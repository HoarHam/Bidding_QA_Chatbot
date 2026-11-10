# 可增加的 Skills

> 本文默认指 **Claude Code 项目级 Skills**（`.claude/skills/<name>/SKILL.md`），
> 用于把本项目反复出现、容易踩坑的开发/验证流程固化下来。
>
> 如果你的意思是「给聊天机器人运行时增加 Anthropic Agent Skills」，见文末附录；
> 当前自定义 Agent 还没有 Skill 加载器，需要额外实现。

## 先说结论

这个项目最适合增加的 Skills，不是「通用写代码」或「通用测试」，而是**承载本项目
特有陷阱的流程型技能**。优先建议 8 个：

| 优先级 | Skill | 触发场景 | 核心价值 |
|---|---|---|---|
| P0 | `bq-release-check` | 提交/发布前 | 一次跑齐后端测试、前端构建、契约同步、迁移检查 |
| P0 | `bq-eval` | 跑 Ragas 评测、看指标 | 管理昂贵的离线/在线评测、解释指标失败 |
| P0 | `bq-rag-debug` | 检索不准、引用不对、缓存异常 | 定位 Qdrant/BM25/精排/缓存问题 |
| P0 | `bq-llm-debug` | provider 报错、回退、标记泄露 | 复现并守住首 token/DSML/GLM 过滤器不变量 |
| P1 | `bq-frontend-verify` | 改前端、流式、上传、滚动 | 真浏览器验证 SSE 与交互 |
| P1 | `bq-db-migrate` | 改模型/加字段/迁移 | 生成、审查、应用 Alembic 迁移 |
| P1 | `bq-ingest` | 语料更新、重建集合 | 安全执行入库，防止误删 Qdrant 集合 |
| P1 | `bq-observability` | Langfuse/trace/反馈 | 验证 trace 落点、反馈分数、隐私开关 |
| P2 | `bq-security-review` | 上线前、新增公开端点/工具 | 检查 IDOR、SSRF、只读边界、公开数据暴露 |
| P2 | `bq-doc-sync` | 改 schema/config/API | 同步前端类型、文档、测试环境键 |
| P2 | `bq-incident` | 服务起不来/端口冲突/Windows 异常 | 快速按本项目已知坑排查 |

---

## 一、Skill 适合做什么，不适合做什么

**适合**

- 多步骤、有顺序、有副作用的流程（迁移、入库、评测、发布检查）。
- 需要反复记住的项目特有约束（SSE 代理不能缓冲、Neo4j 只读事务、迁移用 user_id 过滤）。
- 需要「先确认再执行」的危险操作（`ingest --recreate`、删除集合、停止用户服务）。
- 需要固定输出格式的排查报告。

**不适合**

- 一句话就能完成的命令（`pytest tests/test_bm25.py`），直接跑即可。
- 已经写在 `CLAUDE.md` 且不会经常变化的通用知识。
- 为了「看起来很多」而拆分的重复技能。

---

## 二、推荐 Skill 详情

### 1. `bq-release-check`（P0）

**触发**

> 「提交前检查一下」「跑全量验证」「发布前确认」

**为什么需要**

这个项目的前端、后端、契约、迁移分散在不同目录；只跑 `pytest` 会漏掉：
`frontend/lib/types.ts` 未同步、Alembic 与 ORM 不一致、Next 构建失败、SSE 代理退化。

**流程**

1. 后端：`uv run --no-sync pytest`、`ruff check .`、`ruff format --check .`
2. 契约：确认 `tests/test_contract_sync.py` 通过
3. 数据库：`uv run --no-sync alembic check`
4. 前端：`cd frontend && npx tsc --noEmit && npm run build`
5. 可选 UI：前后端都在 8001/3000 时跑 `npm run verify:ui`
6. 输出「通过/失败 + 失败命令 + 下一步建议」

**边界**

- 不自动 commit/push。
- 不读取/打印 `.env` 密钥。
- 需要停止用户服务时先征求确认。

**建议脚本**

`.claude/skills/bq-release-check/scripts/check.sh`，封装命令和退出码；
PowerShell/Windows 下另备 `.ps1`。

---

### 2. `bq-eval`（P0）

**触发**

> 「跑一下离线评测」「看在线指标」「对比两次 Ragas 结果」

**为什么需要**

评测涉及判分模型、Langfuse trace、Ragas、后台任务、轮询和费用；流程长且容易
误判（把判分失败当成 0 分、用被采样掉的 trace、重复起任务）。

**流程**

1. 检查 `/api/evaluate/config`：`enabled`、判分模型、`langfuse_available`、数据集条数
2. 提醒判分模型与被测模型是否同源
3. 触发离线或在线评测，轮询 run 状态
4. 汇总五项指标：工具准确率、忠诚度、相关性、召回率、平均延迟
5. 读取 `metric_errors`，区分「判分失败」「无数据」「指标为 0」
6. 如与上次 run 对比，输出差值而不是只看绝对值
7. 提示费用/耗时；不自动重复跑

**边界**

- 不跑 12 条全量离线评测除非用户确认费用。
- 不在公开网络环境展示含用户问答的评测详情。
- 不把 Langfuse/智谱密钥写进日志。

**可选脚本**

`scripts/eval_smoke.py`：用 1 条用例验证判分模型和 Langfuse 连通性，跑完清理测试 trace。

---

### 3. `bq-rag-debug`（P0）

**触发**

> 「检索不到」「召回结果很差」「引用不对」「缓存没命中」「换语料后不对」

**为什么需要**

检索链路的静默失败很多：

- Qdrant 集合/维度不匹配；
- BM25 词表 sha1 与集合不一致 → 自动退化纯 dense；
- RRF 分数被误当置信度；
- 精排输入文本与入库不一致；
- 候选缓存 key 包含哪些字段；
- Qdrant 本地模式与服务端分数不同。

**流程**

1. `GET /api/stats` 看集合、点数、维度
2. 用 `POST /api/retrieve` 复现查询，检查 `sources`、`rerank_score`、`elapsed_ms`
3. 检查 BM25 工件与集合 `vocab_sha1` 是否一致
4. 检查候选缓存命中情况（日志/评测 trace 的 `cache_hit`）
5. 必要时对比 `hybrid_search` 原始候选与精排结果
6. 给出「检索/精排/缓存/数据」哪一层的问题，并建议补测试

**边界**

- 不修改 Qdrant 集合，不运行 `--recreate`。
- 不把 RRF 分数解释成置信度。

---

### 4. `bq-llm-debug`（P0）

**触发**

> 「DeepSeek 报错」「回退到智谱失败」「流式中断」「工具调用标记泄露」

**为什么需要**

这些 bug 只在特定 provider/模式下出现，而且不能靠「重试」解决：

- 首 token 之后绝不能换 provider；
- DeepSeek thinking 默认开启，fast 模式要显式关闭；
- DSML / GLM `<tool_call>` 标记可能混入 content 或 reasoning；
- `reasoning_content` 必须回传；
- zai-sdk 是同步的，线程桥接和取消逻辑容易改坏。

**流程**

1. 先跑 `tests/test_chain_fallback.py`、`tests/test_llm_providers.py`、
   `tests/test_dsml.py`、`tests/test_zhipu_markup.py`
2. 用假 provider 复现，不消耗真实额度
3. 检查是否绕过 `llm/dsml.py` / `llm/tool_markup.py`
4. 需要真实调用时先确认密钥可用与费用
5. 输出「问题层（provider/chain/parser/agent）+ 最小修复 + 回归测试」

**边界**

- 不把真实 key、完整 prompt、用户内容写进日志。
- 不删除过滤器「先让回答跑通」——它同时是工具恢复路径。

---

### 5. `bq-frontend-verify`（P1）

**触发**

> 「改聊天界面」「流式不逐字」「滚动不对」「上传/反馈/评测页有问题」

**为什么需要**

本项目踩过两次只有真浏览器才能发现的坑：

- Next 16 拦截跨域开发资源 → 页面能打开但所有按钮失效；
- `rewrites` 缓冲 SSE → 答案最后一次性蹦出。

**流程**

1. `npx tsc --noEmit` + `npm run build`
2. 静态守卫 `tests/test_frontend_config.py`
3. 启动 8001/3000（或 8002 测试端口），跑 `npm run verify:ui`
4. 流式检查看「是否存在中间态样本」，不要用采样间隔判断
5. 涉及滚动/上传/反馈时，用真实浏览器手动或 Playwright 验证
6. 输出失败截图/控制台错误

**边界**

- 不把 Next 代理改回 `rewrites`。
- 不缓冲 `upstream.body`；`params` 必须 `await`。
- 不随便杀用户的 8001/3000 进程。

---

### 6. `bq-db-migrate`（P1）

**触发**

> 「加一个字段」「改消息表」「写迁移」

**为什么需要**

迁移是本项目最容易造成「测试通过、线上缺列」的地方：

- `alembic/env.py` 必须导入 `db.models`；
- `messages.attachments` 用 `none_as_null=True`；
- 迁移必须 nullable 或给 server_default，兼容旧数据；
- 仓储查询必须带 `user_id`。

**流程**

1. 先改模型/仓储并补测试
2. `alembic revision --autogenerate -m "..."`
3. 人工审查迁移（不要盲信 autogenerate）
4. `alembic upgrade head`，再 `alembic check`
5. 跑 SQLite 测试 + 需要时 Postgres 集成测试
6. 更新 `docs/database.md`

**边界**

- 不在生产库直接手写 DDL。
- 不删除/重命名列而不提供回滚策略。

---

### 7. `bq-ingest`（P1）

**触发**

> 「重新入库」「更新语料」「重建 Qdrant 集合」

**为什么需要**

`ingest --recreate` 会清空集合；语料 row 52 的答案只有图片，需要 OOXML 特殊处理；
BM25 词表变化会让旧稀疏向量失效。

**流程**

1. 检查 xlsx 路径、行数、图片、BM25 工件
2. 默认跑幂等 `ingest`；`--recreate` 必须二次确认
3. 入库后检查 `/api/stats` 点数、集合元数据、vocab sha1
4. 跑一条检索冒烟查询
5. 报告新增/更新/清理数量

**边界**

- 没有明确确认绝不执行 `--recreate`。
- 不在生产集合上试 ingest。

---

### 8. `bq-observability`（P1）

**触发**

> 「Langfuse 没数据」「反馈分数没上报」「trace 对不上」

**流程**

1. `/api/health/deps` 看 `langfuse` 配置
2. 检查 `messages.trace_id` 是否落库
3. 用合成内容发一条 trace，确认 observation 类型/input/output 可读
4. 验证 feedback score 的 `score_id`、`user_feedback`、正负值
5. 检查 `LANGFUSE_CAPTURE_CONTENT=false` 时是否只上报长度
6. 清理测试 trace

**边界**

- 不把真实用户问答发到外部项目做调试。
- 不在日志打印 Langfuse key。
- observations API 必须请求 `io` 字段组，否则 input/output 为 None。

---

### 9. `bq-security-review`（P2）

**触发**

> 「上线前检查」「新增公开接口/工具/上传」

**检查清单**

- 所有会话/消息/附件查询是否带 `user_id`；
- 新增路由是否意外公开；
- `/evaluate` 是否暴露用户问答（当前按要求公开，公网必须加代理认证）；
- 上传：magic bytes、大小、zip bomb、路径穿越；
- 网页抓取：SSRF、内网 IP、重定向、超时；
- SQL/Cypher：只读、参数化、行数/超时；
- OCR/网页内容视为不可信，防 prompt injection；
- 日志/健康检查/异常是否经过 `redact()`。

**输出**

按严重级别列出「问题 + 位置 + 修复建议」，不自动改安全策略。

---

### 10. `bq-doc-sync`（P2）

**触发**

> 「改了 schema」「新增 API/配置项」「同步文档」

**流程**

1. `schemas.py` → `frontend/lib/types.ts`
2. 新配置 → `tests/conftest.py` 的 `_ENV_KEYS`
3. 新端点/字段 → `docs/api.md`、`docs/database.md`
4. 架构/行为变化 → `CLAUDE.md`、`docs/*`
5. 跑 `test_contract_sync.py`

**边界**

- 不把 `.env` 密钥写进文档或示例。
- `CLAUDE.md` 被 gitignore，但仍是本地重要上下文。

---

### 11. `bq-incident`（P2，可选）

**触发**

> 「8001 起不来」「端口占用」「Windows 事件循环报错」「连不上 Qdrant/Neo4j」

**流程**

按本项目已知顺序排查：

1. 端口/进程：8001、8002、3000、6333、7687、11434；
2. Windows：是否 SelectorEventLoop、系统代理 `HTTP_TRUST_ENV`；
3. 依赖：`uv sync` 是否被运行中的 exe 锁住；
4. 外部服务：Qdrant/Neo4j/Postgres/Ollama 健康；
5. `curl --ssl-no-revoke` 验证；
6. 不随意杀用户进程。

---

## 三、Skill 的标准结构

推荐目录：

```text
.claude/
└── skills/
    ├── bq-release-check/
    │   ├── SKILL.md
    │   └── scripts/check.sh
    ├── bq-eval/
    │   ├── SKILL.md
    │   └── scripts/eval_smoke.py
    └── ...
```

`SKILL.md` 示例：

```markdown
---
name: bq-release-check
description: 招投标问答项目提交/发布前的全量验证。用于用户说“提交前检查”“发布前跑一遍”“全量验证”时。
---

# 目标

在不提交代码、不读取密钥的前提下，验证后端、前端、契约与迁移。

## 步骤

1. `uv run --no-sync pytest`
2. `uv run --no-sync ruff check . && uv run --no-sync ruff format --check .`
3. `uv run --no-sync alembic check`
4. `cd frontend && npx tsc --noEmit && npm run build`
5. 汇总失败项；不要自动提交。

## 禁止

- 不要打印 `.env` 内容。
- 不要停止用户正在运行的服务；需要时先询问。
```

**写作原则**

- `description` 写清「什么时候用」，这是 Claude 选择 Skill 的主要依据。
- 步骤用命令/判断条件，不写泛泛而谈。
- 明确「禁止」和「需要确认」的动作。
- 大脚本放 `scripts/`，`SKILL.md` 只留决策流程。
- 项目级 Skill 提交到仓库；个人偏好放 `~/.claude/skills`。

---

## 四、建议的实施顺序

**第一批（投入小、复用高）**

1. `bq-release-check`
2. `bq-llm-debug`
3. `bq-frontend-verify`

**第二批（项目核心复杂性）**

4. `bq-eval`
5. `bq-rag-debug`
6. `bq-db-migrate`

**第三批（运维与治理）**

7. `bq-ingest`
8. `bq-observability`
9. `bq-security-review`
10. `bq-doc-sync`

不建议一次建十几个 Skill：Skill 太多会互相竞争触发条件，也会增加维护成本。
先把最常用、最容易出错的 3 个跑顺，再按实际痛点扩展。

---

## 附录：如果指「聊天机器人运行时的 Agent Skills」

当前项目的 Agent 是自定义 `RagAgent` + `ToolRegistry`，**不会自动加载 Claude Code
的 `.claude/skills`**。若要给机器人本身增加 Skill 机制，需要额外实现：

1. `skills/` 目录与 `SKILL.md` 规范（name/description/instructions/resources）；
2. 一个 `load_skill` / `read_skill_resource` 工具，或按需注入 system prompt；
3. 权限与内容边界（Skill 资源是可信项目文件，用户上传文件不可信）；
4. 观测与评测：记录本次回答使用了哪个 Skill。

适合做成运行时 Skill 的内容：

- **招标文件审查清单**：固定章节检查表和输出格式；
- **合规风险规则手册**：可判定的硬规则 + 法条引用模板；
- **质疑函/投诉书模板**：固定格式、必填字段、语气；
- **评测用例编写规范**：问题、reference、expected_tool 的写法；
- **投标文件检查清单**：按项目类型组织的过程性知识。

这些内容更像「可复用的流程与模板」，适合 Skill；而查数据库、查图谱、抓网页仍然
应该保持为 Tool，不要混在一起。
