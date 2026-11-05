# Agent 综合评测（Ragas）

`/evaluate` 是后端直接在 **8001** 端口托管的单页评测面板，**不要求登录**：

- 页面：<http://127.0.0.1:8001/evaluate>
- JSON API：`/api/evaluate/*`（公开）
- 离线数据：`data/processed/evaluate_cases.json`
- 结果文件：`data/evaluations/<run_id>.json`（`data/` 已 gitignore）

面板可运行离线评测、在线评测，查看全部历史记录，并按用户名筛选在线评测的
运行记录与用例明细。

## 两种模式

| | 离线 | 在线 |
|---|---|---|
| 数据来源 | `evaluate_cases.json` 人工用例 | **每个用户**最近 10 条带 Langfuse trace 的真实问答 |
| 被测对象 | 真实 Agent（自己选工具、检索、生成） | 已落库的助手回答 |
| 工具选择准确率 | ✅ 人工 `expected_tool` | ✅ 判分模型基于 Langfuse 工具记录判断 |
| 忠诚度 Faithfulness | ✅ | ✅ |
| 相关性 AnswerRelevancy | ✅ | ✅ |
| 召回率 ContextRecall | ✅ 人工 reference | ✅ 判分模型基于 Langfuse 证据生成 reference |
| 平均延迟 | ✅（本轮生成实测） | ✅（历史 `messages.timings`） |

在线评测依赖 **Langfuse 可观测性**：只选择 `messages.trace_id` 非空的最近
10 条助手回答，从 Langfuse 拉取该 trace 的 tool observation（工具名、参数、
输出正文）。没有 trace 的旧消息不进入在线评测。

## 五个指标

### 1. 工具选择准确率

**离线**：每条用例有 `expected_tool`，数据集里的历史名字归一化到当前工具：

| 数据集标签 | 实际判定 |
|---|---|
| `search_bidding_knowledge` | `search_bidding_knowledge` |
| `search_knowledge_graph` | `query_bidding_graph` |
| `search_postgresql` | `query_bidding_graph`（当前没有 SQL 工具） |
| `ask_clarification` | 不调用任何工具 |

实际调用序列里出现期望工具即算对（允许多查一次补充资料）；
`ask_clarification` 必须一次工具都不调。单条用例 Agent 抛异常 → 判错。

**在线**：判分模型读取 Langfuse 中的实际工具调用（名称/参数/输出）与问题，
判断工具选择是否合理，并给出期望工具；结果写入
`case.tool_selection_correct`，汇总为准确率。

### 2. 忠诚度（Faithfulness）

Ragas `Faithfulness`：把回答拆成原子陈述，逐条检查是否能由检索上下文支持。

### 3. 相关性（AnswerRelevancy）

Ragas `ResponseRelevancy`：由回答反向生成问题，再与原始问题比较语义相似度。
嵌入使用本地 `bge-small-zh-v1.5`（与检索同一模型），不额外调用外部 embedding API。
`strictness=1` 是刻意的：GLM 的 OpenAI 兼容端点不支持一次返回多个候选。

### 4. 召回率（ContextRecall）

Ragas `ContextRecall`：把参考回答拆成陈述，逐条判断能否归因到检索上下文。

- 离线使用数据集里的 `reference`；
- 在线使用判分模型基于 Langfuse 工具输出生成的 `reference_answer`。
  这是**模型生成的参考答案**，不是人工标注，页面/结果里会保留该字段以便审计。

### 5. 平均延迟

- 离线：每条用例从 Agent 开始生成到最终回答的 wall-clock 毫秒。
- 在线：读取历史消息落库的 `messages.timings.total_ms`；旧消息没有 timings 时该条不计入均值。

## 离线评测流程

```text
evaluate_cases.json
   │
   ├─ 逐条调用 RagAgent（fast 模式，联网工具关闭）
   │     ├─ 记录 tool_call / tool_result 正文 / answer / latency
   │     └─ 失败用例记录 error，继续下一条
   │
   ├─ 工具选择准确率（规则判定，无需 LLM）
   │
   └─ 有 reference + context + answer 的用例 → Ragas
         Faithfulness / AnswerRelevancy / ContextRecall
```

- 单轮串行执行，避免并发打爆判分模型；页面轮询进度。
- 评测不创建会话、不写 `messages`/`conversations`，只产生 `data/evaluations/*.json`。
- 判分器在生成前初始化；初始化失败会明确落盘失败原因，而不是悄悄给出空指标。

## 在线评测流程

```text
每个用户最近 10 条带 trace_id 的助手回答
   │
   ├─ 按 trace_id 从 Langfuse 拉取 tool observations
   │     └─ 工具名 / 参数 / 输出正文（fields=io,metadata...）
   │
   ├─ 判分模型：判断工具选择是否正确 + 生成参考答案
   │
   └─ Ragas：Faithfulness / AnswerRelevancy / ContextRecall
         ContextRecall 使用上一步生成的参考答案
```

- 对每个用户分别取最近 10 条问答，总用例数为各用户之和；每条记录带 `user_id` / `username` / `conversation_title`。
- 在线运行没有任何人工标注；工具准确率/召回率的“标准”来自判分模型，页面会展示
  `expected_tool`、`reference` 和 `judge_reasoning` 供核对。
- 页面支持按用户名筛选：历史列表只显示包含该用户用例的运行；点开详情后只展示
  该用户的用例，并重新计算该子集的指标。
- 在线评测需要 `LANGFUSE_ENABLED=true` 且 Langfuse 凭据可用；否则启动按钮置灰并给出提示。

> ⚠️ 按本次需求，`/evaluate` 和 `/api/evaluate/*` **没有登录校验**，且会展示
> 用户问题、回答、判分参考答案等对话内容。只应部署在内网/本机；如果要放到公网，
> 必须在反向代理层增加认证，或重新引入应用层鉴权。

## 配置（`.env`）

```ini
EVALUATE_ENABLED=true
EVALUATE_JUDGE_PROVIDER=zhipu
EVALUATE_JUDGE_MODEL=glm-5.3-flash
EVALUATE_JUDGE_MAX_TOKENS=8192
EVALUATE_JUDGE_TIMEOUT=180
EVALUATE_HISTORY_LIMIT=20
EVALUATE_ONLINE_LIMIT=10
EVALUATE_RAGAS_CACHE_DIR=.cache/ragas
RAGAS_DO_NOT_TRACK=true
```

| 变量 | 说明 |
|---|---|
| `EVALUATE_ENABLED` | 总开关；页面仍可打开，但显示未启用 |
| `EVALUATE_JUDGE_PROVIDER` | 判分 provider（`zhipu` / `deepseek`），使用其 OpenAI 兼容端点 |
| `EVALUATE_JUDGE_MODEL` | 判分模型；**应尽量与被测模型不同源**，否则是自评，指标可能系统性虚高 |
| `EVALUATE_JUDGE_MAX_TOKENS` | 判分单次生成上限；太小会把 JSON 截断，指标变成判分失败 |
| `EVALUATE_JUDGE_TIMEOUT` | 判分请求超时（秒），也作为 Ragas `RunConfig.timeout` |
| `EVALUATE_HISTORY_LIMIT` | 历史列表最多返回条数（不删文件） |
| `EVALUATE_ONLINE_LIMIT` | 在线评测对**每个用户**最多取多少条最近问答 |
| `EVALUATE_RAGAS_CACHE_DIR` | Ragas 磁盘缓存；留空关闭。目录按 `provider-model` 自动分桶 |

`EVALUATE_ENABLED=true` 但判分 provider 的密钥未配置时，`validate_runtime` 会在启动日志中
给出明确告警，评测服务返回 503。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/evaluate` | HTML 面板（公开，无登录表单） |
| GET | `/api/evaluate/config` | 评测配置/可用性/数据集信息 |
| POST | `/api/evaluate/offline` | 启动离线评测，立即返回 run_id |
| POST | `/api/evaluate/online` | 启动全用户在线评测 |
| GET | `/api/evaluate/runs?username=alice` | 评测摘要列表，可按用户名筛选 |
| GET | `/api/evaluate/users` | 历史中出现过的用户及用例数 |
| GET | `/api/evaluate/runs/{run_id}?username=alice` | 单次结果；带用户名时只返回该用户用例并重算指标 |

同一时间只允许一个评测任务；已有任务运行时返回 409。

## 依赖注意事项

- `ragas==0.4.3` 在 import 时引用 `langchain_community.chat_models.vertexai`，
  该模块在 `langchain-community 0.4` 已移除。`pyproject.toml` 显式锁
  `langchain-community>=0.3.31,<0.4`，否则服务启动评测时会 ImportError。
- 判分使用 Instructor 结构化输出；GLM 的 OpenAI 兼容端点已验证可用。
- Langfuse observations API 默认只返回 `core+basic`，`input/output` 会是 None；
  `observability.service.fetch_trace_observations` 显式请求
  `core,basic,time,io,metadata,model,usage,trace_context`。
- 判分是付费 LLM 调用：离线 12 条 × 3 个指标 + 在线 10 条 × (判分+3 指标)
  会产生多轮请求，评测前请确认额度。
- 判分模型的输出 token 上限有限；上下文/回答会截断到 3000/8000 字符。
  某个指标首次判分因 `IncompleteOutputException` 全空时，会用
  `max(16384, 2 × EVALUATE_JUDGE_MAX_TOKENS)` 自动重试一次；仍失败才在结果页列出。
- Langfuse 抓取失败会重试 3 次（每次间隔 1 秒）；仍失败则该条在线用例标记错误，
  不影响同轮其他用例。
