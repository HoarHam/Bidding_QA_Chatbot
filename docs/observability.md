# LLM 调用可观测性（Langfuse）

后端把每一次问答组织成一个 Langfuse trace：根 span 是整轮问答，下面挂检索、
Agent 步骤、每次 LLM 生成与每次工具调用。用户对助手消息的 👍/👎 会作为
score 写回**当初那条 trace**，可以在 Langfuse 里按用户、会话、模式筛选并
对照成本与反馈。

追踪是**旁路能力**：未配置密钥、SDK 初始化失败、网络不可达时全部降级为
no-op，问答照常工作，只是没有外部 trace。

## 配置

`.env`：

```ini
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com
LANGFUSE_ENABLED=true
LANGFUSE_CAPTURE_CONTENT=true
LANGFUSE_TIMEOUT=5
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_ENVIRONMENT=dev
```

| 变量 | 作用 |
|---|---|
| `LANGFUSE_ENABLED` | 总开关；`false` 时即使有 KEY 也不上报 |
| `LANGFUSE_CAPTURE_CONTENT` | `false` 时只上报长度/条数，prompt、回答、工具结果正文不出本机 |
| `LANGFUSE_SAMPLE_RATE` | 按 trace 采样（0~1）。**只影响 span，反馈分数不受影响** |
| `LANGFUSE_TIMEOUT` | 出站 HTTP 超时（秒） |
| `LANGFUSE_ENVIRONMENT` | Langfuse 环境标签（dev / staging / prod） |

`LANGFUSE_BASE_URL` 是自托管/区域云的控制台地址；官方云默认是
`https://cloud.langfuse.com`。

## trace 结构

Agent 路径（默认）：

```text
chat.turn                         span，input=用户问题，metadata=mode/web_search/...
├── history.summary               generation（会话超窗时才有）
├── agent.step.1                  span
│   ├── llm.step.1                generation（provider/model、输入消息、工具调用、usage）
│   └── tool.search_bidding_knowledge  tool（参数、结果、耗时、是否降级）
├── agent.step.2
│   └── llm.step.2                generation（最终回答）
└── ...                           （最多 AGENT_MAX_STEPS 轮）
```

固定检索路径（`AGENT_ENABLED=false`）与一次性 `/api/chat`：

```text
chat.turn
├── retrieval                     span（chunk_id、耗时）
└── llm.generation                generation
```

要点：

- **根 span 携带 `user_id` / `session_id`**（`session_id` 就是会话 id），
  可在 Langfuse 里按用户或会话聚合；`trace_name=chat`，`tags=[chat, 模式]`。
- **每一轮 Agent 都会新建 generation**，输入是该轮实际发给模型的消息
  （含工具结果），因此能看清「模型是基于什么工具结果作答的」。
- 发生 provider 回退时，generation 的 metadata 带 `fallback_attempts`，
  model 是最终实际执行的 provider；失败轮次以 `ERROR` 结束并带脱敏后的错误。
- 根 span 的 output 包含最终回答、来源数、finish_reason、Agent 步数/工具调用数；
  metadata 带分阶段用时（total/thinking/tool/generation）。
- 用户主动停止或断开连接时，根 span 以 `WARNING` / `client disconnected` 结束，
  已产生的部分回答仍会落库。

## 用户反馈打分

`messages.trace_id`（迁移 `0009`）在助手回答落库时保存本轮 trace id。
`POST /api/messages/{id}/feedback` 除了写数据库，还会：

```text
name=user_feedback  data_type=NUMERIC
value:  up=1  down=-1  清除=0
score_id=user-feedback-{message_id}
```

Langfuse 的 scores API 按 `score_id` upsert，所以用户反复切换评价只会更新同一条分数，
不会堆出一串互相矛盾的历史记录。重新生成会换一条新 trace，旧 trace 上的历史分数保留。

## 隐私与安全

- **关闭正文上报**：`LANGFUSE_CAPTURE_CONTENT=false`。此时 input/output 只保留
  角色、键名、列表长度与字符数，工具参数只保留键名与长度。
- 错误/状态消息（`status_message`）统一经过 `log.redact()`，API key 不会进 trace。
- 工具结果的 `artifacts` 只上报来源数量，不上报完整来源结构；OCR 原文属于工具/回答正文，
  同样受 `LANGFUSE_CAPTURE_CONTENT` 控制。
- 日志里不打印 Langfuse 密钥、完整 prompt 或回答。

## 采样与反馈为什么分开

SDK 自带的 `sample_rate` 会把同一 trace 的 score 一起采样掉。这里把 SDK 的
sample_rate 固定为 1.0，按 trace 在本地决定是否创建根 span；即使某条 trace 没被采样，
`messages.trace_id` 仍然保存，反馈分数照常上报（可能形成只有分数的 trace）。
这样 `LANGFUSE_SAMPLE_RATE=0` 也不会丢用户反馈。

## 排障

- `/api/health/deps` 里的 `langfuse` 是**配置级**探测，不做联网请求；
  `ok` 只代表密钥与 SDK 就绪。
- 控制台看不到 trace：先确认 `LANGFUSE_ENABLED=true`、两个 KEY 都属于同一个项目、
  `LANGFUSE_BASE_URL` 与项目区域一致；再看后端日志有没有
  `Langfuse 上报失败（...，不影响业务）`。
- 本机系统代理（`127.0.0.1:10808`）连不通 Langfuse 云时，沿用全局的
  `HTTP_TRUST_ENV=false`：SDK 使用注入的 `httpx.Client(trust_env=...)`。
- 401：KEY 与控制台项目不匹配，或把国际站/国内站/区域云地址混用了。
- 把 `LANGFUSE_SAMPLE_RATE` 设成 0 时只能看到反馈分数、看不到 span，这是预期行为。
- 关闭追踪不会删除已落库的 `trace_id` 列；旧消息没有 trace id 时反馈只落数据库。
