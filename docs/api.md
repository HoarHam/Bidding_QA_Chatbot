# API 说明

服务默认监听 `API_HOST:API_PORT`（`.env` 中为 `0.0.0.0:8001`）。
交互式文档在 `/docs`。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | **SSE 流式问答**（主入口） |
| POST | `/api/chat` | 一次性 JSON 回答 |
| POST | `/api/attachments` | 上传附件（multipart；只存储，发送时才 OCR） |
| POST | `/api/retrieve` | **只检索不生成**，便于单独排查检索质量、不消耗 LLM 额度 |
| GET | `/api/sources/{chunk_id}` | 分块完整记录 |
| GET | `/api/sources/{chunk_id}/images/{index}` | 分块内嵌图片（PNG） |
| GET | `/api/health` | 存活探针（不触碰任何外部依赖） |
| GET | `/api/health/deps` | 各依赖可达性 |
| GET | `/api/stats` | 集合与模型信息（前端 SSR 页头用） |

## SSE 协议

请求：

```json
{"query": "投标保证金的退还期限是多久？", "mode": "fast", "web_search": false, "attachment_ids": [], "history": [], "top_k": null}
```

### 附件

1. 先 `POST /api/attachments`（`multipart/form-data`，字段名 `file`，需 Bearer token）上传；
   返回 `{id, filename, content_type, size}`。支持 PNG/JPEG/WebP/PDF/DOCX，上限 10MB。
2. 聊天请求带上 `attachment_ids: ["<id>"]`（当前最多 1 个）。后端在**开流之前**调用本地
   GLM-OCR 解析；解析失败返回 JSON 错误（400/404/413/415/422/503），**不创建会话、不落用户消息**。
3. 历史消息的 `MessageOut.attachments` 只返回元数据，OCR 文本仅服务端保存（用于同会话追问）。
   同一会话后续不带附件的提问会自动带上最近一次附件的 OCR 文本。

`mode` 取 `fast` 或 `thinking`。`web_search` 为用户端「联网搜索」开关：
只有当服务端 `ENABLE_WEB_SEARCH=true` 且本请求为 `true` 时，模型才会看到联网工具；
关闭时联网工具既不进 schema，也无法被执行。前端用 `/api/stats` 的
`web_search_available` 判断开关是否可点（服务端不可用时置灰）。响应为
`text/event-stream`，使用**具名事件**（前端按事件名分发，不做字符串嗅探）。

**Agent 路径（默认）**的事件顺序：

```
tool_call → tool_result → (sources | web_sources) → provider → (token | reasoning)* → done
```

模型认为不需要检索时（如闲聊），不会有 `tool_call` / `tool_result` / `sources`，
直接 `provider → token* → done`。多轮检索时 `tool_call`/`tool_result`/`sources`/
`web_sources` 会重复出现；**来源帧携带累计列表**（多轮工具调用不会清空面板）。

**老路径（`AGENT_ENABLED=false`）**：`sources → provider → token* → done`，
来源在任何 token 之前发出；该路径忽略 `web_search`。

```
event: tool_call
data: {"name":"search_bidding_knowledge","arguments":{"query":"投标保证金 退还"},"display":"检索知识库：投标保证金 退还"}

event: tool_result
data: {"name":"search_bidding_knowledge","ok":true,"elapsed_ms":412,"summary":"检索到 5 条相关资料","sources":[...],"degraded":false}
```

`tool_call` / `tool_result` 是给人看的：检索加二轮生成要好几秒，
界面没有反馈用户会以为卡住了。`sources` 随 `tool_result` 一并推出，
参考来源面板据此填充。

```
event: sources
data: {"sources":[{"chunk_id":"qa0113-0015","source_label":"招投标问答样例数据0113 · 第17行","question":"...","snippet":"...","excel_row":17,"rrf_score":1.0,"rerank_score":0.997,"images":[]}],"retrieved":5,"elapsed_ms":412}

event: web_sources
data: {"sources":[{"url":"https://example.com/a","title":"政府采购新规解读","snippet":"...","score":0.87,"engine":"tavily","published_date":"2026-09-01"}],"retrieved":3,"elapsed_ms":512}

event: provider
data: {"provider":"deepseek","model":"deepseek-flash","mode":"fast","thinking_enabled":false}

event: fallback
data: {"from":"deepseek","to":"zhipu","reason":"first_token_timeout","attempts":[...]}

event: reasoning
data: {"delta":"先分析问题涉及的法律依据…"}

event: token
data: {"delta":"招标"}

event: error
data: {"code":"stream_interrupted","message":"deepseek 连接中断","partial":true,"provider":"deepseek","attempts":[]}

event: done
data: {"finish_reason":"stop","provider":"deepseek","model":"deepseek-flash","usage":{"prompt_tokens":578,"completion_tokens":44,"total_tokens":622},"elapsed_ms":0,"request_id":""}
```

### 几个刻意的设计

- **`sources` 先于任何 token 发出。** 这样「展示参考来源」不依赖 LLM 的健康状况——
  即使所有模型都挂了，检索结果照样显示。前端在答案还在流式输出时就已经填好了来源面板。
- **`fallback` 是独立事件**，前端据此提示「已从 deepseek 切换到 zhipu」。答案来自备用模型，
  用户有权知道。
- **中文不转义**（`ensure_ascii=False`），线上是原样 UTF-8，不是 `\uXXXX`。
- **心跳**：每 15 秒发一次 `: ping`。思考模式可能长时间静默，没有心跳会被中间代理掐断连接。
- **压缩必须关闭**：SSE 一旦被 gzip 缓冲，就会「看起来正常但一次性返回」。

### 总失败时返回 503 JSON

`StreamingResponse` 一旦发出第一个字节，状态码就定死了。所以后端**先缓冲首个产出**
再决定响应方式：若所有 provider 都在首 token 前失败，返回的是普通 JSON：

```json
{"error": {"code": "all_providers_failed", "message": "...", "attempts": [{"provider": "deepseek", "phase": "first_token", "error_type": "TimeoutError", "elapsed_ms": 15012}]}}
```

前端据 HTTP 状态码判断该按 SSE 还是按普通错误处理。

## 错误码

| code | 含义 |
|---|---|
| `all_providers_failed` | 所有 provider 首 token 前失败（HTTP 503） |
| `stream_interrupted` | 流中途断开，`partial=true` 表示已输出部分正文 |
| `stream_error` | 其他流式错误 |

## 调用示例

```bash
# 纯检索
curl -sS -X POST http://127.0.0.1:8001/api/retrieve \
  -H "content-type: application/json" \
  --data-binary '{"query":"投标保证金的退还期限"}'

# 流式问答（-N 关闭 curl 缓冲）
curl -N -X POST http://127.0.0.1:8001/api/chat/stream \
  -H "content-type: application/json" -H "accept: text/event-stream" \
  --data-binary '{"query":"投标保证金的退还期限","mode":"fast"}'
```

> Windows 下用 `curl` 时若报 `CRYPT_E_REVOCATION_OFFLINE`，加 `--ssl-no-revoke`。
> 这是 Windows Schannel 的证书吊销检查失败，与本项目无关；Python 侧（httpx）不受影响。
