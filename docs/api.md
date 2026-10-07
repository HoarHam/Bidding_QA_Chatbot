# API 说明

服务默认监听 `API_HOST:API_PORT`（`.env` 中为 `0.0.0.0:8001`）。
交互式文档在 `/docs`。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | **SSE 流式问答**（主入口） |
| POST | `/api/chat` | 一次性 JSON 回答 |
| POST | `/api/retrieve` | **只检索不生成**，便于单独排查检索质量、不消耗 LLM 额度 |
| GET | `/api/sources/{chunk_id}` | 分块完整记录 |
| GET | `/api/sources/{chunk_id}/images/{index}` | 分块内嵌图片（PNG） |
| GET | `/api/health` | 存活探针（不触碰任何外部依赖） |
| GET | `/api/health/deps` | 各依赖可达性 |
| GET | `/api/stats` | 集合与模型信息（前端 SSR 页头用） |

## SSE 协议

请求：

```json
{"query": "投标保证金的退还期限是多久？", "mode": "fast", "history": [], "top_k": null}
```

`mode` 取 `fast` 或 `thinking`。响应为 `text/event-stream`，使用**具名事件**
（前端按事件名分发，不做字符串嗅探）。

事件顺序：`sources` → (`provider` | `fallback`) → (`reasoning` | `token`)* → (`done` | `error`)

```
event: sources
data: {"sources":[{"chunk_id":"qa0113-0015","source_label":"招投标问答样例数据0113 · 第17行","question":"...","snippet":"...","excel_row":17,"rrf_score":1.0,"rerank_score":0.997,"images":[]}],"retrieved":5,"elapsed_ms":412}

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
