# 系统架构

第一阶段只做 RAG：把 Excel 里的问答语料入库到 Qdrant，用「稠密 + 稀疏」混合检索召回，
CrossEncoder 精排，再交给可切换的 LLM 流式作答，并在前端展示参考来源。

```mermaid
flowchart LR
    subgraph client["浏览器"]
        UI["Next.js App Router<br/>React + TS + Tailwind"]
    end

    subgraph server["Python 后端 (FastAPI)"]
        API["API 层<br/>/api/chat/stream · SSE"]
        AG["Agent 层<br/>工具调用循环"]
        RET["检索层<br/>hybrid → RRF → rerank"]
        LLM["LLM 层<br/>provider 链 + 回退"]
    end

    subgraph ext["外部依赖"]
        QD[("Qdrant Cloud<br/>bid_qa")]
        DS["DeepSeek<br/>deepseek-flash"]
        ZP["智谱<br/>glm-4.7-flashx"]
        VL["vLLM<br/>本地（仅留接口）"]
    end

    subgraph local["本地模型 (CPU)"]
        EMB["bge-small-zh-v1.5<br/>512 维"]
        RRK["bge-reranker-base<br/>CrossEncoder"]
    end

    UI -->|"POST /api/chat/stream"| API
    API --> AG
    AG -->|"工具：search_bidding_knowledge"| RET
    RET --> EMB
    RET --> QD
    RET --> RRK
    AG --> LLM
    LLM --> DS
    LLM -.降级.-> ZP
    LLM -.降级.-> VL
```

## 一次问答的时序

注意 `sources` 事件在**任何 LLM 调用之前**发出——这样「展示参考来源」这一需求
不依赖 LLM 的健康状况，即使所有模型都挂了，检索结果照样能显示。

```mermaid
sequenceDiagram
    autonumber
    participant U as 浏览器
    participant A as FastAPI
    participant AG as Agent
    participant R as 检索层
    participant Q as Qdrant
    participant L as LLM 链

    U->>A: POST /api/chat/stream
    A->>AG: astream(messages, mode)
    AG->>L: 带 tools 的请求
    alt 模型认为需要检索
        L-->>AG: tool_calls
        AG-->>U: event: tool_call
        AG->>R: 执行检索工具
        R->>Q: Prefetch(dense) + Prefetch(sparse)
        Q-->>R: RRF 融合候选
        R->>R: CrossEncoder 精排
        R-->>AG: Top-K
        AG-->>U: event: tool_result + event: sources
        AG->>L: 回灌工具结果，进入下一轮
    end
    alt 主 provider 首个 token 前失败
        L->>L: 记录 Attempt，切换下一个 provider
        L-->>U: event: fallback
    end
    L-->>U: event: provider
    loop 逐 token
        L-->>U: event: token / reasoning
    end
    L-->>U: event: done
```

> 老路径（`AGENT_ENABLED=false`）没有 Agent 与工具调用这两步：
> 先固定检索、把资料塞进提示词，再直接生成。此时 `sources` 事件在
> 任何 token 之前发出；Agent 路径下来源要等 `tool_result` 才出现。

## 为什么这样分层

- **`agent/` 单独成层**：它编排「模型 ↔ 工具」的往返，不关心检索内部怎么实现，
  也不关心用的是哪家 LLM。新增工具只碰 `agent/`，检索与模型两侧都不用动。
- **`llm/`、`embeddings/`、`retrieval/` 分开放**：三者失败模式和伸缩性完全不同。
  LLM 挂了不该拖垮检索；检索也要能在「假编码器」下独立测试。
- **`ingestion/` 与 `retrieval/` 分离**：入库是批处理 + CLI，检索是每请求异步，
  两者只共享 `ids.py` 和 `qdrant_store.py`。
- **`schemas.py` 放在顶层而非 `api/` 下**：SSE 事件模型被 API 层和前端
  `lib/types.ts` 同时消费，线上格式只有一个地方可查。
- **`integrations/` 单独成目录**：第二阶段的东西集中放，一眼能看出「还没实现」。

## 关键约束（改代码前必读）

1. **`load_dotenv(override=False)` 必须早于任何 `huggingface_hub` / `sentence_transformers`
   的导入。** 本机无法直连 huggingface.co，`HF_ENDPOINT` 指向 hf-mirror 才能下载模型。
   `embeddings/*` 因此全部使用函数内延迟导入。
2. **`pyproject.toml` 里的 `[project.scripts]` 指向 `bidding_qa_chatbot:main`**，
   `main()` 必须留在 `__init__.py`。
3. **`openai` 锁在 2.x**：3.x 改用 `httpx2` 并不再依赖 `httpx`，
   会导致 respx/pytest-httpx 无法拦截，mock 测试会静默打到线上付费 API。
4. **Windows 下所有文件 IO 显式 `encoding="utf-8"`**：本机 locale 是中文，
   Python 默认用 GBK，读中文语料或打印中文会直接崩。
