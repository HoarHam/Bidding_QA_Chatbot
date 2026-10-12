"""对话路由：SSE 流式、一次性、纯检索。

## 首 token 缓冲（这个文件里最重要的设计）

`StreamingResponse` 一旦发出第一个字节，HTTP 状态码就定死了。如果直接开始推流、
结果所有 provider 都失败了，就只能用 HTTP 200 返回一个「错误事件」——
前端拿不到真正的错误状态，也不好做重试 UI。

所以这里先把生产者的**第一个产出**取出来再决定怎么响应：

- 第一个产出是异常 → 返回正常的 `503 JSON`，前端按普通错误处理；
- 第一个产出是事件 → 返回 `StreamingResponse`，状态码此时已确定。

代价是多一个 task 和一次「本来也要等」的等待，延迟基本不变。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from bidding_qa_chatbot.api.sse import HEARTBEAT_FRAME, SSE_HEADERS, encode_event
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.llm.base import AllProvidersFailed, LLMRequest, MidStreamError, StreamEnd
from bidding_qa_chatbot.llm.base import StreamEvent as LLMStreamEvent
from bidding_qa_chatbot.llm.prompts import NO_CONTEXT_REPLY, build_context, build_messages
from bidding_qa_chatbot.log import redact
from bidding_qa_chatbot.schemas import (
    ApiError,
    ApiErrorResponse,
    ChatRequest,
    ChatResponse,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    ProviderEvent,
    RetrieveResponse,
    SourceRef,
    SourcesEvent,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

HEARTBEAT_SECONDS = 15.0


def _to_usage(usage) -> Usage | None:
    return usage.to_model() if usage is not None else None


# ------------------------------------------------------------------ 检索
async def _retrieve(request: Request, body: ChatRequest):
    settings: Settings = request.app.state.settings
    encoder = getattr(request.app.state, "encoder", None)
    if encoder is None:
        raise HTTPException(
            status_code=503,
            detail="本地嵌入模型未就绪，检索能力不可用（请检查模型是否下载成功）",
        )

    from bidding_qa_chatbot.retrieval.service import Retriever

    retriever = Retriever(
        request.app.state.qdrant,
        encoder,
        getattr(request.app.state, "reranker", None),
        settings,
        bm25_artifact=getattr(request.app.state, "bm25", None),
        collection_vocab_sha1=getattr(request.app.state, "collection_vocab_sha1", None),
    )
    return await retriever.retrieve(body.query, top_k=body.top_k)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_only(body: ChatRequest, request: Request) -> RetrieveResponse:
    """只检索不生成。便于单独排查检索质量，不消耗 LLM 配额。"""
    result = await _retrieve(request, body)
    return RetrieveResponse(
        sources=[s.to_ref() for s in result.sources], elapsed_ms=result.elapsed_ms
    )


# ------------------------------------------------------------------ SSE
@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    from bidding_qa_chatbot.llm.chain import ProviderChain

    settings: Settings = request.app.state.settings
    chain: ProviderChain = request.app.state.chain
    request_id = uuid.uuid4().hex

    agent = getattr(request.app.state, "agent", None)
    use_agent = bool(settings.agent_enabled and agent is not None)

    # 老路径的检索是「先固定检索、再生成」，来源在开流前就已知；
    # Agent 路径的检索发生在工具调用里，来源要等 tool_result 才出现。
    initial_sources: list = []
    initial_elapsed = 0

    if use_agent:
        from bidding_qa_chatbot.llm.prompts import build_agent_messages

        source_iter = agent.astream(build_agent_messages(body.query, body.history), body.mode)
    else:
        result = await _retrieve(request, body)
        initial_sources = [s.to_ref() for s in result.sources]
        initial_elapsed = result.elapsed_ms
        context = build_context(result.sources, settings.max_context_chars)
        messages = build_messages(body.query, context, body.history)
        source_iter = chain.stream_with_fallback(LLMRequest(messages=messages, mode=body.mode))

    queue: asyncio.Queue = asyncio.Queue()

    async def producer() -> None:
        try:
            async for item in source_iter:
                await queue.put(("item", item))
        except Exception as exc:  # noqa: BLE001
            await queue.put(("error", exc))
        finally:
            await queue.put(("eof", None))

    task = asyncio.create_task(producer())

    # 先取第一个产出：决定用 503 JSON 还是 SSE。
    first_kind, first_payload = await queue.get()
    if first_kind == "error":
        task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(task, return_exceptions=True)
        return _error_response(request_id, first_payload)

    async def generate() -> AsyncIterator[str]:
        # 老路径下参考来源**先于**任何 token 发出，这样「展示参考来源」
        # 完全不依赖 LLM 的健康状况。Agent 路径没有这个前提（检索是模型决定的），
        # 来源会在 tool_result 那一帧带出来。
        if not use_agent:
            yield encode_event(
                "sources",
                SourcesEvent(
                    sources=initial_sources,
                    retrieved=len(initial_sources),
                    elapsed_ms=initial_elapsed,
                ),
            )

        pending_kind, pending = first_kind, first_payload
        try:
            while True:
                if pending_kind == "eof":
                    break
                if pending_kind == "error":
                    yield _mid_stream_frame(request_id, pending)
                    break

                for frame in _frames_for(pending):
                    yield frame
                if isinstance(pending, MidStreamError | AllProvidersFailed):
                    break
                # StreamEnd 是老路径的收尾；AgentDone 是 Agent 路径的收尾。
                # 两者都代表「本轮结束」，但不能都用 StreamEnd 判断——
                # Agent 中间轮次也会产出 StreamEnd（那一轮要求调用工具），
                # 若按它收尾就会在第一轮工具调用后直接结束、拿不到答案。
                from bidding_qa_chatbot.agent.agent import AgentDone

                if isinstance(pending, AgentDone):
                    break
                if isinstance(pending, StreamEnd) and not use_agent:
                    break

                try:
                    pending_kind, pending = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_SECONDS
                    )
                except TimeoutError:
                    # 思考模式可能长时间静默，心跳防止中间代理掐断连接。
                    if await request.is_disconnected():
                        break
                    yield HEARTBEAT_FRAME
                    pending_kind, pending = await queue.get()
        except asyncio.CancelledError:
            raise
        finally:
            task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "X-Request-ID": request_id},
    )


def _frames_for(item) -> list[str]:
    from bidding_qa_chatbot.agent.agent import AgentDone, ToolCallNotice, ToolResultNotice

    if isinstance(item, LLMStreamEvent):
        name = "reasoning" if item.kind == "reasoning" else "token"
        return [encode_event(name, DeltaEvent(delta=item.text))]

    if isinstance(item, ProviderEvent):
        return [encode_event("provider", item)]

    if isinstance(item, ToolCallNotice):
        return [
            encode_event(
                "tool_call",
                ToolCallEvent(name=item.name, arguments=item.arguments, display=item.display),
            )
        ]

    if isinstance(item, ToolResultNotice):
        frames = [
            encode_event(
                "tool_result",
                ToolResultEvent(
                    name=item.name,
                    ok=item.ok,
                    elapsed_ms=item.elapsed_ms,
                    summary=item.summary,
                    sources=[SourceRef.model_validate(s) for s in item.sources],
                    degraded=item.degraded,
                ),
            )
        ]
        # 参考来源随工具结果一并推给前端，参考来源面板据此填充
        if item.sources:
            frames.append(
                encode_event(
                    "sources",
                    SourcesEvent(
                        sources=[SourceRef.model_validate(s) for s in item.sources],
                        retrieved=len(item.sources),
                        elapsed_ms=item.elapsed_ms,
                    ),
                )
            )
        return frames

    if isinstance(item, AgentDone):
        return [
            encode_event(
                "done",
                DoneEvent(
                    finish_reason=item.finish_reason,
                    provider=item.provider,
                    model=item.model,
                    usage=_to_usage(item.usage),
                    elapsed_ms=0,
                    request_id="",
                ),
            )
        ]

    if isinstance(item, StreamEnd):
        return [
            encode_event(
                "done",
                DoneEvent(
                    finish_reason=item.finish_reason,
                    provider=item.provider,
                    model=item.model,
                    usage=_to_usage(item.usage),
                    elapsed_ms=0,
                    request_id="",
                ),
            )
        ]

    # FallbackEvent 等其余类型直接发
    from pydantic import BaseModel

    if isinstance(item, BaseModel):
        return [encode_event(_event_name(item), item)]
    return []


def _event_name(item) -> str:
    from bidding_qa_chatbot.schemas import FallbackEvent

    if isinstance(item, FallbackEvent):
        return "fallback"
    return "message"


def _mid_stream_frame(request_id: str, exc: Exception) -> str:
    if isinstance(exc, MidStreamError):
        return encode_event(
            "error",
            ErrorEvent(
                code="stream_interrupted",
                message=exc.message,
                partial=True,
                provider=exc.provider,
                attempts=[a.to_model() for a in exc.attempts],
            ),
        )
    return encode_event(
        "error",
        ErrorEvent(code="stream_error", message=redact(str(exc)), partial=False),
    )


def _error_response(request_id: str, exc: Exception) -> JSONResponse:
    if isinstance(exc, AllProvidersFailed):
        body = ApiErrorResponse(
            error=ApiError(
                code="all_providers_failed",
                message=str(exc),
                attempts=[a.to_model() for a in exc.attempts],
            )
        )
    else:
        body = ApiErrorResponse(error=ApiError(code="internal_error", message=redact(str(exc))))
    return JSONResponse(
        status_code=503,
        content=body.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


# ------------------------------------------------------------------ 一次性
@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    import time

    settings: Settings = request.app.state.settings
    chain = request.app.state.chain
    request_id = uuid.uuid4().hex
    started = time.perf_counter()

    result = await _retrieve(request, body)
    context = build_context(result.sources, settings.max_context_chars)
    messages = build_messages(body.query, context, body.history)

    if not result.sources:
        return ChatResponse(
            answer=NO_CONTEXT_REPLY,
            sources=[],
            provider="none",
            model="none",
            finish_reason="no_context",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            request_id=request_id,
        )

    try:
        llm_result = await chain.complete(LLMRequest(messages=messages, mode=body.mode))
    except AllProvidersFailed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        answer=llm_result.text,
        reasoning=llm_result.reasoning,
        sources=[s.to_ref() for s in result.sources],
        provider=llm_result.provider,
        model=llm_result.model,
        finish_reason=llm_result.finish_reason,
        usage=_to_usage(llm_result.usage),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        request_id=request_id,
    )
