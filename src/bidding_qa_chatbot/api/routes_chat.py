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
from bidding_qa_chatbot.attachments import (
    AttachmentContext,
    AttachmentError,
    build_attachment_context,
    prepare_attachments,
)
from bidding_qa_chatbot.auth.deps import CurrentUser
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.db import repository as repo
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
    WebSource,
    WebSourcesEvent,
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


async def _load_attachment_context(request: Request, user, body: ChatRequest) -> AttachmentContext:
    """解析本轮附件并加载最近的历史附件文本。

    必须在 `_resolve_conversation` **之前**调用：OCR 失败时不能创建会话/落用户消息。
    """
    settings: Settings = request.app.state.settings
    store = getattr(request.app.state, "attachment_store", None)
    ocr = getattr(request.app.state, "ocr", None)

    prepared = []
    if body.attachment_ids:
        if ocr is None:
            raise AttachmentError("ocr_unavailable", "附件解析服务未初始化", status_code=503)
        if store is None:
            from bidding_qa_chatbot.attachments import AttachmentStore

            store = AttachmentStore(settings.uploads_dir, max_bytes=settings.max_upload_bytes)
            request.app.state.attachment_store = store
        prepared = await prepare_attachments(
            settings=settings,
            store=store,
            ocr=ocr,
            user_id=user.id,
            attachment_ids=body.attachment_ids,
        )

    # 追问：取最近带附件用户消息的 OCR 文本（必须早于当前消息落库，并按 id 去重）
    prior_rows: list[list[dict]] = []
    if body.conversation_id:
        try:
            conversation_uuid = uuid.UUID(body.conversation_id)
        except ValueError:
            conversation_uuid = None
        db = getattr(request.app.state, "db", None)
        if conversation_uuid is not None and db is not None:
            try:
                async with db.session() as session:
                    prior_rows = await repo.recent_user_attachments(
                        session,
                        user_id=user.id,
                        conversation_id=conversation_uuid,
                        limit=settings.attachment_recent_limit,
                    )
            except Exception as exc:  # noqa: BLE001 — 追问上下文是增强项，失败不阻塞
                logger.warning("读取历史附件失败（本轮继续）: %s", redact(str(exc)))

    return build_attachment_context(
        prepared,
        prior_rows,
        max_current_chars=settings.max_attachment_chars,
        max_prior_chars=settings.max_prior_attachment_chars,
    )


def _attachment_error_response(request_id: str, exc: AttachmentError) -> JSONResponse:
    payload = ApiErrorResponse(error=ApiError(code=exc.code, message=str(exc)))
    response = JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))
    response.headers["X-Request-ID"] = request_id
    return response


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_only(body: ChatRequest, request: Request, user: CurrentUser) -> RetrieveResponse:
    """只检索不生成。便于单独排查检索质量，不消耗 LLM 配额。"""
    result = await _retrieve(request, body)
    return RetrieveResponse(
        sources=[s.to_ref() for s in result.sources], elapsed_ms=result.elapsed_ms
    )


async def _resolve_conversation(
    request: Request,
    user,
    body: ChatRequest,
    *,
    attachments: list[dict] | None = None,
):
    """解析或新建会话，并**立即提交**用户消息（含附件元数据与 OCR 文本）。

    刻意不复用请求作用域的 `DbSession`：它的提交发生在响应**完全发送之后**
    （StreamingResponse 的依赖清理在流结束才执行）。那样会有两个后果：
    流中途断开时用户消息一并丢失；而下面落库助手消息用的是另一个会话，
    会因为读不到尚未提交的会话行而外键失败。

    所以这里用独立的事务，在开流之前就落地。
    """
    db = request.app.state.db
    async with db.session() as session:
        conversation = None
        if body.conversation_id:
            try:
                conv_id = uuid.UUID(body.conversation_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="会话 id 格式非法") from exc
            conversation = await repo.get_conversation(
                session, user_id=user.id, conversation_id=conv_id
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="会话不存在")

        if conversation is None:
            conversation = await repo.create_conversation(session, user_id=user.id)

        await repo.add_message(
            session,
            conversation=conversation,
            role="user",
            content=body.query,
            attachments=attachments,
            title_from=True,  # 首条用户消息顺带定标题
        )
        return str(conversation.id)


async def _persist_assistant(
    request: Request,
    *,
    user,
    conversation_id: str,
    content: str,
    reasoning: str | None,
    sources: list,
    web_sources: list,
    provider: str | None,
    model: str | None,
) -> None:
    """落库助手回答。**失败只记日志，不影响已经发给用户的回答。**

    本地来源与联网来源同存于 `messages.sources`（JSON）：联网项带
    `"kind": "web"` 标记，读取时在 `routes_conversations._message_out` 分流。
    旧数据没有标记 → 按本地来源解析，向后兼容，不需要数据库迁移。
    """
    if not content:
        return
    try:
        combined = [
            *sources,
            *[
                {
                    "kind": "web",
                    **(ws.model_dump(mode="json") if isinstance(ws, WebSource) else dict(ws)),
                }
                for ws in web_sources
            ],
        ]
        db = request.app.state.db
        async with db.session() as session:
            conversation = await repo.get_conversation(
                session, user_id=user.id, conversation_id=uuid.UUID(conversation_id)
            )
            if conversation is None:
                return
            await repo.add_message(
                session,
                conversation=conversation,
                role="assistant",
                content=content,
                reasoning=reasoning,
                sources=combined,
                provider=provider,
                model=model,
            )
    except Exception:  # noqa: BLE001
        logger.exception("助手回答落库失败（回答已正常返回给用户）")


# ------------------------------------------------------------------ SSE
@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user: CurrentUser):
    from bidding_qa_chatbot.llm.chain import ProviderChain

    settings: Settings = request.app.state.settings
    chain: ProviderChain = request.app.state.chain
    request_id = uuid.uuid4().hex

    # 附件必须在会话/用户消息落库**之前**解析：OCR 失败时不产生任何副作用。
    try:
        attachment_ctx = await _load_attachment_context(request, user, body)
    except AttachmentError as exc:
        return _attachment_error_response(request_id, exc)

    # 会话与用户消息在**开流之前**落地（独立事务，见 _resolve_conversation 的说明）
    conversation_id = await _resolve_conversation(
        request, user, body, attachments=attachment_ctx.persisted or None
    )

    agent = getattr(request.app.state, "agent", None)
    use_agent = bool(settings.agent_enabled and agent is not None)

    # 老路径的检索是「先固定检索、再生成」，来源在开流前就已知；
    # Agent 路径的检索发生在工具调用里，来源要等 tool_result 才出现。
    initial_sources: list = []
    initial_elapsed = 0

    if use_agent:
        from bidding_qa_chatbot.llm.prompts import build_agent_messages

        # 提示词与工具集合必须一致：只有真注册/真开启的工具才能出现在提示词里。
        web_tools = {"search_web_keyword", "search_web_semantic"}
        graph_enabled = "query_bidding_graph" in agent.registry
        web_enabled = bool(body.web_search and (web_tools & set(agent.registry.names)))
        allowed_tools = set(agent.registry.names)
        if not web_enabled:
            # 用户没开联网搜索（或服务端没有可用引擎）：连 schema 都不给模型。
            allowed_tools -= web_tools
        source_iter = agent.astream(
            build_agent_messages(
                body.query,
                body.history,
                graph_enabled=graph_enabled,
                web_search_enabled=web_enabled,
                attachment_text=attachment_ctx.current_text,
                prior_attachment_text=attachment_ctx.prior_text,
            ),
            body.mode,
            allowed_tools=allowed_tools,
        )
    else:
        try:
            result = await _retrieve(request, body)
        except HTTPException:
            if not attachment_ctx.has_text:
                raise
            # 附件问题不应被本地编码器缺失拖死：降级为空检索继续作答
            from bidding_qa_chatbot.retrieval.service import RetrievalResult

            logger.warning("检索不可用，但存在附件文本：降级为空检索继续")
            result = RetrievalResult(sources=[], elapsed_ms=0)
        initial_sources = [s.to_ref() for s in result.sources]
        initial_elapsed = result.elapsed_ms
        context = build_context(result.sources, settings.max_context_chars)
        messages = build_messages(
            body.query,
            context,
            body.history,
            attachment_text=attachment_ctx.current_text,
            prior_attachment_text=attachment_ctx.prior_text,
        )
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

    # 边流边累积，用于流结束后把整轮对话落库
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    collected_sources: list = []
    collected_web_sources: list = []
    finished = {"provider": None, "model": None}

    async def generate() -> AsyncIterator[str]:
        # 老路径下参考来源**先于**任何 token 发出，这样「展示参考来源」
        # 完全不依赖 LLM 的健康状况。Agent 路径没有这个前提（检索是模型决定的），
        # 来源会在 tool_result 那一帧带出来。
        if not use_agent:
            collected_sources.extend(initial_sources)
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

                _accumulate(
                    pending,
                    answer_parts,
                    reasoning_parts,
                    collected_sources,
                    collected_web_sources,
                    finished,
                )
                for frame in _frames_for(
                    pending, conversation_id, collected_sources, collected_web_sources
                ):
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
            # 即便中途失败也把已产出的部分存下来——用户已经看到了这段内容，
            # 历史里没有它反而会让回看时对不上。
            with contextlib.suppress(Exception):
                await _persist_assistant(
                    request,
                    user=user,
                    conversation_id=conversation_id,
                    content="".join(answer_parts),
                    reasoning="".join(reasoning_parts) or None,
                    sources=collected_sources,
                    web_sources=collected_web_sources,
                    provider=finished["provider"],
                    model=finished["model"],
                )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "X-Request-ID": request_id},
    )


def _accumulate(
    item,
    answer: list,
    reasoning: list,
    sources: list,
    web_sources: list,
    finished: dict,
) -> None:
    """把流中的条目累积成最终要落库的内容。"""
    from bidding_qa_chatbot.agent.agent import AgentDone, ToolResultNotice

    if isinstance(item, LLMStreamEvent):
        (reasoning if item.kind == "reasoning" else answer).append(item.text)
    elif isinstance(item, ToolResultNotice):
        # Agent 路径下引用来源随工具结果到来。模型常做多次检索（例如先查主问题
        # 再补一个限定词），两次命中同一个 chunk 就会出现重复条目——
        # 那不仅冗余，还会让前端以 chunk_id 为 key 渲染时触发
        # "Encountered two children with the same key"。
        # 按 chunk_id（本地）/ url（联网）去重，保留首次出现。
        seen = {s.chunk_id for s in sources}
        for raw in item.sources:
            source = SourceRef.model_validate(raw)
            if source.chunk_id not in seen:
                seen.add(source.chunk_id)
                sources.append(source)
        seen_urls = {s.url for s in web_sources}
        for raw in item.web_sources:
            web_source = WebSource.model_validate(raw)
            if web_source.url not in seen_urls:
                seen_urls.add(web_source.url)
                web_sources.append(web_source)
    elif isinstance(item, (AgentDone, StreamEnd)):
        # 两者都带 provider/model：老路径收尾是 StreamEnd，Agent 路径是 AgentDone
        finished["provider"] = item.provider
        finished["model"] = item.model


def _frames_for(
    item,
    conversation_id: str | None = None,
    collected_sources: list | None = None,
    collected_web_sources: list | None = None,
) -> list[str]:
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
        # 参考来源随工具结果一并推给前端。**发累计列表**：多轮工具调用时，
        # 前端按事件整体替换，若只发当前这批，后一次调用会把前一批（例如知识库
        # 结果被联网结果）清空——持久化的一直是累计并集，展示也应当一致。
        if collected_sources:
            frames.append(
                encode_event(
                    "sources",
                    SourcesEvent(
                        sources=[SourceRef.model_validate(s) for s in collected_sources],
                        retrieved=len(collected_sources),
                        elapsed_ms=item.elapsed_ms,
                    ),
                )
            )
        if collected_web_sources:
            frames.append(
                encode_event(
                    "web_sources",
                    WebSourcesEvent(
                        sources=[WebSource.model_validate(s) for s in collected_web_sources],
                        retrieved=len(collected_web_sources),
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
                    conversation_id=conversation_id,
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
                    conversation_id=conversation_id,
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
async def chat(
    body: ChatRequest, request: Request, user: CurrentUser
) -> ChatResponse | JSONResponse:
    import time

    settings: Settings = request.app.state.settings
    chain = request.app.state.chain
    request_id = uuid.uuid4().hex
    started = time.perf_counter()

    try:
        attachment_ctx = await _load_attachment_context(request, user, body)
    except AttachmentError as exc:
        return _attachment_error_response(request_id, exc)

    conversation_id = await _resolve_conversation(
        request, user, body, attachments=attachment_ctx.persisted or None
    )

    try:
        result = await _retrieve(request, body)
    except HTTPException:
        if not attachment_ctx.has_text:
            raise
        from bidding_qa_chatbot.retrieval.service import RetrievalResult

        logger.warning("检索不可用，但存在附件文本：降级为空检索继续")
        result = RetrievalResult(sources=[], elapsed_ms=0)

    context = build_context(result.sources, settings.max_context_chars)
    messages = build_messages(
        body.query,
        context,
        body.history,
        attachment_text=attachment_ctx.current_text,
        prior_attachment_text=attachment_ctx.prior_text,
    )
    sources = [s.to_ref() for s in result.sources]

    if not result.sources and not attachment_ctx.has_text:
        await _persist_assistant(
            request,
            user=user,
            conversation_id=conversation_id,
            content=NO_CONTEXT_REPLY,
            reasoning=None,
            sources=[],
            web_sources=[],
            provider="none",
            model="none",
        )
        return ChatResponse(
            answer=NO_CONTEXT_REPLY,
            sources=[],
            provider="none",
            model="none",
            finish_reason="no_context",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            request_id=request_id,
            conversation_id=conversation_id,
        )

    try:
        llm_result = await chain.complete(LLMRequest(messages=messages, mode=body.mode))
    except AllProvidersFailed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    await _persist_assistant(
        request,
        user=user,
        conversation_id=conversation_id,
        content=llm_result.text,
        reasoning=llm_result.reasoning,
        sources=[s.model_dump(mode="json") for s in sources],
        web_sources=[],
        provider=llm_result.provider,
        model=llm_result.model,
    )

    return ChatResponse(
        answer=llm_result.text,
        reasoning=llm_result.reasoning,
        sources=sources,
        provider=llm_result.provider,
        model=llm_result.model,
        finish_reason=llm_result.finish_reason,
        usage=_to_usage(llm_result.usage),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        request_id=request_id,
        conversation_id=conversation_id,
    )
