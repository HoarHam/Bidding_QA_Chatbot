"""综合评测的 Web 面板与 JSON API。

按需求，`/evaluate` 及其 JSON API 不要求登录：页面直接展示全部评测记录；
在线评测对**每个用户**取最近 10 条带 Langfuse trace 的历史问答，并支持按用户名
筛选运行记录与用例明细。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from bidding_qa_chatbot.evaluation.page import PAGE_HTML
from bidding_qa_chatbot.evaluation.runner import EvaluationError, EvaluationService

router = APIRouter(tags=["evaluate"])


def _service(request: Request) -> EvaluationService:
    service = getattr(request.app.state, "evaluation", None)
    if service is None:
        raise HTTPException(status_code=503, detail="评测服务未初始化")
    return service


@router.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page() -> HTMLResponse:
    return HTMLResponse(PAGE_HTML)


@router.get("/api/evaluate/config")
async def evaluate_config(request: Request) -> dict:
    return _service(request).config()


@router.post("/api/evaluate/offline")
async def start_offline(request: Request) -> dict:
    try:
        return _service(request).start_offline()
    except EvaluationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/api/evaluate/online")
async def start_online(request: Request) -> dict:
    try:
        return _service(request).start_online()
    except EvaluationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/api/evaluate/runs")
async def list_runs(
    request: Request,
    username: str | None = Query(default=None, max_length=64),
) -> list[dict]:
    return _service(request).list_runs(username=username)


@router.get("/api/evaluate/users")
async def list_users(request: Request) -> list[dict]:
    return _service(request).list_users()


@router.get("/api/evaluate/runs/{run_id}")
async def get_run(
    request: Request,
    run_id: str,
    username: str | None = Query(default=None, max_length=64),
) -> dict:
    run = _service(request).get_run(run_id, username=username)
    if run is None:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return run
