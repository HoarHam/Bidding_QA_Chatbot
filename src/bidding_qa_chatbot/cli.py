"""命令行入口：ingest / serve / check。

`pyproject.toml` 的 `[project.scripts]` 指向 `bidding_qa_chatbot:main`，
而 `main()` 在 `__init__.py` 里，会转发到这里。改动入口名时两处都要改。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bidding_qa_chatbot.config import ENV_FILE, ROOT_DIR, get_settings, validate_runtime
from bidding_qa_chatbot.log import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bidding-qa-chatbot", description="招投标智能问答机器人（第一阶段 RAG）"
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="把 Excel 语料入库到 Qdrant")
    p_ingest.add_argument("--xlsx", default=None, help="语料路径，默认 data/processed 下那份")
    p_ingest.add_argument("--recreate", action="store_true", help="先删集合再重建（会清空数据）")
    p_ingest.add_argument("--skip-vision", action="store_true", help="跳过图片解析，仅用文本")

    p_serve = sub.add_parser("serve", help="启动 API 服务")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")

    sub.add_parser("check", help="检查配置与各依赖可达性")
    return parser


async def _cmd_ingest(args: argparse.Namespace) -> int:
    from pathlib import Path

    from bidding_qa_chatbot.ingestion.pipeline import DEFAULT_XLSX, run_ingest
    from bidding_qa_chatbot.llm.vision import VisionDescriber
    from bidding_qa_chatbot.retrieval.qdrant_store import build_client

    settings = get_settings()
    problems = validate_runtime(settings)
    if problems:
        for p in problems:
            print(f"  ✗ {p}", file=sys.stderr)
        return 2

    xlsx = Path(args.xlsx) if args.xlsx else DEFAULT_XLSX
    client = build_client(settings)
    try:
        report = await run_ingest(
            client,
            xlsx,
            settings,
            vision=None if args.skip_vision else VisionDescriber(settings),
            recreate=args.recreate,
            skip_vision=args.skip_vision,
        )
    finally:
        await client.close()

    print("\n入库完成")
    print(f"  数据集      {report.dataset}")
    print(f"  问答行数    {report.rows}")
    print(f"  分块数      {report.chunks}")
    print(
        f"  图片        {report.images}（解析成功 {report.vision_ok}，失败 {report.vision_failed}）"
    )
    print(f"  清理陈旧点  {report.deleted_stale}")
    print(f"  BM25 词条   {report.bm25_terms}")
    print(f"  集合        {report.collection}")
    return 0


async def _cmd_check(_: argparse.Namespace) -> int:
    import httpx

    from bidding_qa_chatbot.api.routes_health import health_deps_for_cli

    settings = get_settings()
    print(f"配置文件   {ENV_FILE}  ({'存在' if ENV_FILE.exists() else '不存在'})")
    print(f"项目根目录 {ROOT_DIR}")
    print(f"provider   {' -> '.join(settings.fallback_chain)}")
    print(f"集合       {settings.qdrant_collection} ({settings.qdrant_vector_size} 维)")

    problems = validate_runtime(settings)
    if problems:
        print("\n配置问题:")
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("\n配置校验通过")

    print("\n依赖可达性:")
    async with httpx.AsyncClient() as client:
        for dep in await health_deps_for_cli(settings, client):
            latency = f"{dep.latency_ms}ms" if dep.latency_ms is not None else "-"
            line = f"  {dep.name:<10} {dep.status:<15} {latency:>8}"
            if dep.message:
                line += f"  {dep.message}"
            print(line)

    return 1 if problems else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"启动 API 服务 http://{host}:{port}  (文档 /docs)")
    uvicorn.run(
        "bidding_qa_chatbot.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "ingest":
        return asyncio.run(_cmd_ingest(args))
    if args.command == "check":
        return asyncio.run(_cmd_check(args))
    return 2
