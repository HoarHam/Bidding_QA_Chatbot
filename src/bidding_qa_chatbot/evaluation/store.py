"""评测结果的文件存储。

结果落在 `data/evaluations/*.json`（`data/` 已被 gitignore）。
运行中的进度只在内存里；完成/失败后原子写入磁盘，服务重启后仍能在
`/evaluate` 的历史列表里看到。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.log import redact

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")


class EvaluationStore:
    def __init__(self, settings: Settings) -> None:
        self.directory: Path = settings.evaluations_dir

    def _path(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError("评测运行 id 格式非法")
        return self.directory / f"{run_id}.json"

    def save(self, run: dict[str, Any]) -> None:
        run_id = str(run.get("id", ""))
        path = self._path(run_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(run, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def load(self, run_id: str) -> dict[str, Any] | None:
        try:
            path = self._path(run_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取评测结果失败 %s: %s", path.name, redact(str(exc)))
            return None
        return data if isinstance(data, dict) else None

    def list_summaries(self, *, limit: int, username: str | None = None) -> list[dict[str, Any]]:
        """列出全部评测结果；`username` 非空时只保留包含该用户用例的记录。

        离线结果不属于任何用户，按用户筛选时不返回。
        """
        if not self.directory.exists():
            return []
        runs: list[dict[str, Any]] = []
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("跳过损坏的评测结果 %s: %s", path.name, redact(str(exc)))
                continue
            if not isinstance(data, dict):
                continue
            if username:
                if data.get("mode") != "online":
                    continue
                if not any(str(case.get("username")) == username for case in data.get("cases", [])):
                    continue
            runs.append(summary_of(data))
        runs.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return runs[: max(1, limit)]

    def all_usernames(self) -> dict[str, int]:
        """历史中出现的用户名及用例数，供页面的用户筛选下拉框使用。"""
        if not self.directory.exists():
            return {}
        counts: dict[str, int] = {}
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for case in data.get("cases", []) if isinstance(data, dict) else []:
                name = str(case.get("username") or "").strip()
                if name:
                    counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))


def summary_of(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": run.get("id"),
        "mode": run.get("mode"),
        "status": run.get("status"),
        "created_at": run.get("created_at"),
        "finished_at": run.get("finished_at"),
        "judge": run.get("judge"),
        "evaluated": run.get("evaluated"),
        "summary": run.get("summary") or {},
        "progress": run.get("progress") or {},
        "error": run.get("error"),
        "users": sorted(
            {str(case.get("username")) for case in run.get("cases", []) if case.get("username")}
        ),
    }
