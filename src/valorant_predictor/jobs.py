from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import JOB_STATUS_PATH


JobTarget = Callable[[Callable[[str, int, int, str], None]], dict]


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


class JobManager:
    def __init__(self, status_path: str | Path = JOB_STATUS_PATH):
        self.status_path = Path(status_path)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predictor-job")
        self.state = self._load_state()
        if self.state.get("status") in {"queued", "running"}:
            self.state.update(
                {
                    "status": "interrupted",
                    "message": "The app restarted before this job finished.",
                    "finished_at": _now(),
                }
            )
            self._save_state()

    def _load_state(self) -> dict:
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.status_path)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state, default=str))

    def update(
        self,
        step: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        with self.lock:
            self.state.update(
                {
                    "status": "running",
                    "step": step,
                    "current": int(current),
                    "total": int(total),
                    "progress": float(current / total) if total else 0.0,
                    "message": message,
                    "updated_at": _now(),
                }
            )
            self._save_state()

    def start(self, job_type: str, target: JobTarget) -> tuple[dict, bool]:
        with self.lock:
            if self.state.get("status") in {"queued", "running"}:
                return self.snapshot(), False
            self.state = {
                "job_id": uuid.uuid4().hex,
                "job_type": job_type,
                "status": "queued",
                "step": "queued",
                "current": 0,
                "total": 0,
                "progress": 0.0,
                "message": "Job queued",
                "started_at": _now(),
                "updated_at": _now(),
                "finished_at": "",
                "result": {},
                "error": "",
            }
            self._save_state()
            self.executor.submit(self._run, target)
            return self.snapshot(), True

    def _run(self, target: JobTarget) -> None:
        try:
            self.update("starting", 0, 1, "Starting job")
            result = target(self.update)
            with self.lock:
                self.state.update(
                    {
                        "status": "completed",
                        "step": "complete",
                        "current": 1,
                        "total": 1,
                        "progress": 1.0,
                        "message": result.get("message", "Job completed"),
                        "result": result,
                        "finished_at": _now(),
                        "updated_at": _now(),
                    }
                )
                self._save_state()
        except Exception as exc:
            with self.lock:
                self.state.update(
                    {
                        "status": "failed",
                        "message": str(exc),
                        "error": str(exc),
                        "finished_at": _now(),
                        "updated_at": _now(),
                    }
                )
                self._save_state()


job_manager = JobManager()
