"""Record step duration / cost into the runs table so the cost table matches the logs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from recast.store.db import Store


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def timed_run(
    store: Store | None,
    *,
    task_id: str | None,
    drama_id: str | None,
    step: str,
    model: str | None = None,
    cost: float = 0,
    message: str | None = None,
) -> Iterator[dict[str, Any]]:
    t0 = time.perf_counter()
    started = _now()
    rec: dict[str, Any] = {
        "step": step,
        "model": model,
        "cost": cost,
        "message": message,
        "started_at": started,
        "status": "running",
    }
    row_id = None
    if store is not None:
        row = store.record_run(
            task_id=task_id,
            drama_id=drama_id,
            step=step,
            status="running",
            model=model,
            cost=cost,
            message=message,
            started_at=started,
            finished_at=None,
        )
        row_id = row["id"]
        rec["id"] = row_id
    try:
        yield rec
    except Exception as exc:
        dur = time.perf_counter() - t0
        rec["duration_sec"] = round(dur, 3)
        rec["status"] = "failed"
        rec["error"] = str(exc)
        rec["finished_at"] = _now()
        if store is not None and row_id:
            store.update(
                "runs",
                row_id,
                status="failed",
                duration_sec=rec["duration_sec"],
                error=str(exc),
                finished_at=rec["finished_at"],
                cost=rec.get("cost") or 0,
                model=rec.get("model"),
                message=rec.get("message"),
            )
        raise
    else:
        dur = time.perf_counter() - t0
        rec["duration_sec"] = round(dur, 3)
        rec["status"] = rec.get("status") or "ok"
        rec["finished_at"] = _now()
        if store is not None and row_id:
            store.update(
                "runs",
                row_id,
                status=rec["status"],
                duration_sec=rec["duration_sec"],
                finished_at=rec["finished_at"],
                cost=rec.get("cost") or 0,
                model=rec.get("model"),
                message=rec.get("message"),
                error=rec.get("error"),
            )
