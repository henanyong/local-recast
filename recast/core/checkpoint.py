"""Per-step checkpoints: step name, input fingerprints, output paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from recast.core.fingerprints import file_fingerprint, fingerprint_many, fingerprints_match, loads
from recast.store.db import Store, new_id


def write_checkpoint(
    store: Store,
    task_id: str,
    step: str,
    input_paths: Iterable[str | Path],
    output_paths: Iterable[str | Path],
) -> dict[str, Any]:
    inputs = fingerprint_many(list(input_paths))
    outputs = [str(Path(p)) for p in output_paths]
    existing = store.connect().execute(
        "SELECT * FROM checkpoints WHERE task_id = ? AND step = ?",
        (task_id, step),
    ).fetchone()
    payload = {
        "input_hashes": json.dumps(inputs, ensure_ascii=False),
        "output_paths": json.dumps(outputs, ensure_ascii=False),
    }
    if existing:
        store.update("checkpoints", existing["id"], **payload)
        return store.get("checkpoints", existing["id"])  # type: ignore[return-value]
    return store.insert("checkpoints", id=new_id(), task_id=task_id, step=step, **payload)


def load_checkpoint(store: Store, task_id: str, step: str) -> dict[str, Any] | None:
    cur = store.connect().execute(
        "SELECT * FROM checkpoints WHERE task_id = ? AND step = ?",
        (task_id, step),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def outputs_exist(paths: Iterable[str]) -> bool:
    items = list(paths)
    if not items:
        return False
    return all(Path(p).exists() for p in items)


def should_skip(
    store: Store | None,
    task_id: str | None,
    step: str,
    input_paths: Iterable[str | Path],
    required_outputs: Iterable[str | Path] | None = None,
    *,
    force: bool = False,
) -> bool:
    """Skip a step when the previous checkpoint's fingerprints still match and outputs exist."""
    if force:
        return False
    if store is None or not task_id:
        # Fallback: if required outputs already exist, still skip (CLI without a store).
        if required_outputs:
            return all(Path(p).is_file() for p in required_outputs)
        return False
    row = load_checkpoint(store, task_id, step)
    if not row:
        return False
    recorded = loads(row.get("input_hashes"))
    current = fingerprint_many(list(input_paths))
    if not fingerprints_match(recorded, current):
        return False
    outputs = []
    try:
        outputs = json.loads(row.get("output_paths") or "[]")
    except Exception:
        outputs = []
    if required_outputs:
        outputs = [str(p) for p in required_outputs]
    return outputs_exist(outputs)


def describe_skip(step: str) -> str:
    return f"步骤「{step}」输入指纹未变且产物仍在，已跳过。"
