"""Content fingerprints used by the checkpoint resume logic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Hash the first and last chunks of large files so resume checks stay cheap.
_FULL_LIMIT = 2 * 1024 * 1024
_HEAD = 1024 * 1024
_TAIL = 64 * 1024


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    if not src.is_file():
        return {"path": str(src), "exists": False, "size": 0, "mtime_ns": 0, "sha256": ""}
    st = src.stat()
    digest = hashlib.sha256()
    digest.update(f"{st.st_size}|".encode("utf-8"))
    with src.open("rb") as fh:
        if st.st_size <= _FULL_LIMIT:
            digest.update(fh.read())
        else:
            digest.update(fh.read(_HEAD))
            fh.seek(max(0, st.st_size - _TAIL))
            digest.update(b"")  # separator
            digest.update(fh.read(_TAIL))
    return {
        "path": str(src),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "sha256": digest.hexdigest(),
    }


def fingerprint_many(paths: list[str | Path] | tuple[str | Path, ...]) -> list[dict[str, Any]]:
    return [file_fingerprint(p) for p in paths]


def fingerprints_match(recorded: list[dict[str, Any]] | None, current: list[dict[str, Any]]) -> bool:
    if not recorded:
        return False
    if len(recorded) != len(current):
        return False
    rec = sorted(recorded, key=lambda x: str(x.get("path") or ""))
    cur = sorted(current, key=lambda x: str(x.get("path") or ""))
    for a, b in zip(rec, cur):
        if not b.get("exists"):
            return False
        if a.get("sha256") and b.get("sha256") and a["sha256"] == b["sha256"] and a.get("size") == b.get("size"):
            continue
        return False
    return True


def dumps(fps: list[dict[str, Any]]) -> str:
    return json.dumps(fps, ensure_ascii=False)


def loads(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    return list(data) if isinstance(data, list) else []
