"""ComfyUI HTTP adapter. If the server is not reachable, callers use the animatic fallback."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def comfy_available(base_url: str | None, timeout: float = 1.0) -> bool:
    if not base_url:
        return False
    try:
        with urllib.request.urlopen(_url(base_url, "/system_stats"), timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _request(url: str, data: bytes | None = None, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fill_workflow(workflow: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    """Replace ${var} strings in a ComfyUI workflow JSON."""
    blob = json.dumps(workflow, ensure_ascii=False)
    for key, value in variables.items():
        blob = blob.replace("${" + str(key) + "}", str(value))
    return json.loads(blob)


def submit_workflow(
    base_url: str,
    workflow: dict[str, Any],
    *,
    variables: dict[str, Any] | None = None,
    dest_dir: str | Path | None = None,
    timeout_sec: float = 600.0,
    poll: float = 1.5,
) -> list[Path]:
    """Submit a workflow, wait for completion, and collect output files.

    Returns downloaded file paths. Raises on HTTP / timeout errors.
    """
    if variables:
        workflow = fill_workflow(workflow, variables)
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    raw = _request(_url(base_url, "/prompt"), data=payload, timeout=30)
    info = json.loads(raw.decode("utf-8"))
    prompt_id = info.get("prompt_id") or info.get("promptId")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI 未返回 prompt_id: {info}")
    deadline = time.time() + timeout_sec
    history = {}
    while time.time() < deadline:
        try:
            raw_h = _request(_url(base_url, f"/history/{prompt_id}"), timeout=15)
            history = json.loads(raw_h.decode("utf-8")) or {}
            rec = history.get(prompt_id) or history
            if rec.get("status", {}).get("completed") or rec.get("outputs"):
                break
        except Exception:
            pass
        time.sleep(poll)
    else:
        raise TimeoutError("等待 ComfyUI 工作流超时")
    rec = history.get(prompt_id) or history
    outputs = rec.get("outputs") or {}
    dest = Path(dest_dir or ".")
    dest.mkdir(parents=True, exist_ok=True)
    collected: list[Path] = []
    for node_out in outputs.values():
        for key in ("images", "gifs", "files", "videos"):
            for item in node_out.get(key) or []:
                filename = item.get("filename")
                if not filename:
                    continue
                folder = item.get("subfolder") or ""
                typ = item.get("type") or "output"
                qs = f"filename={filename}&subfolder={folder}&type={typ}"
                try:
                    blob = _request(_url(base_url, f"/view?{qs}"), timeout=60)
                except Exception:
                    continue
                target = dest / filename
                target.write_bytes(blob)
                collected.append(target)
    return collected
