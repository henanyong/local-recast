"""Optional last-step upload. Default OFF. Only the user's own destination.

Provides copy-to-folder and an optional user-defined HTTP webhook / command.
No official-platform login is built in.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")


def upload_default_off(settings: Any) -> bool:
    """True when the upload toggle is off (the default)."""
    return not bool(getattr(settings, "auto_upload", False))


def run_upload(
    source: str | Path,
    *,
    enabled: bool,
    mode: str = "folder",
    folder: str = "",
    webhook: str = "",
    command: str = "",
) -> dict[str, Any]:
    if not enabled:
        return {"ok": True, "skipped": True, "message": "自动上传关闭（默认关闭）。"}
    src = Path(source)
    if not src.exists():
        return {"ok": False, "skipped": False, "error": f"找不到要上传的文件: {src}"}
    mode = (mode or "folder").lower()
    if mode == "folder":
        if not folder:
            return {"ok": False, "error": "未指定上传目标文件夹。"}
        dest_dir = Path(folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if src.is_dir():
            target = dest_dir / src.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            dest = target
        else:
            shutil.copy2(src, dest)
        log.info("已复制到目标文件夹: %s", dest)
        return {"ok": True, "skipped": False, "dest": str(dest), "mode": "folder"}
    if mode == "webhook":
        if not webhook:
            return {"ok": False, "error": "未指定上传 webhook。"}
        data = src.read_bytes() if src.is_file() else b""
        req = urllib.request.Request(webhook, data=data, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("X-Recast-Filename", src.name)
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", 200)
        return {"ok": 200 <= status < 300, "skipped": False, "status": status, "mode": "webhook"}
    if mode == "command":
        if not command:
            return {"ok": False, "error": "未指定上传命令。"}
        # command may contain {src} and {name}
        cmd = command.format(src=str(src), name=src.name)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {
            "ok": proc.returncode == 0,
            "skipped": False,
            "mode": "command",
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
    return {"ok": False, "error": f"未知上传方式: {mode}"}
