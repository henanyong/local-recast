"""Optional LivePortrait (or similar) drive hook for an L2 3D头像资产."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def liveportrait_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("liveportrait") is not None
    except Exception:
        return False


def describe_drive() -> dict[str, Any]:
    ok = liveportrait_available()
    return {
        "available": ok,
        "backend": "liveportrait" if ok else None,
        "message": (
            "已检测到 LivePortrait，可以对 3D头像资产做驱动。"
            if ok
            else "未安装 LivePortrait。3D头像资产仍会保存，以后安装驱动后再用。可在设置 / 自检中查看。"
        ),
    }


def drive_avatar(
    asset_dir: str | Path,
    driving: str | Path,
    dest: str | Path,
) -> dict[str, Any]:
    info = describe_drive()
    if not info["available"]:
        raise RuntimeError(info["message"])
    # Real LivePortrait APIs vary by install; keep a clear hook.
    raise RuntimeError(
        "本机已安装 LivePortrait，但需要按其官方入口自行配置驱动脚本。"
        "资产目录："
        f" {asset_dir}，驱动视频：{driving}，输出：{dest}。"
    )
