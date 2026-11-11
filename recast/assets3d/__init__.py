"""L2 3D头像 and L3 3D道具生成.

L1 remains 「人物替换」(face swap) and must never be described as 「提取了3D模型」.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recast.assets3d.avatar import ASSET_LABEL, build_avatar, load_avatar
from recast.assets3d.drive import describe_drive, drive_avatar, liveportrait_available
from recast.assets3d.gen3d import generate_prop

AVAILABLE = True
LEVEL = "L2+L3"
STATUS = "已建设"


def describe() -> dict[str, Any]:
    drive = describe_drive()
    return {
        "available": True,
        "l2_avatar": ASSET_LABEL,
        "l3_props": "3D道具生成（试点）",
        "l2_drive": drive,
        "note": (
            "L2 生成的是「3D头像资产」（关键点 + 漫反射贴图 + 驱动配置），"
            "不是生产级完整网格，也不是从视频里提取了3D模型。"
            "L3 仅用于道具/场景试点，占位结果不会被当成成品角色模型。"
        ),
    }


def asset_status(path: str | Path | None = None) -> str:
    if not path:
        return "缺失"
    data = load_avatar(path)
    if not data:
        p = Path(path)
        if p.is_file() or (p.is_dir() and (p / "asset.json").is_file()):
            return "已齐"
        return "缺失"
    score = float(data.get("quality_score") or 0)
    if score and score < 0.55:
        return "偏低"
    return "已齐"
