"""L2 「3D头像」asset builder.

Produces a reusable head asset bound to a character id: params, landmarks,
albedo crop, a simple face-card OBJ (not a licensed FLAME mesh), a LivePortrait
drive config, quality score and source note.

Honesty: a landmark + albedo + drive-config asset is 「3D头像资产」, not a
production-ready full mesh, unless the user drops in an official FLAME model.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from recast.assets3d.landmarks import crop_albedo, extract_landmarks, quality_score, write_json

ASSET_LABEL = "3D头像资产"
ASSET_KIND = "3d_avatar"
FLAME_NOTE = """如何放入官方 FLAME 模型（可选）
================================
本程序不附带 FLAME / SMPL-X 等许可证受限模型。

若你已从官方渠道取得 FLAME（https://flame.is.tue.mpg.de/）：
1. 把 generic_model.pkl（或官方发布的等价文件）放到本项目的 models/flame/ 目录
2. 重新对角色执行「生成 3D头像资产」
3. 若检测成功，资产的 production_ready_mesh 仍默认为 false，直到你在资产
   说明里自行确认；本工具不会声称已生成可商用的完整头部网格

没有 FLAME 时，资产包含：
- landmarks.json    2D/5 点关键点
- albedo.png        漫反射脸部裁切
- head_card.obj     带贴图的脸卡（占位网格，不是解剖网格）
- drive_config.json 给 LivePortrait（或同类）驱动用的配置
这称为「3D头像资产」，不是生产级完整网格。
"""


def flame_available(models_dir: str | Path | None = None) -> Path | None:
    roots = []
    if models_dir:
        roots.append(Path(models_dir) / "flame")
    try:
        from recast.config import PROJECT_ROOT

        roots.append(PROJECT_ROOT / "models" / "flame")
    except Exception:
        pass
    names = ("generic_model.pkl", "FLAME_generic.pkl", "flame2020.pkl", "generic.pkl")
    for root in roots:
        if not root.is_dir():
            continue
        for n in names:
            p = root / n
            if p.is_file():
                return p
        found = list(root.glob("*.pkl"))
        if found:
            return found[0]
    return None


def _write_face_card_obj(dest: Path, albedo_name: str = "albedo.png") -> Path:
    """A unit quad face card with UV, clearly a placeholder mesh."""
    obj = dest
    mtl = dest.with_suffix(".mtl")
    mtl.write_text(
        f"newmtl albedo\nKd 1 1 1\nmap_Kd {albedo_name}\n",
        encoding="utf-8",
    )
    obj.write_text(
        "\n".join(
            [
                f"mtllib {mtl.name}",
                "# Placeholder face card for 本地再铸 L2 3D头像资产. Not a FLAME mesh.",
                "o HeadCard",
                "v -0.5 -0.6 0",
                "v  0.5 -0.6 0",
                "v  0.5  0.6 0",
                "v -0.5  0.6 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "vn 0 0 1",
                "usemtl albedo",
                "f 1/1/1 2/2/1 3/3/1 4/4/1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return obj


def build_avatar(
    character_id: str,
    photos: list[str | Path],
    dest_dir: str | Path,
    *,
    display_name: str = "",
    models_dir: str | Path | None = None,
    source_note: str = "",
) -> dict[str, Any]:
    photos = [Path(p) for p in photos if p and Path(p).is_file()]
    if not photos:
        raise FileNotFoundError("请至少提供一张已授权角色的参考脸照片。")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    primary = photos[0]
    landmarks = extract_landmarks(primary)
    albedo = crop_albedo(primary, landmarks, dest / "albedo.png")
    shutil.copy2(primary, dest / "source.jpg") if primary.suffix.lower() in {".jpg", ".jpeg"} else None
    try:
        from PIL import Image

        Image.open(primary).convert("RGB").save(dest / "source.jpg", quality=92)
    except Exception:
        pass
    write_json(dest / "landmarks.json", landmarks)
    _write_face_card_obj(dest / "head_card.obj", albedo_name="albedo.png")
    # optional glb skip (no extra dep); OBJ is enough
    flame = flame_available(models_dir)
    drive_cfg = {
        "backend": "liveportrait",
        "source_image": "albedo.png",
        "required": False,
        "installed": _liveportrait_installed(),
        "note": "若已安装 LivePortrait，可用本配置驱动；未安装时资产仍保留，待以后驱动。",
    }
    write_json(dest / "drive_config.json", drive_cfg)
    (dest / "FLAME说明.txt").write_text(FLAME_NOTE, encoding="utf-8")

    q = quality_score(landmarks, n_photos=len(photos))
    low = q < 0.55
    asset = {
        "character_id": character_id,
        "display_name": display_name,
        "kind": ASSET_KIND,
        "label": ASSET_LABEL,
        "production_ready_mesh": False,
        "backend": "landmarks+albedo+drive_config" + ("+flame-detected" if flame else ""),
        "flame_model": str(flame) if flame else None,
        "quality_score": q,
        "quality_flag": "偏低" if low else "可用",
        "reusable": True,
        "photos": [str(p) for p in photos],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_note": source_note or f"由参考图 {primary.name} 生成的 {ASSET_LABEL}",
        "honesty": (
            "这是「3D头像资产」（关键点 + 漫反射贴图 + 驱动配置），"
            "不是生产级完整头部网格。"
            + (" 检测到本机 models/flame/ 下有 FLAME 文件，但本工具不会在未获授权时捆绑或改写它。" if flame else " 未检测到官方 FLAME 模型。")
        ),
        "files": {
            "landmarks": "landmarks.json",
            "albedo": "albedo.png",
            "obj": "head_card.obj",
            "drive_config": "drive_config.json",
        },
    }
    write_json(dest / "asset.json", asset)
    (dest / "source_note.txt").write_text(asset["source_note"] + "\n" + asset["honesty"] + "\n", encoding="utf-8")
    return {**asset, "path": str(dest / "asset.json"), "dir": str(dest)}


def load_avatar(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if path.is_dir():
        path = path / "asset.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def bind_path_for(character_id: str, models_or_output: str | Path) -> Path:
    return Path(models_or_output) / "l2" / character_id


def _liveportrait_installed() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("liveportrait") is not None or importlib.util.find_spec("src.live_portrait_pipeline") is not None
    except Exception:
        return False
