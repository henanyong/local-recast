"""Bind a character's costume to its 3D model (texture registration).

What "binding" means here, honestly: the costume image is *registered* in
the character folder's meta.json ``model.bindings`` — which costume file
goes on which part of the model, in which texture slot — and the texture
file is validated (exists, is a readable image, sane dimensions). The
texture is NOT written into the GLB binary by this module: no GLB
read/write library (pygltflib / trimesh) is installed, and writing one
wrong would silently corrupt the model. See :func:`bind_costume`'s return
value for the exact Windows steps (Blender, or a pygltflib script).

A binding requires a real GLB model in the character folder first
(imported by hand or auto-generated). Binding a costume to the L2
face-card placeholder is refused — there is no mesh to put it on.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

MIN_SIDE = 64
MAX_SIDE = 4096
DEFAULT_SLOT = "base_color"
# Common parts; anything else is accepted as free text, not rejected.
KNOWN_PARTS = {"全身", "上衣", "下装", "裙子", "外套", "鞋", "脸", "头发"}


class BindingError(ValueError):
    """The costume cannot be bound to the character's 3D model."""


WINDOWS_STEPS = (
    "贴图尚未写入 GLB 文件（本机没有 pygltflib / trimesh，不会硬写以免损坏模型）。"
    "在 Windows 上二选一：① Blender：导入 model.glb → 选中对应网格 → 材质里把 Base Color "
    "换成这张贴图 → 导出 GLB；② 装 pygltflib（pip install pygltflib），用脚本把贴图加进 "
    "GLB 的 images/bufferViews 并指向对应材质的 baseColorTexture。"
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def inspect_texture(path: Path) -> dict[str, Any]:
    """Read an image's format/size. Raises BindingError when unusable."""
    try:
        from PIL import Image
    except Exception as exc:
        raise BindingError(f"读贴图需要 Pillow，本机没有装：{exc}") from exc
    try:
        with Image.open(path) as img:
            img.load()
            info = {
                "format": str(img.format or "").lower(),
                "width": int(img.width),
                "height": int(img.height),
            }
    except (OSError, ValueError) as exc:
        raise BindingError(f"贴图读不懂，不是有效的图片：{path}（{exc}）") from exc
    if info["format"] not in {"png", "jpeg", "jpg", "webp"}:
        raise BindingError(
            f"贴图格式不受支持：{path} 是 {info['format'] or '未知格式'}，请用 PNG 或 JPEG。"
        )
    if info["width"] < MIN_SIDE or info["height"] < MIN_SIDE:
        raise BindingError(
            f"贴图太小：{path} 是 {info['width']}x{info['height']}，每边至少 {MIN_SIDE} 像素。"
        )
    if info["width"] > MAX_SIDE or info["height"] > MAX_SIDE:
        raise BindingError(
            f"贴图太大：{path} 是 {info['width']}x{info['height']}，每边最多 {MAX_SIDE} 像素。"
        )
    return info


def bind_costume(
    folder: str | Path,
    *,
    costume_file: str | Path,
    part: str,
    texture_slot: str | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Register one costume texture on the character's GLB model.

    ``costume_file`` must be an image inside the character folder
    (costume/ or cutout/). ``part`` names the model part the texture goes
    on (e.g. "全身" / "上衣"). Same costume + part replaces the old entry.
    ``source`` is the wardrobe path the texture was picked from, when it
    was copied into the folder first (e.g. "wardrobe/red.png"); it is
    stored so the assembly view can tell the pick is bound.
    """
    from recast.assets3d.glb import validate_glb
    from recast.core.characters import load_meta, save_meta

    folder = Path(folder)
    if not folder.is_dir():
        raise BindingError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
    part = str(part or "").strip()
    if not part:
        raise BindingError("请写清楚贴图绑到模型的哪个部位（比如“全身”“上衣”）。")
    slot = str(texture_slot or DEFAULT_SLOT).strip() or DEFAULT_SLOT
    meta = load_meta(folder) or {}
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    model_path = str(model.get("path") or "").strip()
    if not model_path or not model_path.lower().endswith(".glb"):
        raise BindingError(
            "人物还没有可用的 GLB 模型（只有脸卡占位或没有模型），请先导入或生成 3D 模型再绑定服装。"
        )
    model_file = folder / model_path
    if not model_file.is_file():
        raise BindingError(f"登记的模型文件不存在：{model_file}。请重新导入模型。")
    try:
        validate_glb(model_file)
    except ValueError as exc:
        raise BindingError(f"登记的模型文件已损坏，无法绑定服装：{exc}") from exc

    src = Path(str(costume_file)).expanduser()
    if src.is_symlink():
        raise BindingError(f"拒绝绑定符号链接：{src}")
    try:
        resolved = src.resolve()
    except OSError as exc:
        raise BindingError(f"贴图路径解析失败：{src}（{exc}）") from exc
    if not resolved.is_file():
        raise BindingError(f"找不到贴图文件：{src}")
    try:
        rel = resolved.relative_to(folder.resolve())
    except ValueError:
        raise BindingError(f"贴图必须在人物文件夹里面，拒绝文件夹外的文件：{src}") from None
    info = inspect_texture(resolved)
    warnings: list[str] = []
    if not (_is_power_of_two(info["width"]) and _is_power_of_two(info["height"])):
        warnings.append(
            f"贴图尺寸 {info['width']}x{info['height']} 不是 2 的次方，个别引擎加载会慢或自动缩放，建议改成 512/1024。"
        )
    if part not in KNOWN_PARTS:
        warnings.append(f"部位「{part}」不在常用列表里，已按原样登记。")

    bindings = list(model.get("bindings") or [])
    entry = {
        "costume": str(rel).replace("\\", "/"),
        "part": part,
        "texture_slot": slot,
        "model": model_path,
        "source": str(source or "").replace("\\", "/"),
        "width": info["width"],
        "height": info["height"],
        "format": info["format"],
        "applied_in_glb": False,
        "updated_at": _now(),
    }
    bindings = [
        b for b in bindings
        if not (isinstance(b, dict) and b.get("costume") == entry["costume"] and b.get("part") == part)
    ]
    bindings.append(entry)
    meta["model"] = {**model, "bindings": bindings}
    save_meta(folder, meta)
    log.info("人物「%s」服装已绑定：%s -> %s（%s）。", meta.get("name") or folder.name, rel, part, slot)
    return {
        "character": str(meta.get("name") or folder.name),
        "costume": str(rel).replace("\\", "/"),
        "part": part,
        "texture_slot": slot,
        "model": model_path,
        "source": str(source or "").replace("\\", "/"),
        "width": info["width"],
        "height": info["height"],
        "format": info["format"],
        "applied_in_glb": False,
        "warnings": warnings,
        "note": WINDOWS_STEPS,
    }


def list_bindings(folder: str | Path) -> list[dict[str, Any]]:
    """All costume bindings registered in the character folder's meta.json."""
    from recast.core.characters import load_meta

    meta = load_meta(Path(folder)) or {}
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    bindings = model.get("bindings") or []
    return [dict(b) for b in bindings if isinstance(b, dict)]
