"""Costume variants: one whole-body GLB per outfit, per character.

Honest model of what "clothes" means here: TripoSG generates a whole
person *wearing* an outfit from one image — it cannot generate a separate
garment mesh. So a "costume" is a **dressed-body variant**: the same
character generated again from a different outfit reference image, stored
as ``model/variants/<变体名>.glb`` and registered in meta.json
``model.variants`` (name, path, source reference image, backend,
generated_at). Switching a variant just points ``assembly.body`` at that
variant's GLB, so the assembly view, the 3D preview and downstream code
all read it from the one place they already read the body from.

This reuses the registration-only pattern of recast/assets3d/binding.py:
a variant is *registered* in meta.json, never written into a GLB binary.
Variants are surfaced in the assembly body list by
recast/core/wardrobe.py ``list_bodies`` (it scans ``model/variants/``).

A variant GLB is shape only, like the main model: no texture, no
skeleton, so it cannot be used for body re-rendering either.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from recast.assets3d.generate3d import GenerationError, TripoSGBackend

log = logging.getLogger("recast")

VARIANTS_SUBDIR = Path("model") / "variants"
VARIANT_HONESTY = (
    "衣服是整身变体，不是独立衣片：TripoSG 一次生成穿着这身衣服的整个人，"
    "无贴图、无骨骼。切换变体就是把装配的身体换成对应变体的 GLB。"
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _source_image_label(folder: Path, image: str | Path | None) -> str:
    if image is None:
        return ""
    image = Path(image)
    try:
        return str(image.resolve().relative_to(folder.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return image.name


def variant_filename(name: str) -> str:
    """File name for a variant (Windows-safe, always ``*.glb``)."""
    from recast.core.characters import safe_folder_name

    safe = safe_folder_name(name).strip()
    if not safe or safe == "未命名":
        raise ValueError("变体名字不能为空，请起个名字（比如“红裙”）。")
    return f"{safe}.glb"


def _unique_dest(variants_dir: Path, filename: str) -> Path:
    dest = variants_dir / filename
    if not dest.exists():
        return dest
    stem = dest.stem
    index = 2
    while True:
        candidate = variants_dir / f"{stem}-{index}.glb"
        if not candidate.exists():
            return candidate
        index += 1


def list_costume_variants(folder: str | Path) -> list[dict[str, Any]]:
    """Every registered costume variant for one character folder."""
    from recast.core.characters import load_meta

    meta = load_meta(Path(folder)) or {}
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    variants = model.get("variants") or []
    return [dict(v) for v in variants if isinstance(v, dict)]


def active_costume_variant(folder: str | Path) -> str:
    """Name of the variant the assembly currently uses ("" = main model)."""
    from recast.core.characters import load_meta

    meta = load_meta(Path(folder)) or {}
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    return str(model.get("active_variant") or "").strip()


def register_costume_variant(
    folder: str | Path,
    *,
    name: str,
    glb_path: str | Path,
    source_image: str | Path | None = None,
    backend: str = "",
) -> dict[str, Any]:
    """Register an existing GLB as one character's costume variant.

    ``glb_path`` is validated, then copied into ``model/variants/`` unless
    it is already there. Re-registering the same variant name replaces the
    old entry (the file name is kept when the old file is reused, otherwise
    a fresh unique name is used). The main model and ``assembly.body`` are
    untouched — call :func:`switch_costume_variant` to wear the variant.
    """
    from recast.assets3d.glb import validate_glb
    from recast.core.characters import load_meta, save_meta

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
    name = str(name or "").strip()
    if not name:
        raise ValueError("变体名字不能为空，请起个名字（比如“红裙”）。")
    src = Path(glb_path)
    info = validate_glb(src)
    variants_dir = folder / VARIANTS_SUBDIR
    variants_dir.mkdir(parents=True, exist_ok=True)
    try:
        rel_src = src.resolve().relative_to(variants_dir.resolve())
        dest = variants_dir / rel_src
    except (OSError, ValueError):
        dest = _unique_dest(variants_dir, variant_filename(name))
        shutil.copy2(src, dest)
    rel_path = str(dest.relative_to(folder)).replace("\\", "/")
    meta = load_meta(folder) or {"name": folder.name}
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    variants = [v for v in (model.get("variants") or []) if isinstance(v, dict) and v.get("name") != name]
    entry: dict[str, Any] = {
        "name": name,
        "path": rel_path,
        "source_image": _source_image_label(folder, source_image),
        "backend": str(backend or ""),
        "generated_at": _now(),
        "size": info["size"],
        "note": VARIANT_HONESTY,
    }
    variants.append(entry)
    meta["model"] = {**model, "variants": variants}
    save_meta(folder, meta)
    log.info("人物「%s」已登记换装变体「%s」: %s", meta.get("name") or folder.name, name, dest)
    return {
        "character": str(meta.get("name") or folder.name),
        "variant": name,
        "path": str(dest),
        "source_image": entry["source_image"],
        "backend": entry["backend"],
        "generated_at": entry["generated_at"],
        "size": info["size"],
    }


def generate_costume_variant(
    image: str | Path,
    char_folder: str | Path,
    *,
    variant_name: str,
    backend: Any | None = None,
    repo_dir: str | Path | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Generate one dressed-body variant from an outfit reference image.

    The image should show the whole person wearing the outfit (a full-body
    photo or the character's cutout in that outfit). The backend is
    injectable for tests; the default is TripoSG (needs CUDA + checkout,
    see recast/assets3d/generate3d.py). A failed generation registers
    nothing and leaves no partial file behind.
    """
    from recast.assets3d.glb import validate_glb

    folder = Path(char_folder)
    if not folder.is_dir():
        raise GenerationError(f"找不到人物文件夹：{folder}")
    image = Path(image)
    if not image.is_file():
        raise GenerationError(f"找不到要生成变体的图片：{image}")
    name = str(variant_name or "").strip()
    if not name:
        raise GenerationError("变体名字不能为空，请起个名字（比如“红裙”）。")
    if backend is None:
        backend = TripoSGBackend(repo_dir, python=python)
    variants_dir = folder / VARIANTS_SUBDIR
    variants_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(variants_dir, variant_filename(name))
    try:
        backend.generate(image, dest)
        validate_glb(dest)
    except Exception:
        # No partial file, no registration: a failed variant is invisible.
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return register_costume_variant(
        folder,
        name=name,
        glb_path=dest,
        source_image=image,
        backend=backend.name,
    )


def switch_costume_variant(folder: str | Path, variant_name: str | None) -> dict[str, Any]:
    """Wear a costume variant (or switch back to the main model).

    ``variant_name`` "" / None / "主模型" switches back to the character's
    main ``model/model.glb``. Anything else must be a registered variant
    name. The switch is ``assembly.body`` — the assembly view shows the
    variant as the body right away — plus ``model.active_variant`` in
    meta.json so the choice survives a reload.
    """
    from recast.core.characters import get_assembly, load_meta, save_meta, set_assembly

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
    meta = load_meta(folder) or {"name": folder.name}
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    name = str(variant_name or "").strip()
    if not name or name in ("主模型", "默认"):
        main_rel = str(model.get("path") or "").strip()
        main = folder / main_rel if main_rel else None
        if main is None or not main_rel.lower().endswith(".glb") or not main.is_file():
            raise ValueError(
                "还没有主模型（model/model.glb），无法切回主模型。"
                "请先点「生成 3D 模型」或「导入 3D 模型」。"
            )
        set_assembly(folder, body=str(main))
        meta = load_meta(folder) or meta
        model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
        meta["model"] = {**model, "active_variant": ""}
        save_meta(folder, meta)
        return {
            "character": str(meta.get("name") or folder.name),
            "variant": "",
            "path": main_rel,
            "assembly_body": main_rel,
        }
    variants = list_costume_variants(folder)
    entry = next((v for v in variants if str(v.get("name") or "") == name), None)
    if entry is None:
        available = "、".join(str(v.get("name") or "") for v in variants) or "（还没有变体）"
        raise ValueError(f"没有叫「{name}」的换装变体。已有变体：{available}。")
    rel_path = str(entry.get("path") or "").strip()
    target = folder / rel_path
    if not target.is_file():
        raise ValueError(f"变体「{name}」登记的模型文件不存在：{target}。请重新生成该变体。")
    set_assembly(folder, body=str(target))
    meta = load_meta(folder) or meta
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    meta["model"] = {**model, "active_variant": name}
    save_meta(folder, meta)
    assembly = get_assembly(meta)
    return {
        "character": str(meta.get("name") or folder.name),
        "variant": name,
        "path": rel_path,
        "assembly_body": assembly.get("body") or rel_path,
    }
