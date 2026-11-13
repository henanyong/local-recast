"""Local wardrobe + body library: every costume and every GLB in reach.

The assembly view lets the user pick a costume / body from *all* characters,
not just the selected one, plus two shared libraries under the output root::

    output/wardrobe/   shared costumes (optional)
    output/models/     shared GLB bodies (optional)

Items are plain dicts so the UI and the service layer share one shape:

    {"path": str, "character": str, "name": str, "preview": str | None}

``preview`` is the image itself for a costume, and the cached
``model/preview.png`` for a body (None when no preview has been rendered
yet — the UI then shows a placeholder card, never a fake picture).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recast.core.character_assets import resolve_character_assets
from recast.core.characters import characters_root, list_character_folders
from recast.core.constants import IMAGE_EXTS

WARDROBE_DIR_NAME = "wardrobe"
MODELS_DIR_NAME = "models"


def _scan_dir(folder: Path, suffixes: set[str]) -> list[Path]:
    if not folder.is_dir():
        return []
    try:
        items = sorted(folder.iterdir(), key=lambda p: p.name)
    except OSError:
        return []
    return [p for p in items if p.is_file() and p.suffix.lower() in suffixes]


def list_wardrobe(output_root: str | Path) -> list[dict[str, Any]]:
    """Every costume image: each character's costume/ + shared wardrobe/."""
    root = Path(output_root).expanduser().resolve()
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(path: Path, character: str) -> None:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            return
        seen.add(key)
        items.append(
            {
                "path": str(path),
                "character": character,
                "name": path.name,
                "preview": str(path),
            }
        )

    for entry in list_character_folders(root):
        folder = Path(str(entry["folder"]))
        name = str((entry.get("meta") or {}).get("name") or entry["name"])
        files: list[Path] = []
        try:
            assets = resolve_character_assets(root, entry["name"])
        except Exception:
            assets = None
        if assets is not None:
            files = list(assets.costume_files)
        else:
            files = _scan_dir(folder / "costume", set(IMAGE_EXTS))
        for path in files:
            _add(path, name)
    for path in _scan_dir(root / WARDROBE_DIR_NAME, set(IMAGE_EXTS)):
        _add(path, "共享衣柜")
    items.sort(key=lambda item: (item["character"], item["name"]))
    return items


def list_bodies(output_root: str | Path) -> list[dict[str, Any]]:
    """Every GLB body: each character's model/ + shared models/."""
    root = Path(output_root).expanduser().resolve()
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(path: Path, character: str, folder: Path | None) -> None:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            return
        seen.add(key)
        preview: str | None = None
        if folder is not None:
            candidate = folder / "model" / "preview.png"
            if candidate.is_file():
                preview = str(candidate)
        items.append(
            {
                "path": str(path),
                "character": character,
                "name": path.name,
                "preview": preview,
            }
        )

    for entry in list_character_folders(root):
        folder = Path(str(entry["folder"]))
        name = str((entry.get("meta") or {}).get("name") or entry["name"])
        glbs: list[Path] = []
        try:
            assets = resolve_character_assets(root, entry["name"])
        except Exception:
            assets = None
        if assets is not None and assets.model_path is not None and assets.model_path.suffix.lower() == ".glb":
            glbs = [assets.model_path]
        if not glbs:
            glbs = _scan_dir(folder / "model", {".glb"})
        # Costume variants are whole-body GLBs too (model/variants/<变体名>.glb):
        # the assembly body list is where the user switches between them.
        glbs = [*glbs, *_scan_dir(folder / "model" / "variants", {".glb"})]
        for path in glbs:
            _add(path, name, folder)
    for path in _scan_dir(root / MODELS_DIR_NAME, {".glb"}):
        _add(path, "共享模型库", None)
    items.sort(key=lambda item: (item["character"], item["name"]))
    return items
