"""Manual GLB import: validate a .glb file and register it as a character's 3D model.

GLB is the only manual model format (decided 2026-11-12). Validation checks
the real glTF 2.0 binary header — magic ``glTF``, version 2, declared length
equal to the file size, and a JSON first chunk that parses to an object with
an ``asset`` member. A file that only *ends* in .glb is refused.
"""

from __future__ import annotations

import json
import logging
import shutil
import struct
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
GLB_JSON_CHUNK = 0x4E4F534A  # "JSON" in little-endian uint32
_HEADER = struct.Struct("<4sII")
_CHUNK_HEADER = struct.Struct("<II")


class GlbError(ValueError):
    """The file is not a usable GLB model."""


def validate_glb(path: str | Path) -> dict[str, Any]:
    """Check a file is a glTF 2.0 binary model. Returns header info."""
    path = Path(path)
    if not path.is_file():
        raise GlbError(f"找不到 GLB 文件：{path}")
    if path.is_symlink():
        raise GlbError(f"拒绝导入符号链接：{path}")
    size = path.stat().st_size
    if size < _HEADER.size + _CHUNK_HEADER.size:
        raise GlbError(f"GLB 文件太小，不是有效的模型：{path}（{size} 字节）。")
    with path.open("rb") as fh:
        magic, version, length = _HEADER.unpack(fh.read(_HEADER.size))
        if magic != GLB_MAGIC:
            raise GlbError(f"不是 GLB 文件（文件头不是 glTF）：{path}")
        if version != GLB_VERSION:
            raise GlbError(f"只支持 glTF 2.0，该文件版本是 {version}：{path}")
        if length != size:
            raise GlbError(
                f"GLB 文件长度不对：文件头写的是 {length} 字节，实际是 {size} 字节：{path}"
            )
        chunk_length, chunk_type = _CHUNK_HEADER.unpack(fh.read(_CHUNK_HEADER.size))
        if chunk_type != GLB_JSON_CHUNK:
            raise GlbError(f"GLB 第一个块不是 JSON 块：{path}")
        try:
            payload = json.loads(fh.read(chunk_length) or b"null")
        except (ValueError, OSError) as exc:
            raise GlbError(f"GLB 的 JSON 块读不懂：{path}（{exc}）") from exc
    if not isinstance(payload, dict) or "asset" not in payload:
        raise GlbError(f"GLB 的 JSON 块里没有 asset 信息：{path}")
    return {
        "format": "glb",
        "version": version,
        "size": size,
        "asset_version": str((payload.get("asset") or {}).get("version") or ""),
        "generator": str((payload.get("asset") or {}).get("generator") or ""),
    }


def import_glb(
    src: str | Path,
    char_folder: str | Path,
    *,
    dest_name: str = "model.glb",
) -> dict[str, Any]:
    """Copy a validated GLB into the character folder's model/ and register it.

    Replaces any previously imported model (manual import wins). Updates
    meta.json ``model`` to {source: "manual", format: "glb", ...}.
    """
    from recast.core.characters import load_meta, save_meta

    folder = Path(char_folder)
    if not folder.is_dir():
        raise GlbError(f"找不到人物文件夹：{folder}")
    src = Path(src)
    info = validate_glb(src)
    model_dir = folder / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / dest_name
    shutil.copy2(src, dest)
    meta = load_meta(folder) or {"name": folder.name}
    name = meta.get("name") or folder.name
    meta["model"] = {
        "source": "manual",
        "format": "glb",
        "path": f"model/{dest_name}",
        "note": f"手动导入 {src.name}（{info['size']} 字节，glTF {info['version']}.0）。",
    }
    save_meta(folder, meta)
    log.info("人物「%s」已导入 GLB 模型: %s", name, dest)
    return {
        "character": name,
        "path": str(dest),
        "size": info["size"],
        "format": "glb",
        "source": "manual",
    }
