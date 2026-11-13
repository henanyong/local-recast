"""Tests for manual GLB import (recast/assets3d/glb.py).

Minimal GLB files are built byte by byte in the test — no real 3D model
needed. A valid GLB is: magic "glTF", version 2, total length, then a JSON
chunk holding {"asset": {"version": "2.0"}}.
"""

import json
import struct
from pathlib import Path

import pytest

from recast.assets3d.glb import GlbError, import_glb, validate_glb
from recast.core.characters import ensure_character_folder, load_meta


def _glb_bytes(*, magic=b"glTF", version=2, asset=None, chunk_type=0x4E4F534A) -> bytes:
    payload = json.dumps(asset if asset is not None else {"asset": {"version": "2.0"}}).encode()
    # Pad the JSON chunk to a multiple of 4 with spaces, as the spec requires.
    pad = (-len(payload)) % 4
    payload += b" " * pad
    body = struct.pack("<II", len(payload), chunk_type) + payload
    total = 12 + len(body)
    return struct.pack("<4sII", magic, version, total) + body


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_valid_glb_passes(tmp_path: Path):
    src = _write(tmp_path / "model.glb", _glb_bytes())
    info = validate_glb(src)
    assert info["format"] == "glb"
    assert info["version"] == 2
    assert info["asset_version"] == "2.0"


def test_bad_magic_refused(tmp_path: Path):
    src = _write(tmp_path / "bad.glb", _glb_bytes(magic=b"XXXX"))
    with pytest.raises(GlbError, match="不是 GLB"):
        validate_glb(src)


def test_wrong_version_refused(tmp_path: Path):
    src = _write(tmp_path / "v1.glb", _glb_bytes(version=1))
    with pytest.raises(GlbError, match="glTF 2.0"):
        validate_glb(src)


def test_length_mismatch_refused(tmp_path: Path):
    data = bytearray(_glb_bytes())
    data[8:12] = struct.pack("<I", len(data) + 100)  # declared length != real size
    src = _write(tmp_path / "bad.glb", bytes(data))
    with pytest.raises(GlbError, match="长度不对"):
        validate_glb(src)


def test_non_json_first_chunk_refused(tmp_path: Path):
    src = _write(tmp_path / "bad.glb", _glb_bytes(chunk_type=0x004E4942))  # "BIN"
    with pytest.raises(GlbError, match="JSON"):
        validate_glb(src)


def test_missing_asset_member_refused(tmp_path: Path):
    src = _write(tmp_path / "bad.glb", _glb_bytes(asset={"scenes": []}))
    with pytest.raises(GlbError, match="asset"):
        validate_glb(src)


def test_missing_file_and_truncated_file(tmp_path: Path):
    with pytest.raises(GlbError, match="找不到"):
        validate_glb(tmp_path / "gone.glb")
    src = _write(tmp_path / "tiny.glb", b"glTF")
    with pytest.raises(GlbError, match="太小"):
        validate_glb(src)


def test_import_copies_and_registers_in_meta(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    src = _write(tmp_path / "hero.glb", _glb_bytes())
    report = import_glb(src, folder)
    dest = Path(report["path"])
    assert dest.is_file() and dest.parent.name == "model"
    assert dest.read_bytes() == src.read_bytes()
    meta = load_meta(folder)
    assert meta is not None
    assert meta["model"]["source"] == "manual"
    assert meta["model"]["format"] == "glb"
    assert meta["model"]["path"] == "model/model.glb"


def test_import_replaces_previous_model(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    first = _write(tmp_path / "v1.glb", _glb_bytes(asset={"asset": {"version": "2.0", "generator": "v1"}}))
    second = _write(tmp_path / "v2.glb", _glb_bytes(asset={"asset": {"version": "2.0", "generator": "v2"}}))
    import_glb(first, folder)
    import_glb(second, folder)
    dest = folder / "model" / "model.glb"
    assert dest.read_bytes() == second.read_bytes()
    meta = load_meta(folder)
    assert meta is not None and "v2.glb" in meta["model"]["note"]


def test_import_missing_folder(tmp_path: Path):
    src = _write(tmp_path / "model.glb", _glb_bytes())
    with pytest.raises(GlbError, match="找不到人物文件夹"):
        import_glb(src, tmp_path / "characters" / "nobody")
