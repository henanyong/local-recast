"""Local wardrobe + body library scanning (recast/core/wardrobe.py)."""

import json
import struct
from pathlib import Path

from PIL import Image

from recast.core.characters import ensure_character_folder
from recast.core.wardrobe import list_bodies, list_wardrobe


def _image(path: Path, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def _glb(path: Path) -> Path:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


def test_wardrobe_scans_characters_and_shared(tmp_path: Path):
    root = tmp_path / "out"
    folder_a, _ = ensure_character_folder(root, "温念")
    folder_b, _ = ensure_character_folder(root, "温雪")
    _image(folder_a / "costume" / "red.png")
    _image(folder_a / "costume" / "blue.png")
    _image(folder_b / "costume" / "green.png")
    shared = _image(root / "wardrobe" / "shared.png")

    items = list_wardrobe(root)
    # Sorted by (character, name); "共享衣柜" sorts before "温念" in Unicode order.
    assert [item["name"] for item in items] == ["shared.png", "blue.png", "red.png", "green.png"]
    by_name = {item["name"]: item for item in items}
    assert by_name["red.png"]["character"] == "温念"
    assert by_name["green.png"]["character"] == "温雪"
    assert by_name["shared.png"]["character"] == "共享衣柜"
    # The image itself is the thumbnail.
    assert by_name["shared.png"]["preview"] == str(shared)
    # Same file listed in meta + found by dir scan is not duplicated.
    assert len(items) == 4


def test_bodies_scan_characters_and_shared(tmp_path: Path):
    root = tmp_path / "out"
    folder_a, _ = ensure_character_folder(root, "温念")
    _glb(folder_a / "model" / "model.glb")
    # A rendered preview is picked up when present.
    _image(folder_a / "model" / "preview.png")
    shared = _glb(root / "models" / "fox.glb")

    items = list_bodies(root)
    assert len(items) == 2
    by_name = {item["name"]: item for item in items}
    assert by_name["model.glb"]["character"] == "温念"
    assert by_name["model.glb"]["preview"] == str(folder_a / "model" / "preview.png")
    assert by_name["fox.glb"]["character"] == "共享模型库"
    assert by_name["fox.glb"]["preview"] is None
    assert by_name["fox.glb"]["path"] == str(shared)


def test_wardrobe_and_bodies_empty_without_folders(tmp_path: Path):
    assert list_wardrobe(tmp_path / "out") == []
    assert list_bodies(tmp_path / "out") == []
