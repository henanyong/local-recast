"""Costume -> GLB texture binding: registration + validation, no fake writes."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from PIL import Image

from recast.assets3d.binding import BindingError, bind_costume, list_bindings
from recast.assets3d.glb import import_glb
from recast.core.characters import ensure_character_folder


def _glb_bytes() -> bytes:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    total = 12 + len(body)
    return struct.pack("<4sII", b"glTF", 2, total) + body


def _image(path: Path, size: tuple[int, int] = (256, 256), color=(200, 60, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _character_with_glb(tmp_path: Path, name: str = "温念") -> Path:
    folder, _ = ensure_character_folder(tmp_path / "out", name)
    src = tmp_path / "model.glb"
    src.write_bytes(_glb_bytes())
    import_glb(src, folder)
    return folder


def test_bind_registers_costume_and_validates_size(tmp_path: Path):
    folder = _character_with_glb(tmp_path)
    tex = _image(folder / "costume" / "red-dress.png", (512, 1024))
    report = bind_costume(folder, costume_file=tex, part="全身")
    assert report["costume"] == "costume/red-dress.png"
    assert report["part"] == "全身"
    assert report["texture_slot"] == "base_color"
    assert report["width"] == 512 and report["height"] == 1024
    assert report["applied_in_glb"] is False
    assert "Blender" in report["note"]
    assert list_bindings(folder)[0]["costume"] == "costume/red-dress.png"
    # Rebinding the same costume + part replaces, not duplicates.
    report = bind_costume(folder, costume_file=tex, part="全身", texture_slot="roughness")
    assert len(list_bindings(folder)) == 1
    assert list_bindings(folder)[0]["texture_slot"] == "roughness"


def test_bind_warns_on_non_power_of_two(tmp_path: Path):
    folder = _character_with_glb(tmp_path)
    tex = _image(folder / "costume" / "odd.png", (300, 200))
    report = bind_costume(folder, costume_file=tex, part="上衣")
    assert any("2 的次方" in w for w in report["warnings"])


def test_bind_refuses_without_a_glb_model(tmp_path: Path):
    folder, _ = ensure_character_folder(tmp_path / "out", "温念")
    tex = _image(folder / "costume" / "dress.png")
    with pytest.raises(BindingError, match="还没有可用的 GLB 模型"):
        bind_costume(folder, costume_file=tex, part="全身")


def test_bind_refuses_outside_files_symlinks_and_bad_images(tmp_path: Path):
    folder = _character_with_glb(tmp_path)
    outside = _image(tmp_path / "outside.png")
    with pytest.raises(BindingError, match="人物文件夹里面"):
        bind_costume(folder, costume_file=outside, part="全身")
    link = folder / "costume" / "link.png"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside)
    except OSError:
        # Windows without admin rights or Developer Mode cannot create
        # symlinks (WinError 1314): skip instead of failing the suite.
        pytest.skip("创建符号链接需要管理员权限或 Windows 开发者模式，跳过。")
    with pytest.raises(BindingError, match="符号链接"):
        bind_costume(folder, costume_file=link, part="全身")
    bad = folder / "costume" / "bad.png"
    bad.write_bytes(b"not an image at all")
    with pytest.raises(BindingError, match="读不懂"):
        bind_costume(folder, costume_file=bad, part="全身")
    with pytest.raises(BindingError, match="找不到贴图"):
        bind_costume(folder, costume_file=folder / "costume" / "missing.png", part="全身")
    with pytest.raises(BindingError, match="哪个部位"):
        bind_costume(folder, costume_file=_image(folder / "costume" / "ok.png"), part="  ")


def test_bind_refuses_too_small_and_too_big(tmp_path: Path):
    folder = _character_with_glb(tmp_path)
    tiny = _image(folder / "costume" / "tiny.png", (32, 32))
    with pytest.raises(BindingError, match="太小"):
        bind_costume(folder, costume_file=tiny, part="全身")
    huge = _image(folder / "costume" / "huge.png", (5000, 100))
    with pytest.raises(BindingError, match="太大"):
        bind_costume(folder, costume_file=huge, part="全身")


def test_bind_accepts_cutout_png_as_texture(tmp_path: Path):
    folder = _character_with_glb(tmp_path)
    tex = _image(folder / "cutout" / "wen.png", (256, 256))
    report = bind_costume(folder, costume_file=tex, part="全身")
    assert report["costume"] == "cutout/wen.png"
