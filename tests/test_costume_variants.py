"""Costume variants (recast/assets3d/variants.py): whole-body GLBs per outfit.

Clothes are NOT separate garment meshes — TripoSG can only generate a
whole person wearing an outfit. A variant is one dressed-body GLB
registered in meta.json ``model.variants``; switching one just points
``assembly.body`` at its GLB. Generation is exercised with a stub backend
that writes a minimal valid GLB (no real TripoSG run here).
"""

import json
import struct
from pathlib import Path

import pytest

from recast.assets3d.generate3d import GenerationError, generate_model_glb
from recast.assets3d.variants import (
    active_costume_variant,
    generate_costume_variant,
    list_costume_variants,
    register_costume_variant,
    switch_costume_variant,
)
from recast.core.characters import ensure_character_folder, get_assembly, load_meta
from recast.core.wardrobe import list_bodies


def _glb_bytes() -> bytes:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


class StubBackend:
    name = "stub"

    def check(self):
        return True, ""

    def generate(self, image, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_glb_bytes())
        return dest


class FailingBackend:
    name = "broken"

    def check(self):
        return False, "缺模型"

    def generate(self, image, dest):
        raise GenerationError("缺模型")


def _folder_with_main(tmp_path: Path) -> Path:
    """Character folder with a stub-generated main model (assembly.body set)."""
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    img = tmp_path / "cutout.png"
    img.write_bytes(b"fake-png")
    generate_model_glb(img, folder, backend=StubBackend())
    return folder


def test_register_variant_copies_glb_and_records_meta(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    src = tmp_path / "red-dress.glb"
    src.write_bytes(_glb_bytes())
    ref = tmp_path / "red-dress.png"
    ref.write_bytes(b"fake")
    report = register_costume_variant(
        folder, name="红裙", glb_path=src, source_image=ref, backend="stub"
    )
    dest = Path(report["path"])
    assert dest.is_file()
    assert dest.parent.name == "variants"
    assert dest.name == "红裙.glb"
    variants = list_costume_variants(folder)
    assert len(variants) == 1
    entry = variants[0]
    assert entry["name"] == "红裙"
    assert entry["path"] == "model/variants/红裙.glb"
    assert entry["source_image"] == "red-dress.png"
    assert entry["backend"] == "stub"
    assert entry["generated_at"]
    assert "不是独立衣片" in entry["note"]
    # Registering a variant does not touch the assembly or the main model.
    assert get_assembly(load_meta(folder))["body"] == ""


def test_generate_variant_with_stub_backend_registers(tmp_path: Path):
    folder = _folder_with_main(tmp_path)
    img = tmp_path / "outfit.png"
    img.write_bytes(b"fake-png")
    report = generate_costume_variant(img, folder, variant_name="蓝衣", backend=StubBackend())
    assert report["variant"] == "蓝衣"
    assert Path(report["path"]).is_file()
    assert len(list_costume_variants(folder)) == 1
    # Generating a variant does not change what the character is wearing.
    assert get_assembly(load_meta(folder))["body"] == "model/model.glb"


def test_generate_variant_failure_registers_nothing(tmp_path: Path):
    folder = _folder_with_main(tmp_path)
    img = tmp_path / "outfit.png"
    img.write_bytes(b"fake-png")
    with pytest.raises(GenerationError, match="缺模型"):
        generate_costume_variant(img, folder, variant_name="蓝衣", backend=FailingBackend())
    assert list_costume_variants(folder) == []
    assert not (folder / "model" / "variants" / "蓝衣.glb").exists()


def test_switch_variant_points_assembly_body_at_variant(tmp_path: Path):
    folder = _folder_with_main(tmp_path)
    img = tmp_path / "outfit.png"
    img.write_bytes(b"fake-png")
    generate_costume_variant(img, folder, variant_name="红裙", backend=StubBackend())
    report = switch_costume_variant(folder, "红裙")
    assert report["variant"] == "红裙"
    assert report["assembly_body"] == "model/variants/红裙.glb"
    assert get_assembly(load_meta(folder))["body"] == "model/variants/红裙.glb"
    assert active_costume_variant(folder) == "红裙"
    # Switch back to the main model.
    back = switch_costume_variant(folder, "")
    assert back["variant"] == ""
    assert back["assembly_body"] == "model/model.glb"
    assert get_assembly(load_meta(folder))["body"] == "model/model.glb"
    assert active_costume_variant(folder) == ""


def test_switch_unknown_variant_lists_available(tmp_path: Path):
    folder = _folder_with_main(tmp_path)
    img = tmp_path / "outfit.png"
    img.write_bytes(b"fake-png")
    generate_costume_variant(img, folder, variant_name="红裙", backend=StubBackend())
    with pytest.raises(ValueError, match="没有叫「蓝衣」.*红裙"):
        switch_costume_variant(folder, "蓝衣")
    # Failed switch leaves the assembly on the main model.
    assert get_assembly(load_meta(folder))["body"] == "model/model.glb"


def test_main_generation_preserves_variants(tmp_path: Path):
    folder = _folder_with_main(tmp_path)
    img = tmp_path / "outfit.png"
    img.write_bytes(b"fake-png")
    generate_costume_variant(img, folder, variant_name="红裙", backend=StubBackend())
    switch_costume_variant(folder, "红裙")
    # Regenerate the main model: the variant (and the switch) must survive.
    img2 = tmp_path / "cutout2.png"
    img2.write_bytes(b"fake-png")
    generate_model_glb(img2, folder, backend=StubBackend())
    assert len(list_costume_variants(folder)) == 1
    assert active_costume_variant(folder) == "红裙"
    assert get_assembly(load_meta(folder))["body"] == "model/variants/红裙.glb"


def test_service_generate_and_switch_variant(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    svc.ensure_character_folder("温念")
    folder = tmp_path / "outputs" / "characters" / "温念"
    # Main model first (core call with a stub), so switching back has a target.
    main_img = tmp_path / "cutout.png"
    main_img.write_bytes(b"fake-png")
    generate_model_glb(main_img, folder, backend=StubBackend())
    img = tmp_path / "outfit.png"
    img.write_bytes(b"fake-png")
    report = svc.generate_costume_variant("温念", image=img, variant_name="红裙", backend=StubBackend())
    assert report["variant"] == "红裙"
    assert len(svc.list_costume_variants("温念")) == 1
    switched = svc.switch_costume_variant("温念", variant_name="红裙")
    assert switched["assembly_body"] == "model/variants/红裙.glb"
    back = svc.switch_costume_variant("温念", variant_name="")
    assert back["assembly_body"] == "model/model.glb"


def test_service_check_model_generation_reports_missing(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    check = svc.check_model_generation()
    assert check["ok"] is False
    assert check["reason"]
    assert "torch" in check["reason"] or "GPU" in check["reason"] or "TripoSG" in check["reason"]
    assert "git clone" in check["setup_steps"]


def test_list_bodies_includes_variants(tmp_path: Path):
    folder = _folder_with_main(tmp_path)
    img = tmp_path / "outfit.png"
    img.write_bytes(b"fake-png")
    generate_costume_variant(img, folder, variant_name="红裙", backend=StubBackend())
    bodies = list_bodies(tmp_path)
    paths = [item["path"] for item in bodies]
    assert str(folder / "model" / "model.glb") in paths
    assert str(folder / "model" / "variants" / "红裙.glb") in paths
