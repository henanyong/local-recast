"""Assembly service methods: set/get assembly, wardrobe/bodies, preview."""

import json
import struct
from pathlib import Path

import pytest
from PIL import Image


def _image(path: Path, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def _glb_bytes() -> bytes:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _glb(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_glb_bytes())
    return path


def _service(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _character(svc, name: str = "温念") -> dict:
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=1,
        authorized=1,
        cluster_key=f"cluster:{name}",
    )


def _folder_with_assets(svc, tmp_path: Path, name: str = "温念"):
    ch = _character(svc, name)
    folder = Path(svc.ensure_character_folder(name)["folder"])
    _image(folder / "face" / "a.jpg")
    _image(folder / "face" / "b.jpg")
    _image(folder / "costume" / "dress.png")
    return ch, folder


def test_set_assembly_via_service(tmp_path: Path):
    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    shared_glb = _glb(tmp_path / "outputs" / "models" / "fox.glb")
    report = svc.set_character_assembly("温念", face=folder / "face" / "b.jpg", body=shared_glb)
    assert report["assembly"] == {"face": "face/b.jpg", "body": "models/fox.glb", "costume": ""}
    assert report["binding"]["skipped"] is True  # no costume picked
    got = svc.get_character_assembly("温念")
    assert got["face"] == str(folder / "face" / "b.jpg")
    assert got["body"] == str(shared_glb)


def test_set_assembly_body_must_be_reachable(tmp_path: Path):
    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    outside = _glb(tmp_path / "fox.glb")
    with pytest.raises(ValueError, match="文件夹里面"):
        svc.set_character_assembly("温念", body=outside)


def test_set_assembly_shared_costume_copied_and_bound(tmp_path: Path):
    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    svc.import_model_glb("温念", _glb(tmp_path / "model.glb"))
    shared = _image(tmp_path / "outputs" / "wardrobe" / "red.png")
    report = svc.set_character_assembly("温念", costume=shared)
    assert report["assembly"]["costume"] == "wardrobe/red.png"
    assert report["binding"]["applied"] is True
    # Binding registration points at the copy inside the folder.
    assert report["binding"]["costume"] == "costume/red.png"
    assert (folder / "costume" / "red.png").is_file()
    from recast.assets3d.binding import list_bindings

    assert len(list_bindings(folder)) == 1


def test_set_assembly_costume_without_glb_skips_binding(tmp_path: Path):
    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    report = svc.set_character_assembly("温念", costume=folder / "costume" / "dress.png")
    assert report["assembly"]["costume"] == "costume/dress.png"
    assert report["binding"]["applied"] is False
    assert "GLB" in report["binding"]["reason"]


def test_get_character_assembly_resolves_paths(tmp_path: Path):
    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    svc.import_model_glb("温念", _glb(tmp_path / "model.glb"))
    svc.set_character_assembly("温念", face=folder / "face" / "b.jpg")
    got = svc.get_character_assembly("温念")
    assert got["face"] == str(folder / "face" / "b.jpg")
    assert got["body"] == str(folder / "model" / "model.glb")
    assert got["costume"] == str(folder / "costume" / "dress.png")


def test_service_wardrobe_and_bodies(tmp_path: Path):
    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    svc.import_model_glb("温念", _glb(tmp_path / "model.glb"))
    _image(tmp_path / "outputs" / "wardrobe" / "shared.png")
    assert [item["name"] for item in svc.list_wardrobe()] == ["shared.png", "dress.png"]
    bodies = svc.list_bodies()
    assert [item["name"] for item in bodies] == ["model.glb"]


def test_render_preview_without_body(tmp_path: Path):
    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    report = svc.render_character_preview("温念")
    assert report["rendered"] is False
    assert report["path"] is None


def test_render_preview_with_stub_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import numpy as np

    svc = _service(tmp_path)
    _ch, folder = _folder_with_assets(svc, tmp_path)
    svc.import_model_glb("温念", _glb(tmp_path / "model.glb"))

    def _fake_render(glb_path, dest, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(dest)
        return dest

    monkeypatch.setattr("recast.assets3d.preview.render_model_preview", _fake_render)
    report = svc.render_character_preview("温念")
    assert report["rendered"] is True
    assert report["path"] == str(folder / "model" / "preview.png")
    # Second call reuses the cache instead of rendering again.
    report2 = svc.render_character_preview("温念")
    assert report2["rendered"] is False
    assert report2["path"] == str(folder / "model" / "preview.png")
