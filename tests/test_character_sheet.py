"""Tests for the character sheet (3D turntable + measurable appearance).

The sheet is what the 2D route draws consistency from, so the tests are
about provenance rather than beauty: an angle is either rendered or it is
absent with a reason, and an appearance field is either measured or empty
with a reason. Nothing is allowed to be filled in "plausibly".
"""

import json

import numpy as np
import pytest

from recast.stages.character_sheet import (
    SheetError,
    _upright_rotation,
    appearance_from_photos,
    build_character_sheet,
    render_turntable,
    skin_tone,
)

trimesh = pytest.importorskip("trimesh")


def _write(path, bgr):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", bgr)
    assert ok
    buf.tofile(str(path))


def _gl_available() -> bool:
    try:
        import pyrender

        renderer = pyrender.OffscreenRenderer(8, 8)
        renderer.delete()
        return True
    except Exception:
        return False


needs_gl = pytest.mark.skipif(not _gl_available(), reason="需要离屏 GL 上下文")


def _tall_box_glb(tmp_path):
    """A Z-up tall box: exactly the convention that used to render lying down."""
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 1.4])
    path = tmp_path / "tall.glb"
    mesh.export(path)
    return path


def test_upright_rotation_puts_a_z_up_model_on_its_feet():
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 1.4])
    rotated = mesh.copy()
    rotated.apply_transform(_upright_rotation([mesh]))
    assert rotated.extents[1] == pytest.approx(1.4, abs=1e-5)


def test_upright_rotation_leaves_a_y_up_model_alone():
    mesh = trimesh.creation.box(extents=[0.2, 1.4, 0.2])
    rotated = mesh.copy()
    rotated.apply_transform(_upright_rotation([mesh]))
    assert rotated.extents[1] == pytest.approx(1.4, abs=1e-5)


def test_turntable_reports_a_reason_for_a_missing_file(tmp_path):
    result = render_turntable(tmp_path / "nope.glb")
    assert result["images"] == []
    assert "找不到模型文件" in result["reason"]


@needs_gl
def test_turntable_frames_are_upright_and_fully_visible(tmp_path):
    result = render_turntable(_tall_box_glb(tmp_path), size=128, angles=(0, 90))
    assert result["reason"] is None
    assert [a for a, _ in result["images"]] == [0, 90]

    for _, image in result["images"]:
        assert image.shape == (128, 128, 3)
        foreground = np.abs(image.astype(int) - 107).max(axis=2) > 12
        rows = np.where(foreground.any(axis=1))[0]
        cols = np.where(foreground.any(axis=0))[0]
        assert rows.size > 0 and cols.size > 0
        # A stood-up box reads taller than wide; a lying-down one is wider.
        assert rows.size > cols.size * 1.5
        # And it fits: a few pixels of margin at top and bottom.
        assert rows.min() > 2 and rows.max() < 125


def test_skin_tone_names_a_measured_cheek_patch():
    import cv2

    bgr = np.full((100, 100, 3), (150, 190, 220), dtype=np.uint8)  # BGR light warm
    info = skin_tone(bgr, [20.0, 20.0, 80.0, 90.0])
    assert info["lab"] is not None
    assert info["name"] in {"白皙", "自然", "小麦", "深色"}
    assert info["source"] == "photo"
    # The measurement is in Lab, so it must match what cv2 reports.
    patch = cv2.cvtColor(bgr[58:80, 33:62], cv2.COLOR_BGR2LAB).reshape(-1, 3)
    assert info["lab"][0] == pytest.approx(float(np.median(patch, axis=0)[0]), abs=1.0)


def test_skin_tone_refuses_a_tiny_box():
    bgr = np.zeros((10, 10, 3), np.uint8)
    info = skin_tone(bgr, [0.0, 0.0, 2.0, 2.0])
    assert info["lab"] is None and "脸框太小" in info["reason"]


def test_appearance_measures_palette_and_refuses_to_guess_hair(tmp_path):
    photo = tmp_path / "face" / "a.png"
    _write(photo, np.full((64, 64, 3), (30, 30, 200), dtype=np.uint8))  # red
    out = appearance_from_photos([photo])
    assert out["palette"], "调色板必须从照片里量出来"
    assert out["palette"][0]["color"] == "红色"
    assert out["hair"]["name"] == ""
    assert out["hair"]["source"] == "unmeasured"
    assert out["build"]["name"] == ""


def test_build_sheet_from_photos_only(tmp_path):
    folder = tmp_path / "温念"
    _write(folder / "face" / "01.png", np.full((64, 64, 3), (30, 30, 200), dtype=np.uint8))

    report = build_character_sheet(folder, size=64)

    assert report["character"] == "温念"
    assert report["model_rendered"] is False
    assert "没有 GLB" in report["render_reason"]
    assert report["turns"] == []

    sheet_json = folder / "sheet" / "sheet.json"
    assert sheet_json.is_file()
    saved = json.loads(sheet_json.read_text(encoding="utf-8"))
    assert saved["character"] == "温念"
    assert saved["appearance"]["palette"]

    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["sheet"]["model_rendered"] is False
    assert meta["appearance"]["palette"]


@needs_gl
def test_build_sheet_renders_the_model_when_present(tmp_path):
    folder = tmp_path / "Person_1"
    (folder / "model").mkdir(parents=True)
    _tall_box_glb(folder / "model")
    _write(folder / "face" / "01.png", np.full((48, 48, 3), (60, 120, 160), dtype=np.uint8))

    report = build_character_sheet(folder, size=96)

    assert report["model_rendered"] is True
    assert len(report["turns"]) == 4
    for turn in report["turns"]:
        assert (folder / turn["path"]).is_file()
        assert turn["source"] == "3d"
    assert report["render_reason"] is None


def test_build_sheet_needs_something_to_work_with(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(SheetError, match="设定板"):
        build_character_sheet(folder)


# ----- palette must describe the person, not the studio background -----


def _half_green_photo(path):
    """Left half saturated green, right half neutral grey (a studio plate)."""
    bgr = np.full((96, 96, 3), 200, dtype=np.uint8)
    bgr[:, :48] = (40, 160, 60)  # BGR green
    _write(path, bgr)
    return path


def test_palette_counts_only_the_person_when_a_mask_is_available(tmp_path, monkeypatch):
    import recast.stages.character_sheet as sheet

    photo = _half_green_photo(tmp_path / "face" / "01.png")
    monkeypatch.setattr(
        sheet, "_person_mask", lambda bgr: (np.indices(bgr.shape[:2])[1] < 48)
    )
    out = appearance_from_photos([photo])
    assert out["palette_source"] == "person"
    assert out["palette"][0]["color"] == "绿色"
    assert out["palette"][0]["share"] > 0.9


def test_palette_says_so_when_it_had_to_use_the_whole_photo(tmp_path, monkeypatch):
    """Without a mask the number is still reported, but labelled — a palette
    that is 80% background is worse than no palette, so the caller must know."""
    import recast.stages.character_sheet as sheet

    photo = _half_green_photo(tmp_path / "face" / "01.png")
    monkeypatch.setattr(sheet, "_person_mask", lambda bgr: None)
    out = appearance_from_photos([photo])
    assert out["palette_source"] == "full_photo"
    assert {p["color"] for p in out["palette"]} == {"绿色", "白色"}


# ----- palette must describe the person, not the studio background -----


def _half_green_photo(path):
    """Left half saturated green, right half neutral grey (a studio plate)."""
    bgr = np.full((96, 96, 3), 200, dtype=np.uint8)
    bgr[:, :48] = (40, 160, 60)  # BGR green
    _write(path, bgr)
    return path


def test_palette_counts_only_the_person_when_a_mask_is_available(tmp_path, monkeypatch):
    import recast.stages.character_sheet as sheet

    photo = _half_green_photo(tmp_path / "face" / "01.png")
    monkeypatch.setattr(
        sheet, "_person_mask", lambda bgr: (np.indices(bgr.shape[:2])[1] < 48)
    )
    out = appearance_from_photos([photo])
    assert out["palette_source"] == "person"
    assert out["palette"][0]["color"] == "绿色"
    assert out["palette"][0]["share"] > 0.9


def test_palette_says_so_when_it_had_to_use_the_whole_photo(tmp_path, monkeypatch):
    """Without a mask the number is still reported, but labelled — a palette
    that is 80% background is worse than no palette, so the caller must know."""
    import recast.stages.character_sheet as sheet

    photo = _half_green_photo(tmp_path / "face" / "01.png")
    monkeypatch.setattr(sheet, "_person_mask", lambda bgr: None)
    out = appearance_from_photos([photo])
    assert out["palette_source"] == "full_photo"
    assert {p["color"] for p in out["palette"]} == {"绿色", "白色"}
