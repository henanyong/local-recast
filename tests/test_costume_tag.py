"""Tests for costume auto-tagging (recast/stages/costume.py).

Uses synthetic images and a stub tagger only — no real photos, no model
download. The CLIP-missing error path is exercised for real, with torch
absence simulated via monkeypatch so the test does not depend on whether
the machine happens to have torch installed.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

from recast.core.characters import ensure_character_folder, load_meta
from recast.stages.costume import (
    ClipTagger,
    CostumeError,
    HeuristicTagger,
    crop_costume,
    dominant_colors,
    get_tagger,
    tag_character_folder,
    tag_costume_image,
)


class StubTagger:
    name = "stub"

    def __init__(self, result=None):
        self.result = result or {
            "colors": ["红色"],
            "style": ["古装"],
            "description": "主要颜色：红色；款式：古装。",
        }
        self.seen = []

    def tag(self, image):
        self.seen.append(image.size)
        return dict(self.result)


def _solid(path: Path, color: tuple[int, int, int], size=(100, 100)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_crop_without_bbox_uses_lower_60_percent(tmp_path: Path):
    # Upper 40% red, lower 60% blue: the crop must be (mostly) blue.
    img = Image.new("RGB", (100, 100))
    top = Image.new("RGB", (100, 40), (255, 0, 0))
    bottom = Image.new("RGB", (100, 60), (0, 0, 255))
    img.paste(top, (0, 0))
    img.paste(bottom, (0, 40))
    crop = crop_costume(img)
    assert crop.size == (100, 60)
    colors = dominant_colors(crop)
    assert colors[0]["color"] == "蓝色"


def test_crop_with_face_bbox_starts_below_face(tmp_path: Path):
    img = Image.new("RGB", (200, 200), (0, 0, 255))
    crop = crop_costume(img, face_bbox=(80, 20, 120, 60))
    # Below the chin (y=60 + 10% face height = 64), widened by half a face
    # width on each side (x 60..140).
    assert crop.size == (80, 136)


def test_dominant_colors_solid_red_and_black(tmp_path: Path):
    red = Image.new("RGB", (50, 50), (220, 30, 30))
    assert dominant_colors(red)[0]["color"] == "红色"
    black = Image.new("RGB", (50, 50), (10, 10, 10))
    assert dominant_colors(black)[0]["color"] == "黑色"
    white = Image.new("RGB", (50, 50), (250, 250, 250))
    assert dominant_colors(white)[0]["color"] == "白色"


def test_heuristic_tagger_never_invents_style(tmp_path: Path):
    tagger = HeuristicTagger()
    result = tagger.tag(Image.new("RGB", (50, 50), (220, 30, 30)))
    assert result["colors"] == ["红色"]
    assert result["style"] == []
    assert "红色" in result["description"]


def test_get_tagger_names(tmp_path: Path):
    assert get_tagger("heuristic").name == "heuristic"
    assert get_tagger("clip").name == "clip"
    assert get_tagger("CLIP").name == "clip"
    with pytest.raises(CostumeError, match="未知"):
        get_tagger("nope")


def test_clip_tagger_missing_torch_errors_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Simulate "no torch" regardless of the machine: on a machine where torch
    # IS installed the error would be about openai-clip instead, so the
    # torch-missing path must not depend on the local environment. The real
    # error path must say what to install instead of failing silently or
    # inventing tags.
    monkeypatch.setitem(sys.modules, "torch", None)
    tagger = ClipTagger()
    with pytest.raises(CostumeError, match="torch"):
        tagger.tag(Image.new("RGB", (50, 50), (200, 30, 30)))


def test_tag_costume_image_with_stub_tagger(tmp_path: Path):
    photo = _solid(tmp_path / "photo.jpg", (200, 30, 30))
    stub = StubTagger()
    result = tag_costume_image(photo, tagger=stub)
    assert result["colors"] == ["红色"]
    assert result["style"] == ["古装"]
    assert result["tagger"] == "stub"
    assert stub.seen, "tagger must see the cropped region, not the whole photo"
    assert stub.seen[0] == (100, 60)


def test_tag_costume_image_missing_file(tmp_path: Path):
    with pytest.raises(CostumeError, match="找不到"):
        tag_costume_image(tmp_path / "gone.jpg", tagger=StubTagger())


def test_tag_character_folder_writes_screenshot_and_meta(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    photo = _solid(tmp_path / "photo1.jpg", (200, 30, 30))
    report = tag_character_folder(folder, sources=[photo], tagger=StubTagger())
    assert len(report["tagged"]) == 1
    assert report["description"]
    shot = Path(report["tagged"][0]["screenshot"])
    assert shot.is_file() and shot.parent.name == "costume"
    meta = load_meta(folder)
    assert meta is not None
    assert meta["costume"]["source"] == "auto"
    assert "红色" in meta["costume"]["description"]
    assert meta["costume"]["files"] == [str(shot)]


def test_tag_character_folder_records_failures_without_inventing(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    good = _solid(tmp_path / "good.jpg", (30, 200, 30))
    report = tag_character_folder(
        folder, sources=[good, tmp_path / "missing.jpg"], tagger=StubTagger()
    )
    assert len(report["tagged"]) == 1
    assert len(report["failed"]) == 1
    assert "missing.jpg" in report["failed"][0]["photo"]
    meta = load_meta(folder)
    assert meta is not None and meta["costume"]["description"]


def test_tag_character_folder_needs_full_photos(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    with pytest.raises(CostumeError, match="全身照片"):
        tag_character_folder(folder, sources=[], tagger=StubTagger())
