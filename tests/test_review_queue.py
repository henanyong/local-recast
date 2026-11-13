"""Tests for the manual review queue (recast/core/review.py).

Synthetic images only: a sharp checkerboard scores high, a solid colour
scores low, a tiny crop scores low. No real video, no detection model.
"""

from pathlib import Path

import pytest
from PIL import Image

from recast.core.review import (
    DEFAULT_REVIEW_MIN_SCORE,
    build_review_queue,
    score_sample_file,
)


def _checker(path: Path, size: int = 140) -> Path:
    """High-frequency image: sharp by any measure."""
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if (x // 7 + y // 7) % 2 else 0
    img = img.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _solid(path: Path, size: int = 140, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)
    return path


def test_score_sharp_big_is_high(tmp_path: Path):
    assert score_sample_file(_checker(tmp_path / "big.png", 140)) >= 0.9


def test_score_solid_is_low(tmp_path: Path):
    # Solid colour: no texture at all, Laplacian variance is 0.
    assert score_sample_file(_solid(tmp_path / "solid.png", 140)) < 0.5


def test_score_tiny_is_low_even_when_sharp(tmp_path: Path):
    # 40x40: size_term = sqrt(1600)/120 = 0.33, so even perfectly sharp it
    # cannot reach the 0.5 bar.
    assert score_sample_file(_checker(tmp_path / "tiny.png", 40)) < 0.5


def test_score_unreadable_is_zero(tmp_path: Path):
    assert score_sample_file(tmp_path / "missing.png") == 0.0
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image at all")
    assert score_sample_file(bad) == 0.0


class _Store:
    """Minimal stand-in for the real Store: only list_characters."""

    def __init__(self, characters):
        self._characters = characters

    def list_characters(self):
        return self._characters


def test_queue_lists_unconfirmed_clusters(tmp_path: Path):
    out = tmp_path / "out"
    ch = {
        "id": "c1",
        "name": "人物1",
        "display_name": "人物1",
        "confirmed": 0,
        "assets": [{"kind": "face", "path": str(_checker(tmp_path / "f.jpg"))}],
    }
    items = build_review_queue(_Store([ch]), out)
    assert [i.kind for i in items] == ["cluster"]
    assert items[0].key == "c1"
    assert items[0].thumb is not None


def test_queue_skips_confirmed_characters(tmp_path: Path):
    out = tmp_path / "out"
    ch = {"id": "c1", "name": "温念", "display_name": "温念", "confirmed": 1, "assets": []}
    assert build_review_queue(_Store([ch]), out) == []


def test_queue_flags_low_score_samples(tmp_path: Path):
    out = tmp_path / "out"
    bad = _solid(tmp_path / "bad.png", 140)
    good = _checker(tmp_path / "good.png", 140)
    ch = {
        "id": "c1",
        "name": "温念",
        "display_name": "温念",
        "confirmed": 1,
        "assets": [
            {"kind": "face", "path": str(bad)},
            {"kind": "face", "path": str(good)},
        ],
    }
    items = build_review_queue(_Store([ch]), out)
    assert len(items) == 1
    assert items[0].kind == "sample"
    assert items[0].score is not None and items[0].score < DEFAULT_REVIEW_MIN_SCORE
    assert items[0].thumb == str(bad)


def test_queue_empty_means_step3_done(tmp_path: Path):
    out = tmp_path / "out"
    good = _checker(tmp_path / "good.png", 140)
    ch = {
        "id": "c1",
        "name": "温念",
        "display_name": "温念",
        "confirmed": 1,
        "assets": [{"kind": "face", "path": str(good)}],
    }
    assert build_review_queue(_Store([ch]), out) == []
