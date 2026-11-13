"""Tests for the show-through audit (recast/quality/replace_audit.py).

Each metric is pinned against a scene where the answer is known by
construction: a composited frame that changed the background must score
worse than one that only touched the actor, and a deliberate bright line at
the neck cut must score as a seam.
"""

import numpy as np
import pytest

from recast.quality.replace_audit import (
    audit_composite,
    background_damage,
    render_coverage,
    seam_ratio,
)

NECK_Y = 60
BODY = (slice(60, 120), slice(30, 50))


def _textured_scene(seed=0, h=120, w=80):
    """Person column with film-grain-like noise so gradients are meaningful."""
    rng = np.random.default_rng(seed)
    orig = np.full((h, w, 3), 120, dtype=np.uint8)
    noise = rng.integers(-12, 13, size=(h, w, 3))
    body = (orig + noise).clip(0, 255).astype(np.uint8)
    orig[20:, 30:50] = body[20:, 30:50]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:, 30:50] = 255
    return orig, mask


def _alpha_full(h=120, w=80):
    a = np.zeros((h, w), dtype=np.uint8)
    a[*BODY] = 255
    return a


# ----- background damage -----


def test_untouched_background_scores_zero_damage():
    orig, mask = _textured_scene()
    damage = background_damage(orig, orig.copy(), mask)
    assert damage["mean"] == 0.0
    assert damage["frac"] == 0.0


def test_smeared_background_is_counted():
    orig, mask = _textured_scene()
    out = np.full_like(orig, 200)  # every background pixel repainted
    damage = background_damage(orig, out, mask)
    assert damage["frac"] == pytest.approx(1.0)
    assert damage["mean"] > 30


# ----- seam -----


def test_a_bright_line_at_the_neck_scores_as_a_seam():
    orig, mask = _textured_scene()
    out = orig.copy()
    out[NECK_Y - 1 : NECK_Y + 1, 30:50] = 255
    assert seam_ratio(out, mask, NECK_Y) > 2.0


def test_a_clean_join_scores_near_one():
    orig, mask = _textured_scene()
    ratio = seam_ratio(orig, mask, NECK_Y)
    assert 0.5 < ratio < 2.0


def test_seam_is_neutral_without_a_silhouette():
    orig, mask = _textured_scene()
    empty = np.zeros_like(mask)
    assert seam_ratio(orig, empty, NECK_Y) == 0.0


# ----- coverage -----


def test_full_alpha_covers_the_whole_body():
    _, mask = _textured_scene()
    assert render_coverage(_alpha_full(), mask, NECK_Y) == pytest.approx(1.0)


def test_half_width_render_reports_partial_coverage():
    _, mask = _textured_scene()
    alpha = _alpha_full()
    alpha[:, 30:40] = 0  # only the right half of the body is rendered
    assert render_coverage(alpha, mask, NECK_Y) == pytest.approx(0.5, abs=0.02)


# ----- verdict -----


def test_clean_composite_verdicts_ok():
    orig, mask = _textured_scene()
    result = audit_composite(orig, orig.copy(), mask, NECK_Y, render_alpha=_alpha_full())
    assert result["verdict"] == "ok"
    assert result["problems"] == []
    assert result["render_coverage"] == pytest.approx(1.0)


def test_damaged_and_uncovered_frame_needs_review():
    orig, mask = _textured_scene()
    out = orig.copy()
    out[:, :30] = 90
    alpha = _alpha_full()
    alpha[:, 30:42] = 0
    result = audit_composite(orig, out, mask, NECK_Y, render_alpha=alpha)
    assert result["verdict"] == "review"
    assert any("背景" in p for p in result["problems"])
    assert any("覆盖" in p for p in result["problems"])


def test_audit_works_without_alpha():
    orig, mask = _textured_scene()
    result = audit_composite(orig, orig.copy(), mask, NECK_Y)
    assert result["render_coverage"] is None
    assert result["verdict"] == "ok"


def test_an_already_edgy_neck_is_not_blamed_on_the_composite():
    """Real footage scores ~3 at the neck on its own (collar, necklace, jaw).
    The verdict must judge the increase, not the absolute value."""
    orig, mask = _textured_scene()
    busy = orig.copy()
    busy[NECK_Y - 2 : NECK_Y + 2, 30:50] = 250  # a real neck already has edges
    result = audit_composite(busy, busy.copy(), mask, NECK_Y)
    assert result["seam_ratio"] > 2.0  # absolute value is high ...
    assert result["seam_increase"] == pytest.approx(1.0, abs=0.05)  # ... but unchanged
    assert result["verdict"] == "ok"


def test_a_seam_added_by_compositing_is_flagged():
    orig, mask = _textured_scene()
    out = orig.copy()
    out[NECK_Y - 1 : NECK_Y + 1, 30:50] = 255  # the composite drew a line
    result = audit_composite(orig, out, mask, NECK_Y)
    assert result["seam_increase"] > 1.15
    assert result["verdict"] == "review"
    assert any("接缝" in p for p in result["problems"])
