"""Tests for the composite-quality fixes and the show-through audit.

Two behaviours are pinned here, both of them bugs that produced visible
show-through rather than crashes:

1. ``composite_frame`` must not erase pixels the render already covers.
   The old code inpainted the actor's whole body and smeared the
   background behind it.
2. ``fit_render_to_person`` must close the size loop from the silhouette,
   because the 3D scale cannot be trusted to better than 20-30%.
"""

import numpy as np
import pytest

from recast.stages.rerender import (
    _alpha_bbox,
    composite_frame,
    fit_render_to_person,
)


def _scene(h=120, w=80, body_top=40, x0=35, x1=45):
    """Flat grey background, a 'person' column, and a head block above."""
    orig = np.full((h, w, 3), 120, dtype=np.uint8)
    orig[body_top - 20 :, x0:x1] = (200, 30, 30)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[body_top - 20 :, x0:x1] = 255
    return orig, mask


def _render(x0, x1, top, bottom, h=120, w=80, colour=(30, 30, 200)):
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[top:bottom, x0:x1, :3] = colour
    rgba[top:bottom, x0:x1, 3] = 255
    return rgba


# ----- 1. erase only the uncovered part -----


def test_render_covering_the_body_leaves_background_untouched():
    """The headline fix: nothing is inpainted, so nothing behind is smeared."""
    orig, mask = _scene()
    render = _render(x0=30, x1=50, top=40, bottom=120)  # wider than the actor
    out = composite_frame(orig, render, mask, 40, color_match=False)

    assert (out[:40] == orig[:40]).all()  # head untouched
    # The render's soft edge bleeds 1px outward by design; past that, the
    # background must be bit-identical to the original.
    assert (out[:, :28] == orig[:, :28]).all()  # background left of actor
    assert (out[:, 52:] == orig[:, 52:]).all()  # background right of actor


def test_narrow_render_still_erases_the_leftover_fringe():
    """A slimmer CG body must not leave the actor's own outline beside it."""
    orig, mask = _scene()
    render = _render(x0=38, x1=42, top=40, bottom=120)
    out = composite_frame(orig, render, mask, 40, color_match=False)

    # The actor's body is cols 35-45, the render only 38-42: the two side
    # strips must be repainted, not left as the original actor.
    assert out[80, 36].tolist() != orig[80, 36].tolist()


def test_empty_render_still_inpaints_the_body_away():
    """No render at all is a degenerate case, but it must not leave the
    actor's own body sitting in the frame."""
    orig, mask = _scene()
    render = _render(x0=35, x1=45, top=40, bottom=120)
    render[:, :, 3] = 0
    out = composite_frame(orig, render, mask, 40, color_match=False)
    assert np.abs(out[80, 40].astype(int) - 120).max() <= 12


# ----- 2. fit the render to the measured silhouette -----


def test_fit_refits_render_span_to_neck_and_feet():
    orig, mask = _scene()  # actor body spans rows 40..119
    render = _render(x0=36, x1=44, top=50, bottom=106)  # 56 tall, too small
    fitted = fit_render_to_person(render, mask, 40)

    x0, y0, x1, y1 = _alpha_bbox(fitted[:, :, 3])
    assert y0 == pytest.approx(40, abs=2)  # top lands on the neck cut
    assert y1 == pytest.approx(120, abs=2)  # bottom lands on the actor's feet


def test_fit_skips_a_correction_too_small_to_matter():
    orig, mask = _scene()
    render = _render(x0=36, x1=44, top=41, bottom=120)  # ~2.5% too short
    fitted = fit_render_to_person(render, mask, 40)
    assert np.array_equal(fitted, render)


def test_fit_refuses_an_absurd_scale_instead_of_hiding_it():
    orig, mask = _scene()
    render = _render(x0=38, x1=42, top=100, bottom=120)  # 20 tall vs 80 wanted
    fitted = fit_render_to_person(render, mask, 40)
    assert np.array_equal(fitted, render)  # 4x off: refuse, do not fake it


def test_fit_is_a_noop_without_an_actor_silhouette():
    orig = np.full((120, 80, 3), 120, dtype=np.uint8)
    empty = np.zeros((120, 80), dtype=np.uint8)
    render = _render(x0=36, x1=44, top=50, bottom=106)
    assert np.array_equal(fit_render_to_person(render, empty, 40), render)
