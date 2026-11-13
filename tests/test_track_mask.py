"""Tests for the temporal masker.

The point of the track is that the mask stops flickering. These tests pin the
three mechanisms that make that true — detection scheduling, motion
compensation and smoothing — using a stub segmenter so the suite needs no
model and no GPU.
"""

import numpy as np
import pytest

from recast.stages.person_mask import PersonMaskError
from recast.stages.track_mask import TemporalMasker


def _frame_with_square(offset: int, *, size=(120, 160), side=30):
    frame = np.full((size[0], size[1], 3), 128, dtype=np.uint8)
    frame[40:40 + side, offset:offset + side] = 255
    return frame


def _square_mask(offset: int, *, size=(120, 160), side=30):
    mask = np.zeros(size, dtype=np.uint8)
    mask[40:40 + side, offset:offset + side] = 255
    return mask


def _stub(monkeypatch, *, mask, fail_after=None):
    """Stub segment_person_auto: returns ``mask``, then optionally fails."""
    import recast.stages.track_mask as module

    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise PersonMaskError("stub: 丢了")
        return {
            "mask": mask.copy(),
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "method": "stub",
            "person_count": 1,
            "warning": "",
        }

    monkeypatch.setattr(module, "segment_person_auto", fake)
    return calls


def test_detection_runs_on_schedule_not_every_frame(monkeypatch):
    calls = _stub(monkeypatch, mask=_square_mask(20))
    masker = TemporalMasker(redetect_every=5, smooth=0.0)
    for i in range(12):
        masker(_frame_with_square(20), i)
    assert calls["n"] == 3  # frames 0, 5, 10
    assert masker.report()["detections"] == 3
    assert masker.report()["reuses"] == 9


def test_a_single_frame_blip_does_not_flip_the_mask():
    """Smoothing is what kills boundary shimmer: one frame of noise must not
    move the boundary, or the recolour flashes along the collar."""
    prev = np.zeros((10, 10), dtype=bool)
    new = np.ones((10, 10), dtype=bool)
    assert TemporalMasker.smooth_mask(prev, new, 0.6).sum() == 0  # 0.4 < 0.5
    assert TemporalMasker.smooth_mask(prev, new, 0.0).sum() == 100  # no smoothing


def test_sustained_change_comes_through(monkeypatch):
    masker = TemporalMasker(redetect_every=1, smooth=0.5)
    _stub(monkeypatch, mask=_square_mask(20))
    first = masker(_frame_with_square(20), 0)
    assert (first > 0).sum() > 0
    # After two agreeing frames the EMA crosses 0.5 even with strong smoothing.
    masker(_frame_with_square(20), 1)
    assert (masker(_frame_with_square(20), 2) > 0).sum() > 0


def test_motion_compensation_keeps_the_mask_on_a_moving_subject(monkeypatch):
    """Between detections the mask is translated by phase correlation, so a
    walking actor stays covered instead of being left behind."""
    _stub(monkeypatch, mask=_square_mask(20), fail_after=1)
    masker = TemporalMasker(redetect_every=100, smooth=0.0)

    centroids = []
    for i in range(4):
        mask = masker(_frame_with_square(20 + 4 * i), i)
        xs = np.where(mask > 0)[1]
        centroids.append(float(xs.mean()) if xs.size else -1.0)

    assert all(c > 0 for c in centroids), centroids
    assert centroids[1] > centroids[0] + 1.0, centroids  # followed the square
    assert centroids[-1] > centroids[0] + 5.0, centroids


def test_a_lost_detection_reuses_the_track(monkeypatch):
    _stub(monkeypatch, mask=_square_mask(20), fail_after=1)
    masker = TemporalMasker(redetect_every=1, smooth=0.0)
    first = masker(_frame_with_square(20), 0)
    second = masker(_frame_with_square(20), 1)  # stub raises; track must survive
    assert (first > 0).sum() == (second > 0).sum()
    assert masker.report()["reuses"] == 1


def test_losing_the_person_on_the_first_frame_raises(monkeypatch):
    _stub(monkeypatch, mask=_square_mask(20), fail_after=0)
    masker = TemporalMasker()
    with pytest.raises(PersonMaskError, match="丢了"):
        masker(_frame_with_square(20), 0)


def test_parameters_are_validated():
    with pytest.raises(ValueError, match="redetect_every"):
        TemporalMasker(redetect_every=0)
    with pytest.raises(ValueError, match="smooth"):
        TemporalMasker(smooth=1.0)


def test_report_describes_the_run(monkeypatch):
    _stub(monkeypatch, mask=_square_mask(20))
    masker = TemporalMasker(redetect_every=10, smooth=0.3)
    for i in range(3):
        masker(_frame_with_square(20), i)
    report = masker.report()
    assert report["backend"] == "stub"
    assert report["frames"] == 3
    assert report["redetect_every"] == 10
    assert report["smooth"] == 0.3


def test_adaptive_redetects_early_when_the_subject_keeps_moving(monkeypatch):
    """A fixed 6-frame schedule is fine for a talking head and wrong for a
    walk-out: the drift budget must trigger a re-detect before the mask is
    left behind."""
    _stub(monkeypatch, mask=_square_mask(20))
    masker = TemporalMasker(
        redetect_every=100,  # schedule alone would never fire in this window
        smooth=0.0,
        adaptive=True,
        max_drift_ratio=0.02,  # 2% of 160 px = 3.2 px of drift
        fast_motion_ratio=1.0,  # ignore the jolt path for this test
    )
    for i in range(6):
        masker(_frame_with_square(20 + 4 * i), i)
    assert masker.detections > 1
    assert masker.early_detections >= 1


def test_adaptive_off_keeps_the_plain_schedule(monkeypatch):
    _stub(monkeypatch, mask=_square_mask(20))
    masker = TemporalMasker(
        redetect_every=100, smooth=0.0, adaptive=False, max_drift_ratio=0.01
    )
    for i in range(6):
        masker(_frame_with_square(20 + 4 * i), i)
    assert masker.detections == 1
    assert masker.early_detections == 0


def test_a_jolt_triggers_a_redetect(monkeypatch):
    """A cut or a fast pan produces one large shift; the mask must be re-seeded
    immediately instead of being dragged along the old track."""
    _stub(monkeypatch, mask=_square_mask(20))
    masker = TemporalMasker(
        redetect_every=100, smooth=0.0, adaptive=True, max_drift_ratio=1.0, fast_motion_ratio=0.05
    )
    masker(_frame_with_square(20), 0)
    masker(_frame_with_square(60), 1)  # 40 px jump on a 160 px frame
    assert masker.early_detections == 1
