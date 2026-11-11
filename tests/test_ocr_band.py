"""OCR band, confidence filter, fuzzy merge, and decoded-frame timing."""

from __future__ import annotations

import numpy as np

from recast.extract.ocr import (
    filter_ocr_parts,
    merge_ocr_hits,
    plan_decoded_frames,
    subtitle_band,
    windows_for_hits,
)


def test_subtitle_band_keeps_only_the_bottom():
    frame = np.zeros((100, 16, 3), dtype=np.uint8)
    frame[:72] = 1
    frame[72:] = 9
    band = subtitle_band(frame, top_ratio=0.72)
    assert band.shape[0] == 28
    assert int(band[0, 0, 0]) == 9
    assert int(band[-1, 0, 0]) == 9


def test_low_confidence_parts_are_dropped():
    parts = [("店招", 0.2), ("快起来", 0.91), ("", 0.99)]
    assert filter_ocr_parts(parts, min_conf=0.5) == [("快起来", 0.91)]
    assert filter_ocr_parts(parts, min_conf=0.1)[0] == ("店招", 0.2)


def test_fuzzy_merge_does_not_require_exact_text():
    hits = [
        {"start": 0.0, "end": 0.5, "text": "快起来别装死", "conf": 0.6},
        {"start": 0.4, "end": 0.9, "text": "快起来，别装死！", "conf": 0.92},
        {"start": 2.0, "end": 2.4, "text": "第一集", "conf": 0.95},
    ]
    merged = merge_ocr_hits(hits, similar=0.72, gap=0.8)
    assert len(merged) == 2
    assert merged[0]["text"] == "快起来，别装死！"
    assert merged[0]["end"] == 0.9
    assert merged[1]["text"] == "第一集"


def test_windows_use_decoded_time_not_a_fixed_pad():
    # The sampler asked for 0.4s, but the frame that was actually decoded is at 1.0s.
    hits = [
        {"time": 1.0, "text": "甲", "conf": 0.9},
        {"time": 1.4, "text": "甲", "conf": 0.8},
    ]
    timed = windows_for_hits(hits, max_half=0.45)
    assert timed[0]["start"] == 0.55
    assert abs(timed[0]["end"] - 1.2) < 1e-9
    assert abs(timed[1]["start"] - 1.2) < 1e-9
    assert abs(timed[1]["end"] - 1.85) < 1e-9
    assert 0.4 - 0.6 not in {row["start"] for row in timed}


def test_decoded_frames_walk_forward_only():
    chosen = plan_decoded_frames([0.05, 0.15, 5.0], fps=10.0, frame_count=20)
    indexes = [item[0] for item in chosen]
    assert indexes == sorted(indexes)
    assert chosen[0] == (1, 0.1, 0.05)
    assert chosen[1] == (2, 0.2, 0.15)
    assert all(actual + 1e-9 >= requested for _idx, actual, requested in chosen)
    assert len(chosen) == 2
