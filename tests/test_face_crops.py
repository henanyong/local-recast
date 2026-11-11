"""Reference crops are the best diverse stills, not the first eight in time."""

from recast.extract.faces import score_reference_sample, select_reference_crops


def _sample(frame, det, size, frontal):
    side = size ** 0.5
    return {
        "frame_idx": frame,
        "bbox": [0, 0, side, side],
        "det_score": det,
        "frontal": frontal,
    }


def test_top_crops_prefer_clear_frontal_faces_and_spread_out():
    nearby = [_sample(frame, 0.99, 100 * 100, 1.0) for frame in range(12)]
    later = _sample(40, 0.9, 90 * 90, 1.0)
    tiny = _sample(80, 0.4, 12 * 12, 0.2)
    profile = _sample(60, 0.99, 100 * 100, 0.05)
    chosen = select_reference_crops([*nearby, later, tiny, profile], limit=8, min_frame_gap=8)
    frames = [int(s["frame_idx"]) for s in chosen]
    assert 40 in frames
    assert 80 not in frames
    assert 60 not in frames
    assert max(frames) - min(frames) > 8
    assert len(chosen) == 8
    assert score_reference_sample(nearby[0]) > score_reference_sample(profile)
    assert score_reference_sample(nearby[0]) > score_reference_sample(tiny)
