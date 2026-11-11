"""Short detector gaps are filled; long gaps and separate tracks are not."""

from recast.stages.replace import index_swap_frames
from recast.stages.track import fill_track_gaps, interpolate_track_frames


def _frame(idx: int, box: list[float]) -> dict:
    return {"frame_idx": idx, "bbox": box, "conf": 0.9, "time": idx / 25.0}


def test_short_gap_is_linearly_filled_and_long_gap_is_not():
    short = interpolate_track_frames(
        [_frame(0, [0, 0, 10, 10]), _frame(3, [30, 0, 40, 10])],
        max_gap=10,
    )
    assert [item["frame_idx"] for item in short] == [0, 1, 2, 3]
    assert short[1]["interpolated"] is True
    assert short[1]["bbox"] == [10, 0, 20, 10]
    assert short[2]["bbox"] == [20, 0, 30, 10]

    exactly = interpolate_track_frames(
        [_frame(0, [0, 0, 4, 4]), _frame(11, [11, 0, 15, 4])],
        max_gap=10,
    )
    assert len(exactly) == 12

    long = interpolate_track_frames(
        [_frame(0, [0, 0, 4, 4]), _frame(12, [12, 0, 16, 4])],
        max_gap=10,
    )
    assert [item["frame_idx"] for item in long] == [0, 12]
    assert all(not item.get("interpolated") for item in long)


def test_gap_limit_is_configurable_and_cluster_tracks_both_keep_filled_frames():
    payload = {
        "tracks": [
            {
                "track_id": 3,
                "cluster_id": 1,
                "frames": [_frame(0, [0, 0, 8, 8]), _frame(4, [8, 0, 16, 8])],
            },
            {
                "track_id": 9,
                "cluster_id": 1,
                "frames": [_frame(0, [20, 0, 28, 8]), _frame(4, [28, 0, 36, 8])],
            },
        ]
    }
    tight = fill_track_gaps(payload, max_gap=2)
    assert [item["frame_idx"] for item in tight["tracks"][0]["frames"]] == [0, 4]

    filled = fill_track_gaps(payload, max_gap=10)
    targets = [
        {"face_path": "wen.jpg", "source_key": "wen.jpg", "track_ids": [3, 9]},
    ]
    frames = index_swap_frames(filled, targets)
    assert 2 in frames
    assert {item["track_id"] for item in frames[2]} == {3, 9}
