"""Post-replace samples reach the drift check, and a low score is reported."""

from pathlib import Path

from recast.core.service import RecastService
from recast.quality.face_review import post_replace_review_items, select_replace_samples
from recast.quality.review import run_quality_checks


def _payload() -> dict:
    frames = [
        {"frame_idx": 0, "time": 0.0, "bbox": [0, 0, 20, 20]},
        {"frame_idx": 10, "time": 0.4, "bbox": [2, 2, 22, 22]},
        {"frame_idx": 20, "time": 0.8, "bbox": [4, 4, 24, 24]},
    ]
    return {"tracks": [{"track_id": 3, "frames": frames}, {"track_id": 9, "frames": frames[:1]}]}


def _plan(reference: str) -> dict:
    return {
        "targets": [
            {
                "name": "温念",
                "person_name": "温念",
                "face_path": reference,
                "track_ids": [3, 9],
            }
        ]
    }


def test_replace_samples_cover_the_cluster_and_land_on_a_shot(tmp_path: Path):
    ref = tmp_path / "wen.jpg"
    ref.write_bytes(b"face")
    shots = [{"id": "s1", "shot_code": "F1-L1", "start_time": 0.0, "end_time": 1.0}]
    samples = select_replace_samples(_payload(), _plan(str(ref)), shots, per_face=3)
    assert len(samples) == 3
    assert {s["shot_code"] for s in samples} == {"F1-L1"}
    assert {s["track_id"] for s in samples} >= {3}


def test_low_similarity_is_reported_only_when_pairs_are_passed(tmp_path: Path):
    ref = tmp_path / "wen.jpg"
    ref.write_bytes(b"face")
    shots = [{"id": "s1", "shot_code": "F1-L1", "start_time": 0.0, "end_time": 1.0, "dialogue": "短"}]

    def embed(sample):
        if sample["frame_idx"] == 0:
            return [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0], [0.99, 0.01, 0.0]

    items = post_replace_review_items(
        shots,
        _payload(),
        _plan(str(ref)),
        embed,
        thumbnail_dir=tmp_path / "thumbs",
    )
    drifted = [item for item in items if item.reason == "人脸特征漂移"]
    assert len(drifted) == 1
    assert drifted[0].shot_code == "F1-L1"
    assert drifted[0].score is not None and drifted[0].score < 0.35
    assert "温念" in drifted[0].detail
    assert drifted[0].thumbnail and Path(drifted[0].thumbnail).is_file()
    # The old call, shots only, never produced this finding.
    assert run_quality_checks(shots) == [] or all(item.reason != "人脸特征漂移" for item in run_quality_checks(shots))


def test_review_row_keeps_score_and_thumbnail(tmp_path: Path):
    ref = tmp_path / "wen.jpg"
    ref.write_bytes(b"face")
    shots = [{"id": "s1", "shot_code": "F2-L1", "start_time": 0.0, "end_time": 2.0}]
    items = post_replace_review_items(
        shots,
        _payload(),
        _plan(str(ref)),
        lambda sample: ([1.0, 0.0], [0.0, 1.0]),
        thumbnail_dir=tmp_path / "thumbs",
        per_face=1,
    )
    svc = RecastService(tmp_path / "db.db")
    drama = svc.store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "show"), episode_count=1)
    task = svc.store.insert("tasks", drama_id=drama["id"], episode_name="第01集", video_path=str(tmp_path / "a.mp4"))
    item = next(row for row in items if row.reason == "人脸特征漂移")
    mark = svc.store.insert(
        "quality_marks",
        task_id=task["id"],
        shot_id=item.shot_id,
        shot_code=item.shot_code,
        reason=item.reason,
        detail=item.detail,
        status="待人工检查",
        score=item.score,
        thumbnail=item.thumbnail,
    )
    listed = svc.review_items(task_id=task["id"])
    assert listed[0]["reason"] == "人脸特征漂移"
    assert listed[0]["score"] < 0.35
    assert listed[0]["thumbnail"] == item.thumbnail
    svc.set_review_status(mark["id"], "已忽略")
    assert svc.review_items(task_id=task["id"])[0]["status"] == "已忽略"
