"""One-pass multi-person replace: cluster targets, not a second encode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recast.config import RecastConfig
from recast.core.constants import AUTH_LICENSED, AUTH_OWN_SHOOT
from recast.core.runner import resolve_swap_references, run_episode
from recast.core.service import RecastService
from recast.stages.replace import index_swap_frames, plan_replace_targets, select_target_tracks
from recast.store.db import Store


def _identity() -> dict:
    return {
        "cluster_threshold": 0.45,
        "persons": [
            {
                "name": "温念",
                "auto_name": "Person_1",
                "cluster_id": 1,
                "track_ids": [3, 9],
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "name": "陈秋微",
                "auto_name": "Person_2",
                "cluster_id": 2,
                "track_ids": [4],
                "embedding": [0.0, 1.0, 0.0],
            },
        ],
    }


def _tracks() -> dict:
    return {
        "tracks": [
            {
                "track_id": 3,
                "cluster_id": 1,
                "frames": [{"frame_idx": 0, "bbox": [0, 0, 10, 10]}],
            },
            {
                "track_id": 9,
                "cluster_id": 1,
                "frames": [{"frame_idx": 0, "bbox": [12, 0, 20, 10]}, {"frame_idx": 4, "bbox": [12, 0, 20, 10]}],
            },
            {
                "track_id": 4,
                "cluster_id": 2,
                "frames": [{"frame_idx": 0, "bbox": [30, 0, 40, 10]}] + [
                    {"frame_idx": i, "bbox": [30, 0, 40, 10]} for i in range(1, 40)
                ],
            },
        ]
    }


def _refs() -> list[dict]:
    return [
        {"character_id": "wen", "name": "温念", "face_path": "wen.jpg", "embedding": [1.0, 0.0, 0.0]},
        {"character_id": "chen", "name": "陈秋微", "face_path": "chen.jpg", "embedding": [0.0, 1.0, 0.0]},
    ]


def test_two_people_share_one_replace_plan_and_frame_map():
    payload = _tracks()
    plan = plan_replace_targets(_refs(), _identity(), payload, min_similarity=0.55)
    by_id = {row["character_id"]: row for row in plan["targets"]}
    assert set(by_id) == {"wen", "chen"}
    assert by_id["wen"]["track_ids"] == [3, 9]
    assert by_id["chen"]["track_ids"] == [4]
    assert plan["skipped"] == []

    frames = index_swap_frames(payload, plan["targets"])
    keys = {item["source_key"] for item in frames[0]}
    assert keys == {"wen.jpg", "chen.jpg"}
    assert {item["track_id"] for item in frames[0]} == {3, 9, 4}


def test_cluster_selects_both_tracks_not_the_longest():
    payload = _tracks()
    identity = _identity()
    assert select_target_tracks(payload, None, cluster_id=1, identity=identity) == [3, 9]
    # Same result when only the track records share a cluster_id.
    assert select_target_tracks(payload, None, cluster_id=1) == [3, 9]
    # Explicit track-id narrows that one person and leaves the other cluster intact.
    assert select_target_tracks(payload, 9, cluster_id=1, identity=identity) == [9]
    plan = plan_replace_targets(_refs(), identity, payload, track_id=9, min_similarity=0.55)
    by_id = {row["character_id"]: row for row in plan["targets"]}
    assert by_id["wen"]["track_ids"] == [9]
    assert by_id["chen"]["track_ids"] == [4]
    assert "温念" in (plan.get("track_note") or "")


def test_unmatched_reference_is_reported_and_not_assigned():
    refs = _refs() + [
        {"character_id": "stranger", "name": "路人", "face_path": "stranger.jpg", "embedding": [0.0, 0.0, 1.0]},
    ]
    plan = plan_replace_targets(refs, _identity(), _tracks(), min_similarity=0.55)
    assert {row["character_id"] for row in plan["targets"]} == {"wen", "chen"}
    skipped = plan["skipped"]
    assert len(skipped) == 1
    assert skipped[0]["character_id"] == "stranger"
    assert "已跳过" in skipped[0]["reason"]
    assert "0.000" in skipped[0]["reason"] or "低于" in skipped[0]["reason"]

    ambiguous = plan_replace_targets(
        [{"character_id": "amb", "name": "难分", "face_path": "amb.jpg", "embedding": [1.0, 0.0]}],
        {
            "persons": [
                {"name": "甲", "cluster_id": 1, "track_ids": [1], "embedding": [1.0, 0.0]},
                {"name": "乙", "cluster_id": 2, "track_ids": [2], "embedding": [0.99, 0.1410673597]},
            ]
        },
        {"tracks": [{"track_id": 1, "frames": []}, {"track_id": 2, "frames": []}]},
        min_similarity=0.5,
        margin=0.08,
    )
    assert ambiguous["targets"] == []
    assert "无法确认" in ambiguous["skipped"][0]["reason"]

    forced = plan_replace_targets(
        [{"character_id": "stranger", "name": "路人", "face_path": "s.jpg", "embedding": [0.0, 0.0, 1.0]}],
        _identity(),
        _tracks(),
        track_id=3,
        min_similarity=0.55,
    )
    assert forced["targets"] == []
    assert "没有替换这条轨迹" in (forced.get("track_note") or "")


def test_one_run_maps_both_people_without_a_second_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video-bytes")
    work = tmp_path / "run"
    work.mkdir()
    (work / "meta.json").write_text(
        json.dumps({"width": 16, "height": 16, "fps": 25, "frame_count": 2, "has_audio": False}),
        encoding="utf-8",
    )
    (work / "subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    (work / "identity.json").write_text(json.dumps(_identity(), ensure_ascii=False), encoding="utf-8")
    (work / "tracks_raw.json").write_text(json.dumps(_tracks()), encoding="utf-8")

    face_a = tmp_path / "wen.jpg"
    face_b = tmp_path / "chen.jpg"
    face_c = tmp_path / "extra.jpg"
    face_a.write_bytes(b"face-wen")
    face_b.write_bytes(b"face-chen")
    face_c.write_bytes(b"face-extra")

    store = Store(tmp_path / "store.db")
    drama = store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "show"), episode_count=1)
    task = store.insert("tasks", drama_id=drama["id"], episode_name="第01集", video_path=str(video))
    wen = store.insert("characters", name="温念", display_name="温念", authorized=1, auth_type=AUTH_LICENSED, confirmed=1)
    chen = store.insert("characters", name="陈秋微", display_name="陈秋微", authorized=1, auth_type=AUTH_OWN_SHOOT, confirmed=1)
    extra = store.insert("characters", name="路人", display_name="路人", authorized=0, confirmed=1)
    store.set_character_asset(wen["id"], "face", str(face_a), complete=True)
    store.set_character_asset(chen["id"], "face", str(face_b), complete=True)
    store.set_character_asset(extra["id"], "face", str(face_c), complete=True)

    calls: list[list[str]] = []

    def fake_run(video_arg, info, track_payload, workdir, cfg, face_image=None, track_id=None, **kwargs):
        refs = list(kwargs.get("references") or [])
        ids = [str(row.get("character_id")) for row in refs]
        calls.append(ids)
        out = Path(workdir) / "replaced.mp4"
        # A second single-person call would overwrite this with one id.
        out.write_bytes(("|".join(ids)).encode("utf-8"))
        plan = {
            "targets": [
                {
                    "character_id": row.get("character_id"),
                    "name": row.get("name"),
                    "face_path": row.get("face_path"),
                    "track_ids": [3, 9] if row.get("character_id") == wen["id"] else [4],
                }
                for row in refs
            ],
            "skipped": [],
            "note": "L1 人物替换是本地换脸，不是提取了3D模型。",
        }
        (Path(workdir) / "replace_plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        return out

    monkeypatch.setattr("recast.stages.replace.run", fake_run)
    summary = run_episode(
        video,
        workdir=work,
        config=RecastConfig(),
        stages=["replace"],
        character_ids=[wen["id"], chen["id"]],
        require_auth=True,
        store=store,
        task_id=task["id"],
    )
    assert calls == [[wen["id"], chen["id"]]]
    assert extra["id"] not in calls[0]
    replaced = work / "replaced.mp4"
    assert replaced.read_bytes() == f"{wen['id']}|{chen['id']}".encode("utf-8")
    assert {row["character_id"] for row in summary["replace_plan"]["targets"]} == {wen["id"], chen["id"]}

    run_episode(
        video,
        workdir=work,
        config=RecastConfig(),
        stages=["replace"],
        character_ids=[wen["id"], chen["id"]],
        require_auth=True,
        store=store,
        task_id=task["id"],
    )
    assert len(calls) == 1
    assert replaced.read_bytes() == f"{wen['id']}|{chen['id']}".encode("utf-8")

    recorded = store.connect().execute(
        "SELECT input_hashes FROM checkpoints WHERE task_id = ? AND step = ?",
        (task["id"], "replace"),
    ).fetchone()
    hashed = " ".join(item.get("path") or "" for item in json.loads(recorded["input_hashes"]))
    assert str(face_a) in hashed
    assert str(face_b) in hashed
    assert "identity.json" in hashed
    assert "tracks_raw.json" in hashed

    face_b.write_bytes(b"face-chen-changed")
    run_episode(
        video,
        workdir=work,
        config=RecastConfig(),
        stages=["replace"],
        character_ids=[wen["id"], chen["id"]],
        require_auth=True,
        store=store,
        task_id=task["id"],
    )
    assert len(calls) == 2
    assert calls[1] == [wen["id"], chen["id"]]


def test_resolve_keeps_only_authorized_faces(tmp_path: Path):
    store = Store(tmp_path / "store.db")
    face = tmp_path / "a.jpg"
    face.write_bytes(b"a")
    other = tmp_path / "b.jpg"
    other.write_bytes(b"b")
    ok = store.insert("characters", name="温念", display_name="温念", authorized=1, auth_type=AUTH_LICENSED, confirmed=1)
    denied = store.insert("characters", name="路人", display_name="路人", authorized=0, confirmed=1)
    store.set_character_asset(ok["id"], "face", str(face), complete=True)
    store.set_character_asset(denied["id"], "face", str(other), complete=True)
    rows = resolve_swap_references(store=store, require_auth=True)
    assert [row["character_id"] for row in rows] == [ok["id"]]
    with pytest.raises(PermissionError):
        resolve_swap_references(store=store, character_id=denied["id"], require_auth=True)
    loose = tmp_path / "cli.jpg"
    loose.write_bytes(b"cli")
    cli = resolve_swap_references(store=None, face=loose, require_auth=False)
    assert cli[0]["face_path"] == str(loose)


def test_service_start_queues_every_authorized_character(tmp_path: Path):
    root = tmp_path / "works" / "剧"
    root.mkdir(parents=True)
    (root / "第01集.mp4").write_bytes(b"v")
    svc = RecastService(tmp_path / "db.db")
    svc.scan(root.parent)
    faces = [tmp_path / "wen.jpg", tmp_path / "chen.jpg", tmp_path / "extra.jpg"]
    for path in faces:
        path.write_bytes(b"img")
    wen = svc.add_character_manual("温念", str(faces[0]))
    chen = svc.add_character_manual("陈秋微", str(faces[1]))
    extra = svc.add_character_manual("路人", str(faces[2]))
    svc.confirm_character(wen["id"], authorized=True, auth_type=AUTH_LICENSED)
    svc.confirm_character(chen["id"], authorized=True, auth_type=AUTH_OWN_SHOOT)

    captured: list = []

    def fake_enqueue(jobs):
        captured.extend(jobs)
        return len(jobs)

    svc.scheduler.enqueue = fake_enqueue
    drama = svc.store.list_dramas()[0]
    svc.start([drama["id"]], ["replace"])
    assert len(captured) == 1
    job = captured[0]
    assert job.character_id is None
    assert set(job.character_ids) == {wen["id"], chen["id"]}
    assert extra["id"] not in (job.character_ids or [])

    captured.clear()
    svc.start([drama["id"]], ["replace"], character_id=wen["id"])
    assert len(captured) == 1
    assert captured[0].character_id == wen["id"]
    assert not captured[0].character_ids
