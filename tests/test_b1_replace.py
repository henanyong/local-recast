"""B1 replace: plan / select by GPID instead of by cluster.

Synthetic identities and track payloads only — no models, no video.
GPID_1 spans tracks [1, 2] that used to be two different clusters
(11 and 12); GPID_2 owns track [3].
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recast.stages.replace import (
    _fill_cluster_embeddings,
    _plan_persons,
    index_swap_frames,
    plan_replace_targets,
    select_target_tracks,
)


def _gpid_identity() -> dict:
    return {
        "cluster_threshold": 0.45,
        "min_similarity": 0.55,
        "persons": [
            {"name": "Person_11", "auto_name": "Person_11", "cluster_id": 11,
             "track_ids": [1], "embedding": [1.0, 0.0, 0.0]},
            {"name": "Person_12", "auto_name": "Person_12", "cluster_id": 12,
             "track_ids": [2], "embedding": [1.0, 0.0, 0.0]},
            {"name": "Person_2", "auto_name": "Person_2", "cluster_id": 2,
             "track_ids": [3], "embedding": [0.0, 1.0, 0.0]},
        ],
        "global_persons": [
            {"gpid": 1, "cluster_id": 1, "name": "Person_1", "auto_name": "Person_1",
             "tracklet_ids": [1, 2], "track_ids": [1, 2],
             "embedding": [1.0, 0.0, 0.0], "confidence": 0.9,
             "cluster_ids": [11, 12], "low_confidence": False},
            {"gpid": 2, "cluster_id": 2, "name": "Person_2", "auto_name": "Person_2",
             "tracklet_ids": [3], "track_ids": [3],
             "embedding": [0.0, 1.0, 0.0], "confidence": 0.9,
             "cluster_ids": [2], "low_confidence": False},
        ],
    }


def _tracks() -> dict:
    return {
        "tracks": [
            {"track_id": 1, "cluster_id": 11,
             "frames": [{"frame_idx": 0, "bbox": [0, 0, 10, 10]},
                        {"frame_idx": 1, "bbox": [0, 0, 10, 10]}]},
            {"track_id": 2, "cluster_id": 12,
             "frames": [{"frame_idx": 0, "bbox": [12, 0, 20, 10]}]},
            {"track_id": 3, "cluster_id": 2,
             "frames": [{"frame_idx": 0, "bbox": [30, 0, 40, 10]}]},
        ]
    }


def _ref(emb, character_id="wen", name="温念", face="wen.jpg"):
    return [{"character_id": character_id, "name": name,
             "face_path": face, "embedding": list(emb)}]


# ----- plan matches by GPID -----

def test_plan_matches_gpid_spanning_two_old_clusters():
    plan = plan_replace_targets(_ref([1.0, 0.0, 0.0]), _gpid_identity(), _tracks())
    assert len(plan["targets"]) == 1
    target = plan["targets"][0]
    assert target["gpid"] == 1
    assert target["track_ids"] == [1, 2]
    assert target["cluster_id"] == 1
    assert plan["skipped"] == []


def test_plan_matches_second_gpid_only():
    plan = plan_replace_targets(_ref([0.0, 1.0, 0.0]), _gpid_identity(), _tracks())
    assert len(plan["targets"]) == 1
    assert plan["targets"][0]["gpid"] == 2
    assert plan["targets"][0]["track_ids"] == [3]
    # Only GPID_2's track is swapped; GPID_1's two tracks are untouched.
    frames = index_swap_frames(_tracks(), plan["targets"])
    assert {item["track_id"] for item in frames[0]} == {3}


def test_plan_persons_prefers_global_persons():
    assert [p["gpid"] for p in _plan_persons(_gpid_identity())] == [1, 2]
    # Empty global_persons falls back to the old cluster persons.
    old = {"persons": [{"cluster_id": 7, "track_ids": [1]}]}
    assert _plan_persons({**old, "global_persons": []}) == old["persons"]
    assert _plan_persons(old) == old["persons"]
    assert _plan_persons(None) == []


# ----- select_target_tracks(person_id=...) -----

def test_select_by_person_id_returns_all_gpid_tracks():
    identity = _gpid_identity()
    assert select_target_tracks(_tracks(), person_id=1, identity=identity) == [1, 2]
    assert select_target_tracks(_tracks(), person_id=2, identity=identity) == [3]
    # String form of the id works too.
    assert select_target_tracks(_tracks(), person_id="1", identity=identity) == [1, 2]


def test_select_by_person_id_narrows_with_track_id():
    identity = _gpid_identity()
    assert select_target_tracks(_tracks(), 2, person_id=1, identity=identity) == [2]
    # A track_id outside the person does not narrow (same as cluster behaviour).
    assert select_target_tracks(_tracks(), 3, person_id=1, identity=identity) == [1, 2]


def test_select_by_person_id_beats_cluster_id():
    identity = _gpid_identity()
    # cluster_id=1 would also resolve to GPID_1 here; use a conflicting pair:
    # person 2's tracks, even though cluster_id points at GPID_1.
    assert select_target_tracks(_tracks(), person_id=2, cluster_id=1, identity=identity) == [3]


def test_select_by_person_id_unknown_raises_with_available_ids():
    identity = _gpid_identity()
    with pytest.raises(RuntimeError) as exc:
        select_target_tracks(_tracks(), person_id=99, identity=identity)
    assert "99" in str(exc.value)
    assert "1" in str(exc.value) and "2" in str(exc.value)


def test_select_by_person_id_matches_old_cluster_id():
    old = {
        "persons": [
            {"name": "温念", "cluster_id": 1, "track_ids": [3, 9],
             "embedding": [1.0, 0.0, 0.0]},
        ]
    }
    payload = {"tracks": [{"track_id": 3, "cluster_id": 1, "frames": []},
                          {"track_id": 9, "cluster_id": 1, "frames": []}]}
    assert select_target_tracks(payload, person_id=1, identity=old) == [3, 9]


# ----- fallback: old payload without global_persons -----

def test_plan_falls_back_to_cluster_persons():
    identity = {
        "persons": [
            {"name": "温念", "auto_name": "Person_1", "cluster_id": 1,
             "track_ids": [3, 9], "embedding": [1.0, 0.0, 0.0]},
            {"name": "陈秋微", "auto_name": "Person_2", "cluster_id": 2,
             "track_ids": [4], "embedding": [0.0, 1.0, 0.0]},
        ]
    }
    payload = {
        "tracks": [
            {"track_id": 3, "cluster_id": 1, "frames": []},
            {"track_id": 9, "cluster_id": 1, "frames": []},
            {"track_id": 4, "cluster_id": 2, "frames": []},
        ]
    }
    plan = plan_replace_targets(
        [{"character_id": "wen", "name": "温念", "face_path": "wen.jpg",
          "embedding": [1.0, 0.0, 0.0]}],
        identity, payload, min_similarity=0.55,
    )
    assert len(plan["targets"]) == 1
    target = plan["targets"][0]
    assert target["track_ids"] == [3, 9]
    # No gpid on a cluster person: gpid falls back to the cluster id.
    assert target["gpid"] == 1
    assert target["cluster_id"] == 1


def test_plan_fallback_second_person_still_matches():
    identity = {
        "persons": [
            {"name": "温念", "cluster_id": 1, "track_ids": [3],
             "embedding": [1.0, 0.0, 0.0]},
            {"name": "陈秋微", "cluster_id": 2, "track_ids": [4],
             "embedding": [0.0, 1.0, 0.0]},
        ]
    }
    payload = {"tracks": [{"track_id": 3, "cluster_id": 1, "frames": []},
                          {"track_id": 4, "cluster_id": 2, "frames": []}]}
    plan = plan_replace_targets(
        [{"character_id": "chen", "name": "陈秋微", "face_path": "chen.jpg",
          "embedding": [0.0, 1.0, 0.0]}],
        identity, payload, min_similarity=0.55,
    )
    assert [t["track_ids"] for t in plan["targets"]] == [[4]]
    assert plan["targets"][0]["gpid"] == 2


# ----- wrong-person guard: margin + per-track re-check -----

def test_plan_skips_when_two_gpids_too_close():
    identity = {
        "global_persons": [
            {"gpid": 1, "cluster_id": 1, "name": "Person_1",
             "track_ids": [1], "embedding": [1.0, 0.0, 0.0]},
            {"gpid": 2, "cluster_id": 2, "name": "Person_2",
             "track_ids": [2], "embedding": [0.995, 0.0998, 0.0]},
        ]
    }
    payload = {"tracks": [{"track_id": 1, "cluster_id": 1, "frames": []},
                          {"track_id": 2, "cluster_id": 2, "frames": []}]}
    plan = plan_replace_targets(
        _ref([1.0, 0.0, 0.0]), identity, payload,
        min_similarity=0.55, margin=0.08,
    )
    assert plan["targets"] == []
    assert len(plan["skipped"]) == 1
    assert "无法确认" in plan["skipped"][0]["reason"]


def test_plan_skips_when_best_gpid_below_threshold():
    plan = plan_replace_targets(
        _ref([0.0, 0.0, 1.0]), _gpid_identity(), _tracks(), min_similarity=0.55,
    )
    assert plan["targets"] == []
    assert plan["skipped"]


def test_split_rejects_mismatched_track_inside_gpid():
    identity = _gpid_identity()
    identity["track_embeddings"] = {
        "1": [1.0, 0.0, 0.0],
        "2": [0.0, 1.0, 0.0],  # track 2 looks like GPID_2, not GPID_1
    }
    plan = plan_replace_targets(_ref([1.0, 0.0, 0.0]), identity, _tracks())
    assert len(plan["targets"]) == 1
    target = plan["targets"][0]
    assert target["gpid"] == 1
    assert target["track_ids"] == [1]
    assert [row["track_id"] for row in target["rejected_tracks"]] == [2]


def test_split_keeps_track_without_embedding():
    identity = _gpid_identity()
    identity["track_embeddings"] = {"1": [1.0, 0.0, 0.0]}  # track 2 has none
    plan = plan_replace_targets(_ref([1.0, 0.0, 0.0]), identity, _tracks())
    assert plan["targets"][0]["track_ids"] == [1, 2]
    assert plan["targets"][0]["rejected_tracks"] == []


# ----- index_swap_frames unchanged -----

def test_index_swap_frames_uses_gpid_track_ids():
    plan = plan_replace_targets(_ref([1.0, 0.0, 0.0]), _gpid_identity(), _tracks())
    frames = index_swap_frames(_tracks(), plan["targets"])
    assert sorted(frames) == [0, 1]
    assert {item["track_id"] for item in frames[0]} == {1, 2}
    assert {item["track_id"] for item in frames[1]} == {1}
    assert {item["source_key"] for item in frames[0]} == {"wen.jpg"}


def test_index_swap_frames_ignores_unmatched_tracks():
    frames = index_swap_frames(
        _tracks(),
        [{"face_path": "wen.jpg", "track_ids": [2]}],
    )
    assert sorted(frames) == [0]
    assert [item["track_id"] for item in frames[0]] == [2]


# ----- _fill_cluster_embeddings fills GPID persons -----

class _FakeBackend:
    def __init__(self, emb):
        self.emb = emb
        self.calls: list[str] = []

    def embed_image(self, path):
        self.calls.append(str(path))
        return list(self.emb)


def _write_samples(work: Path, mapping: dict):
    d = work / "face_samples"
    d.mkdir(parents=True)
    (d / "index.json").write_text(json.dumps(mapping), encoding="utf-8")


def test_fill_embeddings_fills_gpid_person(tmp_path: Path):
    identity = {
        "global_persons": [
            {"gpid": 1, "cluster_id": 1, "name": "Person_1",
             "auto_name": "Person_1", "track_ids": [1], "embedding": None},
            {"gpid": 2, "cluster_id": 2, "name": "Person_2",
             "auto_name": "Person_2", "track_ids": [3],
             "embedding": [0.0, 1.0, 0.0]},
        ]
    }
    _write_samples(tmp_path, {"Person_1": ["sample1.jpg"]})
    backend = _FakeBackend([1.0, 0.0, 0.0])
    _fill_cluster_embeddings(backend, identity, tmp_path)
    assert identity["global_persons"][0]["embedding"] == [1.0, 0.0, 0.0]
    assert identity["global_persons"][1]["embedding"] == [0.0, 1.0, 0.0]
    assert backend.calls == ["sample1.jpg"]


def test_fill_embeddings_still_fills_old_cluster_persons(tmp_path: Path):
    identity = {
        "persons": [
            {"cluster_id": 1, "name": "Person_1", "auto_name": "Person_1",
             "track_ids": [1], "embedding": None},
        ]
    }
    _write_samples(tmp_path, {"Person_1": ["sample1.jpg"]})
    backend = _FakeBackend([0.0, 1.0, 0.0])
    _fill_cluster_embeddings(backend, identity, tmp_path)
    assert identity["persons"][0]["embedding"] == [0.0, 1.0, 0.0]
