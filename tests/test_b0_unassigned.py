"""B0-3: faceless tracks share one "未分配" bucket, never each a Person_N.

No InsightFace / GPU / real video here: identity.run() is exercised with a
fake face app (get() returns canned faces, or [] for "no face") and a fake
capture, plus direct tests of the extracted pure function
_assign_unassigned_tracks. ingest_into_library uses a real in-tmp Store;
plan_replace_targets is called directly to lock the skip behaviour.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from recast.config import RecastConfig
from recast.extract.faces import ingest_into_library
from recast.stages import identity as identity_mod
from recast.stages.identity import UNASSIGNED_NAME, _assign_unassigned_tracks
from recast.stages.replace import plan_replace_targets
from recast.store.db import Store

DIM = 64


def _unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _near(base: np.ndarray, seed: int, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(base + rng.normal(scale=scale, size=base.shape))


def _track(tid: int, frame_idx: int, bbox: list[float]) -> dict:
    return {
        "track_id": tid,
        "first_frame": frame_idx,
        "frames": [{"frame_idx": frame_idx, "bbox": list(bbox), "conf": 0.9}],
    }


def _fake_face(emb: np.ndarray) -> SimpleNamespace:
    # Face sits on the head zone of bbox [10, 10, 50, 90] (see _track use).
    return SimpleNamespace(
        embedding=np.asarray(emb, dtype=np.float32),
        bbox=[12.0, 12.0, 48.0, 40.0],
        det_score=0.9,
        pose=None,
    )


class _FakeApp:
    """get() returns the canned faces for the n-th wanted frame, in order."""

    def __init__(self, faces_by_call: list[list]):
        self._faces = list(faces_by_call)
        self._call = 0

    def get(self, frame):
        if self._call < len(self._faces):
            out = self._faces[self._call]
        else:
            out = []
        self._call += 1
        return out


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = list(frames)
        self._idx = 0

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def release(self):
        pass


def _noise_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)


def _run_identity(monkeypatch, tmp_path, tracks, faces_by_call, roster=None):
    """Run identity.run() with the face app + capture mocked out."""
    monkeypatch.setattr(
        identity_mod, "build_face_app", lambda *a, **k: _FakeApp(faces_by_call)
    )
    frames = [_noise_frame(100 + i) for i in range(len(faces_by_call))]
    monkeypatch.setattr(
        identity_mod, "open_capture", lambda video: _FakeCapture(frames)
    )
    monkeypatch.setattr(identity_mod, "vram_report", lambda tag="": "")
    monkeypatch.setattr(identity_mod, "release_cuda", lambda *a: None)
    cfg = RecastConfig()
    cfg.device = "cpu"
    workdir = tmp_path / "work"
    workdir.mkdir()
    payload = identity_mod.run(
        "fake.mp4",
        None,
        {"tracks": tracks},
        workdir,
        cfg,
        roster=roster,
    )
    return payload, workdir


# ------------------------------------------------------- pure function


def test_assign_unassigned_tracks_groups_all_faceless_together():
    cluster_of = {1: 0}
    cluster_first = {0: 0}
    tracks = [
        {"track_id": 2, "first_frame": 5},
        {"track_id": 3, "first_frame": 2},
        {"track_id": 4, "first_frame": 9},
    ]
    first_frame = {1: 0, 2: 5, 3: 2, 4: 9}
    cid = _assign_unassigned_tracks(cluster_of, cluster_first, tracks, first_frame)
    assert cid == 1
    assert cluster_of == {1: 0, 2: 1, 3: 1, 4: 1}
    # Bucket first-appearance is the earliest faceless track, not the last.
    assert cluster_first[1] == 2
    # The real cluster is untouched.
    assert cluster_first[0] == 0


def test_assign_unassigned_tracks_returns_none_when_every_track_has_a_face():
    cluster_of = {1: 0, 2: 0}
    cluster_first = {0: 0}
    tracks = [{"track_id": 1}, {"track_id": 2}]
    cid = _assign_unassigned_tracks(
        cluster_of, cluster_first, tracks, {1: 0, 2: 1}
    )
    assert cid is None
    assert cluster_of == {1: 0, 2: 0}
    assert cluster_first == {0: 0}


def test_assign_unassigned_tracks_empty_input_is_a_noop():
    cluster_of: dict = {}
    cluster_first: dict = {}
    assert _assign_unassigned_tracks(cluster_of, cluster_first, [], {}) is None
    assert cluster_of == {} and cluster_first == {}


# ------------------------------------------------------------- run()


def test_run_all_faceless_tracks_share_one_unassigned_bucket(monkeypatch, tmp_path):
    tracks = [_track(3, 0, [60, 10, 100, 90]), _track(4, 1, [60, 10, 100, 90])]
    payload, workdir = _run_identity(monkeypatch, tmp_path, tracks, [[], []])
    persons = payload["persons"]
    assert len(persons) == 1
    bucket = persons[0]
    assert bucket["name"] == UNASSIGNED_NAME == "未分配"
    assert bucket["auto_name"] == "未分配"
    assert bucket["unassigned"] is True
    assert bucket["embedding"] is None
    assert bucket["face_samples"] == 0
    assert bucket["track_ids"] == [3, 4]
    # No Person_N number was spent on the bucket.
    assert not any(
        str(p.get("auto_name", "")).startswith("Person_") for p in persons
    )
    for track in tracks:
        assert track["name"] == "未分配"
        assert track["cluster_id"] == bucket["cluster_id"]
    # The bucket is kept in identity.json for traceability.
    on_disk = json.loads((workdir / "identity.json").read_text(encoding="utf-8"))
    assert on_disk["persons"] == persons
    assert on_disk["track_names"] == {"3": "未分配", "4": "未分配"}


def test_run_mixed_faced_and_faceless_keeps_person_numbering_gapless(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(41)
    base = _unit(rng.normal(size=DIM))
    tracks = [
        _track(1, 0, [10, 10, 50, 90]),
        _track(2, 1, [10, 10, 50, 90]),
        _track(3, 0, [60, 10, 100, 90]),
        _track(4, 1, [60, 10, 100, 90]),
    ]
    # One face per wanted frame: frame 0 -> track 1, frame 1 -> track 2.
    # Tracks 3/4 (backs of heads) never get a face.
    faces = [[_fake_face(base)], [_fake_face(_near(base, 7))]]
    # A roster hit on a faceless track must NOT rename the bucket.
    payload, _workdir = _run_identity(
        monkeypatch, tmp_path, tracks, faces, roster={"3": "温念"}
    )
    persons = payload["persons"]
    assert [p["auto_name"] for p in persons] == ["Person_1", "未分配"]
    real, bucket = persons
    assert real["track_ids"] == [1, 2]
    assert real["embedding"] is not None
    assert bucket["name"] == "未分配" and bucket["auto_name"] == "未分配"
    assert bucket["unassigned"] is True
    assert bucket["embedding"] is None
    assert bucket["face_samples"] == 0
    assert bucket["track_ids"] == [3, 4]
    assert real["cluster_id"] != bucket["cluster_id"]
    # Faceless tracks were not split off or re-numbered by the outlier check.
    assert len(persons) == 2
    assert [t["name"] for t in tracks] == ["Person_1", "Person_1", "未分配", "未分配"]


# -------------------------------------------------------------- ingest


def _mixed_identity() -> dict:
    rng = np.random.default_rng(42)
    base = _unit(rng.normal(size=DIM))
    return {
        "min_similarity": 0.55,
        "persons": [
            {
                "name": "Person_1",
                "auto_name": "Person_1",
                "cluster_id": 0,
                "track_ids": [1, 2],
                "embedding": [float(x) for x in base],
                "face_samples": 2,
            },
            {
                "name": "未分配",
                "auto_name": "未分配",
                "cluster_id": 1,
                "track_ids": [3, 4],
                "embedding": None,
                "face_samples": 0,
                "unassigned": True,
            },
        ],
        "track_embeddings": {
            "1": [float(x) for x in base],
            "2": [float(x) for x in _near(base, 8)],
        },
    }


def test_ingest_skips_unassigned_bucket_but_keeps_real_persons(tmp_path):
    store = Store(tmp_path / "recast.db")
    out_root = tmp_path / "out"
    ident = _mixed_identity()
    rows = ingest_into_library(
        store,
        ident,
        {"Person_1": ["ep1/a.jpg"], "未分配": []},
        drama_key="ep1",
        output_root=out_root,
    )
    assert len(rows) == 1
    assert len(store.list_characters()) == 1
    assert (rows[0].get("display_name") or rows[0].get("name")) == "Person_1"
    # No store row under the bucket's cluster key, and no folder for it.
    assert store.find_character_by_cluster("ep1:1") is None
    assert not (out_root / "未分配").exists()
    # The bucket person itself is untouched (no merged_into flag etc.).
    assert "merged_into" not in ident["persons"][1]


# -------------------------------------------------------------- replace


def test_plan_replace_targets_skips_embedding_less_person():
    ident = _mixed_identity()
    track_payload = {
        "tracks": [
            {**_track(1, 0, [10, 10, 50, 90]), "cluster_id": 0},
            {**_track(2, 1, [10, 10, 50, 90]), "cluster_id": 0},
            {**_track(3, 0, [60, 10, 100, 90]), "cluster_id": 1},
            {**_track(4, 1, [60, 10, 100, 90]), "cluster_id": 1},
        ]
    }
    rng = np.random.default_rng(43)
    base = np.asarray(ident["persons"][0]["embedding"], dtype=np.float32)
    refs = [
        {
            "face_path": "ref/a.jpg",
            "name": "温念",
            "embedding": [float(x) for x in _near(base, 9)],
        }
    ]
    plan = plan_replace_targets(refs, ident, track_payload)
    assert len(plan["targets"]) == 1
    target = plan["targets"][0]
    assert target["person_name"] == "Person_1"
    assert target["track_ids"] == [1, 2]
    assert not plan["skipped"]
    # No target ever touches a faceless track.
    assert all(tid in (1, 2) for t in plan["targets"] for tid in t["track_ids"])


def test_plan_replace_targets_skips_everything_when_only_unassigned_exists():
    ident = {
        "min_similarity": 0.55,
        "persons": [_mixed_identity()["persons"][1]],
        "track_embeddings": {},
    }
    track_payload = {
        "tracks": [
            {**_track(3, 0, [60, 10, 100, 90]), "cluster_id": 1},
            {**_track(4, 1, [60, 10, 100, 90]), "cluster_id": 1},
        ]
    }
    rng = np.random.default_rng(44)
    base = _unit(rng.normal(size=DIM))
    refs = [
        {"face_path": "ref/a.jpg", "name": "温念", "embedding": [float(x) for x in base]}
    ]
    plan = plan_replace_targets(refs, ident, track_payload)
    assert plan["targets"] == []
    assert len(plan["skipped"]) == 1
