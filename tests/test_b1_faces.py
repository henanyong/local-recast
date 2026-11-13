"""B1 faces: GPID export + ingest + feature persistence.

Synthetic identity payloads + an in-memory Store + a fake capture only.
No InsightFace / GPU / real video. ``identity.run`` is mocked for the
run_library tests; export/ingest are exercised directly elsewhere.
"""

from pathlib import Path

import numpy as np
import pytest

import recast.extract.faces as faces_mod
from recast.config import RecastConfig
from recast.extract.faces import (
    _write_folder_embedding,
    export_face_crops,
    ingest_into_library,
    run_library,
)
from recast.stages import identity as identity_mod
from recast.store.db import Store

DIM = 64


def _unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) for x in _unit(rng.normal(size=DIM))]


def _near(vec: list[float], seed: int, scale: float = 0.01) -> list[float]:
    rng = np.random.default_rng(seed)
    base = np.asarray(vec, dtype=np.float32)
    return [float(x) for x in _unit(base + rng.normal(scale=scale, size=base.shape))]


class _FakeCapture:
    def __init__(self, n_frames: int):
        rng = np.random.default_rng(7)
        self._frames = [
            rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)
            for _ in range(n_frames)
        ]
        self._idx = 0

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def release(self):
        pass


def _sample(frame_idx: int, bbox=None) -> dict:
    return {
        "frame_idx": frame_idx,
        "bbox": list(bbox or [10.0, 10.0, 50.0, 50.0]),
        "det_score": 0.9,
        "frontal": 1.0,
        "sharpness": 200.0,
        "area": 1600.0,
    }


def _track_payload() -> dict:
    """Tracks 1/2/3 with one good face sample each, on far-apart frames."""
    return {
        "tracks": [
            {"track_id": 1, "face_samples": [_sample(0)]},
            {"track_id": 2, "face_samples": [_sample(16)]},
            {"track_id": 3, "face_samples": [_sample(32)]},
        ]
    }


def _old_person(auto: str, cluster_id: int, emb: list[float], tids) -> dict:
    return {
        "name": auto,
        "auto_name": auto,
        "cluster_id": cluster_id,
        "track_ids": list(tids),
        "embedding": list(emb),
        "face_samples": len(tids),
    }


def _gpid_person(gpid: int, emb: list[float], tids, **extra) -> dict:
    person = {
        "gpid": gpid,
        "cluster_id": gpid,
        "name": f"Person_{gpid}",
        "auto_name": f"Person_{gpid}",
        "tracklet_ids": list(tids),
        "track_ids": list(tids),
        "embedding": list(emb) if emb is not None else None,
        "confidence": 0.9,
        "cluster_ids": [0, 1] if len(tids) > 1 else [2],
        "face_samples": len(tids),
        "first_frame": 0,
    }
    person.update(extra)
    return person


def _gpid_ident() -> dict:
    """GPID_1 merges old clusters 0+1 (tracks 1, 2); GPID_2 is track 3."""
    emb1, emb2 = _vec(11), _vec(12)
    return {
        "persons": [
            _old_person("Person_1", 0, emb1, [1]),
            _old_person("Person_2", 1, _near(emb1, 21), [2]),
            _old_person("Person_3", 2, emb2, [3]),
        ],
        "global_persons": [
            _gpid_person(1, emb1, [1, 2]),
            _gpid_person(2, emb2, [3]),
        ],
        "tracklets": [],
        "pending_pairs": [],
    }


def _run_library(monkeypatch, tmp_path, ident, tracks=None):
    monkeypatch.setattr(identity_mod, "run", lambda *a, **k: ident)
    monkeypatch.setattr(
        faces_mod, "open_capture", lambda video: _FakeCapture(33)
    )
    store = Store(":memory:")
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = run_library(
        "fake.mp4",
        None,
        tracks if tracks is not None else _track_payload(),
        workdir,
        RecastConfig(),
        store=store,
        drama_key="ep1",
    )
    return out, store, workdir


# ------------------------------------------------------- run_library: GPID


def test_run_library_exports_by_gpid_not_old_clusters(monkeypatch, tmp_path):
    ident = _gpid_ident()
    out, store, workdir = _run_library(monkeypatch, tmp_path, ident)
    saved = out["_face_samples"]
    # Two GPID folders, not three old-cluster folders.
    assert set(saved) == {"Person_1", "Person_2"}
    # Tracks 1 and 2 (two old clusters) share the GPID_1 folder.
    assert len(saved["Person_1"]) == 2
    assert {Path(p).parent.name for p in saved["Person_1"]} == {"Person_1"}
    assert len(saved["Person_2"]) == 1
    for paths in saved.values():
        for p in paths:
            assert Path(p).is_file()
    # Store rows keyed by GPID, one row per GPID.
    assert len(store.list_characters()) == 2
    assert store.find_character_by_cluster("ep1:1") is not None
    assert store.find_character_by_cluster("ep1:2") is not None
    assert store.find_character_by_cluster("ep1:0") is None
    # The returned ident keeps the old "persons" for traceability.
    assert len(out["persons"]) == 3


def test_run_library_gpid_with_faceless_track_and_empty_gpid(monkeypatch, tmp_path):
    emb1, emb2 = _vec(13), _vec(14)
    ident = {
        "persons": [
            _old_person("Person_1", 0, emb1, [1]),
            _old_person("Person_2", 1, _near(emb1, 22), [2]),
        ],
        "global_persons": [
            # Track 2 has no face samples: GPID_1 exports track 1 only.
            _gpid_person(1, emb1, [1, 2]),
            # No tracks at all: no folder, no row, no crash.
            _gpid_person(2, emb2, []),
        ],
    }
    tracks = {
        "tracks": [
            {"track_id": 1, "face_samples": [_sample(0)]},
            {"track_id": 2, "face_samples": []},
        ]
    }
    out, store, workdir = _run_library(monkeypatch, tmp_path, ident, tracks)
    saved = out["_face_samples"]
    assert set(saved) == {"Person_1"}
    assert len(saved["Person_1"]) == 1
    # The empty GPID exports nothing, but ingest still books a row for it
    # (same as an old cluster person with no samples) — no crash, and no
    # folder/samples for it.
    assert len(store.list_characters()) == 2
    assert store.find_character_by_cluster("ep1:1") is not None
    assert store.find_character_by_cluster("ep1:2") is not None


# -------------------------------------------------- run_library: fallback


def test_run_library_old_payload_without_global_persons(monkeypatch, tmp_path):
    emb1, emb2, emb3 = _vec(15), _vec(16), _vec(17)
    ident = {
        "persons": [
            _old_person("Person_1", 0, emb1, [1]),
            _old_person("Person_2", 1, emb2, [2]),
            _old_person("Person_3", 2, emb3, [3]),
        ]
    }
    out, store, workdir = _run_library(monkeypatch, tmp_path, ident)
    saved = out["_face_samples"]
    assert set(saved) == {"Person_1", "Person_2", "Person_3"}
    assert all(len(v) == 1 for v in saved.values())
    assert len(store.list_characters()) == 3
    assert store.find_character_by_cluster("ep1:0") is not None
    assert store.find_character_by_cluster("ep1:1") is not None
    assert store.find_character_by_cluster("ep1:2") is not None


def test_run_library_empty_global_persons_falls_back(monkeypatch, tmp_path):
    emb1 = _vec(18)
    ident = {
        "persons": [_old_person("Person_1", 0, emb1, [1])],
        "global_persons": [],
    }
    out, store, workdir = _run_library(
        monkeypatch,
        tmp_path,
        ident,
        {"tracks": [{"track_id": 1, "face_samples": [_sample(0)]}]},
    )
    assert set(out["_face_samples"]) == {"Person_1"}
    assert len(store.list_characters()) == 1
    assert store.find_character_by_cluster("ep1:0") is not None


# --------------------------------------------------------- ingest: GPID


def test_ingest_gpid_two_tracks_share_one_row(tmp_path):
    store = Store(":memory:")
    emb = _vec(31)
    ident = {"persons": [_gpid_person(1, emb, [1, 2])]}
    rows = ingest_into_library(
        store, ident, {"Person_1": ["ep1/a.jpg", "ep1/b.jpg"]}, drama_key="ep1"
    )
    assert len(rows) == 1
    assert store.find_character_by_cluster("ep1:1") is not None
    assert len(store.list_characters()) == 1
    assets = [a["path"] for a in rows[0].get("assets") or []]
    assert "ep1/a.jpg" in assets and "ep1/b.jpg" in assets


def test_ingest_two_gpids_make_two_rows_and_reingest_is_idempotent(tmp_path):
    store = Store(":memory:")
    ident = {
        "persons": [
            _gpid_person(1, _vec(32), [1, 2]),
            _gpid_person(2, _vec(33), [3]),
        ]
    }
    samples = {"Person_1": ["ep1/a.jpg"], "Person_2": ["ep1/b.jpg"]}
    rows = ingest_into_library(store, ident, samples, drama_key="ep1")
    assert len(rows) == 2
    assert {r["cluster_key"] for r in rows} == {"ep1:1", "ep1:2"}
    # Same episode again: no duplicate row.
    rows2 = ingest_into_library(store, ident, samples, drama_key="ep1")
    assert len(rows2) == 2
    assert len(store.list_characters()) == 2


def test_ingest_gpid_without_embedding_still_creates_one_row(tmp_path):
    store = Store(":memory:")
    ident = {"persons": [_gpid_person(1, None, [1, 2])]}
    rows = ingest_into_library(
        store, ident, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1"
    )
    assert len(rows) == 1
    assert store.find_character_by_cluster("ep1:1") is not None


# ------------------------------------------------- feature persistence


def _features() -> tuple[list[float], list[float]]:
    body = [round(0.1 * i, 3) for i in range(8)]
    hist = [round(1.0 / 12, 4)] * 12
    return body, hist


def _meta_of(out_root, name):
    from recast.core.characters import character_folder, load_meta

    return load_meta(character_folder(out_root, name))


def test_ingest_new_row_persists_body_and_costume_features(tmp_path):
    from recast.core.characters import ensure_character_folder

    store = Store(":memory:")
    out_root = tmp_path / "out"
    ensure_character_folder(out_root, "Person_1")
    body, hist = _features()
    person = _gpid_person(1, _vec(41), [1], body_embedding=body, costume_hist=hist)
    ingest_into_library(
        store,
        {"persons": [person]},
        {"Person_1": ["ep1/a.jpg"]},
        drama_key="ep1",
        output_root=out_root,
    )
    meta = _meta_of(out_root, "Person_1")
    assert meta is not None
    assert meta["body_embedding"] == body
    assert meta["costume_hist"] == hist
    assert meta["face_embedding"] is not None


def test_ingest_person_without_features_writes_no_feature_keys(tmp_path):
    from recast.core.characters import ensure_character_folder

    store = Store(":memory:")
    out_root = tmp_path / "out"
    ensure_character_folder(out_root, "Person_1")
    person = _gpid_person(1, _vec(42), [1])
    ingest_into_library(
        store,
        {"persons": [person]},
        {"Person_1": ["ep1/a.jpg"]},
        drama_key="ep1",
        output_root=out_root,
    )
    meta = _meta_of(out_root, "Person_1")
    assert meta is not None
    assert "body_embedding" not in meta
    assert "costume_hist" not in meta
    assert meta["face_embedding"] is not None


def test_ingest_features_without_output_root_does_not_raise(tmp_path):
    store = Store(":memory:")
    body, hist = _features()
    person = _gpid_person(1, _vec(43), [1], body_embedding=body, costume_hist=hist)
    rows = ingest_into_library(
        store, {"persons": [person]}, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1"
    )
    assert len(rows) == 1
    assert store.find_character_by_cluster("ep1:1") is not None


def test_ingest_features_without_existing_folder_creates_it_and_persists(
    tmp_path,
):
    # B1 integration: the import pipeline never creates character folders
    # before ingest, so _write_folder_embedding must create the folder
    # when it has features to write — otherwise features would silently
    # never land in meta.json on a fresh import.
    store = Store(":memory:")
    out_root = tmp_path / "out"
    body, hist = _features()
    person = _gpid_person(1, _vec(44), [1], body_embedding=body, costume_hist=hist)
    rows = ingest_into_library(
        store,
        {"persons": [person]},
        {"Person_1": ["ep1/a.jpg"]},
        drama_key="ep1",
        output_root=out_root,
    )
    assert len(rows) == 1
    meta = _meta_of(out_root, "Person_1")
    assert meta is not None
    assert meta["body_embedding"] == body
    assert meta["costume_hist"] == hist


def test_ingest_merge_path_persists_features_into_existing_folder(tmp_path):
    from recast.core.characters import ensure_character_folder

    store = Store(":memory:")
    out_root = tmp_path / "out"
    base = _vec(45)
    # Seed the existing character (no features yet).
    ingest_into_library(
        store,
        {"persons": [_gpid_person(1, base, [1])]},
        {"Person_1": ["ep1/a.jpg"]},
        drama_key="ep1",
        output_root=out_root,
    )
    ensure_character_folder(out_root, "Person_1")
    body, hist = _features()
    # A near-identical embedding merges; features ride along.
    person = _gpid_person(
        2, _near(base, 46), [2], body_embedding=body, costume_hist=hist
    )
    rows = ingest_into_library(
        store,
        {"persons": [person]},
        {"Person_2": ["ep1/b.jpg"]},
        drama_key="ep1",
        output_root=out_root,
    )
    assert len(rows) == 1
    assert person["merged_into"] == "Person_1"
    assert len(store.list_characters()) == 1
    meta = _meta_of(out_root, "Person_1")
    assert meta is not None
    assert meta["body_embedding"] == body
    assert meta["costume_hist"] == hist


def test_write_folder_embedding_features_only_without_face_embedding(tmp_path):
    from recast.core.characters import ensure_character_folder

    out_root = tmp_path / "out"
    folder, _ = ensure_character_folder(out_root, "Person_1")
    body, hist = _features()
    _write_folder_embedding(out_root, "Person_1", None, body_embedding=body, costume_hist=hist)
    meta = _meta_of(out_root, "Person_1")
    assert meta["body_embedding"] == body
    assert meta["costume_hist"] == hist
    assert "face_embedding" not in meta


def test_write_folder_embedding_all_none_is_a_noop(tmp_path):
    from recast.core.characters import ensure_character_folder

    out_root = tmp_path / "out"
    ensure_character_folder(out_root, "Person_1")
    before = _meta_of(out_root, "Person_1")
    _write_folder_embedding(out_root, "Person_1", None)
    _write_folder_embedding(None, "Person_1", [0.5], body_embedding=[0.5])
    assert _meta_of(out_root, "Person_1") == before


# ------------------------------------------------------- unassigned bucket


def test_ingest_gpid_skips_unassigned_bucket(tmp_path):
    store = Store(":memory:")
    out_root = tmp_path / "out"
    ident = {
        "persons": [
            _gpid_person(1, _vec(51), [1, 2]),
            {
                "name": "未分配",
                "auto_name": "未分配",
                "cluster_id": 99,
                "track_ids": [3, 4],
                "embedding": None,
                "face_samples": 0,
                "unassigned": True,
            },
        ]
    }
    rows = ingest_into_library(
        store,
        ident,
        {"Person_1": ["ep1/a.jpg"], "未分配": []},
        drama_key="ep1",
        output_root=out_root,
    )
    assert len(rows) == 1
    assert store.find_character_by_cluster("ep1:99") is None
    assert not (out_root / "characters" / "未分配").exists()


def test_ingest_only_unassigned_returns_no_rows(tmp_path):
    store = Store(":memory:")
    ident = {
        "persons": [
            {
                "name": "未分配",
                "auto_name": "未分配",
                "cluster_id": 99,
                "track_ids": [3, 4],
                "embedding": None,
                "face_samples": 0,
                "unassigned": True,
            }
        ]
    }
    rows = ingest_into_library(store, ident, {"未分配": []}, drama_key="ep1")
    assert rows == []
    assert store.list_characters() == []


# ------------------------------------------------------- export direct check


def test_export_face_crops_groups_gpid_tracks_into_one_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        faces_mod, "open_capture", lambda video: _FakeCapture(33)
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    saved = export_face_crops(
        "fake.mp4", workdir, {"persons": [_gpid_person(1, _vec(61), [1, 2])]},
        _track_payload(),
    )
    assert set(saved) == {"Person_1"}
    assert len(saved["Person_1"]) == 2
    assert {Path(p).parent.name for p in saved["Person_1"]} == {"Person_1"}
