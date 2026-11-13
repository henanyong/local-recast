"""B1 integration: coordinator's cross-worker fixes.

Covers the gaps found when the five B1 workers landed together:
1. identity.run — an all-faceless stitch group is NOT a GPID person; its
   tracklets join the shared "未分配" bucket (and the stale faceless_attach
   pending entry is dropped). A faceless tracklet attached to a FACED
   GPID stays a low-confidence person (positive control).
2. identity.run — roster names reach GPID persons and track names.
3. faces.run_library — output_root is derived from the store and passed
   to ingest, so body/costume features actually land in folder meta.json
   (the folder is created when there are features to write); an
   in-memory store (no derivable root) stays a silent no-op.

Synthetic embeddings and noise frames only; the face app, capture, and
CLIP body pass are mocked out. No InsightFace / GPU / real video here.
"""

import json
from types import SimpleNamespace

import numpy as np

import recast.extract.faces as faces_mod
from recast.config import RecastConfig
from recast.extract.faces import run_library
from recast.stages import identity as identity_mod
from recast.store.db import Store

DIM = 64


def _unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) for x in _unit(rng.normal(size=DIM))]


def _hist(peak: int) -> list[float]:
    v = np.zeros(192, dtype=np.float64)
    v[peak] = 1.0
    return [float(x) for x in v]


def _emb_onehot(base: int, jitter: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(512, dtype=np.float32)
    v[base] = 1.0
    if jitter:
        v = v + rng.normal(0, jitter, size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _track(tid: int, frame_idx: int, bbox=None) -> dict:
    return {
        "track_id": tid,
        "first_frame": frame_idx,
        "frames": [
            {"frame_idx": frame_idx, "bbox": list(bbox or [10.0, 10.0, 50.0, 50.0]), "conf": 0.9}
        ],
    }


def _fake_face(emb: np.ndarray) -> SimpleNamespace:
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
    def __init__(self, frames):
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


def _no_body_attach(video, tracklets, track_payload=None, **kwargs):
    return 0


def _body_attach_by_map(body_map: dict[int, np.ndarray]):
    def _attach(video, tracklets, track_payload=None, **kwargs):
        done = 0
        for t in tracklets:
            vec = body_map.get(int(t["track_id"]))
            if vec is None:
                continue
            t["body_embedding"] = [float(x) for x in _unit(vec)]
            done += 1
        return done

    return _attach


def _run_identity(monkeypatch, tmp_path, tracks, faces_by_call, roster=None, attach=None):
    monkeypatch.setattr(
        identity_mod, "build_face_app", lambda *a, **k: _FakeApp(faces_by_call)
    )
    last_frame = max(
        (int(f["frame_idx"]) for t in tracks for f in t.get("frames") or []),
        default=0,
    )
    frames = [_noise_frame(100 + i) for i in range(last_frame + 1)]
    monkeypatch.setattr(identity_mod, "open_capture", lambda video: _FakeCapture(frames))
    monkeypatch.setattr(identity_mod, "vram_report", lambda tag="": "")
    monkeypatch.setattr(identity_mod, "release_cuda", lambda *a: None)
    monkeypatch.setattr(
        identity_mod, "attach_body_embeddings", attach or _no_body_attach
    )
    cfg = RecastConfig()
    cfg.device = "cpu"
    workdir = tmp_path / "work"
    workdir.mkdir()
    payload = identity_mod.run(
        "fake.mp4", None, {"tracks": tracks}, workdir, cfg, roster=roster
    )
    return payload, workdir


# ------------------------------------------------- all-faceless group -> bucket


def test_all_faceless_group_joins_bucket_not_gpid(monkeypatch, tmp_path):
    # Two faceless tracklets with the same body + near-uniform costume
    # hists merge (body + costume = 2 votes), but the group has no face:
    # it must NOT become a faceless GPID person.
    body = _emb_onehot(7)
    tracks = [_track(1, 0), _track(2, 50)]
    payload, workdir = _run_identity(
        monkeypatch,
        tmp_path,
        tracks,
        faces_by_call=[],
        attach=_body_attach_by_map({1: body, 2: body}),
    )
    assert payload["global_persons"] == []
    assert [p["name"] for p in payload["persons"]] == ["未分配"]
    # The faceless_attach pending entries for both tracklets are stale
    # (their "GPID" was faceless-only) and must be dropped.
    assert payload["pending_pairs"] == []
    assert payload["track_names"] == {"1": "未分配", "2": "未分配"}
    tracklets = {int(t["track_id"]): t for t in payload["tracklets"]}
    assert len(tracklets) == 2
    for t in tracklets.values():
        assert t["unassigned"] is True
        assert t["embedding"] is None
    saved = json.loads((workdir / "identity.json").read_text(encoding="utf-8"))
    assert saved["global_persons"] == []
    assert saved["pending_pairs"] == []


def test_faceless_attach_to_faced_gpid_stays_person(monkeypatch, tmp_path):
    # Positive control: a faceless tracklet attached to a FACED GPID
    # stays a low-confidence person and is NOT bucketed.
    body = _emb_onehot(7)
    tracks = [_track(1, 0), _track(2, 50)]
    payload, _workdir = _run_identity(
        monkeypatch,
        tmp_path,
        tracks,
        faces_by_call=[[_fake_face(_emb_onehot(0))], []],
        attach=_body_attach_by_map({1: body, 2: body}),
    )
    assert len(payload["global_persons"]) == 1
    person = payload["global_persons"][0]
    assert sorted(person["track_ids"]) == [1, 2]
    assert person["low_confidence"] is True
    assert [p["name"] for p in payload["persons"]] == ["Person_1"]
    assert payload["track_names"] == {"1": "Person_1", "2": "Person_1"}
    pending = payload["pending_pairs"]
    assert len(pending) == 1
    assert pending[0]["kind"] == "faceless_attach"
    assert pending[0]["a_track_id"] == 2


# ------------------------------------------------------- roster -> GPID names


def test_roster_name_reaches_gpid_person_and_track(monkeypatch, tmp_path):
    tracks = [_track(1, 0), _track(2, 50)]
    payload, _workdir = _run_identity(
        monkeypatch,
        tmp_path,
        tracks,
        faces_by_call=[
            [_fake_face(_emb_onehot(0))],
            [_fake_face(_emb_onehot(1))],
        ],
        roster={"1": "温念", "99": "路人"},
    )
    assert len(payload["global_persons"]) == 2
    by_auto = {p["auto_name"]: p for p in payload["global_persons"]}
    assert by_auto["Person_1"]["name"] == "温念"
    assert by_auto["Person_1"]["track_ids"] == [1]
    assert by_auto["Person_2"]["name"] == "Person_2"
    assert all(p["name"] != "路人" for p in payload["global_persons"])
    assert payload["track_names"] == {"1": "温念", "2": "Person_2"}
    # Cluster persons keep the roster name too (old behaviour preserved).
    assert {p["name"] for p in payload["persons"]} == {"温念", "Person_2"}


# ------------------------------------------------- run_library feature wiring


def _sample(frame_idx: int) -> dict:
    return {
        "frame_idx": frame_idx,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "det_score": 0.9,
        "frontal": 1.0,
        "sharpness": 200.0,
        "area": 1600.0,
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
        "cluster_ids": [gpid],
        "face_samples": len(tids),
        "first_frame": 0,
    }
    person.update(extra)
    return person


def _run_library_with_store(monkeypatch, tmp_path, store):
    body, hist = _vec(45), _hist(5)
    emb = _vec(11)
    ident = {
        "persons": [
            {
                "name": "Person_1",
                "auto_name": "Person_1",
                "cluster_id": 1,
                "track_ids": [1],
                "embedding": list(emb),
                "face_samples": 1,
            }
        ],
        "global_persons": [
            _gpid_person(1, emb, [1], body_embedding=body, costume_hist=hist)
        ],
        "tracklets": [],
        "pending_pairs": [],
    }
    monkeypatch.setattr(identity_mod, "run", lambda *a, **k: ident)
    rng = np.random.default_rng(7)
    frames = [
        rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)
        for _ in range(33)
    ]
    monkeypatch.setattr(faces_mod, "open_capture", lambda video: _FakeCapture(frames))
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = run_library(
        "fake.mp4",
        None,
        {"tracks": [{"track_id": 1, "face_samples": [_sample(0)]}]},
        workdir,
        RecastConfig(),
        store=store,
        drama_key="ep1",
    )
    return out, body, hist


def test_run_library_persists_features_to_folder_meta(monkeypatch, tmp_path):
    # End-to-end proof of the output_root wiring: a store under
    # <root>/store/recast.db derives root=<tmp>, ingest writes the
    # body/costume features into <root>/characters/Person_1/meta.json
    # (creating the folder — the import pipeline never creates it first).
    store = Store(tmp_path / "store" / "recast.db")
    out, body, hist = _run_library_with_store(monkeypatch, tmp_path, store)
    assert len(store.list_characters()) == 1
    meta_path = tmp_path / "characters" / "Person_1" / "meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["body_embedding"] == body
    assert meta["costume_hist"] == hist
    assert meta["face_embedding"] is not None
    assert out["_face_samples"]["Person_1"]


def test_run_library_memory_store_has_no_output_root(monkeypatch, tmp_path):
    # Boundary: an in-memory store has no derivable output root, so the
    # feature write is a silent no-op — run_library must still succeed
    # and must not create any characters folder.
    store = Store(":memory:")
    out, _body, _hist = _run_library_with_store(monkeypatch, tmp_path, store)
    assert len(store.list_characters()) == 1
    assert not (tmp_path / "characters").exists()
    assert out["_face_samples"]["Person_1"]
