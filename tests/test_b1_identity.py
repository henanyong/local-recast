"""B1 identity: tracklets + costume hist + stitching into GPIDs.

Synthetic embeddings and noise frames only; the face app, capture, and
CLIP body pass are mocked out. No InsightFace / GPU / real video here.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from recast.config import RecastConfig
from recast.stages import identity as identity_mod
from recast.stages.identity import (
    UNASSIGNED_NAME,
    build_tracklets,
    serialize_tracklet,
)
from recast.stages.stitch import (
    aggregate_costume_hist,
    cosine,
    costume_hist_from_crop,
)

DIM = 64


def _unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _near(base: np.ndarray, seed: int, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(base + rng.normal(scale=scale, size=base.shape))


def _emb_onehot(base: int, jitter: float = 0.01, seed: int = 0) -> np.ndarray:
    """512-d one-hot-ish vector; same base ~ cosine 1, other base ~ 0."""
    rng = np.random.default_rng(seed)
    v = np.zeros(512, dtype=np.float32)
    v[base] = 1.0
    if jitter:
        v = v + rng.normal(0, jitter, size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _hist(peak: int) -> list[float]:
    v = np.zeros(192, dtype=np.float64)
    v[peak] = 1.0
    return [float(x) for x in v]


def _track(tid: int, frame_idx: int, bbox: list[float]) -> dict:
    return {
        "track_id": tid,
        "first_frame": frame_idx,
        "frames": [{"frame_idx": frame_idx, "bbox": list(bbox), "conf": 0.9}],
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


def _no_body_attach(video, tracklets, track_payload=None, **kwargs):
    """CLIP unavailable: no body embedding, body vote abstains."""
    return 0


def _body_attach_by_map(body_map: dict[int, np.ndarray]):
    """Fake CLIP pass: fill body_embedding from a track_id -> vector map."""

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


def _run_identity(
    monkeypatch,
    tmp_path,
    tracks,
    faces_by_call,
    roster=None,
    attach=None,
    events=None,
):
    """Run identity.run() with the face app + capture + body pass mocked out."""
    monkeypatch.setattr(
        identity_mod, "build_face_app", lambda *a, **k: _FakeApp(faces_by_call)
    )
    last_frame = max(
        (int(f["frame_idx"]) for t in tracks for f in t.get("frames") or []),
        default=0,
    )
    frames = [_noise_frame(100 + i) for i in range(last_frame + 1)]
    monkeypatch.setattr(
        identity_mod, "open_capture", lambda video: _FakeCapture(frames)
    )
    monkeypatch.setattr(identity_mod, "vram_report", lambda tag="": "")
    if events is not None:
        def _release(*a, **k):
            events.append("release")

        monkeypatch.setattr(identity_mod, "release_cuda", _release)
    else:
        monkeypatch.setattr(identity_mod, "release_cuda", lambda *a: None)
    if attach is None:
        attach = _no_body_attach
    if events is not None:
        real_attach = attach

        def _recording_attach(video, tracklets, track_payload=None, **kwargs):
            events.append("attach")
            return real_attach(video, tracklets, track_payload, **kwargs)

        attach = _recording_attach
    monkeypatch.setattr(identity_mod, "attach_body_embeddings", attach)
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


def _sample_record(emb, frame_idx=0, weight=0.9, det=0.9):
    return {
        "emb": np.asarray(emb, dtype=np.float32),
        "weight": weight,
        "det_score": det,
        "frame_idx": frame_idx,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "sharpness": 500.0,
        "reject_reason": None,
    }


# ------------------------------------------------------- build_tracklets


def test_build_tracklets_has_all_fields_and_ndarray_embedding():
    rng = np.random.default_rng(51)
    base = _unit(rng.normal(size=DIM))
    records = {
        1: [_sample_record(base, 0), _sample_record(_near(base, 1), 5)],
        2: [_sample_record(_near(base, 2), 10)],
    }
    samples_meta = {
        1: [
            {"frame_idx": 0, "sharpness": 500.0, "reject_reason": None},
            {"frame_idx": 5, "sharpness": 400.0, "reject_reason": None},
        ],
        2: [{"frame_idx": 10, "sharpness": 500.0, "reject_reason": None}],
    }
    means = {1: base, 2: _near(base, 2)}
    tracklets = build_tracklets(
        records, samples_meta, means, {1: 0, 2: 0}, {1: 0, 2: 10},
        {1: _hist(0), 2: _hist(0)},
    )
    assert [t["track_id"] for t in tracklets] == [1, 2]
    t1 = tracklets[0]
    assert set(t1) == {
        "track_id", "first_frame", "last_frame", "face_samples",
        "embedding", "body_embedding", "costume_hist", "cluster_id",
        "unassigned",
    }
    assert t1["first_frame"] == 0 and t1["last_frame"] == 5
    assert t1["cluster_id"] == 0 and t1["unassigned"] is False
    assert isinstance(t1["embedding"], np.ndarray)
    assert np.allclose(t1["embedding"], base)
    assert t1["body_embedding"] is None
    assert t1["costume_hist"] == _hist(0)
    assert len(t1["face_samples"]) == 2
    sample = t1["face_samples"][0]
    assert isinstance(sample["emb"], np.ndarray)
    assert sample["frame_idx"] == 0 and sample["det_score"] == 0.9
    assert sample["sharpness"] == 500.0 and sample["reject_reason"] is None
    assert tracklets[1]["first_frame"] == 10 and tracklets[1]["last_frame"] == 10


def test_build_tracklets_faceless_track_is_unassigned_without_embedding():
    tracklets = build_tracklets(
        {}, {}, {}, {}, {7: 42}, {7: None},
    )
    assert len(tracklets) == 1
    t = tracklets[0]
    assert t["track_id"] == 7
    assert t["first_frame"] == 42 and t["last_frame"] == 42
    assert t["face_samples"] == []
    assert t["embedding"] is None
    assert t["body_embedding"] is None
    assert t["costume_hist"] is None
    assert t["cluster_id"] is None
    assert t["unassigned"] is True


def test_build_tracklets_samples_without_embedding_do_not_count_as_faced():
    # All samples failed to produce an embedding: still faceless.
    records = {3: [{"emb": None, "weight": 0.9, "det_score": 0.9,
                    "frame_idx": 0, "bbox": [10.0, 10.0, 50.0, 50.0]}]}
    tracklets = build_tracklets(records, {}, {}, {}, {3: 0}, {})
    assert tracklets[0]["face_samples"] == []
    assert tracklets[0]["embedding"] is None
    assert tracklets[0]["unassigned"] is True


def test_build_tracklets_aggregates_raw_per_frame_costume_hists():
    hists = [_hist(5), _hist(5), _hist(5)]
    tracklets = build_tracklets(
        {}, {}, {}, {}, {1: 0}, {1: hists},
    )
    assert tracklets[0]["costume_hist"] is not None
    assert cosine(tracklets[0]["costume_hist"], _hist(5)) > 0.99
    # Empty / missing costume input stays None.
    assert build_tracklets({}, {}, {}, {}, {2: 0}, {2: []})[0]["costume_hist"] is None
    assert build_tracklets({}, {}, {}, {}, {3: 0}, None)[0]["costume_hist"] is None


# ------------------------------------------------- serialize_tracklet


def test_serialize_tracklet_drops_emb_and_stays_json_safe():
    rng = np.random.default_rng(52)
    base = _unit(rng.normal(size=DIM))
    tracklets = build_tracklets(
        {1: [_sample_record(base, 0)]},
        {1: [{"frame_idx": 0, "sharpness": 500.0, "reject_reason": None}]},
        {1: base}, {1: 0}, {1: 0}, {1: _hist(3)},
    )
    serial = serialize_tracklet(tracklets[0])
    assert "emb" not in serial["face_samples"][0]
    assert serial["face_samples"][0]["frame_idx"] == 0
    assert serial["face_samples"][0]["sharpness"] == 500.0
    assert serial["face_samples"][0]["reject_reason"] is None
    assert isinstance(serial["embedding"], list)
    assert len(serial["embedding"]) == DIM
    assert serial["costume_hist"] == _hist(3)
    # Round-trips through real JSON (what identity.json does).
    assert json.loads(json.dumps(serial)) == serial


def test_serialize_tracklet_faceless_stays_none_everywhere():
    tracklets = build_tracklets({}, {}, {}, {}, {9: 3}, {})
    serial = serialize_tracklet(tracklets[0])
    assert serial["face_samples"] == []
    assert serial["embedding"] is None
    assert serial["body_embedding"] is None
    assert serial["costume_hist"] is None
    json.dumps(serial)


# ------------------------------------------------------- costume hist


def test_costume_hist_same_person_aggregates_close_to_single_frame():
    red = np.zeros((120, 120, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)  # BGR red
    bbox = [10.0, 10.0, 50.0, 90.0]
    single = costume_hist_from_crop(red, bbox)
    assert single is not None
    agg = aggregate_costume_hist([single, single, single])
    assert agg is not None
    assert cosine(agg, single) > 0.99


def test_costume_hist_different_colours_are_far_apart():
    red = np.zeros((120, 120, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)
    blue = np.zeros((120, 120, 3), dtype=np.uint8)
    blue[:, :] = (255, 0, 0)
    bbox = [10.0, 10.0, 50.0, 90.0]
    h_red = costume_hist_from_crop(red, bbox)
    h_blue = costume_hist_from_crop(blue, bbox)
    assert h_red is not None and h_blue is not None
    assert cosine(h_red, h_blue) < 0.5


def test_costume_hist_aggregate_empty_is_none_and_tiny_box_is_none():
    assert aggregate_costume_hist([]) is None
    assert aggregate_costume_hist([None, None]) is None
    frame = _noise_frame(7)
    assert costume_hist_from_crop(frame, [10.0, 10.0, 12.0, 14.0]) is None


# --------------------------------------------------- stitch integration


def test_run_four_fragments_of_one_person_become_one_gpid(monkeypatch, tmp_path):
    rng = np.random.default_rng(53)
    base = _unit(rng.normal(size=DIM))
    tracks = [
        _track(1, 0, [10, 10, 50, 90]),
        _track(2, 10, [10, 10, 50, 90]),
        _track(3, 20, [10, 10, 50, 90]),
        _track(4, 30, [10, 10, 50, 90]),
    ]
    faces = [
        [_fake_face(base)],
        [_fake_face(_near(base, 1))],
        [_fake_face(_near(base, 2))],
        [_fake_face(_near(base, 3))],
    ]
    payload, workdir = _run_identity(monkeypatch, tmp_path, tracks, faces)
    assert len(payload["global_persons"]) == 1
    gpid = payload["global_persons"][0]
    assert gpid["gpid"] == 1
    assert gpid["auto_name"] == "Person_1"
    assert gpid["track_ids"] == [1, 2, 3, 4]
    assert gpid["embedding"] is not None
    # No bucket: every tracklet hung onto the GPID.
    assert not any(p.get("unassigned") for p in payload["persons"])
    assert [t["name"] for t in tracks] == ["Person_1"] * 4
    # Serialized tracklets on disk carry no emb arrays.
    on_disk = json.loads((workdir / "identity.json").read_text(encoding="utf-8"))
    assert len(on_disk["tracklets"]) == 4
    assert all("emb" not in s for t in on_disk["tracklets"] for s in t["face_samples"])
    assert on_disk["global_persons"] == payload["global_persons"]
    assert on_disk["pending_pairs"] == payload["pending_pairs"]


def test_run_faceless_attaches_to_gpid_and_skips_bucket(monkeypatch, tmp_path):
    rng = np.random.default_rng(54)
    base = _unit(rng.normal(size=DIM))
    other = _unit(rng.normal(size=DIM))
    tracks = [
        _track(1, 0, [10, 10, 50, 90]),
        _track(2, 10, [10, 10, 50, 90]),
        _track(3, 20, [10, 10, 50, 90]),  # faceless, same person (body matches)
        _track(4, 30, [60, 10, 100, 90]),  # faceless, nobody matches
        _track(5, 40, [10, 10, 50, 90]),  # a second, faced person
    ]
    faces = [
        [_fake_face(base)],
        [_fake_face(_near(base, 1))],
        [],
        [],
        [_fake_face(other)],
    ]
    body_map = {
        1: _emb_onehot(0, 0.01, 1),
        2: _emb_onehot(0, 0.01, 2),
        3: _emb_onehot(0, 0.01, 3),  # same body as 1/2
        4: _emb_onehot(9, 0.01, 4),  # unique body: matches nobody
        5: _emb_onehot(7, 0.01, 5),
    }
    payload, workdir = _run_identity(
        monkeypatch, tmp_path, tracks, faces,
        attach=_body_attach_by_map(body_map),
    )
    gpids = payload["global_persons"]
    assert [g["auto_name"] for g in gpids] == ["Person_1", "Person_2"]
    assert gpids[0]["track_ids"] == [1, 2, 3]
    assert gpids[1]["track_ids"] == [5]
    # The attached faceless tracklet is low-confidence and needs a human.
    assert gpids[0]["low_confidence"] is True
    assert any(
        p.get("kind") == "faceless_attach" and p["a_track_id"] == 3
        for p in payload["pending_pairs"]
    )
    # Only the tracklet that attached to nothing enters the bucket.
    buckets = [p for p in payload["persons"] if p.get("unassigned")]
    assert len(buckets) == 1
    assert buckets[0]["track_ids"] == [4]
    assert buckets[0]["name"] == UNASSIGNED_NAME
    # Names follow the GPID; the attached faceless track has no cluster
    # id (it never had a face) but still takes the GPID name.
    assert [t["name"] for t in tracks] == [
        "Person_1", "Person_1", "Person_1", "未分配", "Person_2",
    ]
    by_id = {int(t["track_id"]): t for t in tracks}
    assert by_id[3]["cluster_id"] is None
    assert by_id[4]["cluster_id"] == buckets[0]["cluster_id"]
    assert by_id[1]["cluster_id"] == by_id[2]["cluster_id"]
    on_disk = json.loads((workdir / "identity.json").read_text(encoding="utf-8"))
    assert on_disk["track_names"] == {
        "1": "Person_1", "2": "Person_1", "3": "Person_1",
        "4": "未分配", "5": "Person_2",
    }


def test_run_unattachable_faceless_tracks_still_share_one_bucket(
    monkeypatch, tmp_path
):
    tracks = [
        _track(3, 0, [60, 10, 100, 90]),
        _track(4, 1, [60, 10, 100, 90]),
    ]
    payload, _workdir = _run_identity(monkeypatch, tmp_path, tracks, [[], []])
    assert payload["global_persons"] == []
    # The two faceless tracklets share only the costume vote (1 vote):
    # a near-miss for adjudication, not a merge — both stay unassigned.
    assert len(payload["pending_pairs"]) == 1
    assert payload["pending_pairs"][0]["kind"] == "near_miss"
    assert len(payload["persons"]) == 1
    bucket = payload["persons"][0]
    assert bucket["name"] == UNASSIGNED_NAME
    assert bucket["track_ids"] == [3, 4]
    assert [t["name"] for t in tracks] == ["未分配", "未分配"]


def test_run_different_people_stay_separate_gpids(monkeypatch, tmp_path):
    rng = np.random.default_rng(55)
    base = _unit(rng.normal(size=DIM))
    other = _unit(rng.normal(size=DIM))
    tracks = [
        _track(1, 0, [10, 10, 50, 90]),
        _track(2, 10, [10, 10, 50, 90]),
        _track(3, 20, [10, 10, 50, 90]),
    ]
    faces = [
        [_fake_face(base)],
        [_fake_face(_near(base, 1))],
        [_fake_face(other)],
    ]
    payload, _workdir = _run_identity(monkeypatch, tmp_path, tracks, faces)
    assert [g["track_ids"] for g in payload["global_persons"]] == [[1, 2], [3]]
    assert [t["name"] for t in tracks] == ["Person_1", "Person_1", "Person_2"]


# ------------------------------------------------- failure / boundary


def test_run_survives_a_failing_body_pass(monkeypatch, tmp_path):
    """attach_body_embeddings raising must not fail identity: body abstains."""
    rng = np.random.default_rng(56)
    base = _unit(rng.normal(size=DIM))
    tracks = [_track(1, 0, [10, 10, 50, 90]), _track(2, 10, [10, 10, 50, 90])]

    def _boom(video, tracklets, track_payload=None, **kwargs):
        raise RuntimeError("CLIP 炸了")

    payload, _workdir = _run_identity(
        monkeypatch, tmp_path, tracks,
        [[_fake_face(base)], [_fake_face(_near(base, 1))]],
        attach=_boom,
    )
    # Face + costume votes still merge the two fragments.
    assert [g["track_ids"] for g in payload["global_persons"]] == [[1, 2]]
    assert payload["tracklets"][0]["body_embedding"] is None


def test_run_releases_face_app_before_stitching(monkeypatch, tmp_path):
    events: list[str] = []
    rng = np.random.default_rng(57)
    base = _unit(rng.normal(size=DIM))
    tracks = [_track(1, 0, [10, 10, 50, 90]), _track(2, 10, [10, 10, 50, 90])]
    _run_identity(
        monkeypatch, tmp_path, tracks,
        [[_fake_face(base)], [_fake_face(_near(base, 1))]],
        events=events,
    )
    assert events == ["release", "attach"]


def test_run_with_no_tracks_is_empty_but_valid(monkeypatch, tmp_path):
    payload, workdir = _run_identity(monkeypatch, tmp_path, [], [])
    assert payload["persons"] == []
    assert payload["global_persons"] == []
    assert payload["tracklets"] == []
    assert payload["pending_pairs"] == []
    on_disk = json.loads((workdir / "identity.json").read_text(encoding="utf-8"))
    assert on_disk["global_persons"] == [] and on_disk["tracklets"] == []


# ------------------------------------------------------------- regression


def test_cluster_persons_still_follow_the_old_rules(monkeypatch, tmp_path):
    """Clustering is untouched: Person_N by first appearance, outlier split
    still fragments, cluster_id on tracks is the suggestion cluster."""
    rng = np.random.default_rng(58)
    base = _unit(rng.normal(size=DIM))
    other = _unit(rng.normal(size=DIM))
    tracks = [
        _track(1, 0, [10, 10, 50, 90]),
        _track(2, 10, [10, 10, 50, 90]),
        _track(3, 20, [10, 10, 50, 90]),
    ]
    faces = [
        [_fake_face(base)],
        [_fake_face(_near(base, 1))],
        [_fake_face(other)],
    ]
    payload, _workdir = _run_identity(monkeypatch, tmp_path, tracks, faces)
    persons = [p for p in payload["persons"] if not p.get("unassigned")]
    assert [p["auto_name"] for p in persons] == ["Person_1", "Person_2"]
    assert persons[0]["track_ids"] == [1, 2]
    assert persons[1]["track_ids"] == [3]
    assert persons[0]["cluster_id"] != persons[1]["cluster_id"]
    assert persons[0]["first_frame"] == 0
    assert persons[1]["first_frame"] == 20
    # Cluster persons keep their embeddings and confidence.
    assert persons[0]["embedding"] is not None
    assert persons[0]["confidence"] > 0.0
    by_id = {int(t["track_id"]): t for t in tracks}
    assert by_id[1]["cluster_id"] == persons[0]["cluster_id"]
    assert by_id[3]["cluster_id"] == persons[1]["cluster_id"]


def test_cluster_separation_unchanged_when_persons_differ(monkeypatch, tmp_path):
    """Clustering is untouched: a far-apart track gets its own cluster /
    Person_N instead of joining the tight pair (the outlier-split helper
    itself is covered directly in test_cluster_quality.py)."""
    rng = np.random.default_rng(59)
    base = _unit(rng.normal(size=DIM))
    other = _unit(rng.normal(size=DIM))
    # Three tracks, all mutually close enough to land in one cluster by
    # average linkage, but track 3's mean is far from 1/2.
    tracks = [
        _track(1, 0, [10, 10, 50, 90]),
        _track(2, 10, [10, 10, 50, 90]),
        _track(3, 20, [10, 10, 50, 90]),
    ]
    faces = [
        [_fake_face(base)],
        [_fake_face(_near(base, 1, scale=0.05))],
        [_fake_face(_near(base, 2, scale=0.35))],
    ]
    payload, _workdir = _run_identity(monkeypatch, tmp_path, tracks, faces)
    persons = payload["persons"]
    # The outlier either split off (fragment person) or stayed in its own
    # cluster — either way it must not share Person_1's cluster id, and
    # the two tight tracks must stay together.
    tight = next(p for p in persons if 1 in p["track_ids"])
    assert 2 in tight["track_ids"]
    assert 3 not in tight["track_ids"]
    assert tight["auto_name"] == "Person_1"
