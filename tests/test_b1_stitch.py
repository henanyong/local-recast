"""B1 stitcher: three-vote cross-shot stitching into GPIDs."""

import numpy as np
import pytest

from recast.config import RecastConfig, StitchConfig
from recast.stages import stitch
from recast.stages.stitch import (
    aggregate_costume_hist,
    attach_body_embeddings,
    body_vote,
    build_global_persons,
    costume_hist_from_crop,
    costume_vote,
    face_vote,
    pair_decision,
    stitch_tracklets,
)


def _emb(base: int, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """Unit vector around one-hot ``base``; same base ~ high cosine, other base ~ 0."""
    rng = np.random.default_rng(seed)
    v = np.zeros(512, dtype=np.float32)
    v[base] = 1.0
    if jitter:
        v = v + rng.normal(0, jitter, size=512).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v


def _hist(peak: int) -> list[float]:
    """192-dim costume hist with its mass on one hue bin (peaks far apart ~ orthogonal)."""
    v = np.zeros(192, dtype=np.float64)
    v[peak] = 1.0
    return [float(x) for x in v]


def _sample(emb, weight: float = 0.9, det: float = 0.9, frame: int = 0):
    return {
        "emb": np.asarray(emb, dtype=np.float32),
        "weight": weight,
        "det_score": det,
        "frame_idx": frame,
        "bbox": [10.0, 10.0, 50.0, 50.0],
    }


def _tracklet(
    tid: int,
    first: int,
    last: int,
    *,
    face_base: int | None = 0,
    face_jitter: float = 0.01,
    n_samples: int = 3,
    costume_peak: int | None = 0,
    body_base: int | None = None,
    seed: int = 0,
):
    samples = (
        [_sample(_emb(face_base, face_jitter, seed + i), frame=first + i) for i in range(n_samples)]
        if face_base is not None
        else []
    )
    emb = None
    if samples:
        stacked = np.stack([s["emb"] for s in samples])
        mean = stacked.mean(axis=0)
        emb = [float(x) for x in (mean / np.linalg.norm(mean))]
    return {
        "track_id": tid,
        "first_frame": first,
        "last_frame": last,
        "face_samples": samples,
        "embedding": emb,
        "body_embedding": (
            [float(x) for x in _emb(body_base, 0.01, seed)] if body_base is not None else None
        ),
        "costume_hist": _hist(costume_peak) if costume_peak is not None else None,
        "cluster_id": None,
        "unassigned": face_base is None,
    }


# ----- face vote -----

def test_face_vote_same_person_votes():
    a = [_sample(_emb(0, 0.01, i)) for i in range(3)]
    b = [_sample(_emb(0, 0.01, 10 + i)) for i in range(3)]
    res = face_vote(a, b)
    assert res["vote"] is True and res["veto"] is False
    assert res["best"] >= 0.6 and res["median"] >= 0.5


def test_face_vote_different_person_vetoes():
    a = [_sample(_emb(0, 0.01, i)) for i in range(3)]
    b = [_sample(_emb(7, 0.01, 10 + i)) for i in range(3)]
    res = face_vote(a, b)
    assert res["veto"] is True and res["vote"] is False
    assert res["best"] < 0.3


def test_face_vote_abstains_when_a_side_has_no_face():
    res = face_vote([_sample(_emb(0))], [])
    assert res["abstain"] is True and res["vote"] is False and res["veto"] is False


def test_face_vote_best_high_but_median_low_is_no_vote():
    # One lucky pair is close, the rest are far: best-pair alone must not merge.
    a = [_sample(_emb(0, 0.0, 0), weight=0.9)]
    b = [
        _sample(_emb(0, 0.0, 0), weight=0.9),
        _sample(_emb(5, 0.0, 1), weight=0.8),
        _sample(_emb(6, 0.0, 2), weight=0.8),
    ]
    res = face_vote(a, b)
    assert res["best"] >= 0.9
    assert res["median"] < 0.5
    assert res["vote"] is False


# ----- body / costume votes -----

def test_body_vote_threshold_and_abstain():
    assert body_vote(_emb(0, 0.01, 1), _emb(0, 0.01, 2))["vote"] is True
    assert body_vote(_emb(0), _emb(9))["vote"] is False
    assert body_vote(None, _emb(0))["abstain"] is True
    assert body_vote(_emb(0), None)["abstain"] is True


def test_costume_vote_threshold_and_abstain():
    assert costume_vote(_hist(0), _hist(0))["vote"] is True
    assert costume_vote(_hist(0), _hist(100))["vote"] is False
    assert costume_vote(None, _hist(0))["abstain"] is True


# ----- pair decision -----

def test_pair_decision_two_votes_merge():
    a = _tracklet(1, 0, 50, face_base=0, costume_peak=0)
    b = _tracklet(2, 60, 110, face_base=0, costume_peak=0, seed=20)
    dec = pair_decision(a, b)
    assert dec["decision"] == "merge" and dec["votes"] >= 2


def test_pair_decision_one_vote_is_pending():
    # Face agrees but costume differs and no body emb: exactly 1 vote.
    a = _tracklet(1, 0, 50, face_base=0, costume_peak=0)
    b = _tracklet(2, 60, 110, face_base=0, costume_peak=100, seed=20)
    dec = pair_decision(a, b)
    assert dec["decision"] == "pending" and dec["votes"] == 1


def test_pair_decision_face_veto_overrides_two_other_votes():
    # Body + costume agree, but the faces are different people: veto wins.
    a = _tracklet(1, 0, 50, face_base=0, costume_peak=0, body_base=0)
    b = _tracklet(2, 60, 110, face_base=9, costume_peak=0, body_base=0, seed=20)
    dec = pair_decision(a, b)
    assert dec["decision"] == "no_merge"
    assert dec["face"]["veto"] is True


def test_pair_decision_zero_votes_no_merge():
    a = _tracklet(1, 0, 50, face_base=0, costume_peak=0)
    b = _tracklet(2, 60, 110, face_base=9, costume_peak=100, seed=20)
    dec = pair_decision(a, b)
    assert dec["decision"] == "no_merge" and dec["votes"] == 0


# ----- stitching: golden set, overlap guard, faceless -----

def test_golden_four_fragments_merge_into_one_gpid():
    """Daniel's screenshot: Person_22/24/25/26 are one older woman."""
    frags = [
        _tracklet(22 + i, 0, 40, face_base=3, costume_peak=48, seed=100 + i)
        for i in range(4)
    ]
    # Re-space them so no two overlap in time (different shots).
    for i, t in enumerate(frags):
        t["first_frame"] = i * 100
        t["last_frame"] = i * 100 + 40
    other = _tracklet(30, 500, 540, face_base=8, costume_peak=150, seed=200)
    result = stitch_tracklets(frags + [other])
    assert len(result["groups"]) == 2
    assert sorted(result["groups"][0]) == [22, 23, 24, 25] or sorted(
        [g for g in result["groups"] if len(g) == 4][0]
    ) == [22, 23, 24, 25]
    assert result["unassigned"] == []
    persons = build_global_persons(frags + [other], result)
    assert len(persons) == 2
    big = next(p for p in persons if len(p["track_ids"]) == 4)
    assert big["track_ids"] == [22, 23, 24, 25]
    assert big["gpid"] == 1 and big["cluster_id"] == 1
    assert big["embedding"] is not None and big["confidence"] > 0.5


def test_time_overlapping_tracklets_never_merge():
    # Same face, same costume, but on screen at the same time: two people.
    a = _tracklet(1, 0, 100, face_base=0, costume_peak=0)
    b = _tracklet(2, 50, 150, face_base=0, costume_peak=0, seed=20)
    result = stitch_tracklets([a, b])
    assert result["groups"] == [[1], [2]]
    assert result["decisions"] == []


def test_faceless_tracklet_attaches_on_body_plus_costume():
    faced = _tracklet(1, 0, 50, face_base=0, costume_peak=0, body_base=0)
    faceless = _tracklet(2, 60, 110, face_base=None, costume_peak=0, body_base=0, seed=20)
    result = stitch_tracklets([faced, faceless])
    assert result["groups"] == [[1, 2]]
    assert result["unassigned"] == []
    assert any(p["kind"] == "faceless_attach" for p in result["pending"])
    persons = build_global_persons([faced, faceless], result)
    assert persons[0]["low_confidence"] is True
    assert persons[0]["confidence"] <= 0.49


def test_faceless_tracklet_with_no_match_goes_to_unassigned():
    faced = _tracklet(1, 0, 50, face_base=0, costume_peak=0, body_base=0)
    faceless = _tracklet(2, 60, 110, face_base=None, costume_peak=100, body_base=9, seed=20)
    result = stitch_tracklets([faced, faceless])
    assert result["groups"] == [[1], [2]]
    assert result["unassigned"] == [2]
    persons = build_global_persons([faced, faceless], result)
    # The unassigned tracklet is its own group in the stitch result, but
    # build_global_persons skips it — the bucket is not a person.
    assert len(persons) == 1
    assert persons[0]["track_ids"] == [1]


def test_all_faceless_group_is_skipped_not_a_gpid():
    # Two faceless tracklets that merged on body+costume votes form a
    # group with no face at all — it must not become a faceless GPID.
    a = _tracklet(1, 0, 50, face_base=None, costume_peak=0, body_base=7)
    b = _tracklet(2, 60, 110, face_base=None, costume_peak=0, body_base=7)
    result = stitch_tracklets([a, b])
    assert result["groups"] == [[1, 2]]
    assert result["unassigned"] == []
    assert build_global_persons([a, b], result) == []


def test_gpid_numbers_stay_dense_across_skipped_groups():
    # A skipped group between two real groups must not consume a number:
    # the second real person is Person_2, not Person_3.
    a = _tracklet(1, 0, 50, face_base=0, costume_peak=0, body_base=0)
    faceless = _tracklet(2, 60, 110, face_base=None, costume_peak=100, body_base=9, seed=20)
    b = _tracklet(3, 120, 170, face_base=1, costume_peak=50, body_base=5)
    result = stitch_tracklets([a, faceless, b])
    assert result["groups"] == [[1], [2], [3]]
    persons = build_global_persons([a, faceless, b], result)
    assert [p["auto_name"] for p in persons] == ["Person_1", "Person_2"]
    assert [p["gpid"] for p in persons] == [1, 2]


def test_singleton_tracklet_is_its_own_gpid():
    result = stitch_tracklets([_tracklet(5, 0, 30, face_base=0)])
    assert result["groups"] == [[5]]
    persons = build_global_persons([_tracklet(5, 0, 30, face_base=0)], result)
    assert persons[0]["gpid"] == 1 and persons[0]["track_ids"] == [5]


# ----- precision / recall on synthetic pairs -----

def test_synthetic_pair_precision_recall():
    """30 labelled pairs (15 same / 15 different): P >= 0.9, R >= 0.8."""
    cfg = StitchConfig()
    tp = fp = fn = tn = 0
    for i in range(15):
        a = _tracklet(100 + i, 0, 40, face_base=1, costume_peak=24, seed=1000 + i)
        b = _tracklet(200 + i, 100, 140, face_base=1, costume_peak=24, seed=2000 + i)
        dec = pair_decision(a, b, cfg)["decision"]
        if dec == "merge":
            tp += 1
        elif dec == "pending":
            fn += 1  # pending counts as a miss for recall (human must confirm)
        else:
            fn += 1
    for i in range(15):
        a = _tracklet(300 + i, 0, 40, face_base=2, costume_peak=24, seed=3000 + i)
        b = _tracklet(400 + i, 100, 140, face_base=6 + (i % 3), costume_peak=120, seed=4000 + i)
        dec = pair_decision(a, b, cfg)["decision"]
        if dec == "merge":
            fp += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    assert precision >= 0.9, f"precision {precision:.2f} (tp={tp} fp={fp})"
    assert recall >= 0.8, f"recall {recall:.2f} (tp={tp} fn={fn})"


# ----- costume histogram -----

def test_costume_hist_same_crop_same_hist_different_color_differs():
    red = np.zeros((80, 80, 3), dtype=np.uint8)
    red[:, :, 2] = 200  # BGR red
    blue = np.zeros((80, 80, 3), dtype=np.uint8)
    blue[:, :, 0] = 200  # BGR blue
    box = [0.0, 0.0, 80.0, 80.0]
    h_red = costume_hist_from_crop(red, box)
    h_red2 = costume_hist_from_crop(red, box)
    h_blue = costume_hist_from_crop(blue, box)
    assert h_red is not None and h_blue is not None
    assert stitch.cosine(h_red, h_red2) > 0.99
    assert stitch.cosine(h_red, h_blue) < 0.8
    assert costume_hist_from_crop(red, [0.0, 0.0, 4.0, 4.0]) is None
    assert aggregate_costume_hist([h_red, h_red2]) is not None
    assert aggregate_costume_hist([]) is None
    assert aggregate_costume_hist([None, None]) is None


# ----- body embedding attach (best-effort) -----

class _FailingTagger:
    def embed(self, image):
        raise RuntimeError("no CLIP here")


class _FixedTagger:
    def embed(self, image):
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        return [float(x) for x in v]


def _fake_capture(frames):
    class _Cap:
        def __init__(self):
            self.i = 0

        def read(self):
            if self.i >= len(frames):
                return False, None
            f = frames[self.i]
            self.i += 1
            return True, f

        def release(self):
            pass

    return _Cap()


def test_attach_body_embeddings_tagger_failure_is_not_fatal(monkeypatch):
    t = _tracklet(1, 0, 10, face_base=0)
    payload = {"tracks": [{"track_id": 1, "frames": [{"frame_idx": 0, "bbox": [0, 0, 80, 80]}]}]}
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "recast.media.open_capture", lambda video: _fake_capture([frame])
    )
    done = attach_body_embeddings("dummy.mp4", [t], payload, tagger=_FailingTagger())
    assert done == 0
    assert t.get("body_embedding") is None


def test_attach_body_embeddings_fills_embedding(monkeypatch):
    t = _tracklet(1, 0, 10, face_base=0)
    payload = {"tracks": [{"track_id": 1, "frames": [{"frame_idx": 0, "bbox": [0, 0, 80, 80]}]}]}
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "recast.media.open_capture", lambda video: _fake_capture([frame])
    )
    done = attach_body_embeddings("dummy.mp4", [t], payload, tagger=_FixedTagger())
    assert done == 1
    assert t["body_embedding"] is not None
    assert abs(float(np.linalg.norm(np.asarray(t["body_embedding"]))) - 1.0) < 1e-5


# ----- config wiring -----

def test_stitch_config_defaults_from_yaml():
    cfg = RecastConfig.from_yaml()
    assert cfg.stitch.face_best_min == 0.6
    assert cfg.stitch.face_veto == 0.3
    assert cfg.stitch.body_min == 0.75
    assert cfg.stitch.costume_min == 0.8
    assert cfg.stitch.min_votes == 2
