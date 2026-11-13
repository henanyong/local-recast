"""B1 cross-episode merge, three-vote rule (recast/core/merge.py).

Covers score_merge_votes / find_merge_target_votes and the
body_embedding / costume_hist keys load_existing_embeddings reads from
meta.json. All vectors are synthetic — no real model, no video.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from recast.core.merge import (
    find_merge_target,
    find_merge_target_votes,
    load_existing_embeddings,
    score_merge_votes,
)


def _vec(*values: float) -> list[float]:
    """Unit vector from raw components (cosine is then the dot product)."""
    arr = np.array(values, dtype=np.float64)
    arr = arr / np.linalg.norm(arr)
    return [float(x) for x in arr]


def _cos_vec(cos: float) -> list[float]:
    """Unit vector whose cosine to (1, 0, 0) is exactly ``cos``."""
    return [cos, math.sqrt(max(0.0, 1.0 - cos * cos)), 0.0]


def _cand(
    name: str,
    emb: list[float] | None,
    *,
    body: list[float] | None = None,
    costume: list[float] | None = None,
) -> dict:
    return {
        "character_id": f"id:{name}",
        "name": name,
        "embedding": np.array(emb, dtype=np.float32) if emb is not None else None,
        "source": "meta",
        "cluster_key": None,
        "body_embedding": body,
        "costume_hist": costume,
    }


def _new(
    emb: list[float] | None,
    *,
    body: list[float] | None = None,
    costume: list[float] | None = None,
) -> dict:
    return {"embedding": emb, "body_embedding": body, "costume_hist": costume}


# ----- score_merge_votes: three-vote merge -----


def test_votes_merge_face_plus_costume():
    # Face best 0.8, costume 0.9, body missing on both sides -> 2 votes.
    new = _new(_vec(1.0, 0.0, 0.0), costume=_vec(1.0, 0.0, 0.0))
    cand = _cand(
        "温念",
        _cos_vec(0.8),
        costume=_cos_vec(0.9),
    )
    detail = score_merge_votes(new, cand)
    assert detail["decision"] == "merge"
    assert detail["votes"] == 2
    assert detail["face"]["vote"] is True
    assert detail["face"]["best"] == pytest.approx(0.8)
    assert detail["costume"]["vote"] is True
    assert detail["costume"]["sim"] == pytest.approx(0.9)
    assert detail["body"]["abstain"] is True


def test_votes_merge_face_plus_body():
    # Face 0.7 + body 0.9, costume missing on both sides -> 2 votes.
    new = _new(_vec(1.0, 0.0, 0.0), body=_vec(1.0, 0.0, 0.0))
    cand = _cand(
        "温念",
        _cos_vec(0.7),
        body=_cos_vec(0.9),
    )
    detail = score_merge_votes(new, cand)
    assert detail["decision"] == "merge"
    assert detail["votes"] == 2
    assert detail["body"]["sim"] == pytest.approx(0.9)


def test_votes_face_veto_blocks_body_and_costume():
    # Face best 0.0 (< veto 0.3) while body 0.95 and costume 0.95 pass:
    # a face veto is absolute, never merged.
    new = _new(
        _vec(1.0, 0.0, 0.0),
        body=_vec(1.0, 0.0, 0.0),
        costume=_vec(1.0, 0.0, 0.0),
    )
    cand = _cand(
        "温雪",
        _vec(0.0, 1.0, 0.0),
        body=_cos_vec(0.95),
        costume=_cos_vec(0.95),
    )
    detail = score_merge_votes(new, cand)
    assert detail["face"]["veto"] is True
    assert detail["body"]["vote"] is True
    assert detail["costume"]["vote"] is True
    assert detail["decision"] == "no_merge"


def test_votes_single_vote_is_pending_not_merged():
    # Features present on both sides but only the face vote passes
    # (body 0.5 < 0.75, costume 0.5 < 0.8) -> exactly 1 vote -> pending.
    new = _new(
        _vec(1.0, 0.0, 0.0),
        body=_vec(1.0, 0.0, 0.0),
        costume=_vec(1.0, 0.0, 0.0),
    )
    cand = _cand(
        "温念",
        _cos_vec(0.8),
        body=_cos_vec(0.5),
        costume=_cos_vec(0.5),
    )
    detail = score_merge_votes(new, cand)
    assert detail["votes"] == 1
    assert detail["decision"] == "pending"


def test_votes_zero_votes_is_no_merge():
    # Face below the vote bar but above the veto (0.5), body and costume
    # both below their bars -> 0 votes -> no_merge, not pending.
    new = _new(
        _vec(1.0, 0.0, 0.0),
        body=_vec(1.0, 0.0, 0.0),
        costume=_vec(1.0, 0.0, 0.0),
    )
    cand = _cand(
        "路人",
        _cos_vec(0.5),
        body=_cos_vec(0.5),
        costume=_cos_vec(0.5),
    )
    detail = score_merge_votes(new, cand)
    assert detail["votes"] == 0
    assert detail["decision"] == "no_merge"


def test_votes_missing_new_face_abstains():
    # No face embedding on the new side: face abstains, body+costume
    # alone can still merge (2 votes), face best is None.
    new = _new(
        None,
        body=_vec(1.0, 0.0, 0.0),
        costume=_vec(1.0, 0.0, 0.0),
    )
    cand = _cand(
        "温念",
        _vec(1.0, 0.0, 0.0),
        body=_cos_vec(0.9),
        costume=_cos_vec(0.9),
    )
    detail = score_merge_votes(new, cand)
    assert detail["face"]["abstain"] is True
    assert detail["face"]["best"] is None
    assert detail["decision"] == "merge"


# ----- find_merge_target_votes: selection + fallback -----


def test_find_votes_picks_merge_candidate():
    new = _new(_vec(1.0, 0.0, 0.0), costume=_vec(1.0, 0.0, 0.0))
    good = _cand("温念", _cos_vec(0.8), costume=_cos_vec(0.9))
    bad = _cand("温雪", _vec(0.0, 1.0, 0.0), costume=_cos_vec(0.9))
    target, best_sim, detail = find_merge_target_votes(new, [bad, good])
    assert target is good
    assert best_sim == pytest.approx(0.8)
    assert detail is not None and detail["decision"] == "merge"


def test_find_votes_prefers_more_votes_then_face_best():
    # Both merge, but A has 3 votes vs B's 2 -> A wins despite B's
    # slightly better face pair.
    new = _new(
        _vec(1.0, 0.0, 0.0),
        body=_vec(1.0, 0.0, 0.0),
        costume=_vec(1.0, 0.0, 0.0),
    )
    a = _cand(
        "甲",
        _cos_vec(0.8),
        body=_cos_vec(0.9),
        costume=_cos_vec(0.9),
    )
    b = _cand(
        "乙",
        _cos_vec(0.85),
        body=_cos_vec(0.5),
        costume=_cos_vec(0.9),
    )
    target, best_sim, detail = find_merge_target_votes(new, [b, a])
    assert target is a
    assert detail is not None and detail["votes"] == 3

    # Same votes (2), higher face best wins.
    c = _cand(
        "丙",
        _cos_vec(0.7),
        body=_cos_vec(0.5),
        costume=_cos_vec(0.9),
    )
    target, best_sim, detail = find_merge_target_votes(new, [c, b])
    assert target is b
    assert best_sim == pytest.approx(0.85)


def test_find_votes_no_merge_returns_best_face_and_detail():
    # Vetoed candidate + pending candidate: nothing merges, the best
    # face similarity (0.8, the pending one) and its detail come back.
    new = _new(
        _vec(1.0, 0.0, 0.0),
        body=_vec(1.0, 0.0, 0.0),
        costume=_vec(1.0, 0.0, 0.0),
    )
    vetoed = _cand(
        "温雪",
        _vec(0.0, 1.0, 0.0),
        body=_cos_vec(0.95),
        costume=_cos_vec(0.95),
    )
    pending = _cand(
        "温念",
        _cos_vec(0.8),
        body=_cos_vec(0.5),
        costume=_cos_vec(0.5),
    )
    target, best_sim, detail = find_merge_target_votes(new, [vetoed, pending])
    assert target is None
    assert best_sim == pytest.approx(0.8)
    assert detail is not None and detail["decision"] == "pending"


def test_find_votes_fallback_face_threshold_when_no_features():
    # Neither side has any body/costume feature: single-face rule,
    # IdentityConfig.merge_similarity = 0.6.
    new = _new(_vec(1.0, 0.0, 0.0))
    above = _cand("温念", _cos_vec(0.7))
    below = _cand("温雪", _cos_vec(0.3))
    target, best_sim, detail = find_merge_target_votes(new, [below, above])
    assert target is above
    assert best_sim == pytest.approx(0.7)
    assert detail is not None and detail["decision"] == "merge"

    target, best_sim, detail = find_merge_target_votes(new, [below])
    assert target is None
    assert best_sim == pytest.approx(0.3)
    assert detail is not None and detail["decision"] == "no_merge"


def test_find_votes_fallback_when_only_one_side_has_features():
    # New person has a costume feature but the candidate has none:
    # that pair falls back to the face threshold (0.7 >= 0.6 merges),
    # it must not be forced into a 1-vote pending.
    new = _new(_vec(1.0, 0.0, 0.0), costume=_vec(1.0, 0.0, 0.0))
    cand = _cand("温念", _cos_vec(0.7))
    target, best_sim, detail = find_merge_target_votes(new, [cand])
    assert target is cand
    assert best_sim == pytest.approx(0.7)


def test_find_votes_none_embedding_returns_empty():
    new = _new(None, body=_vec(1.0, 0.0, 0.0), costume=_vec(1.0, 0.0, 0.0))
    cand = _cand(
        "温念",
        _vec(1.0, 0.0, 0.0),
        body=_vec(1.0, 0.0, 0.0),
        costume=_vec(1.0, 0.0, 0.0),
    )
    assert find_merge_target_votes(new, [cand]) == (None, 0.0, None)
    # No candidates at all is the same empty result.
    assert find_merge_target_votes(_new(_vec(1.0, 0.0, 0.0)), []) == (None, 0.0, None)


# ----- load_existing_embeddings reads body / costume -----


def _store(path: Path):
    from recast.store.db import Store

    return Store(path)


def _meta_folder(out: Path, name: str, meta: dict) -> None:
    from recast.core.characters import ensure_character_folder, load_meta, save_meta

    folder, _ = ensure_character_folder(out, name)
    current = load_meta(folder) or {}
    current.update(meta)
    save_meta(folder, current)


def test_load_embeddings_reads_body_and_costume_from_meta(tmp_path: Path):
    out = tmp_path / "outputs"
    body = _vec(1.0, 0.0, 0.0)
    hist = _vec(0.6, 0.8, 0.0)
    _meta_folder(
        out,
        "温念",
        {
            "face_embedding": _vec(1.0, 0.0, 0.0),
            "body_embedding": body,
            "costume_hist": hist,
        },
    )
    entries = load_existing_embeddings(None, out)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "温念"
    assert entry["source"] == "meta"
    assert "body_embedding" in entry and "costume_hist" in entry
    assert entry["body_embedding"] == pytest.approx(body)
    assert entry["costume_hist"] == pytest.approx(hist)


def test_load_embeddings_old_meta_without_features_returns_none(tmp_path: Path):
    out = tmp_path / "outputs"
    _meta_folder(out, "温念", {"face_embedding": _vec(1.0, 0.0, 0.0)})
    entries = load_existing_embeddings(None, out)
    assert len(entries) == 1
    assert entries[0]["body_embedding"] is None
    assert entries[0]["costume_hist"] is None


def test_load_embeddings_malformed_features_become_none(tmp_path: Path):
    out = tmp_path / "outputs"
    _meta_folder(
        out,
        "温念",
        {
            "face_embedding": _vec(1.0, 0.0, 0.0),
            "body_embedding": "not-a-list",
            "costume_hist": [1.0, "x", 0.5],
        },
    )
    entries = load_existing_embeddings(None, out)
    assert len(entries) == 1
    assert entries[0]["body_embedding"] is None
    assert entries[0]["costume_hist"] is None


def test_load_embeddings_store_entries_carry_none_features(tmp_path: Path):
    out = tmp_path / "outputs"
    store = _store(out / "store" / "recast.db")
    store.insert(
        "characters",
        name="温念",
        display_name="温念",
        confirmed=0,
        authorized=0,
        cluster_key="cluster:温念",
        face_embedding=json.dumps(_vec(1.0, 0.0, 0.0)),
    )
    entries = load_existing_embeddings(store)
    assert len(entries) == 1
    assert entries[0]["source"] == "store"
    assert entries[0]["body_embedding"] is None
    assert entries[0]["costume_hist"] is None


# ----- find_merge_target itself is unchanged -----


def test_find_merge_target_behavior_unchanged():
    cand = _cand("温念", _vec(1.0, 0.0, 0.0))
    target, sim = find_merge_target(_cos_vec(0.7), [cand], 0.6)
    assert target is cand
    assert sim == pytest.approx(0.7)
    target, sim = find_merge_target(_cos_vec(0.3), [cand], 0.6)
    assert target is None
    assert sim == pytest.approx(0.3)
    # No embedding at all: (None, 0.0), same as before.
    assert find_merge_target(None, [cand], 0.6) == (None, 0.0)
