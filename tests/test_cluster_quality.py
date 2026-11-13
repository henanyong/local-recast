"""Cluster quality: outlier split, sharpness filter, cross-episode merge.

Synthetic embeddings (numpy random vectors + hand-built near/far pairs)
and synthetic face images (PIL checkerboard vs Gaussian-blurred) only.
No real video, no detection model.
"""

import json

import numpy as np
import pytest
from PIL import Image, ImageFilter

from recast.config import IdentityConfig
from recast.core.merge import (
    find_merge_target,
    load_existing_embeddings,
    merge_embeddings,
    parse_embedding,
)
from recast.core.review import _laplacian_variance
from recast.extract.faces import (
    ingest_into_library,
    rejected_samples,
    score_reference_sample,
    select_reference_crops,
)
from recast.stages.identity import (
    quality_weighted_embedding,
    sample_reject_reason,
    sample_weight,
    split_cluster_outliers,
    usable_sample,
)
from recast.store.db import Store

DIM = 64


def _unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _near(base: np.ndarray, seed: int, scale: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(base + rng.normal(scale=scale, size=base.shape))


# ---------------------------------------------------------------- split


def test_split_removes_one_outlier_from_a_mixed_cluster():
    rng = np.random.default_rng(11)
    base = _unit(rng.normal(size=DIM))
    members = {
        1: _near(base, 1),
        2: _near(base, 2),
        3: _unit(rng.normal(size=DIM)),
    }
    kept, outliers = split_cluster_outliers(
        members,
        mean_det={1: 0.9, 2: 0.9, 3: 0.9},
        max_mean_distance=0.45,
        min_member_similarity=0.55,
    )
    assert kept == [1, 2]
    assert [o["track_id"] for o in outliers] == [3]
    assert "0." in outliers[0]["reason"] and "相似度" in outliers[0]["reason"]


def test_split_keeps_a_tight_cluster_whole():
    rng = np.random.default_rng(12)
    base = _unit(rng.normal(size=DIM))
    members = {tid: _near(base, tid) for tid in (1, 2, 3)}
    kept, outliers = split_cluster_outliers(
        members,
        mean_det={1: 0.9, 2: 0.9, 3: 0.9},
        max_mean_distance=0.45,
        min_member_similarity=0.55,
    )
    assert kept == [1, 2, 3]
    assert outliers == []


def test_split_pair_keeps_the_better_detected_member():
    rng = np.random.default_rng(13)
    a = _unit(rng.normal(size=DIM))
    b = _unit(rng.normal(size=DIM))
    kept, outliers = split_cluster_outliers(
        {1: a, 2: b},
        mean_det={1: 0.9, 2: 0.5},
        max_mean_distance=0.45,
        min_member_similarity=0.55,
    )
    assert kept == [1]
    assert [o["track_id"] for o in outliers] == [2]


def test_split_uses_the_config_defaults():
    cfg = IdentityConfig()
    rng = np.random.default_rng(14)
    base = _unit(rng.normal(size=DIM))
    members = {1: _near(base, 1), 2: _near(base, 2), 3: _unit(rng.normal(size=DIM))}
    kept, outliers = split_cluster_outliers(
        members,
        max_mean_distance=cfg.cluster_max_mean_distance,
        min_member_similarity=cfg.cluster_min_member_similarity,
    )
    assert kept == [1, 2]
    assert len(outliers) == 1


# ------------------------------------------------------------- sharpness


def _checker(size: int = 140) -> Image.Image:
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if (x // 7 + y // 7) % 2 else 0
    return img


def _lap(img: Image.Image) -> float:
    return _laplacian_variance(np.asarray(img.convert("L")))


def test_laplacian_scale_separates_sharp_from_blurred():
    sharp = _lap(_checker())
    blurred = _lap(_checker().filter(ImageFilter.GaussianBlur(radius=3)))
    assert sharp > 150.0
    assert blurred < 150.0


def _face_sample(emb, det, area, frontal, sharpness):
    return {
        "emb": np.asarray(emb, dtype=np.float32),
        "det_score": det,
        "area": area,
        "frontal": frontal,
        "sharpness": sharpness,
        "weight": sample_weight(det, area, frontal),
    }


def test_reject_reasons_cover_all_four_floors():
    good = {"det_score": 0.9, "area": 80 * 80, "frontal": 1.0, "sharpness": 500.0}
    assert sample_reject_reason(good) is None
    assert usable_sample(good) is True
    assert sample_reject_reason({**good, "det_score": 0.2}) == "检测分过低"
    assert sample_reject_reason({**good, "area": 10 * 10}) == "太小"
    assert sample_reject_reason({**good, "frontal": 0.05}) == "侧脸"
    assert sample_reject_reason({**good, "sharpness": 20.0}) == "模糊"
    # Sharpness never measured (old payloads): absence never rejects.
    assert sample_reject_reason({"det_score": 0.9, "area": 80 * 80, "frontal": 1.0}) is None
    # Area falls back to the bbox, like score_reference_sample does.
    assert sample_reject_reason(
        {"det_score": 0.9, "frontal": 1.0, "bbox": [0, 0, 80, 80]}
    ) is None
    assert sample_reject_reason(
        {"det_score": 0.9, "frontal": 1.0, "bbox": [0, 0, 10, 10]}
    ) == "太小"


def test_blurry_sample_is_excluded_from_the_cluster_mean():
    sharp = _face_sample([1.0, 0.0], 0.95, 80 * 80, 1.0, _lap(_checker()))
    blur = _face_sample([0.0, 1.0], 0.95, 80 * 80, 1.0, _lap(_checker().filter(ImageFilter.GaussianBlur(radius=3))))
    assert sample_reject_reason(blur) == "模糊"
    mean, used = quality_weighted_embedding([sharp, blur])
    assert len(used) == 1
    assert float(mean[0]) > 0.9


def test_reference_score_prefers_sharp_and_ignores_unmeasured():
    base = {"frame_idx": 0, "bbox": [0, 0, 80, 80], "det_score": 0.9, "frontal": 1.0}
    sharp = {**base, "sharpness": 500.0}
    blur = {**base, "sharpness": 20.0}
    assert score_reference_sample(sharp) > score_reference_sample(blur)
    assert score_reference_sample({**base, "sharpness": 10000.0}) == score_reference_sample(base)


def test_select_reference_crops_skips_blurry_samples():
    good = {"frame_idx": 0, "bbox": [0, 0, 80, 80], "det_score": 0.9, "frontal": 1.0, "sharpness": 500.0}
    blur = {"frame_idx": 40, "bbox": [0, 0, 80, 80], "det_score": 0.99, "frontal": 1.0, "sharpness": 20.0}
    chosen = select_reference_crops([good, blur], limit=8)
    assert [int(s["frame_idx"]) for s in chosen] == [0]
    bad = rejected_samples([good, blur])
    assert len(bad) == 1 and bad[0]["reject_reason"] == "模糊"


# ----------------------------------------------------------------- merge


def test_parse_embedding_rejects_garbage():
    assert parse_embedding(None) is None
    assert parse_embedding("not json") is None
    assert parse_embedding([0.0, 0.0]) is None
    assert parse_embedding([]) is None
    vec = parse_embedding(json.dumps([3.0, 4.0]))
    assert vec is not None and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_find_merge_target_matches_same_person_only():
    rng = np.random.default_rng(21)
    base = _unit(rng.normal(size=DIM))
    candidates = [
        {"character_id": "c1", "name": "温念", "embedding": base, "source": "store"}
    ]
    target, sim = find_merge_target(_near(base, 5), candidates, 0.6)
    assert target is not None and target["name"] == "温念" and sim > 0.9
    target, sim = find_merge_target(_unit(rng.normal(size=DIM)), candidates, 0.6)
    assert target is None and sim < 0.6


def test_merge_embeddings_stays_on_the_same_person():
    rng = np.random.default_rng(22)
    base = _unit(rng.normal(size=DIM))
    merged = merge_embeddings(base, _near(base, 6))
    assert merged is not None
    assert float(np.dot(_unit(merged), base)) > 0.95
    assert np.allclose(merge_embeddings(None, base), _unit(base), atol=1e-6)
    assert np.allclose(merge_embeddings(base, None), _unit(base), atol=1e-6)


def _person(auto: str, cluster_id: int, embedding) -> dict:
    return {
        "name": auto,
        "auto_name": auto,
        "cluster_id": cluster_id,
        "track_ids": [1],
        "embedding": [float(x) for x in embedding] if embedding is not None else None,
    }


def test_ingest_merges_same_person_across_episodes(tmp_path):
    store = Store(tmp_path / "recast.db")
    rng = np.random.default_rng(31)
    base = _unit(rng.normal(size=DIM))
    ident1 = {"persons": [_person("Person_1", 0, base)]}
    rows = ingest_into_library(
        store, ident1, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1"
    )
    assert len(store.list_characters()) == 1
    assert rows[0].get("face_embedding")
    assert "merged_into" not in ident1["persons"][0]

    ident2 = {"persons": [_person("Person_1", 0, _near(base, 9))]}
    rows2 = ingest_into_library(
        store, ident2, {"Person_1": ["ep2/b.jpg"]}, drama_key="ep2"
    )
    assert len(store.list_characters()) == 1
    assert ident2["persons"][0]["merged_into"] == "Person_1"
    assert ident2["persons"][0]["merge_similarity"] > 0.9
    assets = [a for a in rows2[0].get("assets") or [] if a.get("kind") == "face"]
    assert {a["path"] for a in assets} == {"ep1/a.jpg", "ep2/b.jpg"}
    stored = parse_embedding(rows2[0]["face_embedding"])
    # Merged embedding is the mean of the two episodes' embeddings: still
    # clearly the same person (near ~0.93 to base, mean ~0.98), not a copy
    # of either side.
    assert stored is not None and float(np.dot(stored, base)) > 0.95


def test_ingest_creates_a_new_character_for_a_different_person(tmp_path):
    store = Store(tmp_path / "recast.db")
    rng = np.random.default_rng(32)
    base = _unit(rng.normal(size=DIM))
    ingest_into_library(store, {"persons": [_person("Person_1", 0, base)]}, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1")
    other = _unit(rng.normal(size=DIM))
    ident2 = {"persons": [_person("Person_1", 0, other)]}
    ingest_into_library(store, ident2, {"Person_1": ["ep2/b.jpg"]}, drama_key="ep2")
    assert len(store.list_characters()) == 2
    assert "merged_into" not in ident2["persons"][0]


def test_ingest_is_idempotent(tmp_path):
    store = Store(tmp_path / "recast.db")
    rng = np.random.default_rng(33)
    base = _unit(rng.normal(size=DIM))
    ident = {"persons": [_person("Person_1", 0, base)]}
    ingest_into_library(store, ident, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1")
    first = store.list_characters()[0]
    ingest_into_library(store, ident, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1")
    chars = store.list_characters()
    assert len(chars) == 1
    assert chars[0]["face_embedding"] == first["face_embedding"]
    assets = [a for a in chars[0].get("assets") or [] if a.get("kind") == "face"]
    assert len(assets) == 1


def test_ingest_without_embedding_still_creates_a_character(tmp_path):
    store = Store(tmp_path / "recast.db")
    ident = {"persons": [_person("Person_1", 0, None)]}
    rows = ingest_into_library(store, ident, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1")
    assert len(rows) == 1
    assert len(store.list_characters()) == 1


def test_ingest_marks_rejected_crops_with_their_reason(tmp_path):
    store = Store(tmp_path / "recast.db")
    rng = np.random.default_rng(34)
    base = _unit(rng.normal(size=DIM))
    ident = {
        "persons": [_person("Person_1", 0, base)],
        "_rejected_face_samples": {
            "Person_1": [{"path": "ep1/rejected_模糊_00_f3.jpg", "reason": "模糊", "frame_idx": 3}]
        },
    }
    rows = ingest_into_library(
        store,
        ident,
        {"Person_1": ["ep1/00_f0.jpg", "ep1/rejected_模糊_00_f3.jpg"]},
        drama_key="ep1",
    )
    assets = {a["path"]: a for a in rows[0].get("assets") or []}
    assert assets["ep1/rejected_模糊_00_f3.jpg"]["notes"] == "待审核：模糊"
    assert assets["ep1/00_f0.jpg"]["notes"] == "聚类样本"


def test_meta_only_match_does_not_merge_without_a_store_row(tmp_path):
    from recast.core.characters import ensure_character_folder, load_meta, save_meta

    out_root = tmp_path / "out"
    folder, _meta = ensure_character_folder(out_root, "温念")
    rng = np.random.default_rng(35)
    base = _unit(rng.normal(size=DIM))
    meta = load_meta(folder)
    meta["face_embedding"] = [float(x) for x in base]
    save_meta(folder, meta)
    assert load_existing_embeddings(None, out_root)[0]["character_id"] is None

    store = Store(tmp_path / "recast.db")
    ident = {"persons": [_person("Person_1", 0, _near(base, 3))]}
    ingest_into_library(
        store, ident, {"Person_1": ["ep1/a.jpg"]}, drama_key="ep1", output_root=out_root
    )
    assert len(store.list_characters()) == 1
    assert "merged_into" not in ident["persons"][0]
