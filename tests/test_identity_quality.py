"""Cluster means follow sharp frontal samples, and the payload carries a confidence."""

import numpy as np

from recast.stages.identity import (
    CLUSTER_TOP_SAMPLES,
    cluster_distance_and_confidence,
    quality_weighted_embedding,
    sample_weight,
    usable_sample,
)


def _sample(emb, det, area, frontal, weight=None):
    return {
        "emb": np.asarray(emb, dtype=np.float32),
        "det_score": det,
        "area": area,
        "frontal": frontal,
        "weight": sample_weight(det, area, frontal) if weight is None else weight,
    }


def test_sharp_sample_outweighs_a_blurry_one():
    sharp = _sample([1.0, 0.0], 0.95, 80 * 80, 1.0)
    blur = _sample([0.0, 1.0], 0.36, 40 * 40, 0.3)
    mean, used = quality_weighted_embedding([sharp, blur])
    assert mean is not None and len(used) == 2
    assert float(mean[0]) > float(mean[1])
    unweighted = np.array([0.5, 0.5], dtype=np.float32)
    unweighted /= np.linalg.norm(unweighted)
    assert float(mean[0]) > float(unweighted[0])


def test_low_quality_samples_are_dropped_before_the_mean():
    assert usable_sample({"det_score": 0.2, "area": 80 * 80, "frontal": 1.0}) is False
    assert usable_sample({"det_score": 0.9, "area": 8 * 8, "frontal": 1.0}) is False
    assert usable_sample({"det_score": 0.9, "area": 80 * 80, "frontal": 0.05}) is False
    profile = _sample([0.0, 1.0], 0.9, 80 * 80, 0.05)
    sharp = _sample([1.0, 0.0], 0.9, 80 * 80, 1.0)
    mean, used = quality_weighted_embedding([profile, sharp])
    assert len(used) == 1
    assert float(mean[0]) > 0.9


def test_only_the_best_samples_are_clustered():
    samples = [
        _sample([1.0, 0.0], 0.5, 60 * 60, 1.0, weight=float(i))
        for i in range(1, 12)
    ]
    _mean, used = quality_weighted_embedding(samples)
    assert len(used) == CLUSTER_TOP_SAMPLES
    assert min(float(s["weight"]) for s in used) == 4.0


def test_cluster_payload_exposes_distance_and_confidence():
    same = np.array([1.0, 0.0], dtype=np.float32)
    distance, confidence = cluster_distance_and_confidence([same, same], 0.9, 0.45)
    assert distance == 0.0
    assert confidence > 0.9
    other = np.array([0.0, 1.0], dtype=np.float32)
    far, low = cluster_distance_and_confidence([same, other], 0.4, 0.45)
    assert far > distance
    assert low < confidence
