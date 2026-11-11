"""Stage 4: InsightFace embeddings, track clustering, and optional roster names."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from tqdm import tqdm

from recast.config import RecastConfig
from recast.device import release_cuda, requested_device, vram_report
from recast.ffmpeg import VideoInfo
from recast.media import open_capture
from recast.namesutil import apply_names
from recast.runtime import check_cancel

log = logging.getLogger("recast")


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _contains(person: list[float], face: list[float]) -> bool:
    cx, cy = _center(face)
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


# Samples below these floors are blurry, tiny, or turned too far to represent the person.
MIN_DET_SCORE = 0.35
MIN_FACE_AREA = 20 * 20
MIN_FRONTAL = 0.2
CLUSTER_TOP_SAMPLES = 8


def frontal_from_face(face: Any) -> float | None:
    """1 is facing the camera. InsightFace pose is pitch, yaw, roll in degrees."""
    pose = getattr(face, "pose", None)
    if pose is None:
        return None
    try:
        yaw = abs(float(list(pose)[1]))
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, 1.0 - yaw / 75.0)


def sample_weight(det_score: float, area: float, frontal: float | None) -> float:
    det = max(0.0, float(det_score))
    size_term = min(1.0, (max(float(area), 0.0) ** 0.5) / 80.0)
    front = 0.75 if frontal is None else max(0.0, min(1.0, float(frontal)))
    return det * (0.35 + 0.65 * size_term) * (0.4 + 0.6 * front)


def usable_sample(sample: dict[str, Any]) -> bool:
    if float(sample.get("det_score") or 0.0) < MIN_DET_SCORE:
        return False
    if float(sample.get("area") or 0.0) < MIN_FACE_AREA:
        return False
    frontal = sample.get("frontal")
    if frontal is not None and float(frontal) < MIN_FRONTAL:
        return False
    return True


def quality_weighted_embedding(samples: list[dict[str, Any]]) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """Mean of the best samples, weighted by detection score, size, and frontal-ness."""
    if not samples:
        return None, []
    kept = [s for s in samples if usable_sample(s) and s.get("emb") is not None]
    if not kept:
        best = max(samples, key=lambda s: float(s.get("det_score") or 0.0))
        if best.get("emb") is None:
            return None, []
        kept = [best]
    kept = sorted(kept, key=lambda s: float(s.get("weight") or 0.0), reverse=True)[:CLUSTER_TOP_SAMPLES]
    weights = np.array([max(float(s.get("weight") or 0.0), 1e-6) for s in kept], dtype=np.float32)
    matrix = np.stack([np.asarray(s["emb"], dtype=np.float32).reshape(-1) for s in kept])
    mean = (matrix * weights[:, None]).sum(axis=0) / float(weights.sum())
    return _l2_normalize(mean), kept


def cluster_distance_and_confidence(
    vectors: list[np.ndarray],
    mean_det: float,
    threshold: float,
) -> tuple[float, float]:
    """Mean pairwise cosine distance, and a 0..1 confidence for later review."""
    if len(vectors) >= 2:
        dists: list[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dists.append(1.0 - float(np.dot(vectors[i], vectors[j])))
        distance = float(sum(dists) / len(dists))
    else:
        distance = 0.0
    span = max(float(threshold), 1e-6)
    tight = 1.0 - min(1.0, max(0.0, distance) / span)
    quality = min(1.0, max(0.0, float(mean_det)))
    return distance, 0.5 * tight + 0.5 * quality


def _providers(device: str) -> list[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_face_app(model_name: str, det_size: int, device: str):
    from insightface.app import FaceAnalysis

    log.info("加载 InsightFace 模型 %s (det_size=%s, device=%s)", model_name, det_size, device)
    log.info("首次运行会下载 buffalo 权重到用户目录 .insightface/models/ 。")
    app = FaceAnalysis(name=model_name, providers=_providers(device))
    ctx_id = 0 if device == "cuda" else -1
    size = int(det_size)
    app.prepare(ctx_id=ctx_id, det_size=(size, size))
    return app


def _associate(
    faces: list[Any],
    detections: list[dict[str, Any]],
    *,
    min_iou: float | None = None,
) -> list[tuple[int, Any]]:
    """Mutual best match of faces to person tracks. Same rule as L1 replace."""
    from recast.stages.face_match import MIN_FACE_IOU, match_detections

    return match_detections(faces, detections, min_iou=MIN_FACE_IOU if min_iou is None else min_iou)


def _cluster_tracks(
    track_vectors: dict[int, np.ndarray],
    threshold: float,
) -> dict[int, int]:
    """Map track_id -> cluster_id using average-linkage cosine distance."""
    ids = sorted(track_vectors)
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: 0}
    matrix = np.stack([track_vectors[i] for i in ids], axis=0)
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=float(threshold),
    )
    labels = clustering.fit_predict(matrix)
    return {tid: int(lab) for tid, lab in zip(ids, labels)}


def run(
    video: str | Path,
    info: VideoInfo,
    track_payload: dict[str, Any],
    workdir: Path,
    cfg: RecastConfig,
    roster: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    device = requested_device(cfg.device)
    vram_report("身份-加载前")
    app = build_face_app(cfg.identity.model, cfg.identity.det_size, device)
    vram_report("身份-已加载")

    frames_of_interest: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sample_every = max(1, int(cfg.identity.sample_every))
    max_faces = max(1, int(cfg.identity.max_faces_per_track))
    per_track_budget: dict[int, int] = defaultdict(int)

    for track in track_payload.get("tracks") or []:
        tid = int(track["track_id"])
        frames = track.get("frames") or []
        sampled = frames[::sample_every] or frames[:1]
        if len(sampled) > max_faces:
            # even spread
            idxs = np.linspace(0, len(sampled) - 1, max_faces, dtype=int)
            sampled = [sampled[i] for i in idxs]
        for item in sampled:
            frames_of_interest[int(item["frame_idx"])].append(
                {"track_id": tid, "bbox": item["bbox"], "conf": item.get("conf")}
            )
            per_track_budget[tid] += 1

    cap = open_capture(video)

    records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    samples_meta: dict[int, list[dict[str, Any]]] = defaultdict(list)
    wanted = set(frames_of_interest)
    last = max(wanted) if wanted else -1
    frame_idx = 0
    pbar = tqdm(total=len(wanted), desc="身份采样", unit="f")
    while wanted and frame_idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        check_cancel()
        if frame_idx in frames_of_interest:
            dets = frames_of_interest[frame_idx]
            faces = app.get(frame) or []
            for tid, face in _associate(faces, dets, min_iou=float(cfg.identity.min_face_iou)):
                emb = _l2_normalize(np.asarray(face.embedding, dtype=np.float32))
                bbox = [float(x) for x in face.bbox]
                area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                det = float(getattr(face, "det_score", 0.0) or 0.0)
                frontal = frontal_from_face(face)
                weight = sample_weight(det, area, frontal)
                meta = {
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "det_score": det,
                    "area": area,
                    "frontal": frontal,
                    "weight": weight,
                }
                samples_meta[tid].append(meta)
                records[tid].append({**meta, "emb": emb})
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()

    means: dict[int, np.ndarray] = {}
    mean_det: dict[int, float] = {}
    for tid, samples in records.items():
        mean, used = quality_weighted_embedding(samples)
        if mean is None:
            continue
        means[tid] = mean
        mean_det[tid] = float(sum(float(s.get("det_score") or 0.0) for s in used) / max(1, len(used)))
        used_frames = {int(s["frame_idx"]) for s in used}
        for meta in samples_meta.get(tid, []):
            meta["used_for_cluster"] = int(meta["frame_idx"]) in used_frames

    cluster_of = _cluster_tracks(means, cfg.identity.cluster_threshold)

    # Stable Person_N order: by first appearance of any track in the cluster.
    first_frame: dict[int, int] = {}
    for track in track_payload.get("tracks") or []:
        first_frame[int(track["track_id"])] = int(track.get("first_frame") or 0)
    cluster_first: dict[int, int] = {}
    for tid, cid in cluster_of.items():
        cluster_first[cid] = min(cluster_first.get(cid, 10**12), first_frame.get(tid, 10**12))
    # tracks without a face each get their own leftover cluster
    next_cid = (max(cluster_of.values()) + 1) if cluster_of else 0
    for track in track_payload.get("tracks") or []:
        tid = int(track["track_id"])
        if tid not in cluster_of:
            cluster_of[tid] = next_cid
            cluster_first[next_cid] = first_frame.get(tid, 10**12)
            next_cid += 1
    ordered_clusters = sorted(cluster_first, key=lambda c: (cluster_first[c], c))
    cluster_name = {cid: f"Person_{i}" for i, cid in enumerate(ordered_clusters, start=1)}

    roster = dict(roster or {})
    track_names: dict[int, str] = {}
    persons: list[dict[str, Any]] = []
    for cid in ordered_clusters:
        member_ids = [tid for tid, c in cluster_of.items() if c == cid]
        auto_name = cluster_name[cid]
        # prefer a roster hit on any member track id
        name = auto_name
        for tid in member_ids:
            name = apply_names(auto_name, tid, roster)
            if name != auto_name:
                break
        for tid in member_ids:
            track_names[tid] = name
        vecs = [means[tid] for tid in member_ids if tid in means]
        embedding = None
        if vecs:
            weights = np.array(
                [max(mean_det.get(tid, 0.05), 0.05) for tid in member_ids if tid in means],
                dtype=np.float32,
            )
            stacked = np.stack(vecs)
            embedding = [
                float(x)
                for x in _l2_normalize((stacked * weights[:, None]).sum(axis=0) / float(weights.sum()))
            ]
        det_values = [mean_det[tid] for tid in member_ids if tid in mean_det]
        distance, confidence = cluster_distance_and_confidence(
            vecs,
            sum(det_values) / len(det_values) if det_values else 0.0,
            float(cfg.identity.cluster_threshold),
        )
        persons.append(
            {
                "name": name,
                "auto_name": auto_name,
                "cluster_id": cid,
                "track_ids": member_ids,
                "face_samples": sum(len(samples_meta.get(tid, [])) for tid in member_ids),
                "first_frame": cluster_first[cid],
                "embedding": embedding,
                "cluster_distance": distance,
                "confidence": confidence,
            }
        )

    for track in track_payload.get("tracks") or []:
        tid = int(track["track_id"])
        track["name"] = track_names.get(tid, f"Person_{tid}")
        track["cluster_id"] = cluster_of.get(tid)
        track["face_samples"] = samples_meta.get(tid, [])

    payload = {
        "model": cfg.identity.model,
        "cluster_threshold": cfg.identity.cluster_threshold,
        "persons": persons,
        "track_names": {str(k): v for k, v in track_names.items()},
        "track_embeddings": {str(tid): [float(x) for x in means[tid]] for tid in means},
        "note": "未提供花名册时使用 Person_1 / Person_2 自动命名；可在 UI 或 --names 中改名。",
    }
    path = workdir / "identity.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("身份聚类完成: %s 个角色, 详情见 %s", len(persons), path)
    for person in persons:
        log.info(
            "  %s  (原 %s, tracks=%s, 人脸样本=%s)",
            person["name"],
            person["auto_name"],
            person["track_ids"],
            person["face_samples"],
        )

    held = app
    app = None
    release_cuda(held)
    return payload
