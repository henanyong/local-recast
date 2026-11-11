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


def _associate(faces: list[Any], detections: list[dict[str, Any]]) -> list[tuple[int, Any]]:
    """Greedy match of faces to person tracks in the same frame."""
    used_faces: set[int] = set()
    assigned: list[tuple[int, Any]] = []
    for det in detections:
        best_i, best_score = -1, 0.0
        for i, face in enumerate(faces):
            if i in used_faces:
                continue
            fb = [float(x) for x in face.bbox]
            score = _iou(det["bbox"], fb)
            if score < 0.05 and _contains(det["bbox"], fb):
                score = max(score, 0.15)
            if score > best_score:
                best_i, best_score = i, score
        if best_i >= 0 and best_score >= 0.05:
            used_faces.add(best_i)
            assigned.append((int(det["track_id"]), faces[best_i]))
    return assigned


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

    embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
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
            for tid, face in _associate(faces, dets):
                emb = _l2_normalize(np.asarray(face.embedding, dtype=np.float32))
                embeddings[tid].append(emb)
                samples_meta[tid].append(
                    {
                        "frame_idx": frame_idx,
                        "bbox": [float(x) for x in face.bbox],
                        "det_score": float(getattr(face, "det_score", 0.0) or 0.0),
                    }
                )
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()

    means: dict[int, np.ndarray] = {}
    for tid, vecs in embeddings.items():
        if not vecs:
            continue
        means[tid] = _l2_normalize(np.mean(np.stack(vecs, axis=0), axis=0))

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
        persons.append(
            {
                "name": name,
                "auto_name": auto_name,
                "cluster_id": cid,
                "track_ids": member_ids,
                "face_samples": sum(len(samples_meta.get(tid, [])) for tid in member_ids),
                "first_frame": cluster_first[cid],
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
