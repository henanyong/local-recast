"""Build a local character library from InsightFace clusters + face crops."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from recast.config import RecastConfig
from recast.ffmpeg import VideoInfo
from recast.media import open_capture
from recast.runtime import check_cancel
from recast.store.db import Store

log = logging.getLogger("recast")


def _save_crop(frame, bbox: list[float], dest: Path, pad: float = 0.25) -> Path | None:
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Unicode-safe write on Windows
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    dest.write_bytes(buf.tobytes())
    return dest


def score_reference_sample(sample: dict[str, Any]) -> float:
    """Higher is a better reference still: clear, large, and facing the camera."""
    det = float(sample.get("det_score") or 0.0)
    bbox = sample.get("bbox") or [0, 0, 0, 0]
    try:
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError, IndexError):
        width, height = 0.0, 0.0
    area = float(sample["area"]) if sample.get("area") else width * height
    size_term = min(1.0, (max(area, 0.0) ** 0.5) / 120.0)
    frontal = sample.get("frontal")
    front = 0.75 if frontal is None else max(0.0, min(1.0, float(frontal)))
    return det * (0.45 + 0.55 * size_term) * (0.4 + 0.6 * front)


def select_reference_crops(
    samples: list[dict[str, Any]],
    *,
    limit: int = 8,
    min_frame_gap: int = 8,
) -> list[dict[str, Any]]:
    """Top samples by quality, spread across time instead of the first frames."""
    if limit <= 0:
        return []
    ranked = sorted(samples, key=score_reference_sample, reverse=True)
    best = score_reference_sample(ranked[0]) if ranked else 0.0
    floor = best * 0.45
    chosen: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for sample in ranked:
        if len(chosen) >= limit:
            break
        score = score_reference_sample(sample)
        frame = int(sample.get("frame_idx") or 0)
        if best > 0 and score < floor:
            continue
        if any(abs(frame - int(other.get("frame_idx") or 0)) < min_frame_gap for other in chosen):
            deferred.append(sample)
            continue
        chosen.append(sample)
    for sample in deferred:
        if len(chosen) >= limit:
            break
        if best > 0 and score_reference_sample(sample) < floor:
            continue
        chosen.append(sample)
    chosen.sort(key=lambda sample: int(sample.get("frame_idx") or 0))
    return chosen[:limit]


def export_face_crops(
    video: str | Path,
    workdir: Path,
    ident: dict[str, Any],
    track_payload: dict[str, Any],
    *,
    max_per_person: int = 8,
) -> dict[str, list[str]]:
    """Write the best face crops under workdir/face_samples/<auto_name>/."""
    tracks = {int(t["track_id"]): t for t in (track_payload.get("tracks") or [])}
    wanted: dict[int, list[dict[str, Any]]] = {}
    for person in ident.get("persons") or []:
        candidates: list[dict[str, Any]] = []
        for tid in person.get("track_ids") or []:
            for sample in (tracks.get(int(tid)) or {}).get("face_samples") or []:
                candidates.append({**sample, "tid": int(tid), "person": person})
        for sample in select_reference_crops(candidates, limit=max_per_person):
            wanted.setdefault(int(sample["frame_idx"]), []).append(
                {"tid": int(sample["tid"]), "bbox": sample["bbox"], "person": sample["person"]}
            )

    dest_root = workdir / "face_samples"
    dest_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    saved: dict[str, list[str]] = {}
    if not wanted:
        return saved

    cap = open_capture(video)
    last = max(wanted)
    frame_idx = 0
    try:
        while frame_idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            check_cancel()
            if frame_idx in wanted:
                for item in wanted[frame_idx]:
                    person = item["person"]
                    auto = person.get("auto_name") or person.get("name") or f"Person_{item['tid']}"
                    n = counts.get(auto, 0)
                    if n >= max_per_person:
                        continue
                    dest = dest_root / auto / f"{n:02d}_f{frame_idx}.jpg"
                    path = _save_crop(frame, item["bbox"], dest)
                    if path is not None:
                        counts[auto] = n + 1
                        saved.setdefault(auto, []).append(str(path))
            frame_idx += 1
    finally:
        cap.release()

    index = workdir / "face_samples" / "index.json"
    index.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("已导出人脸样本: %s 个角色", len(saved))
    return saved


def ingest_into_library(
    store: Store | None,
    ident: dict[str, Any],
    samples: dict[str, list[str]],
    *,
    task_id: str | None = None,
    drama_key: str | None = None,
) -> list[dict[str, Any]]:
    if store is None:
        return list(ident.get("persons") or [])
    out: list[dict[str, Any]] = []
    for person in ident.get("persons") or []:
        auto = person.get("auto_name") or person.get("name") or "Person"
        cluster_id = person.get("cluster_id")
        key = f"{drama_key or 'global'}:{cluster_id if cluster_id is not None else auto}"
        paths = samples.get(auto) or []
        row = store.upsert_character_candidate(
            cluster_key=key,
            auto_name=auto,
            sample_paths=paths,
            source_task_id=task_id,
        )
        if person.get("name") and person["name"] != auto and not row.get("confirmed"):
            store.update("characters", row["id"], name=person["name"], display_name=person["name"])
            row = store.get_character(row["id"]) or row
        out.append(row)
    return out


def run_library(
    video: str | Path,
    info: VideoInfo,
    track_payload: dict[str, Any],
    workdir: Path,
    cfg: RecastConfig,
    *,
    roster: Mapping[str, str] | None = None,
    store: Store | None = None,
    task_id: str | None = None,
    drama_key: str | None = None,
) -> dict[str, Any]:
    from recast.stages import identity

    ident = identity.run(video, info, track_payload, workdir, cfg, roster=roster)
    samples = export_face_crops(video, workdir, ident, track_payload)
    ingest_into_library(store, ident, samples, task_id=task_id, drama_key=drama_key)
    ident["_face_samples"] = samples
    return ident
