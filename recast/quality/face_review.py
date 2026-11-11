"""Post-replace samples for the face-drift check.

Picks a few frames from each replaced person, then compares a reference
embedding with the embedding of the swapped face. InsightFace is optional:
callers can pass embeddings directly so the review path is testable offline.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Callable

from recast.quality.review import ReviewItem, run_quality_checks

log = logging.getLogger("recast")

Embedder = Callable[[dict[str, Any]], tuple[list[float], list[float]] | None]


def _spread(items: list[Any], count: int) -> list[Any]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indexes: list[int] = []
    for step in range(count):
        index = round(step * (len(items) - 1) / (count - 1))
        if index not in indexes:
            indexes.append(index)
    return [items[index] for index in indexes]


def _shot_at(shots: list[dict[str, Any]] | None, moment: float | None) -> dict[str, Any] | None:
    if moment is None:
        return None
    for shot in shots or []:
        start, end = shot.get("start_time"), shot.get("end_time")
        if start is None or end is None:
            continue
        if float(start) - 0.05 <= float(moment) <= float(end) + 0.05:
            return shot
    return None


def select_replace_samples(
    track_payload: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    shots: list[dict[str, Any]] | None = None,
    *,
    per_face: int = 3,
) -> list[dict[str, Any]]:
    """Frames to compare after a replace. One cluster's tracks are already grouped in the plan."""
    tracks = {int(track["track_id"]): track for track in (track_payload or {}).get("tracks") or [] if track.get("track_id") is not None}
    samples: list[dict[str, Any]] = []
    for target in (plan or {}).get("targets") or []:
        frames: list[tuple[int, dict[str, Any]]] = []
        for raw_id in target.get("track_ids") or []:
            track = tracks.get(int(raw_id))
            if not track:
                continue
            for item in track.get("frames") or []:
                if item.get("bbox"):
                    frames.append((int(raw_id), item))
        frames.sort(key=lambda pair: int(pair[1].get("frame_idx") or 0))
        for track_id, item in _spread(frames, per_face):
            moment = item.get("time")
            try:
                moment_f = float(moment) if moment is not None else None
            except (TypeError, ValueError):
                moment_f = None
            shot = _shot_at(shots, moment_f)
            frame_idx = int(item.get("frame_idx") or 0)
            code = str((shot or {}).get("shot_code") or target.get("person_name") or target.get("name") or f"f{frame_idx}")
            samples.append(
                {
                    "shot_code": code,
                    "shot_id": (shot or {}).get("id"),
                    "frame_idx": frame_idx,
                    "time": moment_f,
                    "bbox": item.get("bbox"),
                    "track_id": track_id,
                    "reference": target.get("face_path"),
                    "person_name": target.get("person_name") or target.get("name"),
                }
            )
    return samples


def _copy_reference_thumb(sample: dict[str, Any], directory: Path) -> str | None:
    reference = sample.get("reference")
    if not reference or not Path(str(reference)).is_file():
        return sample.get("thumbnail")
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{sample.get('shot_code')}_f{sample.get('frame_idx')}.jpg"
    try:
        shutil.copyfile(reference, dest)
    except OSError as exc:
        log.info("终检缩略图未写出: %s", exc)
        return None
    return str(dest)


def post_replace_review_items(
    shots: list[dict[str, Any]] | None,
    track_payload: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    embedder: Embedder | None,
    *,
    thumbnail_dir: str | Path | None = None,
    per_face: int = 3,
    face_threshold: float = 0.35,
    max_cps: float = 4.0,
) -> list[ReviewItem]:
    """Run drift on post-replace samples, plus the existing shot checks."""
    samples = select_replace_samples(track_payload, plan, shots, per_face=per_face)
    thumb_root = Path(thumbnail_dir) if thumbnail_dir else None
    pairs: list[dict[str, Any]] = []
    for sample in samples:
        if thumb_root is not None:
            sample["thumbnail"] = _copy_reference_thumb(sample, thumb_root)
        vectors = embedder(sample) if embedder is not None else None
        if not vectors:
            continue
        library, observed = vectors
        pairs.append(
            {
                **sample,
                "library_embedding": library,
                "observed_embedding": observed,
            }
        )
    return run_quality_checks(
        list(shots or []),
        face_pairs=pairs,
        max_cps=max_cps,
        face_threshold=face_threshold,
    )


def try_embed_sample(sample: dict[str, Any], video: str | Path | None = None) -> tuple[list[float], list[float]] | None:
    """InsightFace embeddings of the reference still and one replaced frame.

    Returns None when InsightFace or the frame is unavailable. Weights are not
    downloaded or committed by this check.
    """
    reference = sample.get("reference")
    if not reference or not Path(str(reference)).is_file():
        return None
    if not video or not Path(video).is_file():
        return None
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        log.info("未安装 InsightFace，换脸相似度未计算: %s", exc)
        return None
    try:
        import cv2
    except Exception as exc:
        log.info("未安装 OpenCV，换脸相似度未计算: %s", exc)
        return None
    app = getattr(try_embed_sample, "_app", None)
    if app is None:
        app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        setattr(try_embed_sample, "_app", app)

    def _vec(image) -> list[float] | None:
        faces = app.get(image) or []
        if not faces:
            return None
        faces = sorted(faces, key=lambda face: float(getattr(face, "det_score", 0.0) or 0.0), reverse=True)
        emb = getattr(faces[0], "normed_embedding", None)
        if emb is None:
            emb = getattr(faces[0], "embedding", None)
        if emb is None:
            return None
        return [float(x) for x in emb]

    ref = cv2.imread(str(reference))
    if ref is None:
        return None
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample.get("frame_idx") or 0))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    bbox = sample.get("bbox") or []
    if len(bbox) >= 4:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            crop = frame[y1:y2, x1:x2]
            thumb = sample.get("thumbnail")
            if thumb:
                cv2.imwrite(str(thumb), crop)
            observed = _vec(crop)
        else:
            observed = _vec(frame)
    else:
        observed = _vec(frame)
    library = _vec(ref)
    if library is None or observed is None:
        return None
    return library, observed
