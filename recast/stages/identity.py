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
from recast.core.review import _laplacian_variance
from recast.device import release_cuda, requested_device, vram_report
from recast.ffmpeg import VideoInfo
from recast.media import open_capture
from recast.namesutil import apply_names
from recast.runtime import check_cancel
from recast.stages.stitch import (
    aggregate_costume_hist,
    attach_body_embeddings,
    build_global_persons,
    costume_hist_from_crop,
    stitch_tracklets,
)

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
# Laplacian-variance floor for sharpness. Same scale as
# recast/core/review.py score_sample_file (sharp photos are typically
# > 150, blurry ones < 50); see sample_reject_reason for how the two
# measurements line up. The value itself lives in IdentityConfig
# (min_sharpness); this constant is the default when no config is given.
MIN_SHARPNESS = 150.0
CLUSTER_TOP_SAMPLES = 8
# Name of the shared bucket for tracks that never yielded a face
# (back of the head, long shot, face occluded). It is not a person and
# never takes a Person_N number; see _assign_unassigned_tracks.
UNASSIGNED_NAME = "未分配"


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


def _face_sharpness(frame, bbox: list[float]) -> float | None:
    """Laplacian variance of the face region, same scale as review.py.

    review.py's :func:`score_sample_file` measures the saved crop file;
    this measures the same face straight from the video frame at detection
    time (detection score / bbox / frontal-ness are not persisted anywhere,
    so the file score cannot be recomputed later). Same function
    (:func:`recast.core.review._laplacian_variance`), same 150 scale — the
    numbers differ only by the crop margin (tight box here, 25% padding in
    the saved crop), same order of magnitude, directly comparable.

    None means "not measured" (empty box), not "blurry".
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return _laplacian_variance(gray)


def sample_reject_reason(sample: dict[str, Any], *, min_sharpness: float = MIN_SHARPNESS) -> str | None:
    """Why this sample must not join the cluster core, or None when usable.

    Reasons, in check order: "检测分过低" / "太小" / "侧脸" / "模糊".
    The reason is recorded on the sample meta and in identity.json, and
    travels with the exported crop, so the review queue can show why the
    sample was held back. A missing "sharpness" key means "never measured"
    (old payloads, unit tests): only a measured value below the floor
    rejects — absence never does.
    """
    if float(sample.get("det_score") or 0.0) < MIN_DET_SCORE:
        return "检测分过低"
    area = sample.get("area")
    if area is None:
        # Same fallback as score_reference_sample: derive it from the box
        # when the sample never recorded one.
        bbox = sample.get("bbox") or [0, 0, 0, 0]
        try:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1])
            )
        except (TypeError, ValueError, IndexError):
            area = 0.0
    if float(area or 0.0) < MIN_FACE_AREA:
        return "太小"
    frontal = sample.get("frontal")
    if frontal is not None and float(frontal) < MIN_FRONTAL:
        return "侧脸"
    sharpness = sample.get("sharpness")
    if sharpness is not None and float(sharpness) < min_sharpness:
        return "模糊"
    return None


def usable_sample(sample: dict[str, Any], *, min_sharpness: float = MIN_SHARPNESS) -> bool:
    return sample_reject_reason(sample, min_sharpness=min_sharpness) is None


def quality_weighted_embedding(
    samples: list[dict[str, Any]], *, min_sharpness: float = MIN_SHARPNESS
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """Mean of the best samples, weighted by detection score, size, and frontal-ness.

    Samples held back by :func:`sample_reject_reason` (blurry, tiny,
    profile, low detection score) never join ``kept``. Only when every
    sample of a track is held back does the old fallback apply — the best
    detectable face is used so the track keeps an embedding at all, and
    its rejection reason stays recorded on the sample meta.
    """
    if not samples:
        return None, []
    kept = [s for s in samples if usable_sample(s, min_sharpness=min_sharpness) and s.get("emb") is not None]
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


def split_cluster_outliers(
    member_embeddings: dict[int, np.ndarray],
    *,
    mean_det: dict[int, float] | None = None,
    max_mean_distance: float,
    min_member_similarity: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Re-check one finished cluster and split out members that don't belong.

    Average linkage only guarantees the *between-group* distance at merge
    time, so a finished cluster can still hide an outlier (e.g. two faces
    of one person plus one face of another). Two triggers, either one
    splits:

    - the cluster's mean pairwise cosine distance is above
      ``max_mean_distance``;
    - a member's mean cosine similarity to the other members is below
      ``min_member_similarity``.

    The worst member (lowest mean similarity to the rest, lowest detection
    score breaking ties) is removed first, then the cluster is re-checked,
    until it passes or one member is left. Removed members never become
    characters on their own: the caller gives each one a fresh fragment
    cluster that lands in the review queue as unconfirmed.

    Returns (kept track ids, outliers as [{track_id, reason}]).
    Members without an embedding cannot be judged and always stay.
    """
    ids = sorted(member_embeddings)
    if len(ids) < 2:
        return ids, []
    kept = list(ids)
    outliers: list[dict[str, Any]] = []
    det = mean_det or {}

    def _mean_sim_to_rest(members: list[int], tid: int) -> float:
        vec = _l2_normalize(np.asarray(member_embeddings[tid], dtype=np.float64))
        others = [o for o in members if o != tid]
        sims = [
            float(np.dot(vec, _l2_normalize(np.asarray(member_embeddings[o], dtype=np.float64))))
            for o in others
        ]
        return float(sum(sims) / len(sims)) if sims else 0.0

    while len(kept) >= 2:
        vecs = [np.asarray(member_embeddings[tid], dtype=np.float32) for tid in kept]
        distance, _ = cluster_distance_and_confidence(vecs, 0.0, max_mean_distance)
        sims = {tid: _mean_sim_to_rest(kept, tid) for tid in kept}
        worst = min(kept, key=lambda tid: (sims[tid], float(det.get(tid, 0.0))))
        if sims[worst] >= min_member_similarity and distance <= max_mean_distance:
            break
        if sims[worst] >= min_member_similarity:
            reason = (
                f"聚类平均成对距离 {distance:.2f} > {max_mean_distance:.2f}，"
                f"成员与其它成员平均相似度 {sims[worst]:.2f}"
            )
        else:
            reason = (
                f"成员与聚类内其它成员平均相似度 {sims[worst]:.2f} "
                f"< {min_member_similarity:.2f}"
            )
        outliers.append({"track_id": worst, "reason": reason})
        kept.remove(worst)
    return kept, outliers


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


def _assign_unassigned_tracks(
    cluster_of: dict[int, int],
    cluster_first: dict[int, int],
    tracks: list[dict[str, Any]],
    first_frame: dict[int, int],
) -> int | None:
    """Put every track without a face into one shared cluster.

    Tracks that never yielded a face (back of the head, long shot, face
    occluded) have no embedding, so they cannot be clustered by similarity
    and must not each become a Person_N. They all share one cluster id;
    the caller names that cluster ``未分配`` and never gives it a Person_N
    number. Returns the shared cluster id, or None when every track has
    a face. Mutates ``cluster_of`` / ``cluster_first`` in place.
    """
    faceless = [
        int(track["track_id"])
        for track in tracks
        if int(track["track_id"]) not in cluster_of
    ]
    if not faceless:
        return None
    cid = (max(cluster_of.values()) + 1) if cluster_of else 0
    for tid in faceless:
        cluster_of[tid] = cid
    cluster_first[cid] = min(first_frame.get(tid, 10**12) for tid in faceless)
    return cid


def _coerce_costume_hist(value: Any) -> list[float] | None:
    """One tracklet's costume hist: aggregated (list of floats) or raw per-frame list.

    ``run()`` passes the aggregated dict; tests may pass the raw per-frame
    lists instead — both are accepted, empty/missing means None.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value and isinstance(
        value[0], (list, tuple, np.ndarray)
    ):
        return aggregate_costume_hist([list(map(float, h)) for h in value])
    try:
        vec = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return vec or None


def build_tracklets(
    records: Mapping[int, list[dict[str, Any]]],
    samples_meta: Mapping[int, list[dict[str, Any]]],
    means: Mapping[int, np.ndarray],
    cluster_of: Mapping[int, int],
    first_frame: Mapping[int, int],
    costume_hists: Mapping[int, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the in-memory tracklets handed to the stitcher (pure function).

    One tracklet per track id seen in ``records`` / ``samples_meta`` /
    ``means`` / ``first_frame`` / ``costume_hists``. ``embedding`` is the
    quality-weighted mean (an ``np.ndarray`` in memory — the JSON payload
    uses :func:`serialize_tracklet`); a track with no face samples has
    ``embedding`` None and ``unassigned`` True. ``cluster_id`` is the
    clustering suggestion (``_cluster_tracks`` + outlier split), kept
    unchanged — stitching, not clustering, decides the GPID.
    """
    costume_hists = costume_hists or {}
    tids = set(first_frame) | set(records) | set(samples_meta) | set(means)
    tids |= set(costume_hists)
    tracklets: list[dict[str, Any]] = []
    for tid in sorted(int(t) for t in tids):
        recs = list(records.get(tid) or [])
        metas = list(samples_meta.get(tid) or [])
        meta_by_frame = {int(m.get("frame_idx") or 0): m for m in metas}
        face_samples: list[dict[str, Any]] = []
        for rec in recs:
            if rec.get("emb") is None:
                continue
            sample = {
                "emb": np.asarray(rec["emb"], dtype=np.float32),
                "weight": float(rec.get("weight") or 0.0),
                "det_score": float(rec.get("det_score") or 0.0),
                "frame_idx": int(rec.get("frame_idx") or 0),
                "bbox": [float(x) for x in (rec.get("bbox") or [0, 0, 0, 0])],
            }
            meta = meta_by_frame.get(sample["frame_idx"]) or {}
            for key in ("sharpness", "reject_reason"):
                if key in rec or key in meta:
                    # Keep the key even when the value is None: the
                    # serialized tracklet keeps sharpness/reject_reason.
                    sample[key] = rec.get(key, meta.get(key))
            face_samples.append(sample)
        frames = [s["frame_idx"] for s in face_samples]
        frames += [int(m.get("frame_idx") or 0) for m in metas]
        first = int(first_frame.get(tid, frames[0] if frames else 0))
        last = max(frames) if frames else first
        mean = means.get(tid)
        tracklets.append(
            {
                "track_id": tid,
                "first_frame": first,
                "last_frame": last,
                "face_samples": face_samples,
                "embedding": (
                    np.asarray(mean, dtype=np.float32) if mean is not None else None
                ),
                "body_embedding": None,
                "costume_hist": _coerce_costume_hist(costume_hists.get(tid)),
                "cluster_id": cluster_of.get(tid),
                "unassigned": not face_samples,
            }
        )
    return tracklets


def serialize_tracklet(tracklet: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of a tracklet: embeddings become float lists and the
    per-sample ``emb`` arrays are dropped (they would bloat identity.json —
    the track mean already lives in ``track_embeddings``)."""
    samples = []
    for sample in tracklet.get("face_samples") or []:
        samples.append({k: v for k, v in sample.items() if k != "emb"})
    emb = tracklet.get("embedding")
    body = tracklet.get("body_embedding")
    hist = tracklet.get("costume_hist")
    return {
        **tracklet,
        "face_samples": samples,
        "embedding": [float(x) for x in np.asarray(emb).reshape(-1)] if emb is not None else None,
        "body_embedding": [float(x) for x in body] if body is not None else None,
        "costume_hist": [float(x) for x in hist] if hist is not None else None,
    }


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
    # Per-track costume histograms, one per sampled frame (aggregated into
    # one hist per tracklet after the loop, for the stitcher's costume vote).
    costume_hists_raw: dict[int, list[list[float]]] = defaultdict(list)
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
            # Costume hist for every sampled track frame, face or no face:
            # the costume vote is what lets a faceless tracklet attach.
            for det in dets:
                try:
                    hist = costume_hist_from_crop(frame, det["bbox"])
                except Exception as exc:  # noqa: BLE001 - one bad crop skips
                    log.debug("服装直方图失败（track %s）：%s", det["track_id"], exc)
                    hist = None
                if hist is not None:
                    costume_hists_raw[int(det["track_id"])].append(hist)
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
                    "sharpness": _face_sharpness(frame, bbox),
                }
                meta["reject_reason"] = sample_reject_reason(
                    meta, min_sharpness=float(cfg.identity.min_sharpness)
                )
                samples_meta[tid].append(meta)
                records[tid].append({**meta, "emb": emb})
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()

    means: dict[int, np.ndarray] = {}
    mean_det: dict[int, float] = {}
    for tid, samples in records.items():
        mean, used = quality_weighted_embedding(
            samples, min_sharpness=float(cfg.identity.min_sharpness)
        )
        if mean is None:
            continue
        means[tid] = mean
        mean_det[tid] = float(sum(float(s.get("det_score") or 0.0) for s in used) / max(1, len(used)))
        used_frames = {int(s["frame_idx"]) for s in used}
        for meta in samples_meta.get(tid, []):
            meta["used_for_cluster"] = int(meta["frame_idx"]) in used_frames

    # InsightFace is done: sampling + means are finished, stitching does
    # not need it, and releasing now frees VRAM for the CLIP body pass.
    held = app
    app = None
    release_cuda(held)

    costume_hists: dict[int, list[float] | None] = {
        tid: aggregate_costume_hist(hists) for tid, hists in costume_hists_raw.items()
    }

    cluster_of = _cluster_tracks(means, cfg.identity.cluster_threshold)

    # Stable Person_N order: by first appearance of any track in the cluster.
    first_frame: dict[int, int] = {}
    for track in track_payload.get("tracks") or []:
        first_frame[int(track["track_id"])] = int(track.get("first_frame") or 0)
    cluster_first: dict[int, int] = {}
    for tid, cid in cluster_of.items():
        cluster_first[cid] = min(cluster_first.get(cid, 10**12), first_frame.get(tid, 10**12))
    # The "未分配" bucket is assigned AFTER stitching now (B1): only
    # tracklets the stitcher could not attach to any GPID go in the
    # bucket — a faceless tracklet that hangs onto a GPID on body+costume
    # votes is a person, not unassigned. unassigned_cid stays None until
    # the post-stitch call below.
    unassigned_cid: int | None = None
    next_cid = (max(cluster_of.values()) + 1) if cluster_of else 0
    # 聚类内校验：平均链接只保证合并时的组间距离，组内仍可能混进离群成员
    # （截图里一个文件夹并排三张不同人的脸就是这么来的）。离群成员拆出来，
    # 各自成为未确认的碎片聚类——进待审核队列，不直接成人物。
    # 没有 embedding 的成员（没采到脸）无法判断，留在原聚类。
    split_info: dict[int, dict[str, Any]] = {}
    for cid in sorted(set(cluster_of.values())):
        members = [tid for tid, c in cluster_of.items() if c == cid and tid in means]
        if len(members) < 2:
            continue
        _kept, outliers = split_cluster_outliers(
            {tid: means[tid] for tid in members},
            mean_det=mean_det,
            max_mean_distance=float(cfg.identity.cluster_max_mean_distance),
            min_member_similarity=float(cfg.identity.cluster_min_member_similarity),
        )
        if not outliers:
            continue
        for out in outliers:
            frag_cid = next_cid
            next_cid += 1
            cluster_of[out["track_id"]] = frag_cid
            cluster_first[frag_cid] = first_frame.get(out["track_id"], 10**12)
            split_info[frag_cid] = {
                "split_from": cid,
                "reason": out["reason"],
            }
            log.info("聚类 %s 拆出离群成员 track=%s：%s", cid, out["track_id"], out["reason"])
        split_info[cid] = {
            **split_info.get(cid, {}),
            "split_off": [o["track_id"] for o in outliers],
        }
    ordered_clusters = sorted(cluster_first, key=lambda c: (cluster_first[c], c))
    # The "未分配" bucket never takes a Person_N number: real clusters
    # are numbered Person_1..N by first appearance with no gaps, and the
    # bucket person is appended after them below.
    real_clusters = [c for c in ordered_clusters if c != unassigned_cid]
    cluster_name = {cid: f"Person_{i}" for i, cid in enumerate(real_clusters, start=1)}

    roster = dict(roster or {})
    track_names: dict[int, str] = {}
    persons: list[dict[str, Any]] = []
    for cid in real_clusters:
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
        person = {
            "name": name,
            "auto_name": auto_name,
            "cluster_id": cid,
            "track_ids": member_ids,
            "face_samples": sum(len(samples_meta.get(tid, [])) for tid in member_ids),
            "rejected_face_samples": sum(
                1
                for tid in member_ids
                for meta in samples_meta.get(tid, [])
                if meta.get("reject_reason")
            ),
            "first_frame": cluster_first[cid],
            "embedding": embedding,
            "cluster_distance": distance,
            "confidence": confidence,
        }
        info = split_info.get(cid) or {}
        if "split_from" in info:
            person["split_from"] = cluster_name[info["split_from"]]
            person["split_reason"] = info["reason"]
        if info.get("split_off"):
            person["split_off_track_ids"] = info["split_off"]
        persons.append(person)

    # ---- B1 跟踪优先：把 tracklet 拼成 GPID（聚类只是合并建议）----
    tracklets = build_tracklets(
        records, samples_meta, means, cluster_of, first_frame, costume_hists
    )
    try:
        attached = attach_body_embeddings(video, tracklets, track_payload)
        log.info("身体外观：%s 条 tracklet 拿到 CLIP embedding。", attached)
    except Exception as exc:  # noqa: BLE001 - best effort, body vote abstains
        log.info("身体外观失败，身体票弃权：%s", exc)
    stitch_result = stitch_tracklets(tracklets, cfg.stitch)
    # Unassigned tracklets are singletons in the stitch groups (they joined
    # no merge edge); they must not become GPID persons — the caller builds
    # the shared "未分配" bucket from them below.
    unassigned_ids = set(stitch_result.get("unassigned") or [])
    # A group with no face sample at all has no identity to attach to:
    # its tracklets join the shared "未分配" bucket too (build_global_persons
    # already skips such groups, so this only feeds the bucket below and
    # the stale-pending filter).
    _by_id = {int(t.get("track_id")): t for t in tracklets}
    for group in stitch_result.get("groups") or []:
        if not any((_by_id.get(int(tid)) or {}).get("face_samples") for tid in group):
            unassigned_ids.update(int(tid) for tid in group)
    global_persons = build_global_persons(tracklets, stitch_result)
    # Roster names apply to GPID persons the same way they applied to
    # cluster persons: a roster hit on any member track id renames the
    # GPID (auto_name stays Person_N, so track["name"] keeps the roster
    # name instead of falling back to Person_N).
    for person in global_persons:
        auto_name = person.get("auto_name") or person.get("name") or ""
        name = auto_name
        for tid in person.get("track_ids") or []:
            name = apply_names(auto_name, int(tid), roster)
            if name != auto_name:
                break
        person["name"] = name
    # A faceless_attach pending entry whose tracklet ended up in the
    # bucket (its "GPID" turned out to be faceless-only) is stale —
    # the attach never happened, so it is not a decision to make.
    pending_pairs: list[dict[str, Any]] = []
    for pair in stitch_result.get("pending") or []:
        aid = pair.get("a_track_id")
        if (
            pair.get("kind") == "faceless_attach"
            and aid is not None
            and int(aid) in unassigned_ids
        ):
            continue
        pending_pairs.append(pair)
    log.info(
        "拼接完成：%s 个 GPID，%s 对待裁决，%s 条未分配。",
        len(global_persons),
        len(pending_pairs),
        len(unassigned_ids),
    )

    # Only the stitcher's unassigned tracklets enter the bucket — faceless
    # tracks that attached to a GPID stay persons (see _assign_unassigned_tracks).
    if unassigned_ids:
        bucket_tracks = [
            t
            for t in (track_payload.get("tracks") or [])
            if int(t["track_id"]) in unassigned_ids
        ]
        unassigned_cid = _assign_unassigned_tracks(
            cluster_of, cluster_first, bucket_tracks, first_frame
        )

    if unassigned_cid is not None:
        # One shared bucket for all faceless tracks, kept in identity.json
        # for traceability. No embedding, no face samples, no Person_N
        # number, and no roster rename — it is not a person.
        member_ids = [tid for tid, c in cluster_of.items() if c == unassigned_cid]
        for tid in member_ids:
            track_names[tid] = UNASSIGNED_NAME
        persons.append(
            {
                "name": UNASSIGNED_NAME,
                "auto_name": UNASSIGNED_NAME,
                "cluster_id": unassigned_cid,
                "track_ids": member_ids,
                "face_samples": sum(len(samples_meta.get(tid, [])) for tid in member_ids),
                "rejected_face_samples": 0,
                "first_frame": cluster_first[unassigned_cid],
                "embedding": None,
                "cluster_distance": 0.0,
                "confidence": 0.0,
                "unassigned": True,
            }
        )

    # Track name follows the GPID first (tracking-first identity), then
    # the "未分配" bucket, then the cluster suggestion as a fallback.
    # cluster_id stays the cluster id — the suggestion, not the GPID.
    gpid_name = {
        int(tid): person.get("name") or person.get("auto_name")
        for person in global_persons
        for tid in person.get("track_ids") or []
    }
    for track in track_payload.get("tracks") or []:
        tid = int(track["track_id"])
        if tid in gpid_name:
            track["name"] = gpid_name[tid]
        elif tid in (unassigned_ids or set()):
            track["name"] = UNASSIGNED_NAME
        else:
            track["name"] = track_names.get(tid, f"Person_{tid}")
        track_names[tid] = track["name"]
        track["cluster_id"] = cluster_of.get(tid)
        track["face_samples"] = samples_meta.get(tid, [])

    # 被挡下的样本：原因随样本走，供导出裁剪和待审核队列使用。
    rejected_samples: list[dict[str, Any]] = []
    for tid, metas in samples_meta.items():
        for meta in metas:
            if not meta.get("reject_reason"):
                continue
            rejected_samples.append(
                {
                    "track_id": tid,
                    "frame_idx": int(meta.get("frame_idx") or 0),
                    "reason": meta["reject_reason"],
                    "sharpness": meta.get("sharpness"),
                    "area": float(meta.get("area") or 0.0),
                    "det_score": float(meta.get("det_score") or 0.0),
                    "frontal": meta.get("frontal"),
                }
            )
    rejected_samples.sort(key=lambda r: (r["track_id"], r["frame_idx"]))

    payload = {
        "model": cfg.identity.model,
        "cluster_threshold": cfg.identity.cluster_threshold,
        "cluster_max_mean_distance": cfg.identity.cluster_max_mean_distance,
        "cluster_min_member_similarity": cfg.identity.cluster_min_member_similarity,
        "min_similarity": float(cfg.replace.min_similarity),
        "persons": persons,
        "tracklets": [serialize_tracklet(t) for t in tracklets],
        "global_persons": global_persons,
        "pending_pairs": pending_pairs,
        "rejected_samples": rejected_samples,
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

    return payload
