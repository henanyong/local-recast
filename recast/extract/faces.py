"""Build a local character library from InsightFace clusters + face crops."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from recast.config import IdentityConfig, RecastConfig
from recast.core.merge import (
    find_merge_target,
    load_existing_embeddings,
    merge_embeddings,
    parse_embedding,
)
from recast.ffmpeg import VideoInfo
from recast.media import open_capture
from recast.runtime import check_cancel
from recast.stages.identity import MIN_SHARPNESS, sample_reject_reason
from recast.store.db import Store

log = logging.getLogger("recast")

# How many held-back crops per person are exported for manual review.
# They are not reference stills — the filename carries the reason
# (rejected_模糊_...) and rejected.json carries the structured record.
REJECTED_CROPS_PER_PERSON = 4


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
    """Higher is a better reference still: clear, large, and facing the camera.

    The sharpness term uses the same scale as review.py's file score
    (``0.4 + 0.6 * min(1, laplacian / 150)``). A sample whose sharpness was
    never measured keeps a neutral factor of 1.0; a measured blurry sample
    scores low here and is excluded outright by select_reference_crops.
    """
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
    sharpness = sample.get("sharpness")
    if sharpness is None:
        sharp_term = 1.0
    else:
        sharp_term = 0.4 + 0.6 * min(1.0, max(0.0, float(sharpness)) / 150.0)
    return det * (0.45 + 0.55 * size_term) * (0.4 + 0.6 * front) * sharp_term


def rejected_samples(
    samples: list[dict[str, Any]],
    *,
    min_sharpness: float = MIN_SHARPNESS,
) -> list[dict[str, Any]]:
    """Samples held back from the cluster core, each with its reason.

    The reason comes from :func:`recast.stages.identity.sample_reject_reason`
    ("模糊" / "太小" / "侧脸" / "检测分过低"), recomputed from the sample
    dict so it also works on payloads that predate the recorded reason.
    """
    out: list[dict[str, Any]] = []
    for sample in samples:
        reason = sample.get("reject_reason") or sample_reject_reason(
            sample, min_sharpness=min_sharpness
        )
        if reason:
            out.append({**sample, "reject_reason": reason})
    return out


def select_reference_crops(
    samples: list[dict[str, Any]],
    *,
    limit: int = 8,
    min_frame_gap: int = 8,
    min_sharpness: float = MIN_SHARPNESS,
) -> list[dict[str, Any]]:
    """Top samples by quality, spread across time instead of the first frames.

    Held-back samples (blurry, tiny, profile, low detection score) are
    excluded here — see :func:`rejected_samples` for where they go instead.
    """
    if limit <= 0:
        return []
    usable = [
        s
        for s in samples
        if not sample_reject_reason(s, min_sharpness=min_sharpness)
    ]
    ranked = sorted(usable, key=score_reference_sample, reverse=True)
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
    rejected_out: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, list[str]]:
    """Write the best face crops under workdir/face_samples/<auto_name>/.

    Grouping is per ``ident["persons"]`` entry, so B1 GPID persons work
    unchanged: a GPID whose ``track_ids`` span several old clusters
    still exports into a single folder named by its own ``auto_name``.

    Held-back samples are exported too (up to REJECTED_CROPS_PER_PERSON
    per person, filename ``rejected_<原因>_...``) and listed in the
    returned ``saved`` mapping, so they flow into the store and the
    character folder. They do NOT enter the review queue by default
    (B0): ``build_review_queue`` skips held-back samples unless
    ``include_rejected=True`` (library window checkbox "显示低质量样本",
    default off). The structured record (path + reason) is
    written to ``face_samples/rejected.json`` and, when ``rejected_out``
    is given, filled into it as ``{auto_name: [{path, reason, frame_idx}]}``.
    """
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
        bad = sorted(
            rejected_samples(candidates),
            key=score_reference_sample,
            reverse=True,
        )[:REJECTED_CROPS_PER_PERSON]
        for sample in bad:
            wanted.setdefault(int(sample["frame_idx"]), []).append(
                {
                    "tid": int(sample["tid"]),
                    "bbox": sample["bbox"],
                    "person": sample["person"],
                    "rejected_reason": sample["reject_reason"],
                }
            )

    dest_root = workdir / "face_samples"
    dest_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    saved: dict[str, list[str]] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}
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
                    reason = item.get("rejected_reason")
                    if reason:
                        n = rejected_counts.get(auto, 0)
                        if n >= REJECTED_CROPS_PER_PERSON:
                            continue
                        dest = dest_root / auto / f"rejected_{reason}_{n:02d}_f{frame_idx}.jpg"
                    else:
                        n = counts.get(auto, 0)
                        if n >= max_per_person:
                            continue
                        dest = dest_root / auto / f"{n:02d}_f{frame_idx}.jpg"
                    path = _save_crop(frame, item["bbox"], dest)
                    if path is not None:
                        if reason:
                            rejected_counts[auto] = n + 1
                            rejected.setdefault(auto, []).append(
                                {
                                    "path": str(path),
                                    "reason": reason,
                                    "frame_idx": frame_idx,
                                }
                            )
                        else:
                            counts[auto] = n + 1
                        saved.setdefault(auto, []).append(str(path))
            frame_idx += 1
    finally:
        cap.release()

    index = workdir / "face_samples" / "index.json"
    index.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    if rejected:
        rejected_path = workdir / "face_samples" / "rejected.json"
        rejected_path.write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
        if rejected_out is not None:
            rejected_out.update(rejected)
        log.info("已导出被挡下的人脸样本: %s 个角色", len(rejected))
    log.info("已导出人脸样本: %s 个角色", len(saved))
    return saved


def ingest_into_library(
    store: Store | None,
    ident: dict[str, Any],
    samples: dict[str, list[str]],
    *,
    task_id: str | None = None,
    drama_key: str | None = None,
    merge_similarity: float | None = None,
    output_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Write one episode's persons into the character store.

    Cross-episode merge runs first: a new cluster whose embedding matches
    an existing character at or above ``merge_similarity`` (default
    ``IdentityConfig.merge_similarity``) is merged into that character —
    its samples are added to the existing row, no new folder/row is
    created, and the stored embedding is updated with the weighted mean.
    Only store rows can be merged into; a match that exists only as a
    character folder (no store row) falls through to creating a new row.
    No embedding (no face sampled) also falls through to the old path.
    The one exception is the shared "未分配" bucket (``unassigned: True``):
    it is skipped entirely — no store row, no folder, no embedding.

    The stored embedding is only (re)written when the ingest actually
    added new sample paths: re-running the same episode adds nothing and
    changes nothing, so the operation is idempotent.
    """
    if store is None:
        return list(ident.get("persons") or [])
    if merge_similarity is None:
        merge_similarity = IdentityConfig().merge_similarity
    candidates = load_existing_embeddings(store, output_root)
    rejected_map: dict[str, list[dict[str, Any]]] = ident.get("_rejected_face_samples") or {}
    out: list[dict[str, Any]] = []
    for person in ident.get("persons") or []:
        if person.get("unassigned"):
            # The shared "未分配" bucket is not a person: no store row,
            # no character folder, no embedding. It only exists in
            # identity.json for traceability of the faceless tracks.
            log.info(
                "跳过「未分配」桶（轨迹 %s），不建人物库记录。",
                person.get("track_ids") or [],
            )
            continue
        auto = person.get("auto_name") or person.get("name") or "Person"
        cluster_id = person.get("cluster_id")
        key = f"{drama_key or 'global'}:{cluster_id if cluster_id is not None else auto}"
        paths = samples.get(auto) or []
        embedding = person.get("embedding")

        target, best_sim = (None, 0.0)
        if embedding is not None:
            target, best_sim = find_merge_target(embedding, candidates, float(merge_similarity))
        if target is not None and not target.get("character_id"):
            log.info(
                "人物「%s」与已有人物文件夹「%s」相似度 %.2f，但库里没有对应行，先建新人物",
                auto,
                target["name"],
                best_sim,
            )
            target = None
        if target is not None and not target.get("cluster_key"):
            log.info(
                "人物「%s」与已有人物「%s」相似度 %.2f，但该行没有 cluster_key，先建新人物",
                auto,
                target["name"],
                best_sim,
            )
            target = None

        # B1: extra per-person features for the three-vote merge. They are
        # persisted to the folder meta, never used for the merge decision
        # here (find_merge_target above stays face-only).
        extra_features = {
            key: person[key]
            for key in ("body_embedding", "costume_hist")
            if person.get(key) is not None
        }
        if target is not None:
            # Merge into the existing character row.
            existing = store.find_character_by_cluster(target["cluster_key"])
            before = {
                a.get("path")
                for a in (existing or {}).get("assets") or []
                if a.get("kind") == "face"
            }
            row = store.upsert_character_candidate(
                cluster_key=target["cluster_key"],
                auto_name=target["name"],
                sample_paths=paths,
                source_task_id=task_id,
            )
            new_paths = [p for p in paths if p and p not in before]
            merged = merge_embeddings(
                target["embedding"],
                embedding,
                old_count=max(1, len(before)),
                new_count=max(1, len(new_paths)),
            )
            wrote_features = False
            if new_paths and merged is not None:
                store.update("characters", row["id"], face_embedding=merged)
                row = store.get_character(row["id"]) or row
                # Later persons in this same batch compare against the
                # updated embedding, not the stale one.
                target["embedding"] = parse_embedding(merged)
                _write_folder_embedding(
                    output_root, target["name"], merged, **extra_features
                )
                wrote_features = bool(extra_features)
            if extra_features and not wrote_features:
                # No face-embedding write happened (e.g. idempotent re-run
                # with no new samples, or a faceless GPID) — the features
                # still belong in the folder meta.
                _write_folder_embedding(
                    output_root, target["name"], None, **extra_features
                )
            person["merged_into"] = target["name"]
            person["merged_character_id"] = target["character_id"]
            person["merge_similarity"] = round(best_sim, 3)
            row = dict(row)
            row["merged_into"] = target["name"]
            row["merge_similarity"] = round(best_sim, 3)
            log.info(
                "人物「%s」并入已有人物「%s」（相似度 %.2f），新增样本 %s 个",
                auto,
                target["name"],
                best_sim,
                len(new_paths),
            )
        else:
            existing = store.find_character_by_cluster(key)
            before = {
                a.get("path")
                for a in (existing or {}).get("assets") or []
                if a.get("kind") == "face"
            }
            row = store.upsert_character_candidate(
                cluster_key=key,
                auto_name=auto,
                sample_paths=paths,
                source_task_id=task_id,
            )
            if person.get("name") and person["name"] != auto and not row.get("confirmed"):
                store.update("characters", row["id"], name=person["name"], display_name=person["name"])
                row = store.get_character(row["id"]) or row
            new_paths = [p for p in paths if p and p not in before]
            wrote_features = False
            if new_paths and embedding is not None:
                old_emb = row.get("face_embedding")
                parsed_new = parse_embedding(embedding)
                if old_emb and parsed_new is not None:
                    merged = merge_embeddings(
                        old_emb,
                        embedding,
                        old_count=max(1, len(before)),
                        new_count=len(new_paths),
                    )
                elif parsed_new is not None:
                    merged = [float(x) for x in parsed_new]
                else:
                    merged = None
                if merged is not None:
                    store.update("characters", row["id"], face_embedding=merged)
                    row = store.get_character(row["id"]) or row
                    # A later episode can now match this character.
                    candidates.append(
                        {
                            "character_id": row["id"],
                            "name": row.get("display_name") or row.get("name") or auto,
                            "embedding": parse_embedding(merged),
                            "source": "store",
                            "cluster_key": key,
                        }
                    )
                    _write_folder_embedding(
                        output_root,
                        row.get("display_name") or auto,
                        merged,
                        **extra_features,
                    )
                    wrote_features = bool(extra_features)
            if extra_features and not wrote_features:
                _write_folder_embedding(
                    output_root,
                    row.get("display_name") or auto,
                    None,
                    **extra_features,
                )
        _mark_rejected_assets(store, row, rejected_map.get(auto) or [])
        row = store.get_character(row["id"]) or row
        out.append(row)
    return out


def _write_folder_embedding(
    output_root: str | Path | None,
    name: str,
    embedding: list[float] | None,
    *,
    body_embedding: list[float] | None = None,
    costume_hist: list[float] | None = None,
) -> None:
    """Mirror the stored embedding into the character folder's meta.json.

    Only when the caller passes ``output_root``. When there is something
    to write, the folder is created (idempotent) — the import pipeline
    never creates character folders before ingest, so "only when the
    folder already exists" would silently drop every feature on a fresh
    import. Failures are logged, not raised — the store row is the source
    of truth.

    B1: ``body_embedding`` (CLIP body appearance) and ``costume_hist``
    (HSV costume histogram) are persisted under the same key names when
    given, so the three-vote cross-episode merge can read them back from
    the folder later. Passing ``embedding=None`` with only the extra
    features writes just those features (a faceless GPID attached on
    body+costume votes only has no face embedding to mirror).
    """
    if output_root is None:
        return
    if embedding is None and body_embedding is None and costume_hist is None:
        return
    try:
        from recast.core.characters import (
            ensure_character_folder,
            load_meta,
            save_meta,
        )

        folder, _meta = ensure_character_folder(output_root, name)
        meta = load_meta(folder)
        if meta is None:
            return
        if embedding is not None:
            meta["face_embedding"] = [float(x) for x in embedding]
        if body_embedding is not None:
            meta["body_embedding"] = [float(x) for x in body_embedding]
        if costume_hist is not None:
            meta["costume_hist"] = [float(x) for x in costume_hist]
        save_meta(folder, meta)
    except Exception as exc:  # folder/meta problems must not break ingest
        log.warning("写人物文件夹 embedding 失败（%s）：%s", name, exc)


def _mark_rejected_assets(
    store: Store, row: dict[str, Any], rejected: list[dict[str, Any]]
) -> None:
    """Tag held-back crops on their asset rows with the reason.

    The review queue's file score only measures size + sharpness, so a
    held-back profile / low-det crop would otherwise look like an
    ordinary sample. The note ("待审核：模糊" etc.) carries the reason
    where the queue UI can show it; the filename and rejected.json carry
    it too. Rows already marked are left alone.
    """
    if not rejected:
        return
    by_path = {item.get("path"): item.get("reason") for item in rejected if item.get("path")}
    if not by_path:
        return
    fresh = store.get_character(row["id"]) or row
    for asset in fresh.get("assets") or []:
        reason = by_path.get(asset.get("path"))
        if not reason or asset.get("kind") != "face":
            continue
        if "待审核" in str(asset.get("notes") or ""):
            continue
        store.update("character_assets", asset["id"], notes=f"待审核：{reason}")


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
    # B1: identity.json v2 carries cross-shot global persons (GPID).
    # Export + ingest group by GPID, so tracks that were split across
    # old clusters land in one Person_N folder / store row. An old
    # payload without "global_persons" falls back to "persons" unchanged.
    gpid_persons = ident.get("global_persons")
    if gpid_persons:
        gpid_ident = {**ident, "persons": list(gpid_persons)}
    else:
        gpid_ident = ident
    rejected: dict[str, list[dict[str, Any]]] = {}
    samples = export_face_crops(video, workdir, gpid_ident, track_payload, rejected_out=rejected)
    gpid_ident["_rejected_face_samples"] = rejected
    ident["_rejected_face_samples"] = rejected
    from recast.core.merge import resolve_output_root

    ingest_into_library(
        store,
        gpid_ident,
        samples,
        task_id=task_id,
        drama_key=drama_key,
        merge_similarity=float(cfg.identity.merge_similarity),
        # B1: features only land in folder meta.json when the ingest
        # knows the output root (the store path derives it); without
        # this, body/costume features never persist and the three-vote
        # cross-episode merge can never read them back.
        output_root=resolve_output_root(store),
    )
    ident["_face_samples"] = samples
    return ident
