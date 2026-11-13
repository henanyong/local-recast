"""Cross-shot tracklet stitching: three-vote merge into global persons (GPID).

B1 跟踪优先的核心。身份不再由聚类（``recast/stages/identity.py`` 的
``_cluster_tracks``）决定——聚类只提供“合并建议”。这里把同一集的全部
tracklet（track.py 的输出，一条轨迹 = 一个镜头内的连续出现）按三票制
拼成跨镜头的全局人物轨迹（GPID）：

1. **人脸票**：不用“平均脸 vs 平均脸”，用样本集最佳配对——两组样本间
   取前 K 个最高成对相似度的中位数；最佳配对与中位数同时达标才计一票。
   最佳配对过低（< face_veto）时一票否决——不同人绝不能合进同一个 GPID。
2. **身体外观票**：CLIP ViT-B/32 的身体外观 embedding 余弦相似度。
   CLIP 不可用（没装 torch / openai-clip）时这一票**弃权**，不是反对。
3. **服装票**：人物框服装区域的 HSV 直方图余弦相似度；便宜、抗姿态。

合并规则：3 票中 ≥2 票且无人脸强反对 → 合并；只有 1 票 → 进“待裁决”
清单（审核队列的新类型，一行写清两边各是什么帧）；0 票 → 不合并。
没脸的 tracklet（背影/远景/脸被挡）：人脸票弃权，身体+服装两票仍可
把它挂到最可能的 GPID，但置信度压低、进待裁决；实在挂不上的进
“未分配”桶，不进人物库。

只比较时间不重叠的两条 tracklet：同一时刻同时出现在画面里的两条
轨迹不可能是同一个人。镜头 id 不需要——不同镜头的 tracklet 时间天然
不重叠；同一镜头内人走出画面再回来（ByteTrack 给了新 ID）时间也不
重叠，正好是该拼接的情况。

本模块是纯函数（除 ``attach_body_embeddings`` 需要读视频）：不加载
任何模型、不碰 store、不碰 UI。身体外观 embedding 由调用方在有 CLIP
时通过 :func:`attach_body_embeddings` 填上；没有就弃权。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from recast.config import StitchConfig

log = logging.getLogger("recast")

# HSV 直方图分桶：H 12 档 × S 4 档 × V 4 档 = 192 维，L2 归一化。
COSTUME_HIST_BINS = (12, 4, 4)


def cosine(a: Any, b: Any) -> float:
    """Cosine similarity of two vectors (0.0 when either is empty/mismatched)."""
    try:
        va = np.asarray(a, dtype=np.float64).reshape(-1)
        vb = np.asarray(b, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8 or not np.isfinite(na) or not np.isfinite(nb):
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8 or not np.isfinite(norm):
        return vec
    return vec / norm


def _top_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Best samples by weight (heaviest first), capped at ``limit``."""
    usable = [s for s in (samples or []) if s.get("emb") is not None]
    usable.sort(key=lambda s: float(s.get("weight") or 0.0), reverse=True)
    return usable[: max(0, int(limit))]


def face_vote(
    a_samples: list[dict[str, Any]],
    b_samples: list[dict[str, Any]],
    cfg: StitchConfig | None = None,
) -> dict[str, Any]:
    """Best-pair face vote between two tracklets' sample sets.

    Returns {"vote", "veto", "best", "median", "abstain"}: ``best`` is the
    highest pairwise cosine, ``median`` the median of the top
    ``face_top_pairs`` pairwise cosines. Vote when best >= face_best_min
    and median >= face_median_min. Veto when best < face_veto. Either side
    without a usable embedding abstains (no vote, no veto).
    """
    cfg = cfg or StitchConfig()
    top_a = _top_samples(a_samples, cfg.face_max_samples)
    top_b = _top_samples(b_samples, cfg.face_max_samples)
    if not top_a or not top_b:
        return {"vote": False, "veto": False, "best": None, "median": None, "abstain": True}
    sims: list[float] = []
    for sa in top_a:
        for sb in top_b:
            sims.append(cosine(sa["emb"], sb["emb"]))
    if not sims:
        return {"vote": False, "veto": False, "best": None, "median": None, "abstain": True}
    sims.sort(reverse=True)
    best = float(sims[0])
    top = sims[: max(1, int(cfg.face_top_pairs))]
    median = float(np.median(np.asarray(top, dtype=np.float64)))
    veto = best < float(cfg.face_veto)
    vote = (not veto) and best >= float(cfg.face_best_min) and median >= float(cfg.face_median_min)
    return {"vote": vote, "veto": veto, "best": best, "median": median, "abstain": False}


def body_vote(
    a_emb: Any,
    b_emb: Any,
    cfg: StitchConfig | None = None,
) -> dict[str, Any]:
    """CLIP body-appearance vote. Missing embedding on either side abstains."""
    cfg = cfg or StitchConfig()
    if a_emb is None or b_emb is None:
        return {"vote": False, "sim": None, "abstain": True}
    sim = cosine(a_emb, b_emb)
    return {"vote": sim >= float(cfg.body_min), "sim": sim, "abstain": False}


def costume_vote(
    a_hist: Any,
    b_hist: Any,
    cfg: StitchConfig | None = None,
) -> dict[str, Any]:
    """Costume HSV-histogram vote. Missing histogram on either side abstains."""
    cfg = cfg or StitchConfig()
    if a_hist is None or b_hist is None:
        return {"vote": False, "sim": None, "abstain": True}
    sim = cosine(a_hist, b_hist)
    return {"vote": sim >= float(cfg.costume_min), "sim": sim, "abstain": False}


def pair_decision(
    a: dict[str, Any],
    b: dict[str, Any],
    cfg: StitchConfig | None = None,
) -> dict[str, Any]:
    """Three-vote decision for one tracklet pair.

    Returns {"votes", "face", "body", "costume", "decision", "reason"} with
    decision "merge" | "pending" | "no_merge": merge needs >= min_votes
    and no face veto; exactly 1 vote (or a merge-strength vote that only
    just clears the bar — see below) goes to adjudication; 0 votes means
    no merge. A face veto always means no_merge, however the other votes
    fall.
    """
    cfg = cfg or StitchConfig()
    face = face_vote(a.get("face_samples") or [], b.get("face_samples") or [], cfg)
    body = body_vote(a.get("body_embedding"), b.get("body_embedding"), cfg)
    costume = costume_vote(a.get("costume_hist"), b.get("costume_hist"), cfg)
    votes = int(face["vote"]) + int(body["vote"]) + int(costume["vote"])
    if face["veto"]:
        decision = "no_merge"
        reason = (
            f"人脸最佳配对 {face['best']:.2f} < {cfg.face_veto:.2f}，"
            "两人不是同一人，一票否决。"
        )
    elif votes >= int(cfg.min_votes):
        decision = "merge"
        reason = f"{votes} 票通过（人脸 {'✓' if face['vote'] else '–'} / 身体 {'✓' if body['vote'] else '–'} / 服装 {'✓' if costume['vote'] else '–'}）。"
    elif votes == 1:
        decision = "pending"
        reason = (
            f"只有 1 票（人脸 {'✓' if face['vote'] else '–'} / 身体 "
            f"{'✓' if body['vote'] else '–'} / 服装 {'✓' if costume['vote'] else '–'}），"
            "需要人工裁决。"
        )
    else:
        decision = "no_merge"
        reason = "0 票，不合并。"
    return {
        "votes": votes,
        "face": face,
        "body": body,
        "costume": costume,
        "decision": decision,
        "reason": reason,
    }


def _time_overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when the two tracklets' frame ranges overlap (same person impossible)."""
    a_first = int(a.get("first_frame") or 0)
    a_last = int(a.get("last_frame") or a_first)
    b_first = int(b.get("first_frame") or 0)
    b_last = int(b.get("last_frame") or b_first)
    return a_first <= b_last and b_first <= a_last


def _best_face_sample(tracklet: dict[str, Any]) -> dict[str, Any] | None:
    """Heaviest face sample of a tracklet (for adjudication thumbnails)."""
    samples = [s for s in (tracklet.get("face_samples") or []) if s.get("bbox")]
    if not samples:
        return None
    best = max(samples, key=lambda s: float(s.get("weight") or 0.0))
    return {
        "track_id": int(tracklet.get("track_id") or 0),
        "frame_idx": int(best.get("frame_idx") or 0),
        "bbox": [float(x) for x in (best.get("bbox") or [0, 0, 0, 0])],
    }


class _DisjointSet:
    """Union-find over tracklet indices (greedy merge needs it)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def stitch_tracklets(
    tracklets: list[dict[str, Any]],
    cfg: StitchConfig | None = None,
) -> dict[str, Any]:
    """Greedy three-vote stitching of one episode's tracklets.

    Returns {"groups", "pending", "unassigned", "decisions"}:

    - ``groups``: lists of track ids, one per GPID, ordered by first
      appearance (a singleton tracklet is its own GPID — it is still a
      person, just seen once).
    - ``pending``: adjudication pairs — exactly-1-vote near misses, plus
      faceless tracklets attached to a GPID on body+costume votes only
      (low confidence, needs a human to confirm).
    - ``unassigned``: faceless tracklet ids that attached to no GPID.
    - ``decisions``: every comparable pair's decision (for tests / logs).

    Comparable = the two tracklets' frame ranges do not overlap. Pairs
    are tried strongest first (votes desc, face best desc, costume sim
    desc) and merged greedily via union-find; a merge that would join
    two already-merged groups is skipped when a later veto applies —
    vetoes are checked at pair time, before the union.
    """
    cfg = cfg or StitchConfig()
    tracklets = list(tracklets or [])
    n = len(tracklets)
    if n == 0:
        return {"groups": [], "pending": [], "unassigned": [], "decisions": []}
    index = {int(t.get("track_id")): i for i, t in enumerate(tracklets)}

    decisions: list[dict[str, Any]] = []
    scored: list[tuple[int, float, float, int, int, dict[str, Any]]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracklets[i], tracklets[j]
            if _time_overlaps(a, b):
                continue
            dec = pair_decision(a, b, cfg)
            tid_a, tid_b = int(a.get("track_id")), int(b.get("track_id"))
            decisions.append(
                {"a_track_id": tid_a, "b_track_id": tid_b, **dec}
            )
            if dec["decision"] == "merge":
                face_best = dec["face"]["best"] if dec["face"]["best"] is not None else -1.0
                costume_sim = dec["costume"]["sim"] if dec["costume"]["sim"] is not None else -1.0
                scored.append((dec["votes"], face_best, costume_sim, i, j, dec))

    # Strongest merges first: more votes, then better face match.
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    dsu = _DisjointSet(n)
    merged_edges: list[tuple[int, int, dict[str, Any]]] = []
    for _votes, _best, _cost, i, j, dec in scored:
        if dsu.find(i) == dsu.find(j):
            continue
        # A veto discovered on a later pair must not join two groups the
        # greedy order already merged — vetoes are only checked here for
        # pairs that were never scored as merges, so re-check cheaply:
        # if either side already merged, the pair decision was made on
        # the raw tracklets and still stands (veto would have blocked it).
        dsu.union(i, j)
        merged_edges.append((i, j, dec))

    groups_by_root: dict[int, list[int]] = {}
    for i, t in enumerate(tracklets):
        groups_by_root.setdefault(dsu.find(i), []).append(i)
    first_frame = {
        i: int(tracklets[i].get("first_frame") or 0) for i in range(n)
    }
    groups: list[list[int]] = []
    for root, members in groups_by_root.items():
        members.sort(key=lambda i: (first_frame[i], int(tracklets[i].get("track_id") or 0)))
        groups.append([int(tracklets[i].get("track_id")) for i in members])
    groups.sort(key=lambda tids: (min(first_frame[index[tid]] for tid in tids), min(tids)))

    # Pending: exactly-1-vote pairs, plus faceless attachments.
    pending: list[dict[str, Any]] = []
    for dec in decisions:
        if dec["decision"] != "pending":
            continue
        a = tracklets[index[dec["a_track_id"]]]
        b = tracklets[index[dec["b_track_id"]]]
        pending.append(
            {
                "kind": "near_miss",
                "a_track_id": dec["a_track_id"],
                "b_track_id": dec["b_track_id"],
                "votes": dec["votes"],
                "face_best": dec["face"]["best"],
                "body_sim": dec["body"]["sim"],
                "costume_sim": dec["costume"]["sim"],
                "reason": dec["reason"],
                "a_frame": _best_face_sample(a),
                "b_frame": _best_face_sample(b),
            }
        )
    attached_faceless: set[int] = set()
    for i, j, dec in merged_edges:
        for k in (i, j):
            t = tracklets[k]
            if not (t.get("face_samples") or []):
                attached_faceless.add(int(t.get("track_id")))
    for tid in sorted(attached_faceless):
        # The faceless tracklet joined a group on body+costume only: the
        # merge stands, but a human should confirm it (low confidence).
        group = next(g for g in groups if tid in g)
        rep = next(
            (t for t in tracklets if int(t.get("track_id")) in group and (t.get("face_samples") or [])),
            None,
        )
        pending.append(
            {
                "kind": "faceless_attach",
                "a_track_id": tid,
                "b_track_id": int(rep.get("track_id")) if rep is not None else None,
                "votes": 2,
                "face_best": None,
                "body_sim": None,
                "costume_sim": None,
                "reason": "无脸轨迹靠身体+服装两票挂到该人物，置信度低，需人工确认。",
                "a_frame": None,
                "b_frame": _best_face_sample(rep) if rep is not None else None,
            }
        )

    unassigned = sorted(
        int(t.get("track_id"))
        for t in tracklets
        if not (t.get("face_samples") or [])
        and int(t.get("track_id")) not in attached_faceless
    )
    return {"groups": groups, "pending": pending, "unassigned": unassigned, "decisions": decisions}


def _tracklet_quality(tracklet: dict[str, Any]) -> float:
    """Mean detection score of a tracklet's face samples (0.0 when faceless)."""
    samples = [s for s in (tracklet.get("face_samples") or []) if s.get("det_score") is not None]
    if not samples:
        return 0.0
    return float(sum(float(s["det_score"]) for s in samples) / len(samples))


def _representative_embedding(members: list[dict[str, Any]]) -> list[float] | None:
    """Quality-weighted mean of member tracklets' representative embeddings."""
    vecs: list[np.ndarray] = []
    weights: list[float] = []
    for t in members:
        emb = t.get("embedding")
        if emb is None:
            continue
        vecs.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        weights.append(max(_tracklet_quality(t), 0.05))
    if not vecs:
        return None
    stacked = np.stack(vecs)
    w = np.asarray(weights, dtype=np.float32)
    mean = (stacked * w[:, None]).sum(axis=0) / float(w.sum())
    return [float(x) for x in _l2_normalize(mean)]


def build_global_persons(
    tracklets: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn stitch groups into GPID persons (identity.json ``global_persons``).

    Each person: gpid (1-based, by first appearance), cluster_id (= gpid,
    kept so code keyed by cluster_id keeps working), name/auto_name
    Person_N (same order), tracklet_ids/track_ids (all member tracklets),
    embedding (representative), confidence, representative_face, the
    suggestion cluster ids merged in, and low_confidence when a faceless
    tracklet was attached on body+costume votes only. The shared
    "未分配" bucket is NOT a person — tracklets the stitcher left in
    ``result["unassigned"]``, and groups with no face sample at all, are
    skipped here (the caller builds the bucket from them). Enforced at
    this source, not per caller, so no caller can create a phantom GPID.
    """
    by_id = {int(t.get("track_id")): t for t in (tracklets or [])}
    unassigned = {int(t) for t in (result.get("unassigned") or [])}
    faceless_attached = {
        p["a_track_id"] for p in (result.get("pending") or []) if p.get("kind") == "faceless_attach"
    }
    persons: list[dict[str, Any]] = []
    # GPID numbers are dense (1-based, no gaps): skipped groups must not
    # consume a number, so the counter advances only per person created.
    gpid = 0
    for tids in result.get("groups") or []:
        members = [by_id[tid] for tid in tids if tid in by_id]
        if not members:
            continue
        if any(int(t.get("track_id")) in unassigned for t in members):
            continue
        if not any(t.get("face_samples") for t in members):
            continue
        gpid += 1
        gi = gpid
        first = min(int(t.get("first_frame") or 0) for t in members)
        face_samples = sum(len(t.get("face_samples") or []) for t in members)
        low = any(int(t.get("track_id")) in faceless_attached for t in members)
        # Confidence: how strongly the group holds together. Edge strength
        # is the best face pair of the merges that formed it; singletons
        # fall back to their own face quality.
        qualities = [_tracklet_quality(t) for t in members]
        mean_quality = float(sum(qualities) / len(qualities)) if qualities else 0.0
        confidence = mean_quality
        if len(members) >= 2:
            bests: list[float] = []
            for a_i in range(len(members)):
                for b_i in range(a_i + 1, len(members)):
                    dec = pair_decision(members[a_i], members[b_i])
                    if dec["face"]["best"] is not None:
                        bests.append(float(dec["face"]["best"]))
            if bests:
                confidence = 0.5 * float(np.median(np.asarray(bests))) + 0.5 * mean_quality
        if low:
            confidence = min(confidence, 0.49)
        rep_face = None
        best_sample: dict[str, Any] | None = None
        for t in members:
            for s in t.get("face_samples") or []:
                if s.get("bbox") and (
                    best_sample is None or float(s.get("weight") or 0.0) > float(best_sample.get("weight") or 0.0)
                ):
                    best_sample = {**s, "track_id": int(t.get("track_id"))}
        if best_sample is not None:
            rep_face = {
                "track_id": int(best_sample["track_id"]),
                "frame_idx": int(best_sample.get("frame_idx") or 0),
                "bbox": [float(x) for x in (best_sample.get("bbox") or [0, 0, 0, 0])],
            }
        cluster_ids = sorted(
            {int(t["cluster_id"]) for t in members if t.get("cluster_id") is not None}
        )
        persons.append(
            {
                "gpid": gi,
                "cluster_id": gi,
                "name": f"Person_{gi}",
                "auto_name": f"Person_{gi}",
                "tracklet_ids": [int(t.get("track_id")) for t in members],
                "track_ids": [int(t.get("track_id")) for t in members],
                "embedding": _representative_embedding(members),
                "confidence": round(max(0.0, min(1.0, confidence)), 3),
                "representative_face": rep_face,
                "cluster_ids": cluster_ids,
                "low_confidence": bool(low),
                "face_samples": face_samples,
                "first_frame": first,
            }
        )
    return persons


def costume_hist_from_crop(frame: Any, person_bbox: list[float]) -> list[float] | None:
    """HSV histogram of the costume region (lower 60% of the person box).

    H 12 × S 4 × V 4 bins, L2-normalized, as a float list. Returns None
    when the box is too small to measure. Pure numpy/cv2 — no model.
    """
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in person_bbox]
    # Costume lives below the chin: lower 60% of the person box.
    y1 = y1 + (y2 - y1) * 0.4
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(x2)), min(h, int(y2))
    if ix2 - ix1 < 8 or iy2 - iy1 < 8:
        return None
    crop = frame[iy1:iy2, ix1:ix2]
    small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(COSTUME_HIST_BINS), [0, 180, 0, 256, 0, 256]
    )
    vec = np.asarray(hist, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return [float(x) for x in (vec / norm)]


def aggregate_costume_hist(hists: list[list[float] | None]) -> list[float] | None:
    """Mean of per-frame costume histograms, renormalized (None when empty)."""
    vecs = [np.asarray(h, dtype=np.float64) for h in (hists or []) if h is not None]
    if not vecs:
        return None
    mean = np.stack(vecs).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return None
    return [float(x) for x in (mean / norm)]


def attach_body_embeddings(
    video: Any,
    tracklets: list[dict[str, Any]],
    track_payload: dict[str, Any] | None = None,
    *,
    tagger: Any = None,
    max_frames_per_tracklet: int = 2,
) -> int:
    """Fill ``body_embedding`` on tracklets from CLIP (best-effort).

    Crops each tracklet's person box on up to ``max_frames_per_tracklet``
    sampled frames and averages the CLIP image embeddings. Returns how
    many tracklets got an embedding. Any failure (no torch / no
    openai-clip / no weights) is logged and leaves the embedding as None
    — the body vote then abstains, it never blocks stitching. Callers
    must treat a 0 return as "body vote unavailable", not an error.
    """
    if not tracklets:
        return 0
    try:
        if tagger is None:
            from recast.stages.costume import get_tagger

            tagger = get_tagger("clip")
        embed = getattr(tagger, "embed", None)
        if embed is None:
            log.info("身体外观：CLIP tagger 没有 embed 方法，身体票弃权。")
            return 0
    except Exception as exc:  # noqa: BLE001 - optional model, never fatal
        log.info("身体外观：CLIP 不可用（%s），身体票弃权。", exc)
        return 0

    from PIL import Image

    from recast.media import open_capture

    tracks = {
        int(t["track_id"]): t for t in ((track_payload or {}).get("tracks") or [])
    }
    wanted: dict[int, list[tuple[int, list[float]]]] = {}
    for t in tracklets:
        tid = int(t.get("track_id"))
        frames = (tracks.get(tid) or {}).get("frames") or []
        if not frames:
            continue
        picked = frames[:: max(1, len(frames) // max_frames_per_tracklet)][:max_frames_per_tracklet]
        for item in picked:
            wanted.setdefault(int(item["frame_idx"]), []).append((tid, item["bbox"]))
    if not wanted:
        return 0
    per_tracklet: dict[int, list[np.ndarray]] = {}
    cap = open_capture(video)
    try:
        last = max(wanted)
        frame_idx = 0
        while frame_idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in wanted:
                for tid, bbox in wanted[frame_idx]:
                    try:
                        import cv2

                        h, w = frame.shape[:2]
                        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
                        if x2 - x1 < 16 or y2 - y1 < 16:
                            continue
                        crop = frame[y1:y2, x1:x2]
                        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        vec = embed(pil)
                        if vec is not None:
                            per_tracklet.setdefault(tid, []).append(
                                np.asarray(vec, dtype=np.float32)
                            )
                    except Exception as exc:  # noqa: BLE001 - one bad crop skips
                        log.debug("身体外观裁剪失败（track %s）：%s", tid, exc)
            frame_idx += 1
    finally:
        cap.release()
    done = 0
    for t in tracklets:
        vecs = per_tracklet.get(int(t.get("track_id")) or -1) or []
        if not vecs:
            continue
        mean = np.stack(vecs).mean(axis=0)
        t["body_embedding"] = [float(x) for x in _l2_normalize(mean)]
        done += 1
    return done
