"""跨集归并：新聚类先比对已有人物，够像就并入，不够像才建新文件夹。

多集时同一个人会被拆成多个文件夹，因为每一集的 ingest 都是按
``{drama_key}:{cluster_id}`` 直接建新人物，从不比对已有人物。这里补上
这一步：建新人物之前，拿新聚类的 embedding 和已有人物的 embedding
逐一比余弦相似度。

判定流程（调用方见 :func:`recast.extract.faces.ingest_into_library`）：

1. 取新聚类的 embedding（identity.json 里 ``persons[].embedding``）。
   没有 embedding（没采到脸）无法比对，直接走建新文件夹的老路。
2. 读已有人物的 embedding：优先 store 里 ``characters.face_embedding``
   列（ingest 时写入）；其次人物文件夹 ``meta.json`` 的
   ``face_embedding``（供 store 里还没有的老人物用）。
3. 相似度最高的人物达到 ``IdentityConfig.merge_similarity``（默认
   0.6）就并入：样本拷进那个人物（不建新文件夹），meta 更新，存的
   embedding 用新旧加权平均更新；达不到才建新文件夹。
4. 幂等：同一批数据跑两次，第二次没有新样本可加（资产按路径去重），
   embedding 不再更新，不产生重复人物。

阈值为什么是 0.6：同一人 InsightFace 余弦相似度通常 > 0.6，不同人
< 0.3，0.6 落在中间偏严的一侧——宁可漏并（人工在审核队列里点一下
合并）也不错并（两个人混进一个文件夹更难拆）。

三票制（B1）：人物文件夹 meta.json 顶层新增 ``body_embedding``（CLIP
身体外观）与 ``costume_hist``（HSV 服装直方图）后，
:func:`find_merge_target_votes` 在新人物与候选双方都至少有一项身体/
服装特征时走 :func:`recast.stages.stitch.pair_decision` 的三票制（人脸
最佳配对 + 身体 + 服装，≥2 票且无人脸否决才合并）；否则按对回退到
上面的单人脸阈值。:func:`find_merge_target` 本身保持原样作回退。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("recast")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embedding vectors (0 when either is empty)."""
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def parse_embedding(value: Any) -> np.ndarray | None:
    """Parse a stored embedding (list, JSON string, ndarray) to a unit vector.

    Returns None for anything missing, malformed, or all-zero — a bad
    stored value must never match (or crash) a merge decision.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    try:
        vec = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vec.size == 0:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8 or not np.isfinite(norm):
        return None
    return (vec / norm).astype(np.float32)


def parse_feature_list(value: Any) -> list[float] | None:
    """Parse a stored body embedding / costume histogram (float list).

    Returns None for anything missing, malformed, empty, or non-finite —
    a bad stored value must abstain from the three-vote decision, never
    vote on garbage.
    """
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return [float(x) for x in arr]


def embedding_to_json(embedding: Any) -> str:
    """Serialize an embedding for the store / meta.json (list of floats)."""
    vec = parse_embedding(embedding)
    if vec is None:
        return "[]"
    return json.dumps([float(x) for x in vec], ensure_ascii=False)


def merge_embeddings(
    old: Any,
    new: Any,
    *,
    old_count: int = 1,
    new_count: int = 1,
) -> list[float] | None:
    """Weighted mean of two embeddings, L2-normalized, as a float list.

    Counts weight the mean toward the side built from more samples. Either
    side missing falls back to the other side unchanged.
    """
    old_vec = parse_embedding(old)
    new_vec = parse_embedding(new)
    if old_vec is None:
        return [float(x) for x in new_vec] if new_vec is not None else None
    if new_vec is None:
        return [float(x) for x in old_vec]
    if old_vec.size != new_vec.size:
        return [float(x) for x in old_vec]
    total = max(1, int(old_count) + int(new_count))
    merged = (old_vec.astype(np.float64) * int(old_count) + new_vec.astype(np.float64) * int(new_count)) / total
    norm = float(np.linalg.norm(merged))
    if norm < 1e-8:
        return [float(x) for x in old_vec]
    return [float(x) for x in (merged / norm)]


def load_existing_embeddings(
    store: Any,
    output_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Every existing character that has a stored embedding.

    Each entry: {"character_id", "name", "embedding", "source",
    "body_embedding", "costume_hist"} where source is "store"
    (``characters.face_embedding``) or "meta" (``meta.json``). The store
    wins when both exist for one character; entries without a usable
    embedding are skipped, never guessed. Meta-only entries have
    ``character_id`` / ``cluster_key`` None: they can be compared, but
    only a store row can be merged into.

    ``body_embedding`` (CLIP body appearance) and ``costume_hist`` (HSV
    histogram) are read from the top level of ``meta.json`` only — the
    store has no such columns, so store entries always carry None for
    both keys (the keys are still present). A missing or malformed
    value becomes None, never a guessed vector.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(
        character_id: str | None,
        name: str,
        raw: Any,
        *,
        source: str,
        body: Any = None,
        costume: Any = None,
    ) -> dict[str, Any] | None:
        emb = parse_embedding(raw)
        if emb is None:
            return None
        key = character_id or f"name:{name}"
        if key in seen:
            return None
        seen.add(key)
        entry = {
            "character_id": character_id,
            "name": name,
            "embedding": emb,
            "source": source,
            "cluster_key": None,
            "body_embedding": parse_feature_list(body),
            "costume_hist": parse_feature_list(costume),
        }
        out.append(entry)
        return entry

    if store is not None:
        try:
            for ch in store.list_characters():
                entry = _add(
                    ch.get("id"),
                    ch.get("display_name") or ch.get("name") or ch.get("id") or "",
                    ch.get("face_embedding"),
                    source="store",
                )
                if entry is not None:
                    entry["cluster_key"] = ch.get("cluster_key")
        except Exception as exc:  # store unreadable: meta.json is the fallback
            log.warning("读人物库 embedding 失败：%s", exc)

    if output_root is not None:
        try:
            from recast.core.characters import list_character_folders

            for entry in list_character_folders(output_root):
                meta = entry.get("meta") or {}
                raw = meta.get("face_embedding")
                if raw is None and isinstance(meta.get("face"), dict):
                    raw = meta["face"].get("embedding")
                # Body / costume live only at the top level of meta.json
                # (meta["face"] never carries them); missing or malformed
                # values become None in the entry.
                _add(
                    None,
                    entry["name"],
                    raw,
                    source="meta",
                    body=meta.get("body_embedding"),
                    costume=meta.get("costume_hist"),
                )
        except Exception as exc:  # folders unreadable: store entries are enough
            log.warning("读人物文件夹 embedding 失败：%s", exc)
    return out


def find_merge_target(
    embedding: Any,
    candidates: list[dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any] | None, float]:
    """Best-matching existing character, or (None, best similarity).

    ``embedding`` is the new cluster's embedding; ``candidates`` come from
    :func:`load_existing_embeddings`. The match must reach ``threshold``;
    the best below-threshold score is still returned so the caller can
    record how close the decision was.
    """
    vec = parse_embedding(embedding)
    if vec is None:
        return None, 0.0
    best: dict[str, Any] | None = None
    best_sim = 0.0
    for cand in candidates:
        sim = cosine_similarity(vec, cand["embedding"])
        if sim > best_sim:
            best_sim = sim
            best = cand
    if best is not None and best_sim >= float(threshold):
        return best, best_sim
    return None, best_sim


def _has_body_or_costume(features: dict[str, Any] | None) -> bool:
    """True when at least one body/costume feature is present (non-empty)."""
    if not features:
        return False
    for key in ("body_embedding", "costume_hist"):
        value = features.get(key)
        if value is None:
            continue
        try:
            if len(value) > 0:
                return True
        except TypeError:
            return True
    return False


def score_merge_votes(
    new_features: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    cfg: Any = None,
) -> dict[str, Any]:
    """Three-vote score for one new person vs one existing character.

    ``new_features`` is {"embedding", "body_embedding", "costume_hist"}
    (missing keys count as None); ``candidate`` is one entry from
    :func:`load_existing_embeddings`. Both sides are reshaped into the
    tracklet-style dicts :func:`recast.stages.stitch.pair_decision`
    expects — the new person's face embedding becomes a single
    face sample (weight 1.0); an absent face embedding means an empty
    sample list, so the face vote abstains. The ``pair_decision`` result
    is returned unchanged (votes / face / body / costume / decision /
    reason).
    """
    from recast.stages.stitch import pair_decision

    def _side(features: dict[str, Any] | None) -> dict[str, Any]:
        features = features or {}
        emb = parse_embedding(features.get("embedding"))
        return {
            "face_samples": [{"emb": emb, "weight": 1.0}] if emb is not None else [],
            "body_embedding": parse_feature_list(features.get("body_embedding")),
            "costume_hist": parse_feature_list(features.get("costume_hist")),
        }

    return pair_decision(_side(new_features), _side(candidate), cfg)


def _face_only_detail(
    new_emb: np.ndarray,
    cand_emb: Any,
    threshold: float,
) -> dict[str, Any]:
    """Fallback detail for a pair without body/costume on either side.

    Same rule as :func:`find_merge_target` for one candidate — cosine
    similarity against ``threshold`` (``IdentityConfig.merge_similarity``)
    — shaped like a ``pair_decision`` result so callers can log and rank
    vote-scored and fallback-scored candidates uniformly.
    """
    cand_vec = parse_embedding(cand_emb)
    sim = cosine_similarity(new_emb, cand_vec) if cand_vec is not None else 0.0
    merged = sim >= float(threshold)
    return {
        "votes": 1 if merged else 0,
        "face": {"vote": merged, "veto": False, "best": sim, "median": sim, "abstain": False},
        "body": {"vote": False, "sim": None, "abstain": True},
        "costume": {"vote": False, "sim": None, "abstain": True},
        "decision": "merge" if merged else "no_merge",
        "reason": (
            f"无身体/服装特征，回退单人脸阈值比对：相似度 {sim:.2f}"
            f"{' ≥ ' if merged else ' < '}{float(threshold):.2f}。"
        ),
    }


def find_merge_target_votes(
    new_person: dict[str, Any] | None,
    candidates: list[dict[str, Any]] | None,
    cfg: Any = None,
) -> tuple[dict[str, Any] | None, float, dict[str, Any] | None]:
    """Best cross-episode merge target, three-vote when possible.

    ``new_person`` is {"embedding", "body_embedding", "costume_hist"};
    ``candidates`` are entries from :func:`load_existing_embeddings`.
    Returns (target | None, best face similarity, detail | None).

    Per candidate pair, the three-vote rule
    (:func:`score_merge_votes`) applies only when the new person AND
    that candidate each carry at least one body/costume feature;
    otherwise the pair falls back to the single-face rule of
    :func:`find_merge_target` (threshold
    ``IdentityConfig().merge_similarity``, default 0.6 — never the
    stitch thresholds). Among candidates whose decision is "merge",
    the one with the most votes wins, ties broken by the best face
    pair; with no merge, the best face similarity and that candidate's
    detail are returned. A missing new face embedding cannot be
    compared at all: (None, 0.0, None).
    """
    from recast.config import IdentityConfig

    new_person = new_person or {}
    new_emb = parse_embedding(new_person.get("embedding"))
    if new_emb is None:
        return None, 0.0, None
    threshold = float(IdentityConfig().merge_similarity)
    scored: list[tuple[int, float, dict[str, Any], dict[str, Any]]] = []
    for cand in candidates or []:
        if _has_body_or_costume(new_person) and _has_body_or_costume(cand):
            detail = score_merge_votes(new_person, cand, cfg)
        else:
            detail = _face_only_detail(new_emb, cand.get("embedding"), threshold)
        face_best = detail["face"]["best"]
        face_best = 0.0 if face_best is None else float(face_best)
        scored.append((int(detail["votes"]), face_best, cand, detail))
    if not scored:
        return None, 0.0, None
    merges = [s for s in scored if s[3]["decision"] == "merge"]
    if merges:
        merges.sort(key=lambda s: (-s[0], -s[1]))
        votes, best, cand, detail = merges[0]
        return cand, best, detail
    best_face = max(s[1] for s in scored)
    best_cands = [s for s in scored if s[1] == best_face]
    best_cands.sort(key=lambda s: -s[0])
    _votes, best, cand, detail = best_cands[0]
    return None, best, detail


def resolve_output_root(store: Any = None, output_root: str | Path | None = None) -> Path | None:
    """Derive the output root (the directory holding ``characters/``).

    Explicit ``output_root`` wins. Otherwise the store's database path is
    used: the default layout is ``<output_root>/store/recast.db`` (see
    ``recast.store.db.default_db_path`` and ``Settings.output_root``), so
    a store path whose parent directory is named ``store`` resolves to
    its grandparent. Any other layout returns None — a guessed root must
    never be used to write meta.json files into the wrong folders.
    """
    if output_root is not None:
        return Path(output_root).expanduser()
    path = getattr(store, "path", None) if store is not None else None
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.parent.name == "store":
        return path.parent.parent
    return None


def sync_folder_embeddings(
    store: Any, output_root: str | Path | None = None
) -> dict[str, Any]:
    """Mirror store ``face_embedding`` values into folder meta.json files.

    The pipeline (``run_library`` → ``ingest_into_library``) writes the
    embedding to the store row but never passes ``output_root``, so the
    folder mirror used to silently never happen. This closes that gap:
    every store character with a usable embedding gets it written into
    ``<output_root>/characters/<name>/meta.json`` (only when it differs;
    nothing is created — a missing folder or meta.json is skipped with a
    reason, never invented).

    Returns ``{"written": [names], "skipped": [{"name", "reason"}],
    "error": str | None}``. ``error`` is set (with the reason written out)
    when the output root cannot be derived or the store cannot be read —
    the mirror is skipped, never silently dropped.
    """
    root = resolve_output_root(store, output_root)
    if root is None:
        return {
            "written": [],
            "skipped": [],
            "error": (
                "推导不出输出根目录：没有传入 output_root，store 的路径也不在"
                " <输出根>/store/recast.db 的默认位置；meta.json 镜像已跳过。"
            ),
        }
    from recast.core.characters import character_folder, load_meta, save_meta

    try:
        characters = store.list_characters() if store is not None else []
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"written": [], "skipped": [], "error": f"读人物库失败：{exc}"}
    written: list[str] = []
    skipped: list[dict[str, str]] = []
    for ch in characters or []:
        name = ch.get("display_name") or ch.get("name") or ""
        label = name or str(ch.get("id") or "未知人物")
        emb = parse_embedding(ch.get("face_embedding"))
        if emb is None:
            skipped.append({"name": label, "reason": "库里没有可用的 face_embedding"})
            continue
        try:
            folder = character_folder(root, name) if name else None
            if folder is None or not folder.is_dir():
                skipped.append({"name": label, "reason": "没有人物文件夹"})
                continue
            meta = load_meta(folder)
            if meta is None:
                skipped.append({"name": label, "reason": "人物文件夹里没有 meta.json"})
                continue
            current = parse_embedding(meta.get("face_embedding"))
            if current is not None and current.shape == emb.shape:
                import numpy as _np

                if bool(_np.allclose(current, emb)):
                    skipped.append({"name": label, "reason": "已一致"})
                    continue
            meta["face_embedding"] = [float(x) for x in emb]
            save_meta(folder, meta)
            written.append(label)
        except Exception as exc:  # noqa: BLE001 - one bad folder must not stop the rest
            skipped.append({"name": label, "reason": f"写 meta.json 失败：{exc}"})
    return {"written": written, "skipped": skipped, "error": None}
