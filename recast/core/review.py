"""Manual review queue: the unclear stuff goes here, not into the library.

Two kinds of items land in the queue:

1. Unconfirmed character clusters (``confirmed = 0`` in the store) — the
   extractor was not sure who this is, so a human has to name it, merge it
   into an existing character, or discard it.
2. Low-quality face samples — a crop that is too small or too blurry to be
   a trustworthy reference. See :func:`score_sample_file` for how the score
   is computed.

Why a file-based score instead of :func:`recast.extract.faces.score_reference_sample`?
The extraction-time detection score (``det_score`` / bbox / frontal) is not
persisted anywhere: ``character_assets`` rows only store kind + path, and
``face_samples/index.json`` only stores paths. Re-running detection just to
score a crop would drag the whole InsightFace model back in. The file-based
score below measures the same two things the extraction score cares about
(size and clarity) from the crop file itself, with no model.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

REVIEW_TRASH_DIR = "review_trash"
# Discarded samples go one level deeper (review_trash/discarded/) so the
# queue never shows them again; corrected samples wait in review_trash/
# itself and DO show up until someone assigns, re-creates or discards them.
REVIEW_TRASH_DISCARDED = "discarded"
# A crop is "clear" when it is at least this big and sharp. The threshold
# derivation (see score_sample_file): a sharp 120x120 crop scores 1.0, a
# sharp 60x60 crop scores 0.5, a blurry small crop scores well below 0.5.
# 0.5 therefore keeps "usable but not great" and flags "clearly bad".
DEFAULT_REVIEW_MIN_SCORE = 0.5


@dataclass
class ReviewItem:
    """One row in the manual review queue."""

    kind: str  # "cluster" | "sample" | "stitch"
    key: str  # stable id: character id (cluster) or file path (sample)
    character_id: str | None
    character_name: str
    thumb: str | None
    score: float | None
    source: str = ""
    gpid: str | None = None  # B1: 所属全局人物编号（cluster_key "{drama}:{数字}" 的数字部分）

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "character_id": self.character_id,
            "character_name": self.character_name,
            "thumb": self.thumb,
            "score": self.score,
            "source": self.source,
        }


def _laplacian_variance(gray) -> float:
    """Variance of the Laplacian (numpy only, no cv2 needed).

    Sharp images have strong second derivatives (typically > 150); blurry
    ones are flat (typically < 50). A solid-colour image scores 0.
    """
    import numpy as np

    a = np.asarray(gray, dtype=np.float64)
    if a.size == 0:
        return 0.0
    padded = np.pad(a, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * a
    )
    return float(lap.var())


def score_sample_file(path: str | Path) -> float:
    """Quality of one face crop file, 0.0 (useless) to ~1.0 (good).

    score = size_term * (0.4 + 0.6 * sharpness)

    - size_term mirrors ``score_reference_sample``: ``min(1, sqrt(area)/120)``,
      so a 120x120 crop scores 1.0 and a 60x60 crop scores 0.5. A tiny crop
      cannot be a trustworthy reference no matter how sharp it is.
    - sharpness = ``min(1, laplacian_variance / 150)``; sharp photos are
      typically > 150, blurry ones < 50, solid colour is 0.
    - The 0.4 floor means a large but soft crop still scores ~0.4-0.5
      (borderline, human decides); a small AND blurry crop scores < 0.3.

    Returns 0.0 when the file cannot be read at all (missing/corrupt) — an
    unreadable crop is not a reference.
    """
    from PIL import Image

    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            w, h = gray.size
            if w <= 0 or h <= 0:
                return 0.0
            size_term = min(1.0, math.sqrt(w * h) / 120.0)
            sharp = min(1.0, _laplacian_variance(gray) / 150.0)
            return round(size_term * (0.4 + 0.6 * sharp), 3)
    except Exception as exc:  # unreadable file: not a usable sample
        log.warning("读不懂人脸样本 %s：%s", path, exc)
        return 0.0


def _is_within(path: Path, root: Path) -> bool:
    """True when ``path`` (resolved) sits inside ``root`` (resolved)."""
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _face_sample_files(store: Any, output_root: str | Path) -> list[dict[str, Any]]:
    """Every face sample file, deduped by resolved path.

    Sources: the store's ``face`` assets and the character folders' ``face/``
    (the folder may hold files the store never registered, e.g. hand-copied).
    Each entry: {character_id, character_name, path, reviewed, rejected}.

    ``reviewed`` is True when a human already decided about this sample:
    the store asset row's notes contain "已审核" (set by move_face_sample
    when a sample is assigned / split / created into a character). Decided
    samples do not come back into the queue — otherwise assigning a bad
    sample to another character would never clear its row, because the
    file itself is still low-score. Discarding needs no marker: the file
    leaves the scanned locations.

    ``rejected`` is True when extraction held this crop back (blurry /
    too small / profile / low detection score): the store asset row's
    notes contain "待审核：" (set by _mark_rejected_assets), or the file
    name starts with "rejected_" (folder-scanned files carry the reason
    in the name). Rejected samples are hidden from the queue by default
    (see build_review_queue) — they are low-score by design, so showing
    them would flood the queue with rows nobody needs to decide on.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(character_id: str | None, name: str, raw: Any, *, reviewed: bool = False, rejected: bool = False) -> None:
        p = Path(str(raw))
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen or not p.is_file() or p.is_symlink():
            return
        seen.add(key)
        # Folder-scanned rejected crops carry the marker in the file name
        # even when no store row (and no note) exists for them.
        rejected = rejected or p.name.startswith("rejected_")
        out.append({"character_id": character_id, "character_name": name, "path": str(p), "reviewed": reviewed, "rejected": rejected})

    for ch in store.list_characters():
        name = ch.get("display_name") or ch.get("name") or ch["id"]
        for a in ch.get("assets") or []:
            if a.get("kind") == "face" and a.get("path"):
                _add(ch["id"], name, a["path"], reviewed="已审核" in str(a.get("notes") or ""), rejected="待审核：" in str(a.get("notes") or ""))
    # Character folders (may hold files the store never registered).
    try:
        from recast.core.characters import list_character_folders

        for entry in list_character_folders(output_root):
            meta = entry.get("meta") or {}
            face = meta.get("face") if isinstance(meta.get("face"), dict) else {}
            for raw in face.get("files") or []:
                _add(None, entry["name"], raw)
            folder = Path(entry["folder"])
            for p in sorted((folder / "face").glob("*")) if (folder / "face").is_dir() else []:
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    _add(None, entry["name"], p)
    except Exception as exc:  # folders unreadable: store assets are enough
        log.warning("扫描人物文件夹取人脸样本失败：%s", exc)
    return out


def _gpid_from_cluster_key(cluster_key: Any) -> str | None:
    """cluster_key "{drama}:{数字}" -> 数字部分（字符串），取不到为 None。

    非数字后缀（如 "ep1:Person_1"、"manual:温念"）、无冒号、非字符串、
    空值都返回 None：只有纯数字后缀才能对应一个全局人物编号。
    """
    if not isinstance(cluster_key, str) or ":" not in cluster_key:
        return None
    suffix = cluster_key.split(":", 1)[1]
    return suffix if suffix.isdigit() else None


def _stitch_item(pair: Any) -> ReviewItem | None:
    """一个 pending 拼接对 -> 一行 kind="stitch" 的 ReviewItem。

    key=f"stitch:{a_track_id}:{b_track_id}"；score 取 face_best（没有
    时为 None，如 faceless_attach）；source 取 pair 的 reason。a_track_id
    缺失时无法成行，返回 None（跳过该对）。
    """
    if not isinstance(pair, dict):
        return None
    a = pair.get("a_track_id")
    if a is None:
        return None
    face_best = pair.get("face_best")
    try:
        score = float(face_best) if face_best is not None else None
    except (TypeError, ValueError):
        score = None
    return ReviewItem(
        kind="stitch",
        key=f"stitch:{a}:{pair.get('b_track_id')}",
        character_id=None,
        character_name="待裁决拼接",
        thumb=None,
        score=score,
        source=str(pair.get("reason") or ""),
        gpid=None,
    )


def build_review_queue(
    store: Any,
    output_root: str | Path,
    *,
    min_score: float = DEFAULT_REVIEW_MIN_SCORE,
    include_rejected: bool = False,
    pending_pairs: list[dict[str, Any]] | None = None,
) -> list[ReviewItem]:
    """The manual review queue: unclear clusters first, then bad samples.

    - One "cluster" row per unconfirmed character (``confirmed = 0``).
    - One "sample" row per face crop whose file score is below ``min_score``.
    Rows carry a thumbnail path and a source string so the UI can show
    "who / where / how bad" at a glance. An empty queue means step 3
    (人物审核确认) is done.

    ``include_rejected`` (default False): extraction-held-back samples
    (blurry / too small / profile / low detection score, marked
    "待审核：" in the asset notes or named ``rejected_*``) are hidden by
    default — they are low-score by design, so showing them floods the
    queue with rows nobody needs to decide on (Daniel's screenshot: 68
    of 90 rows). Pass True to show them again, e.g. from the library
    window's "显示低质量样本" checkbox. Two things are NOT affected by
    the flag: ordinary low-score samples (not rejected) still queue, and
    correction-moved samples waiting in ``review_trash/`` always queue
    (they were moved out for belonging to the wrong person, not for
    being blurry).
    ``pending_pairs`` (default None): stitch.py 的 pending 拼接对列表
    （identity.json "pending_pairs"）。每对追加一行 kind="stitch" 的
    ReviewItem，插在聚类行之后、样本行之前；为 None 或空时行为与改前
    完全一致。
    """
    items: list[ReviewItem] = []
    characters = store.list_characters()
    by_id = {ch["id"]: ch for ch in characters}

    for ch in characters:
        if ch.get("confirmed"):
            continue
        name = ch.get("display_name") or ch.get("name") or ch["id"]
        thumb = None
        for a in ch.get("assets") or []:
            if a.get("kind") == "face" and a.get("path") and Path(str(a["path"])).is_file():
                thumb = str(a["path"])
                break
        items.append(
            ReviewItem(
                kind="cluster",
                key=ch["id"],
                character_id=ch["id"],
                character_name=name,
                thumb=thumb,
                score=None,
                source=f"聚类「{name}」未确认",
                gpid=_gpid_from_cluster_key(ch.get("cluster_key")),
            )
        )

    # 待裁决的拼接对：聚类行之后、样本行之前。
    for pair in pending_pairs or []:
        item = _stitch_item(pair)
        if item is not None:
            items.append(item)

    for entry in _face_sample_files(store, output_root):
        if entry.get("reviewed"):
            continue
        if entry.get("rejected") and not include_rejected:
            continue
        score = score_sample_file(entry["path"])
        if score >= min_score:
            continue
        items.append(
            ReviewItem(
                kind="sample",
                key=entry["path"],
                character_id=entry["character_id"],
                character_name=entry["character_name"],
                thumb=entry["path"],
                score=score,
                source=f"样本来自「{entry['character_name']}」",
            )
        )

    # Samples a human moved out of a character (correction): they wait in
    # review_trash/ until they are assigned, made into a new character, or
    # discarded (which moves them to review_trash/discarded/, not scanned).
    # They show up whatever their score is — they were moved out because
    # they belong to the wrong person, not because they are blurry.
    try:
        trash = Path(output_root).expanduser().resolve() / REVIEW_TRASH_DIR
        if trash.is_dir():
            for p in sorted(trash.iterdir()):
                if not p.is_file() or p.is_symlink():
                    continue
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                items.append(
                    ReviewItem(
                        kind="sample",
                        key=str(p),
                        character_id=None,
                        character_name="待审核（已移出）",
                        thumb=str(p),
                        score=score_sample_file(p),
                        source="移出的样本（待复核）",
                    )
                )
    except OSError:
        pass
    return items


def review_trash_path(output_root: str | Path, name: str, *, discarded: bool = False) -> Path:
    """Where a moved-out / discarded sample goes (never deleted outright).

    Correction moves land in ``review_trash/`` itself (the queue shows them
    again); discards land in ``review_trash/discarded/`` (out of the queue).
    """
    trash = Path(output_root).expanduser().resolve() / REVIEW_TRASH_DIR
    if discarded:
        trash = trash / REVIEW_TRASH_DISCARDED
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        n = 1
        while (trash / f"{stem}_{n}{suffix}").exists():
            n += 1
        dest = trash / f"{stem}_{n}{suffix}"
    return dest


# ----- 队列的批量 / 筛选 / 分组辅助函数（UI 用） -----
# 只读纯函数：不改 build_review_queue 的返回结构（已有测试依赖它）。

# 分数段边界（UI 筛选框的字面含义）：<0.3 明显差，0.3–0.5 边缘。
SCORE_BAND_LOW = 0.3
SCORE_BAND_MID_MAX = 0.5


def _item_get(item: Any, key: str) -> Any:
    """ReviewItem 或 dict 都能读（UI 传 dict，core 测试传 ReviewItem）。"""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def filter_review_items(
    items: list[Any],
    *,
    kinds: set[str] | None = None,
    score_band: str = "all",
) -> list[Any]:
    """按类型 / 分数段筛队列（不改原列表顺序）。

    - kinds: {"cluster"} / {"sample"} / None（全部）。
    - score_band: "all" | "low"（分数 < 0.3）| "mid"（0.3–0.5）。
      聚类行没有文件分数（score 是 None），分数段筛不过滤它们：
      未确认的聚类不管多清楚都得有人拍板，不能被分数筛藏起来。
    """
    out: list[Any] = []
    for item in items:
        kind = _item_get(item, "kind")
        if kinds is not None and kind not in kinds:
            continue
        if score_band in ("low", "mid"):
            score = _item_get(item, "score")
            if score is not None:
                if score_band == "low" and not (score < SCORE_BAND_LOW):
                    continue
                if score_band == "mid" and not (
                    SCORE_BAND_LOW <= score < SCORE_BAND_MID_MAX
                ):
                    continue
        out.append(item)
    return out


def group_review_items(items: list[Any]) -> list[tuple[tuple[str, str], str, list[Any]]]:
    """按（类型，人物）分组，首次出现顺序。

    返回 [(group_key, title, [items...]), ...]，group_key 是
    (kind, character_name)，title 如 "聚类 · 人物1" / "样本 · 温念"。
    队列里的三类来源（未确认聚类、低分样本、纠错移出的样本）都按
    这个键归组：移出的样本 character_name 是 "待审核（已移出）"，自成一组。
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    order: list[tuple[str, str]] = []
    for item in items:
        kind = _item_get(item, "kind") or "sample"
        name = _item_get(item, "character_name") or "未命名"
        key = (kind, name)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    out: list[tuple[tuple[str, str], str, list[Any]]] = []
    for kind, name in order:
        title = f"{'聚类' if kind == 'cluster' else '样本'} · {name}"
        out.append(((kind, name), title, groups[(kind, name)]))
    return out


def _layer_sort_key(item: Any) -> tuple[int, int, int, str]:
    """decisions 层内的排序键：stitch 在前，聚类在后；同类内先按 gpid
    （数字升序，有 gpid 的在前），再按名字。gpid 是字符串，"10" 应排
    在 "3" 之后，故按数值比；无 gpid / 非数字 gpid 的排在同类末尾。
    """
    kind = _item_get(item, "kind")
    kind_rank = 0 if kind == "stitch" else 1
    gpid = _item_get(item, "gpid")
    if isinstance(gpid, str) and gpid.isdigit():
        return (kind_rank, 0, int(gpid), str(_item_get(item, "character_name") or ""))
    return (kind_rank, 1, 0, str(_item_get(item, "character_name") or ""))


def layer_review_items(items: list[Any]) -> list[tuple[str, str, list[Any]]]:
    """把队列分成两层（UI 分区显示用），不改 build_review_queue 的返回结构。

    - ("decisions", "待确认（人物与拼接）", [kind 为 cluster 或 stitch
      的行，stitch 行在前，同类内按 gpid/名字排序])
    - ("samples", "低分样本", [kind 为 sample 的行，原顺序])

    空层不返回；两层都空时返回 []。接受 ReviewItem 或 dict（与
    filter/group 辅助函数一致，用 _item_get 读字段）。
    """
    decisions = [i for i in items if _item_get(i, "kind") in ("cluster", "stitch")]
    samples = [i for i in items if _item_get(i, "kind") == "sample"]
    out: list[tuple[str, str, list[Any]]] = []
    if decisions:
        out.append(("decisions", "待确认（人物与拼接）", sorted(decisions, key=_layer_sort_key)))
    if samples:
        out.append(("samples", "低分样本", list(samples)))
    return out


def high_confidence_items(
    items: list[Any],
    *,
    min_score: float = DEFAULT_REVIEW_MIN_SCORE,
) -> list[Any]:
    """一键确认的候选：只含聚类行。

    规则：聚类行 + 有代表脸缩略图 + 那张脸的清晰度分 >= min_score
    （默认 0.5，即 review_min_score）。样本行不在候选里：进队列的样本
    本来就是低分（或被纠错移出），没有"高置信度"的样本。

    注意：提取时算的 cluster_distance / confidence（见
    recast/stages/identity.py cluster_distance_and_confidence）没有存进
    人物表，队列里拿不到真正的聚类距离；这里用"代表脸够清楚"作代理。
    想用真距离的话，得先让提取阶段把 confidence 写进人物行，那是
    identity.py 的事，不在本函数管辖内。
    """
    out: list[Any] = []
    for item in items:
        if _item_get(item, "kind") != "cluster":
            continue
        thumb = _item_get(item, "thumb")
        if not thumb:
            continue
        try:
            if score_sample_file(thumb) >= min_score:
                out.append(item)
        except Exception:
            continue
    return out
