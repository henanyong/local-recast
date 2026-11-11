"""Face-embedding drift, optional costume/style flags, dialogue overflow → 待人工检查."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from recast.templates.duration import DialogueOverflow, check_shots


DEFAULT_FACE_THRESHOLD = 0.35


@dataclass
class ReviewItem:
    shot_code: str
    reason: str
    detail: str = ""
    shot_id: str | None = None
    score: float | None = None
    thumbnail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "shot_code": self.shot_code,
            "reason": self.reason,
            "detail": self.detail,
            "shot_id": self.shot_id,
            "score": self.score,
            "thumbnail": self.thumbnail,
        }


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    x = list(a)
    y = list(b)
    if not x or not y or len(x) != len(y):
        return 0.0
    dot = sum(i * j for i, j in zip(x, y))
    nx = math.sqrt(sum(i * i for i in x))
    ny = math.sqrt(sum(j * j for j in y))
    if nx <= 0 or ny <= 0:
        return 0.0
    return dot / (nx * ny)


def check_face_drift(
    pairs: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_FACE_THRESHOLD,
) -> list[ReviewItem]:
    """pairs: {shot_code, library_embedding, observed_embedding, shot_id?}"""
    items: list[ReviewItem] = []
    for p in pairs:
        lib = p.get("library_embedding") or p.get("library")
        obs = p.get("observed_embedding") or p.get("observed")
        if lib is None or obs is None:
            continue
        sim = cosine_similarity(lib, obs)
        if sim < threshold:
            code = str(p.get("shot_code") or p.get("shot_id") or "")
            who = p.get("person_name") or p.get("name") or ""
            who_bit = f"「{who}」" if who else "换后的脸"
            items.append(
                ReviewItem(
                    shot_code=code,
                    reason="人脸特征漂移",
                    detail=(
                        f"{who_bit}与参考脸相似度 {sim:.3f}，低于阈值 {threshold:.2f}。"
                        "请看缩略图确认是否换错脸，并可单镜重跑。"
                    ),
                    shot_id=p.get("shot_id"),
                    score=sim,
                    thumbnail=p.get("thumbnail"),
                )
            )
    return items


def check_dialogue_overflow(shots: list[dict[str, Any]], *, max_cps: float = 4.0) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for issue in check_shots(shots, max_cps=max_cps):
        items.append(
            ReviewItem(
                shot_code=issue.shot_code,
                reason="对白超出时长",
                detail=issue.message,
            )
        )
    return items


def check_costume_flags(shots: list[dict[str, Any]]) -> list[ReviewItem]:
    """Optional heuristic flags from shot metadata (no vision model required)."""
    items: list[ReviewItem] = []
    for sh in shots:
        flags = sh.get("style_flags") or sh.get("costume_flags") or []
        for flag in flags:
            items.append(
                ReviewItem(
                    shot_code=str(sh.get("shot_code") or ""),
                    reason="服装 / 风格标记",
                    detail=str(flag),
                    shot_id=sh.get("id"),
                )
            )
    return items


def run_quality_checks(
    shots: list[dict[str, Any]],
    *,
    face_pairs: list[dict[str, Any]] | None = None,
    max_cps: float = 4.0,
    face_threshold: float = DEFAULT_FACE_THRESHOLD,
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    if face_pairs:
        items.extend(check_face_drift(face_pairs, threshold=face_threshold))
    items.extend(check_dialogue_overflow(shots, max_cps=max_cps))
    items.extend(check_costume_flags(shots))
    return items
