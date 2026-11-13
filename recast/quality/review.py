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


#: 合成前唇同步警告的容差（秒）。配音比镜头长出这么多以内不警告——
#: 与 ``recast.templates.duration.extend_timeline_for_audio`` 认定“可忽略”
#: 的 0.05 秒是同一数值，拼贴时小于它的超长不会被当成节奏问题。
LIPSYNC_TOLERANCE_SEC = 0.05


@dataclass
class LipSyncWarning:
    """One line whose dubbed audio outruns its shot slot.

    This is a *duration* warning only: it says the voice track is longer
    than the picture it must fit into. It does NOT check mouth shapes —
    true lip-sync (frame-by-frame mouth alignment) is a later step.
    """

    shot_code: str
    text: str
    speaker: str
    audio_sec: float
    slot_sec: float
    over_sec: float
    message: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            shown = self.text if len(self.text) <= 18 else self.text[:18] + "…"
            who = f"「{self.speaker}」" if self.speaker else ""
            self.message = (
                f"{self.shot_code} {who}「{shown}」配音 {self.audio_sec:.2f} 秒，"
                f"镜头 {self.slot_sec:.2f} 秒，超出 {self.over_sec:.2f} 秒。"
                "请改短台词或拆到下一镜；继续合成只会让这句盖住下一镜的开头，"
                "且本检查不管口型对不对。"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "shot_code": self.shot_code,
            "text": self.text,
            "speaker": self.speaker,
            "audio_sec": round(self.audio_sec, 3),
            "slot_sec": round(self.slot_sec, 3),
            "over_sec": round(self.over_sec, 3),
            "message": self.message,
        }


def _slot_seconds(line: dict[str, Any]) -> float:
    """Shot slot for one dialogue line.

    Same order as ``recast.templates.duration.shot_slot_seconds``:
    start/end wins, ``duration_sec`` is the fallback. Missing slot is 0 —
    a line with audio but no slot at all is an overrun of the whole line.
    """
    start = line.get("start_time")
    end = line.get("end_time")
    try:
        if start is not None and end is not None and float(end) > float(start):
            return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        pass
    try:
        return max(0.0, float(line.get("duration_sec") or 0))
    except (TypeError, ValueError):
        return 0.0


def check_lipsync_warnings(
    lines: list[dict[str, Any]],
    *,
    tolerance_sec: float = LIPSYNC_TOLERANCE_SEC,
) -> list[LipSyncWarning]:
    """Flag lines whose *actual dubbed audio* outruns their shot slot.

    阈值规则（写死在这里，二选一的结果）：**配音时长 > 镜头时长 + 容差**。
    用的是配音完成后的真实 wav 时长（``audio_duration``），不是
    ``check_dialogue_overflow`` 那套“>4 字/秒”的文字预估——那套在配音前
    按字数估算，本检查在合成前按真实配音时长核对。容差默认 0.05 秒，
    与 ``extend_timeline_for_audio`` 的忽略阈值一致。

    还没配音的行（``audio_duration`` 缺失或为 0）跳过不判——判不了，
    文字超长由 ``check_dialogue_overflow`` 另外管。输入行只读字典，
    不改任何字段；输出按行号/镜头顺序原样排列。
    """
    warnings: list[LipSyncWarning] = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        try:
            audio = float(line.get("audio_duration") or 0)
        except (TypeError, ValueError):
            continue
        if audio <= 0:
            continue
        slot = _slot_seconds(line)
        over = audio - slot
        if over <= tolerance_sec:
            continue
        code = str(line.get("shot_code") or line.get("code") or "（未编号镜头）")
        warnings.append(
            LipSyncWarning(
                shot_code=code,
                text=str(line.get("text") or line.get("dialogue") or ""),
                speaker=str(line.get("speaker") or line.get("character") or ""),
                audio_sec=audio,
                slot_sec=slot,
                over_sec=round(over, 3),
            )
        )
    return warnings


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
