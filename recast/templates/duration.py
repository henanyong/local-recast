"""Word-count vs shot-duration pre-check. Never auto-speed audio."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Conservative spoken-Chinese rate. Faster than this sounds rushed.
DEFAULT_MAX_CPS = 4.0
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass
class DialogueOverflow:
    shot_code: str
    text: str
    duration_sec: float
    char_count: int
    max_chars: int
    max_cps: float
    message: str

    def __str__(self) -> str:
        return self.message


def char_count(text: str) -> int:
    """Count spoken characters: CJK + letters + digits, no punctuation or spaces."""
    if not text:
        return 0
    n = 0
    for ch in text:
        if ch.isspace():
            continue
        if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf"):
            n += 1
            continue
        # other letters (fullwidth etc.)
        if ch.isalpha():
            n += 1
    return n


def check_dialogue_duration(
    text: str,
    duration_sec: float,
    *,
    shot_code: str = "",
    max_cps: float = DEFAULT_MAX_CPS,
) -> DialogueOverflow | None:
    """Return an overflow record if the line cannot fit the slot at max_cps.

    Callers must prompt the user to edit the shot or split the line.
    This function never recommends speeding up audio.
    """
    if duration_sec is None or duration_sec <= 0:
        if not (text or "").strip():
            return None
        # no usable slot
        duration_sec = 0.0
    count = char_count(text or "")
    max_chars = int(duration_sec * max_cps) if duration_sec > 0 else 0
    if count <= max_chars:
        return None
    code = shot_code or "（未编号镜头）"
    msg = (
        f"对白字数超出镜头时长：{code} 共 {count} 字，"
        f"时长 {duration_sec:.1f} 秒（按每秒最多 {max_cps:g} 字计算，上限 {max_chars} 字）。"
        "请修改分镜、缩短台词或把这句话拆到下一镜，不要加速音频。"
    )
    return DialogueOverflow(
        shot_code=code,
        text=text or "",
        duration_sec=float(duration_sec),
        char_count=count,
        max_chars=max_chars,
        max_cps=max_cps,
        message=msg,
    )


def check_shots(shots: list[dict], *, max_cps: float = DEFAULT_MAX_CPS) -> list[DialogueOverflow]:
    issues: list[DialogueOverflow] = []
    for sh in shots:
        text = sh.get("dialogue") or sh.get("text") or ""
        if not str(text).strip():
            continue
        dur = sh.get("duration_sec")
        if dur is None:
            start, end = sh.get("start_time"), sh.get("end_time")
            if start is not None and end is not None:
                dur = float(end) - float(start)
        issue = check_dialogue_duration(
            str(text),
            float(dur or 0),
            shot_code=str(sh.get("shot_code") or sh.get("code") or ""),
            max_cps=max_cps,
        )
        if issue:
            issues.append(issue)
    return issues
