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


def shot_slot_seconds(shot: dict) -> float:
    """Visual slot used for mixing. Subtitle-only end times do not widen this."""
    start = shot.get("start_time")
    end = shot.get("end_time")
    if start is not None and end is not None and float(end) > float(start):
        return float(end) - float(start)
    dur = shot.get("duration_sec")
    if dur not in (None, ""):
        return max(0.0, float(dur))
    return 0.0


def suggest_split(text: str, slot_sec: float, audio_sec: float) -> str | None:
    """A punctuation split the user can confirm. Does not cut or speed the wav."""
    if not text or audio_sec <= 0 or slot_sec <= 0 or audio_sec <= slot_sec:
        return None
    marks = "，。！？；、,.!?;"
    if not any(ch in text for ch in marks):
        return None
    target = int(round(len(text) * min(0.85, slot_sec / audio_sec)))
    target = min(max(target, 1), len(text) - 1)
    best_at = None
    best_distance = 10**9
    for index, ch in enumerate(text):
        if ch not in marks:
            continue
        distance = abs(index - target)
        if distance < best_distance:
            best_distance = distance
            best_at = index
    if best_at is None:
        return None
    left = text[: best_at + 1].strip()
    right = text[best_at + 1 :].strip()
    if not left or not right:
        return None
    return f"{left} ｜ {right}"


def _shift_shot(shot: dict, delta: float) -> None:
    if abs(delta) < 1e-9:
        return
    for key in ("start_time", "end_time", "narration_start", "narration_end"):
        if shot.get(key) is not None:
            shot[key] = float(shot[key]) + delta


def _extend_shot_to_audio(shot: dict, audio_sec: float) -> float:
    start = float(shot.get("start_time") or shot.get("narration_start") or 0.0)
    new_end = start + audio_sec
    extra = new_end - (start + shot_slot_seconds(shot))
    shot["end_time"] = new_end
    shot["duration_sec"] = audio_sec
    narr_end = shot.get("narration_end")
    if narr_end is None or float(narr_end) < new_end:
        shot["narration_end"] = new_end
    return max(0.0, extra)


def extend_timeline_for_audio(shots: list[dict]) -> list[dict]:
    """Move later shots so a longer wav does not play over the next line.

    The wav itself is never time-stretched. Each overflow is recorded so the
    user can confirm the new rhythm or split the sentence and dub again.
    """
    confirmations: list[dict] = []
    order = sorted(
        range(len(shots)),
        key=lambda i: (
            float(shots[i].get("start_time") if shots[i].get("start_time") is not None else shots[i].get("narration_start") or i),
            i,
        ),
    )
    shift = 0.0
    for index in order:
        shot = shots[index]
        _shift_shot(shot, shift)
        audio = float(shot.get("audio_duration") or 0.0)
        if audio <= 0:
            continue
        slot = shot_slot_seconds(shot)
        if audio <= slot + 0.05:
            continue
        extra = _extend_shot_to_audio(shot, audio)
        shift += extra
        text = str(shot.get("dialogue") or shot.get("text") or "")
        code = str(shot.get("shot_code") or shot.get("code") or "（未编号镜头）")
        split = suggest_split(text, slot, audio)
        message = (
            f"镜头 {code} 配音实际 {audio:.2f} 秒，原镜头 {slot:.2f} 秒。"
            f"已自动把本镜延长 {extra:.2f} 秒，并把后面的镜头后移，没有加速音频。请确认成片节奏。"
        )
        if split:
            message += f" 也可以把这句拆成：{split}"
        shot["needs_confirm"] = True
        shot["timeline_note"] = message
        confirmations.append(
            {
                "shot_code": code,
                "action": "extend",
                "extra_sec": extra,
                "slot_sec": slot,
                "audio_sec": audio,
                "split_suggestion": split,
                "needs_confirm": True,
                "message": message,
            }
        )
    return confirmations


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
