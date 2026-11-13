"""Per-character voice timbre: pitch / rate / style.

Timbre lives in the character folder's meta.json ``voice.timbre``:

    {"pitch": 0.0, "rate": 1.0, "style": ""}

``pitch`` is semitones relative to the voice default (-12..+12).
``rate`` is a speed multiplier (0.5..2.0); audio is never sped up to fit a
slot — rate only changes how fast the line is *spoken*.
``style`` is a free-form hint (e.g. "冷淡" / "温柔"); engines that do not
understand it ignore it and say so in the result note.

How each engine applies it (see recast/generate/tts.py):

- edge-tts: SSML prosody via Communicate(rate=, pitch=). Pitch goes as
  semitones ("+2.0st"), rate as percent ("+10%").
- Piper: ``--length-scale`` is the inverse of the speaking rate
  (length-scale 0.5 = twice as fast). Piper has no pitch control.
- Reference-wav clone (RECAST_TTS_CLONE): the external cloner gets no
  timbre parameters; cloning keeps the reference voice as-is.
- pyttsx3: speaking rate in words per minute (default 200). No pitch.
"""

from __future__ import annotations

from typing import Any

PITCH_MIN = -12.0
PITCH_MAX = 12.0
RATE_MIN = 0.5
RATE_MAX = 2.0
STYLE_MAX_LEN = 40
DEFAULT_WPM = 200
DEFAULT_TIMBRE: dict[str, Any] = {"pitch": 0.0, "rate": 1.0, "style": ""}


def normalize_timbre(raw: Any) -> dict[str, Any]:
    """Clamp a timbre dict to the supported ranges. Bad input becomes default."""
    out = dict(DEFAULT_TIMBRE)
    if not isinstance(raw, dict):
        return out
    try:
        pitch = float(raw.get("pitch", 0.0))
    except (TypeError, ValueError):
        pitch = 0.0
    try:
        rate = float(raw.get("rate", 1.0))
    except (TypeError, ValueError):
        rate = 1.0
    out["pitch"] = max(PITCH_MIN, min(PITCH_MAX, pitch))
    out["rate"] = max(RATE_MIN, min(RATE_MAX, rate))
    style = str(raw.get("style") or "").strip()
    out["style"] = style[:STYLE_MAX_LEN]
    return out


def is_default(timbre: dict[str, Any] | None) -> bool:
    """True when nothing was changed from the voice default."""
    t = normalize_timbre(timbre)
    return t["pitch"] == 0.0 and t["rate"] == 1.0 and not t["style"]


def edge_tts_kwargs(timbre: dict[str, Any] | None) -> dict[str, str]:
    """rate/pitch kwargs for edge_tts.Communicate (empty when default)."""
    t = normalize_timbre(timbre)
    kwargs: dict[str, str] = {}
    if t["rate"] != 1.0:
        pct = round((t["rate"] - 1.0) * 100)
        kwargs["rate"] = f"{pct:+d}%"
    if t["pitch"] != 0.0:
        kwargs["pitch"] = f"{t['pitch']:+.1f}st"
    return kwargs


def piper_length_scale(rate: float) -> float:
    """Piper --length-scale for a speaking-rate multiplier (inverse)."""
    try:
        r = float(rate)
    except (TypeError, ValueError):
        r = 1.0
    r = max(RATE_MIN, min(RATE_MAX, r))
    return max(0.25, min(4.0, 1.0 / r))


def pyttsx3_wpm(rate: float) -> int:
    """pyttsx3 speaking rate (words per minute) for a rate multiplier."""
    try:
        r = float(rate)
    except (TypeError, ValueError):
        r = 1.0
    r = max(RATE_MIN, min(RATE_MAX, r))
    return max(50, min(400, int(round(DEFAULT_WPM * r))))


def timbre_note(timbre: dict[str, Any] | None, engine: str) -> str:
    """What a given engine does with this timbre (for the result message)."""
    t = normalize_timbre(timbre)
    if is_default(t):
        return ""
    if engine == "edge-tts":
        bits = []
        if t["rate"] != 1.0:
            bits.append(f"语速 {t['rate']}x")
        if t["pitch"] != 0.0:
            bits.append(f"音调 {t['pitch']:+.1f} 半音")
        if t["style"]:
            bits.append(f"风格「{t['style']}」未应用（edge-tts 不支持该风格参数）")
        return "已按音色合成：" + "、".join(bits) + "。" if bits else ""
    if engine == "piper":
        bits = []
        if t["rate"] != 1.0:
            bits.append(f"语速 {t['rate']}x")
        if t["pitch"] != 0.0:
            bits.append("音调未应用（Piper 不支持调音调）")
        if t["style"]:
            bits.append(f"风格「{t['style']}」未应用（Piper 不支持风格参数）")
        return "已按音色合成：" + "、".join(bits) + "。" if bits else ""
    if engine == "local-clone":
        return "克隆音色不支持调参，已用参考 wav 的原音色合成。"
    if engine == "pyttsx3":
        bits = []
        if t["rate"] != 1.0:
            bits.append(f"语速 {t['rate']}x")
        if t["pitch"] != 0.0:
            bits.append("音调未应用（pyttsx3 不支持调音调）")
        if t["style"]:
            bits.append(f"风格「{t['style']}」未应用")
        return "已按音色合成：" + "、".join(bits) + "。" if bits else ""
    return ""
