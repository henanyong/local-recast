"""Align OCR (burned-in captions) with ASR (speech) into one subtitle track."""

from __future__ import annotations

import difflib
import math
import re
from typing import Any, Iterable

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[，。！？、,.!?;；：:“”\"'（）()【】\[\]…—\-]")


def _norm(text: str) -> str:
    text = _PUNCT.sub("", text or "")
    text = _WS.sub("", text)
    return text.strip().lower()


def text_similarity(a: str, b: str) -> float:
    """Sequence-aware similarity in 0..1. Order matters; a shared character bag does not."""
    left, right = _norm(a), _norm(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0.0, hi - lo)


def _duration(start: float, end: float) -> float:
    return max(0.0, end - start)


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = _overlap(a_start, a_end, b_start, b_end)
    union = _duration(a_start, a_end) + _duration(b_start, b_end) - inter
    return inter / union if union > 0 else 0.0


def confidence_score(seg: dict[str, Any]) -> float | None:
    """Map ASR log-prob and OCR 0..1 scores onto one 0..1 scale.

    ``avg_logprob`` (faster-whisper) becomes ``exp(avg_logprob)``. An explicit
    ``conf`` in 0..1 is used as-is. A negative ``conf`` is treated as a log-prob.
    Missing scores stay unknown so a tie can fall back to the ASR line.
    """
    logprob = seg.get("avg_logprob")
    if logprob is not None:
        try:
            return _clamp01(math.exp(float(logprob)))
        except (TypeError, ValueError, OverflowError):
            pass
    raw = seg.get("conf")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        try:
            return _clamp01(math.exp(value))
        except OverflowError:
            return 0.0
    if value > 1.0:
        return _clamp01(value / 100.0)
    return value


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _as_seg(item: dict[str, Any], source: str) -> dict[str, Any]:
    start = float(item.get("start", item.get("start_time") or 0.0))
    end = float(item.get("end", item.get("end_time") or start))
    text = str(item.get("text") or "").strip()
    conf = item.get("conf")
    logprob = item.get("avg_logprob")
    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None
    try:
        log_f = float(logprob) if logprob is not None else None
    except (TypeError, ValueError):
        log_f = None
    # Older callers stored avg_logprob in conf and left avg_logprob empty.
    if log_f is None and conf_f is not None and conf_f < 0:
        log_f = conf_f
        conf_f = None
    return {
        "start": start,
        "end": max(end, start),
        "text": text,
        "source": source,
        "conf": conf_f,
        "avg_logprob": log_f,
    }


def _ocr_is_better(ocr_seg: dict[str, Any], asr_seg: dict[str, Any]) -> bool:
    """True only when OCR's confidence is strictly higher. Ties keep ASR."""
    ocr_score = confidence_score(ocr_seg)
    asr_score = confidence_score(asr_seg)
    if ocr_score is None:
        return False
    if asr_score is None:
        return True
    return ocr_score > asr_score


def merge_subtitles(
    asr: Iterable[dict[str, Any]] | None,
    ocr: Iterable[dict[str, Any]] | None,
    *,
    iou_threshold: float = 0.3,
    similar: float = 0.72,
) -> list[dict[str, Any]]:
    """ASR is the base speech track.

    OCR is added when it does not overlap an ASR line, or the texts are not the
    same sequence. When they overlap in time and the text is the same line,
    the higher confidence wins. Equal confidence keeps ASR.
    """
    asr_segs = [_as_seg(s, "asr") for s in (asr or []) if str(s.get("text") or "").strip()]
    ocr_segs = [_as_seg(s, "ocr") for s in (ocr or []) if str(s.get("text") or "").strip()]
    kept: list[dict[str, Any]] = [dict(s) for s in asr_segs]

    for ocr_seg in ocr_segs:
        if not _norm(ocr_seg["text"]):
            continue
        best_i = -1
        best_iou = 0.0
        for i, asr_seg in enumerate(kept):
            iou = _iou(ocr_seg["start"], ocr_seg["end"], asr_seg["start"], asr_seg["end"])
            if iou < iou_threshold or iou < best_iou:
                continue
            if text_similarity(ocr_seg["text"], asr_seg["text"]) < similar:
                continue
            best_i = i
            best_iou = iou
        if best_i < 0:
            kept.append(dict(ocr_seg))
            continue
        if _ocr_is_better(ocr_seg, kept[best_i]):
            chosen = dict(ocr_seg)
            chosen["source"] = "ocr"
            kept[best_i] = chosen

    kept.sort(key=lambda s: (s["start"], s["end"], s["text"]))
    for i, seg in enumerate(kept):
        seg["id"] = i
    return kept
