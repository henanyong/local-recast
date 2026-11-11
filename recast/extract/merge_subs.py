"""Align OCR (burned-in captions) with ASR (speech) into one subtitle track."""

from __future__ import annotations

import re
from typing import Any, Iterable

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[，。！？、,.!?;；：:“”\"'（）()【】\[\]…—\-]")


def _norm(text: str) -> str:
    text = _PUNCT.sub("", text or "")
    text = _WS.sub("", text)
    return text.strip().lower()


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


def _as_seg(item: dict[str, Any], source: str) -> dict[str, Any]:
    start = float(item.get("start", item.get("start_time") or 0.0))
    end = float(item.get("end", item.get("end_time") or start))
    text = str(item.get("text") or "").strip()
    conf = item.get("conf", item.get("avg_logprob"))
    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None
    return {"start": start, "end": max(end, start), "text": text, "source": source, "conf": conf_f}


def merge_subtitles(
    asr: Iterable[dict[str, Any]] | None,
    ocr: Iterable[dict[str, Any]] | None,
    *,
    iou_threshold: float = 0.3,
    similar: float = 0.72,
) -> list[dict[str, Any]]:
    """ASR is the base speech track. OCR is added when it fills a gap or carries
    clearly different burned-in text (titles, location cards, etc.).
    """
    asr_segs = [_as_seg(s, "asr") for s in (asr or []) if str(s.get("text") or "").strip()]
    ocr_segs = [_as_seg(s, "ocr") for s in (ocr or []) if str(s.get("text") or "").strip()]
    kept: list[dict[str, Any]] = [dict(s, source="asr") for s in asr_segs]

    for ocr_seg in ocr_segs:
        otext = _norm(ocr_seg["text"])
        if not otext:
            continue
        conflict = False
        for asr_seg in asr_segs:
            iou = _iou(ocr_seg["start"], ocr_seg["end"], asr_seg["start"], asr_seg["end"])
            if iou < iou_threshold:
                continue
            atext = _norm(asr_seg["text"])
            if not atext:
                continue
            # character-level overlap ratio (Chinese-friendly, no tokenizer)
            shorter, longer = (otext, atext) if len(otext) <= len(atext) else (atext, otext)
            ratio = (sum(1 for ch in shorter if ch in longer) / len(shorter)) if shorter else 0.0
            if ratio >= similar:
                conflict = True
                break
            # heavily overlapping in time and reasonably similar → treat as duplicate
            if iou >= 0.6 and ratio >= 0.4:
                conflict = True
                break
        if not conflict:
            kept.append(dict(ocr_seg, source="ocr"))

    kept.sort(key=lambda s: (s["start"], s["end"], s["text"]))
    # re-index
    for i, seg in enumerate(kept):
        seg["id"] = i
    return kept
