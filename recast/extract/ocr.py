"""On-screen caption OCR. RapidOCR/ONNX if installed; otherwise a graceful skip."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from recast.ffmpeg import VideoInfo

log = logging.getLogger("recast")

# Bottom band of the frame. Signs and posters above this line are not captions.
DEFAULT_BAND_TOP = 0.72
DEFAULT_MIN_CONF = 0.5
DEFAULT_MERGE_SIMILAR = 0.72
DEFAULT_MERGE_GAP = 0.8
# Half-window around the frame that was actually decoded, not a fixed ±0.6s.
DEFAULT_WINDOW_HALF = 0.45


def ocr_available() -> tuple[bool, str]:
    try:
        import rapidocr  # noqa: F401

        return True, "rapidocr"
    except Exception:
        pass
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return True, "rapidocr_onnxruntime"
    except Exception as exc:
        return False, str(exc)


def _build_engine():
    try:
        from rapidocr import RapidOCR

        return RapidOCR(), "rapidocr"
    except Exception:
        pass
    try:
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR(), "rapidocr_onnxruntime"
    except Exception as exc:
        raise RuntimeError(
            f"未安装 RapidOCR，已跳过画面 OCR。可稍后执行: pip install rapidocr  （当前错误: {exc}）"
        ) from exc


def _sample_times(shots: list[dict[str, Any]], info: VideoInfo, extra_fps: float = 0.5) -> list[float]:
    times: list[float] = []
    for sh in shots or []:
        start = float(sh.get("start_time") or 0.0)
        end = float(sh.get("end_time") or start)
        mid = (start + end) / 2.0
        times.append(mid)
        # long shots get a couple of extra samples
        dur = max(0.0, end - start)
        if dur >= 4.0:
            times.append(start + dur * 0.25)
            times.append(start + dur * 0.75)
    if not times and info.duration > 0:
        step = 1.0 / max(extra_fps, 0.1)
        t = step / 2.0
        while t < info.duration:
            times.append(t)
            t += max(2.0, step)
    # unique, sorted
    uniq: list[float] = []
    for t in sorted(times):
        if not uniq or abs(t - uniq[-1]) > 0.2:
            uniq.append(t)
    return uniq


def _parse_rapidocr(result: Any) -> list[tuple[str, float]]:
    """Return (text, score) pairs from various RapidOCR result shapes."""
    items: list[tuple[str, float]] = []
    payload = result
    if isinstance(result, tuple) and result:
        payload = result[0]
    if payload is None:
        return items
    if hasattr(payload, "txts"):
        txts = list(getattr(payload, "txts") or [])
        scores = list(getattr(payload, "scores") or [1.0] * len(txts))
        for t, s in zip(txts, scores):
            if t:
                items.append((str(t), float(s or 0.0)))
        return items
    for row in payload or []:
        text = None
        score = 0.0
        if isinstance(row, dict):
            text = row.get("text") or row.get("txt")
            score = float(row.get("score") or row.get("confidence") or 0.0)
        elif isinstance(row, (list, tuple)):
            # typical: [box, text, score]
            if len(row) >= 3 and isinstance(row[1], str):
                text = row[1]
                try:
                    score = float(row[2])
                except (TypeError, ValueError):
                    score = 0.0
            elif len(row) >= 2 and isinstance(row[1], str):
                text = row[1]
        if text and str(text).strip():
            items.append((str(text).strip(), score))
    return items


def subtitle_band(frame: Any, *, top_ratio: float = DEFAULT_BAND_TOP) -> Any:
    """Crop to the bottom subtitle band. ``top_ratio`` is the first row kept."""
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim < 2 or arr.shape[0] <= 1:
        return arr
    ratio = min(max(float(top_ratio), 0.0), 0.95)
    y0 = int(arr.shape[0] * ratio)
    if y0 >= arr.shape[0]:
        y0 = arr.shape[0] - 1
    return arr[y0:]


def filter_ocr_parts(
    parts: list[tuple[str, float]],
    *,
    min_conf: float = DEFAULT_MIN_CONF,
) -> list[tuple[str, float]]:
    kept: list[tuple[str, float]] = []
    for text, score in parts:
        if not str(text).strip():
            continue
        try:
            conf = float(score)
        except (TypeError, ValueError):
            continue
        if conf < min_conf:
            continue
        kept.append((str(text).strip(), conf))
    return kept


def plan_decoded_frames(
    sample_times: list[float],
    *,
    fps: float,
    frame_count: int,
) -> list[tuple[int, float, float]]:
    """Walk frames forward and pick the first decoded frame at or after each sample.

    Returns ``(frame_index, actual_time, requested_time)``. The index never moves
    backward. A sample past the last frame is dropped instead of seeking.
    """
    if frame_count <= 0:
        return []
    rate = fps if fps and fps > 0 else 25.0
    frame_idx = 0
    chosen: list[tuple[int, float, float]] = []
    for requested in sample_times:
        while frame_idx < frame_count - 1 and (frame_idx / rate) + 1e-9 < requested:
            frame_idx += 1
        actual = frame_idx / rate
        if actual + 1e-9 < requested:
            continue
        chosen.append((frame_idx, actual, float(requested)))
    return chosen


def windows_for_hits(
    hits: list[dict[str, Any]],
    *,
    max_half: float = DEFAULT_WINDOW_HALF,
) -> list[dict[str, Any]]:
    """Time each hit from the decoded frame, halfway to its neighbors, capped."""
    out: list[dict[str, Any]] = []
    for i, hit in enumerate(hits):
        row = dict(hit)
        t = float(hit.get("time", hit.get("start") or 0.0))
        prev_t = float(hits[i - 1].get("time", hits[i - 1].get("start") or 0.0)) if i else None
        next_t = float(hits[i + 1]["time"]) if i + 1 < len(hits) and hits[i + 1].get("time") is not None else None
        if next_t is None and i + 1 < len(hits):
            next_t = float(hits[i + 1].get("start") or t)
        back = max_half if prev_t is None else min(max_half, max(0.0, (t - prev_t) / 2.0))
        fwd = max_half if next_t is None else min(max_half, max(0.0, (next_t - t) / 2.0))
        row["time"] = t
        row["start"] = max(0.0, t - back)
        row["end"] = t + max(fwd, 0.05)
        out.append(row)
    return out


def merge_ocr_hits(
    segments: list[dict[str, Any]],
    *,
    similar: float = DEFAULT_MERGE_SIMILAR,
    gap: float = DEFAULT_MERGE_GAP,
) -> list[dict[str, Any]]:
    """Merge consecutive hits by sequence similarity, not exact text equality."""
    from recast.extract.merge_subs import text_similarity

    merged: list[dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        row = dict(seg)
        row["text"] = text
        if not merged:
            merged.append(row)
            continue
        prev = merged[-1]
        close = float(row.get("start") or 0.0) <= float(prev.get("end") or 0.0) + gap
        if close and text_similarity(prev.get("text") or "", row["text"]) >= similar:
            prev["end"] = max(float(prev.get("end") or 0.0), float(row.get("end") or 0.0))
            if float(row.get("conf") or 0.0) > float(prev.get("conf") or 0.0):
                prev["text"] = row["text"]
                prev["conf"] = row.get("conf")
        else:
            merged.append(row)
    return merged


def _ocr_band(engine: Any, frame: Any, *, band_top: float, min_conf: float) -> tuple[str, float] | None:
    band = subtitle_band(frame, top_ratio=band_top)
    try:
        result = engine(band)
    except Exception as exc:
        log.warning("OCR 字幕带失败: %s", exc)
        return None
    parts = filter_ocr_parts(_parse_rapidocr(result), min_conf=min_conf)
    if not parts:
        return None
    text = " ".join(text for text, _score in parts)
    conf = sum(score for _text, score in parts) / len(parts)
    return text, conf


def run(
    video: str | Path,
    workdir: Path,
    info: VideoInfo,
    shots: list[dict[str, Any]] | None = None,
    *,
    band_top: float = DEFAULT_BAND_TOP,
    min_conf: float = DEFAULT_MIN_CONF,
    merge_similar: float = DEFAULT_MERGE_SIMILAR,
    merge_gap: float = DEFAULT_MERGE_GAP,
    window_half: float = DEFAULT_WINDOW_HALF,
) -> tuple[Path, list[dict[str, Any]]]:
    """OCR sampled frames and write ocr.json / ocr.srt. Always writes files."""
    from recast.device import release_cuda
    from recast.media import open_capture
    from recast.runtime import check_cancel
    from recast.stages.subtitles import write_srt

    workdir.mkdir(parents=True, exist_ok=True)
    json_path = workdir / "ocr.json"
    srt_path = workdir / "ocr.srt"
    ok, reason = ocr_available()
    if not ok:
        log.warning("未安装 RapidOCR，跳过画面 OCR（%s）。字幕将仅使用语音转写。", reason)
        payload = {"backend": None, "skipped": True, "reason": reason, "segments": []}
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_srt([], srt_path)
        return json_path, []

    engine, backend = _build_engine()
    times = _sample_times(list(shots or []), info)
    cap = open_capture(video)
    fps = info.fps or 25.0
    if fps <= 0:
        fps = 25.0
    pending = list(times)
    hits: list[dict[str, Any]] = []
    frame_idx = 0
    try:
        while pending:
            check_cancel()
            ok_f, frame = cap.read()
            if not ok_f or frame is None:
                break
            actual = frame_idx / fps
            due = False
            while pending and actual + 1e-9 >= pending[0]:
                pending.pop(0)
                due = True
            if due:
                recognized = _ocr_band(engine, frame, band_top=band_top, min_conf=min_conf)
                if recognized:
                    text, conf = recognized
                    hits.append(
                        {
                            "time": actual,
                            "text": text,
                            "conf": conf,
                            "frame_idx": frame_idx,
                        }
                    )
            frame_idx += 1
    finally:
        cap.release()
        held = engine
        engine = None
        release_cuda(held)

    timed = windows_for_hits(hits, max_half=window_half)
    merged = merge_ocr_hits(timed, similar=merge_similar, gap=merge_gap)

    payload = {
        "backend": backend,
        "skipped": False,
        "band_top": band_top,
        "min_conf": min_conf,
        "segments": merged,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(merged, srt_path)
    log.info("OCR 完成（%s）: %s 段，写出 %s", backend, len(merged), srt_path)
    return json_path, merged
