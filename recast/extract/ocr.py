"""On-screen caption OCR. RapidOCR/ONNX if installed; otherwise a graceful skip."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from recast.ffmpeg import VideoInfo

log = logging.getLogger("recast")


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


def run(
    video: str | Path,
    workdir: Path,
    info: VideoInfo,
    shots: list[dict[str, Any]] | None = None,
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
    segments: list[dict[str, Any]] = []
    try:
        for t in times:
            check_cancel()
            frame_idx = max(0, int(round(t * fps)))
            cap.set(1, frame_idx)  # cv2.CAP_PROP_POS_FRAMES
            ok_f, frame = cap.read()
            if not ok_f or frame is None:
                continue
            try:
                result = engine(frame)
            except TypeError:
                # some versions want a path
                result = engine(frame)
            except Exception as exc:
                log.warning("OCR 帧 %s 失败: %s", frame_idx, exc)
                continue
            parts = _parse_rapidocr(result)
            if not parts:
                continue
            text = " ".join(p[0] for p in parts)
            conf = sum(p[1] for p in parts) / max(1, len(parts))
            # each sample covers a small window around the frame
            start = max(0.0, t - 0.6)
            end = t + 0.6
            segments.append({"start": start, "end": end, "text": text, "conf": conf, "frame_idx": frame_idx})
    finally:
        cap.release()
        held = engine
        engine = None
        release_cuda(held)

    # merge consecutive similar OCR hits
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if merged and seg["text"] == merged[-1]["text"] and seg["start"] <= merged[-1]["end"] + 0.8:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))

    payload = {"backend": backend, "skipped": False, "segments": merged}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(merged, srt_path)
    log.info("OCR 完成（%s）: %s 段，写出 %s", backend, len(merged), srt_path)
    return json_path, merged
