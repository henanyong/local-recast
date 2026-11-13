"""Stage 2: faster-whisper transcription -> SRT + timestamped JSON."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from recast.config import RecastConfig
from recast.device import release_cuda, requested_device, vram_report
from recast.runtime import check_cancel

log = logging.getLogger("recast")


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list[dict[str, Any]], path: Path) -> Path:
    lines: list[str] = []
    index = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        index += 1
        lines.append(str(index))
        lines.append(f"{_fmt_ts(float(seg['start']))} --> {_fmt_ts(float(seg['end']))}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 字幕编辑器（第 ⑤ 步“声音与台词”用；对话框见 recast/desktop/subtitle_dialog.py）
#
# 数据形状与流水线其余部分一致：{"start": float, "end": float, "text": str}，
# 其余键（id / words 等）原样保留。所有编辑操作都是纯函数：返回新列表，
# 不改原列表；非法操作抛 ValueError（中文原因），调用方直接显示给用户。
# ---------------------------------------------------------------------------

MAX_CHARS_PER_SEC = 4.0
"""字幕语速上限（字/秒），与台词溢出校验（dialogue.py）同一把尺。"""


def _parse_ts(text: str) -> float:
    """Parse an SRT timestamp ("00:00:01,234" or "00:00:01.234") to seconds."""
    raw = str(text or "").strip().replace(".", ",")
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"时间戳格式不对：{text!r}，应为 时:分:秒,毫秒，例如 00:00:01,234")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_ms = parts[2].split(",")
        if len(sec_ms) != 2:
            raise ValueError
        seconds = int(sec_ms[0])
        millis = int(sec_ms[1])
    except ValueError:
        raise ValueError(f"时间戳格式不对：{text!r}，应为 时:分:秒,毫秒，例如 00:00:01,234") from None
    if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60 or millis < 0 or millis >= 1000:
        raise ValueError(f"时间戳数值不合法：{text!r}")
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_srt(source: str | Path) -> list[dict[str, Any]]:
    """Read SRT text (or a path to an .srt file) into segment dicts.

    Raises ValueError with a Chinese reason on malformed input, or
    FileNotFoundError when ``source`` looks like a path that does not exist.
    """
    path = Path(source)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    elif isinstance(source, str) and "\n" not in source and source.strip().lower().endswith(".srt"):
        raise FileNotFoundError(f"字幕文件不存在：{source}")
    else:
        text = str(source)
    blocks = [b for b in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n") if b.strip()]
    segments: list[dict[str, Any]] = []
    for block_no, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 2:
            raise ValueError(f"SRT 第 {block_no} 段不完整（只有 {len(lines)} 行）")
        ts_idx = 0 if "-->" in lines[0] else 1
        if ts_idx >= len(lines) or "-->" not in lines[ts_idx]:
            raise ValueError(f"SRT 第 {block_no} 段没有时间戳行（应为 开始 --> 结束）")
        try:
            start_raw, end_raw = [p.strip() for p in lines[ts_idx].split("-->")]
            start = _parse_ts(start_raw)
            end = _parse_ts(end_raw)
        except ValueError as exc:
            raise ValueError(f"SRT 第 {block_no} 段时间戳不对：{exc}") from None
        body = "\n".join(lines[ts_idx + 1:]).strip()
        segments.append({"start": start, "end": end, "text": body})
    return segments


def validate_segments(segments: list[dict[str, Any]]) -> list[str]:
    """Hard legality checks. Returns Chinese reasons; empty means OK to save."""
    errors: list[str] = []
    rows: list[tuple[int, float, float]] = []
    for idx, seg in enumerate(segments):
        line = idx + 1
        text = str(seg.get("text") or "").strip()
        if not text:
            errors.append(f"第 {line} 行文字为空")
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
        except (TypeError, ValueError):
            errors.append(f"第 {line} 行时间不是数字")
            continue
        if start < 0:
            errors.append(f"第 {line} 行开始时间不能为负（{start:g} 秒）")
        if end <= start:
            errors.append(f"第 {line} 行结束时间必须晚于开始时间（{_fmt_ts(start)} --> {_fmt_ts(end)}）")
        rows.append((line, start, end))
    for idx in range(1, len(rows)):
        if rows[idx][1] < rows[idx - 1][1] - 1e-3:
            errors.append(f"第 {rows[idx][0]} 行开始时间比上一行早，顺序不对")
    ordered = sorted(rows, key=lambda r: r[1])
    for (la, _sa, ea), (lb, sb, _eb) in zip(ordered, ordered[1:]):
        if ea > sb + 1e-3:
            errors.append(f"第 {la} 行与第 {lb} 行时间重叠（第 {la} 行到 {_fmt_ts(ea)}，第 {lb} 行从 {_fmt_ts(sb)} 开始）")
    return errors


def save_segments(
    segments: list[dict[str, Any]],
    srt_path: str | Path,
    json_path: str | Path | None = None,
) -> Path:
    """Validate, then rewrite the SRT (and subtitles.json when given).

    Raises ValueError with the Chinese reasons when validation fails;
    in that case nothing is written.
    """
    errors = validate_segments(segments)
    if errors:
        raise ValueError("字幕不合法，拒绝保存：\n" + "\n".join(errors))
    srt = write_srt(segments, Path(srt_path))
    if json_path is not None:
        jp = Path(json_path)
        payload: dict[str, Any] = {"language": None, "segments": []}
        if jp.is_file():
            try:
                loaded = json.loads(jp.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload.update(loaded)
            except (ValueError, OSError):
                pass
        payload["segments"] = [dict(seg) for seg in segments]
        jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return srt


def _segment_at(segments: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not 0 <= index < len(segments):
        raise IndexError(f"没有第 {index + 1} 行字幕（共 {len(segments)} 行）")
    return segments[index]


def _clean_copy(seg: dict[str, Any]) -> dict[str, Any]:
    """Copy a segment, dropping keys that stop being true after an edit."""
    row = dict(seg)
    row.pop("id", None)
    row.pop("words", None)
    return row


def edit_segment_text(
    segments: list[dict[str, Any]], index: int, new_text: str
) -> list[dict[str, Any]]:
    """Replace one line's text. Raises ValueError when the new text is empty."""
    _segment_at(segments, index)
    text = str(new_text or "").strip()
    if not text:
        raise ValueError(f"第 {index + 1} 行文字不能为空")
    out = [dict(s) for s in segments]
    out[index]["text"] = text
    return out


def edit_segment_times(
    segments: list[dict[str, Any]], index: int, start: float, end: float
) -> list[dict[str, Any]]:
    """Replace one line's start/end. Raises ValueError on illegal times."""
    _segment_at(segments, index)
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        raise ValueError(f"第 {index + 1} 行时间必须是数字") from None
    if start_f < 0:
        raise ValueError(f"第 {index + 1} 行开始时间不能为负")
    if end_f <= start_f:
        raise ValueError(f"第 {index + 1} 行结束时间必须晚于开始时间")
    out = [dict(s) for s in segments]
    out[index]["start"] = start_f
    out[index]["end"] = end_f
    return out


def split_segment(
    segments: list[dict[str, Any]], index: int, split_at: float | None = None
) -> list[dict[str, Any]]:
    """Split one line into two.

    Without ``split_at`` the text is cut in half by character count and the
    time is split proportionally. With ``split_at`` (seconds, inside the
    line's range) the text is cut at the matching character position.
    """
    row = _segment_at(segments, index)
    text = str(row.get("text") or "").strip()
    start = float(row.get("start"))
    end = float(row.get("end"))
    if len(text) < 2:
        raise ValueError(f"第 {index + 1} 行只有一个字，无法拆分")
    if split_at is None:
        half = max(1, len(text) // 2)
        boundary = start + (end - start) * half / len(text)
        first_text, second_text = text[:half], text[half:]
    else:
        try:
            boundary = float(split_at)
        except (TypeError, ValueError):
            raise ValueError(f"拆分点必须是数字（第 {index + 1} 行）") from None
        if not start < boundary < end:
            raise ValueError(f"拆分点必须在第 {index + 1} 行的时间范围内（{_fmt_ts(start)} --> {_fmt_ts(end)}）")
        cut = max(1, min(len(text) - 1, round(len(text) * (boundary - start) / (end - start))))
        first_text, second_text = text[:cut], text[cut:]
    first = _clean_copy(row)
    first.update({"start": start, "end": boundary, "text": first_text})
    second = _clean_copy(row)
    second.update({"start": boundary, "end": end, "text": second_text})
    out = [dict(s) for s in segments]
    out[index:index + 1] = [first, second]
    return out


def _join_text(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    import re

    cjk = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
    if cjk.search(a[-1]) or cjk.search(b[0]):
        return a + b
    return a + " " + b


def merge_segments(segments: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    """Merge line ``index`` with the next line. Raises on the last line."""
    first = _segment_at(segments, index)
    if index + 1 >= len(segments):
        raise IndexError(f"第 {index + 1} 行是最后一行，没有下一行可以合并")
    second = segments[index + 1]
    merged = _clean_copy(first)
    merged.update(
        {
            "start": min(float(first["start"]), float(second["start"])),
            "end": max(float(first["end"]), float(second["end"])),
            "text": _join_text(
                str(first.get("text") or "").strip(), str(second.get("text") or "").strip()
            ),
        }
    )
    out = [dict(s) for s in segments]
    out[index:index + 2] = [merged]
    return out


def delete_segment(segments: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    """Remove one line."""
    _segment_at(segments, index)
    return [dict(s) for i, s in enumerate(segments) if i != index]


def insert_segment(
    segments: list[dict[str, Any]], index: int, start: float, end: float, text: str
) -> list[dict[str, Any]]:
    """Insert a new line before ``index`` (``index == len(segments)`` appends)."""
    if not 0 <= index <= len(segments):
        raise IndexError(f"插入位置超出范围：{index + 1}（共 {len(segments)} 行）")
    body = str(text or "").strip()
    if not body:
        raise ValueError("插入的字幕文字不能为空")
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        raise ValueError("插入的字幕时间必须是数字") from None
    if start_f < 0:
        raise ValueError("插入的字幕开始时间不能为负")
    if end_f <= start_f:
        raise ValueError("插入的字幕结束时间必须晚于开始时间")
    out = [dict(s) for s in segments]
    out.insert(index, {"start": start_f, "end": end_f, "text": body})
    return out


def check_subtitle_timeline(
    segments: list[dict[str, Any]],
    shots: list[dict[str, Any]] | None = None,
    *,
    max_chars_per_sec: float = MAX_CHARS_PER_SEC,
) -> list[dict[str, Any]]:
    """Warning-level timeline check (does not block saving).

    Returns [{"line", "kind", "reason"}] for three problem kinds:
    "overlap" (two lines overlap), "too_long" (faster than
    ``max_chars_per_sec``), "shot_boundary" (a line spans a cut point).
    Shot dicts use the same keys as ``snap_segments_to_shots``:
    "start_time" / "end_time".
    """
    issues: list[dict[str, Any]] = []
    rows: list[tuple[int, float, float, str]] = []
    for idx, seg in enumerate(segments):
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text") or "").strip()
        rows.append((idx + 1, start, end, text))
        dur = end - start
        if dur > 0 and text:
            rate = len(text) / dur
            if rate > max_chars_per_sec:
                issues.append(
                    {
                        "line": idx + 1,
                        "kind": "too_long",
                        "reason": (
                            f"第 {idx + 1} 行语速过快：每秒 {rate:.1f} 字"
                            f"（超过 {max_chars_per_sec:g} 字/秒），建议拆行或缩短文字"
                        ),
                    }
                )
    ordered = sorted(rows, key=lambda r: r[1])
    for (la, _sa, ea, _ta), (lb, sb, _eb, _tb) in zip(ordered, ordered[1:]):
        if ea > sb + 1e-3:
            issues.append(
                {
                    "line": lb,
                    "kind": "overlap",
                    "reason": f"第 {la} 行与第 {lb} 行时间重叠（第 {la} 行到 {_fmt_ts(ea)}，第 {lb} 行从 {_fmt_ts(sb)} 开始）",
                }
            )
    boundaries: list[float] = []
    for shot in shots or []:
        for key in ("start_time", "end_time"):
            if shot.get(key) is not None:
                try:
                    boundaries.append(float(shot[key]))
                except (TypeError, ValueError):
                    continue
    for line, start, end, _text in rows:
        for boundary in boundaries:
            if start + 1e-3 < boundary < end - 1e-3:
                issues.append(
                    {
                        "line": line,
                        "kind": "shot_boundary",
                        "reason": f"第 {line} 行横跨切镜点（{boundary:.2f} 秒），建议按镜头拆分",
                    }
                )
                break
    return sorted(issues, key=lambda i: (i["line"], i["kind"]))


def detect_burned_subtitles(
    video: str | Path,
    *,
    band_top: float = 0.85,
    min_conf: float = 0.6,
    sample_count: int = 3,
    ocr_fn: Any | None = None,
) -> bool | None:
    """Detect burned-in subtitles in the bottom band of a video.

    Samples ``sample_count`` frames across the duration and OCRs the bottom
    ``(1 - band_top)`` of each frame (default bottom 15%). Returns True
    (found), False (checked, none found), or None (cannot tell: OCR
    unavailable or no frame could be read). ``ocr_fn`` takes a BGR band
    image and returns the recognised text (or None); tests inject a stub,
    otherwise the OCR stage's own band recogniser is used.
    """
    from recast.ffmpeg import probe_video
    from recast.media import open_capture

    try:
        info = probe_video(video)
    except Exception as exc:
        log.warning("无法探测视频，跳过烧录字幕检测：%s", exc)
        return None
    duration = float(info.duration or 0.0)
    if duration <= 0:
        return None

    if ocr_fn is None:
        try:
            from recast.extract import ocr as ocr_mod
        except Exception as exc:
            log.warning("无法导入 OCR 模块，跳过烧录字幕检测：%s", exc)
            return None
        ok, reason = ocr_mod.ocr_available()
        if not ok:
            log.warning("OCR 不可用，跳过烧录字幕检测（%s）。", reason)
            return None
        try:
            engine, _backend = ocr_mod._build_engine()
        except Exception as exc:
            log.warning("OCR 引擎加载失败，跳过烧录字幕检测：%s", exc)
            return None

        def ocr_fn(band: Any, _engine: Any = engine, _ocr_mod: Any = ocr_mod) -> str | None:
            hit = _ocr_mod._ocr_band(_engine, band, band_top=band_top, min_conf=min_conf)
            return hit[0] if hit else None

    fractions = [(i + 0.5) / max(1, sample_count) for i in range(max(1, sample_count))]
    times = sorted({min(max(duration * f, 0.5), max(duration - 0.5, 0.5)) for f in fractions})
    try:
        cap = open_capture(video)
    except Exception as exc:
        log.warning("无法打开视频，跳过烧录字幕检测：%s", exc)
        return None
    found = False
    read_any = False
    try:
        import cv2

        from recast.extract.ocr import subtitle_band

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok_f, frame = cap.read()
            if not ok_f or frame is None:
                continue
            read_any = True
            try:
                band = subtitle_band(frame, top_ratio=band_top)
            except Exception:
                band = frame
            try:
                text = ocr_fn(band)
            except Exception as exc:
                log.warning("OCR 识别失败，跳过该帧：%s", exc)
                continue
            if text and str(text).strip():
                log.info("在 %.2f 秒处检测到底部烧录字幕：%s", t, str(text).strip()[:40])
                found = True
                break
    finally:
        cap.release()
    if not read_any:
        return None
    return found


def _load_model_with_timeout(model_cls: Any, model_name: str, model_kwargs: dict[str, Any], timeout_sec: float) -> Any:
    """Construct a WhisperModel, giving up after ``timeout_sec`` seconds.

    faster-whisper / huggingface_hub expose no total-download timeout, so the
    download runs in a daemon thread and the caller stops waiting at the cap.
    The thread itself cannot be killed, but the stage fails fast with a clear
    error instead of hanging forever on an unreachable huggingface.co.
    """
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["model"] = model_cls(model_name, **model_kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised on the caller thread
            outcome["error"] = exc

    thread = threading.Thread(target=_target, name="whisper-download", daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, float(timeout_sec)))
    if thread.is_alive():
        raise TimeoutError(f"下载 Whisper 模型超时（超过 {timeout_sec:g} 秒未完成）")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["model"]


def build_initial_prompt(names: Any) -> str | None:
    """Short name list for Whisper. Empty when the roster has no names."""
    seen: list[str] = []
    have: set[str] = set()
    for raw in names or []:
        name = str(raw).strip()
        if not name or name in have:
            continue
        have.add(name)
        seen.append(name)
    if not seen:
        return None
    return ("人名：" + "、".join(seen))[:224]


def filter_no_speech(
    segments: list[dict[str, Any]],
    *,
    max_no_speech_prob: float = 0.6,
) -> list[dict[str, Any]]:
    """Drop segments Whisper itself marks as non-speech. ``> threshold``, so 0.6 stays."""
    kept: list[dict[str, Any]] = []
    for seg in segments:
        try:
            prob = float(seg.get("no_speech_prob") or 0.0)
        except (TypeError, ValueError):
            prob = 0.0
        if prob > max_no_speech_prob:
            continue
        kept.append(seg)
    return kept


def _word_span(seg: dict[str, Any]) -> tuple[float, float] | None:
    spans: list[tuple[float, float]] = []
    for word in seg.get("words") or []:
        if not str(word.get("word") or "").strip():
            continue
        try:
            start = float(word.get("start") or 0.0)
            end = float(word.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        if end > start:
            spans.append((start, end))
    if not spans:
        return None
    return min(start for start, _end in spans), max(end for _start, end in spans)


def _nearest_boundary(moment: float, boundaries: list[float], max_shift: float) -> float:
    best: float | None = None
    best_d = float(max_shift)
    for raw in boundaries:
        boundary = float(raw)
        distance = abs(boundary - moment)
        if distance <= best_d:
            best_d = distance
            best = boundary
    return moment if best is None else best


def snap_segments_to_shots(
    segments: list[dict[str, Any]],
    shots: list[dict[str, Any]] | None,
    *,
    max_shift: float = 0.35,
) -> list[dict[str, Any]]:
    """Use word timestamps to tighten a line, then nudge edges onto a nearby cut.

    Segments without word times are left alone. This does not change playback speed.
    """
    boundaries: list[float] = []
    for shot in shots or []:
        for key in ("start_time", "end_time"):
            if shot.get(key) is not None:
                boundaries.append(float(shot[key]))
    out: list[dict[str, Any]] = []
    for seg in segments:
        row = dict(seg)
        span = _word_span(row)
        if span is None:
            out.append(row)
            continue
        word_start, word_end = span
        try:
            original = float(row.get("end") or 0.0) - float(row.get("start") or 0.0)
        except (TypeError, ValueError):
            original = 0.0
        word_dur = word_end - word_start
        if original > 0 and abs(original - word_dur) / original > 0.35:
            row["duration_note"] = "已按词级时间收紧"
        start = _nearest_boundary(word_start, boundaries, max_shift)
        end = _nearest_boundary(word_end, boundaries, max_shift)
        if end <= start + 0.05:
            start, end = word_start, word_end
        row["start"] = start
        row["end"] = end
        out.append(row)
    return out


def run(
    audio_wav: Path | None,
    workdir: Path,
    cfg: RecastConfig,
    *,
    language: str | None = None,
    names: Any = None,
) -> tuple[Path | None, Path | None, list[dict[str, Any]]]:
    """Transcribe audio. language=None or 'auto' lets Whisper detect (Chinese-friendly)."""
    if audio_wav is None or not Path(audio_wav).is_file():
        log.warning("没有音轨，跳过字幕阶段。")
        empty = workdir / "subtitles.json"
        empty.write_text(
            json.dumps({"language": None, "segments": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        srt = workdir / "subtitles.srt"
        srt.write_text("", encoding="utf-8")
        return srt, empty, []

    from faster_whisper import WhisperModel

    device = requested_device(cfg.whisper.device or cfg.device)
    compute = cfg.whisper.compute_type
    if device == "cpu" and compute in {"float16", "float32"}:
        compute = "int8"
        log.warning("CPU 模式下将 Whisper compute_type 改为 int8。")

    lang = None if not language or language.lower() in {"auto", "none"} else language
    log.info(
        "加载 faster-whisper 模型 %s (device=%s, compute=%s, lang=%s)",
        cfg.whisper.model,
        device,
        compute,
        lang or "auto",
    )
    vram_report("字幕-加载前")
    model_kwargs = {
        "device": device,
        "compute_type": compute,
        "cpu_threads": cfg.cpu_threads,
    }
    try:
        # 本地已有缓存时不要再访问 Hugging Face，避免网络不通时一直卡住。
        model = WhisperModel(cfg.whisper.model, local_files_only=True, **model_kwargs)
        log.info("已使用本地 Whisper 缓存，未访问 Hugging Face。")
    except Exception as exc:
        timeout_sec = float(getattr(cfg.whisper, "download_timeout_sec", 60) or 60)
        log.warning("本地 Whisper 缓存不可用（%s），尝试联网下载（限时 %s 秒）。", exc, timeout_sec)
        try:
            model = _load_model_with_timeout(WhisperModel, cfg.whisper.model, model_kwargs, timeout_sec)
        except TimeoutError:
            raise RuntimeError(
                f"下载 Whisper 模型超时（超过 {timeout_sec:g} 秒）。"
                "请检查网络，或先把 faster-whisper 模型放到本机缓存后再试。"
            )
        except Exception as download_exc:
            raise RuntimeError(
                "无法加载 Whisper 模型：本地缓存不可用，且无法从 Hugging Face 下载。"
                "请检查网络，或先把 faster-whisper 模型放到本机缓存后再试。"
            ) from download_exc
    vram_report("字幕-已加载")

    prompt = build_initial_prompt(names)
    transcribe_kwargs: dict[str, Any] = {
        "language": lang,
        "beam_size": max(1, int(cfg.whisper.beam_size)),
        "vad_filter": bool(cfg.whisper.vad_filter),
        "word_timestamps": True,
    }
    if prompt:
        transcribe_kwargs["initial_prompt"] = prompt
    segments_iter, info = model.transcribe(str(audio_wav), **transcribe_kwargs)

    segments: list[dict[str, Any]] = []
    for seg in segments_iter:
        check_cancel()
        words = []
        for word in (seg.words or []):
            words.append(
                {
                    "start": float(word.start or 0.0),
                    "end": float(word.end or 0.0),
                    "word": word.word,
                    "probability": float(getattr(word, "probability", 0.0) or 0.0),
                }
            )
        segments.append(
            {
                "id": int(seg.id),
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
                "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                "words": words,
            }
        )
        log.info("  [%s -> %s] %s", f"{seg.start:.2f}", f"{seg.end:.2f}", (seg.text or "").strip())

    threshold = float(getattr(cfg.whisper, "no_speech_prob", 0.6) or 0.6)
    before = len(segments)
    segments = filter_no_speech(segments, max_no_speech_prob=threshold)
    if len(segments) != before:
        log.info("已丢弃 %s 段无语音（no_speech_prob > %.2f）。", before - len(segments), threshold)

    payload = {
        "language": getattr(info, "language", lang),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "model": cfg.whisper.model,
        "beam_size": int(cfg.whisper.beam_size),
        "initial_prompt": prompt,
        "segments": segments,
    }
    json_path = workdir / "subtitles.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    srt_path = write_srt(segments, workdir / "subtitles.srt")
    log.info("字幕已写出: %s  (%s 段, 语言=%s)", srt_path, len(segments), payload["language"])

    held = model
    model = None
    release_cuda(held)
    return srt_path, json_path, segments
