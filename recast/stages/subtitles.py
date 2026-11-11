"""Stage 2: faster-whisper transcription -> SRT + timestamped JSON."""

from __future__ import annotations

import json
import logging
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


def run(
    audio_wav: Path | None,
    workdir: Path,
    cfg: RecastConfig,
    *,
    language: str | None = None,
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
        log.warning("本地 Whisper 缓存不可用（%s），尝试联网下载。", exc)
        log.info("若 huggingface.co 无法访问，这一步可能会长时间无响应。")
        try:
            model = WhisperModel(cfg.whisper.model, **model_kwargs)
        except Exception as download_exc:
            raise RuntimeError(
                "无法加载 Whisper 模型：本地缓存不可用，且无法从 Hugging Face 下载。"
                "请检查网络，或先把 faster-whisper 模型放到本机缓存后再试。"
            ) from download_exc
    vram_report("字幕-已加载")

    segments_iter, info = model.transcribe(
        str(audio_wav),
        language=lang,
        beam_size=max(1, int(cfg.whisper.beam_size)),
        vad_filter=bool(cfg.whisper.vad_filter),
        word_timestamps=True,
    )

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

    payload = {
        "language": getattr(info, "language", lang),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "model": cfg.whisper.model,
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
