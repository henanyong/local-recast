"""Pluggable TTS: optional local engine, imported wav table, or skip (still export SRT).

No cloud TTS key is required. Audio is never auto-sped up.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("recast")


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or 1)
    except Exception:
        try:
            from recast.ffmpeg import probe_video

            info = probe_video(path)
            return float(getattr(info, "duration", 0) or 0)
        except Exception:
            return 0.0


def _try_edge_tts(text: str, dest: Path, voice: str | None = None) -> bool:
    try:
        import asyncio
        import edge_tts  # type: ignore
    except Exception:
        return False
    voice = voice or "zh-CN-XiaoxiaoNeural"

    async def _run() -> None:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(str(dest))

    try:
        asyncio.run(_run())
        return dest.is_file() and dest.stat().st_size > 0
    except Exception as exc:
        log.info("edge-tts 不可用: %s", exc)
        return False


def _try_pyttsx3(text: str, dest: Path) -> bool:
    try:
        import pyttsx3  # type: ignore
    except Exception:
        return False
    try:
        engine = pyttsx3.init()
        # do not set a high rate — never speed up to fit a slot
        dest.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, str(dest))
        engine.runAndWait()
        return dest.is_file() and dest.stat().st_size > 0
    except Exception as exc:
        log.info("pyttsx3 不可用: %s", exc)
        return False


def synthesize_line(
    text: str,
    dest: str | Path,
    *,
    voice: str | None = None,
    wav_source: str | Path | None = None,
    allow_engines: bool = True,
) -> dict[str, Any]:
    """Produce a wav for one line. Prefer an imported wav, then a local engine, else skip."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if wav_source and Path(wav_source).is_file():
        src = Path(wav_source)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes()) if src.suffix.lower() == ".wav" else _ffmpeg_copy(src, dest)
        return {"ok": True, "path": str(dest), "source": "import", "duration": _duration(dest), "skipped": False}
    if not (text or "").strip():
        return {"ok": True, "path": None, "source": "empty", "duration": 0.0, "skipped": True}
    if allow_engines:
        if _try_edge_tts(text, dest, voice=voice):
            return {"ok": True, "path": str(dest), "source": "edge-tts", "duration": _duration(dest), "skipped": False}
        if _try_pyttsx3(text, dest):
            return {"ok": True, "path": str(dest), "source": "pyttsx3", "duration": _duration(dest), "skipped": False}
    log.info("未安装本地 TTS，也没有导入对白 wav，已跳过配音，仍会导出 SRT。")
    return {
        "ok": True,
        "path": None,
        "source": "skip",
        "duration": 0.0,
        "skipped": True,
        "message": "已跳过配音（未安装本地 TTS，也没有导入对白 wav），仍会写出 SRT。",
    }


def _ffmpeg_copy(src: Path, dest: Path) -> Path:
    from recast.ffmpeg import run_ffmpeg

    run_ffmpeg(["-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(dest)])
    return dest


def load_wav_table(path: str | Path) -> dict[str, Path]:
    """Map shot_code -> wav. Accepts a folder, a YAML/JSON table, or 'CODE=path' lines."""
    path = Path(path)
    mapping: dict[str, Path] = {}
    if path.is_dir():
        for f in path.iterdir():
            if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".aac", ".flac"}:
                mapping[f.stem] = f
        return mapping
    if not path.is_file():
        return mapping
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        import json

        data = json.loads(text)
        for k, v in (data.items() if isinstance(data, dict) else []):
            if v:
                mapping[str(k)] = Path(str(v))
        return mapping
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            for k, v in data.items():
                if v:
                    mapping[str(k)] = Path(str(v))
            return mapping
    except Exception:
        pass
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            mapping[k.strip()] = Path(v.strip())
    return mapping


def synthesize_table(
    shots: list[dict[str, Any]],
    dest_dir: str | Path,
    *,
    wav_table: dict[str, Path] | None = None,
    allow_engines: bool = True,
) -> dict[str, Any]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = []
    skipped = 0
    for sh in shots:
        text = (sh.get("dialogue") or sh.get("text") or "").strip()
        code = sh.get("shot_code") or f"S{sh.get('index_no', 0)}"
        src = None
        if wav_table:
            src = wav_table.get(str(code)) or wav_table.get(str(sh.get("speaker") or ""))
        out = dest_dir / f"{code}.wav"
        rec = synthesize_line(text, out, voice=None, wav_source=src, allow_engines=allow_engines)
        sh["audio_path"] = rec.get("path")
        sh["audio_source"] = rec.get("source")
        sh["audio_duration"] = rec.get("duration")
        if rec.get("skipped"):
            skipped += 1
        results.append({"shot_code": code, **rec})
    return {"results": results, "skipped": skipped, "dir": str(dest_dir)}
