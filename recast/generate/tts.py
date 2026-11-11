"""Dialogue audio for 本地再铸. Audio is never auto-sped up.

edge-tts is Microsoft's online service, not a local engine and not a cloud key
in this app. Offline, use one of:

- a per-line wav (wav table) or ``shutil.copyfile`` import
- a per-character reference wav plus a local cloner in ``RECAST_TTS_CLONE``
- a Piper ``.onnx`` the user places at ``models/piper/`` or ``RECAST_PIPER_MODEL``
  (CPU, not loaded into the 8GB GPU budget; do not commit the weights)
- pyttsx3, which is local but robotic

If none of those are available, dubbing is skipped and the SRT is still written.
"""

from __future__ import annotations

import logging
import os
import shutil
import wave
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
_PYTTSX3: Any = None


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


def _pyttsx3_engine() -> Any:
    """One engine for the process. Creating one per line is slow and leaky."""
    global _PYTTSX3
    if _PYTTSX3 is None:
        import pyttsx3  # type: ignore

        _PYTTSX3 = pyttsx3.init()
    return _PYTTSX3


def _try_pyttsx3(text: str, dest: Path) -> bool:
    try:
        engine = _pyttsx3_engine()
    except Exception:
        return False
    try:
        # do not set a high rate — never speed up to fit a slot
        dest.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, str(dest))
        engine.runAndWait()
        return dest.is_file() and dest.stat().st_size > 0
    except Exception as exc:
        log.info("pyttsx3 不可用: %s", exc)
        return False


def discover_piper_model(voice: str | None = None) -> Path | None:
    """A user-supplied Piper onnx. Nothing here is downloaded or committed."""
    if voice:
        candidate = Path(voice)
        if candidate.suffix.lower() == ".onnx" and candidate.is_file():
            return candidate
    env = os.environ.get("RECAST_PIPER_MODEL", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    from recast.config import PROJECT_ROOT

    folder = PROJECT_ROOT / "models" / "piper"
    if folder.is_dir():
        found = sorted(folder.glob("*.onnx"))
        if found:
            return found[0]
    return None


def _try_piper(text: str, dest: Path, model: Path) -> bool:
    exe = shutil.which("piper")
    if not exe:
        return False
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [exe, "--model", str(model), "--output_file", str(dest)],
        input=text,
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


def _try_local_clone(text: str, dest: Path, reference: Path) -> bool:
    """Optional offline cloner. ``RECAST_TTS_CLONE`` is a local program, not a cloud API."""
    command = os.environ.get("RECAST_TTS_CLONE", "").strip()
    if not command or not reference.is_file():
        return False
    import shlex
    import subprocess

    argv = shlex.split(command) + ["--ref", str(reference), "--text", text, "--out", str(dest)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


def classify_voice(voice: str | None) -> dict[str, Any]:
    """Split a character ``voice`` field into an online id, a piper model, or a reference wav."""
    raw = str(voice or "").strip()
    if not raw:
        return {"edge_voice": None, "piper_model": None, "reference": None, "raw": ""}
    path = Path(raw)
    suffix = path.suffix.lower()
    if suffix == ".onnx" and path.is_file():
        return {"edge_voice": None, "piper_model": path, "reference": None, "raw": raw}
    if suffix in _AUDIO_SUFFIXES and path.is_file():
        return {"edge_voice": None, "piper_model": None, "reference": path, "raw": raw}
    return {"edge_voice": raw, "piper_model": None, "reference": None, "raw": raw}


def voices_by_speaker(characters: list[dict[str, Any]] | None) -> dict[str, str]:
    """Map character name and id to the JSON ``voice`` field."""
    mapping: dict[str, str] = {}
    for character in characters or []:
        voice = str(character.get("voice") or "").strip()
        if not voice:
            continue
        for key in (character.get("name"), character.get("id"), character.get("display_name")):
            if key:
                mapping[str(key)] = voice
    return mapping


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
            if src.suffix.lower() == ".wav":
                shutil.copyfile(src, dest)
            else:
                _ffmpeg_copy(src, dest)
        return {"ok": True, "path": str(dest), "source": "import", "duration": _duration(dest), "skipped": False}
    if not (text or "").strip():
        return {"ok": True, "path": None, "source": "empty", "duration": 0.0, "skipped": True}
    spec = classify_voice(voice)
    note = ""
    if allow_engines:
        reference = spec["reference"]
        if reference is not None and _try_local_clone(text, dest, reference):
            return {
                "ok": True,
                "path": str(dest),
                "source": "local-clone",
                "duration": _duration(dest),
                "skipped": False,
                "voice": str(reference),
            }
        if reference is not None:
            note = "已记录角色参考 wav，但没有设置 RECAST_TTS_CLONE，不能离线克隆音色。"
        if spec["piper_model"] is not None and _try_piper(text, dest, spec["piper_model"]):
            return {
                "ok": True,
                "path": str(dest),
                "source": "piper",
                "duration": _duration(dest),
                "skipped": False,
                "voice": str(spec["piper_model"]),
                "message": note,
            }
        if spec["edge_voice"] and _try_edge_tts(text, dest, voice=spec["edge_voice"]):
            return {
                "ok": True,
                "path": str(dest),
                "source": "edge-tts",
                "duration": _duration(dest),
                "skipped": False,
                "voice": spec["edge_voice"],
                "message": "edge-tts 是微软在线语音，不是本地引擎。",
            }
        local_model = None if spec["piper_model"] is not None else discover_piper_model(None)
        if local_model is not None and _try_piper(text, dest, local_model):
            return {
                "ok": True,
                "path": str(dest),
                "source": "piper",
                "duration": _duration(dest),
                "skipped": False,
                "voice": str(local_model),
                "message": note or "使用本机 piper 模型，未联网。",
            }
        if not spec["edge_voice"] and _try_edge_tts(text, dest, voice=None):
            return {
                "ok": True,
                "path": str(dest),
                "source": "edge-tts",
                "duration": _duration(dest),
                "skipped": False,
                "voice": "zh-CN-XiaoxiaoNeural",
                "message": "edge-tts 是微软在线语音，不是本地引擎。",
            }
        if _try_pyttsx3(text, dest):
            return {
                "ok": True,
                "path": str(dest),
                "source": "pyttsx3",
                "duration": _duration(dest),
                "skipped": False,
                "message": (note + " 已用本机 pyttsx3，音色机械。").strip(),
            }
    log.info("没有可用的离线配音，也没有导入对白 wav。edge-tts 需要联网。已跳过配音，仍会导出 SRT。")
    return {
        "ok": True,
        "path": None,
        "source": "skip",
        "duration": 0.0,
        "skipped": True,
        "message": "已跳过配音。edge-tts 是微软在线服务；离线请把 piper 的 .onnx 放到 models/piper/（不要提交权重），或导入 wav / 设置 RECAST_TTS_CLONE。仍会写出 SRT。",
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
    voices: dict[str, str] | None = None,
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
        voice = str(sh.get("voice") or "").strip()
        if not voice and voices:
            voice = voices.get(str(sh.get("speaker") or "")) or voices.get(str(sh.get("character_id") or "")) or ""
        out = dest_dir / f"{code}.wav"
        rec = synthesize_line(text, out, voice=voice or None, wav_source=src, allow_engines=allow_engines)
        sh["audio_path"] = rec.get("path")
        sh["audio_source"] = rec.get("source")
        sh["audio_duration"] = rec.get("duration")
        if rec.get("skipped"):
            skipped += 1
        results.append({"shot_code": code, **rec})
    return {"results": results, "skipped": skipped, "dir": str(dest_dir)}
