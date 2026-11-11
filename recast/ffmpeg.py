"""ffmpeg / ffprobe wrappers. Paths are handled in a Windows-safe way."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

log = logging.getLogger("recast")

# Optional extra directory (Settings / PyInstaller bundle / RECAST_FFMPEG_DIR).
_FFMPEG_DIR: str | None = None


class FFmpegError(RuntimeError):
    pass


def set_ffmpeg_dir(path: str | Path | None) -> None:
    """Point the wrappers at a fixed ffmpeg directory (used by the desktop Settings)."""
    global _FFMPEG_DIR
    _FFMPEG_DIR = str(path) if path else None


def _exe_name(base: str) -> str:
    return base + ".exe" if os.name == "nt" else base


def _search_exe(base: str) -> str | None:
    name = _exe_name(base)
    env_key = "RECAST_FFMPEG" if base == "ffmpeg" else "RECAST_FFPROBE"
    env = os.environ.get(env_key) or os.environ.get("RECAST_FFMPEG_DIR")
    candidates: list[Path] = []
    if _FFMPEG_DIR:
        candidates.append(Path(_FFMPEG_DIR) / name)
        candidates.append(Path(_FFMPEG_DIR))
    if env:
        ev = Path(env)
        candidates.append(ev if ev.name.lower().startswith(base) else ev / name)
        candidates.append(ev / name)
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
        for root in (base_dir, base_dir / "ffmpeg", base_dir / "_internal" / "ffmpeg"):
            candidates.append(root / name)
    which = shutil.which(base) or shutil.which(name)
    if which:
        return which
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def which_ffmpeg() -> str:
    exe = _search_exe("ffmpeg")
    if not exe:
        raise FFmpegError(
            "找不到 ffmpeg。请安装并把 ffmpeg 所在目录加入 PATH，或在设置里指定目录。"
            " Windows 推荐: https://www.gyan.dev/ffmpeg/builds/"
        )
    return exe


def which_ffprobe() -> str:
    exe = _search_exe("ffprobe")
    if not exe:
        raise FFmpegError("找不到 ffprobe（通常和 ffmpeg 在同一目录）。请把它加入 PATH，或在设置里指定目录。")
    return exe


def run_ffmpeg(args: Sequence[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [which_ffmpeg(), "-hide_banner", "-y", *args]
    log.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stderr = (result.stderr or "")[-4000:]
        raise FFmpegError(f"ffmpeg 失败 (exit {result.returncode}):\n{stderr}")
    return result


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_fps(rate: str | None) -> float:
    if not rate or rate in {"0/0", "N/A"}:
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def probe_video(path: str | Path) -> VideoInfo:
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"找不到视频: {src}")
    cmd = [
        which_ffprobe(),
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(src),
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe 失败: {result.stderr[-2000:]}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise FFmpegError(f"文件没有视频流: {src}")

    fps = _parse_fps(video.get("avg_frame_rate")) or _parse_fps(video.get("r_frame_rate"))
    duration = 0.0
    for candidate in (video.get("duration"), fmt.get("duration")):
        if candidate not in (None, "N/A", ""):
            try:
                duration = float(candidate)
                break
            except ValueError:
                continue
    nb = video.get("nb_frames")
    try:
        frame_count = int(nb) if nb not in (None, "N/A", "") else 0
    except ValueError:
        frame_count = 0
    if frame_count <= 0 and duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    return VideoInfo(
        path=str(src.resolve()),
        width=width,
        height=height,
        fps=float(fps or 25.0),
        duration=float(duration),
        frame_count=frame_count,
        has_audio=audio is not None,
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str((audio or {}).get("codec_name") or ""),
    )


def extract_audio(video: str | Path, wav_path: str | Path, sample_rate: int = 16000) -> Path | None:
    """Extract mono PCM wav for Whisper. Returns None if the source has no audio."""
    info = probe_video(video)
    out = Path(wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not info.has_audio:
        log.warning("源视频没有音轨，跳过音频提取。")
        return None
    run_ffmpeg(
        [
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )
    return out


def mux_av(
    video: str | Path,
    output: str | Path,
    audio: str | Path | None = None,
    subtitle: str | Path | None = None,
    *,
    burn_subs: bool = False,
    crf: int = 18,
    copy_audio: bool = True,
) -> Path:
    """Mux video + optional audio and write sidecar / burned SRT."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = ["-i", str(video)]
    if audio is not None and Path(audio).is_file():
        args += ["-i", str(audio)]

    filters: list[str] = []
    if burn_subs and subtitle is not None and Path(subtitle).is_file():
        filters.append(subtitles_filter(Path(subtitle), dest_dir=out.parent))
    if filters:
        args += ["-vf", ",".join(filters)]

    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium"]
    if audio is not None and Path(audio).is_file():
        if copy_audio and not _needs_reencode_audio(Path(audio)):
            args += ["-c:a", "copy"]
        else:
            args += ["-c:a", "aac", "-b:a", "192k"]
        args += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    else:
        args += ["-an"]
    args += ["-movflags", "+faststart", str(out)]
    run_ffmpeg(args)
    return out


def _needs_reencode_audio(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in {".wav", ".flac", ".pcm"}


def prepare_subtitle_file(subtitle: str | Path, dest_dir: str | Path | None = None) -> Path:
    """Copy an SRT onto a pure-ASCII path so the subtitles filter stays valid."""
    from recast.media import is_ascii_path

    src = Path(subtitle)
    if dest_dir is None or not is_ascii_path(dest_dir):
        dest = Path(tempfile.mkdtemp(prefix="recast_sub_")) / "burn_subs.srt"
    else:
        dest = Path(dest_dir) / "burn_subs.srt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dest.exists() and dest.resolve() == src.resolve():
            return dest
    except OSError:
        pass
    shutil.copy2(src, dest)
    return dest


def _escape_subtitles_path(path: Path) -> str:
    """Escape a path for ffmpeg's subtitles filter (Windows drive letters included)."""
    text = Path(path).resolve().as_posix().replace(chr(92), "/")
    special = {chr(92), ':', chr(39), ',', '[', ']', ';'}
    escaped: list[str] = []
    for ch in text:
        if ch in special:
            escaped.append(chr(92) + ch)
        else:
            escaped.append(ch)
    return chr(39) + ''.join(escaped) + chr(39)


def subtitles_filter(path: str | Path, dest_dir: str | Path | None = None) -> str:
    """Return a subtitles= filter argument using an ASCII copy of the SRT."""
    ascii_srt = prepare_subtitle_file(path, dest_dir)
    return "subtitles=" + _escape_subtitles_path(ascii_srt)

def copy_sidecar_srt(subtitle: str | Path, video_out: str | Path) -> Path | None:
    src = Path(subtitle)
    if not src.is_file():
        return None
    dest = Path(video_out).with_suffix(".srt")
    dest.write_bytes(src.read_bytes())
    return dest


def open_ffmpeg_writer(path: str | Path, width: int, height: int, fps: float, crf: int = 18):
    """Start an ffmpeg process that accepts raw BGR24 frames on stdin."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        which_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        str(out),
    ]
    log.debug("ffmpeg writer: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc
