"""Stage 7: mux audio, write sidecar SRT, optionally burn subtitles, output.mp4."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from recast.config import RecastConfig
from recast.ffmpeg import FFmpegError, copy_sidecar_srt, mux_av, probe_video, run_ffmpeg

log = logging.getLogger("recast")


def run(
    video: str | Path,
    source_video: str | Path,
    workdir: Path,
    cfg: RecastConfig,
    output: str | Path,
    subtitle: str | Path | None = None,
    *,
    burn_subs: bool | None = None,
    has_audio: bool | None = None,
) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    burn = cfg.compose.burn_subs if burn_subs is None else bool(burn_subs)

    if has_audio is None:
        try:
            has_audio = probe_video(source_video).has_audio
        except Exception:
            has_audio = False
    use_audio = bool(has_audio) and cfg.compose.audio != "none" and Path(source_video).is_file()
    audio_src: Path | None = Path(source_video) if use_audio else None

    srt_in = None
    if subtitle and Path(subtitle).is_file() and Path(subtitle).stat().st_size > 0:
        srt_in = Path(subtitle)

    from recast.media import same_media_file

    same_file = same_media_file(video, source_video)
    if same_file and not burn:
        args = ["-i", str(source_video), "-c", "copy", "-movflags", "+faststart", str(out)]
        try:
            run_ffmpeg(args)
        except FFmpegError:
            log.warning("流复制失败，改为重编码。")
            mux_av(
                video,
                out,
                audio=audio_src,
                subtitle=srt_in,
                burn_subs=False,
                crf=cfg.compose.crf,
                copy_audio=True,
            )
    else:
        mux_av(
            video,
            out,
            audio=audio_src,
            subtitle=srt_in,
            burn_subs=bool(burn and srt_in is not None),
            crf=cfg.compose.crf,
            copy_audio=True,
        )

    sidecar = None
    if srt_in is not None:
        sidecar = copy_sidecar_srt(srt_in, out)
        shutil.copy2(srt_in, workdir / "output.srt")

    log.info("成片已写出: %s%s", out, f" ，字幕旁路 {sidecar}" if sidecar else "")
    return out
