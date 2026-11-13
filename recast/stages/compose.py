"""Stage 7: mux audio, write sidecar SRT, optionally burn subtitles, output.mp4.

When burning new subtitles, the original video's own burned-in subtitles
(if any) are covered first (see ``cover_burned_subs`` / ``cover_mode``),
so the new line does not print on top of the old one.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from recast.config import RecastConfig
from recast.ffmpeg import (
    COVER_MODES,
    FFmpegError,
    apply_shot_enhance,
    cover_subtitle_band,
    copy_sidecar_srt,
    enhance_filter_for_shots,
    mux_av,
    probe_video,
    run_ffmpeg,
)
from recast.stages.subtitles import detect_burned_subtitles

log = logging.getLogger("recast")


def enhance_script_output(
    video: str | Path,
    workdir: str | Path,
    enhance_shots: list[dict] | None,
    *,
    crf: int = 18,
    use_nvenc: bool | None = None,
) -> Path:
    """N24: apply the per-shot enhance filter to a stitched script output.

    The script path stitches per-shot clips in
    ``recast.generate.stitch.stitch_and_dub`` and never calls ``run``
    above, so its enhance flags (``workdir/shots_enhance.json``, read by
    ``runner.enhance_shots_for_task``) would otherwise be silently
    ignored. This is the script path's equivalent of the import path's
    pre-pass: the same real ffmpeg filter
    (``recast.ffmpeg.apply_shot_enhance``, hqdn3d + unsharp) on the same
    time ranges, run once over the stitched timeline after stitching.
    It is a post-pass, not a per-clip pre-stitch pass — the stitcher has
    no enhance hook and touching it is out of scope for N24 — so when
    subtitles were burned during stitching they are re-encoded too
    (the filter is light denoise + sharpen; subtitle pixels move
    slightly). The caller moves the returned file over the original
    output when it differs.

    Returns the input path unchanged (no re-encode, no file written)
    when there is no usable enhance range. Raises FileNotFoundError
    when the video is missing and lets ffmpeg errors propagate so the
    caller can record the failure instead of silently shipping an
    unenhanced cut.
    """
    src = Path(video)
    if enhance_filter_for_shots(enhance_shots) is None:
        log.info("剧本路逐镜头增强：没有开着的镜头，不做处理。")
        return src
    if not src.is_file():
        raise FileNotFoundError(f"要增强的成片不存在：{video}")
    out = Path(workdir) / "enhanced_script.mp4"
    apply_shot_enhance(
        src, out, enhance_shots or [],
        crf=crf, use_nvenc=use_nvenc,
    )
    log.info(
        "剧本路逐镜头增强已应用（%s 个镜头）：%s",
        len(enhance_shots or []), out,
    )
    return out


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
    cover_burned_subs: bool | None = None,
    cover_mode: str = "blur",
    enhance_shots: list[dict] | None = None,
) -> Path:
    """Mux audio, write sidecar SRT, optionally burn subtitles.

    ``cover_burned_subs`` controls covering the original video's burned-in
    subtitle band before new subtitles are burned (only when burning):
    None = auto-detect (default: cover only when OCR finds text in the
    bottom 15% of the frame; OCR unavailable means no cover, with a
    warning); True = always cover; False = never cover.
    ``cover_mode`` is "blur" (default) or "bar" (black bar); "inpaint"
    is a placeholder and raises NotImplementedError.
    ``enhance_shots`` is the per-shot post-process switch (design 4.2):
    a list of ``{"shot_code": str, "start": float, "end": float}`` for the
    shots whose enhance switch is ON (default: every shot off). The
    enhance filter (light denoise + sharpen, see
    ``recast.ffmpeg.apply_shot_enhance``) runs on those time ranges only,
    as a pre-pass before cover / mux — this is the wiring point for the
    import-video path, whose picture is one whole video (replaced.mp4 or
    the source alias) and has no per-shot files. The script path's
    per-shot clips (``shots_silent/<shot_code>.mp4``) are stitched by
    ``recast.generate.stitch.stitch_and_dub``, not by this function, so
    the script path applies the same time-range filter as a post-pass on
    the stitched output instead — see ``enhance_script_output`` (N24),
    called from ``runner.dispatch_job`` after the script episode is
    stitched. Same filter, same ``shots_enhance.json`` flags; only the
    point in the pipeline differs, because the stitcher itself has no
    enhance hook.
    """
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

    video_for_mux = Path(video)
    enhanced = False
    if enhance_filter_for_shots(enhance_shots) is not None:
        enhanced_path = Path(workdir) / "enhanced.mp4"
        apply_shot_enhance(
            video, enhanced_path, enhance_shots or [],
            crf=cfg.compose.crf, use_nvenc=cfg.compose.use_nvenc,
        )
        video_for_mux = enhanced_path
        enhanced = True
        log.info(
            "逐镜头增强已应用（%s 个镜头）：%s",
            len(enhance_shots or []), enhanced_path,
        )

    if burn and srt_in is not None:
        if cover_mode not in COVER_MODES:
            raise ValueError(f"未知的遮盖模式：{cover_mode}，可选 blur / bar / inpaint")
        if cover_burned_subs is None:
            detected = detect_burned_subtitles(video)
            if detected is True:
                should_cover = True
                log.info("检测到底部烧录字幕，烧新字幕前先遮盖（模式=%s）。", cover_mode)
            elif detected is False:
                should_cover = False
                log.info("未检测到底部烧录字幕，不遮盖，直接烧新字幕。")
            else:
                should_cover = False
                log.warning("无法判断有无烧录字幕（OCR 不可用），不遮盖直接烧新字幕。")
        else:
            should_cover = bool(cover_burned_subs)
        if should_cover:
            covered = Path(workdir) / "covered_subs.mp4"
            cover_subtitle_band(
                video_for_mux, covered, mode=cover_mode, use_nvenc=cfg.compose.use_nvenc
            )
            video_for_mux = covered
            log.info("已遮盖原烧录字幕带（模式=%s）：%s", cover_mode, covered)

    from recast.media import same_media_file

    same_file = same_media_file(video, source_video)
    if same_file and not burn and not enhanced:
        args = ["-i", str(source_video), "-c", "copy", "-movflags", "+faststart", str(out)]
        try:
            run_ffmpeg(args)
        except FFmpegError:
            log.warning("流复制失败，改为重编码。")
            mux_av(
                video_for_mux,
                out,
                audio=audio_src,
                subtitle=srt_in,
                burn_subs=False,
                crf=cfg.compose.crf,
                copy_audio=True,
                use_nvenc=cfg.compose.use_nvenc,
            )
    else:
        mux_av(
            video_for_mux,
            out,
            audio=audio_src,
            subtitle=srt_in,
            burn_subs=bool(burn and srt_in is not None),
            crf=cfg.compose.crf,
            copy_audio=True,
            use_nvenc=cfg.compose.use_nvenc,
        )

    sidecar = None
    if srt_in is not None:
        sidecar = copy_sidecar_srt(srt_in, out)
        shutil.copy2(srt_in, workdir / "output.srt")

    log.info("成片已写出: %s%s", out, f" ，字幕旁路 {sidecar}" if sidecar else "")
    return out
