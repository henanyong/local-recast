"""Stage 1: probe the video and extract a 16 kHz mono wav for Whisper."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from recast.ffmpeg import VideoInfo, ensure_constant_frame_rate, extract_audio, probe_video

log = logging.getLogger("recast")


def run(video: str | Path, workdir: Path) -> tuple[VideoInfo, Path | None]:
    workdir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video)
    source = Path(video)
    if info.is_vfr:
        log.warning("检测到可变帧率（约 %.3f fps），先转成恒定帧率再继续。", info.fps)
        cfr = workdir / "source_cfr.mp4"
        source = ensure_constant_frame_rate(video, cfr, info.fps)
        info = probe_video(source)
    log.info(
        "视频 %sx%s  %.3f fps  %.2fs  %s 帧  编码=%s  音轨=%s",
        info.width,
        info.height,
        info.fps,
        info.duration,
        info.frame_count,
        info.video_codec or "?",
        "有" if info.has_audio else "无",
    )
    wav = extract_audio(source, workdir / "audio.wav", sample_rate=16000)
    payload = info.to_dict()
    from recast.media import load_source_meta

    payload.update({k: v for k, v in load_source_meta(workdir).items() if v})
    (workdir / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return info, wav
