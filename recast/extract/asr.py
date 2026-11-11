"""ASR wrapper around the existing faster-whisper stage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from recast.config import RecastConfig


def run(
    audio_wav: Path | None,
    workdir: Path,
    cfg: RecastConfig,
    *,
    language: str | None = None,
) -> tuple[Path | None, Path | None, list[dict[str, Any]]]:
    from recast.stages import subtitles

    srt_path, json_path, segments = subtitles.run(audio_wav, workdir, cfg, language=language)
    # Also keep a clearly named ASR copy so OCR/ASR dual tracks are obvious.
    if srt_path and Path(srt_path).is_file():
        dest = workdir / "asr.srt"
        if Path(srt_path).resolve() != dest.resolve():
            shutil.copy2(srt_path, dest)
    if json_path and Path(json_path).is_file():
        dest_j = workdir / "asr.json"
        if Path(json_path).resolve() != dest_j.resolve():
            shutil.copy2(json_path, dest_j)
    return srt_path, json_path, segments
