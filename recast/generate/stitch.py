"""Stitch silent clips in shot order, lay dialogue/narration, align subtitles, export + SRT.

Pipeline order is fixed: silent picture first, then dub. Templates can change
dialogue text, not this order. Audio is never auto-sped up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from recast.ffmpeg import copy_sidecar_srt, run_ffmpeg
from recast.templates.duration import check_shots, extend_timeline_for_audio

log = logging.getLogger("recast")


def write_srt(segments: list[dict[str, Any]], dest: Path) -> Path:
    from recast.stages.subtitles import write_srt as _write

    _write(segments, dest)
    return dest


def shots_to_segments(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    for sh in shots:
        text = (sh.get("dialogue") or sh.get("text") or "").strip()
        if not text:
            continue
        start = sh.get("narration_start", sh.get("start_time"))
        end = sh.get("narration_end", sh.get("end_time"))
        if start is None:
            start = 0.0
        if end is None:
            dur = float(sh.get("duration_sec") or sh.get("audio_duration") or 0)
            end = float(start) + max(dur, 0.5)
        # If audio is longer than the slot, do NOT speed up: extend the subtitle
        # (and let the caller extend the picture if they choose). Flag is left to quality.
        audio_dur = float(sh.get("audio_duration") or 0)
        if audio_dur and float(end) - float(start) < audio_dur:
            end = float(start) + audio_dur
        segs.append({"start": float(start), "end": float(end), "text": text})
    return segs


def _concat_list(clips: list[Path], dest: Path) -> Path:
    lst = dest.parent / (dest.stem + "_concat.txt")
    lines = []
    for c in clips:
        s = str(c.resolve()).replace("'", "'\\''")
        lines.append(f"file '{s}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dest)])
    except Exception:
        run_ffmpeg(
            [
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest),
            ]
        )
    return dest


def _mix_audio(shots: list[dict[str, Any]], dest: Path, total_duration: float) -> Path | None:
    """Lay each wav at the shot start time. No atempo / speed change."""
    items: list[tuple[float, Path]] = []
    for sh in shots:
        path = sh.get("audio_path")
        if not path or not Path(path).is_file():
            continue
        start = float(sh.get("narration_start", sh.get("start_time") or 0) or 0)
        items.append((start, Path(path)))
    if not items:
        return None
    # Build an amix filter with adelay (ms)
    args: list[str] = ["-y"]
    filters = []
    for i, (start, path) in enumerate(items):
        args += ["-i", str(path)]
        delay = max(0, int(round(start * 1000)))
        filters.append(f"[{i}:a]adelay={delay}|{delay},aformat=sample_fmts=fltp:sample_rates=24000:channel_layouts=mono[a{i}]")
    mix_ins = "".join(f"[a{i}]" for i in range(len(items)))
    filters.append(f"{mix_ins}amix=inputs={len(items)}:duration=longest:dropout_transition=0,apad=whole_dur={total_duration:.3f}[aout]")
    args += ["-filter_complex", ";".join(filters), "-map", "[aout]", "-t", f"{total_duration:.3f}", str(dest)]
    run_ffmpeg(args)
    return dest


def stitch_and_dub(
    shots: list[dict[str, Any]],
    workdir: str | Path,
    *,
    silent: str | Path | None = None,
    output: str | Path | None = None,
    burn_subs: bool = False,
    precheck_dialogue: bool = True,
    max_cps: float = 4.0,
) -> dict[str, Any]:
    """Stitch (if needed), mix dialogue/narration, write episode + sidecar SRT.

    If precheck_dialogue finds overflow, raise DialogueOverflowBundle — never speed up.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if precheck_dialogue:
        issues = check_shots(shots, max_cps=max_cps)
        if issues:
            from recast.templates.duration import DialogueOverflow

            class DialogueOverflowBundle(ValueError):
                def __init__(self, items: list[DialogueOverflow]):
                    self.issues = items
                    super().__init__("\n".join(str(i) for i in items))

            raise DialogueOverflowBundle(issues)

    clips = []
    for sh in shots:
        p = sh.get("output_path")
        if p and Path(p).is_file():
            clips.append(Path(p))
    silent_path = Path(silent) if silent else workdir / "silent.mp4"
    if not silent_path.is_file() and clips:
        _concat_list(clips, silent_path)
    if not silent_path.is_file():
        raise FileNotFoundError("还没有静音画面，请先完成「按剧本生成（先静音）」。")

    # Real wav length, not the character-count estimate. Later shots move back
    # so two lines are not mixed on top of each other. Audio is not sped up.
    confirmations = extend_timeline_for_audio(shots)
    for note in confirmations:
        log.warning("%s", note["message"])

    # total duration from shots
    last_end = 0.0
    for sh in shots:
        end = sh.get("narration_end", sh.get("end_time"))
        if end is not None:
            last_end = max(last_end, float(end))
        else:
            start = float(sh.get("start_time") or last_end)
            last_end = start + float(sh.get("duration_sec") or 0)
    if last_end <= 0:
        last_end = 1.0

    mixed = workdir / "dialogue_mix.wav"
    audio = None
    try:
        audio = _mix_audio(shots, mixed, last_end)
    except Exception as exc:
        log.warning("对白混音失败（将导出无配音成片，SRT 仍会写）: %s", exc)
        audio = None

    segs = shots_to_segments(shots)
    srt = workdir / "subtitles.srt"
    write_srt(segs, srt)
    # also a script-source copy
    write_srt(segs, workdir / "script.srt")

    output = Path(output) if output else workdir / "output.mp4"
    args = ["-y", "-i", str(silent_path)]
    if audio and Path(audio).is_file():
        args += ["-i", str(audio), "-c:v", "copy", "-c:a", "aac", "-shortest"]
    else:
        args += ["-c:v", "copy", "-an"]
    if burn_subs and srt.is_file():
        # burning requires re-encode
        from recast.ffmpeg import subtitles_filter

        filt = subtitles_filter(srt, dest_dir=workdir)
        args = ["-y", "-i", str(silent_path)]
        if audio and Path(audio).is_file():
            args += ["-i", str(audio)]
        args += ["-vf", filt, "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if audio and Path(audio).is_file():
            args += ["-c:a", "aac", "-shortest"]
        else:
            args += ["-an"]
    args.append(str(output))
    run_ffmpeg(args)
    copy_sidecar_srt(srt, output)
    marker = workdir / "DELIVERY.txt"
    note = "成片与旁路 SRT 已写出。"
    if (workdir / "ANIMATIC.txt").is_file():
        note += " 画面是「分镜预览成片」，不是完整图生视频。"
    marker.write_text(note + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "srt": str(srt),
        "silent": str(silent_path),
        "audio": str(audio) if audio and Path(audio).is_file() else None,
        "segments": len(segs),
        "confirmations": confirmations,
    }
