"""Jianying draft plus interchange files other NLEs can open.

The draft_content.json layout follows the field names commonly used by
Jianying 5.x drafts (microsecond ranges, relative media paths, separate
video and audio tracks). This environment has not opened the file in
Jianying, so the export does not claim that any version imports it.

timeline.edl is CMX 3600. timeline_fcp.xml is FCP 7 xmeml. Both use paths
relative to this folder.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring


TARGET_NOTE = (
    "draft_content.json 按剪映 5.x 常见草稿字段组织（相对路径、微秒时间范围）。"
    "本环境没有打开剪映，不能声称某一版本已经验证能导入。"
    "时间线请同时用 timeline.edl（CMX 3600）或 timeline_fcp.xml（FCP7 xmeml）。"
)

README = """剪映工程导出说明
================

本目录由「本地再铸」写出，包含：

- project.json       简单时间线清单。路径相对于本目录。
- timeline.xml       同上内容的 XML
- timeline.edl       CMX 3600，Premiere、DaVinci Resolve 等可以打开
- timeline_fcp.xml   FCP7 xmeml，多数非线工具可以打开
- draft_info.json    剪映草稿字段的尽力描述
- draft_content.json 按剪映 5.x 常见 draft_content 字段组织的草稿
- media/             成片、分镜、对白、SRT 的副本

关于剪映版本
  草稿字段对齐的是剪映 5.x 常见的 draft_content 结构。
  写出这份工程的环境没有安装、也没有打开过剪映，
  因此不能把「已在某个剪映版本验证能打开」写进交付说明。
  若剪映打不开，请改用 timeline.edl 或 timeline_fcp.xml。

路径
  素材路径都是相对本目录的，把整个文件夹挪走不会把链接写死成旧绝对路径。

音频
  每条对白按该段自己的时长写入，不用整段成片的总时长去填每一条。
"""


def _dur_us(seconds: float) -> int:
    return int(round(float(seconds) * 1_000_000))


def _rel_to(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        rel = Path(os.path.relpath(Path(path).resolve(), Path(root).resolve()))
    except Exception:
        rel = Path(path).name
    return rel.as_posix()


def _copy(src: Path, dest_dir: Path) -> Path | None:
    if not src or not Path(src).is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(src).name
    if dest.resolve() != Path(src).resolve():
        shutil.copy2(src, dest)
    return dest


def build_audio_segments(
    shots: list[dict[str, Any]] | None,
    *,
    mix_duration: float | None = None,
) -> list[dict[str, Any]]:
    """One audio segment per dialogue line, using that line's own duration.

    The episode picture length is never substituted for a missing line length.
    A single mix file with no per-line timing becomes one clip of the mix's
    own duration.
    """
    segments: list[dict[str, Any]] = []
    for i, sh in enumerate(shots or []):
        dialogue = str(sh.get("dialogue") or sh.get("text") or "").strip()
        has_line = bool(dialogue) or bool(sh.get("audio_path")) or sh.get("audio_duration") not in (None, "")
        if not has_line:
            continue
        raw_dur = sh.get("audio_duration")
        if raw_dur in (None, ""):
            if sh.get("end_time") is not None and sh.get("start_time") is not None and dialogue:
                raw_dur = float(sh["end_time"]) - float(sh["start_time"])
            elif sh.get("duration_sec") not in (None, "") and dialogue:
                raw_dur = sh.get("duration_sec")
            else:
                continue
        try:
            dur = float(raw_dur)
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        start = sh.get("narration_start", sh.get("start_time"))
        try:
            start_f = 0.0 if start is None else float(start)
        except (TypeError, ValueError):
            start_f = 0.0
        segments.append(
            {
                "index": i,
                "shot_code": sh.get("shot_code") or f"S{i+1:02d}",
                "start": start_f,
                "duration": dur,
                "source": str(sh["audio_path"]) if sh.get("audio_path") else None,
            }
        )
    if segments:
        return segments
    if mix_duration and float(mix_duration) > 0:
        return [
            {
                "index": 0,
                "shot_code": "mix",
                "start": 0.0,
                "duration": float(mix_duration),
                "source": None,
            }
        ]
    return []


def _frames_to_tc(seconds: float, fps: int) -> str:
    rate = max(1, int(fps))
    frames = int(round(float(seconds) * rate))
    ff = frames % rate
    whole = frames // rate
    mm, ss = divmod(whole, 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def render_edl(
    *,
    title: str,
    fps: int,
    clips: list[dict[str, Any]],
    audio_segments: list[dict[str, Any]],
) -> str:
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    event = 1
    for clip in clips:
        start = float(clip["start"])
        dur = float(clip["duration"])
        src_in = _frames_to_tc(0, fps)
        src_out = _frames_to_tc(dur, fps)
        rec_in = _frames_to_tc(start, fps)
        rec_out = _frames_to_tc(start + dur, fps)
        name = clip.get("path") or clip.get("shot_code") or "video"
        lines.append(f"{event:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {name}")
        event += 1
    for seg in audio_segments:
        start = float(seg["start"])
        dur = float(seg["duration"])
        src_in = _frames_to_tc(0, fps)
        src_out = _frames_to_tc(dur, fps)
        rec_in = _frames_to_tc(start, fps)
        rec_out = _frames_to_tc(start + dur, fps)
        name = seg.get("path") or seg.get("shot_code") or "audio"
        lines.append(f"{event:03d}  AX       A     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {name}")
        event += 1
    lines.append("")
    return "\n".join(lines)


def _xml_text(parent: Element, tag: str, text: str) -> Element:
    el = SubElement(parent, tag)
    el.text = text
    return el


def render_fcp_xml(
    *,
    title: str,
    width: int,
    height: int,
    fps: int,
    clips: list[dict[str, Any]],
    audio_segments: list[dict[str, Any]],
) -> bytes:
    rate = max(1, int(fps))
    root = Element("xmeml", {"version": "5"})
    sequence = SubElement(root, "sequence")
    _xml_text(sequence, "name", title)
    rate_el = SubElement(sequence, "rate")
    _xml_text(rate_el, "timebase", str(rate))
    _xml_text(rate_el, "ntsc", "FALSE")
    media = SubElement(sequence, "media")
    video = SubElement(media, "video")
    vformat = SubElement(video, "format")
    sample = SubElement(vformat, "samplecharacteristics")
    _xml_text(sample, "width", str(width))
    _xml_text(sample, "height", str(height))
    vtrack = SubElement(video, "track")
    for clip in clips:
        _fcp_clip(vtrack, clip, rate, kind="video")
    audio = SubElement(media, "audio")
    if audio_segments:
        atrack = SubElement(audio, "track")
        for seg in audio_segments:
            _fcp_clip(atrack, seg, rate, kind="audio")
    body = tostring(root, encoding="utf-8")
    return b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body


def _fcp_clip(track: Element, clip: dict[str, Any], fps: int, *, kind: str) -> None:
    start = int(round(float(clip["start"]) * fps))
    dur = max(1, int(round(float(clip["duration"]) * fps)))
    item = SubElement(track, "clipitem")
    _xml_text(item, "name", str(clip.get("shot_code") or clip.get("path") or kind))
    _xml_text(item, "duration", str(dur))
    rate = SubElement(item, "rate")
    _xml_text(rate, "timebase", str(fps))
    _xml_text(rate, "ntsc", "FALSE")
    _xml_text(item, "start", str(start))
    _xml_text(item, "end", str(start + dur))
    _xml_text(item, "in", "0")
    _xml_text(item, "out", str(dur))
    file_el = SubElement(item, "file")
    _xml_text(file_el, "name", Path(str(clip.get("path") or "media")).name)
    path = str(clip.get("path") or "")
    _xml_text(file_el, "pathurl", path)


def _new_id() -> str:
    return uuid.uuid4().hex


def _draft_content(
    *,
    title: str,
    width: int,
    height: int,
    fps: int,
    clips: list[dict[str, Any]],
    audio_segments: list[dict[str, Any]],
    srt_path: str | None,
) -> dict[str, Any]:
    videos = []
    audios = []
    v_segments = []
    a_segments = []
    for clip in clips:
        material_id = _new_id()
        dur = _dur_us(clip["duration"])
        videos.append(
            {
                "id": material_id,
                "path": clip.get("path") or "",
                "duration": dur,
                "type": "video",
                "width": width,
                "height": height,
            }
        )
        v_segments.append(
            {
                "id": _new_id(),
                "material_id": material_id,
                "target_timerange": {"start": _dur_us(clip["start"]), "duration": dur},
                "source_timerange": {"start": 0, "duration": dur},
            }
        )
    for seg in audio_segments:
        material_id = _new_id()
        dur = _dur_us(seg["duration"])
        audios.append(
            {
                "id": material_id,
                "path": seg.get("path") or "",
                "duration": dur,
                "type": "audio",
            }
        )
        a_segments.append(
            {
                "id": _new_id(),
                "material_id": material_id,
                "target_timerange": {"start": _dur_us(seg["start"]), "duration": dur},
                "source_timerange": {"start": 0, "duration": dur},
            }
        )
    texts = []
    t_segments = []
    if srt_path:
        text_id = _new_id()
        texts.append({"id": text_id, "path": srt_path, "type": "subtitle"})
        t_segments.append({"id": _new_id(), "material_id": text_id})
    total = 0
    for clip in clips:
        total = max(total, _dur_us(float(clip["start"]) + float(clip["duration"])))
    for seg in audio_segments:
        total = max(total, _dur_us(float(seg["start"]) + float(seg["duration"])))
    return {
        "id": _new_id(),
        "draft_name": title,
        "fps": float(fps),
        "duration": total,
        "canvas_config": {"width": width, "height": height, "ratio": "original"},
        "materials": {"videos": videos, "audios": audios, "texts": texts},
        "tracks": [
            {"type": "video", "segments": v_segments},
            {"type": "audio", "segments": a_segments},
            {"type": "text", "segments": t_segments},
        ],
        "platform": {"app": "本地再铸", "target": "Jianying 5.x draft_content field layout", "verified": False},
    }


def export_jianying_draft(
    dest_dir: str | Path,
    *,
    shots: list[dict[str, Any]] | None = None,
    output_video: str | Path | None = None,
    silent_video: str | Path | None = None,
    audio: str | Path | None = None,
    srt: str | Path | None = None,
    title: str = "本地再铸导出",
    width: int = 720,
    height: int = 1280,
    fps: int = 24,
) -> Path:
    dest = Path(dest_dir)
    media = dest / "media"
    dest.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)

    copied_output = _copy(Path(output_video), media) if output_video else None
    copied_silent = _copy(Path(silent_video), media) if silent_video else None
    copied_audio = _copy(Path(audio), media) if audio else None
    copied_srt = _copy(Path(srt), media) if srt else None

    copied_sources: dict[str, Path] = {}
    if copied_audio and audio:
        copied_sources[str(Path(audio).resolve())] = copied_audio

    clips: list[dict[str, Any]] = []
    cursor = 0.0
    for i, sh in enumerate(shots or []):
        start = float(sh.get("start_time") if sh.get("start_time") is not None else cursor)
        dur = float(sh.get("duration_sec") or 0)
        if dur <= 0 and sh.get("end_time") is not None:
            dur = float(sh["end_time"]) - start
        if dur <= 0:
            dur = 1.0
        src = sh.get("output_path")
        local = _copy(Path(src), media / "shots") if src and Path(src).is_file() else None
        clips.append(
            {
                "index": i,
                "shot_code": sh.get("shot_code") or f"S{i+1:02d}",
                "start": start,
                "duration": dur,
                "path": _rel_to(local, dest) if local else None,
                "dialogue": sh.get("dialogue") or "",
                "speaker": sh.get("speaker") or "",
            }
        )
        cursor = start + dur
    total = cursor

    segments = build_audio_segments(shots)
    audio_rows: list[dict[str, Any]] = []
    for seg in segments:
        local = None
        source = seg.get("source")
        if source and Path(source).is_file():
            key = str(Path(source).resolve())
            local = copied_sources.get(key)
            if local is None:
                local = _copy(Path(source), media / "audio")
                if local is not None:
                    copied_sources[key] = local
        elif copied_audio is not None:
            local = copied_audio
        audio_rows.append(
            {
                "index": seg["index"],
                "shot_code": seg["shot_code"],
                "start": seg["start"],
                "duration": seg["duration"],
                "path": _rel_to(local, dest) if local else _rel_to(copied_audio, dest),
            }
        )

    project = {
        "title": title,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": total,
        "video": _rel_to(copied_output, dest),
        "silent": _rel_to(copied_silent, dest),
        "audio": _rel_to(copied_audio, dest),
        "srt": _rel_to(copied_srt, dest),
        "clips": clips,
        "audio_segments": audio_rows,
        "note": TARGET_NOTE,
        "paths": "relative",
        "jianying_verified": False,
    }
    (dest / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")

    root = Element(
        "timeline",
        {
            "title": title,
            "width": str(width),
            "height": str(height),
            "fps": str(fps),
            "paths": "relative",
        },
    )
    vtrack = SubElement(root, "video")
    for c in clips:
        el = SubElement(
            vtrack,
            "clip",
            {
                "shot_code": str(c["shot_code"]),
                "start": f"{c['start']:.3f}",
                "duration": f"{c['duration']:.3f}",
            },
        )
        if c.get("path"):
            el.set("path", c["path"])
    atrack = SubElement(root, "audio")
    for seg in audio_rows:
        attrs = {
            "shot_code": str(seg["shot_code"]),
            "start": f"{float(seg['start']):.3f}",
            "duration": f"{float(seg['duration']):.3f}",
        }
        if seg.get("path"):
            attrs["path"] = str(seg["path"])
        SubElement(atrack, "clip", attrs)
    strack = SubElement(root, "subtitles")
    if copied_srt:
        strack.set("path", _rel_to(copied_srt, dest) or "")
    xml = tostring(root, encoding="utf-8")
    (dest / "timeline.xml").write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml)

    (dest / "timeline.edl").write_text(
        render_edl(title=title, fps=fps, clips=[c for c in clips if c.get("path")], audio_segments=audio_rows),
        encoding="utf-8",
    )
    (dest / "timeline_fcp.xml").write_bytes(
        render_fcp_xml(
            title=title,
            width=width,
            height=height,
            fps=fps,
            clips=[c for c in clips if c.get("path")],
            audio_segments=audio_rows,
        )
    )

    draft_clips = clips or (
        [
            {
                "shot_code": "main",
                "start": 0.0,
                "duration": total,
                "path": _rel_to(copied_output or copied_silent, dest),
            }
        ]
        if (copied_output or copied_silent) and total > 0
        else []
    )
    draft_info = {
        "draft_name": title,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_us": _dur_us(total),
        "paths": "relative",
        "tracks": [
            {
                "type": "video",
                "segments": [
                    {
                        "id": c["shot_code"],
                        "target_timerange": {"start": _dur_us(c["start"]), "duration": _dur_us(c["duration"])},
                        "source_timerange": {"start": 0, "duration": _dur_us(c["duration"])},
                        "material": {"path": c.get("path") or ""},
                    }
                    for c in draft_clips
                ],
            },
            {
                "type": "audio",
                "segments": [
                    {
                        "id": seg["shot_code"],
                        "target_timerange": {"start": _dur_us(seg["start"]), "duration": _dur_us(seg["duration"])},
                        "source_timerange": {"start": 0, "duration": _dur_us(seg["duration"])},
                        "material": {"path": seg.get("path") or ""},
                    }
                    for seg in audio_rows
                ],
            },
            {
                "type": "subtitle",
                "segments": (
                    [{"id": "srt", "material": {"path": _rel_to(copied_srt, dest) or ""}}] if copied_srt else []
                ),
            },
        ],
        "exporter": {
            "app": "本地再铸",
            "format": "draft_info.json",
            "target": "Jianying 5.x draft field layout",
            "guaranteed_to_open": False,
            "verified": False,
            "note": TARGET_NOTE,
        },
    }
    (dest / "draft_info.json").write_text(json.dumps(draft_info, ensure_ascii=False, indent=2), encoding="utf-8")
    content = _draft_content(
        title=title,
        width=width,
        height=height,
        fps=fps,
        clips=[c for c in draft_clips if c.get("path")],
        audio_segments=audio_rows,
        srt_path=_rel_to(copied_srt, dest),
    )
    (dest / "draft_content.json").write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "README.txt").write_text(README, encoding="utf-8")
    return dest
