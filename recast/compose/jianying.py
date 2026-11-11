"""Jianying / CapCut-CN draft exporter.

This writes a documented draft folder that lists the timeline (video clips,
audio, SRT) in both a simple project.json / timeline.xml and a draft_info.json
inspired by common Jianying draft fields.

It is NOT guaranteed to open in every Jianying version — the official format
changes frequently and has not been verified against a specific release here.
The delivery folder is still always produced so you can hand the timeline to
another tool or try opening it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring


README = """剪映 / 剪映国际（CapCut-CN）工程导出说明
========================================

本目录由「本地再铸」写出，包含：

- project.json     简单时间线清单（镜头、音轨、字幕），结构稳定，便于对照
- timeline.xml     同上内容的 XML
- draft_info.json  接近常见剪映草稿字段的一份工程描述（素材路径、轨道、时长）
- media/           成片、分镜静音片段、对白音频、SRT 的副本

关于能否直接用剪映打开
  剪映草稿格式随版本变化，本导出器没有对某一确定版本做打开验证，
  因此不承诺「在每一个剪映版本里都能一键打开」。
  请先用 project.json / timeline.xml 核对该时间线；若你的剪映版本
  能识别 draft_info.json，可以尝试把本目录放到剪映草稿文件夹后再打开。

时间线内容
  视频轨：按分镜顺序排列的静音镜头（若只有成片，则放整段成片）
  音频轨：对白 / 旁白（若已生成）
  字幕轨：SRT 条目
"""


def _dur_us(seconds: float) -> int:
    return int(round(float(seconds) * 1_000_000))


def _copy(src: Path, dest_dir: Path) -> Path | None:
    if not src or not Path(src).is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(src).name
    if dest.resolve() != Path(src).resolve():
        shutil.copy2(src, dest)
    return dest


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
                "path": str(local) if local else None,
                "dialogue": sh.get("dialogue") or "",
                "speaker": sh.get("speaker") or "",
            }
        )
        cursor = start + dur
    total = cursor
    if total <= 0 and copied_output:
        total = 0.0

    project = {
        "title": title,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": total,
        "video": str(copied_output) if copied_output else None,
        "silent": str(copied_silent) if copied_silent else None,
        "audio": str(copied_audio) if copied_audio else None,
        "srt": str(copied_srt) if copied_srt else None,
        "clips": clips,
        "note": "简单时间线清单。剪映草稿格式未对具体版本做打开验证。",
    }
    (dest / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")

    # timeline.xml
    root = Element("timeline", {"title": title, "width": str(width), "height": str(height), "fps": str(fps)})
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
    if copied_audio:
        SubElement(atrack, "clip", {"path": str(copied_audio), "start": "0", "duration": f"{total:.3f}"})
    strack = SubElement(root, "subtitles")
    if copied_srt:
        strack.set("path", str(copied_srt))
    xml = tostring(root, encoding="utf-8")
    (dest / "timeline.xml").write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml)

    draft_info = {
        "draft_name": title,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_us": _dur_us(total),
        "tracks": [
            {
                "type": "video",
                "segments": [
                    {
                        "id": c["shot_code"],
                        "target_timerange": {"start": _dur_us(c["start"]), "duration": _dur_us(c["duration"])},
                        "material": {"path": c["path"] or (str(copied_silent) if copied_silent else str(copied_output) if copied_output else "")},
                    }
                    for c in clips
                ]
                or (
                    [
                        {
                            "id": "main",
                            "target_timerange": {"start": 0, "duration": _dur_us(total)},
                            "material": {"path": str(copied_output or copied_silent or "")},
                        }
                    ]
                    if (copied_output or copied_silent)
                    else []
                ),
            },
            {
                "type": "audio",
                "segments": (
                    [
                        {
                            "id": "dialogue",
                            "target_timerange": {"start": 0, "duration": _dur_us(total)},
                            "material": {"path": str(copied_audio)},
                        }
                    ]
                    if copied_audio
                    else []
                ),
            },
            {
                "type": "subtitle",
                "segments": (
                    [{"id": "srt", "material": {"path": str(copied_srt)}}] if copied_srt else []
                ),
            },
        ],
        "exporter": {
            "app": "本地再铸",
            "format": "draft_info.json (best-effort)",
            "guaranteed_to_open": False,
            "note": "未对具体剪映版本做打开验证，请同时参考 project.json。",
        },
    }
    (dest / "draft_info.json").write_text(json.dumps(draft_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "README.txt").write_text(README, encoding="utf-8")
    return dest
