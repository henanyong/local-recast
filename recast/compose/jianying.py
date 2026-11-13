"""Jianying draft plus interchange files other NLEs can open.

The draft_content.json layout follows the field names commonly used by
Jianying 5.x drafts (microsecond ranges, relative media paths, separate
video and audio tracks). This environment has not opened the file in
Jianying, so the export does not claim that any version imports it.

timeline.edl is CMX 3600. timeline_fcp.xml is FCP 7 xmeml. Both use paths
relative to this folder.

先校验再建稿、写后自检（G9）
---------------------------
建稿入口 ``export_jianying_draft`` 默认先跑校验、写完跑自检，两步不过都
抛 ``JianyingDraftError``（附带中文原因列表），不写半截工程、不静默成功。

service 层 / UI 下一轮的调用方式（只调这里，不用自己拼检查）::

    from recast.compose.jianying import (
        JianyingDraftError,
        export_jianying_draft,
        validate_for_jianying,
        verify_jianying_draft,
    )

    # 1. 导出按钮先调校验，把 errors 逐条显示给用户，通过了才建稿。
    report = validate_for_jianying(
        dest_dir,
        shots=shots,
        output_video=output,
        audio=audio,
        srt=srt,
        dialogue=workdir / "dialogue.json",   # 生成路线才传；再铸路线不传
        reference_assets=[face_path, model_glb],  # 替换涉及的参考脸 / GLB
    )
    if not report.ok:
        show(report.errors)   # 每条都是中文，已写明文件、行号或时间段
        return
    # 2. 建稿（内部会再跑一次校验，写完跑自检；不过就抛异常）。
    dest = export_jianying_draft(dest_dir, shots=shots, ...)
    # 3. 也可以单独对已写出的草稿做自检（比如用户从磁盘恢复的旧草稿）。
    check = verify_jianying_draft(dest_dir)
    if not check.ok:
        show(check.errors)

注意：``JianyingDraftError`` 不要静默吞掉；校验不过时 ``export_jianying_draft``
一个文件也不写。台词表与字幕的条数/时间对齐只在“生成路线”成立（台词就是
字幕源头），再铸路线不要传 dialogue，否则会因来源不同而误报。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring


class JianyingDraftError(RuntimeError):
    """剪映导出在校验或自检阶段失败；``issues`` 是中文原因列表。"""

    def __init__(self, issues: list[str] | str):
        if isinstance(issues, str):
            issues = [issues]
        self.issues = list(issues)
        super().__init__("；".join(self.issues))


@dataclass
class JianyingCheckResult:
    """一次校验或自检的结果；``ok`` 为真才表示可以继续。"""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def messages(self) -> list[str]:
        lines = [f"错误：{m}" for m in self.errors]
        lines += [f"提醒：{m}" for m in self.warnings]
        lines += [f"未检查：{m}" for m in self.skipped]
        return lines


def _fmt_sec(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    mm, ss = divmod(whole, 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


_SRT_TS = re.compile(r"^(\d+):(\d{2}):(\d{2})[,.](\d{3})$")


def _srt_ts_to_sec(text: str) -> float | None:
    match = _SRT_TS.match(text.strip())
    if not match:
        return None
    hh, mm, ss, ms = (int(g) for g in match.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def _parse_srt(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse an SRT file. Returns (segments, errors) with block numbers."""
    errors: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return [], [f"字幕文件读不出来：{path}（{exc}）"]
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", raw.strip()) if b.strip()]
    if not blocks:
        return [], [f"字幕文件是空的：{path}"]
    segments: list[dict[str, Any]] = []
    for num, block in enumerate(blocks, start=1):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            errors.append(f"字幕第 {num} 条格式不对（只有 {len(lines)} 行）：{path}")
            continue
        stamp = lines[1] if lines[0].isdigit() else lines[0]
        text_lines = lines[2:] if lines[0].isdigit() else lines[1:]
        if "-->" not in stamp:
            errors.append(f"字幕第 {num} 条没有时间戳（应为“开始 --> 结束”）：{path}")
            continue
        start_s, _, end_s = stamp.partition("-->")
        start = _srt_ts_to_sec(start_s)
        end = _srt_ts_to_sec(end_s)
        if start is None or end is None:
            errors.append(
                f"字幕第 {num} 条时间戳读不懂：“{stamp.strip()}”（应为 00:00:00,000 格式）：{path}"
            )
            continue
        text = " ".join(text_lines).strip()
        if not text:
            errors.append(f"字幕第 {num} 条没有文字（{_fmt_sec(start)}–{_fmt_sec(end)}）：{path}")
            continue
        if end <= start:
            errors.append(
                f"字幕第 {num} 条结束时间早于或等于开始时间（{_fmt_sec(start)}–{_fmt_sec(end)}）：{path}"
            )
            continue
        if segments and start < segments[-1]["end"] - 0.001:
            errors.append(
                f"字幕第 {num} 条开始时间（{_fmt_sec(start)}）早于上一条的结束时间"
                f"（{_fmt_sec(segments[-1]['end'])}），时间重叠了：{path}"
            )
            continue
        segments.append({"index": num, "start": start, "end": end, "text": text})
    return segments, errors


def _parse_dialogue_file(path: Path) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Parse dialogue.json. Returns (lines, errors); no shots_plan fallback."""
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "[]")
    except ValueError as exc:
        return None, [f"台词表读不懂（不是合法 JSON）：{path}（{exc}）"]
    except OSError as exc:
        return None, [f"台词表读不出来：{path}（{exc}）"]
    if not isinstance(data, list):
        return None, [f"台词表不是一个列表：{path}"]
    return [dict(line) for line in data], []


def _probe_duration(path: Path) -> float | None:
    """Best-effort media duration via ffprobe; None means unmeasurable."""
    try:
        from recast.ffmpeg import probe_video

        info = probe_video(path)
    except Exception:
        return None
    try:
        dur = float(info.duration)
    except (AttributeError, TypeError, ValueError):
        return None
    return dur if dur > 0 else None


def _check_media_file(
    result: JianyingCheckResult,
    path: str | Path | None,
    label: str,
) -> Path | None:
    """Existence + readability + non-empty check for one input file."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        result.errors.append(f"{label}不存在：{p}")
        return None
    if not p.is_file():
        result.errors.append(f"{label}不是一个文件：{p}")
        return None
    try:
        if p.stat().st_size == 0:
            result.errors.append(f"{label}是空文件（0 字节）：{p}")
            return None
        with p.open("rb") as fh:
            fh.read(1)
    except OSError as exc:
        result.errors.append(f"{label}读不出来：{p}（{exc}）")
        return None
    return p


def validate_for_jianying(
    dest_dir: str | Path,
    *,
    shots: list[dict[str, Any]] | None = None,
    output_video: str | Path | None = None,
    silent_video: str | Path | None = None,
    audio: str | Path | None = None,
    srt: str | Path | None = None,
    dialogue: str | Path | None = None,
    reference_assets: list[str | Path] | None = None,
    title: str = "本地再铸导出",
) -> JianyingCheckResult:
    """Validate every input before building a Jianying draft.

    Nothing is written here. Any error means "do not build": the caller
    should show ``result.errors`` and stop. Warnings do not block the build
    but name things the draft will silently drop (e.g. a shot with no
    rendered file). ``dialogue`` is only for the script-generation route,
    where dialogue lines are the subtitle source; do not pass it on the
    recast route or the count/time alignment check will misfire.
    """
    result = JianyingCheckResult()
    dest = Path(dest_dir)
    if not title.strip():
        result.errors.append("导出标题是空的，请填一个标题再导出。")

    # 2. 源视频 / 静音视频 / 音频：存在、可读、非空。
    output_p = _check_media_file(result, output_video, "成片视频")
    silent_p = _check_media_file(result, silent_video, "静音视频")
    _check_media_file(result, audio, "混音音频")

    # 3. 分镜：时长合法、顺序对、引用的文件存在。
    shots = list(shots or [])
    timeline_end = 0.0
    prev_start: float | None = None
    for i, sh in enumerate(shots):
        code = str(sh.get("shot_code") or f"S{i + 1:02d}")
        try:
            start = float(sh.get("start_time") if sh.get("start_time") is not None else timeline_end)
        except (TypeError, ValueError):
            result.errors.append(f"{code} 的开始时间读不懂：{sh.get('start_time')!r}")
            continue
        try:
            dur = float(sh.get("duration_sec") or 0)
        except (TypeError, ValueError):
            result.errors.append(f"{code} 的时长读不懂：{sh.get('duration_sec')!r}")
            continue
        if dur <= 0 and sh.get("end_time") is not None:
            try:
                dur = float(sh["end_time"]) - start
            except (TypeError, ValueError):
                dur = 0
        if start < 0:
            result.errors.append(f"{code} 的开始时间是负数（{start} 秒）。")
        if dur <= 0:
            result.errors.append(f"{code} 没有有效时长（开始 {_fmt_sec(start)}，时长 {dur} 秒），导不出这个镜头。")
            continue
        if prev_start is not None and start < prev_start - 0.001:
            result.errors.append(f"{code} 的开始时间（{_fmt_sec(start)}）比上一个镜头还早，时间轴顺序乱了。")
        prev_start = start
        timeline_end = max(timeline_end, start + dur)
        src = sh.get("output_path")
        if src:
            _check_media_file(result, src, f"{code} 的镜头成片")
        else:
            result.warnings.append(f"{code} 没有镜头成片文件，导出时该镜头不会出现在剪映草稿里。")
        audio_path = sh.get("audio_path")
        if audio_path:
            _check_media_file(result, audio_path, f"{code} 的对白音频")
        for key, label in (
            ("face_path", "参考脸"),
            ("reference_face", "参考脸"),
            ("body_model", "身体模型"),
            ("model_path", "身体模型"),
        ):
            ref = sh.get(key)
            if ref:
                _check_media_file(result, ref, f"{code} 的{label}")
        if sh.get("dialogue") and sh.get("audio_duration") in (None, "") and not sh.get("audio_path"):
            result.warnings.append(f"{code} 有台词但没有配音文件，剪映里这句将是静音。")

    # 4. 字幕 SRT：存在、时间戳合法、文字非空、时间不重叠。
    srt_segments: list[dict[str, Any]] = []
    if srt is not None:
        srt_p = _check_media_file(result, srt, "字幕文件")
        if srt_p is not None:
            srt_segments, srt_errors = _parse_srt(srt_p)
            result.errors.extend(srt_errors)

    # 5. 台词表 dialogue.json：结构对、每句有文字有时间；如有字幕则对齐。
    if dialogue is not None:
        dlg_p = _check_media_file(result, dialogue, "台词表")
        if dlg_p is not None:
            lines, dlg_errors = _parse_dialogue_file(dlg_p)
            result.errors.extend(dlg_errors)
            if lines is not None:
                for line in lines:
                    code = str(line.get("shot_code") or "（未编号）")
                    text = str(line.get("text") or line.get("dialogue") or "").strip()
                    if not text:
                        result.errors.append(f"台词表里 {code} 这一句没有文字：{dlg_p}")
                    try:
                        ls = float(line.get("start_time") or 0)
                        le = float(line.get("end_time") or 0)
                    except (TypeError, ValueError):
                        result.errors.append(f"台词表里 {code} 的时间读不懂：{dlg_p}")
                        continue
                    if le <= ls:
                        result.errors.append(
                            f"台词表里 {code} 的结束时间（{_fmt_sec(le)}）早于或等于开始时间"
                            f"（{_fmt_sec(ls)}）：{dlg_p}"
                        )
                if srt_segments:
                    if len(lines) != len(srt_segments):
                        result.errors.append(
                            f"台词表有 {len(lines)} 句，字幕有 {len(srt_segments)} 条，条数对不上："
                            f"台词表 {dlg_p}，字幕 {srt}"
                        )
                    else:
                        for line, seg in zip(lines, srt_segments):
                            code = str(line.get("shot_code") or "（未编号）")
                            try:
                                ls = float(line.get("start_time") or 0)
                                le = float(line.get("end_time") or 0)
                            except (TypeError, ValueError):
                                continue
                            if le <= seg["start"] or ls >= seg["end"]:
                                result.errors.append(
                                    f"台词表里 {code}（{_fmt_sec(ls)}–{_fmt_sec(le)}）"
                                    f"和字幕第 {seg['index']} 条（{_fmt_sec(seg['start'])}–"
                                    f"{_fmt_sec(seg['end'])}）时间对不上：台词表 {dlg_p}，字幕 {srt}"
                                )

    # 6. 人物替换涉及的参考脸 / GLB：存在；GLB 再验文件头。
    for asset in reference_assets or []:
        p = _check_media_file(result, asset, "替换参考资产")
        if p is None:
            continue
        if p.suffix.lower() == ".glb":
            try:
                from recast.assets3d.glb import validate_glb

                validate_glb(p)
            except Exception as exc:  # GlbError 或导入失败都当作该项不过
                result.errors.append(f"身体模型不是有效的 GLB：{p}（{exc}）")

    # 7. 导出时长与源视频时长基本一致，防止静默截断。
    source = output_p or silent_p
    source_dur = _probe_duration(source) if source is not None else None
    if timeline_end > 0 and source_dur is not None:
        diff = abs(timeline_end - source_dur)
        if diff > max(1.0, 0.05 * source_dur):
            result.errors.append(
                f"导出时长 {_fmt_sec(timeline_end)}（{timeline_end:.1f} 秒）与源视频"
                f" {_fmt_sec(source_dur)}（{source_dur:.1f} 秒）相差 {diff:.1f} 秒，"
                "可能被静默截断了，请检查分镜时长再导出。"
            )
    elif timeline_end > 0:
        result.skipped.append("源视频时长测不出来（不是标准视频文件或没有 ffprobe），没法核对导出时长。")
    elif source_dur is not None and source_dur > 0:
        result.warnings.append(
            f"没有分镜，按成片时长 {_fmt_sec(source_dur)} 导出一整个镜头。"
        )
        timeline_end = source_dur
    if timeline_end <= 0 and not result.errors:
        result.errors.append("没有可导出的镜头或成片时长，导不出任何内容。")

    # 1. 输出目录：必须能建、能写。放在最后——只有其他项都通过才建目录，
    # 校验不过时磁盘上什么都不留，不写半截工程。
    if not result.errors:
        try:
            if dest.exists() and not dest.is_dir():
                result.errors.append(f"输出位置已被一个文件占了，不是目录：{dest}")
            else:
                dest.mkdir(parents=True, exist_ok=True)
                probe = dest / ".write-test"
                probe.write_bytes(b"ok")
                probe.unlink()
        except OSError as exc:
            result.errors.append(f"输出目录建不出来或写不进去：{dest}（{exc}）")
    return result


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


def verify_jianying_draft(dest_dir: str | Path) -> JianyingCheckResult:
    """Re-read a written Jianying draft and check it is self-consistent.

    Checks: every expected file exists and is non-zero; the JSON/XML files
    parse; every material path is non-empty and resolves to a real file
    under the draft folder; every timeline segment points at a material
    that exists (no dangling references); the clip/audio/text counts agree
    across project.json, draft_info.json and draft_content.json; the
    timeline is in order; the durations agree across the three files.
    """
    result = JianyingCheckResult()
    dest = Path(dest_dir)
    if not dest.is_dir():
        result.errors.append(f"草稿目录不存在：{dest}")
        return result

    expected = [
        "project.json",
        "timeline.xml",
        "timeline.edl",
        "timeline_fcp.xml",
        "draft_info.json",
        "draft_content.json",
        "README.txt",
    ]
    for name in expected:
        p = dest / name
        if not p.is_file():
            result.errors.append(f"草稿缺文件：{name}（{dest} 里没有）")
        elif p.stat().st_size == 0:
            result.errors.append(f"草稿文件是空的：{name}（{dest}）")

    def _load_json(name: str) -> dict[str, Any] | None:
        p = dest / name
        if not p.is_file() or p.stat().st_size == 0:
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            result.errors.append(f"{name} 不是合法 JSON，读不懂：{dest}（{exc}）")
            return None
        if not isinstance(data, dict):
            result.errors.append(f"{name} 的顶层不是一个对象：{dest}")
            return None
        return data

    project = _load_json("project.json")
    draft_info = _load_json("draft_info.json")
    content = _load_json("draft_content.json")

    for name in ("timeline.xml", "timeline_fcp.xml"):
        p = dest / name
        if p.is_file() and p.stat().st_size > 0:
            try:
                from xml.etree.ElementTree import parse as _xml_parse

                _xml_parse(p)
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                result.errors.append(f"{name} 不是合法 XML：{dest}（{exc}）")

    def _resolve(rel: str | None, where: str) -> Path | None:
        if not rel:
            result.errors.append(f"{where} 引用的素材路径是空的。")
            return None
        target = (dest / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if not target.is_file():
            result.errors.append(f"{where} 指向的文件不存在：{rel}（应在 {dest} 里）")
            return None
        if target.stat().st_size == 0:
            result.errors.append(f"{where} 指向的是空文件：{rel}")
            return None
        return target

    # project.json：每个媒体字段、每个有路径的镜头、每段音频都要能找到。
    if project is not None:
        for key in ("video", "silent", "audio", "srt"):
            if project.get(key):
                _resolve(str(project[key]), f"project.json 的 {key}")
        clips = project.get("clips") or []
        if not isinstance(clips, list):
            result.errors.append("project.json 的 clips 不是一个列表。")
            clips = []
        prev_start: float | None = None
        for i, clip in enumerate(clips):
            code = str(clip.get("shot_code") or f"第 {i + 1} 个镜头")
            if clip.get("path"):
                _resolve(str(clip["path"]), f"project.json 镜头 {code}")
            try:
                start = float(clip.get("start") or 0)
                dur = float(clip.get("duration") or 0)
            except (TypeError, ValueError):
                result.errors.append(f"project.json 镜头 {code} 的时间读不懂。")
                continue
            if dur <= 0:
                result.errors.append(f"project.json 镜头 {code} 的时长是 {dur} 秒。")
            if prev_start is not None and start < prev_start - 0.001:
                result.errors.append(f"project.json 镜头 {code} 的开始时间比上一个镜头还早，时间轴顺序乱了。")
            prev_start = start
        for i, seg in enumerate(project.get("audio_segments") or []):
            code = str(seg.get("shot_code") or f"第 {i + 1} 段音频")
            if seg.get("path"):
                _resolve(str(seg["path"]), f"project.json 音频 {code}")

    # draft_content.json：素材登记与时间轴段互相引用，不能有空引用或断链。
    if content is not None:
        materials = content.get("materials") or {}
        videos = materials.get("videos") or []
        audios = materials.get("audios") or []
        texts = materials.get("texts") or []
        material_ids: set[str] = set()
        for kind, items in (("视频", videos), ("音频", audios), ("字幕", texts)):
            for j, item in enumerate(items):
                mid = str(item.get("id") or "")
                if not mid:
                    result.errors.append(f"draft_content.json 第 {j + 1} 个{kind}素材没有 id。")
                    continue
                if mid in material_ids:
                    result.errors.append(f"draft_content.json 有重复的素材 id：{mid}")
                material_ids.add(mid)
                _resolve(str(item.get("path") or ""), f"draft_content.json {kind}素材 {mid}")
        tracks = content.get("tracks") or []
        for track in tracks:
            ttype = str(track.get("type") or "未知")
            for seg in track.get("segments") or []:
                mid = str(seg.get("material_id") or "")
                if not mid:
                    result.errors.append(f"draft_content.json {ttype}轨有一段没有 material_id，挂不上素材。")
                elif mid not in material_ids:
                    result.errors.append(
                        f"draft_content.json {ttype}轨有一段指向不存在的素材 id：{mid}，是断链。"
                    )
                timerange = seg.get("target_timerange")
                if timerange is None:
                    # 字幕轨的段没有时间范围：它引用的是整份 SRT 文件，不是时间线片段。
                    continue
                try:
                    if float(timerange.get("duration") or 0) <= 0:
                        result.errors.append(f"draft_content.json {ttype}轨有一段时长是 0。")
                except (TypeError, ValueError):
                    result.errors.append(f"draft_content.json {ttype}轨有一段时间读不懂。")
        # 数量对齐：视频/音频/字幕条数三处文件要一致。
        if project is not None:
            clips_with_path = [c for c in (project.get("clips") or []) if c.get("path")]
            if len(videos) != len(clips_with_path):
                result.errors.append(
                    f"视频素材数对不上：project.json 有 {len(clips_with_path)} 个带路径的镜头，"
                    f"draft_content.json 登记了 {len(videos)} 个视频素材。"
                )
            audio_rows = project.get("audio_segments") or []
            if len(audios) != len(audio_rows):
                result.errors.append(
                    f"音频素材数对不上：project.json 有 {len(audio_rows)} 段音频，"
                    f"draft_content.json 登记了 {len(audios)} 个音频素材。"
                )
            has_srt = bool(project.get("srt"))
            if has_srt and not texts:
                result.errors.append("project.json 说有字幕，但 draft_content.json 里一个字幕素材都没登记。")
            if not has_srt and texts:
                result.errors.append("draft_content.json 登记了字幕素材，但 project.json 里没有字幕文件。")
        # 时长对齐：三个文件的总时长要一致（允许 1 毫秒舍入）。
        try:
            content_dur = float(content.get("duration") or 0) / 1_000_000
        except (TypeError, ValueError):
            content_dur = -1
        if content_dur < 0:
            result.errors.append("draft_content.json 的 duration 读不懂。")
        elif project is not None:
            try:
                project_dur = float(project.get("duration_sec") or 0)
            except (TypeError, ValueError):
                project_dur = -1
            if project_dur >= 0 and abs(project_dur - content_dur) > 0.001:
                result.errors.append(
                    f"总时长对不上：project.json 是 {project_dur:.3f} 秒，"
                    f"draft_content.json 是 {content_dur:.3f} 秒。"
                )
        if draft_info is not None:
            try:
                info_dur = float(draft_info.get("duration_us") or 0) / 1_000_000
            except (TypeError, ValueError):
                info_dur = -1
            if info_dur >= 0 and content_dur >= 0 and abs(info_dur - content_dur) > 0.001:
                result.errors.append(
                    f"总时长对不上：draft_info.json 是 {info_dur:.3f} 秒，"
                    f"draft_content.json 是 {content_dur:.3f} 秒。"
                )
            info_video_segs = (draft_info.get("tracks") or [{}])[0].get("segments") or []
            project_clips = project.get("clips") or []
            # 写出逻辑：无分镜但有成片且总时长 > 0 时，draft_info 会写一段 "main"。
            expected_video_segs = len(project_clips)
            if not project_clips and (project.get("video") or project.get("silent")):
                try:
                    if float(project.get("duration_sec") or 0) > 0:
                        expected_video_segs = 1
                except (TypeError, ValueError):
                    pass
            if len(info_video_segs) != expected_video_segs:
                result.errors.append(
                    f"视频段数对不上：project.json 对应 {expected_video_segs} 段，"
                    f"draft_info.json 视频轨有 {len(info_video_segs)} 段。"
                )

    edl = dest / "timeline.edl"
    if edl.is_file() and edl.stat().st_size > 0 and project is not None:
        text = edl.read_text(encoding="utf-8", errors="replace")
        for clip in (project.get("clips") or []):
            if clip.get("path") and "FROM CLIP NAME" not in text:
                result.errors.append("timeline.edl 里没有镜头记录，时间线可能是空的。")
                break
    return result


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
    total_us: int | None = None,
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
    if total_us:
        # 草稿总时长是时间线全长：即使最后一个镜头没有成片文件（不登记
        # 素材），时间线本身仍是完整的；只按带路径的镜头算会静默缩短。
        total = max(total, total_us)
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
    dialogue: str | Path | None = None,
    reference_assets: list[str | Path] | None = None,
    title: str = "本地再铸导出",
    width: int = 720,
    height: int = 1280,
    fps: int = 24,
    output_root: str | Path | None = None,
    store: Any | None = None,
    validate: bool = True,
    verify: bool = True,
) -> Path:
    """Write a Jianying 5.x draft plus EDL / FCP XML interchange files.

    When ``output_root`` is given, each shot's speaker is canonicalized to
    its character folder's display name (a store id or auto name becomes the
    folder name); without a usable folder the speaker is left unchanged.

    ``validate`` runs :func:`validate_for_jianying` first and raises
    :class:`JianyingDraftError` before writing anything when an input is
    bad. ``verify`` runs :func:`verify_jianying_draft` after writing and
    raises when the written draft fails self-check. Pass ``dialogue`` only
    on the script-generation route (dialogue lines are the subtitle source
    there); ``reference_assets`` are face / GLB files the replace step used.
    """
    from recast.core.character_assets import canonical_speaker_name

    if validate:
        report = validate_for_jianying(
            dest_dir,
            shots=shots,
            output_video=output_video,
            silent_video=silent_video,
            audio=audio,
            srt=srt,
            dialogue=dialogue,
            reference_assets=reference_assets,
            title=title,
        )
        if not report.ok:
            raise JianyingDraftError(report.errors)

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
        speaker = str(sh.get("speaker") or "")
        if output_root is not None and speaker:
            speaker = canonical_speaker_name(output_root, speaker, store=store)
        clips.append(
            {
                "index": i,
                "shot_code": sh.get("shot_code") or f"S{i+1:02d}",
                "start": start,
                "duration": dur,
                "path": _rel_to(local, dest) if local else None,
                "dialogue": sh.get("dialogue") or "",
                "speaker": speaker,
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
        total_us=_dur_us(total) if total > 0 else None,
    )
    (dest / "draft_content.json").write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "README.txt").write_text(README, encoding="utf-8")
    if verify:
        check = verify_jianying_draft(dest)
        if not check.ok:
            raise JianyingDraftError(check.errors)
    return dest
