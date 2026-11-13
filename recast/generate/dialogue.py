"""Per-line dialogue editing and selective re-dubbing.

Dialogue lives in the episode workdir's ``dialogue.json`` — one line per
shot, grouped by speaker (character):

    {"shot_code": "F2-L2", "speaker": "温念", "character": "温念",
     "text": "你干什么！", "start_time": 5.0, "end_time": 8.0,
     "duration_sec": 3.0, "audio_path": ".../voice/lines/F2-L2.wav",
     "audio_duration": 2.4, "dirty": false}

Editing a line only marks it dirty. Re-dubbing synthesizes *only* the dirty
(or explicitly named) lines with the speaker's current voice + timbre, writes
each new wav to that character's ``voice/lines/<shot_code>.wav``, registers it
in the character folder's meta.json, and re-mixes the whole episode track
(``dialogue_mix.wav`` + SRT) from all lines. Audio is never sped up: a line
whose new audio is longer than its slot pushes later lines back, exactly
like the first dub.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from recast.ffmpeg import run_ffmpeg

log = logging.getLogger("recast")

DIALOGUE_FILENAME = "dialogue.json"


def _line_from_shot(sh: dict[str, Any]) -> dict[str, Any]:
    start = float(sh.get("narration_start", sh.get("start_time") or 0) or 0)
    end = sh.get("narration_end", sh.get("end_time"))
    duration = float(sh.get("duration_sec") or 0)
    if end is None:
        end = start + (duration or float(sh.get("audio_duration") or 0) or 0.5)
    return {
        "shot_code": str(sh.get("shot_code") or f"S{sh.get('index_no', 0)}"),
        "speaker": str(sh.get("speaker") or ""),
        "character": str(sh.get("character_id") or sh.get("speaker") or ""),
        "text": str(sh.get("dialogue") or sh.get("text") or ""),
        "start_time": start,
        "end_time": float(end),
        "duration_sec": float(end) - start,
        "audio_path": sh.get("audio_path") or None,
        "audio_duration": float(sh.get("audio_duration") or 0),
        "dirty": False,
    }


def load_dialogue(workdir: str | Path) -> list[dict[str, Any]]:
    """Read dialogue.json; build it from shots_plan.json when absent."""
    workdir = Path(workdir)
    path = workdir / DIALOGUE_FILENAME
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "[]")
        except ValueError as exc:
            raise ValueError(f"读不懂对白文件 {path}：{exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"对白文件 {path} 不是一个列表。")
        return [dict(line) for line in data]
    plan = workdir / "shots_plan.json"
    if not plan.is_file():
        raise FileNotFoundError(
            f"找不到对白（{path} 不存在，也没有 shots_plan.json）。请先跑一次配音。"
        )
    try:
        shots = json.loads(plan.read_text(encoding="utf-8") or "[]")
    except ValueError as exc:
        raise ValueError(f"读不懂分镜计划 {plan}：{exc}") from exc
    if isinstance(shots, dict):
        shots = shots.get("shots") or []
    lines = [_line_from_shot(sh) for sh in (shots or [])]
    save_dialogue(workdir, lines)
    return lines


def save_dialogue(workdir: str | Path, lines: list[dict[str, Any]]) -> Path:
    """Write dialogue.json."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / DIALOGUE_FILENAME
    path.write_text(
        json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def edit_line(workdir: str | Path, shot_code: str, new_text: str) -> dict[str, Any]:
    """Change one line's text. Only marks it dirty; nothing is re-dubbed here."""
    text = str(new_text or "").strip()
    if not text:
        raise ValueError("对白文本不能为空。改成空就是删掉这句，请用删除而不是改空。")
    lines = load_dialogue(workdir)
    for line in lines:
        if line.get("shot_code") == shot_code:
            old = line.get("text") or ""
            line["text"] = text
            line["dirty"] = True
            save_dialogue(workdir, lines)
            log.info("对白 %s 已改（%s -> %s），待重配。", shot_code, old, text)
            return dict(line)
    raise FileNotFoundError(f"对白里没有 {shot_code} 这一句。")


def resolve_speaker_voice(
    store: Any,
    output_root: str | Path,
    speaker: str,
) -> tuple[str | None, dict[str, Any]]:
    """(voice, timbre) for a speaker name: store character, else folder meta.

    Precedence is kept from P2: the store character's voice wins, the folder
    only fills in when the store has none. The folder lookup itself lives in
    the shared resolver (``recast.core.character_assets``) so the folder
    parsing is not duplicated across the application lines.
    """
    from recast.core.character_assets import resolve_character_assets
    from recast.generate.timbre import normalize_timbre

    voice: str | None = None
    if store is not None:
        try:
            for ch in store.list_characters():
                if str(ch.get("display_name") or ch.get("name") or "") == speaker:
                    voice = str(ch.get("voice") or ch.get("voice_id") or "").strip() or None
                    break
        except Exception:
            voice = None
    assets = resolve_character_assets(output_root, speaker, store=store)
    if assets is None:
        return voice, normalize_timbre(None)
    if voice is None:
        voice = assets.tts_voice
    return voice, assets.timbre


def redub_lines(
    workdir: str | Path,
    *,
    shot_codes: list[str] | None = None,
    output_root: str | Path | None = None,
    store: Any = None,
    synthesize: Callable[..., dict[str, Any]] | None = None,
    remix: bool = True,
) -> dict[str, Any]:
    """Re-dub only the changed lines, then re-mix the episode track.

    Lines are picked by ``shot_codes`` or by their dirty flag. Each new wav
    goes to ``<character>/voice/lines/<shot_code>.wav`` and is registered in
    that character folder's meta.json ``voice.dubbed_lines``. ``synthesize``
    defaults to recast.generate.tts.synthesize_line (tests inject a stub).
    """
    from recast.core.characters import character_folder, load_meta, save_meta
    from recast.generate.tts import synthesize_line as _synthesize

    workdir = Path(workdir)
    lines = load_dialogue(workdir)
    if output_root is None:
        raise ValueError("重配音需要 output_root（人物文件夹的根目录）。")
    synth = synthesize or _synthesize
    wanted = {str(c) for c in (shot_codes or [])}
    targets = [
        line
        for line in lines
        if (wanted and line.get("shot_code") in wanted)
        or (not wanted and line.get("dirty"))
    ]
    report: dict[str, Any] = {"redubbed": [], "skipped": [], "changed": len(targets)}
    for line in targets:
        code = str(line.get("shot_code"))
        speaker = str(line.get("speaker") or line.get("character") or "")
        voice, timbre = resolve_speaker_voice(store, output_root, speaker)
        folder = character_folder(output_root, speaker or "未命名")
        lines_dir = folder / "voice" / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)
        dest = lines_dir / f"{code}.wav"
        rec = synth(
            str(line.get("text") or ""),
            dest,
            voice=voice,
            timbre=timbre,
        )
        if rec.get("skipped") or not rec.get("path"):
            report["skipped"].append(
                {"shot_code": code, "speaker": speaker, "reason": rec.get("message") or "配音跳过"}
            )
            continue
        line["audio_path"] = rec.get("path")
        line["audio_duration"] = float(rec.get("duration") or 0)
        line["dirty"] = False
        # Register the line wav in the character folder's meta.
        if folder.is_dir():
            meta = load_meta(folder) or {}
            voice_meta = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
            dubbed = dict(voice_meta.get("dubbed_lines") or {})
            dubbed[code] = str(dest)
            meta["voice"] = {**voice_meta, "dubbed_lines": dubbed}
            save_meta(folder, meta)
        report["redubbed"].append(
            {"shot_code": code, "speaker": speaker, "path": str(dest),
             "duration": line["audio_duration"], "source": rec.get("source")}
        )
    save_dialogue(workdir, lines)
    if remix:
        report["remix"] = remix_dialogue(workdir)
    log.info(
        "重配音完成：重配 %s 句，跳过 %s 句。",
        len(report["redubbed"]),
        len(report["skipped"]),
    )
    return report


def remix_dialogue(workdir: str | Path) -> dict[str, Any]:
    """Re-mix every line's wav into dialogue_mix.wav + SRT (never sped up).

    A line whose audio outgrew its slot pushes later lines back, exactly
    like the first dub. If workdir/output.mp4 exists, its audio track is
    replaced with the new mix (video stream copied untouched).
    """
    from recast.generate.stitch import mix_filter_graph, shots_to_segments, write_srt
    from recast.templates.duration import extend_timeline_for_audio

    workdir = Path(workdir)
    lines = load_dialogue(workdir)
    for line in lines:
        line.setdefault("dialogue", line.get("text") or "")
    confirmations = extend_timeline_for_audio(lines)
    for note in confirmations:
        log.warning("%s", note["message"])
    items: list[tuple[float, Path]] = []
    for line in lines:
        path = line.get("audio_path")
        if not path or not Path(str(path)).is_file():
            continue
        items.append((float(line.get("start_time") or 0), Path(str(path))))
    result: dict[str, Any] = {
        "mixed": None,
        "srt": None,
        "remuxed": None,
        "remux_error": None,
        "confirmations": confirmations,
    }
    if not items:
        save_dialogue(workdir, lines)
        return result
    last_end = 0.0
    for line in lines:
        end = float(line.get("end_time") or 0)
        start = float(line.get("start_time") or 0)
        last_end = max(last_end, end, start + float(line.get("audio_duration") or 0))
    total = max(last_end, 1.0)
    args: list[str] = ["-y"]
    delays: list[int] = []
    for start, path in items:
        args += ["-i", str(path)]
        delays.append(max(0, int(round(start * 1000))))
    graph = mix_filter_graph(delays, total)
    mixed = workdir / "dialogue_mix.wav"
    args += ["-filter_complex", graph, "-map", "[aout]", "-t", f"{total:.3f}", str(mixed)]
    run_ffmpeg(args)
    result["mixed"] = str(mixed)
    segs = shots_to_segments(lines)
    srt = workdir / "subtitles.srt"
    write_srt(segs, srt)
    result["srt"] = str(srt)
    save_dialogue(workdir, lines)
    output = workdir / "output.mp4"
    if output.is_file():
        tmp = workdir / "output.remux.mp4"
        try:
            run_ffmpeg([
                "-y", "-i", str(output), "-i", str(mixed),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp),
            ])
            os.replace(tmp, output)
            result["remuxed"] = str(output)
        except Exception as exc:
            result["remux_error"] = f"成片重混音轨失败（对白混音 dialogue_mix.wav 和 SRT 已更新）：{exc}"
            log.warning("%s", result["remux_error"])
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return result
