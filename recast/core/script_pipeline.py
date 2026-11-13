"""Second entry: generate a new episode from script / storyboard + dialogue table.

Silent picture first, then dub. Both entries join at replace / compose / export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from recast.core.constants import ST_NEED_ASSETS, ST_REVIEW
from recast.core.state_machine import format_missing
from recast.core.timing import timed_run
from recast.runtime import check_cancel
from recast.store.db import Store
from recast.templates.duration import check_shots
from recast.templates.fill import apply_pack_to_shots
from recast.templates.pack import TemplatePack, load_pack, pin_pack_for_task
from recast.templates.parse import attach_dialogue, parse_dialogue_table, parse_storyboard
from recast.templates.validate import validate_pack

log = logging.getLogger("recast")

ProgressFn = Callable[[str, str, float], None]


def _export_specs(pack: TemplatePack) -> dict[str, Any]:
    script = pack.script or {}
    export = script.get("export") or {}
    return {
        "width": int(export.get("width") or 720),
        "height": int(export.get("height") or 1280),
        "fps": int(export.get("fps") or 24),
        "target_duration_sec": float(script.get("target_duration_sec") or export.get("duration") or 90),
    }


def load_or_parse_shots(
    pack: TemplatePack,
    *,
    storyboard: str | Path | None = None,
    dialogue: str | Path | None = None,
) -> list[dict[str, Any]]:
    shots = list(pack.shots or [])
    if storyboard:
        parsed = parse_storyboard(storyboard)
        # storyboard wins if pack shots are empty or the user passed an override
        if parsed:
            shots = parsed
    if not shots and (pack.root / "storyboard.txt").is_file():
        shots = parse_storyboard(pack.root / "storyboard.txt")
    lines: list[dict[str, Any]] = []
    if dialogue:
        lines = parse_dialogue_table(dialogue)
    elif (pack.root / "dialogue.txt").is_file():
        lines = parse_dialogue_table(pack.root / "dialogue.txt")
    narration = (pack.script or {}).get("narration_segments") or ["F1"]
    if lines:
        shots = attach_dialogue(shots, lines, narration_segments=narration)
    return apply_pack_to_shots(shots, pack.to_dict())


def material_errors(pack: TemplatePack, shots: list[dict[str, Any]]) -> list[str]:
    issues = validate_pack({**pack.to_dict(), "shots": shots}, pack_root=pack.root)
    missing: list[str] = []
    for iss in issues:
        if iss.is_error:
            missing.extend(iss.missing_files or [])
            if iss.code != "missing_refs" and iss.code != "missing_ref_files":
                # keep the message for one-action / fight
                missing.append(iss.message) if not iss.missing_files else None
    # unique file names first
    files = []
    others = []
    seen = set()
    for m in missing:
        if not m:
            continue
        if m not in seen:
            seen.add(m)
            if m.endswith((".png", ".jpg", ".jpeg", ".webp")) or "缺少" not in str(m):
                # format_missing later; keep file-like tokens separate
                if any(str(m).endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    files.append(str(m))
                else:
                    others.append(str(m))
            else:
                others.append(str(m))
    return files, others, issues


def run_script_episode(
    *,
    workdir: str | Path,
    pack_root: str | Path,
    stages: list[str],
    storyboard: str | Path | None = None,
    dialogue: str | Path | None = None,
    wav_table: str | Path | None = None,
    store: Store | None = None,
    task_id: str | None = None,
    drama_id: str | None = None,
    comfyui_url: str | None = None,
    burn_subs: bool = False,
    max_cps: float = 4.0,
    allow_tts: bool = True,
    rerun_shot: str | None = None,
    progress: ProgressFn | None = None,
    output: str | Path | None = None,
    settings: Any | None = None,
    crf: int = 18,
    use_nvenc: bool | None = None,
) -> dict[str, Any]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    wanted = list(stages)

    def mark(stage: str, message: str, frac: float) -> None:
        log.info("[%s] %s", stage, message)
        if progress:
            progress(stage, message, max(0.0, min(1.0, frac)))

    # Resolve the pinned pack if this task already snapshotted one.
    pin_dir = workdir / "templates_used"
    if pin_dir.is_dir() and (pin_dir / "pack.json").is_file():
        pack = load_pack(pin_dir)
    else:
        pack = load_pack(pack_root)

    specs = _export_specs(pack)
    summary: dict[str, Any] = {
        "entry": "script",
        "pack": pack.name,
        "template_version": pack.version,
        "workdir": str(workdir),
        "wording": "按剧本生成：先静音画面，再贴对白。默认离线回退是「分镜预览成片」，不是完整图生视频。",
    }

    # ----- template -----
    if "template" in wanted or "scan" in wanted or "extract" in wanted or "generate" in wanted:
        with timed_run(store, task_id=task_id, drama_id=drama_id, step="模板套用", model=f"pack@{pack.version}"):
            mark("template", f"套用模板「{pack.name}」版本 {pack.version}", 0.2)
            if not pin_dir.is_dir():
                pin_pack_for_task(pack, workdir)
                pack = load_pack(pin_dir)
            shots = load_or_parse_shots(pack, storyboard=storyboard, dialogue=dialogue)
            if rerun_shot:
                # keep all shots; generate will only rebuild the named one
                pass
            files, others, issues = material_errors(pack, shots)
            errors = [i for i in issues if i.is_error]
            if files:
                msg = format_missing(files)
                raise PermissionError(msg)
            hard = [i for i in errors if i.code in {"one_action", "fight_two_person"}]
            if hard:
                raise ValueError("\n".join(i.message for i in hard))
            if store and task_id:
                store.replace_shots(task_id, shots)
                store.update(
                    "tasks",
                    task_id,
                    template_pack_id=str(pack.root),
                    template_version=pack.version,
                    pack_snapshot=str(pin_dir),
                    entry_type="script",
                )
            (workdir / "shots_plan.json").write_text(
                json.dumps(shots, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            summary["shots"] = len(shots)
            mark("template", f"已套用 {len(shots)} 个分镜，版本已钉住为 {pack.version}", 1.0)
    else:
        shots = json.loads((workdir / "shots_plan.json").read_text(encoding="utf-8")) if (workdir / "shots_plan.json").is_file() else load_or_parse_shots(pack, storyboard=storyboard, dialogue=dialogue)

    # ----- generate silent -----
    silent = workdir / "silent.mp4"
    output_root = getattr(settings, "output_root", None)
    if "generate" in wanted:
        check_cancel()
        from recast.generate.comfy import comfy_available, submit_workflow
        from recast.generate.animatic import ANIMATIC_LABEL, ANIMATIC_NOTE, build_animatic, build_shot_clip

        use_comfy = bool(comfyui_url) and comfy_available(comfyui_url)
        model_name = "comfyui" if use_comfy else "animatic"
        with timed_run(store, task_id=task_id, drama_id=drama_id, step="按剧本生成（先静音）", model=model_name) as rec:
            if use_comfy and pack.workflows:
                mark("generate", "已检测到 ComfyUI，正在提交工作流…", 0.2)
                wf = pack.workflows.get("txt2img") or pack.workflows.get("i2v") or next(iter(pack.workflows.values()))
                # Even with Comfy, we still write clips shot-by-shot; if a shot fails, fall back.
                clips_dir = workdir / "shots_silent"
                clips_dir.mkdir(exist_ok=True)
                for i, sh in enumerate(shots):
                    if rerun_shot and sh.get("shot_code") != rerun_shot:
                        continue
                    check_cancel()
                    code = sh.get("shot_code") or f"S{i+1:02d}"
                    try:
                        files = submit_workflow(
                            comfyui_url,
                            wf,
                            variables={
                                "prompt": sh.get("filled_prompt") or sh.get("prompt") or "",
                                "shot_code": code,
                                "width": specs["width"],
                                "height": specs["height"],
                            },
                            dest_dir=clips_dir / code,
                        )
                        video = next((f for f in files if f.suffix.lower() in {".mp4", ".webm", ".mov"}), None)
                        if video:
                            dest = clips_dir / f"{code}.mp4"
                            if video.resolve() != dest.resolve():
                                dest.write_bytes(video.read_bytes())
                            sh["output_path"] = str(dest)
                            sh["animatic"] = False
                            continue
                    except Exception as exc:
                        log.warning("ComfyUI 镜头 %s 失败，改用分镜预览成片: %s", code, exc)
                    refs = pack.resolve_shot_refs(sh, output_root=output_root)
                    build_shot_clip(
                        sh,
                        clips_dir / f"{code}.mp4",
                        refs=refs,
                        width=specs["width"],
                        height=specs["height"],
                        fps=specs["fps"],
                        crf=crf,
                        use_nvenc=use_nvenc,
                    )
                    sh["animatic"] = True
                from recast.generate.animatic import _concat

                _concat(
                    [Path(s["output_path"]) for s in shots if s.get("output_path")],
                    silent,
                    crf=crf,
                    use_nvenc=use_nvenc,
                )
                rec["message"] = "ComfyUI 工作流（个别失败镜头已回退为分镜预览成片）"
                rec["model"] = "comfyui"
            else:
                mark("generate", "未连接 ComfyUI，正在生成「分镜预览成片」（参考静帧推拉，不是完整图生视频）", 0.3)
                if rerun_shot:
                    # rebuild one clip then restitch
                    target = next((s for s in shots if s.get("shot_code") == rerun_shot), None)
                    if target is None:
                        raise ValueError(f"找不到要重跑的镜头 {rerun_shot}")
                    refs = pack.resolve_shot_refs(target, output_root=output_root)
                    dest = workdir / "shots_silent" / f"{rerun_shot}.mp4"
                    build_shot_clip(
                        target,
                        dest,
                        refs=refs,
                        width=specs["width"],
                        height=specs["height"],
                        fps=specs["fps"],
                        crf=crf,
                        use_nvenc=use_nvenc,
                    )
                    from recast.generate.animatic import _concat

                    _concat(
                        [Path(s.get("output_path") or workdir / "shots_silent" / f"{s.get('shot_code')}.mp4") for s in shots],
                        silent,
                        crf=crf,
                        use_nvenc=use_nvenc,
                    )
                else:
                    result = build_animatic(
                        shots,
                        workdir,
                        pack=pack,
                        width=specs["width"],
                        height=specs["height"],
                        fps=specs["fps"],
                        crf=crf,
                        use_nvenc=use_nvenc,
                        output_root=output_root,
                    )
                    silent = Path(result["silent"]) if result.get("silent") else silent
                rec["message"] = ANIMATIC_NOTE
                rec["model"] = "animatic"
            if store and task_id:
                store.replace_shots(task_id, shots)
            (workdir / "shots_plan.json").write_text(
                json.dumps(shots, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            summary["silent"] = str(silent) if silent.is_file() else None
            summary["picture_kind"] = "comfyui" if use_comfy else ANIMATIC_LABEL
            mark("generate", f"静音画面已写出（{summary['picture_kind']}）", 1.0)

    # ----- dub (never first; never auto-speed) -----
    if "dub" in wanted or "compose" in wanted:
        check_cancel()
        overflows = check_shots(shots, max_cps=max_cps)
        if overflows and "dub" in wanted:
            msg = "\n".join(str(o) for o in overflows)
            if store and task_id:
                for o in overflows:
                    store.insert(
                        "quality_marks",
                        task_id=task_id,
                        shot_code=o.shot_code,
                        reason="对白超出时长",
                        detail=o.message,
                        status=ST_REVIEW,
                    )
            raise ValueError(msg)
        with timed_run(store, task_id=task_id, drama_id=drama_id, step="贴轨与字幕", model="tts/ffmpeg") as rec:
            mark("dub", "贴对白 / 旁白并导出 SRT（不加速音频）", 0.3)
            from recast.core.character_assets import resolve_voices
            from recast.generate.tts import load_wav_table, synthesize_table
            from recast.generate.stitch import stitch_and_dub

            from recast.generate.tts import voices_by_speaker

            speakers = sorted(
                {
                    str(sh.get("speaker") or sh.get("character_id") or "")
                    for sh in shots
                }
                - {""}
            )
            voices, timbres = resolve_voices(
                output_root,
                speakers,
                store=store,
                fallback=voices_by_speaker(pack.characters),
            )
            table = load_wav_table(wav_table) if wav_table else None
            tts = synthesize_table(
                shots,
                workdir / "dub",
                wav_table=table,
                voices=voices,
                timbres=timbres,
                allow_engines=allow_tts,
            )
            if tts.get("skipped"):
                log.info("有 %s 条对白跳过了配音，仍会写出 SRT。", tts["skipped"])
                rec["message"] = f"跳过 {tts['skipped']} 条配音，已导出 SRT"
            out = Path(output) if output else workdir / "output.mp4"
            result = stitch_and_dub(
                shots,
                workdir,
                silent=silent if silent.is_file() else workdir / "silent.mp4",
                output=out,
                burn_subs=burn_subs,
                precheck_dialogue=False,  # already checked
                max_cps=max_cps,
                crf=crf,
                use_nvenc=use_nvenc,
            )
            rec["model"] = "ffmpeg+tts"
            summary.update(result)
            for note in result.get("confirmations") or []:
                log.warning("%s", note.get("message"))
                if store and task_id:
                    store.insert(
                        "quality_marks",
                        task_id=task_id,
                        shot_code=note.get("shot_code"),
                        reason="配音超出镜头",
                        detail=note.get("message"),
                        status=ST_REVIEW,
                    )
            if store and task_id:
                store.replace_shots(task_id, shots)
                if result.get("output"):
                    store.update("tasks", task_id, output_file=result["output"])
            mark("dub", f"成片已写出: {result.get('output')}", 1.0)

    return summary
