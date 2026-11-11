"""V0.1 episode runner: extract → library → authorized 人物替换 → compose, with checkpoints."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from recast.config import RecastConfig, resolve_config_path
from recast.core.checkpoint import describe_skip, should_skip, write_checkpoint
from recast.core.constants import (
    STAGE_GROUP_IDS,
    STAGE_LABELS,
    ST_NEED_ASSETS,
)
from recast.core.roster import parse_roster
from recast.core.state_machine import classify_error
from recast.device import apply_thread_caps, release_cuda, requested_device, vram_report
from recast.runtime import CancelledError, check_cancel
from recast.store.db import Store

log = logging.getLogger("recast")

ProgressFn = Callable[[str, str, float], None]

# Stages that run_episode itself implements.
RUNNER_STAGES = ("scan", "extract", "library", "pose", "replace", "compose")


def select_compose_picture(
    video_alias: Path,
    replaced_path: Path | None,
    *,
    skip_replace: bool,
    has_face: bool,
) -> Path:
    """When replace is skipped or no reference face is set, never reuse a leftover replaced.mp4."""
    if skip_replace or not has_face:
        return Path(video_alias)
    if replaced_path is not None and Path(replaced_path).is_file():
        return Path(replaced_path)
    return Path(video_alias)



# CLI / legacy aliases kept so `python -m recast process --only ...` still works.
_ALIASES = {
    "01": "scan",
    "1": "scan",
    "check": "scan",
    "preprocess": "extract",
    "ingest": "extract",
    "02": "extract",
    "2": "extract",
    "subtitles": "extract",
    "whisper": "extract",
    "scenes": "extract",
    "ocr": "extract",
    "asr": "extract",
    "03": "library",
    "3": "library",
    "tracks": "library",
    "track": "library",
    "identity": "library",
    "faces": "library",
    "04": "replace",
    "4": "replace",
    "swap": "replace",
    "05": "compose",
    "5": "compose",
    "pose": "pose",
    "keypoints": "pose",
    "06": "template",
    "6": "template",
    "template": "template",
    "07": "generate",
    "7": "generate",
    "generate": "generate",
    "animatic": "generate",
    "08": "dub",
    "8": "dub",
    "dub": "dub",
    "09": "export",
    "9": "export",
    "export": "export",
    "cost": "export",
    "jianying": "export",
    "upload": "upload",
    "all": "*",
}


def normalize_stages(stages: Sequence[str] | None) -> list[str]:
    if not stages:
        return list(STAGE_GROUP_IDS)
    out: list[str] = []
    include_pose = False
    for raw in stages:
        key = _ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
        if key in {"*", "all"}:
            return list(STAGE_GROUP_IDS)
        if key == "pose":
            include_pose = True
            continue
        if key not in STAGE_GROUP_IDS and key not in {"pose", "upload"}:
            raise ValueError(f"未知阶段: {raw}。可选: {', '.join(STAGE_GROUP_IDS)}")
        if key not in out:
            out.append(key)
    plan = [s for s in STAGE_GROUP_IDS if s in out]
    if include_pose and "pose" not in plan:
        if "library" in plan:
            idx = plan.index("library") + 1
            plan.insert(idx, "pose")
        else:
            plan.append("pose")
    if "upload" in out and "upload" not in plan:
        plan.append("upload")
    return plan


def _progress(cb: ProgressFn | None, stage: str, message: str, fraction: float) -> None:
    label = STAGE_LABELS.get(stage, stage)
    log.info("[%s] %s", label, message)
    if cb is not None:
        cb(label, message, max(0.0, min(1.0, fraction)))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _reference_from_character(ch: dict[str, Any]) -> dict[str, Any] | None:
    for asset in ch.get("assets") or []:
        path = asset.get("path")
        if asset.get("kind") == "face" and path and Path(str(path)).is_file():
            return {
                "character_id": ch.get("id"),
                "name": ch.get("display_name") or ch.get("name") or Path(str(path)).stem,
                "face_path": str(path),
            }
    return None


def _same_face(rows: list[dict[str, Any]], face_path: str) -> bool:
    try:
        target = Path(face_path).resolve()
    except OSError:
        target = Path(face_path)
    for row in rows:
        try:
            if Path(str(row.get("face_path"))).resolve() == target:
                return True
        except OSError:
            if str(row.get("face_path")) == face_path:
                return True
    return False


def _character_reference(store: Store, character_id: str, *, require_auth: bool) -> dict[str, Any]:
    ch = store.get_character(character_id)
    if not ch:
        raise RuntimeError(f"找不到人物: {character_id}")
    authorized = bool(ch.get("authorized") and ch.get("auth_type"))
    if require_auth and not authorized:
        raise PermissionError(
            "人物替换仅能用于已勾选授权的角色（自有拍摄 / 已授权 / 自有生成角色）。"
        )
    row = _reference_from_character(ch)
    if row is None:
        label = ch.get("display_name") or ch.get("name") or character_id
        raise RuntimeError(
            ST_NEED_ASSETS
            + f"：人物「{label}」缺少参考脸，请在人物库中确认姓名、勾选授权并指定参考脸后再执行人物替换。"
        )
    return row


def _reference_from_face_file(store: Store | None, face: Path, *, require_auth: bool) -> dict[str, Any]:
    if require_auth and store is not None:
        matched: dict[str, Any] | None = None
        for ch in store.list_characters():
            for asset in ch.get("assets") or []:
                path = asset.get("path")
                if not (asset.get("kind") == "face" and path):
                    continue
                try:
                    same = Path(str(path)).resolve() == face.resolve()
                except OSError:
                    same = str(path) == str(face)
                if same:
                    matched = ch
                    break
            if matched is not None:
                break
        if matched is None:
            raise PermissionError(
                "请先在人物库中为该参考脸确认姓名并勾选授权，再执行人物替换。"
            )
        if not (matched.get("authorized") and matched.get("auth_type")):
            raise PermissionError(
                "人物替换仅能用于已勾选授权的角色（自有拍摄 / 已授权 / 自有生成角色）。"
            )
        row = _reference_from_character(matched)
        if row is not None:
            return row
    return {
        "character_id": None,
        "name": face.stem,
        "face_path": str(face),
    }


def resolve_swap_references(
    *,
    store: Store | None,
    character_id: str | None = None,
    character_ids: list[str] | None = None,
    face: str | Path | None = None,
    require_auth: bool = False,
) -> list[dict[str, Any]]:
    """Authorized reference faces for one replace pass.

    An explicit character id is checked and included. When none is given and a
    store is available, every authorized character that has a reference face is
    included. A raw face file is only added when authorization allows it.
    """
    explicit: list[str] = []
    seen: set[str] = set()
    for cid in list(character_ids or []):
        if cid and str(cid) not in seen:
            explicit.append(str(cid))
            seen.add(str(cid))
    if character_id and str(character_id) not in seen:
        explicit.append(str(character_id))
        seen.add(str(character_id))

    rows: list[dict[str, Any]] = []
    if store is not None and explicit:
        for cid in explicit:
            try:
                rows.append(_character_reference(store, cid, require_auth=require_auth))
            except RuntimeError:
                # A single requested id with no file can still fall through to --face.
                if not (len(explicit) == 1 and face and Path(face).is_file()):
                    raise
    elif store is not None and not face:
        for ch in store.list_characters():
            if not (ch.get("authorized") and ch.get("auth_type")):
                continue
            row = _reference_from_character(ch)
            if row is not None:
                rows.append(row)

    if face and Path(face).is_file():
        raw = _reference_from_face_file(store, Path(face), require_auth=require_auth)
        if not _same_face(rows, raw["face_path"]):
            rows.append(raw)
    return rows


def authorized_faces(store: Store | None, character_id: str | None = None) -> list[dict[str, Any]]:
    if store is None:
        return []
    rows = []
    chars = [store.get_character(character_id)] if character_id else store.list_characters()
    for ch in chars:
        if not ch:
            continue
        if not (ch.get("authorized") and ch.get("auth_type")):
            continue
        faces = [a for a in (ch.get("assets") or []) if a.get("kind") == "face" and a.get("path") and Path(str(a["path"])).is_file()]
        if faces:
            rows.append({**ch, "face_path": faces[0]["path"]})
    return rows


def resolve_swap_face(
    *,
    store: Store | None,
    character_id: str | None,
    face: str | Path | None,
    require_auth: bool,
) -> Path | None:
    rows = resolve_swap_references(
        store=store,
        character_id=character_id,
        face=face,
        require_auth=require_auth,
    )
    if not rows:
        return None
    return Path(rows[0]["face_path"])


def run_episode(
    video: str | Path,
    *,
    workdir: str | Path,
    config: str | Path | RecastConfig | None = None,
    stages: Sequence[str] | None = None,
    face: str | Path | None = None,
    character_id: str | None = None,
    character_ids: list[str] | None = None,
    track_id: int | None = None,
    names: str | Path | dict[str, str] | None = None,
    lang: str | None = "auto",
    burn_subs: bool | None = None,
    skip_replace: bool = False,
    require_auth: bool = False,
    with_pose: bool = False,
    store: Store | None = None,
    task_id: str | None = None,
    drama_key: str | None = None,
    progress: ProgressFn | None = None,
    output: str | Path | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run one episode with per-step checkpoints and CUDA cache clears."""
    src = Path(video).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"找不到输入视频: {src}")
    run_dir = Path(workdir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(config, RecastConfig):
        cfg = config
    else:
        cfg = RecastConfig.from_yaml(resolve_config_path(config))

    apply_thread_caps(cfg.cpu_threads, cfg.opencv_threads)
    device = requested_device(cfg.device)
    roster_map: dict[str, str] = {}
    if names:
        roster_map = parse_roster(names).name_map() if not isinstance(names, dict) else {str(k): str(v) for k, v in names.items()}

    wanted = normalize_stages(list(stages or []) + (["pose"] if with_pose else []))
    wanted = [s for s in wanted if s in RUNNER_STAGES]
    from recast.pipeline import resolve_plan

    plan = resolve_plan(wanted, run_dir, force=force_rerun)
    if not plan:
        from recast.pipeline import _existing_summary

        log.info("所选阶段的产物已存在，无需重跑。")
        _progress(progress, wanted[-1] if wanted else "scan", "所选阶段的产物已存在，无需重跑", 1.0)
        return _existing_summary(src, run_dir, Path(output).expanduser().resolve() if output else run_dir / "output.mp4", cfg, device)

    wanted = plan
    output_path = Path(output).expanduser().resolve() if output else run_dir / "output.mp4"

    summary: dict[str, Any] = {
        "input": str(src),
        "workdir": str(run_dir),
        "output": str(output_path) if output_path.is_file() else None,
        "config": str(cfg.path) if cfg.path else None,
        "device": device,
        "vram_gb": cfg.vram_gb,
        "stages": wanted,
        "wording": "本流程的替换是 L1 人物替换（本地换脸），不是提取了3D模型。",
    }

    from recast.media import alias_input
    from recast.pipeline import find_input_alias, load_track_payload, load_video_info

    check_cancel()
    video_alias = find_input_alias(run_dir, src) or alias_input(src, run_dir)
    info = load_video_info(run_dir, video_alias)
    wav = run_dir / "audio.wav"
    wav_path = wav if wav.is_file() else None

    if wanted:
        n = len(wanted)
        spans = {stage: (i / n, (i + 1) / n) for i, stage in enumerate(wanted)}
    else:
        spans = {"scan": (0.0, 1.0)}

    def mark(stage: str, message: str, frac: float) -> None:
        lo, hi = spans.get(stage, (0.0, 1.0))
        _progress(progress, stage, message, lo + (hi - lo) * max(0.0, min(1.0, frac)))

    def maybe_skip(step: str, inputs: list[Path], outputs: list[Path]) -> bool:
        if should_skip(store, task_id, step, inputs, outputs, force=force_rerun):
            log.info(describe_skip(step))
            return True
        return False

    def remember(step: str, inputs: list[Path], outputs: list[Path]) -> None:
        if store is not None and task_id:
            write_checkpoint(store, task_id, step, inputs, outputs)

    # ----- 01 scan / material note is handled by the service; runner just records the video fingerprint -----
    if "scan" in wanted:
        mark("scan", "记录源视频指纹并确认文件存在", 0.2)
        if not maybe_skip("scan", [src], [video_alias]):
            check_cancel()
            video_alias = find_input_alias(run_dir, src) or alias_input(src, run_dir)
            remember("scan", [src], [video_alias])
        mark("scan", "扫描检查完成", 1.0)

    # ----- 02 extract: ingest + scenes + ASR + OCR + merge -----
    if "extract" in wanted:
        mark("extract", "预处理：探测视频并提取音轨", 0.05)
        if not maybe_skip("ingest", [video_alias], [run_dir / "meta.json"]):
            check_cancel()
            from recast.stages import ingest

            info, wav_path = ingest.run(video_alias, run_dir)
            remember("ingest", [video_alias], [run_dir / "meta.json"])
        else:
            info = load_video_info(run_dir, video_alias) or info
            wav_path = wav if wav.is_file() else wav_path
        if info is None:
            raise RuntimeError("缺少预处理产物，请先运行「02 解析提取」。")
        summary["video"] = info.to_dict()

        mark("extract", "切镜检测", 0.25)
        shots_path = run_dir / "shots.json"
        if not maybe_skip("scenes", [video_alias], [shots_path]):
            check_cancel()
            from recast.extract.scenes import detect_scenes

            payload = detect_scenes(video_alias, run_dir, info=info)
            if store is not None and task_id:
                store.replace_shots(task_id, payload.get("shots") or [])
            remember("scenes", [video_alias], [shots_path])
        shots = (_load_json(shots_path) or {}).get("shots") or []
        summary["shots"] = len(shots)

        mark("extract", "语音转写（ASR）", 0.5)
        asr_json = run_dir / "subtitles.json"
        if not maybe_skip("asr", [p for p in (wav_path, video_alias) if p and Path(p).exists()], [asr_json]):
            check_cancel()
            from recast.extract import asr

            srt_path, sub_json, segments = asr.run(
                wav_path,
                run_dir,
                cfg,
                language=lang,
                names=list(roster_map.values()),
            )
            if store is not None and task_id:
                store.replace_subtitles(task_id, "asr", segments)
            remember("asr", [p for p in (wav_path, video_alias) if p and Path(p).exists()], [asr_json, run_dir / "subtitles.srt"])
        else:
            segments = ((_load_json(asr_json) or {}).get("segments")) or []
        summary["asr_segments"] = len(segments)

        mark("extract", "画面 OCR（烧录字幕）", 0.75)
        ocr_json = run_dir / "ocr.json"
        if not maybe_skip("ocr", [video_alias, shots_path] if shots_path.is_file() else [video_alias], [ocr_json]):
            check_cancel()
            from recast.extract import ocr

            _, ocr_segments = ocr.run(video_alias, run_dir, info, shots)
            if store is not None and task_id:
                store.replace_subtitles(task_id, "ocr", ocr_segments)
            remember("ocr", [video_alias], [ocr_json, run_dir / "ocr.srt"])
        else:
            ocr_segments = ((_load_json(ocr_json) or {}).get("segments")) or []

        mark("extract", "对齐 OCR 与 ASR 字幕", 0.9)
        merged_json = run_dir / "subtitles_merged.json"
        if not maybe_skip("merge", [asr_json, ocr_json], [merged_json, run_dir / "subtitles.srt"]):
            from recast.extract.merge_subs import merge_subtitles
            from recast.stages.subtitles import filter_no_speech, snap_segments_to_shots, write_srt

            asr_segs = ((_load_json(asr_json) or {}).get("segments")) or segments
            asr_segs = filter_no_speech(
                asr_segs,
                max_no_speech_prob=float(getattr(cfg.whisper, "no_speech_prob", 0.6) or 0.6),
            )
            asr_segs = snap_segments_to_shots(
                asr_segs,
                shots,
                max_shift=float(getattr(cfg.whisper, "edge_snap", 0.35) or 0.35),
            )
            merged = merge_subtitles(asr_segs, ocr_segments)
            merged_json.write_text(
                json.dumps({"source": "merged", "segments": merged}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_srt(merged, run_dir / "subtitles.srt")
            write_srt(merged, run_dir / "merged.srt")
            if store is not None and task_id:
                store.replace_subtitles(task_id, "merged", merged)
            remember("merge", [asr_json, ocr_json], [merged_json, run_dir / "subtitles.srt"])
            summary["subtitles"] = {"srt": str(run_dir / "subtitles.srt"), "segments": len(merged)}
        else:
            merged = ((_load_json(merged_json) or {}).get("segments")) or []
            summary["subtitles"] = {"srt": str(run_dir / "subtitles.srt"), "segments": len(merged)}
        release_cuda()
        vram_report("提取后")
        mark("extract", "解析提取完成", 1.0)

    # ----- 03 library -----
    if "library" in wanted:
        if info is None:
            info = load_video_info(run_dir, video_alias)
        if info is None:
            raise RuntimeError("人物提取需要先完成「02 解析提取」。")
        mark("library", "行人检测与跟踪", 0.15)
        tracks_raw = run_dir / "tracks_raw.json"
        if not maybe_skip("track", [video_alias], [tracks_raw]):
            check_cancel()
            from recast.stages import track

            track_payload = track.run(video_alias, info, run_dir, cfg)
            remember("track", [video_alias], [tracks_raw])
            release_cuda()
        else:
            track_payload = load_track_payload(run_dir) or {}
        mark("library", "人脸聚类并写入人物库", 0.6)
        ident_path = run_dir / "identity.json"
        if not maybe_skip("identity", [tracks_raw, video_alias], [ident_path]):
            check_cancel()
            from recast.extract.faces import run_library

            ident = run_library(
                video_alias,
                info,
                track_payload if track_payload else (load_track_payload(run_dir) or {}),
                run_dir,
                cfg,
                roster=roster_map,
                store=store,
                task_id=task_id,
                drama_key=drama_key,
            )
            remember("identity", [tracks_raw, video_alias], [ident_path])
            release_cuda()
        else:
            ident = _load_json(ident_path) or {}
            # still try to ingest existing samples into the library
            if store is not None:
                from recast.extract.faces import ingest_into_library

                index_path = run_dir / "face_samples" / "index.json"
                samples = (_load_json(index_path) or {}) if index_path.is_file() else {}
                ingest_into_library(store, ident, samples, task_id=task_id, drama_key=drama_key)
        summary["persons"] = (ident or {}).get("persons")
        summary["tracks"] = len((load_track_payload(run_dir) or {}).get("tracks") or [])
        vram_report("人物库后")
        mark("library", "人物提取与建库完成，请在人物库中确认姓名并勾选授权", 1.0)

    # ----- optional pose -----
    if "pose" in wanted:
        if info is None:
            info = load_video_info(run_dir, video_alias)
        track_payload = load_track_payload(run_dir)
        if info is None or track_payload is None:
            raise RuntimeError("动作关键点需要先完成人物跟踪。")
        mark("pose", "估计轨迹关键点（可选）", 0.2)
        tracks_json = run_dir / "tracks.json"
        if not maybe_skip("pose", [video_alias, run_dir / "tracks_raw.json"], [tracks_json]):
            check_cancel()
            from recast.stages import pose

            pose.run(video_alias, info, track_payload, run_dir, cfg)
            remember("pose", [video_alias, run_dir / "tracks_raw.json"], [tracks_json])
            release_cuda()
        mark("pose", "关键点完成", 1.0)

    # ----- 04 人物替换 (L1 local face swap, never described as extracting a 3D model) -----
    swap_refs: list[dict[str, Any]] = []
    replaced_path: Path | None = None
    if "replace" in wanted and not skip_replace:
        mark("replace", "准备人物替换（L1 本地换脸，不是提取了3D模型）", 0.05)
        swap_refs = resolve_swap_references(
            store=store,
            character_id=character_id,
            character_ids=character_ids,
            face=face,
            require_auth=require_auth,
        )
        if not swap_refs:
            raise RuntimeError(
                ST_NEED_ASSETS + "：缺少已授权角色的参考脸，请在人物库中确认姓名、勾选授权并指定参考脸后再执行人物替换。"
            )
        if info is None:
            info = load_video_info(run_dir, video_alias)
        track_payload = load_track_payload(run_dir)
        if info is None or track_payload is None:
            raise RuntimeError("人物替换需要先完成「03 人物提取与建库」。")
        from recast.stages.replace import replace_checkpoint_inputs

        face_paths = [Path(str(row["face_path"])) for row in swap_refs]
        replace_inputs = replace_checkpoint_inputs(video_alias, run_dir, face_paths)
        if (run_dir / "replaced.mp4").is_file():
            replaced_path = run_dir / "replaced.mp4"
        if not maybe_skip("replace", replace_inputs, [run_dir / "replaced.mp4"]):
            check_cancel()
            from recast.stages import replace

            names = "、".join(str(row.get("name") or Path(str(row["face_path"])).name) for row in swap_refs)
            log.info(
                "执行 L1 人物替换（本地换脸），参考脸 %s。这不是 3D 头像，也不是提取了3D模型。",
                names,
            )
            replaced_path = replace.run(
                video_alias,
                info,
                track_payload,
                run_dir,
                cfg,
                references=swap_refs,
                track_id=track_id,
            )
            remember("replace", replace_inputs, [Path(replaced_path)])
            release_cuda()
        plan_doc = _load_json(run_dir / "replace_plan.json") or {}
        summary["replace_plan"] = plan_doc
        summary["replaced"] = str(replaced_path or (run_dir / "replaced.mp4"))
        vram_report("人物替换后")
        mark("replace", "人物替换完成", 1.0)
    elif "replace" in wanted and skip_replace:
        log.info("已跳过人物替换。")
    do_replace = bool(swap_refs) and not skip_replace

    # ----- 05 compose -----
    if "compose" in wanted:
        if info is None:
            info = load_video_info(run_dir, video_alias)
        mark("compose", "混流音轨并写出成片", 0.3)
        srt = run_dir / "subtitles.srt"
        picture = select_compose_picture(
            video_alias,
            replaced_path,
            skip_replace=skip_replace or not do_replace,
            has_face=do_replace,
        )
        if not maybe_skip("compose", [Path(picture), src], [output_path]):
            check_cancel()
            from recast.stages import compose

            final = compose.run(
                picture,
                src if src.is_file() else video_alias,
                run_dir,
                cfg,
                output_path,
                subtitle=srt if srt.is_file() else None,
                burn_subs=burn_subs,
                has_audio=bool(info.has_audio) if info is not None else None,
            )
            archived = run_dir / "output.mp4"
            if Path(final).resolve() != archived.resolve():
                try:
                    if not archived.exists():
                        shutil.copy2(final, archived)
                except OSError:
                    pass
            remember("compose", [Path(picture), src], [Path(final)])
            summary["output"] = str(final)
        else:
            summary["output"] = str(output_path if output_path.is_file() else run_dir / "output.mp4")
        mark("compose", f"成片已写出: {summary.get('output')}", 1.0)

    summary["output"] = summary.get("output") or (str(output_path) if output_path.is_file() else None)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _progress(progress, "compose" if "compose" in wanted else (wanted[-1] if wanted else "scan"), "完成", 1.0)
    vram_report("结束")
    return summary


def classify_and_message(exc: BaseException) -> tuple[str, str]:
    kind = classify_error(exc)
    return kind, str(exc)



def split_resource_stages(stages: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a plan into CPU-preprocess, GPU, and CPU-post, preserving relative order."""
    from recast.core.constants import CPU_POST_STAGES, CPU_PRE_STAGES, GPU_STAGES

    pre, gpu, post = [], [], []
    for s in stages:
        if s in GPU_STAGES:
            gpu.append(s)
        elif s in CPU_POST_STAGES or s in {"compose", "dub", "export", "upload"}:
            post.append(s)
        else:
            pre.append(s)
    return pre, gpu, post


def _run_export_stages(
    *,
    workdir: Path,
    stages: list[str],
    store: Store | None,
    task_id: str | None,
    drama_id: str | None,
    shots: list[dict] | None,
    output: Path | None,
    silent: Path | None = None,
    audio: Path | None = None,
    srt: Path | None = None,
    title: str = "本地再铸导出",
    settings=None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    from recast.compose.cost import write_cost_xlsx
    from recast.compose.jianying import export_jianying_draft
    from recast.core.timing import timed_run

    result: dict[str, Any] = {}
    if "export" not in stages and "upload" not in stages:
        return result
    if "export" in stages:
        with timed_run(store, task_id=task_id, drama_id=drama_id, step="成本表 / 剪映工程导出", model="openpyxl") as rec:
            _progress(progress, "export", "写出成本表与剪映工程", 0.3)
            dest_root = workdir / "delivery"
            dest_root.mkdir(parents=True, exist_ok=True)
            runs = store.list_runs(task_id=task_id) if store and task_id else (store.list_runs(drama_id=drama_id) if store and drama_id else [])
            xlsx = dest_root / "cost.xlsx"
            try:
                write_cost_xlsx(runs, xlsx, title=f"{title} 成本表")
            except RuntimeError as exc:
                rec["message"] = str(exc)
                rec["status"] = "ok"
                xlsx = dest_root / "cost.csv"
            jy = export_jianying_draft(
                dest_root / "jianying_draft",
                shots=shots or [],
                output_video=output,
                silent_video=silent,
                audio=audio,
                srt=srt or (workdir / "subtitles.srt" if (workdir / "subtitles.srt").is_file() else None),
                title=title,
            )
            rec["message"] = f"成本表 {xlsx}；剪映工程 {jy}（未对具体剪映版本做打开验证）"
            result["cost_xlsx"] = str(xlsx)
            result["jianying"] = str(jy)
            _progress(progress, "export", rec["message"], 1.0)
    if "upload" in stages and settings is not None and getattr(settings, "auto_upload", False):
        from recast.upload import run_upload
        from recast.core.timing import timed_run as _tr

        src = output or workdir
        with _tr(store, task_id=task_id, drama_id=drama_id, step="上传", model=getattr(settings, "upload_mode", "folder")) as rec:
            up = run_upload(
                src,
                enabled=True,
                mode=getattr(settings, "upload_mode", "folder"),
                folder=getattr(settings, "upload_folder", ""),
                webhook=getattr(settings, "upload_webhook", ""),
                command=getattr(settings, "upload_command", ""),
            )
            rec["message"] = up.get("message") or up.get("dest") or up.get("error")
            rec["status"] = "ok" if up.get("ok") else "failed"
            result["upload"] = up
    elif "upload" in stages:
        log.info("自动上传关闭（默认关闭），已跳过。")
        result["upload"] = {"ok": True, "skipped": True}
    return result


def dispatch_job(
    *,
    video: str | Path | None,
    workdir: str | Path,
    config=None,
    stages: list[str] | None = None,
    entry_type: str = "import",
    pack_root: str | Path | None = None,
    storyboard: str | Path | None = None,
    dialogue: str | Path | None = None,
    wav_table: str | Path | None = None,
    face=None,
    character_id=None,
    character_ids: list[str] | None = None,
    track_id=None,
    names=None,
    lang: str | None = "auto",
    burn_subs: bool | None = None,
    skip_replace: bool = False,
    require_auth: bool = False,
    with_pose: bool = False,
    store: Store | None = None,
    task_id: str | None = None,
    drama_key: str | None = None,
    progress: ProgressFn | None = None,
    output: str | Path | None = None,
    comfyui_url: str | None = None,
    max_cps: float = 4.0,
    rerun_shot: str | None = None,
    settings=None,
) -> dict[str, Any]:
    """Run either the import-video path or the generate-from-script path."""
    wanted = normalize_stages(list(stages or []) + (["pose"] if with_pose else []))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if entry_type == "script" or (pack_root and (not video or not Path(video).is_file())):
        from recast.core.script_pipeline import run_script_episode

        script_stages = [s for s in wanted if s in {"scan", "extract", "template", "generate", "dub", "compose", "replace", "export", "upload"}]
        # template + generate + dub first; replace may run on the silent/output afterwards
        gen_stages = [s for s in script_stages if s != "replace" and s != "export" and s != "upload"]
        if not gen_stages and any(s in wanted for s in ("template", "generate", "dub", "compose", "scan", "extract")):
            gen_stages = [s for s in wanted if s not in {"replace", "export", "upload", "library", "pose"}]
        summary = run_script_episode(
            workdir=workdir,
            pack_root=pack_root,
            stages=gen_stages or ["template", "generate", "dub"],
            storyboard=storyboard,
            dialogue=dialogue,
            wav_table=wav_table,
            store=store,
            task_id=task_id,
            drama_id=drama_key,
            comfyui_url=comfyui_url,
            burn_subs=bool(burn_subs),
            max_cps=max_cps,
            rerun_shot=rerun_shot,
            progress=progress,
            output=output,
            settings=settings,
        )
        # optional L1 replace on the generated silent/output picture joins here
        if "replace" in wanted and not skip_replace and (workdir / "silent.mp4").is_file():
            try:
                summary_rep = run_episode(
                    workdir / "silent.mp4",
                    workdir=workdir / "replace_work",
                    config=config,
                    stages=["replace", "compose"],
                    face=face,
                    character_id=character_id,
                    character_ids=character_ids,
                    track_id=track_id,
                    names=names,
                    lang=lang,
                    burn_subs=burn_subs,
                    skip_replace=False,
                    require_auth=require_auth,
                    store=store,
                    task_id=task_id,
                    drama_key=drama_key,
                    progress=progress,
                    output=Path(output) if output else workdir / "output_replaced.mp4",
                )
                summary["replace"] = summary_rep
            except Exception as exc:
                log.warning("生成后再做人物替换未执行: %s", exc)
        shots = []
        plan = workdir / "shots_plan.json"
        if plan.is_file():
            shots = json.loads(plan.read_text(encoding="utf-8")).get("shots") if False else json.loads(plan.read_text(encoding="utf-8"))
            if isinstance(shots, dict):
                shots = shots.get("shots") or []
        summary.update(
            _run_export_stages(
                workdir=workdir,
                stages=wanted,
                store=store,
                task_id=task_id,
                drama_id=drama_key,
                shots=shots if isinstance(shots, list) else [],
                output=Path(summary.get("output") or output or workdir / "output.mp4"),
                silent=workdir / "silent.mp4" if (workdir / "silent.mp4").is_file() else None,
                audio=workdir / "dialogue_mix.wav" if (workdir / "dialogue_mix.wav").is_file() else None,
                srt=workdir / "subtitles.srt" if (workdir / "subtitles.srt").is_file() else None,
                title=Path(workdir).parent.name,
                settings=settings,
                progress=progress,
            )
        )
        return summary

    # import-video path: existing extract + replace, then optional template pin / export
    if not video:
        raise FileNotFoundError("导入现成视频的任务需要输入视频。")
    import_stages = [s for s in wanted if s in {"scan", "extract", "library", "pose", "replace", "compose"}]
    if import_stages:
        from recast.core.constants import STAGE_LABELS
        from recast.core.timing import timed_run

        label = " / ".join(STAGE_LABELS.get(s, s) for s in import_stages)
        with timed_run(store, task_id=task_id, drama_id=drama_key, step=label):
            summary = run_episode(
            video,
            workdir=workdir,
            config=config,
            stages=import_stages,
            face=face,
            character_id=character_id,
            character_ids=character_ids,
            track_id=track_id,
            names=names,
            lang=lang,
            burn_subs=burn_subs,
            skip_replace=skip_replace or "replace" not in import_stages,
            require_auth=require_auth,
            with_pose=with_pose,
            store=store,
            task_id=task_id,
            drama_key=drama_key,
            progress=progress,
            output=output,
        )
    if "template" in wanted and pack_root:
        from recast.templates.pack import load_pack, pin_pack_for_task
        from recast.core.timing import timed_run

        with timed_run(store, task_id=task_id, drama_id=drama_key, step="模板套用"):
            pack = load_pack(pack_root)
            pin_pack_for_task(pack, workdir)
            if store and task_id:
                store.update("tasks", task_id, template_version=pack.version, template_pack_id=str(pack.root), pack_snapshot=str(Path(workdir) / "templates_used"))
            log.info("已为导入任务钉住模板版本 %s", pack.version)
    if "generate" in wanted and entry_type != "script":
        log.info("当前任务是导入现成视频，已跳过「按剧本生成」。")
    if "dub" in wanted and entry_type != "script":
        log.info("导入视频的音轨已在合成导出阶段处理，已跳过单独贴轨。")

    # quality marks after replace, including swapped-face vs reference similarity
    if store and task_id and "replace" in wanted:
        from recast.pipeline import load_track_payload
        from recast.quality.face_review import post_replace_review_items, try_embed_sample

        shots = store.shots_for_task(task_id)
        work = Path(workdir)
        plan_path = work / "replace_plan.json"
        plan = _load_json(plan_path) if plan_path.is_file() else {}
        track_payload = load_track_payload(work) or {}
        replaced = work / "replaced.mp4"
        thumbs = work / "review_thumbs"

        def _embed(sample: dict) -> tuple[list[float], list[float]] | None:
            return try_embed_sample(sample, replaced if replaced.is_file() else None)

        items = post_replace_review_items(
            shots,
            track_payload,
            plan if isinstance(plan, dict) else {},
            _embed,
            thumbnail_dir=thumbs,
        )
        for it in items:
            store.insert(
                "quality_marks",
                task_id=task_id,
                shot_id=it.shot_id,
                shot_code=it.shot_code,
                reason=it.reason,
                detail=it.detail,
                status="待人工检查",
                score=it.score,
                thumbnail=it.thumbnail,
            )
        if items:
            log.warning("质量检查标记了 %s 个镜头，请在终检窗口查看。", len(items))

    summary = summary or {}
    out_path = Path(summary.get("output") or output or Path(workdir) / "output.mp4")
    shots = store.shots_for_task(task_id) if store and task_id else []
    summary.update(
        _run_export_stages(
            workdir=Path(workdir),
            stages=wanted,
            store=store,
            task_id=task_id,
            drama_id=drama_key,
            shots=shots,
            output=out_path if out_path.is_file() else None,
            srt=Path(workdir) / "subtitles.srt" if (Path(workdir) / "subtitles.srt").is_file() else None,
            title=Path(video).stem if video else "本地再铸导出",
            settings=settings,
            progress=progress,
        )
    )
    return summary
