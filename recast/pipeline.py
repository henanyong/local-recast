"""End-to-end pipeline: ingest -> subtitles -> track -> identity -> pose -> replace -> compose.

Stages can be run all at once or one workbench group at a time, reusing artifacts
in the same workdir. Cooperative cancel is honoured between stages and inside
the long frame loops.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from recast.config import PROJECT_ROOT, RecastConfig, resolve_config_path
from recast.core.constants import STAGE_GROUP_IDS, STAGE_GROUPS, STAGE_LABELS
from recast.device import apply_thread_caps, requested_device, vram_report
from recast.ffmpeg import VideoInfo
from recast.namesutil import parse_names
from recast.runtime import CancelledError, check_cancel

log = logging.getLogger("recast")

# Re-exported for the web workbench and tests.
_GROUP_LABEL = dict(STAGE_LABELS)

def _stage_imports():
    from recast.media import alias_input
    from recast.stages import compose, identity, ingest, pose, replace, subtitles, track
    return alias_input, compose, identity, ingest, pose, replace, subtitles, track

ProgressFn = Callable[[str, str, float], None]


def setup_logging(workdir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("recast")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    logger.propagate = False
    if workdir is not None:
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(workdir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _progress(cb: ProgressFn | None, stage: str, message: str, fraction: float) -> None:
    log.info("[%s] %s", stage, message)
    if cb is not None:
        cb(stage, message, max(0.0, min(1.0, fraction)))


def _make_workdir(input_video: Path, out: Path | None, workdir: Path | None) -> Path:
    from recast.media import ascii_stem, is_ascii_path, write_source_record

    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        write_source_record(workdir, input_video)
        return workdir
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{ascii_stem(input_video)}"
    if out is not None and is_ascii_path(out.parent):
        base = out.parent / "recast-runs" / name
    else:
        if out is not None and not is_ascii_path(out.parent):
            log.info("成片目录含非 ASCII 字符，工作目录改放到项目 outputs 下以避免中文路径问题。")
        base = PROJECT_ROOT / "outputs" / name
    base.mkdir(parents=True, exist_ok=True)
    write_source_record(base, input_video)
    return base


def find_input_alias(workdir: Path, src: Path) -> Path | None:
    from recast.media import find_input_alias as _find

    return _find(workdir, src)


def load_video_info(workdir: Path, fallback: Path) -> VideoInfo | None:
    meta = workdir / "meta.json"
    if not meta.is_file():
        return None
    data = json.loads(meta.read_text(encoding="utf-8"))
    return VideoInfo(
        path=str(data.get("path") or fallback),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        fps=float(data.get("fps") or 25.0),
        duration=float(data.get("duration") or 0.0),
        frame_count=int(data.get("frame_count") or 0),
        has_audio=bool(data.get("has_audio")),
        video_codec=str(data.get("video_codec") or ""),
        audio_codec=str(data.get("audio_codec") or ""),
    )


def load_track_payload(workdir: Path) -> dict[str, Any] | None:
    preferred = workdir / "tracks.json"
    raw = workdir / "tracks_raw.json"
    path = preferred if preferred.is_file() else raw if raw.is_file() else None
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    ident_path = workdir / "identity.json"
    if ident_path.is_file():
        ident = json.loads(ident_path.read_text(encoding="utf-8"))
        names = ident.get("track_names") or {}
        for item in payload.get("tracks") or []:
            tid = str(item.get("track_id"))
            if tid in names and names[tid]:
                item["name"] = names[tid]
        payload["_identity"] = ident
    return payload


def artifact_status(workdir: Path) -> dict[str, bool]:
    return {
        "scan": (workdir / "input.mp4").is_file() or any(workdir.glob("input.*")),
        "extract": (workdir / "meta.json").is_file() and (workdir / "subtitles.srt").is_file(),
        "library": (workdir / "identity.json").is_file(),
        "replace": (workdir / "replaced.mp4").is_file(),
        "compose": (workdir / "output.mp4").is_file(),
        "preprocess": (workdir / "meta.json").is_file(),
        "subtitles": (workdir / "subtitles.json").is_file(),
        "tracks": (workdir / "tracks_raw.json").is_file() and (workdir / "identity.json").is_file(),
        "pose": (workdir / "tracks.json").is_file(),
        "replaced": (workdir / "replaced.mp4").is_file(),
    }


def normalize_stages(stages: Sequence[str] | None) -> list[str]:
    from recast.core.runner import normalize_stages as _norm
    return _norm(stages)


def normalize_stages_legacy(stages: Sequence[str] | None) -> list[str]:
    if not stages:
        return list(STAGE_GROUP_IDS)
    aliases = {
        "01": "scan",
        "ingest": "extract",
        "1": "scan",
        "02": "extract",
        "whisper": "extract",
        "subtitles": "extract",
        "2": "extract",
        "03": "library",
        "track": "library",
        "identity": "library",
        "tracks": "library",
        "3": "library",
        "04": "replace",
        "keypoints": "pose",
        "4": "replace",
        "05": "compose",
        "replace": "replace",
        "5": "compose",
        "all": "*",
        "preprocess": "extract",
    }
    out: list[str] = []
    for raw in stages:
        key = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
        if key in {"*", "all"}:
            return list(STAGE_GROUP_IDS)
        if key not in STAGE_GROUP_IDS:
            raise ValueError(f"未知阶段: {raw}。可选: {', '.join(STAGE_GROUP_IDS)}")
        if key not in out:
            out.append(key)
    return [s for s in STAGE_GROUP_IDS if s in out]


# Stages whose artifacts we can reliably detect and therefore skip when present.
_SKIPPABLE_STAGES = {
    "scan",
    "extract",
    "library",
    "pose",
    # replace stays in the plan when replaced.mp4 already exists so the
    # checkpoint fingerprint (every reference face, identity, tracks) can
    # decide. maybe_skip still avoids a second encode when those inputs match.
    "compose",
    "preprocess",
    "subtitles",
    "tracks",
    "replaced",
}

_PREREQS = {
    "scan": [],
    "extract": [],
    "library": ["extract"],
    "pose": ["extract", "library"],
    "replace": ["extract", "library"],
    "compose": ["extract"],
    "preprocess": [],
    "subtitles": ["preprocess"],
    "tracks": ["preprocess"],
}


def _plan_order() -> list[str]:
    order = list(STAGE_GROUP_IDS)
    if "pose" not in order:
        if "library" in order:
            order.insert(order.index("library") + 1, "pose")
        else:
            order.append("pose")
    if "upload" not in order:
        order.append("upload")
    return order


def resolve_plan(wanted: Sequence[str], workdir: Path, *, force: bool = False) -> list[str]:
    """Insert missing prerequisites; skip stages whose artifacts already exist unless *force*."""
    wanted_set = set(wanted)
    have = artifact_status(workdir)
    needed: set[str] = set()
    for stage in wanted:
        for pre in _PREREQS.get(stage, []):
            if not have.get(pre):
                needed.add(pre)
        if force or stage not in _SKIPPABLE_STAGES or not have.get(stage):
            needed.add(stage)
    plan: list[str] = []
    for stage in _plan_order():
        if stage in needed and stage not in plan:
            # Prerequisites that already have artifacts can stay skipped.
            if stage not in wanted_set and have.get(stage) and not force:
                continue
            plan.append(stage)
    return plan


def _load_segments(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("segments") or [])
    except Exception:
        return []


def process(
    input_video: str | Path,
    *,
    face: str | Path | None = None,
    out: str | Path | None = None,
    config: str | Path | RecastConfig | None = None,
    track_id: int | None = None,
    names: str | Path | dict[str, str] | None = None,
    skip_replace: bool = False,
    lang: str | None = "auto",
    burn_subs: bool | None = None,
    progress: ProgressFn | None = None,
    stages: Sequence[str] | None = None,
    workdir: str | Path | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """CLI-compatible entry. V0.1 path goes through the checkpointed episode runner."""
    from recast.core.runner import run_episode

    src = Path(input_video).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"找不到输入视频: {src}")
    output_path = Path(out).expanduser().resolve() if out else None
    run_dir = _make_workdir(src, output_path, Path(workdir).expanduser().resolve() if workdir else None)
    if output_path is None:
        output_path = run_dir / "output.mp4"
    setup_logging(run_dir)
    wanted = list(stages) if stages else None
    if skip_replace and wanted:
        wanted = [s for s in wanted if s not in {"replace"}]
    return run_episode(
        src,
        workdir=run_dir,
        config=config,
        stages=wanted,
        face=face,
        track_id=track_id,
        names=names,
        lang=lang,
        burn_subs=burn_subs,
        skip_replace=skip_replace,
        require_auth=False,
        progress=progress,
        output=output_path,
        force_rerun=force_rerun,
    )



def _existing_summary(
    src: Path,
    run_dir: Path,
    output_path: Path,
    cfg: RecastConfig,
    device: str,
) -> dict[str, Any]:
    existing = run_dir / "summary.json"
    if existing.is_file():
        try:
            data = json.loads(existing.read_text(encoding="utf-8"))
            data["artifacts"] = artifact_status(run_dir)
            return data
        except Exception:
            pass
    out = output_path if output_path.is_file() else (run_dir / "output.mp4")
    return {
        "input": str(src),
        "workdir": str(run_dir),
        "output": str(out) if out.is_file() else None,
        "config": str(cfg.path) if cfg.path else None,
        "device": device,
        "vram_gb": cfg.vram_gb,
        "stages": [],
        "artifacts": artifact_status(run_dir),
        "reused": True,
    }


def group_labels() -> list[dict[str, str]]:
    return [{"id": gid, "label": label} for gid, label, _ in STAGE_GROUPS]
