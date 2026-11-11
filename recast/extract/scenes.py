"""Shot / scene detection: PySceneDetect first, FFmpeg scene filter as fallback."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from recast.ffmpeg import VideoInfo, probe_video, run_ffmpeg
from recast.runtime import check_cancel

log = logging.getLogger("recast")

# ContentDetector's historical default and this project's FFmpeg default describe
# the same sensitivity. A higher threshold means fewer cuts on both backends.
CONTENT_ANCHOR = 27.0
FFMPEG_SCORE_AT_CONTENT_27 = 0.30
DEFAULT_MIN_SHOT_SEC = 0.4

BACKEND_LABELS = {
    "pyscenedetect": "PySceneDetect",
    "ffmpeg": "FFmpeg",
    "ffmpeg+single": "FFmpeg（整段一个镜头）",
    "pyscenedetect+single": "PySceneDetect（整段一个镜头）",
}


def ffmpeg_scene_score(content_threshold: float) -> float:
    """Map a PySceneDetect content threshold onto FFmpeg's 0–1 scene score.

    27 maps to 0.30, which is the pair this project used as its two defaults.
    The scale is linear through that anchor so changing the one user threshold
    moves both detectors together.
    """
    score = float(content_threshold) * (FFMPEG_SCORE_AT_CONTENT_27 / CONTENT_ANCHOR)
    return min(0.99, max(0.01, score))


def backend_label(name: str | None) -> str:
    if not name:
        return ""
    return BACKEND_LABELS.get(name, name)


def _reindex(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, shot in enumerate(shots):
        row = dict(shot)
        row["index_no"] = i
        out.append(row)
    return out


def merge_short_shots(shots: list[dict[str, Any]], min_shot_sec: float) -> list[dict[str, Any]]:
    """Fold flashes and other sub-minimum shots into a neighbor.

    A short first shot extends the next one. Later short shots extend the
    previous one. The whole timeline stays covered.
    """
    rows = [dict(shot) for shot in shots]
    if min_shot_sec <= 0 or len(rows) < 2:
        return _reindex(rows)

    def _dur(shot: dict[str, Any]) -> float:
        return float(shot["end_time"]) - float(shot["start_time"])

    while len(rows) >= 2 and _dur(rows[0]) < min_shot_sec:
        rows[1]["start_time"] = rows[0]["start_time"]
        rows[1]["start_frame"] = rows[0]["start_frame"]
        rows.pop(0)
    merged: list[dict[str, Any]] = [rows[0]]
    for shot in rows[1:]:
        if _dur(shot) < min_shot_sec:
            merged[-1]["end_time"] = shot["end_time"]
            merged[-1]["end_frame"] = shot["end_frame"]
        else:
            merged.append(shot)
    return _reindex(merged)


def _shots_payload(
    video: Path,
    info: VideoInfo,
    shots: list[dict[str, Any]],
    backend: str,
    *,
    threshold: float,
    ffmpeg_score: float,
    min_shot_sec: float,
) -> dict[str, Any]:
    return {
        "video": str(video),
        "backend": backend,
        "backend_label": backend_label(backend),
        "threshold": float(threshold),
        "ffmpeg_score": float(ffmpeg_score),
        "min_shot_sec": float(min_shot_sec),
        "fps": info.fps,
        "duration": info.duration,
        "frame_count": info.frame_count,
        "shots": shots,
        "count": len(shots),
    }


def _frame_at(info: VideoInfo, seconds: float) -> int:
    return max(0, int(round(seconds * (info.fps or 25.0))))


def _shots_from_cuts(cut_times: list[float], info: VideoInfo) -> list[dict[str, Any]]:
    duration = float(info.duration or 0.0)
    bounds = [0.0, *sorted(t for t in cut_times if 0.0 < t < duration - 0.05), duration if duration > 0 else (cut_times[-1] if cut_times else 0.0)]
    # unique
    uniq: list[float] = []
    for t in bounds:
        if not uniq or abs(t - uniq[-1]) > 0.05:
            uniq.append(t)
    if len(uniq) < 2:
        end = duration if duration > 0 else 0.0
        return [{"index_no": 0, "start_time": 0.0, "end_time": end, "start_frame": 0, "end_frame": _frame_at(info, end)}]
    shots: list[dict[str, Any]] = []
    for i in range(len(uniq) - 1):
        start, end = uniq[i], uniq[i + 1]
        shots.append(
            {
                "index_no": i,
                "start_time": float(start),
                "end_time": float(end),
                "start_frame": _frame_at(info, start),
                "end_frame": _frame_at(info, end),
            }
        )
    return shots


def _detect_pyscenedetect(video: Path, info: VideoInfo, threshold: float) -> list[dict[str, Any]] | None:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except Exception as exc:
        log.info("未安装 PySceneDetect，将使用 FFmpeg 切镜回退（%s）", exc)
        return None
    check_cancel()
    try:
        video_s = open_video(str(video))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=float(threshold)))
        manager.detect_scenes(video_s, show_progress=False)
        scenes = manager.get_scene_list()
    except Exception as exc:
        log.warning("PySceneDetect 失败，回退 FFmpeg: %s", exc)
        return None
    if not scenes:
        return _shots_from_cuts([], info)
    shots: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(scenes):
        start_s = start.get_seconds()
        end_s = end.get_seconds()
        shots.append(
            {
                "index_no": i,
                "start_time": float(start_s),
                "end_time": float(end_s),
                "start_frame": int(start.get_frames()),
                "end_frame": int(end.get_frames()),
            }
        )
    return shots


_FFMPEG_PTS = re.compile(r"pts_time:([0-9.+-eE]+)")


def _detect_ffmpeg(video: Path, info: VideoInfo, score: float = 0.3) -> list[dict[str, Any]]:
    """Use ffmpeg's scene filter; parse showinfo pts of selected frames."""
    # We only need timestamps of scene-change frames. showinfo prints to stderr.
    from recast.ffmpeg import which_ffmpeg
    import subprocess

    check_cancel()
    cmd = [
        which_ffmpeg(),
        "-hide_banner",
        "-i",
        str(video),
        "-filter:v",
        f"select='gt(scene,{score})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    log.info("FFmpeg 切镜回退: scene>%s", score)
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    times: list[float] = []
    blob = (result.stderr or "") + "\n" + (result.stdout or "")
    for match in _FFMPEG_PTS.finditer(blob):
        try:
            times.append(float(match.group(1)))
        except ValueError:
            continue
    return _shots_from_cuts(times, info)


def detect_scenes(
    video: str | Path,
    workdir: Path,
    *,
    info: VideoInfo | None = None,
    threshold: float = CONTENT_ANCHOR,
    ffmpeg_score: float | None = None,
    min_shot_sec: float = DEFAULT_MIN_SHOT_SEC,
) -> dict[str, Any]:
    src = Path(video)
    workdir.mkdir(parents=True, exist_ok=True)
    meta = info or probe_video(src)
    score = ffmpeg_scene_score(threshold) if ffmpeg_score is None else float(ffmpeg_score)
    shots = _detect_pyscenedetect(src, meta, threshold)
    backend = "pyscenedetect"
    if shots is None:
        shots = _detect_ffmpeg(src, meta, score)
        backend = "ffmpeg"
    if not shots:
        shots = _shots_from_cuts([], meta)
        backend = backend + "+single"
    shots = merge_short_shots(shots, float(min_shot_sec))
    payload = _shots_payload(
        src,
        meta,
        shots,
        backend,
        threshold=float(threshold),
        ffmpeg_score=score,
        min_shot_sec=float(min_shot_sec),
    )
    path = workdir / "shots.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "切镜完成（%s，阈值 %.2f / FFmpeg scene>%.3f，最短 %.2fs）: %s 个镜头，写出 %s",
        backend_label(backend),
        float(threshold),
        score,
        float(min_shot_sec),
        len(shots),
        path,
    )
    return payload
