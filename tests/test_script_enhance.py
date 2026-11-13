"""N24: per-shot enhance on the script path + clean shots can be flagged.

The script path stitches per-shot clips in ``stitch_and_dub`` and never
calls ``stages.compose.run``, so its enhance flags
(``workdir/shots_enhance.json``) used to be silently ignored, and the
only switch UI lived on review-flagged rows. The fix wires the same real
ffmpeg filter (hqdn3d + unsharp, via ``apply_shot_enhance``) as a
post-pass on the stitched output in ``runner.dispatch_job``, and the
flags file — not the review queue — is the source of truth, so any shot
in the plan (including a clean one) can be switched on.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_clip(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:d={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
    )
    return path


def _mean_abs_diff(a: Path, b: Path, at: float) -> float:
    import numpy as np

    def frame(video: Path) -> "np.ndarray":
        raw = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", str(at), "-i", str(video), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True,
        ).stdout
        return np.frombuffer(raw, dtype=np.uint8).astype(int)

    return float(abs(frame(a) - frame(b)).mean())


@pytest.fixture()
def svc(tmp_path: Path):
    from recast.core.service import RecastService

    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


@pytest.fixture()
def script_task(svc, tmp_path: Path):
    """A script task whose shots were never review-flagged (clean shots)."""
    drama = svc.store.insert(
        "dramas", origin_name="剧", root_path=str(tmp_path / "show"), episode_count=1
    )
    workdir = tmp_path / "w1"
    task = svc.store.insert(
        "tasks",
        drama_id=drama["id"],
        episode_name="e1",
        video_path=str(tmp_path / "a.mp4"),
        workdir=str(workdir),
        entry_type="script",
    )
    svc.store.replace_shots(
        task["id"],
        [
            {"shot_code": "S01", "start_time": 0.0, "end_time": 1.0},
            {"shot_code": "S02", "start_time": 1.0, "end_time": 2.0},
        ],
    )
    return svc, task, workdir


def _cfg_no_nvenc():
    from recast.config import RecastConfig

    cfg = RecastConfig()
    cfg.compose.use_nvenc = False
    return cfg


# ---------------------------------------------------------------------------
# 1. compose.enhance_script_output: the post-pass itself (real ffmpeg)
# ---------------------------------------------------------------------------


def test_enhance_script_output_applies_real_filter(tmp_path: Path):
    from recast.stages.compose import enhance_script_output

    src = _make_clip(tmp_path / "src.mp4")
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = enhance_script_output(
        src, workdir, [{"shot_code": "S01", "start": 0.0, "end": 1.0}],
        crf=18, use_nvenc=False,
    )
    assert out.is_file() and out != src
    assert out.read_bytes() != src.read_bytes()
    # Pixels move much more inside the flagged range than outside it.
    assert _mean_abs_diff(src, out, 0.5) > _mean_abs_diff(src, out, 1.5) * 2


def test_enhance_script_output_noop_without_flags(tmp_path: Path):
    from recast.stages.compose import enhance_script_output

    src = _make_clip(tmp_path / "src.mp4")
    workdir = tmp_path / "work"
    workdir.mkdir()
    assert enhance_script_output(src, workdir, [], crf=18, use_nvenc=False) == src
    assert enhance_script_output(src, workdir, None, crf=18, use_nvenc=False) == src
    # Unusable ranges only -> also a no-op, no intermediate file written.
    assert enhance_script_output(
        src, workdir, [{"shot_code": "S01", "start": 1.0, "end": 1.0}],
        crf=18, use_nvenc=False,
    ) == src
    assert not (workdir / "enhanced_script.mp4").exists()


def test_enhance_script_output_missing_video_raises(tmp_path: Path):
    from recast.stages.compose import enhance_script_output

    with pytest.raises(FileNotFoundError):
        enhance_script_output(
            tmp_path / "nope.mp4", tmp_path,
            [{"shot_code": "S01", "start": 0.0, "end": 1.0}],
            crf=18, use_nvenc=False,
        )


# ---------------------------------------------------------------------------
# 2. Clean shots can be flagged without ever entering the review queue
# ---------------------------------------------------------------------------


def test_clean_shot_can_be_flagged_without_review(script_task):
    svc, task, workdir = script_task
    # No quality_marks / review items exist for this task at all.
    assert svc.store.list_rows("quality_marks", "task_id = ?", (task["id"],)) == []
    assert svc.set_shot_enhance(task["id"], "S02", True) == {"S02": True}
    assert svc.shot_enhance_flags(task["id"]) == {"S02": True}
    assert json.loads((workdir / "shots_enhance.json").read_text(encoding="utf-8")) == {"S02": True}
    # And it can be switched back off again.
    assert svc.set_shot_enhance(task["id"], "S02", False) == {}


def test_enhance_shots_for_task_reads_script_plan_and_narration_times(tmp_path: Path):
    from recast.core.runner import enhance_shots_for_task

    workdir = tmp_path / "w"
    workdir.mkdir()
    # Flags written directly to the file (the non-UI entry point): no
    # store at all, shots only in the script plan, times only as
    # narration_start / narration_end.
    (workdir / "shots_enhance.json").write_text(
        json.dumps({"S02": True}), encoding="utf-8"
    )
    (workdir / "shots_plan.json").write_text(
        json.dumps([
            {"shot_code": "S01", "start_time": 0.0, "end_time": 1.0},
            {"shot_code": "S02", "narration_start": 1.0, "narration_end": 2.0},
        ]),
        encoding="utf-8",
    )
    assert enhance_shots_for_task(None, None, workdir) == [
        {"shot_code": "S02", "start": 1.0, "end": 2.0}
    ]


# ---------------------------------------------------------------------------
# 3. dispatch_job script path: flags reach the stitched output
# ---------------------------------------------------------------------------


def _fake_stitcher(tmp_path: Path, workdir: Path, *, corrupt: bool = False):
    """Stand-in for run_script_episode: writes a stitched output + plan.

    Returns (fake, orig_clip) where orig_clip is an identical,
    unenhanced copy of what the fake wrote as the episode output.
    """
    orig = tmp_path / "orig.mp4"

    def fake(**kwargs):
        wd = Path(kwargs["workdir"])
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "shots_plan.json").write_text(
            json.dumps([
                {"shot_code": "S01", "start_time": 0.0, "end_time": 1.0},
                {"shot_code": "S02", "start_time": 1.0, "end_time": 2.0},
            ]),
            encoding="utf-8",
        )
        out = wd / "output.mp4"
        if corrupt:
            out.write_bytes(b"")
        else:
            _make_clip(out)
            orig.write_bytes(out.read_bytes())
        return {"entry": "script", "output": str(out)}

    return fake, orig


def test_script_dispatch_applies_enhance_to_flagged_clean_shot(
    script_task, tmp_path: Path, monkeypatch
):
    from recast.core.runner import dispatch_job

    svc, task, workdir = script_task
    # S01 is a clean shot (never review-flagged) — flag it anyway.
    svc.set_shot_enhance(task["id"], "S01", True)
    fake, orig = _fake_stitcher(tmp_path, workdir)
    monkeypatch.setattr("recast.core.script_pipeline.run_script_episode", fake)
    summary = dispatch_job(
        video=None,
        workdir=workdir,
        config=_cfg_no_nvenc(),
        stages=["dub", "compose"],
        entry_type="script",
        pack_root=tmp_path / "pack",
        store=svc.store,
        task_id=task["id"],
    )
    assert summary.get("enhance_applied") is True
    assert summary.get("enhanced_shots") == ["S01"]
    out = Path(summary["output"])
    assert out.is_file()
    assert not (workdir / "enhanced_script.mp4").exists()  # moved over output
    assert out.read_bytes() != orig.read_bytes()
    # The real filter ran, and only inside the flagged range.
    assert _mean_abs_diff(orig, out, 0.5) > _mean_abs_diff(orig, out, 1.5) * 2


def test_script_dispatch_leaves_unflagged_output_alone(
    script_task, tmp_path: Path, monkeypatch
):
    from recast.core.runner import dispatch_job

    svc, task, workdir = script_task
    fake, orig = _fake_stitcher(tmp_path, workdir)
    monkeypatch.setattr("recast.core.script_pipeline.run_script_episode", fake)
    summary = dispatch_job(
        video=None,
        workdir=workdir,
        config=_cfg_no_nvenc(),
        stages=["dub", "compose"],
        entry_type="script",
        pack_root=tmp_path / "pack",
        store=svc.store,
        task_id=task["id"],
    )
    assert "enhance_applied" not in summary
    assert Path(summary["output"]).read_bytes() == orig.read_bytes()
    assert not (workdir / "enhanced_script.mp4").exists()


def test_script_dispatch_enhance_failure_keeps_output(
    script_task, tmp_path: Path, monkeypatch
):
    from recast.core.runner import dispatch_job

    svc, task, workdir = script_task
    svc.set_shot_enhance(task["id"], "S01", True)
    fake, _orig = _fake_stitcher(tmp_path, workdir, corrupt=True)
    monkeypatch.setattr("recast.core.script_pipeline.run_script_episode", fake)
    summary = dispatch_job(
        video=None,
        workdir=workdir,
        config=_cfg_no_nvenc(),
        stages=["dub", "compose"],
        entry_type="script",
        pack_root=tmp_path / "pack",
        store=svc.store,
        task_id=task["id"],
    )
    # The episode is not failed by the enhance step: the unenhanced
    # output survives and the failure is recorded, not silent.
    assert summary.get("enhance_applied") is False
    assert summary.get("enhance_error")
    assert Path(summary["output"]).is_file()
