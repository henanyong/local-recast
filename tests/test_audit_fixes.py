"""Regression tests for the workbench / pipeline audit fixes."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from recast.media import alias_input, find_input_alias, same_media_file, workdir_leaf, write_source_record
from recast.pipeline import _make_workdir, artifact_status, resolve_plan
from recast.runtime import CancelledError, bind_job, clear_stop, request_stop, stopping
from recast.stages.subtitles import write_srt


def _touch_video(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_workdir_leaf_distinguishes_same_stem():
    a = Path("/shows/S01/01.mp4")
    b = Path("/shows/S02/01.mp4")
    assert workdir_leaf(a, 1) != workdir_leaf(b, 2)
    assert workdir_leaf(a, 1).startswith("001_")
    assert workdir_leaf(b, 2).startswith("002_")


def test_alias_input_does_not_reuse_other_source(tmp_path: Path):
    src1 = _touch_video(tmp_path / "S01" / "01.mp4", b"CLIP-ONE")
    src2 = _touch_video(tmp_path / "S02" / "01.mp4", b"CLIP-TWO")
    work = tmp_path / "run"
    alias1 = alias_input(src1, work)
    assert alias1.is_file()
    assert alias1.read_bytes() == b"CLIP-ONE"
    alias2 = alias_input(src2, work)
    assert alias2 != alias1 or same_media_file(alias2, src2)
    assert alias2.read_bytes() == b"CLIP-TWO"
    assert find_input_alias(work, src1) is not None
    assert find_input_alias(work, src1).read_bytes() == b"CLIP-ONE"
    found2 = find_input_alias(work, src2)
    assert found2 is not None
    assert found2.read_bytes() == b"CLIP-TWO"


def test_same_media_file_follows_hardlink(tmp_path: Path):
    src = _touch_video(tmp_path / "src.mp4", b"data")
    dest = tmp_path / "input.mp4"
    dest.hardlink_to(src)
    assert same_media_file(src, dest)
    other = _touch_video(tmp_path / "other.mp4", b"data")
    assert not same_media_file(src, other)


def test_scan_does_not_drop_tasks_from_other_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import recast.workbench as wb

    monkeypatch.setattr(wb, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(wb, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(wb, "ASSETS_DIR", tmp_path / "assets")
    wb.BOARD = wb.Board()

    root_a = tmp_path / "剧集A"
    root_b = tmp_path / "剧集B"
    _touch_video(root_a / "S01" / "01.mp4", b"a1")
    _touch_video(root_a / "S02" / "01.mp4", b"a2")
    # folder tasks come from subdirs of the scan root
    show_a = root_a / "show"
    _touch_video(show_a / "S01" / "01.mp4", b"a1")
    _touch_video(show_a / "S02" / "01.mp4", b"a2")
    show_b = root_b / "show"
    _touch_video(show_b / "b1.mp4", b"b")

    wb.scan_directory(root_a)
    for t in wb.BOARD.tasks:
        t["settings"]["face"] = str(tmp_path / "face.jpg")
        t["settings"]["names"] = "1=张三"
    ids_a = {t["id"] for t in wb.BOARD.tasks}
    assert ids_a

    wb.scan_directory(root_b)
    paths = {t["path"] for t in wb.BOARD.tasks}
    assert str(show_a) in paths
    assert str(show_b) in paths
    kept = [t for t in wb.BOARD.tasks if t["path"] == str(show_a)]
    assert kept
    assert kept[0]["settings"]["face"] == str(tmp_path / "face.jpg")
    assert kept[0]["settings"]["names"] == "1=张三"
    assert kept[0]["id"] in ids_a


def test_clip_workdir_unique_for_same_stem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import recast.workbench as wb

    monkeypatch.setattr(wb, "RUNS_DIR", tmp_path / "runs")
    c1 = _touch_video(tmp_path / "S01" / "01.mp4", b"one")
    c2 = _touch_video(tmp_path / "S02" / "01.mp4", b"two")
    task = {"id": "abc123", "clips": [str(c1), str(c2)], "workdir_map": {}}
    d1 = wb._clip_workdir(task, c1, 1)
    d2 = wb._clip_workdir(task, c2, 2)
    assert d1 != d2
    assert d1.name.startswith("001_")
    assert d2.name.startswith("002_")
    assert d1.is_dir() and d2.is_dir()


def test_stop_does_not_clear_queue_or_leave_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import recast.workbench as wb

    monkeypatch.setattr(wb, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(wb, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(wb, "ASSETS_DIR", tmp_path / "assets")
    wb.BOARD = wb.Board()
    clear_stop()
    bind_job()

    video_a = _touch_video(tmp_path / "a.mp4", b"A")
    video_b = _touch_video(tmp_path / "b.mp4", b"B")
    snap_a = wb.scan_directory(video_a, merge=False)
    id_a = snap_a["tasks"][0]["id"]
    snap_b = wb.scan_directory(video_b, merge=True)
    tasks = {t["path"]: t for t in snap_b["tasks"]}
    id_b = tasks[str(video_b)]["id"]
    # restore A if merge kept it
    assert wb.BOARD.get_task(id_a) is not None

    started = threading.Event()
    calls: list[str] = []

    def fake_run(task, clip, stages, idx, total, force_rerun=False):
        calls.append(task["id"])
        if task["id"] == id_a:
            started.set()
            deadline = time.time() + 3
            while time.time() < deadline:
                if stopping():
                    raise CancelledError("已停止当前任务")
                time.sleep(0.02)
            raise TimeoutError("first job never saw stop")
        return {"output": str(tmp_path / "out.mp4"), "workdir": str(tmp_path)}

    monkeypatch.setattr(wb, "_run_one_clip", fake_run)

    wb.start_jobs([id_a], ["scan"])
    assert started.wait(timeout=3)
    wb.start_jobs([id_b], ["scan"])
    assert any(item[0] == id_b for item in wb.BOARD.job_queue) or wb.BOARD.current_id == id_b
    wb.stop_current()

    deadline = time.time() + 5
    while time.time() < deadline:
        task_b = wb.BOARD.get_task(id_b)
        if task_b and task_b["status"] == wb.STATUS_DONE and not stopping():
            break
        if not (wb._worker_thread and wb._worker_thread.is_alive()) and not stopping():
            break
        time.sleep(0.05)

    task_a = wb.BOARD.get_task(id_a)
    task_b = wb.BOARD.get_task(id_b)
    assert task_a["status"] == wb.STATUS_PENDING
    assert stopping() is False
    assert id_b in calls or task_b["status"] in {wb.STATUS_DONE, wb.STATUS_RUNNING, wb.STATUS_PENDING}
    # The previously queued B must not have been discarded by job_queue.clear().
    assert task_b["status"] != wb.STATUS_FAILED
    # Give the worker a moment to pick B up if it hasn't finished yet.
    if task_b["status"] == wb.STATUS_PENDING:
        deadline = time.time() + 3
        while time.time() < deadline and task_b["status"] == wb.STATUS_PENDING:
            time.sleep(0.05)
            task_b = wb.BOARD.get_task(id_b)
    assert task_b["status"] == wb.STATUS_DONE
    assert id_b in calls


def test_resolve_plan_skips_existing_unless_forced(tmp_path: Path):
    work = tmp_path / "run"
    work.mkdir()
    (work / "input.mp4").write_bytes(b"x")
    (work / "meta.json").write_text("{}", encoding="utf-8")
    (work / "subtitles.srt").write_text("1\n", encoding="utf-8")
    (work / "subtitles.json").write_text("{}", encoding="utf-8")
    (work / "tracks_raw.json").write_text("{}", encoding="utf-8")
    (work / "identity.json").write_text("{}", encoding="utf-8")
    (work / "tracks.json").write_text("{}", encoding="utf-8")
    (work / "output.mp4").write_bytes(b"out")

    assert resolve_plan(["extract"], work) == []
    assert resolve_plan(["library"], work) == []
    assert resolve_plan(["compose"], work) == []
    assert resolve_plan(["extract", "library", "compose"], work) == []
    assert "extract" in resolve_plan(["extract"], work, force=True)
    assert "compose" in resolve_plan(["compose"], work, force=True)

    # A later stage with a missing prereq still pulls the prereq in.
    (work / "identity.json").unlink()
    plan = resolve_plan(["library"], work)
    assert "library" in plan
    assert "extract" not in plan  # extract artifacts still present


def test_artifact_status_compose_requires_output_mp4(tmp_path: Path):
    work = tmp_path / "run"
    work.mkdir()
    (work / "summary.json").write_text("{}", encoding="utf-8")
    (work / "subtitles.srt").write_text("", encoding="utf-8")
    status = artifact_status(work)
    assert status["compose"] is False
    (work / "output.mp4").write_bytes(b"x")
    assert artifact_status(work)["compose"] is True


def test_select_compose_picture_ignores_stale_replace(tmp_path: Path):
    from recast.core.runner import select_compose_picture

    video = _touch_video(tmp_path / "input.mp4", b"orig")
    replaced = _touch_video(tmp_path / "replaced.mp4", b"old-face")
    assert select_compose_picture(video, replaced, skip_replace=True, has_face=False) == video
    assert select_compose_picture(video, replaced, skip_replace=False, has_face=False) == video
    assert select_compose_picture(video, replaced, skip_replace=False, has_face=True) == replaced


def test_write_srt_skips_empty_without_jumping_index(tmp_path: Path):
    segs = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "  "},
        {"start": 2.0, "end": 3.0, "text": "world"},
    ]
    path = write_srt(segs, tmp_path / "a.srt")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("1\n")
    assert "\n2\n" in text
    assert "\n3\n" not in text
    assert "hello" in text and "world" in text


def test_make_workdir_ascii_when_out_parent_is_chinese(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import recast.pipeline as pipeline

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    src = _touch_video(tmp_path / "源视频.mp4", b"v")
    out = tmp_path / "剧集" / "成片.mp4"
    out.parent.mkdir()
    work = pipeline._make_workdir(src, out, None)
    assert "剧集" not in work.as_posix()
    assert work.is_relative_to(tmp_path / "outputs")
    meta = work / "source_meta.json"
    assert meta.is_file()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["display_name"] == "源视频"
    assert data["original_name"] == "源视频.mp4"


def test_removing_face_reverts_need_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import recast.workbench as wb

    monkeypatch.setattr(wb, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(wb, "ASSETS_DIR", tmp_path / "assets")
    wb.BOARD = wb.Board()
    video = _touch_video(tmp_path / "x.mp4", b"x")
    face = tmp_path / "face.jpg"
    face.write_bytes(b"img")
    snap = wb.scan_directory(video, merge=False)
    tid = snap["tasks"][0]["id"]
    wb.update_settings(tid, {"face": str(face)})
    task = wb.BOARD.get_task(tid)
    # pending is fine when face is present
    assert task["status"] in {wb.STATUS_PENDING, wb.STATUS_NEED_ASSETS}
    # no face → 待补资料 (default plan includes replace/compose)
    task["status"] = wb.STATUS_PENDING
    wb.update_settings(tid, {"face": None})
    task = wb.BOARD.get_task(tid)
    assert task["status"] == wb.STATUS_NEED_ASSETS


def test_should_skip_respects_force(tmp_path: Path):
    from recast.core.checkpoint import should_skip

    out = tmp_path / "done.txt"
    out.write_text("ok", encoding="utf-8")
    assert should_skip(None, None, "x", [], [out]) is True
    assert should_skip(None, None, "x", [], [out], force=True) is False


def test_unknown_config_keys_are_logged(tmp_path: Path, caplog):
    from recast.config import RecastConfig

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "device: cpu\nvram_gb: 8\ndetect:\n  imgsz: 640\n  img_size: 512\nunknown_top: 1\n",
        encoding="utf-8",
    )
    caplog.set_level("WARNING", logger="recast")
    RecastConfig.from_yaml(path)
    text = caplog.text
    assert "img_size" in text
    assert "unknown_top" in text


def test_subtitle_escape_covers_special_chars(tmp_path: Path):
    from recast.ffmpeg import _escape_subtitles_path, prepare_subtitle_file, subtitles_filter

    # Use an ASCII dest so we can inspect the escaped result of the copy.
    src = tmp_path / "subs,file[1].srt"
    src.write_text("1\n", encoding="utf-8")
    dest_dir = tmp_path / "ascii_out"
    dest_dir.mkdir()
    copied = prepare_subtitle_file(src, dest_dir)
    assert copied.name == "burn_subs.srt"
    escaped = _escape_subtitles_path(copied)
    assert escaped.startswith("'") and escaped.endswith("'")
    filt = subtitles_filter(src, dest_dir=dest_dir)
    assert filt.startswith("subtitles=")
    # original special-character name is not used in the filter
    assert "subs,file" not in filt
