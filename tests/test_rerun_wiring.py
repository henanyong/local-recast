"""Tests for the task-history re-run execution wiring (round: service/config).

``RecastService.rerun_task`` builds the re-run dict, inserts it and starts
it through the scheduler with the same Job construction as 开始执行.
The scheduler's real ``enqueue`` would start a worker thread, so these
tests capture the enqueue call instead of running the job — no video,
no runner, no execution. The dialog tests click the real button against
the real service method with the same capture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


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


def _drama(svc, name: str = "演示剧") -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, new_name=name,
        root_path=f"C:/demo/{name}", episode_count=1,
    )


def _task(svc, drama_id: str, *, episode: str, status: str, **extra) -> dict:
    fields: dict = {
        "drama_id": drama_id,
        "episode_name": episode,
        "video_path": f"C:/demo/{episode}.mp4",
        "status": status,
        "progress": "等待执行",
        "progress_pct": 0,
    }
    fields.update(extra)
    return svc.store.insert("tasks", **fields)


def _capture_enqueue(svc):
    """Replace scheduler.enqueue with a recorder; returns the captured jobs."""
    captured: list = []

    def fake_enqueue(jobs):
        captured.extend(jobs)
        return len(jobs)

    svc.scheduler.enqueue = fake_enqueue  # type: ignore[method-assign]
    return captured


def _authorized_character(svc, tmp_path: Path, name: str = "温念") -> dict:
    face = tmp_path / "face.jpg"
    face.write_bytes(b"fake-face")
    ch = svc.store.insert(
        "characters", name=name, display_name=name, confirmed=1,
        authorized=1, auth_type="自有拍摄", cluster_key=f"manual:{name}",
    )
    svc.store.set_character_asset(ch["id"], "face", str(face), complete=True)
    return ch


def test_rerun_inserts_and_enqueues_without_replace_when_no_auth(svc, tmp_path: Path):
    from recast.core.constants import STAGE_GROUP_IDS

    drama = _drama(svc)
    old = _task(
        svc, drama["id"], episode="第1集", status="失败", error="换脸模型加载失败",
        entry_type="import", source_dialogue="台词.json",
        source_storyboard="分镜.json", template_pack_id="packA",
        template_version="v3", pack_snapshot="packA-snap",
    )
    captured = _capture_enqueue(svc)
    result = svc.rerun_task(old["id"])

    assert result["started"] is True
    assert result["reason"] is None
    assert result["id"] != old["id"]
    new_row = svc.store.get("tasks", result["id"])
    assert new_row is not None
    assert new_row["status"] == "待执行"
    assert new_row["progress_pct"] == 0
    assert new_row["retry_count"] == 0
    assert new_row["error"] is None
    assert new_row["drama_id"] == drama["id"]
    assert new_row["video_path"] == old["video_path"]
    assert new_row["entry_type"] == "import"
    assert new_row["source_dialogue"] == "台词.json"
    assert new_row["source_storyboard"] == "分镜.json"
    assert new_row["template_pack_id"] == "packA"
    # No authorized face: replace is dropped, the rest runs (same rule as
    # 开始执行 with the default stage list).
    assert len(captured) == 1
    job = captured[0]
    assert job.drama_id == drama["id"]
    assert job.task_id == result["id"]
    assert job.stages == [s for s in STAGE_GROUP_IDS if s != "replace"]
    assert job.character_ids is None
    assert job.entry_type == "import"
    assert job.pack_root == "packA-snap"
    assert job.storyboard == "分镜.json"
    assert job.dialogue == "台词.json"
    assert job.require_auth is True


def test_rerun_keeps_replace_with_authorized_face(svc, tmp_path: Path):
    from recast.core.constants import STAGE_GROUP_IDS

    drama = _drama(svc)
    old = _task(svc, drama["id"], episode="第1集", status="失败")
    ch = _authorized_character(svc, tmp_path)
    captured = _capture_enqueue(svc)
    result = svc.rerun_task(old["id"])

    assert result["started"] is True
    assert len(captured) == 1
    job = captured[0]
    assert job.stages == STAGE_GROUP_IDS
    assert job.character_ids == [ch["id"]]


def test_rerun_missing_task_raises(svc):
    with pytest.raises(ValueError, match="找不到任务"):
        svc.rerun_task("no-such-task")


def test_rerun_missing_drama_raises_and_inserts_nothing(svc):
    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集", status="失败")
    # Orphan the task: delete its drama with FK checks off (the schema
    # would otherwise cascade-delete the task row itself).
    con = svc.store.connect()
    con.execute("PRAGMA foreign_keys=OFF")
    try:
        con.execute("DELETE FROM dramas WHERE id = ?", (drama["id"],))
        con.commit()
    finally:
        con.execute("PRAGMA foreign_keys=ON")
    before = len(svc.store.list_rows("tasks"))
    with pytest.raises(ValueError, match="没有所属剧目"):
        svc.rerun_task(task["id"])
    assert len(svc.store.list_rows("tasks")) == before


def test_rerun_enqueue_refused_reports_not_started(svc):
    drama = _drama(svc)
    old = _task(svc, drama["id"], episode="第1集", status="失败")
    svc.scheduler.enqueue = lambda jobs: 0  # type: ignore[method-assign]
    result = svc.rerun_task(old["id"])

    assert result["started"] is False
    assert "未能排队" in (result["reason"] or "")
    # The row is still inserted — only the start did not happen.
    assert svc.store.get("tasks", result["id"]) is not None


def _dialog(svc, tmp_path: Path):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="失败", error="换脸模型加载失败")
    dlg = TaskHistoryDialog(svc)
    dlg.table.setCurrentCell(0, 0)
    return dlg


def test_dialog_rerun_button_starts_and_refreshes(svc, tmp_path: Path):
    captured = _capture_enqueue(svc)
    dlg = _dialog(svc, tmp_path)
    before = dlg.table.rowCount()
    dlg.rerun_btn.click()

    assert dlg.last_rerun_task is not None
    assert dlg.last_rerun_task["started"] is True
    assert "已排队" in dlg.status.text()
    assert len(captured) == 1
    # The new row shows up in the list after refresh.
    assert dlg.table.rowCount() == before + 1


def test_dialog_rerun_parked_shows_reason(svc, tmp_path: Path):
    dlg = _dialog(svc, tmp_path)
    svc.rerun_task = lambda task_id: {  # type: ignore[method-assign]
        "id": "new-id", "started": False,
        "reason": "缺少已授权角色的参考脸", "stages": [],
    }
    dlg.rerun_btn.click()

    assert "未启动" in dlg.status.text()
    assert "缺少已授权角色的参考脸" in dlg.status.text()


def test_dialog_rerun_missing_task_shows_error(svc, tmp_path: Path):
    dlg = _dialog(svc, tmp_path)

    def _boom(task_id):
        raise ValueError("找不到任务：gone")

    svc.rerun_task = _boom  # type: ignore[method-assign]
    dlg.rerun_btn.click()

    assert "找不到任务" in dlg.status.text()
