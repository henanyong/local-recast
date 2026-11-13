"""Tests for the task history panel (core queries + dialog).

Uses a throwaway SQLite file in tmp_path — never the user's real library.
Synthetic tasks only; no runner, no video, no execution.
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


def test_list_newest_first(svc, tmp_path: Path):
    from recast.core.task_history import list_history

    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="已完成",
          created_at="2026-11-10 10:00:00", updated_at="2026-11-10 10:05:00")
    _task(svc, drama["id"], episode="第2集", status="失败",
          created_at="2026-11-11 10:00:00", updated_at="2026-11-11 10:05:00",
          error="换脸模型加载失败")
    _task(svc, drama["id"], episode="第3集", status="处理中",
          created_at="2026-11-12 10:00:00", updated_at="2026-11-12 10:05:00")
    rows = list_history(svc.store)
    assert [r["episode_name"] for r in rows] == ["第3集", "第2集", "第1集"]
    assert rows[0]["drama_name"] == "演示剧"
    assert rows[1]["status_kind"] == "failed"
    assert rows[1]["error"] == "换脸模型加载失败"


def test_filter_by_kind_and_by_stored_text(svc):
    from recast.core.task_history import list_history

    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="已完成")
    _task(svc, drama["id"], episode="第2集", status="失败")
    _task(svc, drama["id"], episode="第3集", status="处理中")
    assert [r["episode_name"] for r in list_history(svc.store, status="failed")] == ["第2集"]
    assert [r["episode_name"] for r in list_history(svc.store, status="失败")] == ["第2集"]
    assert [r["episode_name"] for r in list_history(svc.store, status="done")] == ["第1集"]
    assert len(list_history(svc.store, status="pending")) == 0


def test_unknown_status_shown_verbatim_not_hidden(svc):
    from recast.core.task_history import list_history

    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="奇怪状态")
    rows = list_history(svc.store)
    assert len(rows) == 1
    assert rows[0]["status_kind"] == "unknown"
    assert rows[0]["status_label"] == "奇怪状态"


def test_config_snapshot_marks_unrecorded_fields(svc):
    from recast.core.task_history import get_history_entry

    drama = _drama(svc)
    row = _task(svc, drama["id"], episode="第1集", status="已完成")
    snap = get_history_entry(svc.store, row["id"])["config_snapshot"]
    assert snap["模板"] == "未记录"
    assert snap["选的人物"] == "未记录"
    assert snap["阶段清单"] == "未记录"
    assert snap["视频"] == "C:/demo/第1集.mp4"

    row2 = _task(
        svc, drama["id"], episode="第2集", status="已完成",
        template_pack_id="packA", template_version="v3",
        entry_type="import", source_script="剧本.txt",
    )
    snap2 = get_history_entry(svc.store, row2["id"])["config_snapshot"]
    assert snap2["模板"] == "packA（版本 v3）"
    assert snap2["入口类型"] == "import"
    assert snap2["剧本"] == "剧本.txt"
    assert snap2["台词表"] == "未记录"


def test_count_history(svc):
    from recast.core.task_history import count_history

    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="已完成")
    _task(svc, drama["id"], episode="第2集", status="失败")
    assert count_history(svc.store) == 2
    assert count_history(svc.store, status="failed") == 1
    assert count_history(svc.store, status="done") == 1


def test_build_rerun_task_same_config_fresh_state_not_inserted(svc):
    from recast.core.task_history import build_rerun_task

    drama = _drama(svc)
    row = _task(
        svc, drama["id"], episode="第2集", status="失败",
        progress="换脸失败", progress_pct=45, retry_count=2,
        error="显存不足", template_pack_id="packA", template_version="v3",
        entry_type="import", workdir="C:/works/ep2",
    )
    rerun = build_rerun_task(svc.store, row["id"])
    assert rerun["id"] != row["id"]
    assert rerun["drama_id"] == drama["id"]
    assert rerun["episode_name"] == "第2集"
    assert rerun["video_path"] == row["video_path"]
    assert rerun["template_pack_id"] == "packA"
    assert rerun["template_version"] == "v3"
    assert rerun["workdir"] == "C:/works/ep2"
    assert rerun["status"] == "待执行"
    assert rerun["progress"] == "等待执行"
    assert rerun["progress_pct"] == 0
    assert rerun["retry_count"] == 0
    assert rerun["error"] is None
    # Pure builder: nothing was inserted.
    assert svc.store.get("tasks", rerun["id"]) is None
    assert len(svc.store.list_rows("tasks")) == 1


def test_build_rerun_task_missing_raises(svc):
    from recast.core.task_history import build_rerun_task

    with pytest.raises(ValueError, match="找不到任务"):
        build_rerun_task(svc.store, "no-such-task")


def test_delete_removes_row_only_not_output_file(svc, tmp_path: Path):
    from recast.core.task_history import delete_history_entry

    drama = _drama(svc)
    out_file = tmp_path / "out.mp4"
    out_file.write_bytes(b"fake-video")
    row = _task(svc, drama["id"], episode="第1集", status="已完成",
                output_file=str(out_file))
    assert delete_history_entry(svc.store, row["id"]) is True
    assert svc.store.get("tasks", row["id"]) is None
    assert out_file.exists()  # 成片文件未动
    assert svc.store.get("dramas", drama["id"]) is not None  # 剧集行保留
    assert delete_history_entry(svc.store, row["id"]) is False


def test_pagination_limit_offset(svc):
    from recast.core.task_history import list_history

    drama = _drama(svc)
    for i in range(5):
        _task(svc, drama["id"], episode=f"第{i + 1}集", status="已完成",
              created_at=f"2026-11-1{i} 10:00:00", updated_at=f"2026-11-1{i} 10:00:00")
    page1 = list_history(svc.store, limit=2)
    page2 = list_history(svc.store, limit=2, offset=2)
    assert [r["episode_name"] for r in page1] == ["第5集", "第4集"]
    assert [r["episode_name"] for r in page2] == ["第3集", "第2集"]


def test_dialog_lists_tasks_and_shows_detail(svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="已完成",
          created_at="2026-11-10 10:00:00", updated_at="2026-11-10 10:00:00")
    _task(svc, drama["id"], episode="第2集", status="失败",
          error="换脸模型加载失败：显存不足", template_version="v3",
          created_at="2026-11-11 10:00:00", updated_at="2026-11-11 10:00:00")
    dlg = TaskHistoryDialog(svc)
    assert dlg.table.rowCount() == 2
    assert dlg.count_label.text() == "共 2 条"
    # Select the failed row (newest first: 第2集 is row 0).
    dlg.table.setCurrentCell(0, 0)
    assert "换脸模型加载失败" in dlg.detail.toPlainText()
    assert "未记录" in dlg.detail.toPlainText()  # 选的人物 / 阶段清单无记录
    assert "v3" in dlg.detail.toPlainText()
    dlg.close()


def test_dialog_rerun_inserts_and_starts(svc, monkeypatch):
    """Re-run now inserts the row and queues it (wired this round).

    The scheduler's real enqueue would start a worker thread, so the test
    captures the enqueue call instead of running the job.
    """
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    row = _task(svc, drama["id"], episode="第2集", status="失败",
                error="换脸失败", template_version="v3")
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    dlg = TaskHistoryDialog(svc)
    dlg.table.setCurrentCell(0, 0)
    dlg.rerun_btn.click()
    rerun = dlg.last_rerun_task
    assert rerun is not None
    assert rerun["id"] != row["id"]
    assert rerun["status"] == "待执行"
    assert rerun["template_version"] == "v3"
    assert rerun["started"] is True
    # Inserted and queued.
    assert svc.store.get("tasks", rerun["id"]) is not None
    assert len(svc.store.list_rows("tasks")) == 2
    assert len(captured) == 1
    assert captured[0].task_id == rerun["id"]
    assert "已排队" in dlg.status.text()
    dlg.close()


def test_dialog_delete_asks_first_and_keeps_output_file(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    out_file = tmp_path / "out.mp4"
    out_file.write_bytes(b"fake-video")
    _task(svc, drama["id"], episode="第1集", status="已完成",
          output_file=str(out_file))
    dlg = TaskHistoryDialog(svc)
    assert dlg.table.rowCount() == 1
    dlg.table.setCurrentCell(0, 0)

    # "No" keeps the row.
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    dlg.delete_btn.click()
    assert dlg.table.rowCount() == 1

    # "Yes" deletes the row only; the output file stays.
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    dlg.delete_btn.click()
    assert dlg.table.rowCount() == 0
    assert out_file.exists()
    dlg.close()


def test_dialog_status_filter(svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="已完成")
    _task(svc, drama["id"], episode="第2集", status="失败")
    _task(svc, drama["id"], episode="第3集", status="处理中")
    dlg = TaskHistoryDialog(svc)
    assert dlg.table.rowCount() == 3
    dlg.filter_box.setCurrentIndex(2)  # "失败"
    assert dlg.table.rowCount() == 1
    assert dlg.table.item(0, 0).text().endswith("第2集")
    dlg.close()
