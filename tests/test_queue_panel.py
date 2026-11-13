"""Tests for the queue panel (G3 / PM 9.2).

Uses a throwaway SQLite file in tmp_path — never the user's real library.
Synthetic tasks only; no runner, no video, no execution. The scheduler's
real enqueue is captured by monkeypatch so a retry click never runs a job.
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


def _row_for(dlg, episode: str) -> int:
    """Table row index whose title ends with the given episode name."""
    for i in range(dlg.table.rowCount()):
        item = dlg.table.item(i, 0)
        if item is not None and item.text().endswith(episode):
            return i
    raise AssertionError(f"no row for {episode!r} in panel")


def _retry_button(dlg, row_index: int):
    from PySide6.QtWidgets import QPushButton

    widget = dlg.table.cellWidget(row_index, 5)
    return widget if isinstance(widget, QPushButton) else None


def test_panel_lists_mixed_status_rows_match_store(svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="待执行",
          progress="等待执行",
          created_at="2026-11-10 10:00:00", updated_at="2026-11-10 10:00:00")
    _task(svc, drama["id"], episode="第2集", status="处理中",
          progress="人物替换中", progress_pct=45,
          created_at="2026-11-11 10:00:00", updated_at="2026-11-11 10:00:00")
    _task(svc, drama["id"], episode="第3集", status="已完成",
          progress="已完成",
          created_at="2026-11-12 10:00:00", updated_at="2026-11-12 10:00:00")
    _task(svc, drama["id"], episode="第4集", status="失败",
          progress="人物替换失败", error="显存不足",
          created_at="2026-11-12 11:00:00", updated_at="2026-11-12 11:00:00")
    dlg = QueuePanelDialog(svc)
    # P3/N33: [仅进行中] is checked by default, so the done row is hidden.
    assert dlg.only_in_progress_checkbox.isChecked() is True
    assert dlg.table.rowCount() == 3
    assert dlg.count_label.text() == "共 3 条"
    with pytest.raises(AssertionError):
        _row_for(dlg, "第3集")

    # Unchecking the filter shows all rows, including the done one.
    dlg.only_in_progress_checkbox.setChecked(False)
    assert dlg.table.rowCount() == 4
    assert dlg.count_label.text() == "共 4 条"

    # Status label, current step and failure reason all match the store.
    assert dlg.table.item(_row_for(dlg, "第1集"), 1).text() == "待执行"
    assert dlg.table.item(_row_for(dlg, "第2集"), 1).text() == "处理中"
    assert dlg.table.item(_row_for(dlg, "第2集"), 2).text() == "人物替换中"
    assert dlg.table.item(_row_for(dlg, "第3集"), 1).text() == "已完成"
    failed = _row_for(dlg, "第4集")
    assert dlg.table.item(failed, 1).text() == "失败"
    assert dlg.table.item(failed, 2).text() == "人物替换失败"
    assert dlg.table.item(failed, 4).text() == "显存不足"
    # Drama name is on every row.
    assert dlg.table.item(0, 0).text().startswith("演示剧 — ")
    dlg.close()


def test_progress_bar_only_when_pct_stored(svc):
    from PySide6.QtWidgets import QProgressBar

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="处理中",
          progress="人物替换中", progress_pct=45)
    _task(svc, drama["id"], episode="第2集", status="待执行",
          progress="等待执行", progress_pct=0)
    dlg = QueuePanelDialog(svc)

    running = _row_for(dlg, "第1集")
    bar = dlg.table.cellWidget(running, 3)
    assert isinstance(bar, QProgressBar)
    assert bar.value() == 45

    # No stored percentage -> "—", never a fabricated 0% bar.
    pending = _row_for(dlg, "第2集")
    assert dlg.table.cellWidget(pending, 3) is None
    assert dlg.table.item(pending, 3).text() == "—"
    dlg.close()


def test_failed_row_has_resume_retry_and_it_calls_service_start(svc, monkeypatch):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    row = _task(svc, drama["id"], episode="第2集", status="失败",
                progress="人物替换失败", error="换脸模型加载失败",
                entry_type="import")
    _task(svc, drama["id"], episode="第1集", status="已完成", progress="已完成")

    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    dlg = QueuePanelDialog(svc)
    failed = _row_for(dlg, "第2集")
    assert dlg.table.item(failed, 4).text() == "换脸模型加载失败"

    retry = _retry_button(dlg, failed)
    assert retry is not None
    assert retry.text() == "从失败步续跑"
    assert "不新建任务" in retry.toolTip()
    retry.click()

    # Same-task resume via service.start: the failed task is queued,
    # no new task row is created, and rerun_task was never involved.
    assert len(captured) == 1
    assert captured[0].task_id == row["id"]
    assert svc.store.get("tasks", row["id"]) is not None
    assert len(svc.store.list_rows("tasks")) == 2
    assert dlg.last_retry is not None
    assert dlg.last_retry["started"] is True
    assert dlg.last_retry["task_id"] == row["id"]
    assert "从失败步续跑" in dlg.status.text()
    dlg.close()


def test_retry_button_absent_on_non_failed_rows(svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="处理中",
          progress="人物替换中", progress_pct=30)
    _task(svc, drama["id"], episode="第2集", status="已完成", progress="已完成")
    _task(svc, drama["id"], episode="第3集", status="待执行", progress="等待执行")
    dlg = QueuePanelDialog(svc)
    # P3/N33: the done row is hidden by the default [仅进行中] filter.
    dlg.only_in_progress_checkbox.setChecked(False)
    for episode in ("第1集", "第2集", "第3集"):
        row_index = _row_for(dlg, episode)
        assert _retry_button(dlg, row_index) is None
        assert dlg.table.item(row_index, 5).text() == "—"
    dlg.close()


def test_empty_queue_shows_empty_hint(svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    dlg = QueuePanelDialog(svc)
    assert dlg.table.rowCount() == 0
    assert dlg.count_label.text() == "共 0 条"
    assert dlg.empty_label.isHidden() is False
    # P3/N33: with the default [仅进行中] filter on, the empty hint says
    # there is nothing in progress (not the old "see history" pointer).
    assert "没有进行中的任务" in dlg.empty_label.text()
    assert "取消勾选" in dlg.empty_label.text()
    # With the filter off and a truly empty store, the old hint applies.
    dlg.only_in_progress_checkbox.setChecked(False)
    assert "队列为空" in dlg.empty_label.text()
    assert "任务历史" in dlg.empty_label.text()
    dlg.close()


def test_refresh_picks_up_new_task(svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="待执行")
    dlg = QueuePanelDialog(svc)
    assert dlg.table.rowCount() == 1
    assert dlg.empty_label.isHidden() is True

    _task(svc, drama["id"], episode="第2集", status="处理中",
          progress="人物替换中", progress_pct=10)
    dlg.refresh()
    assert dlg.table.rowCount() == 2
    assert dlg.count_label.text() == "共 2 条"
    assert _row_for(dlg, "第2集") >= 0
    assert dlg.table.item(_row_for(dlg, "第2集"), 2).text() == "人物替换中"
    dlg.close()


def test_retry_failure_is_reported_in_status_line_not_a_popup(svc, monkeypatch):
    """service.start raising must not crash the panel or pop a modal."""
    from PySide6.QtWidgets import QMessageBox

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第2集", status="失败",
          progress="人物替换失败", error="显存不足", entry_type="import")
    monkeypatch.setattr(
        svc, "start", lambda *a, **k: (_ for _ in ()).throw(ValueError("没有可执行的任务")),
    )
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not pop a modal"))),
    )
    dlg = QueuePanelDialog(svc)
    retry = _retry_button(dlg, _row_for(dlg, "第2集"))
    assert retry is not None
    retry.click()
    assert dlg.last_retry is not None
    assert dlg.last_retry["started"] is False
    assert "没有可执行的任务" in dlg.status.text()
    assert dlg.table.rowCount() == 1  # panel still intact
    dlg.close()
