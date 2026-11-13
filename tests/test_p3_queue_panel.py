"""P3/N33 tests for the queue panel: auto-refresh timer, [仅进行中]
filter, and filter-aware empty text.

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


def _row_for(dlg, episode: str) -> int:
    for i in range(dlg.table.rowCount()):
        item = dlg.table.item(i, 0)
        if item is not None and item.text().endswith(episode):
            return i
    raise AssertionError(f"no row for {episode!r} in panel")


def _episodes(dlg) -> list[str]:
    return [
        dlg.table.item(i, 0).text().rsplit(" — ", 1)[-1]
        for i in range(dlg.table.rowCount())
    ]


def test_only_in_progress_checked_by_default_and_hides_done(svc):
    """Default filter shows unfinished tasks only; unchecking shows all."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="待执行")
    _task(svc, drama["id"], episode="第2集", status="处理中",
          progress="人物替换中", progress_pct=45)
    _task(svc, drama["id"], episode="第3集", status="已完成", progress="已完成")
    _task(svc, drama["id"], episode="第4集", status="失败",
          progress="人物替换失败", error="显存不足")

    dlg = QueuePanelDialog(svc)
    assert dlg.only_in_progress_checkbox.isChecked() is True
    assert dlg.table.rowCount() == 3
    assert dlg.count_label.text() == "共 3 条"
    assert set(_episodes(dlg)) == {"第1集", "第2集", "第4集"}
    # A failed row is unfinished: it stays visible so its retry button
    # remains reachable under the default filter.
    assert _row_for(dlg, "第4集") >= 0

    dlg.only_in_progress_checkbox.setChecked(False)
    assert dlg.table.rowCount() == 4
    assert dlg.count_label.text() == "共 4 条"
    assert _row_for(dlg, "第3集") >= 0

    dlg.only_in_progress_checkbox.setChecked(True)
    assert dlg.table.rowCount() == 3
    with pytest.raises(AssertionError):
        _row_for(dlg, "第3集")
    dlg.close()


def test_failed_row_still_has_retry_under_default_filter(svc, monkeypatch):
    """Boundary: the default filter must not hide failed rows (and their
    [从失败步续跑] button) behind the done rows it removes."""
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    row = _task(svc, drama["id"], episode="第2集", status="失败",
                progress="人物替换失败", error="显存不足", entry_type="import")
    _task(svc, drama["id"], episode="第1集", status="已完成", progress="已完成")
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    dlg = QueuePanelDialog(svc)
    assert dlg.table.rowCount() == 1
    failed = _row_for(dlg, "第2集")
    retry = dlg.table.cellWidget(failed, 5)
    assert isinstance(retry, QPushButton)
    assert retry.text() == "从失败步续跑"
    retry.click()
    assert len(captured) == 1
    assert captured[0].task_id == row["id"]
    assert dlg.last_retry is not None and dlg.last_retry["started"] is True
    dlg.close()


def test_auto_refresh_timer_picks_up_new_task_and_never_pops_modal(
    svc, monkeypatch,
):
    """The QTimer only re-reads the table: a new task appears after a
    timer tick, and no modal is popped even with QMessageBox rigged to
    explode."""
    from PySide6.QtWidgets import QMessageBox

    from recast.desktop.queue_panel import (
        AUTO_REFRESH_INTERVAL_MS,
        QueuePanelDialog,
    )

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="待执行")
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("timer refresh must not pop a modal"))),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("timer refresh must not pop a modal"))),
    )
    dlg = QueuePanelDialog(svc)
    assert AUTO_REFRESH_INTERVAL_MS == 2000
    assert dlg.auto_refresh_checkbox.isChecked() is True
    assert dlg._auto_refresh_timer.interval() == 2000
    assert dlg._auto_refresh_timer.isActive() is True
    assert dlg.table.rowCount() == 1

    _task(svc, drama["id"], episode="第2集", status="处理中",
          progress="人物替换中", progress_pct=10)
    dlg._auto_refresh_timer.timeout.emit()
    assert dlg.table.rowCount() == 2
    assert _row_for(dlg, "第2集") >= 0
    dlg.close()


def test_auto_refresh_checkbox_stops_and_restarts_timer(svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    dlg = QueuePanelDialog(svc)
    assert dlg._auto_refresh_timer.isActive() is True

    dlg.auto_refresh_checkbox.setChecked(False)
    assert dlg._auto_refresh_timer.isActive() is False
    # Manual refresh still works with auto-refresh off.
    assert dlg.table.rowCount() == 0

    dlg.auto_refresh_checkbox.setChecked(True)
    assert dlg._auto_refresh_timer.isActive() is True
    dlg.close()


def test_timer_stops_when_dialog_closes(svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    dlg = QueuePanelDialog(svc)
    assert dlg._auto_refresh_timer.isActive() is True
    dlg.close()
    assert dlg._auto_refresh_timer.isActive() is False


def test_empty_text_follows_filter_state(svc):
    """Boundary: only done tasks + filter on -> 'no in-progress tasks'
    (no contradictory 'see history' pointer); filter off + truly empty
    store -> the old hint that points at task history."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="已完成", progress="已完成")
    dlg = QueuePanelDialog(svc)

    # Filter on (default): the done row is hidden, the panel is empty,
    # and the hint must say there is nothing in progress — not send the
    # user to 「任务历史」 while the filter itself hides history.
    assert dlg.table.rowCount() == 0
    assert dlg.empty_label.isHidden() is False
    assert "没有进行中的任务" in dlg.empty_label.text()
    assert "取消勾选" in dlg.empty_label.text()
    assert "任务历史" not in dlg.empty_label.text()

    # Filter off: the done row appears and the hint disappears.
    dlg.only_in_progress_checkbox.setChecked(False)
    assert dlg.table.rowCount() == 1
    assert dlg.empty_label.isHidden() is True
    dlg.close()

    # Truly empty store, filter off: the old hint (with the history
    # pointer) is the correct one.
    svc.store.delete("tasks", svc.store.list_rows("tasks")[0]["id"])
    dlg2 = QueuePanelDialog(svc)
    dlg2.only_in_progress_checkbox.setChecked(False)
    assert dlg2.table.rowCount() == 0
    assert "队列为空" in dlg2.empty_label.text()
    assert "任务历史" in dlg2.empty_label.text()
    dlg2.close()
