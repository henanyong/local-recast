"""P4/N38 tests for the queue panel: refresh must preserve the selected
row (by task id) and the scroll position, and must only rewrite cells
whose content changed — progress bars and retry buttons keep their
object identity when their row stays put.

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


def _task_id_for(dlg, episode: str) -> str:
    from PySide6.QtCore import Qt

    item = dlg.table.item(_row_for(dlg, episode), 0)
    assert item is not None
    return item.data(Qt.ItemDataRole.UserRole)


def _selected_task_id(dlg):
    from PySide6.QtCore import Qt

    row = dlg.table.currentRow()
    if row < 0:
        return None
    item = dlg.table.item(row, 0)
    return item.data(Qt.ItemDataRole.UserRole) if item is not None else None


def _display_ids(dlg) -> list[str]:
    return [row["id"] for row in dlg._rows]


def _show_scrollable(dlg):
    """Show the dialog small enough that many rows need scrolling, and
    pump the event loop so the scrollbar range is real."""
    from PySide6.QtWidgets import QApplication

    dlg.resize(920, 300)
    dlg.show()
    QApplication.processEvents()
    assert dlg.table.verticalScrollBar().maximum() > 0, (
        "test setup: table must be scrollable (add more rows or shrink "
        "the dialog) or the scroll assertions below are vacuous"
    )


def _make_many(svc, drama_id: str, n: int, *, status: str = "待执行") -> list[dict]:
    """``n`` tasks with staggered timestamps so display order is
    deterministic (``list_history`` sorts by ``updated_at`` DESC)."""
    tasks = []
    for i in range(1, n + 1):
        stamp = f"2026-11-12 10:{i:02d}:00"
        tasks.append(_task(
            svc, drama_id, episode=f"第{i}集", status=status,
            created_at=stamp, updated_at=stamp,
        ))
    return tasks


# ----- A. selection follows the task id across refresh -----

def test_refresh_keeps_selection_on_same_task_even_when_row_moves(svc):
    """Positive: the selected task's progress ticks (which bumps its
    ``updated_at`` and moves it to the top) — selection must follow the
    task id, not the old row number, and the new step text must show."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="待执行",
          created_at="2026-11-12 10:01:00", updated_at="2026-11-12 10:01:00")
    running = _task(svc, drama["id"], episode="第2集", status="处理中",
                    progress="人物替换中", progress_pct=30,
                    created_at="2026-11-12 10:02:00",
                    updated_at="2026-11-12 10:02:00")
    _task(svc, drama["id"], episode="第3集", status="待执行",
          created_at="2026-11-12 10:03:00", updated_at="2026-11-12 10:03:00")

    dlg = QueuePanelDialog(svc)
    dlg.table.selectRow(_row_for(dlg, "第2集"))
    assert _selected_task_id(dlg) == running["id"]

    svc.store.update("tasks", running["id"], progress="字幕烧录中", progress_pct=60)
    dlg.refresh()

    assert dlg.table.rowCount() == 3
    assert _selected_task_id(dlg) == running["id"]
    assert _row_for(dlg, "第2集") == 0  # moved to the top, selection followed
    assert dlg.table.item(0, 2).text() == "字幕烧录中"
    dlg.close()


def test_refresh_with_no_changes_keeps_selection_row_and_items(svc):
    """Boundary: a refresh with no data change at all must not disturb
    the selection, the row index, or any cell item object."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="待执行")
    _task(svc, drama["id"], episode="第2集", status="处理中",
          progress="人物替换中", progress_pct=30)
    _task(svc, drama["id"], episode="第3集", status="待执行")

    dlg = QueuePanelDialog(svc)
    row = _row_for(dlg, "第3集")
    dlg.table.selectRow(row)
    items = [dlg.table.item(row, c) for c in (0, 1, 2, 4)]
    assert all(item is not None for item in items)

    dlg.refresh()

    assert _selected_task_id(dlg) == _task_id_for(dlg, "第3集")
    assert _row_for(dlg, "第3集") == row
    assert [dlg.table.item(row, c) for c in (0, 1, 2, 4)] == items
    dlg.close()


# ----- B. scroll position survives refresh -----

def test_refresh_keeps_scroll_position_when_nothing_changed(svc):
    """Positive: 40 rows, scrolled halfway, refresh with no data change
    (in-place path) — the vertical scrollbar value must not move."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _make_many(svc, drama["id"], 40)

    dlg = QueuePanelDialog(svc)
    _show_scrollable(dlg)
    vbar = dlg.table.verticalScrollBar()
    vbar.setValue(vbar.maximum() // 2)
    assert vbar.value() > 0

    dlg.refresh()

    assert dlg.table.verticalScrollBar().value() == vbar.maximum() // 2
    assert dlg.table.selectionModel().selectedRows() == []
    dlg.close()


def test_refresh_restores_scroll_when_new_row_arrives(svc):
    """Boundary: a new task arrives at the top (rebuild path) — the
    scrollbar must be restored to the saved value, and a refresh with no
    prior selection must not invent one."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _make_many(svc, drama["id"], 40)

    dlg = QueuePanelDialog(svc)
    _show_scrollable(dlg)
    vbar = dlg.table.verticalScrollBar()
    saved = vbar.maximum() // 2
    vbar.setValue(saved)
    assert vbar.value() == saved

    _task(svc, drama["id"], episode="第41集", status="处理中",
          progress="人物替换中", progress_pct=10)
    dlg.refresh()

    assert dlg.table.rowCount() == 41
    assert _row_for(dlg, "第41集") == 0
    assert dlg.table.verticalScrollBar().value() == saved
    assert dlg.table.selectionModel().selectedRows() == []
    dlg.close()


# ----- C. selected row disappears (task completes under the filter) -----

def test_selected_row_completed_falls_to_next_row_and_scroll_holds(svc):
    """Positive: the selected task finishes, so the [仅进行中] filter
    drops its row — selection must fall to the nearest same-kind row
    (the one that slid into its place) and the scroll position must
    not jump."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _make_many(svc, drama["id"], 40)

    dlg = QueuePanelDialog(svc)
    _show_scrollable(dlg)
    ids = _display_ids(dlg)
    k = 20
    doomed, neighbour = ids[k], ids[k + 1]
    dlg.table.selectRow(k)
    vbar = dlg.table.verticalScrollBar()
    saved = vbar.maximum() // 2
    vbar.setValue(saved)
    assert vbar.value() == saved

    svc.store.update("tasks", doomed, status="已完成", progress="已完成")
    dlg.refresh()

    assert dlg.table.rowCount() == 39
    assert _selected_task_id(dlg) == neighbour
    assert dlg.table.verticalScrollBar().value() == saved
    dlg.close()


def test_last_selected_row_completed_empties_table_and_clears_selection(svc):
    """Boundary: the only selected task finishes and the table empties —
    selection is cleared (not left pointing at a dead row), no crash,
    scrollbar back at 0."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    only = _task(svc, drama["id"], episode="第1集", status="处理中",
                 progress="人物替换中", progress_pct=50)

    dlg = QueuePanelDialog(svc)
    dlg.table.selectRow(0)
    assert _selected_task_id(dlg) == only["id"]

    svc.store.update("tasks", only["id"], status="已完成", progress="已完成")
    dlg.refresh()

    assert dlg.table.rowCount() == 0
    assert dlg.count_label.text() == "共 0 条"
    assert dlg.table.selectionModel().selectedRows() == []
    assert dlg.table.currentRow() == -1
    assert dlg.table.verticalScrollBar().value() == 0
    dlg.close()


# ----- D. only changed cells are rewritten (widget identity) -----

def test_unchanged_refresh_reuses_items_bar_and_retry_button(svc):
    """Positive: with no data change, every item, the progress bar and
    the retry button must be the very same objects after refresh."""
    from PySide6.QtWidgets import QProgressBar, QPushButton

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="处理中",
          progress="人物替换中", progress_pct=45)
    _task(svc, drama["id"], episode="第2集", status="失败",
          progress="人物替换失败", error="显存不足")

    dlg = QueuePanelDialog(svc)
    running = _row_for(dlg, "第1集")
    failed = _row_for(dlg, "第2集")
    bar = dlg.table.cellWidget(running, 3)
    retry = dlg.table.cellWidget(failed, 5)
    assert isinstance(bar, QProgressBar)
    assert isinstance(retry, QPushButton)
    items = {
        (r, c): dlg.table.item(r, c)
        for r in (running, failed) for c in (0, 1, 2, 4)
    }

    dlg.refresh()

    assert dlg.table.cellWidget(_row_for(dlg, "第1集"), 3) is bar
    assert dlg.table.cellWidget(_row_for(dlg, "第2集"), 5) is retry
    for (r, c), item in items.items():
        assert dlg.table.item(r, c) is item
    dlg.close()


def test_changed_progress_updates_bar_value_but_keeps_widget_and_items(svc):
    """Positive: a progress tick rewrites only the value/text — the bar
    and the name item stay the same objects."""
    from PySide6.QtWidgets import QProgressBar

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    running = _task(svc, drama["id"], episode="第1集", status="处理中",
                    progress="人物替换中", progress_pct=30,
                    created_at="2026-11-12 10:02:00",
                    updated_at="2026-11-12 10:02:00")
    _task(svc, drama["id"], episode="第2集", status="待执行",
          created_at="2026-11-12 10:01:00", updated_at="2026-11-12 10:01:00")

    dlg = QueuePanelDialog(svc)
    row = _row_for(dlg, "第1集")
    bar = dlg.table.cellWidget(row, 3)
    name_item = dlg.table.item(row, 0)
    assert isinstance(bar, QProgressBar)
    assert bar.value() == 30

    # Keep the row on top (explicit updated_at) so the tick does not
    # reorder the table: this is the pure in-place path.
    svc.store.update("tasks", running["id"], progress="字幕烧录中",
                     progress_pct=70, updated_at="2026-11-12 10:02:00")
    dlg.refresh()

    assert _row_for(dlg, "第1集") == row
    assert dlg.table.cellWidget(row, 3) is bar
    assert bar.value() == 70
    assert dlg.table.item(row, 2).text() == "字幕烧录中"
    assert dlg.table.item(row, 0) is name_item
    dlg.close()


def test_failed_row_keeps_retry_button_across_refresh(svc):
    """Positive: a failed row whose error text changes keeps its retry
    button object — the handler binds task_id, not a row number, so the
    kept button still retries the same task."""
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    failed = _task(svc, drama["id"], episode="第2集", status="失败",
                   progress="人物替换失败", error="显存不足",
                   created_at="2026-11-12 10:02:00",
                   updated_at="2026-11-12 10:02:00")
    _task(svc, drama["id"], episode="第1集", status="待执行",
          created_at="2026-11-12 10:01:00", updated_at="2026-11-12 10:01:00")

    dlg = QueuePanelDialog(svc)
    row = _row_for(dlg, "第2集")
    retry = dlg.table.cellWidget(row, 5)
    assert isinstance(retry, QPushButton)
    assert retry.property("task_id") == failed["id"]

    svc.store.update("tasks", failed["id"], error="换脸模型加载失败",
                     updated_at="2026-11-12 10:02:00")
    dlg.refresh()

    assert dlg.table.cellWidget(_row_for(dlg, "第2集"), 5) is retry
    assert retry.property("task_id") == failed["id"]
    assert dlg.table.item(_row_for(dlg, "第2集"), 4).text() == "换脸模型加载失败"
    dlg.close()


def test_row_becoming_failed_gets_button_and_row_recovering_loses_it(svc):
    """Boundary: pending -> failed creates a retry button bound to the
    task id; failed -> pending removes it again and restores "—"."""
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    row_task = _task(svc, drama["id"], episode="第1集", status="待执行")

    dlg = QueuePanelDialog(svc)
    row = _row_for(dlg, "第1集")
    assert dlg.table.cellWidget(row, 5) is None
    assert dlg.table.item(row, 5).text() == "—"

    svc.store.update("tasks", row_task["id"], status="失败",
                     progress="人物替换失败", error="显存不足",
                     updated_at="2026-11-13 00:00:00")
    dlg.refresh()
    row = _row_for(dlg, "第1集")
    retry = dlg.table.cellWidget(row, 5)
    assert isinstance(retry, QPushButton)
    assert retry.text() == "从失败步续跑"
    assert retry.property("task_id") == row_task["id"]
    assert dlg.table.item(row, 4).text() == "显存不足"

    svc.store.update("tasks", row_task["id"], status="待执行",
                     progress="等待执行", error="",
                     updated_at="2026-11-13 00:00:00")
    dlg.refresh()
    row = _row_for(dlg, "第1集")
    assert dlg.table.cellWidget(row, 5) is None
    assert dlg.table.item(row, 5).text() == "—"
    assert dlg.table.item(row, 4).text() == ""
    dlg.close()


def test_progress_bar_appears_and_disappears_without_rebuilding_row(svc):
    """Boundary: 0% -> 50% creates a bar, 50% -> 0% removes it and
    restores "—", while the name item object stays the same."""
    from PySide6.QtWidgets import QProgressBar

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    running = _task(svc, drama["id"], episode="第1集", status="处理中",
                    progress="人物替换中", progress_pct=0)

    dlg = QueuePanelDialog(svc)
    row = _row_for(dlg, "第1集")
    assert dlg.table.cellWidget(row, 3) is None
    assert dlg.table.item(row, 3).text() == "—"
    name_item = dlg.table.item(row, 0)

    svc.store.update("tasks", running["id"], progress_pct=50,
                     updated_at="2026-11-13 00:00:00")
    dlg.refresh()
    row = _row_for(dlg, "第1集")
    bar = dlg.table.cellWidget(row, 3)
    assert isinstance(bar, QProgressBar)
    assert bar.value() == 50
    assert dlg.table.item(row, 0) is name_item

    svc.store.update("tasks", running["id"], progress_pct=0,
                     updated_at="2026-11-13 00:00:00")
    dlg.refresh()
    row = _row_for(dlg, "第1集")
    assert dlg.table.cellWidget(row, 3) is None
    assert dlg.table.item(row, 3).text() == "—"
    assert dlg.table.item(row, 0) is name_item
    dlg.close()


def test_reordered_rows_update_in_place_and_selection_follows_task(svc):
    """Failure case for the naive 'same order' rule: a background row's
    progress tick bumps its ``updated_at`` and jumps it to the top.
    Row count is unchanged, so there must be no rebuild — the selection
    follows the task id and the per-position item objects survive."""
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="待执行",
          created_at="2026-11-12 10:01:00", updated_at="2026-11-12 10:01:00")
    middle = _task(svc, drama["id"], episode="第2集", status="处理中",
                   progress="人物替换中", progress_pct=20,
                   created_at="2026-11-12 10:02:00",
                   updated_at="2026-11-12 10:02:00")
    _task(svc, drama["id"], episode="第3集", status="待执行",
          created_at="2026-11-12 10:03:00", updated_at="2026-11-12 10:03:00")

    dlg = QueuePanelDialog(svc)
    # Display order is newest-first: 第3集, 第2集, 第1集.
    assert _row_for(dlg, "第2集") == 1
    dlg.table.selectRow(1)
    top_name_item = dlg.table.item(0, 0)

    svc.store.update("tasks", middle["id"], progress="字幕烧录中",
                     progress_pct=55)
    dlg.refresh()

    assert dlg.table.rowCount() == 3
    assert _row_for(dlg, "第2集") == 0  # ticked row jumped to the top
    assert _selected_task_id(dlg) == middle["id"]
    assert dlg.table.item(0, 2).text() == "字幕烧录中"
    # Per-position items were reused, not rebuilt: the old top row's
    # name item object is still the item at row 0 (now showing 第2集).
    assert dlg.table.item(0, 0) is top_name_item
    dlg.close()
