"""Queue panel dialog: live view of the batch queue (G3 / PM 9.2).

One row per episode — drama name, status, current step, progress,
failure reason, and a retry button on failed rows. This is the
"in-progress" counterpart of the task history dialog
(``recast/desktop/task_history_dialog.py``): the history panel looks
back at past tasks with config snapshots and a full re-run; this panel
shows where every task stands right now so a running batch does not
have to be followed through the log plus an after-the-fact failure card.

Read-only display plus one existing retry entry point; it never runs
tasks itself and never touches scheduling logic:

* Rows are read through ``recast.core.task_history.list_history`` —
  the same read helper the history dialog uses. No second way of
  reading the ``tasks`` table is introduced here.
* The retry button on a failed row is "从失败步续跑": it calls
  ``service.start([drama_id], stages)`` — the same entry point the
  main window's failure card uses. The same task resumes and already
  finished steps are skipped by the runner's checkpoints; no new task
  is created. A full re-run that creates a new task is a different
  operation ("一键复跑", ``service.rerun_task``) and lives in the
  history panel; the retry tooltip here says so, so the two are not
  confused again (P1/N2 lesson).
* Progress is honest: the progress bar is only shown when the task row
  actually carries a ``progress_pct`` greater than 0. A row with no
  stored percentage shows "—" in the progress column — never a
  fabricated 0% or an invented bar. The current-step text (the row's
  ``progress`` field) is always shown, with or without a percentage.

No modal popups from any timer or refresh in this module (P1 lesson:
never exec a modal from a paint / timer refresh). The one exception is
the auth gate in ``_retry_failed`` (N32): the user clicked [从失败步续跑],
so the shared ``auth_gate_ok`` dialog may pop — a click handler, never
the auto-refresh timer. Refresh happens when the dialog opens, when the
user presses [刷新], when a caller calls ``refresh()``, and — since P3/N33
— every 2 s from a QTimer while [自动刷新] is checked. The timer only
re-reads the table (``refresh()``); it never pops a dialog, never starts
a task, and never touches scheduling. The dialog is modal (``dlg.exec()``)
but the timer lives in the GUI thread, so it also fires inside the modal
event loop. Unchecking [自动刷新] stops the timer; closing the dialog
stops it too.

Next-round wiring (main window entry button)::

    from recast.desktop.queue_panel import QueuePanelDialog

    dlg = QueuePanelDialog(self.service, parent=self)
    dlg.exec()
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from recast.core.constants import ST_FAILED, STAGE_GROUP_IDS
from recast.core.entry_shared import ENTRY_STAGES
from recast.core.service import RecastService
from recast.core.task_history import list_history
from recast.desktop.auth_gate_dialog import auth_gate_ok
from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap

# 入口–阶段映射的唯一来源在 recast.core.entry_shared（ENTRY_STAGES，
# N35 去重，P3）：这里不再留拷贝，import 跑前 6 个阶段，script 跑后 5 个。

COLS = ["剧目 / 集", "状态", "当前步骤", "进度", "失败原因", "操作"]

#: Text shown when the queue has no rows at all.
EMPTY_TEXT = (
    "队列为空：还没有任务。批量执行开始后，每集的状态、当前步骤和进度会显示在这里；"
    "已完成的任务请到「任务历史」查看。"
)

#: Text shown when [仅进行中] is checked and no unfinished task exists.
#: Deliberately does NOT point at 「任务历史」 — with the filter on, the
#: panel is an in-progress view, so "go look at history" would contradict
#: it (P3/N33). It tells the user how to see everything instead.
EMPTY_IN_PROGRESS_TEXT = (
    "没有进行中的任务。取消勾选「仅进行中」可查看全部任务（含已完成）。"
)

#: Auto-refresh interval (P3/N33): 2 s is frequent enough to follow a
#: running batch without hammering the tasks table on every frame.
AUTO_REFRESH_INTERVAL_MS = 2000

#: Canonical ``status_kind`` values (from ``recast.core.task_history``)
#: that count as "in progress" for the [仅进行中] filter. Everything
#: except "done" is unfinished — including "failed" (a failed row still
#: needs its [从失败步续跑] button) and "unknown" (a row whose status the
#: history helper could not classify must never be hidden by this panel).
_IN_PROGRESS_KINDS: frozenset[str] = frozenset(
    {"need_assets", "pending", "running", "review", "failed", "unknown"}
)


class QueuePanelDialog(QDialog):
    """[队列面板]：批量跑进行中的每集状态 / 当前步骤 / 进度 / 失败原因。"""

    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self._rows: list[dict[str, Any]] = []
        #: Last retry attempt from a failed row (``{"task_id",
        #: "drama_id", "started", "reason"}``); None until a retry is tried.
        self.last_retry: dict[str, Any] | None = None
        self.setWindowTitle("队列面板")
        # P5/B3: this IS a QDialog (not an embedded panel), so it gets
        # the dialog treatment — top row + table + status scroll, the
        # [关闭] bar is pinned at the bottom, dialog clamped to the
        # screen (was resize(920, 520)). The table keeps its own
        # scrollbar for long queues; the 2 s auto-refresh timer is
        # untouched.
        cap_dialog_to_screen(self, 920, 520)

        content = QWidget()
        layout = QVBoxLayout(content)

        top = QHBoxLayout()
        self.count_label = QLabel()
        top.addWidget(self.count_label)
        self.scheduler_label = QLabel()
        top.addWidget(self.scheduler_label)
        top.addStretch(1)
        # P3/N33: [仅进行中] defaults to checked — without it, finished
        # rows bury the running ones after a few batches.
        self.only_in_progress_checkbox = QCheckBox("仅进行中")
        self.only_in_progress_checkbox.setObjectName("queueOnlyInProgress")
        self.only_in_progress_checkbox.setChecked(True)
        self.only_in_progress_checkbox.setToolTip(
            "只列未完成的任务（待执行 / 处理中 / 待人工检查 / 失败等）；"
            "取消勾选可查看全部任务（含已完成）。"
        )
        self.only_in_progress_checkbox.toggled.connect(self.refresh)
        top.addWidget(self.only_in_progress_checkbox)
        # P3/N33: auto-refresh switch for the timer below.
        self.auto_refresh_checkbox = QCheckBox("自动刷新（2 秒）")
        self.auto_refresh_checkbox.setObjectName("queueAutoRefresh")
        self.auto_refresh_checkbox.setChecked(True)
        self.auto_refresh_checkbox.setToolTip(
            "每 2 秒只刷新一次表格内容，不会弹出任何窗口；"
            "取消勾选可停止自动刷新，仍可点 [刷新] 手动刷新。"
        )
        self.auto_refresh_checkbox.toggled.connect(self._set_auto_refresh)
        top.addWidget(self.auto_refresh_checkbox)
        refresh = QPushButton("刷新")
        refresh.setObjectName("queueRefresh")
        refresh.clicked.connect(self.refresh)
        top.addWidget(refresh)
        layout.addLayout(top)

        self.empty_label = QLabel(EMPTY_TEXT)
        self.empty_label.setWordWrap(True)
        self.empty_label.setObjectName("queueEmpty")
        layout.addWidget(self.empty_label)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(3, 130)
        self.table.setColumnWidth(4, 200)
        self.table.setColumnWidth(5, 130)
        layout.addWidget(self.table, 1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setObjectName("queueStatus")
        layout.addWidget(self.status)

        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        row.addWidget(self.close_btn)
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))

        self.refresh()

        # P3/N33: table-only auto-refresh timer. It lives in the GUI
        # thread (a child of this dialog), so it also fires inside the
        # modal ``exec()`` event loop. Its slot is ``refresh()`` — a
        # read-only table update that never pops a modal (P1 lesson).
        # Since P4/N38 the refresh preserves the selection (by task id)
        # and the scroll position: unchanged rows are updated in place,
        # and a rebuild restores both afterwards.
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(AUTO_REFRESH_INTERVAL_MS)
        self._auto_refresh_timer.timeout.connect(self.refresh)
        if self.auto_refresh_checkbox.isChecked():
            self._auto_refresh_timer.start()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the auto-refresh timer when the dialog closes."""
        timer = getattr(self, "_auto_refresh_timer", None)
        if timer is not None:
            timer.stop()
        super().closeEvent(event)

    def _set_auto_refresh(self, enabled: bool) -> None:
        """Start/stop the table-only auto-refresh timer."""
        if enabled:
            self._auto_refresh_timer.start()
        else:
            self._auto_refresh_timer.stop()

    # ----- data -----

    def _snapshot_view_state(self) -> dict[str, Any]:
        """Capture selection + scroll position before a table refresh.

        The selected row is identified by task id — never by row number,
        because rows move when ``list_history`` re-sorts by ``updated_at``
        — so it can be found again after the rebuild even if its index
        moved. (P4/N38)
        """
        state: dict[str, Any] = {
            "task_id": None,
            "status_kind": None,
            "row_index": -1,
            "v_value": self.table.verticalScrollBar().value(),
            "h_value": self.table.horizontalScrollBar().value(),
        }
        row_index = self.table.currentRow()
        selected = self.table.selectionModel().selectedRows()
        if selected:
            row_index = selected[0].row()
        if 0 <= row_index < self.table.rowCount():
            state["row_index"] = row_index
            item = self.table.item(row_index, 0)
            if item is not None:
                state["task_id"] = item.data(Qt.ItemDataRole.UserRole)
            if 0 <= row_index < len(self._rows):
                state["status_kind"] = self._rows[row_index].get("status_kind")
        return state

    def _restore_view_state(
        self, state: dict[str, Any], rows: list[dict[str, Any]]
    ) -> None:
        """Restore selection + scroll position after a table rebuild.

        Same task id -> re-select that row wherever it moved. If the
        selected task is gone (e.g. it finished and the [仅进行中] filter
        hid it), select the nearest row of the same status kind, else the
        row now sitting at the old index; with no rows left the selection
        is simply cleared. Scroll position is restored *after* selecting,
        because selecting a row may itself scroll the view — the saved
        reading position wins. (P4/N38)
        """
        target = -1
        task_id = state.get("task_id")
        if task_id is not None:
            for i, row in enumerate(rows):
                if row["id"] == task_id:
                    target = i
                    break
        if target < 0 and rows and state.get("row_index", -1) >= 0:
            old_index = state["row_index"]
            old_kind = state.get("status_kind")
            same_kind = [
                i for i, row in enumerate(rows)
                if row.get("status_kind") == old_kind
            ]
            if same_kind:
                target = min(same_kind, key=lambda i: (abs(i - old_index), i))
            else:
                target = max(0, min(old_index, len(rows) - 1))
        if target >= 0:
            self.table.selectRow(target)
        else:
            self.table.clearSelection()
        vbar = self.table.verticalScrollBar()
        vbar.setValue(min(state.get("v_value", 0), vbar.maximum()))
        hbar = self.table.horizontalScrollBar()
        hbar.setValue(min(state.get("h_value", 0), hbar.maximum()))

    def _rebuild_rows(self, rows: list[dict[str, Any]]) -> None:
        """Full rebuild, used only when the row set changed.

        Drops all rows first so no stale cell widget survives at a
        recycled index, then fills fresh items/widgets. (P4/N38)
        """
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            title = row["episode_name"] or row["video_path"] or row["id"]
            if row["drama_name"]:
                title = f"{row['drama_name']} — {title}"
            name_item = QTableWidgetItem(title)
            name_item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.table.setItem(i, 0, name_item)
            status_item = QTableWidgetItem(row["status_label"])
            status_item.setForeground(QColor(row["status_colour"]))
            self.table.setItem(i, 1, status_item)
            self.table.setItem(i, 2, QTableWidgetItem(row["progress"] or "—"))
            self._set_progress_cell(i, row)
            self.table.setItem(i, 4, QTableWidgetItem(row["error"] or ""))
            self._set_action_cell(i, row)

    def _update_rows_in_place(
        self, old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]
    ) -> None:
        """Update only the cells whose content changed (P4/N38).

        Only called when the row set is unchanged (same task ids; their
        order may have moved, since ``list_history`` sorts by
        ``updated_at``). Cells are rewritten by position from the new
        rows, so row count, item/widget objects (for rows that did not
        move), selection and scroll position are never disturbed — a
        running task's progress bar and a failed row's [从失败步续跑]
        button keep their identity when the row stays put.
        """
        for i, (_old, new) in enumerate(zip(old_rows, new_rows)):
            title = new["episode_name"] or new["video_path"] or new["id"]
            if new["drama_name"]:
                title = f"{new['drama_name']} — {title}"
            name_item = self.table.item(i, 0)
            if name_item is None:
                name_item = QTableWidgetItem(title)
                name_item.setData(Qt.ItemDataRole.UserRole, new["id"])
                self.table.setItem(i, 0, name_item)
            else:
                if name_item.text() != title:
                    name_item.setText(title)
                if name_item.data(Qt.ItemDataRole.UserRole) != new["id"]:
                    name_item.setData(Qt.ItemDataRole.UserRole, new["id"])
            status_item = self.table.item(i, 1)
            if status_item is None:
                status_item = QTableWidgetItem(new["status_label"])
                status_item.setForeground(QColor(new["status_colour"]))
                self.table.setItem(i, 1, status_item)
            else:
                if status_item.text() != new["status_label"]:
                    status_item.setText(new["status_label"])
                if (
                    status_item.foreground().color().name()
                    != QColor(new["status_colour"]).name()
                ):
                    status_item.setForeground(QColor(new["status_colour"]))
            step_text = new["progress"] or "—"
            step_item = self.table.item(i, 2)
            if step_item is None:
                self.table.setItem(i, 2, QTableWidgetItem(step_text))
            elif step_item.text() != step_text:
                step_item.setText(step_text)
            self._set_progress_cell(i, new)
            error_text = new["error"] or ""
            error_item = self.table.item(i, 4)
            if error_item is None:
                self.table.setItem(i, 4, QTableWidgetItem(error_text))
            elif error_item.text() != error_text:
                error_item.setText(error_text)
            self._set_action_cell(i, new)

    def refresh(self) -> None:
        """Re-read the tasks table and refresh the rows.

        Read-only: never writes to the store, never pops anything.
        When [仅进行中] is checked (the default), only unfinished tasks
        are shown — the filter uses the ``status_kind`` that
        ``list_history`` already computed, no second status judgement is
        invented here. The empty hint follows the filter state (P3/N33).

        Selection + scroll are preserved (P4/N38): when the row set is
        unchanged (same task ids, even if their order moved — a progress
        tick bumps ``updated_at`` and re-sorts the list) only cells whose
        content changed are rewritten, so selection, scroll position and
        widget identity survive; when rows were added or removed the
        table is rebuilt and the selection (by task id) and scroll
        position are restored afterwards.
        """
        state = self._snapshot_view_state()
        all_rows = list_history(self.service.store, limit=200)
        if self.only_in_progress_checkbox.isChecked():
            new_rows = [
                row for row in all_rows
                if row.get("status_kind") in _IN_PROGRESS_KINDS
            ]
        else:
            new_rows = all_rows
        self.count_label.setText(f"共 {len(new_rows)} 条")
        self._refresh_scheduler_label()
        if new_rows:
            self.empty_label.setVisible(False)
        else:
            if self.only_in_progress_checkbox.isChecked():
                self.empty_label.setText(EMPTY_IN_PROGRESS_TEXT)
            else:
                self.empty_label.setText(EMPTY_TEXT)
            self.empty_label.setVisible(True)
        old_rows = self._rows
        old_ids = [r["id"] for r in old_rows]
        new_ids = [r["id"] for r in new_rows]
        if len(new_ids) == len(old_ids) and set(new_ids) == set(old_ids):
            # Same task set (order may have changed: list_history sorts
            # by updated_at, so a task whose progress just ticked jumps
            # to the top). Cells are rewritten in place by position and
            # the selection is re-found by task id; no rebuild.
            self._update_rows_in_place(old_rows, new_rows)
            self._restore_view_state(state, new_rows)
        else:
            self.table.setUpdatesEnabled(False)
            try:
                self._rebuild_rows(new_rows)
            finally:
                self.table.setUpdatesEnabled(True)
            self._restore_view_state(state, new_rows)
        self._rows = new_rows

    def _refresh_scheduler_label(self) -> None:
        """One-line scheduler summary (running / queued), best effort.

        A service without a ``snapshot()`` (e.g. a test double) simply
        leaves the label blank instead of breaking the panel.
        """
        snapshot = getattr(self.service, "snapshot", None)
        if snapshot is None:
            self.scheduler_label.setText("")
            return
        try:
            sched = (snapshot() or {}).get("scheduler") or {}
        except Exception:
            self.scheduler_label.setText("")
            return
        if sched.get("running"):
            current = sched.get("current_progress") or ""
            text = f"调度器：进行中（排队 {sched.get('queued', 0)}）"
            if current:
                text += f"　{current}"
            self.scheduler_label.setText(text)
        elif sched.get("queued"):
            self.scheduler_label.setText(f"调度器：排队 {sched.get('queued')}")
        else:
            self.scheduler_label.setText("调度器：空闲")

    def _set_progress_cell(self, row_index: int, row: dict[str, Any]) -> None:
        """Progress column: a real bar only when a percentage is stored.

        ``progress_pct`` of 0 / missing is indistinguishable from "no
        progress information was ever recorded", so it shows "—" rather
        than a fabricated 0% bar.

        Reuse-aware (P4/N38): an existing bar for the same row keeps its
        identity and only its value is updated; a bar is only created or
        removed when the row gains/loses a stored percentage.
        """
        try:
            pct = float(row.get("progress_pct") or 0)
        except (TypeError, ValueError):
            pct = 0
        if pct > 0:
            value = max(0, min(100, int(pct)))
            bar = self.table.cellWidget(row_index, 3)
            if (
                isinstance(bar, QProgressBar)
                and bar.property("task_id") == row["id"]
            ):
                if bar.value() != value:
                    bar.setValue(value)
                return
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(value)
            bar.setFormat("%p%")
            # Task binding, like the retry button's: a bar reused after
            # a reorder must belong to the task now shown in this row.
            bar.setProperty("task_id", row["id"])
            self.table.setCellWidget(row_index, 3, bar)
        else:
            if self.table.cellWidget(row_index, 3) is not None:
                self.table.setCellWidget(row_index, 3, None)
            item = self.table.item(row_index, 3)
            if item is None:
                self.table.setItem(row_index, 3, QTableWidgetItem("—"))
            elif item.text() != "—":
                item.setText("—")

    def _set_action_cell(self, row_index: int, row: dict[str, Any]) -> None:
        """Failed rows get [从失败步续跑]; every other row gets "—".

        Reuse-aware (P4/N38): a failed row that stays failed keeps its
        retry button object — the click handler binds ``task_id``, not a
        row number, so the kept button still retries the same task. A
        button is only created when the row becomes failed, and only
        removed when it stops being failed.
        """
        if row.get("status_kind") != "failed":
            if self.table.cellWidget(row_index, 5) is not None:
                self.table.setCellWidget(row_index, 5, None)
            item = self.table.item(row_index, 5)
            if item is None:
                self.table.setItem(row_index, 5, QTableWidgetItem("—"))
            elif item.text() != "—":
                item.setText("—")
            return
        existing = self.table.cellWidget(row_index, 5)
        if (
            isinstance(existing, QPushButton)
            and existing.property("task_id") == row["id"]
        ):
            return
        retry = QPushButton("从失败步续跑")
        retry.setObjectName("queueRetry")
        retry.setToolTip(
            "只重跑失败步及之后：已完成的步骤按检查点跳过，不从头重跑"
            "（同一任务续跑，不新建任务；新建任务全量重跑请到任务历史用「一键复跑」）。"
        )
        retry.setProperty("task_id", row["id"])
        retry.setProperty("drama_id", row.get("drama_id") or "")
        retry.clicked.connect(
            lambda _checked=False, task_id=row["id"]: self._retry_failed(task_id)
        )
        self.table.setCellWidget(row_index, 5, retry)

    # ----- actions -----

    def _go_to_library(self) -> None:
        """[去人物库补授权]：先关面板，再让主窗口打开人物库。

        面板是主窗口打开的模态对话框，人物库也由主窗口打开：先
        ``close()`` 关掉面板，再调父窗口（主窗口）的 ``_open_library``。
        面板不是主窗口打开的（parent 没有 ``_open_library``），就只关
        面板、不另起人物库——这种上下文里打不开人物库是已知边界。
        """
        self.close()
        opener = getattr(self.parent(), "_open_library", None)
        if callable(opener):
            opener()

    def _retry_failed(self, task_id: str) -> None:
        """Resume one failed task from its failed step (same task).

        Walks the existing ``service.start`` entry point — the same one
        the main window's failure card uses — with the stages that apply
        to the task's entry type. Checkpoints skip finished steps, so
        this is a resume, not a re-run. Before ``service.start`` it goes
        through the shared auth gate (N32) when replace will run; a
        failure to start is reported in the status line only, and the
        gate itself is the only modal this panel pops — only from this
        click handler, never from the auto-refresh timer.
        """
        row = next((r for r in self._rows if r["id"] == task_id), None)
        drama_id = (row or {}).get("drama_id") or ""
        if not drama_id:
            self.last_retry = {"task_id": task_id, "drama_id": "", "started": False,
                               "reason": "该任务没有所属剧目，无法续跑。"}
            self.status.setText(self.last_retry["reason"])
            return
        entry = (row or {}).get("entry_type") or ""
        if not entry:
            try:
                drama = self.service.store.get("dramas", drama_id) or {}
                entry = drama.get("entry_type") or ""
            except Exception:
                entry = ""
        stages = None
        if entry in ENTRY_STAGES:
            stages = [sid for sid in STAGE_GROUP_IDS if sid in ENTRY_STAGES[entry]]
        # N32：重试也是替换入口，service.start 之前先走与主窗口同一
        # 个授权关口（共用 auth_gate_ok）。关口跟的是“替换”这个动作：
        # 剧本路重试没有 replace 阶段，不拦；入口类型不明时按默认全
        # 阶段（含 replace）算，同样过关口。零已授权时 [继续] 灰掉，
        # 被跳过的人物由关口写进持久的 auth_skips。
        if stages is None or "replace" in stages:
            if not auth_gate_ok(
                self, self.service, [drama_id],
                prompt=True, on_go_library=self._go_to_library,
            ):
                self.last_retry = {"task_id": task_id, "drama_id": drama_id,
                                   "started": False, "reason": "授权关口未通过"}
                self.status.setText(
                    "从失败步续跑已停止：授权关口未通过——有未授权人物，"
                    "请先去人物库补授权。"
                )
                return
        try:
            if stages:
                self.service.start([drama_id], stages)
            else:
                self.service.start([drama_id])
        except Exception as exc:
            self.last_retry = {"task_id": task_id, "drama_id": drama_id,
                               "started": False, "reason": str(exc)}
            self.status.setText(f"从失败步续跑失败：{exc}")
            return
        name = (row or {}).get("drama_name") or drama_id
        title = (row or {}).get("episode_name") or ""
        self.last_retry = {"task_id": task_id, "drama_id": drama_id,
                           "started": True, "reason": None}
        self.status.setText(
            f"「{name}{(' — ' + title) if title else ''}」已从失败步续跑："
            "只重跑失败步及之后（同一任务续跑，不新建任务；"
            "新建任务全量重跑请到任务历史用「一键复跑」）。"
        )
        self.refresh()
