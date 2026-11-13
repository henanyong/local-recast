"""Task history dialog: browse past tasks, inspect config snapshots, re-run.

Read-only over the ``tasks`` table except for two explicit user actions:

* [一键复跑] builds a re-run task dict from the selected history entry
  (same config, fresh id, status reset to 待执行), inserts it and starts
  it through ``service.rerun_task`` — the same scheduler entry point as
  the main window's 开始执行. The status line says plainly whether the
  re-run really started (queued) or was parked (e.g. replace-auth gate)
  and why.
* [查看跳过记录] opens a read-only viewer (``AuthSkipsDialog``) listing
  the auth-gate skips for the selected task (name + reason + time);
  with no records it says “本次无跳过记录” instead of showing blank.
* [删除历史记录] deletes only the history row (dependent shot / subtitle /
  checkpoint rows go away with the foreign-key cascade). The output file on
  disk is never touched, and the drama row is kept. Deletion asks first.

Next-round wiring (main window entry button)::

    from recast.desktop.task_history_dialog import TaskHistoryDialog

    dlg = TaskHistoryDialog(self.service, parent=self)
    dlg.exec()
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from recast.core.service import RecastService
from recast.core.task_history import (
    NO_AUTH_SKIPS_TEXT,
    build_config_snapshot,
    count_history,
    delete_history_entry,
    format_auth_skips,
    get_auth_skips_for_task,
    get_history_entry,
    list_history,
)
from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap

# Filter combo entries: (label shown, value passed to list_history).
FILTERS: list[tuple[str, str | None]] = [
    ("全部状态", None),
    ("已完成", "done"),
    ("失败", "failed"),
    ("处理中", "running"),
    ("待执行", "pending"),
    ("待补资料", "need_assets"),
    ("待人工检查", "review"),
]

COLS = ["任务", "状态", "当前步骤", "开始", "最近更新", "失败原因"]


class AuthSkipsDialog(QDialog):
    """[查看跳过记录]：只读列出一次授权关口跳过的人名、原因与时间。

    ``skips_text`` is the pre-formatted text from
    :func:`recast.core.task_history.format_auth_skips` (or
    ``NO_AUTH_SKIPS_TEXT`` when there is nothing to show). Pure viewer:
    it never writes to the store.
    """

    def __init__(self, task_title: str, skips_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"跳过记录 — {task_title}")
        # P5/B3: body scrolls, the [关闭] bar is pinned at the bottom,
        # dialog clamped to the screen (was resize(520, 300)).
        cap_dialog_to_screen(self, 520, 300)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(QLabel("授权关口跳过的人物（只读，不改库）："))
        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setPlainText(skips_text or NO_AUTH_SKIPS_TEXT)
        content_layout.addWidget(self.body, 1)
        buttons = QWidget()
        buttons_layout = QVBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        buttons_layout.addWidget(close)
        self.close_btn = close
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))


class TaskHistoryDialog(QDialog):
    """[任务历史]：回看跑过的任务、配置快照、复跑或删除历史记录。"""

    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self._rows: list[dict[str, Any]] = []
        #: Last re-run result from [一键复跑] (the inserted task row plus
        #: ``started`` / ``reason`` / ``stages``); None until clicked.
        self.last_rerun_task: dict[str, Any] | None = None
        #: Last [查看跳过记录] viewer opened (for tests / inspection).
        self.last_skips_dialog: AuthSkipsDialog | None = None
        self.setWindowTitle("任务历史")
        # P5/B3: filter + table + detail + status scroll, the
        # [一键复跑]/[删除历史记录]/[查看跳过记录]/[关闭] bar is pinned at
        # the bottom, dialog clamped to the screen (was resize(860, 560)).
        # The table keeps its own scrollbar for long histories.
        cap_dialog_to_screen(self, 860, 560)

        content = QWidget()
        layout = QVBoxLayout(content)
        top = QHBoxLayout()
        top.addWidget(QLabel("状态筛选"))
        self.filter_box = QComboBox()
        for label, _value in FILTERS:
            self.filter_box.addItem(label)
        self.filter_box.currentIndexChanged.connect(self._refresh)
        top.addWidget(self.filter_box)
        self.count_label = QLabel()
        top.addWidget(self.count_label)
        top.addStretch(1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh)
        top.addWidget(refresh)
        layout.addLayout(top)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 180)
        self.table.setColumnWidth(3, 150)
        self.table.setColumnWidth(4, 150)
        self.table.currentItemChanged.connect(self._on_select)
        layout.addWidget(self.table, 1)

        layout.addWidget(QLabel("详情（配置快照 + 失败原因全文）"))
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(160)
        layout.addWidget(self.detail)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.rerun_btn = QPushButton("一键复跑（新建任务）")
        self.rerun_btn.setToolTip("新建一个任务行，按同一配置全量重跑（与主窗口「从失败步续跑」不同：那是同一任务续跑，不新建任务）。")
        self.rerun_btn.clicked.connect(self._rerun_selected)
        row.addWidget(self.rerun_btn)
        self.delete_btn = QPushButton("删除历史记录")
        self.delete_btn.setToolTip("只删历史行，不删成片文件。")
        self.delete_btn.clicked.connect(self._delete_selected)
        row.addWidget(self.delete_btn)
        self.skips_btn = QPushButton("查看跳过记录")
        self.skips_btn.setToolTip("列出本次授权关口跳过的人名、原因与时间（只读，不改库）；无记录时显示“本次无跳过记录”。")
        self.skips_btn.clicked.connect(self._show_auth_skips)
        row.addWidget(self.skips_btn)
        row.addStretch(1)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        row.addWidget(self.close_btn)
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))

        self._refresh()

    # ----- data -----

    def _current_filter(self) -> str | None:
        return FILTERS[self.filter_box.currentIndex()][1]

    def _refresh(self) -> None:
        status = self._current_filter()
        self._rows = list_history(self.service.store, status=status, limit=200)
        total = count_history(self.service.store, status=status)
        self.count_label.setText(f"共 {total} 条")
        self.table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            title = row["episode_name"] or row["video_path"]
            if row["drama_name"]:
                title = f"{row['drama_name']} — {title}"
            name_item = QTableWidgetItem(title)
            name_item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.table.setItem(i, 0, name_item)
            status_item = QTableWidgetItem(row["status_label"])
            status_item.setForeground(QColor(row["status_colour"]))
            self.table.setItem(i, 1, status_item)
            self.table.setItem(i, 2, QTableWidgetItem(row["progress"]))
            self.table.setItem(i, 3, QTableWidgetItem(row["created_at"]))
            self.table.setItem(i, 4, QTableWidgetItem(row["updated_at"]))
            self.table.setItem(i, 5, QTableWidgetItem(row["error"]))
        self._on_select()

    def _selected_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]["id"]

    def _on_select(self) -> None:
        task_id = self._selected_id()
        has_selection = task_id is not None
        self.rerun_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.skips_btn.setEnabled(has_selection)
        if not has_selection:
            self.detail.setPlainText("")
            return
        entry = get_history_entry(self.service.store, task_id)
        if entry is None:
            self.detail.setPlainText("该任务已不在库中。")
            return
        lines = [f"任务：{entry['episode_name'] or entry['video_path']}"]
        lines.append(f"状态：{entry['status_label']}　当前步骤：{entry['progress'] or '—'}")
        lines.append(f"重试次数：{entry['retry_count']}")
        skips = entry.get("auth_skips") or []
        if skips:
            lines.append(f"跳过记录：共 {len(skips)} 条，点 [查看跳过记录] 看人名、原因与时间。")
        else:
            lines.append(f"跳过记录：{NO_AUTH_SKIPS_TEXT}")
        lines.append("")
        lines.append("配置快照：")
        snapshot = entry.get("config_snapshot") or build_config_snapshot(entry)
        for key, value in snapshot.items():
            lines.append(f"  {key}：{value}")
        if entry["error"]:
            lines.append("")
            lines.append("失败原因：")
            lines.append(entry["error"])
        self.detail.setPlainText("\n".join(lines))

    # ----- actions -----

    def _build_auth_skips_dialog(self, task_id: str) -> AuthSkipsDialog:
        """Build (but do not show) the skip-records viewer for a task.

        Kept separate from :meth:`_show_auth_skips` so tests can inspect
        the viewer without running a modal event loop. Read-only: it only
        reads ``auth_skips`` through :func:`get_auth_skips_for_task`.
        """
        entry = get_history_entry(self.service.store, task_id)
        title = (entry["episode_name"] if entry else "") or task_id
        if entry is None:
            return AuthSkipsDialog(title, "该任务已不在库中。", parent=self)
        text = entry.get("auth_skips_text") or format_auth_skips(
            get_auth_skips_for_task(self.service.store, task_id)
        )
        return AuthSkipsDialog(title, text, parent=self)

    def _show_auth_skips(self) -> None:
        task_id = self._selected_id()
        if task_id is None:
            return
        dlg = self._build_auth_skips_dialog(task_id)
        self.last_skips_dialog = dlg
        dlg.exec()

    def _rerun_selected(self) -> None:
        task_id = self._selected_id()
        if task_id is None:
            return
        try:
            result = self.service.rerun_task(task_id)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        # Inserted and (usually) queued by the service; say plainly which.
        self.last_rerun_task = result
        if result.get("started"):
            self.status.setText(
                f"已复跑并启动：新任务 {result['id']} 已排队执行"
                "（与主窗口「开始执行」走同一调度入口）。"
            )
        else:
            self.status.setText(
                f"复跑任务 {result['id']} 已入库，但未启动："
                f"{result.get('reason') or '未知原因'}"
            )
        self._refresh()

    def _delete_selected(self) -> None:
        task_id = self._selected_id()
        if task_id is None:
            return
        entry = get_history_entry(self.service.store, task_id)
        title = (entry["episode_name"] if entry else "") or task_id
        answer = QMessageBox.question(
            self,
            "删除历史记录",
            f"删除历史记录「{title}」？\n只删历史行，不删成片文件。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if delete_history_entry(self.service.store, task_id):
            self.status.setText(f"已删除历史记录「{title}」（成片文件未动）。")
        else:
            self.status.setText("该任务已不在库中。")
        self._refresh()
