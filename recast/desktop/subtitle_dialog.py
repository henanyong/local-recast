"""Subtitle editor dialog: edit SRT rows (text / times / split / merge / insert / delete).

The dialog only calls functions from ``recast.stages.subtitles``; it holds no
file logic of its own. It edits ``subtitles.srt`` in place and, when a
``subtitles.json`` sits next to it, keeps that in sync too.

Next-round wiring (step 5 "声音与台词"): the step panel opens this dialog
with the workdir's ``subtitles.srt`` and the episode's shot list::

    from recast.desktop.subtitle_dialog import SubtitleDialog
    dlg = SubtitleDialog(workdir / "subtitles.srt", shots=shots)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        # SRT (+ subtitles.json) already saved; then compose with cover.
        # The cover mode comes from configuration, never a hardcoded value:
        # ComposeConfig.cover_mode (yaml preset or the settings-dialog
        # override) translated by cover_options_for_compose ("off" means
        # never cover; anything else keeps auto-detection).
        from recast.core.runner import cover_options_for_compose
        cover_burned_subs, cover_mode = cover_options_for_compose(
            cfg.compose.cover_mode
        )
        compose.run(..., subtitle=srt, burn_subs=True,
                    cover_burned_subs=cover_burned_subs, cover_mode=cover_mode)

Timeline problems (overlap / too fast / spanning a cut) are shown in red;
they warn but do not block saving. Hard errors (empty text, end <= start,
overlapping times) block saving with a Chinese reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap
from recast.stages.subtitles import (
    _fmt_ts,
    _parse_ts,
    check_subtitle_timeline,
    delete_segment,
    insert_segment,
    merge_segments,
    parse_srt,
    save_segments,
    split_segment,
    validate_segments,
)

_ISSUE_BG = QColor(122, 30, 30)
_BAD_TIME_BG = QColor(122, 30, 30)


class SubtitleDialog(QDialog):
    """Edit one SRT file row by row."""

    def __init__(
        self,
        srt_path: str | Path,
        parent: QWidget | None = None,
        shots: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.srt_path = Path(srt_path)
        self.shots = list(shots or [])
        self.segments: list[dict[str, Any]] = parse_srt(self.srt_path)
        self._rebuilding = False
        self._refreshing = False
        self.setWindowTitle(f"编辑字幕 — {self.srt_path.name}")
        # P5/B3: content (table + hint + row ops + status) scrolls, the
        # Save/Cancel bar is pinned at the bottom, dialog clamped to the
        # screen (was resize(720, 460)). The table keeps its own
        # scrollbar for long SRT files.
        cap_dialog_to_screen(self, 720, 460)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        self.table = QTableWidget(0, 3, content)
        self.table.setHorizontalHeaderLabels(["开始", "结束", "文字"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 130)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._sync_buttons)

        hint = QLabel("双击单元格可改；时间格式 00:00:01,234。红色行是时间轴问题（重叠 / 语速过快 / 横跨切镜点），仅提醒，不阻止保存。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b90a0;")

        ops = QHBoxLayout()
        self.split_btn = QPushButton("拆分")
        self.split_btn.clicked.connect(self._split_current)
        self.merge_btn = QPushButton("合并下一行")
        self.merge_btn.clicked.connect(self._merge_current)
        self.delete_btn = QPushButton("删除")
        self.delete_btn.clicked.connect(self._delete_current)
        self.insert_btn = QPushButton("插入")
        self.insert_btn.clicked.connect(self._insert_after_current)
        for btn in (self.split_btn, self.merge_btn, self.delete_btn, self.insert_btn):
            ops.addWidget(btn)
        ops.addStretch(1)

        self.status = QLabel()
        self.status.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons

        content_layout.addWidget(self.table, 1)
        content_layout.addWidget(hint)
        content_layout.addLayout(ops)
        content_layout.addWidget(self.status)
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))

        self._rebuild_table()
        self._refresh_checks()

    # -- table <-> segments ---------------------------------------------------

    def _rebuild_table(self) -> None:
        self._rebuilding = True
        try:
            self.table.setRowCount(len(self.segments))
            for row, seg in enumerate(self.segments):
                start_item = QTableWidgetItem(_fmt_ts(float(seg["start"])))
                start_item.setData(Qt.ItemDataRole.UserRole, float(seg["start"]))
                end_item = QTableWidgetItem(_fmt_ts(float(seg["end"])))
                end_item.setData(Qt.ItemDataRole.UserRole, float(seg["end"]))
                text_item = QTableWidgetItem(str(seg.get("text") or ""))
                self.table.setItem(row, 0, start_item)
                self.table.setItem(row, 1, end_item)
                self.table.setItem(row, 2, text_item)
        finally:
            self._rebuilding = False
        self._sync_buttons()

    def _read_table(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            text_item = self.table.item(row, 2)
            out.append(
                {
                    "start": float(start_item.data(Qt.ItemDataRole.UserRole)),
                    "end": float(end_item.data(Qt.ItemDataRole.UserRole)),
                    "text": (text_item.text() if text_item else "").strip(),
                }
            )
        return out

    def _current_row(self) -> int:
        row = self.table.currentRow()
        return row if 0 <= row < self.table.rowCount() else -1

    # -- checks ---------------------------------------------------------------

    def _refresh_checks(self) -> None:
        if self._rebuilding or self._refreshing:
            return
        self._refreshing = True
        try:
            segments = self._read_table()
            errors = validate_segments(segments)
            issues = check_subtitle_timeline(segments, self.shots or None)
            by_line: dict[int, list[str]] = {}
            for issue in issues:
                by_line.setdefault(int(issue["line"]), []).append(str(issue["reason"]))
            for row in range(self.table.rowCount()):
                reasons = by_line.get(row + 1, [])
                bg = _ISSUE_BG if reasons else None
                for col in range(3):
                    item = self.table.item(row, col)
                    if item is None:
                        continue
                    item.setBackground(bg if bg is not None else QColor(0, 0, 0, 0))
                    item.setToolTip("\n".join(reasons) if reasons else "")
            if errors:
                self.status.setText("不合法，无法保存：\n" + "\n".join(errors))
                self.status.setStyleSheet("color:#ff7a7a;")
            elif issues:
                self.status.setText(
                    f"时间轴提醒（{len(issues)} 条，不阻止保存）：\n" + "\n".join(i["reason"] for i in issues)
                )
                self.status.setStyleSheet("color:#e8c15a;")
            else:
                self.status.setText(f"共 {len(segments)} 行，检查通过。")
                self.status.setStyleSheet("")
        finally:
            self._refreshing = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._rebuilding:
            return
        if item.column() in (0, 1):
            try:
                value = _parse_ts(item.text())
            except ValueError as exc:
                item.setBackground(_BAD_TIME_BG)
                self.status.setText(str(exc))
                self.status.setStyleSheet("color:#ff7a7a;")
                return
            self._rebuilding = True
            try:
                item.setData(Qt.ItemDataRole.UserRole, value)
                item.setText(_fmt_ts(value))
                # Background is owned by _refresh_checks (it paints issue rows
                # red); touching it here would re-trigger itemChanged and loop.
            finally:
                self._rebuilding = False
        self._refresh_checks()

    def _sync_buttons(self) -> None:
        has_rows = self.table.rowCount() > 0
        current = self._current_row()
        self.split_btn.setEnabled(has_rows and current >= 0)
        self.merge_btn.setEnabled(has_rows and 0 <= current < self.table.rowCount() - 1)
        self.delete_btn.setEnabled(has_rows and current >= 0)
        self.insert_btn.setEnabled(True)

    # -- operations -----------------------------------------------------------

    def _apply(self, segments: list[dict[str, Any]], select_row: int | None = None) -> None:
        self.segments = segments
        self._rebuild_table()
        if select_row is not None and 0 <= select_row < self.table.rowCount():
            self.table.setCurrentCell(select_row, 2)
        self._refresh_checks()

    def _split_current(self) -> None:
        row = self._current_row()
        if row < 0:
            return
        try:
            self._apply(split_segment(self._read_table(), row), select_row=row + 1)
        except (ValueError, IndexError) as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color:#ff7a7a;")

    def _merge_current(self) -> None:
        row = self._current_row()
        if row < 0:
            return
        try:
            self._apply(merge_segments(self._read_table(), row), select_row=row)
        except (ValueError, IndexError) as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color:#ff7a7a;")

    def _delete_current(self) -> None:
        row = self._current_row()
        if row < 0:
            return
        try:
            self._apply(delete_segment(self._read_table(), row), select_row=min(row, self.table.rowCount() - 2))
        except (ValueError, IndexError) as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color:#ff7a7a;")

    def _insert_after_current(self) -> None:
        segments = self._read_table()
        row = self._current_row()
        if row < 0:
            start, end = 0.0, 2.0
            at = len(segments)
        else:
            start = float(segments[row]["end"]) + 0.2
            end = start + 2.0
            at = row + 1
        try:
            self._apply(
                insert_segment(segments, at, start, end, "新字幕"),
                select_row=at,
            )
        except (ValueError, IndexError) as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color:#ff7a7a;")

    # -- save -----------------------------------------------------------------

    def _save(self) -> None:
        segments = self._read_table()
        json_path = self.srt_path.with_name("subtitles.json")
        try:
            save_segments(segments, self.srt_path, json_path if json_path.is_file() else None)
        except ValueError as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color:#ff7a7a;")
            return
        self.accept()
