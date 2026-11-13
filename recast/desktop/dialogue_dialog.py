"""Edit one character's dialogue lines and re-dub only the changed ones."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from recast.core.service import RecastService
from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap
from recast.generate.dialogue import load_dialogue


class _RedubSignal(QObject):
    finished = Signal(object)


class DialogueDialog(QDialog):
    """[编辑对话]：按人物改对话文本，改完只重配改动的句。

    先选分集工作目录（里面有 dialogue.json 或 shots_plan.json），默认只列
    该人物的台词；改完点 [保存改动]（只标 dirty，不配音），再点 [重配音]
    只重配改过的句，别的不动，音频绝不加速。
    """

    COLS = ["镜头", "说话人", "对白文本", "状态"]

    def __init__(self, service: RecastService, character: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.character = character
        self._sync_tasks = False
        self._redub_running = False
        self.workdir: str | None = None
        self._lines: list[dict[str, Any]] = []
        self.setWindowTitle(f"编辑对话 — {character}")
        # P5/B3: content (workdir row + table + status) scrolls, the
        # [保存改动]/[重配音]/[关闭] bar is pinned at the bottom, dialog
        # clamped to the screen (was resize(720, 480)). The table keeps
        # its own scrollbar for long dialogue lists.
        cap_dialog_to_screen(self, 720, 480)

        content = QWidget()
        layout = QVBoxLayout(content)
        top = QHBoxLayout()
        top.addWidget(QLabel("分集工作目录"))
        self.workdir_edit = QLineEdit()
        self.workdir_edit.setPlaceholderText("选含 dialogue.json 的分集工作目录")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._pick_workdir)
        load = QPushButton("加载对白")
        load.clicked.connect(self._load)
        top.addWidget(self.workdir_edit, 1)
        top.addWidget(browse)
        top.addWidget(load)
        layout.addLayout(top)
        self.only_mine = QCheckBox("只看该人物的台词")
        self.only_mine.setChecked(True)
        self.only_mine.toggled.connect(self._fill_table)
        layout.addWidget(self.only_mine)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(3, 80)
        layout.addWidget(self.table, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.save_btn = QPushButton("保存改动")
        self.save_btn.clicked.connect(self._save)
        self.redub_btn = QPushButton("重配音")
        self.redub_btn.setObjectName("primary")
        self.redub_btn.clicked.connect(self._redub)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.reject)
        row.addWidget(self.save_btn)
        row.addWidget(self.redub_btn)
        row.addStretch(1)
        row.addWidget(self.close_btn)
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))
        self._redub_signal = _RedubSignal()
        self._redub_signal.finished.connect(self._on_redub_finished)
        self._refresh_buttons()

    # ----- data -----

    def _pick_workdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择分集工作目录", self.workdir_edit.text() or str(Path.home()))
        if path:
            self.workdir_edit.setText(path)

    def _load(self) -> None:
        workdir = self.workdir_edit.text().strip()
        if not workdir:
            self.status.setText("请先选择分集工作目录。")
            return
        try:
            self._lines = load_dialogue(workdir)
        except Exception as exc:
            self.status.setText(f"加载对白失败：{exc}")
            return
        self.workdir = workdir
        self._fill_table()
        self.status.setText(f"已加载 {len(self._lines)} 句对白。")
        self._refresh_buttons()

    def _visible_lines(self) -> list[dict[str, Any]]:
        if not self.only_mine.isChecked():
            return list(self._lines)
        return [
            line
            for line in self._lines
            if str(line.get("speaker") or line.get("character") or "") == self.character
        ]

    def _fill_table(self) -> None:
        lines = self._visible_lines()
        self.table.setRowCount(0)
        for line in lines:
            row = self.table.rowCount()
            self.table.insertRow(row)
            code = QTableWidgetItem(str(line.get("shot_code") or ""))
            code.setFlags(code.flags() & ~Qt.ItemFlag.ItemIsEditable)
            code.setData(Qt.ItemDataRole.UserRole, str(line.get("shot_code") or ""))
            self.table.setItem(row, 0, code)
            speaker = QTableWidgetItem(str(line.get("speaker") or line.get("character") or ""))
            speaker.setFlags(speaker.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, speaker)
            self.table.setItem(row, 2, QTableWidgetItem(str(line.get("text") or "")))
            state = QTableWidgetItem("已改" if line.get("dirty") else "")
            state.setFlags(state.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 3, state)

    def _refresh_buttons(self) -> None:
        has_workdir = bool(self.workdir)
        self.save_btn.setEnabled(has_workdir)
        self.redub_btn.setEnabled(has_workdir and not self._redub_running)

    # ----- actions -----

    def _save(self) -> None:
        if not self.workdir:
            self.status.setText("请先加载对白。")
            return
        changed = 0
        errors: list[str] = []
        for row in range(self.table.rowCount()):
            code_item = self.table.item(row, 0)
            text_item = self.table.item(row, 2)
            if code_item is None or text_item is None:
                continue
            code = str(code_item.data(Qt.ItemDataRole.UserRole) or "")
            new_text = text_item.text().strip()
            original = next(
                (str(l.get("text") or "") for l in self._lines if str(l.get("shot_code") or "") == code),
                "",
            )
            if new_text == original or not code:
                continue
            try:
                self.service.edit_dialogue_line(self.workdir, code, new_text)
                changed += 1
            except Exception as exc:
                errors.append(f"{code}：{exc}")
        if errors:
            self.status.setText(f"保存了 {changed} 句，{len(errors)} 句失败：{'；'.join(errors)}")
        else:
            self.status.setText(f"已保存 {changed} 句改动（已标 dirty，尚未重配音）。" if changed else "没有改动。")
        try:
            self._lines = load_dialogue(self.workdir)
        except Exception:
            pass
        self._fill_table()

    def _redub(self) -> None:
        if not self.workdir or self._redub_running:
            return
        self._redub_running = True
        self._refresh_buttons()
        self.status.setText("正在重配音…（只重配改过的句）")
        workdir = self.workdir
        if self._sync_tasks:
            try:
                report = self.service.redub_dialogue(workdir)
                outcome: dict = {"ok": True, "report": report}
            except Exception as exc:
                outcome = {"ok": False, "message": str(exc)}
            self._on_redub_finished(outcome)
            return

        def work() -> None:
            try:
                report = self.service.redub_dialogue(workdir)
                outcome = {"ok": True, "report": report}
            except Exception as exc:
                outcome = {"ok": False, "message": str(exc)}
            self._redub_signal.finished.emit(outcome)

        threading.Thread(target=work, name="recast-dialogue-redub", daemon=True).start()

    def _on_redub_finished(self, outcome: dict) -> None:
        self._redub_running = False
        self._refresh_buttons()
        if not outcome.get("ok"):
            self.status.setText(f"重配音失败：{outcome.get('message')}")
            return
        report = outcome.get("report") or {}
        redubbed = report.get("redubbed") or []
        skipped = report.get("skipped") or []
        remix = report.get("remix") or {}
        parts = [f"重配 {len(redubbed)} 句，跳过 {len(skipped)} 句"]
        if remix.get("mixed"):
            parts.append(f"已混音：{remix['mixed']}")
        if remix.get("remux_error"):
            parts.append(f"成片重混失败：{remix['remux_error']}")
        elif remix.get("remuxed"):
            parts.append("成片音轨已更新")
        self.status.setText("；".join(parts) + "。")
        if self.workdir:
            try:
                self._lines = load_dialogue(self.workdir)
            except Exception:
                pass
            self._fill_table()
