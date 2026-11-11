"""Cost table window: rows come from the runs table and match cost.xlsx."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from recast.compose.cost import cost_rows, totals_from_runs, write_cost_xlsx
from recast.core.service import RecastService


class CostWindow(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None, drama_id: str | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.drama_id = drama_id
        self.setWindowTitle("成本表")
        self.resize(860, 480)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["步骤", "时长(秒)", "云费用", "模型", "状态", "开始时间", "说明"])
        self.total = QLabel()
        export = QPushButton("导出 cost.xlsx")
        export.clicked.connect(self._export)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.reload)
        row = QHBoxLayout()
        row.addWidget(self.total, 1)
        row.addWidget(refresh)
        row.addWidget(export)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("数字来自 runs 日志，与导出的成本表一致。本机电耗按步骤时长记录，云费用仅在实际调用后写入。"))
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        self.reload()

    def reload(self) -> None:
        runs = self.service.cost_runs(drama_id=self.drama_id)
        rows = cost_rows(runs)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for c, key in enumerate(("step", "duration_sec", "cost", "model", "status", "started_at", "message")):
                self.table.setItem(i, c, QTableWidgetItem(str(r.get(key, ""))))
        tot = totals_from_runs(runs)
        self.total.setText(f"合计  {tot['count']} 步 · 时长 {tot['duration_sec']} 秒 · 云费用 {tot['cost']}")

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出成本表", str(Path.home() / "cost.xlsx"), "Excel (*.xlsx)")
        if not path:
            return
        try:
            write_cost_xlsx(self.service.cost_runs(drama_id=self.drama_id), path)
            QMessageBox.information(self, "已导出", path)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
