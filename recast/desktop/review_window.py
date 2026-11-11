"""Review queue for flagged shots, with single-shot rerun."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from recast.core.service import RecastService


class ReviewWindow(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None, drama_id: str | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.drama_id = drama_id
        self.setWindowTitle("质量终检")
        self.resize(840, 460)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["镜头", "原因", "详情", "任务"])
        self.table.setColumnWidth(0, 100)
        self.table.setColumnWidth(1, 140)
        rerun = QPushButton("重跑所选镜头")
        rerun.clicked.connect(self._rerun)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.reload)
        row = QHBoxLayout()
        row.addWidget(rerun)
        row.addWidget(refresh)
        row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("生成或人物替换后，低相似人脸、对白超时、服装/风格标记会进入这里。可单镜重跑。"))
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        self.reload()

    def reload(self) -> None:
        items = self.service.review_items(drama_id=self.drama_id)
        self.table.setRowCount(len(items))
        self._rows = items
        for i, r in enumerate(items):
            self.table.setItem(i, 0, QTableWidgetItem(str(r.get("shot_code") or "")))
            self.table.setItem(i, 1, QTableWidgetItem(str(r.get("reason") or "")))
            self.table.setItem(i, 2, QTableWidgetItem(str(r.get("detail") or "")))
            self.table.setItem(i, 3, QTableWidgetItem(str(r.get("task_id") or "")))

    def _rerun(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(getattr(self, "_rows", [])):
            QMessageBox.information(self, "重跑", "请先选中一个被标记的镜头。")
            return
        rec = self._rows[row]
        try:
            self.service.rerun_shot(rec["task_id"], rec.get("shot_code") or "")
            QMessageBox.information(self, "已排队", f"镜头 {rec.get('shot_code')} 已加入重跑队列。")
        except Exception as exc:
            QMessageBox.warning(self, "无法重跑", str(exc))
