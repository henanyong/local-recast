"""Review queue for flagged shots, with single-shot rerun."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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
        self._enhance_flags: dict[str, dict[str, bool]] = {}
        self.setWindowTitle("质量终检")
        self.resize(980, 520)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["镜头", "原因", "相似度", "参考脸", "详情", "状态", "增强"])
        self.table.setColumnWidth(0, 90)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 70)
        self.table.setColumnWidth(3, 88)
        self.table.verticalHeader().setDefaultSectionSize(76)
        rerun = QPushButton("重跑所选镜头")
        rerun.clicked.connect(self._rerun)
        accept = QPushButton("通过")
        accept.clicked.connect(lambda: self._set_status("已通过"))
        ignore = QPushButton("忽略")
        ignore.clicked.connect(lambda: self._set_status("已忽略"))
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.reload)
        row = QHBoxLayout()
        row.addWidget(rerun)
        row.addWidget(accept)
        row.addWidget(ignore)
        row.addWidget(refresh)
        row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("低相似人脸会带上参考脸缩略图和分数。选中一行后可重跑该镜头，或标为通过 / 忽略。"))
        layout.addWidget(QLabel("“增强”列：开——合成时对这个镜头做轻量增强（去噪 + 锐化），默认关；只影响该镜头，不用整段重跑。"))
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
            score = r.get("score")
            score_text = "" if score is None else f"{float(score):.3f}"
            self.table.setItem(i, 2, QTableWidgetItem(score_text))
            thumb = QLabel("无图")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            path = r.get("thumbnail")
            if path and Path(str(path)).is_file():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    thumb.setPixmap(pix.scaled(72, 72, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    thumb.setToolTip(str(path))
            self.table.setCellWidget(i, 3, thumb)
            self.table.setItem(i, 4, QTableWidgetItem(str(r.get("detail") or "")))
            self.table.setItem(i, 5, QTableWidgetItem(str(r.get("status") or "")))
            self.table.setCellWidget(i, 6, self._enhance_box(r))

    def _enhance_box(self, rec: dict) -> QCheckBox:
        """逐镜头后处理开关（PM 第 2 轮 4.2）：只影响这一个镜头，默认关。"""
        box = QCheckBox()
        box.setToolTip("开：合成时对这个镜头做轻量增强（去噪 + 锐化）。")
        task_id = str(rec.get("task_id") or "")
        shot = str(rec.get("shot_code") or "")
        flags = self._enhance_flags.get(task_id)
        if flags is None:
            try:
                flags = self.service.shot_enhance_flags(task_id)
            except Exception:
                flags = {}
            self._enhance_flags[task_id] = dict(flags or {})
        box.setChecked(bool((flags or {}).get(shot, False)))
        box.stateChanged.connect(
            lambda _st, t=task_id, s=shot, b=box: self._set_enhance(t, s, b)
        )
        return box

    def _set_enhance(self, task_id: str, shot_code: str, box: QCheckBox) -> None:
        wanted = box.isChecked()
        try:
            self.service.set_shot_enhance(task_id, shot_code, wanted)
        except Exception as exc:
            box.blockSignals(True)
            box.setChecked(not wanted)
            box.blockSignals(False)
            QMessageBox.warning(self, "无法设置增强", str(exc))
            return
        self._enhance_flags.setdefault(task_id, {})[shot_code] = wanted

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

    def _set_status(self, status: str) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(getattr(self, "_rows", [])):
            QMessageBox.information(self, "终检", "请先选中一个被标记的镜头。")
            return
        rec = self._rows[row]
        try:
            self.service.set_review_status(rec["id"], status)
        except Exception as exc:
            QMessageBox.warning(self, "无法更新", str(exc))
            return
        self.reload()
