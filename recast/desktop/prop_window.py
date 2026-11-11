"""L3 3D道具生成 pilot window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from recast.core.service import RecastService


class PropWindow(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("3D道具生成（试点）")
        self.resize(560, 360)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("道具或场景物件的描述，例如：破旧木门、土屋内的硬木床")
        self.image = QLineEdit()
        self.dest = QLineEdit(str(Path(service.settings.output_root) / "l3" / "prop"))
        form = QFormLayout()
        form.addRow("提示词", self.prompt)
        form.addRow("参考图（可选）", self._browse(self.image))
        form.addRow("输出目录", self._browse(self.dest, dir=True))
        run = QPushButton("生成试点/占位结果")
        run.setObjectName("primary")
        run.clicked.connect(self._run)
        self.log = QLabel()
        self.log.setWordWrap(True)
        note = QLabel(
            "这是 L3「3D道具生成」试点。若未配置 ComfyUI Hunyuan3D / Tripo 工作流，"
            "会写出一份任务说明和带贴图的简单盒体，明确标记为「试点/占位结果」，"
            "不会当成成品角色模型，也不会自动绑骨骼。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b90a0;")
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(run)
        layout.addWidget(self.log)
        layout.addWidget(note)

    def _browse(self, edit: QLineEdit, dir: bool = False) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        btn = QPushButton("浏览…")

        def pick() -> None:
            if dir:
                path = QFileDialog.getExistingDirectory(self, "选择目录", edit.text() or str(Path.home()))
            else:
                path, _ = QFileDialog.getOpenFileName(self, "选择参考图", edit.text() or str(Path.home()))
            if path:
                edit.setText(path)

        btn.clicked.connect(pick)
        row.addWidget(btn)
        return w

    def _run(self) -> None:
        try:
            rec = self.service.generate_prop(
                self.dest.text().strip() or str(Path(self.service.settings.output_root) / "l3" / "prop"),
                prompt=self.prompt.toPlainText().strip(),
                image=self.image.text().strip() or None,
            )
            self.log.setText(f"已写出：{rec.get('dir')}  ·  {rec.get('honesty')}")
        except Exception as exc:
            QMessageBox.warning(self, "生成失败", str(exc))
