"""Create a generate-from-script task from a template pack + storyboard / dialogue table."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QMessageBox,
)

from recast.core.service import RecastService
from recast.config import PROJECT_ROOT
from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap


class GenerateDialog(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("按剧本生成新集")
        # P5/B3: content scrolls, the OK/Cancel bar is pinned at the
        # bottom, dialog clamped to the screen (was resize(640, 320)).
        cap_dialog_to_screen(self, 640, 320)
        self.name = QLineEdit("开局即打·第1集")
        self.pack = QLineEdit(str(PROJECT_ROOT / "examples" / "开局即打-第1集模板"))
        self.storyboard = QLineEdit()
        self.dialogue = QLineEdit()
        content = QWidget()
        form = QFormLayout(content)
        form.addRow("剧目名称", self.name)
        form.addRow("模板包目录", self._browse(self.pack, "选择模板包目录", dir=True))
        form.addRow("分镜文本/JSON", self._browse(self.storyboard, "选择分镜文件"))
        form.addRow("对白表", self._browse(self.dialogue, "选择对白表"))
        note = QLabel(
            "会先套用四级模板并校验参考图，再生成静音画面，最后贴对白。"
            "若未连接 ComfyUI，将使用「分镜预览成片」（参考静帧推拉），不是完整图生视频。"
            "对白字数超出镜头时长时会停在待人工检查，不会加速音频。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b90a0;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))

    def _browse(self, edit: QLineEdit, title: str, dir: bool = False) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        btn = QPushButton("浏览…")

        def pick() -> None:
            if dir:
                path = QFileDialog.getExistingDirectory(self, title, edit.text() or str(Path.home()))
            else:
                path, _ = QFileDialog.getOpenFileName(self, title, edit.text() or str(Path.home()))
            if path:
                edit.setText(path)

        btn.clicked.connect(pick)
        row.addWidget(btn)
        return w

    def apply(self) -> dict | None:
        pack = self.pack.text().strip()
        if not pack:
            QMessageBox.warning(self, "缺少模板包", "请选择模板包目录。")
            return None
        try:
            return self.service.create_script_drama(
                self.name.text().strip() or "未命名剧目",
                pack,
                storyboard=self.storyboard.text().strip() or None,
                dialogue=self.dialogue.text().strip() or None,
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法创建任务", str(exc))
            return None
