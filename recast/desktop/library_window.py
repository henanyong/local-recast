"""Character library: confirm names, authorize, inspect face samples side by side."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from recast.core.constants import AUTH_TYPES, ASSET_LABELS
from recast.core.service import RecastService, asset_flags


class LibraryWindow(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("人物库")
        self.resize(900, 620)
        self._current_id: str | None = None

        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_select)

        self.name_edit = QLineEdit()
        self.auth = QCheckBox("已授权，可用于人物替换")
        self.auth_type = QComboBox()
        self.auth_type.addItem("（未选择）", "")
        for t in AUTH_TYPES:
            self.auth_type.addItem(t, t)
        self.flags = QLabel()
        self.flags.setWordWrap(True)
        self.samples = QWidget()
        self.samples_layout = QHBoxLayout(self.samples)
        self.samples_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.samples)
        scroll.setMinimumHeight(220)

        save = QPushButton("保存确认")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        add_face = QPushButton("指定参考脸…")
        add_face.clicked.connect(self._pick_face)
        add_char = QPushButton("手动添加人物…")
        add_char.clicked.connect(self._add_manual)
        self.l2_btn = QPushButton("生成 3D头像资产")
        self.l2_btn.clicked.connect(self._build_l2)

        left = QVBoxLayout()
        left.addWidget(QLabel("人物列表（聚类候选需确认姓名）"))
        left.addWidget(self.list, 1)
        left.addWidget(add_char)

        right = QVBoxLayout()
        right.addWidget(QLabel("显示名称"))
        right.addWidget(self.name_edit)
        right.addWidget(self.auth)
        right.addWidget(QLabel("授权类型（自有拍摄 / 已授权 / 自有生成角色）"))
        right.addWidget(self.auth_type)
        right.addWidget(self.flags)
        right.addWidget(QLabel("人脸样本（并排对照，便于确认是否同一人）"))
        right.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.addWidget(add_face)
        row.addWidget(self.l2_btn)
        row.addWidget(save)
        row.addStretch(1)
        right.addLayout(row)
        hint = QLabel(
            "L1「人物替换」是本地换脸，不是提取了3D模型。"
            "L2「3D头像资产」是关键点 + 漫反射贴图 + 驱动配置，不是生产级完整网格；未放入官方 FLAME 时也不会假装是。"
            "质量偏低会标成「偏低」。同一人物 ID 可跨两集复用。未勾选授权的人物不能替换，也不能生成 3D头像。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b90a0;")
        right.addWidget(hint)

        layout = QHBoxLayout(self)
        lw = QWidget()
        lw.setLayout(left)
        rw = QWidget()
        rw.setLayout(right)
        layout.addWidget(lw, 1)
        layout.addWidget(rw, 2)
        self.reload()

    def reload(self) -> None:
        current = self._current_id
        self.list.clear()
        for ch in self.service.characters():
            name = ch.get("display_name") or ch.get("name") or ch["id"]
            flags = []
            if ch.get("confirmed"):
                flags.append("已确认")
            if ch.get("authorized"):
                flags.append("已授权")
            else:
                flags.append("未授权")
            item = QListWidgetItem(f"{name}  ·  {' / '.join(flags)}")
            item.setData(Qt.ItemDataRole.UserRole, ch["id"])
            self.list.addItem(item)
            if current and ch["id"] == current:
                self.list.setCurrentItem(item)
        if self.list.count() and self.list.currentItem() is None:
            self.list.setCurrentRow(0)

    def _on_select(self, item: QListWidgetItem | None) -> None:
        if item is None:
            self._current_id = None
            return
        cid = item.data(Qt.ItemDataRole.UserRole)
        self._current_id = cid
        ch = next((c for c in self.service.characters() if c["id"] == cid), None)
        if not ch:
            return
        self.name_edit.setText(ch.get("display_name") or ch.get("name") or "")
        self.auth.setChecked(bool(ch.get("authorized")))
        idx = self.auth_type.findData(ch.get("auth_type") or "")
        self.auth_type.setCurrentIndex(max(0, idx))
        flags = asset_flags(ch)
        parts = [f"{ASSET_LABELS.get(k, k)}：{v}" for k, v in flags.items()]
        self.flags.setText("资料完整度    " + "    ".join(parts))
        self._show_samples(ch)

    def _show_samples(self, ch: dict) -> None:
        while self.samples_layout.count():
            item = self.samples_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        paths = [a.get("path") for a in (ch.get("assets") or []) if a.get("kind") == "face" and a.get("path")]
        if not paths:
            lab = QLabel("还没有人脸样本。完成「03 人物提取与建库」后会自动出现，也可手动指定参考脸。")
            lab.setWordWrap(True)
            self.samples_layout.addWidget(lab)
            return
        for path in paths[:12]:
            box = QVBoxLayout()
            img = QLabel()
            img.setFixedSize(140, 160)
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(str(path))
            if not pix.isNull():
                img.setPixmap(pix.scaled(140, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                img.setText(Path(path).name)
            cap = QLabel(Path(path).name)
            cap.setStyleSheet("color:#8b90a0; font-size:11px;")
            wrap = QWidget()
            box.addWidget(img)
            box.addWidget(cap)
            wrap.setLayout(box)
            self.samples_layout.addWidget(wrap)
        self.samples_layout.addStretch(1)

    def _save(self) -> None:
        if not self._current_id:
            return
        try:
            self.service.confirm_character(
                self._current_id,
                display_name=self.name_edit.text().strip(),
                authorized=self.auth.isChecked(),
                auth_type=self.auth_type.currentData() or None,
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法保存", str(exc))
            return
        self.reload()

    def _pick_face(self) -> None:
        if not self._current_id:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择参考脸", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return
        try:
            self.service.confirm_character(self._current_id, face_path=path)
        except Exception as exc:
            QMessageBox.warning(self, "无法设置参考脸", str(exc))
            return
        self.reload()

    def _add_manual(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择参考脸（可取消，稍后补充）", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        name = Path(path).stem if path else "新人物"
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "手动添加人物", "人物名称：", text=name)
        if not ok or not str(name).strip():
            return
        self.service.add_character_manual(str(name).strip(), path or None)
        self.reload()

    def _build_l2(self) -> None:
        if not self._current_id:
            QMessageBox.information(self, "3D头像资产", "请先选择一个已授权人物。")
            return
        if self.service.settings.is_basic_tier():
            QMessageBox.information(
                self,
                "当前为基础档",
                "无独立 GPU 时仍可尝试生成「3D头像资产」（关键点 + 贴图 + 驱动配置），但 L2 驱动已禁用。",
            )
        try:
            asset = self.service.build_l2(self._current_id)
        except Exception as exc:
            QMessageBox.warning(self, "无法生成 3D头像资产", str(exc))
            return
        QMessageBox.information(
            self,
            "已生成 3D头像资产",
            f"{asset.get('honesty')}\n\n质量分：{asset.get('quality_score')}（{asset.get('quality_flag')}）\n目录：{asset.get('dir')}",
        )
        self.reload()
