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
        self.resize(960, 680)
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
        self.l2_del_btn = QPushButton("删除 3D头像资产")
        self.l2_del_btn.setObjectName("danger")
        self.l2_del_btn.clicked.connect(self._delete_l2)
        clean_l2 = QPushButton("清理未使用的 3D头像资产")
        clean_l2.clicked.connect(self._clean_unused_l2)
        self.delete_char_btn = QPushButton("删除人物")
        self.delete_char_btn.setObjectName("danger")
        self.delete_char_btn.setEnabled(False)
        self.delete_char_btn.clicked.connect(self._delete_character)
        clean_samples = QPushButton("清理失效样本")
        clean_samples.clicked.connect(self._clean_invalid_samples)

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
        l2_row = QHBoxLayout()
        l2_row.addWidget(self.l2_del_btn)
        l2_row.addWidget(clean_l2)
        l2_row.addStretch(1)
        right.addLayout(l2_row)
        char_row = QHBoxLayout()
        char_row.addWidget(self.delete_char_btn)
        char_row.addWidget(clean_samples)
        char_row.addStretch(1)
        right.addLayout(char_row)
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
        self.delete_char_btn.setEnabled(self.list.currentItem() is not None)

    def _on_select(self, item: QListWidgetItem | None) -> None:
        if item is None:
            self._current_id = None
            self.delete_char_btn.setEnabled(False)
            self.name_edit.clear()
            self.auth.setChecked(False)
            self.auth_type.setCurrentIndex(0)
            self.flags.clear()
            self._show_samples({"assets": []}, empty="请从左侧选择人物。")
            return
        self.delete_char_btn.setEnabled(True)
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

    def _show_samples(self, ch: dict, *, empty: str | None = None) -> None:
        while self.samples_layout.count():
            item = self.samples_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        paths = [a.get("path") for a in (ch.get("assets") or []) if a.get("kind") == "face" and a.get("path")]
        if not paths:
            lab = QLabel(empty or "还没有人脸样本。完成「03 人物提取与建库」后会自动出现，也可手动指定参考脸。")
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

    def _delete_l2(self) -> None:
        if not self._current_id:
            QMessageBox.information(self, "删除 3D头像资产", "请先选择一个人物。")
            return
        ch = next((c for c in self.service.characters() if c["id"] == self._current_id), None)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or self._current_id
        if not _confirm(
            self,
            "删除 3D头像资产",
            f"确定删除「{name}」的 3D头像资产？\n\n"
            "只会删除该人物的 3D头像文件，人脸照片会保留。"
            "删除后 3D 状态变为「缺失」，之后可以重新生成。",
            accept="删除",
        ):
            return
        try:
            self.service.delete_l2(self._current_id)
        except Exception as exc:
            QMessageBox.warning(self, "无法删除 3D头像资产", str(exc))
            return
        QMessageBox.information(
            self,
            "已删除 3D头像资产",
            f"已删除「{name}」的 3D头像资产。人脸照片仍保留，3D 状态为「缺失」。",
        )
        self.reload()

    def _delete_character(self) -> None:
        if not self._current_id:
            QMessageBox.information(self, "删除人物", "请先选择一个人物。")
            return
        ch = next((c for c in self.service.characters() if c["id"] == self._current_id), None)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or self._current_id
        if not _confirm(
            self,
            "删除人物",
            f"确定把「{name}」移出人物库？\n\n"
            "这个人会离开人物库，数据库里的人物记录和资料记录也会去掉。\n"
            "若有 3D头像资产，只删除位于 L2 目录内、且通过安全检查的那一份。\n"
            "生成的人脸样本（输出目录中的 face_samples）会删除。\n"
            "从其他位置指定的参考脸不会删除。剧集、任务和成片不会删除。",
            accept="删除",
        ):
            return
        try:
            self.service.delete_character(self._current_id)
        except Exception as exc:
            QMessageBox.warning(self, "无法删除人物", str(exc))
            return
        self._current_id = None
        self.reload()
        QMessageBox.information(self, "已移出人物库", f"已将「{name}」移出人物库。")

    def _clean_invalid_samples(self) -> None:
        try:
            plan = self.service.list_invalid_samples()
        except Exception as exc:
            QMessageBox.warning(self, "无法检查失效样本", str(exc))
            return
        if not plan.get("stale_count") and not plan.get("remove_count"):
            QMessageBox.information(self, "清理失效样本", "没有失效样本。")
            return
        if not _confirm(self, "清理失效样本", _cleanup_detail(plan), accept="清理"):
            return
        try:
            result = self.service.clean_invalid_samples()
        except Exception as exc:
            QMessageBox.warning(self, "无法清理失效样本", str(exc))
            return
        self._current_id = None
        self.reload()
        QMessageBox.information(
            self,
            "已清理失效样本",
            f"已清理 {result.get('stale_count', 0)} 条失效资料，"
            f"移出 {result.get('remove_count', 0)} 个人物。\n"
            "仍有人脸文件的人物留在人物库中，只去掉失效资料。",
        )

    def _clean_unused_l2(self) -> None:
        try:
            items = self.service.list_unused_l2()
        except Exception as exc:
            QMessageBox.warning(self, "无法检查 3D头像资产", str(exc))
            return
        if not items:
            QMessageBox.information(self, "清理未使用的 3D头像资产", "没有未使用的 3D头像资产。")
            return
        shown = items[:30]
        lines = []
        for item in shown:
            label = item.get("display_name") or item.get("name") or item.get("character_id") or "未命名"
            cid = item.get("character_id") or ""
            if cid and cid != label:
                label = f"{label}（{cid}）"
            reason = item.get("reason") or ""
            lines.append(f"· {label}：{reason}" if reason else f"· {label}")
        extra = ""
        if len(items) > len(shown):
            extra = f"\n… 另有 {len(items) - len(shown)} 项"
        detail = (
            f"将删除 {len(items)} 个未使用的 3D头像资产：\n\n"
            + "\n".join(lines)
            + extra
            + "\n\n不会删除仍在使用的 3D头像，也不会删除人脸、服装、声音，或 L2 目录以外的文件。\n确定删除吗？"
        )
        if not _confirm(self, "清理未使用的 3D头像资产", detail, accept="删除"):
            return
        try:
            result = self.service.delete_unused_l2(only=[item.get("path") or "" for item in items])
        except Exception as exc:
            QMessageBox.warning(self, "无法清理 3D头像资产", str(exc))
            return
        QMessageBox.information(
            self,
            "已清理 3D头像资产",
            f"已删除 {result.get('count', 0)} 个未使用的 3D头像资产。",
        )
        self.reload()


def _capped_lines(items: list[dict], *, limit: int = 30) -> list[str]:
    lines = [f"· {item.get('label') or item.get('name') or item.get('id')}" for item in items[:limit]]
    extra = len(items) - len(lines)
    if extra > 0:
        lines.append(f"… 另有 {extra} 项")
    return lines


def _cleanup_detail(plan: dict) -> str:
    kind_counts = plan.get("kind_counts") or {}
    kind_text = "、".join(f"{label} {count}" for label, count in kind_counts.items())
    summary = f"失效资料 {plan.get('stale_count', 0)} 条"
    if kind_text:
        summary += f"（{kind_text}）"
    summary += f"，将移出人物库 {plan.get('remove_count', 0)} 人（没有仍有效的人脸文件）。"
    parts = [summary]
    stale = list(plan.get("stale_assets") or [])
    if stale:
        parts.append("失效资料：\n" + "\n".join(_capped_lines(stale)))
    remove = list(plan.get("remove_characters") or [])
    if remove:
        parts.append("将移出人物库（没有仍有效的人脸文件，包括已确认和已授权的人物）：\n" + "\n".join(_capped_lines(remove)))
    keep = list(plan.get("keep_characters") or [])
    if keep:
        parts.append("以下人物仍有人脸文件，会留在人物库，只去掉失效资料：\n" + "\n".join(_capped_lines(keep)))
    parts.append(
        "不会删除仍存在的人脸文件，也不会删除剧集、任务或成片。\n"
        "输出目录以外的参考照片不会删除。仍有效的 3D头像目录不会在这一步删除，"
        "之后可以用「清理未使用的 3D头像资产」处理。\n"
        "确定清理吗？"
    )
    return "\n\n".join(parts)


def _confirm(parent: QWidget, title: str, text: str, *, accept: str, reject: str = "取消") -> bool:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.No)
    yes = box.button(QMessageBox.StandardButton.Yes)
    no = box.button(QMessageBox.StandardButton.No)
    if yes is not None:
        yes.setText(accept)
    if no is not None:
        no.setText(reject)
    return box.exec() == QMessageBox.StandardButton.Yes
