"""Simple first-run self-check: GPU tier, disk, ffmpeg, works directory."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from recast.core.engine import first_run_report, suggest_tier, detect_vram_gb
from recast.core.service import RecastService


def maybe_first_run(service: RecastService, parent: QWidget | None = None) -> None:
    if service.settings.first_run_done:
        return
    dlg = FirstRunDialog(service, parent)
    if dlg.exec():
        dlg.apply()


class FirstRunDialog(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("首次运行自检")
        self.resize(620, 460)
        report = first_run_report(service.settings)
        snap = report["snapshot"]

        layout = QVBoxLayout(self)
        intro = QLabel(
            "欢迎使用本地再铸 V1.0。这是环境自检：硬件档、磁盘、ffmpeg、模型缺失项和作品目录。核心流程不需要云密钥即可离线跑完。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        gpu = snap.get("gpu_name") or "未检测到 CUDA GPU"
        vram = snap.get("vram_gb")
        vram_s = f"{vram:.1f} GB" if vram is not None else "未知"
        suggested = suggest_tier(detect_vram_gb(), snap.get("gpu_name"))
        form.addRow("显卡", QLabel(f"{gpu}  ·  显存 {vram_s}"))
        labels = {"basic": "基础档（无独立 GPU）", "4060ti-16g": "推荐档 · 4060 Ti 16GB", "4060ti-8g": "主流档 · 4060 Ti 8GB（推荐）"}
        form.addRow("建议硬件档", QLabel(labels.get(suggested, suggested)))
        form.addRow("ffmpeg", QLabel("已找到" if (snap.get("ffmpeg") or {}).get("ok") else "未找到，请安装或在设置里指定目录"))
        disk = snap.get("disk_free_gb")
        form.addRow("磁盘剩余", QLabel(f"{disk:.1f} GB" if disk is not None else "未知"))
        layout.addLayout(form)

        if report["issues"]:
            issues = QLabel("需要留意：\n- " + "\n- ".join(report["issues"]))
            issues.setWordWrap(True)
            issues.setStyleSheet("color:#ffb347;")
            layout.addWidget(issues)
        if report.get("fixes"):
            fixes = QLabel("可行动的修复：\n- " + "\n- ".join(report["fixes"]))
            fixes.setWordWrap(True)
            layout.addWidget(fixes)

        self.works = QLineEdit(service.settings.works_root)
        row = QHBoxLayout()
        row.addWidget(QLabel("作品根目录"))
        row.addWidget(self.works, 1)
        btn = QPushButton("浏览…")

        def pick() -> None:
            path = QFileDialog.getExistingDirectory(self, "选择作品根目录", self.works.text() or str(Path.home()))
            if path:
                self.works.setText(path)

        btn.clicked.connect(pick)
        row.addWidget(btn)
        layout.addLayout(row)

        note = QLabel(
            "无独显时会降为基础档：不会崩溃，但会禁用 L2 驱动和 ComfyUI 重生成，改用「分镜预览成片」。"
            "L1「人物替换」是本地换脸，不是提取了3D模型。L2 才叫「3D头像」，L3 叫「3D道具生成」。"
            "自动上传默认关闭。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b90a0;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def apply(self) -> None:
        snap = first_run_report(self.service.settings)["snapshot"]
        suggested = suggest_tier(detect_vram_gb(), snap.get("gpu_name"))
        self.service.update_settings(
            first_run_done=True,
            works_root=self.works.text().strip() or self.service.settings.works_root,
            hardware_tier=suggested,
        )
