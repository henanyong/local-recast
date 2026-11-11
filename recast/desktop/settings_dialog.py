"""Settings: hardware tier, directories, concurrency, optional cloud / upload / ComfyUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from recast.cloud.channels import estimate_cost
from recast.cloud.secrets import SecretStore
from recast.core.service import RecastService


class SettingsDialog(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.secrets = SecretStore()
        self.setWindowTitle("设置")
        self.resize(620, 680)
        s = service.settings

        self.tier = QComboBox()
        self.tier.addItem("基础档（无独立 GPU，禁用重生成 / L2 驱动）", "basic")
        self.tier.addItem("主流档 · 4060 Ti 8GB（推荐，默认）", "4060ti-8g")
        self.tier.addItem("推荐档 · 4060 Ti 16GB（仅确认是 16GB 时使用）", "4060ti-16g")
        idx = self.tier.findData(s.hardware_tier)
        self.tier.setCurrentIndex(max(0, idx))

        self.works = QLineEdit(s.works_root)
        self.output = QLineEdit(s.output_root)
        self.models = QLineEdit(s.models_dir)
        self.ffmpeg = QLineEdit(s.ffmpeg_dir)
        self.comfy = QLineEdit(getattr(s, "comfyui_url", "") or "")
        self.comfy.setPlaceholderText("例如 http://127.0.0.1:8188，留空则用分镜预览成片")
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 2)
        self.concurrency.setValue(s.clamped_concurrency())
        self.lang = QComboBox()
        self.lang.addItem("自动检测（对中文友好）", "auto")
        self.lang.addItem("中文", "zh")
        self.lang.addItem("英文", "en")
        li = self.lang.findData(s.lang)
        self.lang.setCurrentIndex(max(0, li))
        self.burn = QCheckBox("把字幕烧进成片（旁路 SRT 始终会写）")
        self.burn.setChecked(bool(s.burn_subs))
        self.pose = QCheckBox("额外导出动作关键点（可选，默认关闭）")
        self.pose.setChecked(bool(s.with_pose))

        layout = QVBoxLayout(self)

        hard = QGroupBox("硬件档与本地引擎")
        hf = QFormLayout(hard)
        hf.addRow("硬件档", self.tier)
        hf.addRow("任务并发（GPU）", self._hint_row(self.concurrency, "默认 1 个 GPU 任务；CPU 可预处理下一集"))
        hf.addRow("字幕语言", self.lang)
        hf.addRow("", self.burn)
        hf.addRow("", self.pose)
        layout.addWidget(hard)

        dirs = QGroupBox("目录与本地生成")
        df = QFormLayout(dirs)
        df.addRow("作品根目录", self._browse_row(self.works, "选择作品根目录"))
        df.addRow("成片输出目录", self._browse_row(self.output, "选择输出目录"))
        df.addRow("模型目录", self._browse_row(self.models, "选择模型目录"))
        df.addRow("ffmpeg 目录", self._browse_row(self.ffmpeg, "选择 ffmpeg 所在目录（可空，默认走 PATH）"))
        df.addRow("ComfyUI HTTP", self.comfy)
        layout.addWidget(dirs)

        cloud = QGroupBox("云通道（默认关闭；密钥加密保存，不进日志和导出）")
        cf = QFormLayout(cloud)
        self.cloud_enabled = QCheckBox("启用云备选通道（每次调用前会弹出预估并需确认）")
        self.cloud_enabled.setChecked(bool(getattr(s, "cloud_enabled", False)))
        cf.addRow(self.cloud_enabled)
        self.volc_key = QLineEdit()
        self.volc_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.volc_key.setPlaceholderText("火山 Ark API Key / SK，可空")
        self.deep_key = QLineEdit()
        self.deep_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.deep_key.setPlaceholderText("DeepSeek API Key，可空")
        self.oai_key = QLineEdit()
        self.oai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.oai_key.setPlaceholderText("OpenAI 兼容 API Key，可空")
        self.oai_base = QLineEdit(s.cloud_openai.base_url if getattr(s, "cloud_openai", None) else "")
        self.oai_base.setPlaceholderText("https://api.openai.com/v1 或自建兼容地址")
        # do not load secrets into the line edit; only show configured or not
        volc = self.secrets.get("volcengine")
        deep = self.secrets.get("deepseek")
        oai = self.secrets.get("generic")
        cf.addRow("火山引擎", self._hint_row(self.volc_key, "已配置" if volc.get("api_key") or volc.get("sk") else "未配置"))
        cf.addRow("DeepSeek", self._hint_row(self.deep_key, "已配置" if deep.get("api_key") else "未配置"))
        cf.addRow("OpenAI 兼容 Key", self.oai_key)
        cf.addRow("OpenAI 兼容地址", self.oai_base)
        layout.addWidget(cloud)

        up = QGroupBox("自动上传（默认关闭，只发往你自己指定的目录 / webhook / 命令）")
        uf = QFormLayout(up)
        self.auto_upload = QCheckBox("任务完成后自动上传")
        self.auto_upload.setChecked(bool(getattr(s, "auto_upload", False)))
        self.upload_mode = QComboBox()
        self.upload_mode.addItem("复制到文件夹", "folder")
        self.upload_mode.addItem("HTTP webhook", "webhook")
        self.upload_mode.addItem("自定义命令", "command")
        mi = self.upload_mode.findData(getattr(s, "upload_mode", "folder"))
        self.upload_mode.setCurrentIndex(max(0, mi))
        self.upload_folder = QLineEdit(getattr(s, "upload_folder", "") or "")
        self.upload_webhook = QLineEdit(getattr(s, "upload_webhook", "") or "")
        self.upload_command = QLineEdit(getattr(s, "upload_command", "") or "")
        self.upload_command.setPlaceholderText("例如 scp {src} user@host:dir/{name}")
        uf.addRow(self.auto_upload)
        uf.addRow("方式", self.upload_mode)
        uf.addRow("目标文件夹", self._browse_row(self.upload_folder, "选择上传目标文件夹"))
        uf.addRow("Webhook URL", self.upload_webhook)
        uf.addRow("上传命令", self.upload_command)
        layout.addWidget(up)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_row(self, edit: QLineEdit, title: str) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        btn = QPushButton("浏览…")

        def pick() -> None:
            path = QFileDialog.getExistingDirectory(self, title, edit.text() or str(Path.home()))
            if path:
                edit.setText(path)

        btn.clicked.connect(pick)
        row.addWidget(btn)
        return w

    def _hint_row(self, widget: QWidget, hint: str) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget)
        lab = QLabel(hint)
        lab.setStyleSheet("color:#8b90a0;")
        row.addWidget(lab, 1)
        return w

    def apply(self) -> None:
        if self.volc_key.text().strip():
            self.secrets.set("volcengine", {"api_key": self.volc_key.text().strip()})
        if self.deep_key.text().strip():
            self.secrets.set("deepseek", {"api_key": self.deep_key.text().strip()})
        if self.oai_key.text().strip() or self.oai_base.text().strip():
            rec = self.secrets.get("generic")
            if self.oai_key.text().strip():
                rec["api_key"] = self.oai_key.text().strip()
            if self.oai_base.text().strip():
                rec["base_url"] = self.oai_base.text().strip()
            self.secrets.set("generic", rec)
        volc_cfg = bool(self.secrets.configured("volcengine") or self.volc_key.text().strip())
        deep_cfg = bool(self.secrets.configured("deepseek") or self.deep_key.text().strip())
        oai_cfg = bool(self.secrets.configured("generic") or self.oai_key.text().strip())
        self.service.update_settings(
            hardware_tier=self.tier.currentData(),
            works_root=self.works.text().strip(),
            output_root=self.output.text().strip(),
            models_dir=self.models.text().strip(),
            ffmpeg_dir=self.ffmpeg.text().strip(),
            comfyui_url=self.comfy.text().strip(),
            concurrency=int(self.concurrency.value()),
            lang=self.lang.currentData(),
            burn_subs=self.burn.isChecked(),
            with_pose=self.pose.isChecked(),
            cloud_enabled=self.cloud_enabled.isChecked(),
            auto_upload=self.auto_upload.isChecked(),
            auto_upload_enabled=True,
            upload_mode=self.upload_mode.currentData(),
            upload_folder=self.upload_folder.text().strip(),
            upload_webhook=self.upload_webhook.text().strip(),
            upload_command=self.upload_command.text().strip(),
            cloud_volcengine={"name": "火山引擎", "configured": volc_cfg, "enabled": self.cloud_enabled.isChecked() and volc_cfg, "status": "已配置" if volc_cfg else "未配置"},
            cloud_deepseek={"name": "DeepSeek", "configured": deep_cfg, "enabled": self.cloud_enabled.isChecked() and deep_cfg, "status": "已配置" if deep_cfg else "未配置"},
            cloud_openai={"name": "OpenAI 兼容", "configured": oai_cfg, "enabled": self.cloud_enabled.isChecked() and oai_cfg, "status": "已配置" if oai_cfg else "未配置", "base_url": self.oai_base.text().strip()},
        )


def confirm_cloud_estimate(parent: QWidget | None, estimate: dict) -> bool:
    msg = estimate.get("message") or f"预估费用 {estimate.get('estimate')} USD，确认后才调用。"
    return QMessageBox.question(parent, "确认云调用", msg) == QMessageBox.StandardButton.Yes
