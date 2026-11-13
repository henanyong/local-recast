"""Settings: hardware tier, directories, concurrency, optional cloud / upload / ComfyUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
from recast.config import EXPORT_PRESETS, EXPORT_PRESET_LABELS
from recast.core.service import RecastService
from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap


class SettingsDialog(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.secrets = SecretStore()
        self.setWindowTitle("设置")
        cap_dialog_to_screen(self, 640, 860)
        s = service.settings
        from recast.config import RecastConfig, resolve_config_path

        try:
            preset = RecastConfig.from_yaml(resolve_config_path(s.config_file()))
        except Exception:
            preset = None
        crf_value = int(s.export_crf if s.export_crf else (preset.compose.crf if preset else 18))
        nvenc_value = bool(preset.compose.use_nvenc) if s.use_nvenc is None else bool(s.use_nvenc)
        cover_value = s.cover_mode if getattr(s, "cover_mode", None) in {"blur", "bar", "off"} else (preset.compose.cover_mode if preset else "blur")
        if cover_value not in {"blur", "bar", "off"}:
            cover_value = "blur"
        scene_value = float(s.scene_threshold if s.scene_threshold is not None else (preset.scene.threshold if preset else 27))
        min_shot_value = float(s.min_shot_sec if s.min_shot_sec is not None else (preset.scene.min_shot_sec if preset else 0.4))
        pose_quality = s.pose_quality or (preset.pose.quality if preset else "accuracy")
        pose_every = int(s.pose_sample_every or (preset.pose.sample_every if preset else 2))
        detect_every = int(s.detect_every or (preset.replace.detect_every if preset else 4))

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
        self.cover_mode = QComboBox()
        self.cover_mode.addItem("模糊遮盖原烧录字幕", "blur")
        self.cover_mode.addItem("黑条遮盖原烧录字幕", "bar")
        self.cover_mode.addItem("不遮盖原烧录字幕", "off")
        self.cover_mode.setCurrentIndex(max(0, self.cover_mode.findData(cover_value)))
        self.pose = QCheckBox("额外导出动作关键点（可选，默认关闭）")
        self.pose.setChecked(bool(s.with_pose))

        # P5/B3: all GroupBox content lives in one scrollable content widget;
        # the OK/Cancel bar is pinned at the bottom outside the scroll area,
        # so it stays visible and clickable on a 1280x720 screen.
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # -- 导出画质预设（设计 8.5/10.5）：一点同时改 CRF / 检测间隔；
        # 预设不改并发（N28），并发只由上面「任务并发（GPU）」单独改 --
        self._export_preset = s.export_preset if getattr(s, "export_preset", None) in EXPORT_PRESETS else None
        self._applying_preset = False
        preset_group = QGroupBox("导出画质预设")
        preset_row = QHBoxLayout(preset_group)
        self.preset_buttons: dict[str, QPushButton] = {}
        for key in ("high", "balanced", "fast"):
            btn = QPushButton(EXPORT_PRESET_LABELS[key])
            btn.setToolTip(
                f"CRF {EXPORT_PRESETS[key]['crf']}、换脸全图检测间隔 "
                f"{EXPORT_PRESETS[key]['detect_every']}（不改任务并发）"
            )
            btn.clicked.connect(lambda _checked=False, k=key: self._apply_export_preset(k))
            self.preset_buttons[key] = btn
            preset_row.addWidget(btn)
        self.preset_label = QLabel()
        preset_row.addWidget(self.preset_label, 1)
        content_layout.addWidget(preset_group)

        # -- 设置预设（命名快照）：存当前可调设置、加载、删除 --
        named_group = QGroupBox("设置预设")
        named_row = QHBoxLayout(named_group)
        self.named_preset_combo = QComboBox()
        self.named_preset_combo.setMinimumWidth(160)
        named_row.addWidget(self.named_preset_combo, 1)
        self.named_save_btn = QPushButton("保存当前为预设…")
        self.named_save_btn.clicked.connect(self._save_named_preset)
        named_row.addWidget(self.named_save_btn)
        self.named_load_btn = QPushButton("加载")
        self.named_load_btn.clicked.connect(self._load_named_preset)
        named_row.addWidget(self.named_load_btn)
        self.named_delete_btn = QPushButton("删除")
        self.named_delete_btn.clicked.connect(self._delete_named_preset)
        named_row.addWidget(self.named_delete_btn)
        content_layout.addWidget(named_group)

        hard = QGroupBox("硬件档与本地引擎")
        hf = QFormLayout(hard)
        hf.addRow("硬件档", self.tier)
        hf.addRow("任务并发（GPU）", self._hint_row(self.concurrency, "默认 1 个 GPU 任务；CPU 可预处理下一集"))
        hf.addRow("字幕语言", self.lang)
        hf.addRow("", self.burn)
        hf.addRow("遮盖原烧录字幕", self.cover_mode)
        hf.addRow("", self.pose)
        content_layout.addWidget(hard)

        precision = QGroupBox("导出与精度")
        pf = QFormLayout(precision)
        self.nvenc = QCheckBox("优先 NVENC（不可用或失败时回退 libx264）")
        self.nvenc.setChecked(nvenc_value)
        self.crf = QSpinBox()
        self.crf.setRange(1, 51)
        self.crf.setValue(crf_value)
        self.scene_threshold = QDoubleSpinBox()
        self.scene_threshold.setRange(1.0, 80.0)
        self.scene_threshold.setSingleStep(1.0)
        self.scene_threshold.setDecimals(1)
        self.scene_threshold.setValue(scene_value)
        self.min_shot = QDoubleSpinBox()
        self.min_shot.setRange(0.0, 10.0)
        self.min_shot.setSingleStep(0.1)
        self.min_shot.setDecimals(2)
        self.min_shot.setValue(min_shot_value)
        self.pose_quality = QComboBox()
        self.pose_quality.addItem("逐帧 + 平滑（精度）", "accuracy")
        self.pose_quality.addItem("抽帧（速度）", "speed")
        pq = self.pose_quality.findData(pose_quality if pose_quality in {"accuracy", "speed"} else "accuracy")
        self.pose_quality.setCurrentIndex(max(0, pq))
        self.pose_every = QSpinBox()
        self.pose_every.setRange(1, 30)
        self.pose_every.setValue(max(1, pose_every))
        self.detect_every = QSpinBox()
        self.detect_every.setRange(1, 60)
        self.detect_every.setValue(max(1, detect_every))
        pf.addRow("", self.nvenc)
        pf.addRow("CRF / NVENC CQ", self.crf)
        pf.addRow("切镜阈值", self._hint_row(self.scene_threshold, "PySceneDetect 与 FFmpeg 共用"))
        pf.addRow("最短镜头（秒）", self.min_shot)
        pf.addRow("姿态", self.pose_quality)
        pf.addRow("抽帧间隔", self._hint_row(self.pose_every, "只在「抽帧」时使用"))
        pf.addRow("换脸全图检测间隔", self._hint_row(self.detect_every, "中间帧只检测轨迹框"))
        content_layout.addWidget(precision)

        dirs = QGroupBox("目录与本地生成")
        df = QFormLayout(dirs)
        df.addRow("作品根目录", self._browse_row(self.works, "选择作品根目录"))
        df.addRow("成片输出目录", self._browse_row(self.output, "选择输出目录"))
        df.addRow("模型目录", self._browse_row(self.models, "选择模型目录"))
        df.addRow("ffmpeg 目录", self._browse_row(self.ffmpeg, "选择 ffmpeg 所在目录（可空，默认走 PATH）"))
        df.addRow("ComfyUI HTTP", self.comfy)
        content_layout.addWidget(dirs)

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
        content_layout.addWidget(cloud)

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
        content_layout.addWidget(up)

        self._refresh_preset_label()
        self._refresh_named_preset_combo()
        # 手动改 CRF / 检测间隔任一处 -> 预设回到“自定义”。
        # 注意：_apply_export_preset 期间用 _applying_preset 屏蔽这条路。
        # 并发不接这条路（N28）：并发不在预设里，改并发不影响画质预设状态。
        self.crf.valueChanged.connect(self._on_preset_field_changed)
        self.detect_every.valueChanged.connect(self._on_preset_field_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))

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

    # -- 导出画质预设 ---------------------------------------------------------

    def _refresh_preset_label(self) -> None:
        if self._export_preset in EXPORT_PRESET_LABELS:
            self.preset_label.setText(f"当前：{EXPORT_PRESET_LABELS[self._export_preset]}")
        else:
            self.preset_label.setText("当前：自定义")

    def _apply_export_preset(self, name: str) -> None:
        values = EXPORT_PRESETS[name]
        self._applying_preset = True
        try:
            self.crf.setValue(int(values["crf"]))
            self.detect_every.setValue(int(values["detect_every"]))
            # N28: 预设不碰并发，self.concurrency 保持用户当前值。
            self._export_preset = name
        finally:
            self._applying_preset = False
        self._refresh_preset_label()

    def _on_preset_field_changed(self, _value: int) -> None:
        if self._applying_preset:
            return
        if self._export_preset is not None:
            self._export_preset = None
            self._refresh_preset_label()

    # -- 设置预设（命名快照） -------------------------------------------------

    def _refresh_named_preset_combo(self, select: str | None = None) -> None:
        names = self.service.settings_mgr.list_presets()
        self.named_preset_combo.clear()
        for name in names:
            self.named_preset_combo.addItem(name, name)
        if select and select in names:
            self.named_preset_combo.setCurrentIndex(self.named_preset_combo.findData(select))
        self.named_load_btn.setEnabled(bool(names))
        self.named_delete_btn.setEnabled(bool(names))

    def _save_named_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "保存为预设", "预设名字：")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            QMessageBox.warning(self, "名字为空", "预设名字不能为空。")
            return
        saved = self.service.settings_mgr.save_preset(
            name,
            confirm_overwrite=lambda n: QMessageBox.question(
                self, "覆盖预设", f"预设「{n}」已存在，是否覆盖？"
            )
            == QMessageBox.StandardButton.Yes,
        )
        if saved:
            self._refresh_named_preset_combo(select=name)

    def _load_named_preset(self) -> None:
        name = self.named_preset_combo.currentData()
        if not name:
            return
        try:
            loaded = self.service.settings_mgr.load_preset(str(name))
        except KeyError as exc:
            QMessageBox.warning(self, "找不到预设", str(exc))
            self._refresh_named_preset_combo()
            return
        self._refresh_from_settings(loaded)

    def _delete_named_preset(self) -> None:
        name = self.named_preset_combo.currentData()
        if not name:
            return
        answer = QMessageBox.question(
            self, "删除预设", f"确定删除预设「{name}」吗？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.settings_mgr.delete_preset(str(name))
        except KeyError as exc:
            QMessageBox.warning(self, "找不到预设", str(exc))
        self._refresh_named_preset_combo()

    def _refresh_from_settings(self, s) -> None:
        """加载命名预设后，让对话框控件跟着变（不只改内存里的值）。"""
        self._applying_preset = True
        try:
            self.burn.setChecked(bool(s.burn_subs))
            if getattr(s, "cover_mode", None) in {"blur", "bar", "off"}:
                self.cover_mode.setCurrentIndex(max(0, self.cover_mode.findData(s.cover_mode)))
            if getattr(s, "use_nvenc", None) is not None:
                self.nvenc.setChecked(bool(s.use_nvenc))
            if getattr(s, "export_crf", None) is not None:
                self.crf.setValue(int(s.export_crf))
            if getattr(s, "scene_threshold", None) is not None:
                self.scene_threshold.setValue(float(s.scene_threshold))
            if getattr(s, "min_shot_sec", None) is not None:
                self.min_shot.setValue(float(s.min_shot_sec))
            if getattr(s, "pose_quality", None):
                idx = self.pose_quality.findData(s.pose_quality)
                if idx >= 0:
                    self.pose_quality.setCurrentIndex(idx)
            if getattr(s, "pose_sample_every", None) is not None:
                self.pose_every.setValue(max(1, int(s.pose_sample_every)))
            if getattr(s, "detect_every", None) is not None:
                self.detect_every.setValue(max(1, int(s.detect_every)))
            # max_cps 没有对话框控件：值已由 load_preset 写进设置，只是不显示。
            preset = getattr(s, "export_preset", None)
            if preset in EXPORT_PRESETS:
                # 预设只管画质两项；并发不在预设里，不动 self.concurrency。
                self.crf.setValue(int(EXPORT_PRESETS[preset]["crf"]))
                self.detect_every.setValue(int(EXPORT_PRESETS[preset]["detect_every"]))
                self._export_preset = preset
            else:
                self._export_preset = None
        finally:
            self._applying_preset = False
        self._refresh_preset_label()

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
            cover_mode=self.cover_mode.currentData(),
            use_nvenc=self.nvenc.isChecked(),
            export_crf=int(self.crf.value()),
            export_preset=self._export_preset,
            scene_threshold=float(self.scene_threshold.value()),
            min_shot_sec=float(self.min_shot.value()),
            pose_quality=self.pose_quality.currentData(),
            pose_sample_every=int(self.pose_every.value()),
            detect_every=int(self.detect_every.value()),
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
