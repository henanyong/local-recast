"""Persisted application settings (hardware tier, directories, concurrency, optional cloud / upload)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from recast.config import DEFAULT_CONFIG, PROJECT_ROOT
from recast.core.constants import DEFAULT_CONCURRENCY, MAX_CONCURRENCY, TIER_BASIC, TIER_MAIN, TIER_RECO
from recast.store.db import Store


def _cfg_rel(name: str) -> str:
    return str((PROJECT_ROOT / "configs" / name).as_posix())


@dataclass
class CloudChannel:
    name: str
    configured: bool = False
    enabled: bool = False
    status: str = "未配置"
    base_url: str = ""
    model: str = ""


@dataclass
class AppSettings:
    hardware_tier: str = TIER_MAIN  # basic | 4060ti-8g | 4060ti-16g
    config_path: str = field(default_factory=lambda: str(DEFAULT_CONFIG))
    works_root: str = ""
    output_root: str = field(default_factory=lambda: str(PROJECT_ROOT / "outputs"))
    models_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "models"))
    ffmpeg_dir: str = ""
    concurrency: int = DEFAULT_CONCURRENCY
    first_run_done: bool = False
    # N21：审核队列清空自动跳转提示，用户勾了 [不再提示] 后不再弹。
    review_jump_muted: bool = False
    lang: str = "auto"
    burn_subs: bool = False
    with_pose: bool = False
    # Upload: default OFF. User may enable copy-to-folder / webhook / command.
    auto_upload: bool = False
    auto_upload_enabled: bool = True  # the control is available in V1.0
    upload_mode: str = "folder"  # folder | webhook | command
    upload_folder: str = ""
    upload_webhook: str = ""
    upload_command: str = ""
    # Cloud: default OFF. Secrets live in SecretStore, not here.
    cloud_enabled: bool = False
    comfyui_url: str = ""
    default_pack: str = ""
    max_cps: float = 4.0
    # None follows the yaml preset. The settings dialog writes an explicit value.
    export_crf: int | None = None
    use_nvenc: bool | None = None
    # None follows the yaml preset (ComposeConfig.cover_mode). The settings
    # dialog writes "blur" | "bar" | "off".
    cover_mode: str | None = None
    # Export quality preset: "high" | "balanced" | "fast", None = 未选/自定义。
    # 选了预设就按预设（见 apply_runtime_settings）；手动改 CRF / 检测间隔 /
    # 并发任一字段后，本字段自动回到 None（见 SettingsManager.update）。
    export_preset: str | None = None
    scene_threshold: float | None = None
    min_shot_sec: float | None = None
    pose_quality: str = ""
    pose_sample_every: int | None = None
    detect_every: int | None = None
    cloud_volcengine: CloudChannel = field(default_factory=lambda: CloudChannel("火山引擎"))
    cloud_deepseek: CloudChannel = field(default_factory=lambda: CloudChannel("DeepSeek"))
    cloud_openai: CloudChannel = field(default_factory=lambda: CloudChannel("OpenAI 兼容"))

    def config_file(self) -> str:
        if self.hardware_tier == TIER_RECO:
            return str(PROJECT_ROOT / "configs" / "4060ti-16g.yaml")
        if self.hardware_tier in {TIER_MAIN, TIER_BASIC}:
            return str(PROJECT_ROOT / "configs" / "4060ti-8g.yaml")
        return self.config_path or str(DEFAULT_CONFIG)

    def clamped_concurrency(self) -> int:
        return max(1, min(MAX_CONCURRENCY, int(self.concurrency or 1)))

    def is_basic_tier(self) -> bool:
        return self.hardware_tier == TIER_BASIC

    def generate_heavy_allowed(self) -> bool:
        return not self.is_basic_tier()

    def l2_drive_allowed(self) -> bool:
        return not self.is_basic_tier()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppSettings":
        raw = dict(data or {})
        for key in ("cloud_volcengine", "cloud_deepseek", "cloud_openai"):
            if isinstance(raw.get(key), dict):
                raw[key] = CloudChannel(
                    **{k: v for k, v in raw[key].items() if k in CloudChannel.__dataclass_fields__}
                )
        allowed = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        settings = cls(**allowed)
        # Defaults stay off if missing; do not force-disable a user who turned them on.
        if not settings.cloud_volcengine.configured:
            settings.cloud_volcengine.status = settings.cloud_volcengine.status or "未配置"
        if not settings.cloud_deepseek.configured:
            settings.cloud_deepseek.status = settings.cloud_deepseek.status or "未配置"
        if not settings.cloud_openai.configured:
            settings.cloud_openai.status = settings.cloud_openai.status or "未配置"
        return settings


class SettingsManager:
    KEY = "app_settings"
    PRESETS_KEY = "setting_presets"
    # Fields a named setting preset snapshots (design: settings presets).
    # Deliberately the tunable export/quality fields only — not directories,
    # tiers, upload targets or cloud state.
    PRESET_FIELDS: tuple[str, ...] = (
        "export_crf",
        "use_nvenc",
        "cover_mode",
        "scene_threshold",
        "min_shot_sec",
        "pose_quality",
        "pose_sample_every",
        "detect_every",
        "export_preset",
        "burn_subs",
        "max_cps",
    )
    # The two fields an export preset drives. Writing any one of them
    # without naming export_preset in the same call means "the user edited
    # a single value by hand", so the preset drops back to custom (None).
    # N28: "concurrency" is NOT preset-driven anymore — presets never touch it.
    PRESET_DRIVEN_FIELDS: tuple[str, ...] = ("export_crf", "detect_every")

    def __init__(self, store: Store) -> None:
        self.store = store
        self._current = AppSettings.from_dict(store.get_setting(self.KEY, {}) or {})

    @property
    def current(self) -> AppSettings:
        return self._current

    def save(self, settings: AppSettings | None = None) -> AppSettings:
        self._current = settings or self._current
        self.store.set_setting(self.KEY, self._current.to_dict())
        return self._current

    def update(self, **fields: Any) -> AppSettings:
        data = self._current.to_dict()
        data.update(fields)
        if "export_preset" not in fields and any(
            k in fields for k in self.PRESET_DRIVEN_FIELDS
        ):
            # Manual edit of one preset-driven field -> custom, not a preset.
            data["export_preset"] = None
        return self.save(AppSettings.from_dict(data))

    # -- named setting presets (snapshots) -----------------------------------

    def list_presets(self) -> list[str]:
        stored = self.store.get_setting(self.PRESETS_KEY, {}) or {}
        if not isinstance(stored, dict):
            return []
        return sorted(str(name) for name in stored)

    def save_preset(
        self,
        name: str,
        *,
        confirm_overwrite: Callable[[str], bool] | None = None,
    ) -> bool:
        """Snapshot the current tunable settings under ``name``.

        Returns True when saved, False when the user declined an overwrite.
        Empty names raise ValueError. When ``name`` already exists and
        ``confirm_overwrite`` is given, it is called with the name and must
        return True to overwrite; without a callback an existing name is
        overwritten directly (the dialog passes a QMessageBox callback).
        """
        clean = (name or "").strip()
        if not clean:
            raise ValueError("预设名字不能为空。")
        stored = self.store.get_setting(self.PRESETS_KEY, {}) or {}
        if not isinstance(stored, dict):
            stored = {}
        if clean in stored and confirm_overwrite is not None:
            if not confirm_overwrite(clean):
                return False
        snapshot = {
            field: getattr(self._current, field) for field in self.PRESET_FIELDS
        }
        stored[clean] = snapshot
        self.store.set_setting(self.PRESETS_KEY, stored)
        return True

    def load_preset(self, name: str) -> AppSettings:
        """Apply a saved preset to the current settings. Unknown name raises KeyError."""
        stored = self.store.get_setting(self.PRESETS_KEY, {}) or {}
        if not isinstance(stored, dict) or name not in stored:
            raise KeyError(f"找不到设置预设：{name}")
        snapshot = stored[name]
        if not isinstance(snapshot, dict):
            raise KeyError(f"设置预设已损坏：{name}")
        fields = {k: v for k, v in snapshot.items() if k in self.PRESET_FIELDS}
        return self.update(**fields)

    def delete_preset(self, name: str) -> None:
        """Delete a saved preset. Unknown name raises KeyError."""
        stored = self.store.get_setting(self.PRESETS_KEY, {}) or {}
        if not isinstance(stored, dict) or name not in stored:
            raise KeyError(f"找不到设置预设：{name}")
        del stored[name]
        self.store.set_setting(self.PRESETS_KEY, stored)
