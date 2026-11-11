"""Persisted application settings (hardware tier, directories, concurrency, optional cloud / upload)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
        return self.save(AppSettings.from_dict(data))
