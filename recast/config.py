"""YAML config loading for 4060 Ti 8GB / 16GB presets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("recast")

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "4060ti-8g.yaml"


def _warn_unknown(keys: list[str], where: str) -> None:
    if keys:
        log.warning("配置忽略未知键%s: %s", f" ({where})" if where else "", ", ".join(keys))


def _merge(dc_cls: type, data: dict[str, Any] | None, *, where: str = "") -> Any:
    if not data:
        return dc_cls()
    allowed = {f.name for f in fields(dc_cls)}
    _warn_unknown([k for k in data if k not in allowed], where or dc_cls.__name__)
    return dc_cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class WhisperConfig:
    model: str = "small"
    compute_type: str = "float16"
    device: str = "cuda"
    beam_size: int = 1
    vad_filter: bool = True


@dataclass
class DetectConfig:
    model: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.4
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"


@dataclass
class IdentityConfig:
    model: str = "buffalo_s"
    det_size: int = 512
    cluster_threshold: float = 0.45
    max_faces_per_track: int = 24
    sample_every: int = 5


@dataclass
class PoseConfig:
    backend: str = "auto"
    mode: str = "lightweight"
    overlay: bool = True
    export_smpl_hook: bool = True


@dataclass
class ReplaceConfig:
    backend: str = "inswapper"
    model: str = "inswapper_128.onnx"
    model_url: str = (
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    )


@dataclass
class ComposeConfig:
    crf: int = 18
    audio: str = "original"
    burn_subs: bool = False


@dataclass
class RecastConfig:
    device: str = "cuda"
    vram_gb: int = 8
    cpu_threads: int = 4
    opencv_threads: int = 2
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    replace: ReplaceConfig = field(default_factory=ReplaceConfig)
    compose: ComposeConfig = field(default_factory=ComposeConfig)
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "RecastConfig":
        cfg_path = Path(path) if path else DEFAULT_CONFIG
        if not cfg_path.is_file():
            raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known_top = {
            "device",
            "vram_gb",
            "cpu_threads",
            "opencv_threads",
            "whisper",
            "detect",
            "identity",
            "pose",
            "replace",
            "compose",
        }
        _warn_unknown([k for k in raw if k not in known_top], cfg_path.name)
        cfg = cls(
            device=str(raw.get("device", "cuda")),
            vram_gb=int(raw.get("vram_gb", 8)),
            cpu_threads=int(raw.get("cpu_threads", 4)),
            opencv_threads=int(raw.get("opencv_threads", 2)),
            whisper=_merge(WhisperConfig, raw.get("whisper"), where="whisper"),
            detect=_merge(DetectConfig, raw.get("detect"), where="detect"),
            identity=_merge(IdentityConfig, raw.get("identity"), where="identity"),
            pose=_merge(PoseConfig, raw.get("pose"), where="pose"),
            replace=_merge(ReplaceConfig, raw.get("replace"), where="replace"),
            compose=_merge(ComposeConfig, raw.get("compose"), where="compose"),
            path=cfg_path,
            raw=raw,
        )
        return cfg


def resolve_config_path(value: str | Path | None) -> Path:
    if value is None:
        return DEFAULT_CONFIG
    path = Path(value)
    if path.is_file():
        return path
    alt = PROJECT_ROOT / value
    if alt.is_file():
        return alt
    named = PROJECT_ROOT / "configs" / path.name
    if named.is_file():
        return named
    raise FileNotFoundError(f"找不到配置文件: {value}")
