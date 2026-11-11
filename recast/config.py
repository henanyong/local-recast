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
    # Drop a segment when Whisper's no-speech probability is above this.
    no_speech_prob: float = 0.6
    # How far a word edge may move to meet a shot cut. Audio is never sped up.
    edge_snap: float = 0.35


@dataclass
class DetectConfig:
    model: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.4
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    # Fill missing boxes across gaps no longer than this. Longer holes stay empty.
    max_gap_frames: int = 10
    # The per-frame detection log is large. Off by default; tracks still keep their boxes.
    keep_raw_frames: bool = False


@dataclass
class IdentityConfig:
    model: str = "buffalo_s"
    det_size: int = 512
    cluster_threshold: float = 0.45
    max_faces_per_track: int = 24
    sample_every: int = 5
    # Face must clear this IoU against the head zone before it is bound to a track.
    min_face_iou: float = 0.2


@dataclass
class PoseConfig:
    backend: str = "auto"
    mode: str = "lightweight"
    overlay: bool = True
    export_smpl_hook: bool = True
    # accuracy: every frame plus smoothing. speed: sample every sample_every frames.
    quality: str = "accuracy"
    sample_every: int = 2
    smooth: bool = True
    min_pose_iou: float = 0.3


@dataclass
class ReplaceConfig:
    backend: str = "inswapper"
    model: str = "inswapper_128.onnx"
    model_url: str = (
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    )
    # Full-frame detection runs every N frames. Intermediate frames search the track crop.
    detect_every: int = 4
    # Weight of the current swapped face. 1 disables temporal blending.
    temporal_alpha: float = 0.62


@dataclass
class SceneConfig:
    # PySceneDetect content threshold. The FFmpeg fallback is calibrated from this
    # (27 -> scene score 0.30) so the two backends share one sensitivity.
    threshold: float = 27.0
    min_shot_sec: float = 0.4


@dataclass
class ComposeConfig:
    crf: int = 18
    use_nvenc: bool = True
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
    scene: SceneConfig = field(default_factory=SceneConfig)
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
            "scene",
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
            scene=_merge(SceneConfig, raw.get("scene"), where="scene"),
            compose=_merge(ComposeConfig, raw.get("compose"), where="compose"),
            path=cfg_path,
            raw=raw,
        )
        return cfg


def apply_runtime_settings(cfg: RecastConfig, settings: Any | None) -> RecastConfig:
    """Copy optional desktop overrides onto a loaded preset. Unset values keep the yaml."""
    if settings is None:
        return cfg
    crf = getattr(settings, "export_crf", None)
    if crf not in (None, "", 0):
        cfg.compose.crf = int(crf)
    nvenc = getattr(settings, "use_nvenc", None)
    if nvenc is not None:
        cfg.compose.use_nvenc = bool(nvenc)
    threshold = getattr(settings, "scene_threshold", None)
    if threshold not in (None, ""):
        cfg.scene.threshold = float(threshold)
    min_shot = getattr(settings, "min_shot_sec", None)
    if min_shot not in (None, ""):
        cfg.scene.min_shot_sec = float(min_shot)
    quality = getattr(settings, "pose_quality", None)
    if quality:
        cfg.pose.quality = str(quality)
    every = getattr(settings, "pose_sample_every", None)
    if every not in (None, "", 0):
        cfg.pose.sample_every = max(1, int(every))
    detect = getattr(settings, "detect_every", None)
    if detect not in (None, "", 0):
        cfg.replace.detect_every = max(1, int(detect))
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
