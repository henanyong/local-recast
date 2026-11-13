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
    # Wall-clock cap for the online model download after a local cache miss.
    # The download runs in a worker thread; past this many seconds the stage
    # fails fast instead of hanging on an unreachable huggingface.co.
    download_timeout_sec: float = 60.0


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
    model: str = "buffalo_l"
    det_size: int = 512
    cluster_threshold: float = 0.45
    max_faces_per_track: int = 24
    sample_every: int = 5
    # Face must clear this IoU against the head zone before it is bound to a track.
    min_face_iou: float = 0.2
    # Manual review queue: a face crop whose file score (see
    # recast/core/review.py score_sample_file) is below this goes to人工审核.
    # 0.5 keeps "usable but not great" (a sharp 60x60 crop scores 0.5) and
    # flags small or blurry crops (a blurry 60x60 scores ~0.26).
    review_min_score: float = 0.5
    # Post-cluster validation: after AgglomerativeClustering, each cluster
    # is re-checked member by member (see split_cluster_outliers).
    # cluster_max_mean_distance is the safety net on the mean pairwise
    # cosine distance. It equals cluster_threshold on purpose: average
    # linkage only merges groups whose mean distance is at/below the
    # threshold, so a finished cluster's mean distance is already at/below
    # it in theory — this catches numerical drift and future linkage
    # changes, not ordinary clusters.
    cluster_max_mean_distance: float = 0.45
    # A member whose mean cosine similarity to the other members is below
    # this is split out as an unconfirmed fragment for manual review.
    # 1 - cluster_threshold = 0.55: below the similarity the clustering
    # itself demands when it decides two groups belong together, so by the
    # clustering's own standard this member does not belong to the group.
    # Same-person InsightFace similarity is usually > 0.6, different
    # people < 0.3; 0.55 sits between them.
    cluster_min_member_similarity: float = 0.55
    # Cross-episode merge: before creating a new character folder, the new
    # cluster's embedding is compared with every existing character
    # (see recast/core/merge.py). At or above this cosine similarity it
    # merges into that character instead of creating a new folder.
    # Same person usually > 0.6, different people < 0.3; 0.6 errs on the
    # side of NOT merging (a missed merge is fixed by one click in the
    # review queue, a wrong merge mixes two people into one folder).
    merge_similarity: float = 0.6
    # Sharpness floor (variance of the Laplacian, same scale as
    # recast/core/review.py _laplacian_variance: sharp photos are
    # typically > 150, blurry ones < 50). A sample below this never joins
    # the cluster core; it is recorded with reason "模糊" for review.
    min_sharpness: float = 150.0


@dataclass
class StitchConfig:
    # Face vote (best-pair over the two tracklets' sample sets): the best
    # pair must reach face_best_min and the median of the top
    # face_top_pairs pairs must reach face_median_min. Best pair below
    # face_veto vetoes the merge outright (different people).
    face_best_min: float = 0.6
    face_median_min: float = 0.5
    face_veto: float = 0.3
    # Body appearance (CLIP ViT-B/32) cosine vote.
    body_min: float = 0.75
    # Costume HSV-histogram cosine vote.
    costume_min: float = 0.8
    # Votes needed to merge (of the three: face / body / costume).
    min_votes: int = 2
    # How many top face pairs the median is taken over.
    face_top_pairs: int = 3
    # Max face samples per tracklet used in best-pair scoring.
    face_max_samples: int = 8


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
    # Minimum cosine similarity for a confident reference-to-cluster match.
    # Independent of identity.cluster_threshold, which is only for clustering.
    min_similarity: float = 0.55


@dataclass
class MattingConfig:
    # rembg model name (u2net default). Empty models_dir = rembg default cache.
    model: str = "u2net"
    models_dir: str = ""


@dataclass
class DiarizationConfig:
    # Audio is resampled to this rate before splitting.
    sample_rate: int = 16000
    min_speech_sec: float = 0.3
    min_silence_sec: float = 0.3
    merge_gap_sec: float = 0.5
    max_speakers: int = 8


@dataclass
class CostumeConfig:
    # Tagger: "clip" (local CLIP zero-shot, needs torch + clip) or
    # "heuristic" (colour-only, no model). Missing CLIP is a hard error.
    tagger: str = "clip"
    clip_model: str = "ViT-B/32"
    models_dir: str = ""


@dataclass
class Model3dConfig:
    # Auto-generation backend: "triposg" (needs CUDA GPU >= 8GB VRAM).
    backend: str = "triposg"
    # Directory holding a TripoSG checkout (its scripts/inference_triposg.py).
    # Empty = not installed.
    triposg_repo: str = ""
    # Python to run TripoSG with. Empty = the interpreter running local-recast.
    # Set this to the TripoSG conda env's python when TripoSG lives in its
    # own environment (recommended: conda create -n tripoSG python=3.10).
    triposg_python: str = ""


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
    # Cover the original video's burned-in subtitle band before burning new
    # subtitles: "blur" (default) | "bar" (black bar) | "off" (never cover).
    # "off" is translated by the caller (see cover_options_for_compose);
    # compose.run itself only knows blur / bar / inpaint.
    cover_mode: str = "blur"


@dataclass
class RecastConfig:
    device: str = "cuda"
    vram_gb: int = 8
    cpu_threads: int = 4
    opencv_threads: int = 2
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    replace: ReplaceConfig = field(default_factory=ReplaceConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    compose: ComposeConfig = field(default_factory=ComposeConfig)
    matting: MattingConfig = field(default_factory=MattingConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    costume: CostumeConfig = field(default_factory=CostumeConfig)
    model3d: Model3dConfig = field(default_factory=Model3dConfig)
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
            "stitch",
            "pose",
            "replace",
            "scene",
            "compose",
            "matting",
            "diarization",
            "costume",
            "model3d",
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
            stitch=_merge(StitchConfig, raw.get("stitch"), where="stitch"),
            pose=_merge(PoseConfig, raw.get("pose"), where="pose"),
            replace=_merge(ReplaceConfig, raw.get("replace"), where="replace"),
            scene=_merge(SceneConfig, raw.get("scene"), where="scene"),
            compose=_merge(ComposeConfig, raw.get("compose"), where="compose"),
            matting=_merge(MattingConfig, raw.get("matting"), where="matting"),
            diarization=_merge(DiarizationConfig, raw.get("diarization"), where="diarization"),
            costume=_merge(CostumeConfig, raw.get("costume"), where="costume"),
            model3d=_merge(Model3dConfig, raw.get("model3d"), where="model3d"),
            path=cfg_path,
            raw=raw,
        )
        return cfg


# 导出画质预设（设计 8.5 / 10.5）：一键切 CRF + 换脸全图检测间隔。
# 预设不碰并发（N28）：并发是硬件风险项，只由用户在设置里改；
# "balanced" 的两值就是现有默认值（yaml CRF 18、ReplaceConfig.detect_every 4），
# 所以和现有默认没有冲突。
EXPORT_PRESETS: dict[str, dict[str, int]] = {
    "high": {"crf": 16, "detect_every": 2},
    "balanced": {"crf": 18, "detect_every": 4},
    "fast": {"crf": 23, "detect_every": 8},
}
EXPORT_PRESET_LABELS: dict[str, str] = {
    "high": "高清",
    "balanced": "均衡",
    "fast": "快速预览",
}


def apply_runtime_settings(cfg: RecastConfig, settings: Any | None) -> RecastConfig:
    """Copy optional desktop overrides onto a loaded preset. Unset values keep the yaml.

    Export preset rule (design 8.5/10.5): when ``settings.export_preset``
    names a known preset, the preset wins over the individual CRF /
    detect_every fields — the whole point of one click is that the two
    values move together. A preset has no "custom" state of its own:
    editing any single field afterwards clears ``export_preset`` back to
    None (see ``SettingsManager.update`` and the settings dialog, which
    both do that); with ``export_preset`` None the individual fields apply.
    Presets never touch concurrency (N28): it stays whatever the user set
    in settings, so switching quality presets cannot silently change how
    many GPU tasks run in parallel.
    """
    if settings is None:
        return cfg
    preset = EXPORT_PRESETS.get(getattr(settings, "export_preset", None) or "")
    if preset is not None:
        cfg.compose.crf = int(preset["crf"])
        cfg.replace.detect_every = max(1, int(preset["detect_every"]))
    else:
        crf = getattr(settings, "export_crf", None)
        if crf not in (None, "", 0):
            cfg.compose.crf = int(crf)
        detect = getattr(settings, "detect_every", None)
        if detect not in (None, "", 0):
            cfg.replace.detect_every = max(1, int(detect))
    nvenc = getattr(settings, "use_nvenc", None)
    if nvenc is not None:
        cfg.compose.use_nvenc = bool(nvenc)
    cover = getattr(settings, "cover_mode", None)
    if cover in {"blur", "bar", "off"}:
        cfg.compose.cover_mode = cover
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
