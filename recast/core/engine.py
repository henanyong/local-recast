"""Local engine status, hardware-tier detection, disk space, model presence."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from recast import __version__
from recast.config import RecastConfig, resolve_config_path
from recast.core.constants import ENG_IDLE, ENG_MISSING, ENG_RUNNING
from recast.core.settings import AppSettings
from recast.models.registry import status_report


def detect_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return None
    return None


def detect_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def suggest_tier(vram_gb: float | None, gpu_name: str | None = None) -> str:
    """Map detected hardware to basic / mainstream (4060ti-8g) / recommend (16g)."""
    if vram_gb is None and not gpu_name:
        return "basic"
    if vram_gb is not None and vram_gb >= 14:
        return "4060ti-16g"
    return "4060ti-8g"


def capability_of(tier: str) -> dict:
    basic = tier == "basic"
    return {
        "tier": tier,
        "l2_drive": not basic,
        "generate_heavy": not basic,
        "replace": not basic,  # L1 swap needs a GPU in practice; CPU is possible but very slow
        "animatic": True,
        "extract": True,
        "message": (
            "当前是基础档（无独立 GPU）。切镜、OCR、ASR、拼接导出可用；"
            "L2 头像驱动和 ComfyUI 等重生成已禁用，将使用「分镜预览成片」离线回退。"
            if basic
            else ""
        ),
    }


def disk_free_gb(path: str | Path) -> float | None:
    try:
        target = Path(path)
        if not target.exists():
            target = target.parent if target.parent.exists() else Path.cwd()
        usage = shutil.disk_usage(str(target))
        return usage.free / (1024**3)
    except Exception:
        return None


def ffmpeg_status(ffmpeg_dir: str = "") -> dict[str, Any]:
    from recast.ffmpeg import set_ffmpeg_dir

    if ffmpeg_dir:
        set_ffmpeg_dir(ffmpeg_dir)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    # also honour override
    try:
        from recast import ffmpeg as ffm

        if ffmpeg_dir:
            exe = Path(ffmpeg_dir) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if exe.is_file():
                ffmpeg = str(exe)
            probe = Path(ffmpeg_dir) / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
            if probe.is_file():
                ffprobe = str(probe)
    except Exception:
        pass
    return {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "ok": bool(ffmpeg and ffprobe)}


def module_status() -> dict[str, bool]:
    names = {
        "torch": "torch",
        "faster_whisper": "faster_whisper",
        "ultralytics": "ultralytics",
        "insightface": "insightface",
        "onnxruntime": "onnxruntime",
        "cv2": "cv2",
        "scenedetect": "scenedetect",
        "rapidocr": "rapidocr",
        "PySide6": "PySide6",
    }
    out: dict[str, bool] = {}
    for label, mod in names.items():
        try:
            if mod == "rapidocr":
                from recast.extract.ocr import ocr_available

                out[label] = ocr_available()[0]
            else:
                __import__(mod)
                out[label] = True
        except Exception:
            out[label] = False
    return out


_HEAVY_TTL = 5.0
_heavy_cache: tuple[float, tuple, dict[str, Any]] | None = None
_comfy_cache: dict[str, tuple[float, bool]] = {}


def clear_engine_cache() -> None:
    """Drop cached GPU / config / ComfyUI probe results."""
    global _heavy_cache
    _heavy_cache = None
    _comfy_cache.clear()


def cached_comfy_available(base_url: str | None, *, ttl: float = _HEAVY_TTL, now: float | None = None) -> bool:
    """ComfyUI's probe uses a 1s timeout. Reuse the last answer for a few seconds."""
    from recast.generate.comfy import comfy_available

    key = base_url or ""
    clock = time.monotonic() if now is None else now
    hit = _comfy_cache.get(key)
    if hit is not None and clock - hit[0] < ttl:
        return hit[1]
    ok = bool(key) and comfy_available(base_url)
    _comfy_cache[key] = (clock, ok)
    return ok


def _heavy_snapshot(settings: AppSettings, cfg: RecastConfig | None) -> dict[str, Any]:
    global _heavy_cache
    key = (
        settings.hardware_tier,
        settings.ffmpeg_dir,
        settings.output_root,
        getattr(settings, "comfyui_url", "") or "",
        settings.config_file(),
        getattr(settings, "export_crf", None),
        getattr(settings, "use_nvenc", None),
        getattr(settings, "scene_threshold", None),
        getattr(settings, "min_shot_sec", None),
    )
    now = time.monotonic()
    if _heavy_cache is not None and _heavy_cache[1] == key and now - _heavy_cache[0] < _HEAVY_TTL:
        return _heavy_cache[2]
    vram = detect_vram_gb()
    gpu = detect_gpu_name()
    if cfg is None:
        try:
            from recast.config import apply_runtime_settings

            cfg = apply_runtime_settings(
                RecastConfig.from_yaml(resolve_config_path(settings.config_file())),
                settings,
            )
        except Exception:
            cfg = None
    models = status_report(cfg)
    ffm = ffmpeg_status(settings.ffmpeg_dir)
    mods = module_status()
    from recast.assets3d.drive import liveportrait_available

    comfy_url = getattr(settings, "comfyui_url", "") or ""
    scene = {
        "threshold": float(cfg.scene.threshold) if cfg is not None else 27.0,
        "min_shot_sec": float(cfg.scene.min_shot_sec) if cfg is not None else 0.4,
        "installed": "pyscenedetect" if mods.get("scenedetect") else "ffmpeg",
    }
    encode = {
        "crf": int(cfg.compose.crf) if cfg is not None else 18,
        "use_nvenc": bool(cfg.compose.use_nvenc) if cfg is not None else True,
    }
    data = {
        "vram_gb": vram,
        "gpu_name": gpu,
        "ffmpeg": ffm,
        "modules": mods,
        "models": models,
        "disk_free_gb": disk_free_gb(settings.output_root or "."),
        "comfyui": {"url": comfy_url, "ok": cached_comfy_available(comfy_url or None)},
        "liveportrait": liveportrait_available(),
        "scene": scene,
        "encode": encode,
    }
    _heavy_cache = (now, key, data)
    return data


def engine_snapshot(
    settings: AppSettings | None = None,
    *,
    running: bool = False,
    cfg: RecastConfig | None = None,
) -> dict[str, Any]:
    settings = settings or AppSettings()
    heavy = _heavy_snapshot(settings, cfg)
    vram = heavy["vram_gb"]
    gpu = heavy["gpu_name"]
    ffm = heavy["ffmpeg"]
    mods = heavy["modules"]
    models = heavy["models"]
    missing_models = [m["label"] for m in models if not m["present"] and m["key"] in {"inswapper_128"}]
    # inswapper absence is not fatal until replace; engine title uses 模型缺失 only when
    # a required runtime piece (ffmpeg / torch) is gone.
    if running:
        state = ENG_RUNNING
    elif not ffm["ok"] or not mods.get("torch"):
        state = ENG_MISSING
    else:
        state = ENG_IDLE
    cap = capability_of(settings.hardware_tier)
    if settings.hardware_tier == "basic":
        tier_label = "基础档（无独立 GPU）"
    elif settings.hardware_tier == "4060ti-16g":
        tier_label = "推荐档 · 4060 Ti 16GB"
    else:
        tier_label = "主流档 · 4060 Ti 8GB（推荐）"
    if vram is not None and vram < 10 and settings.hardware_tier == "4060ti-16g":
        tier_label += "（当前卡看起来不是 16GB）"
    if vram is None and not gpu and settings.hardware_tier != "basic":
        tier_label += "（未检测到独显，建议改为基础档）"

    return {
        "product": "本地再铸",
        "version": __version__,
        "state": state,
        "running": running,
        "hardware_tier": settings.hardware_tier,
        "hardware_label": tier_label,
        "capability": cap,
        "gpu_name": gpu,
        "vram_gb": vram,
        "python": sys.version.split()[0],
        "ffmpeg": ffm,
        "modules": mods,
        "models": models,
        "missing_optional": missing_models,
        "disk_free_gb": heavy["disk_free_gb"],
        "cuda_ok": bool(mods.get("torch") and (vram is not None)),
        "comfyui": heavy["comfyui"],
        "liveportrait": heavy["liveportrait"],
        "scene": heavy["scene"],
        "encode": heavy["encode"],
        "cloud_enabled": bool(getattr(settings, "cloud_enabled", False)),
        "auto_upload": bool(getattr(settings, "auto_upload", False)),
    }


def first_run_report(settings: AppSettings) -> dict[str, Any]:
    snap = engine_snapshot(settings)
    issues: list[str] = []
    fixes: list[str] = []
    if not snap["ffmpeg"]["ok"]:
        issues.append("找不到 ffmpeg / ffprobe。")
        fixes.append("请安装 ffmpeg 并把 bin 目录加入 PATH，或在设置里指定目录。Windows 可从 https://www.gyan.dev/ffmpeg/builds/ 下载。")
    if snap["vram_gb"] is None and not snap.get("gpu_name"):
        issues.append("未检测到独立 GPU，已按基础档处理（不会崩溃）。")
        fixes.append("切镜 / 字幕 / 分镜预览成片仍可运行。人物替换、L2 驱动和 ComfyUI 重生成已禁用。若这是目标电脑，请安装 NVIDIA 驱动和 PyTorch CUDA 轮子后改回主流档。")
    elif snap["vram_gb"] is not None and snap["vram_gb"] < 10 and settings.hardware_tier == "4060ti-16g":
        issues.append("当前显存看起来不足 16GB，建议改用 4060 Ti 8GB 配置。")
        fixes.append("在设置里把硬件档改成「主流档 · 4060 Ti 8GB」。")
    free = snap.get("disk_free_gb")
    if free is not None and free < 5:
        issues.append(f"输出目录所在磁盘剩余约 {free:.1f} GB，建议至少预留数 GB。")
        fixes.append("请清理磁盘或把作品 / 输出目录改到空间更大的固态硬盘。空间过低时任务会失败并写明原因。")
    if not settings.works_root:
        issues.append("尚未选择作品根目录。")
        fixes.append("请在下方或设置里选择一个用于存放剧目和成片的文件夹。")
    mods = snap.get("modules") or {}
    if not mods.get("torch"):
        issues.append("未安装 PyTorch。")
        fixes.append("请运行 scripts\\install_windows.bat，或手动安装 CUDA 版 PyTorch。")
    missing_models = [m for m in (snap.get("models") or []) if not m.get("present") and m.get("key") == "inswapper_128"]
    if missing_models:
        issues.append("人物替换权重尚未下载（首次执行「04 人物替换」时会自动获取）。")
        fixes.append("保持网络畅通，或把 inswapper_128 放到模型目录后重新自检。")
    if not snap.get("liveportrait"):
        fixes.append("未安装 LivePortrait：L2「3D头像资产」仍可生成，但不能当场驱动。可选安装后再用。")
    comfy = snap.get("comfyui") or {}
    if not comfy.get("ok"):
        fixes.append("未检测到 ComfyUI HTTP 服务：按剧本生成将使用「分镜预览成片」离线回退，而不是完整图生视频。")
    return {
        "snapshot": snap,
        "issues": issues,
        "fixes": fixes,
        "ok": not issues,
        "suggested_tier": suggest_tier(snap.get("vram_gb"), snap.get("gpu_name")),
    }
