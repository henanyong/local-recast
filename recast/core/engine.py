"""Local engine status, hardware-tier detection, disk space, model presence."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import importlib.util

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
    """Report whether each heavy optional module is installed.

    No module is imported here (except the rapidocr branch, see below):
    presence is decided by ``sys.modules`` membership or
    ``importlib.util.find_spec`` (which only locates the module, it does not
    execute its code). This matters because ``module_status`` runs both on the
    main thread (video import) and on the 400ms snapshot refresh thread, and
    concurrent ``__import__`` collided with PySide6's shiboken import hook
    and corrupted the heap (SIGSEGV, see SEGFAULT diagnosis 2026-11-13).

    The result is cached process-wide in ``_module_status_cache``; use
    :func:`clear_engine_cache` to reset it. Cache hits return a *copy* so a
    caller mutating the dict cannot poison the cache.
    """
    global _module_status_cache
    if _module_status_cache is not None:
        return dict(_module_status_cache)
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
                # Kept as-is per the segfault diagnosis: ocr_available() does
                # not import any heavy module itself.
                from recast.extract.ocr import ocr_available

                out[label] = ocr_available()[0]
            elif mod in sys.modules:
                out[label] = True
            else:
                out[label] = importlib.util.find_spec(mod) is not None
        except Exception:
            out[label] = False
    _module_status_cache = out
    return dict(out)


# Process-wide cache for module_status(); find_spec is cheap, so clearing it
# in clear_engine_cache() is safe and gives tests a reset hook.
_module_status_cache: dict[str, bool] | None = None
# Serialises _heavy_snapshot() computation and all engine-cache writes, so at
# most one thread is ever inside detect_vram_gb()/detect_gpu_name()'s real
# ``import torch`` or any other heavy probe at a time. RLock because
# _heavy_snapshot() calls cached_comfy_available(), which takes the same lock.
_heavy_lock = threading.RLock()


_HEAVY_TTL = 5.0
_heavy_cache: tuple[float, tuple, dict[str, Any]] | None = None
_comfy_cache: dict[str, tuple[float, bool]] = {}


def clear_engine_cache() -> None:
    """Drop cached GPU / config / ComfyUI probe results."""
    global _heavy_cache, _module_status_cache
    with _heavy_lock:
        _heavy_cache = None
        _comfy_cache.clear()
        _module_status_cache = None


def cached_comfy_available(base_url: str | None, *, ttl: float = _HEAVY_TTL, now: float | None = None) -> bool:
    """ComfyUI's probe uses a 1s timeout. Reuse the last answer for a few seconds."""
    from recast.generate.comfy import comfy_available

    key = base_url or ""
    clock = time.monotonic() if now is None else now
    with _heavy_lock:
        hit = _comfy_cache.get(key)
        if hit is not None and clock - hit[0] < ttl:
            return hit[1]
    # Probe outside the lock (it has a 1s timeout); two threads racing here
    # both probe and write the same answer, which is benign.
    ok = bool(key) and comfy_available(base_url)
    with _heavy_lock:
        _comfy_cache[key] = (clock, ok)
    return ok


def _heavy_snapshot(settings: AppSettings, cfg: RecastConfig | None, *, light: bool = False) -> dict[str, Any]:
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
    if light and _heavy_cache is not None and _heavy_cache[1] == key:
        # Light mode (400ms timer tick): reuse the cached snapshot even
        # past the 5s TTL, as long as the key still matches -- the engine
        # panel values only change when a key field changes, and a key
        # change (or an empty cache) falls through to the normal locked
        # recompute below. Ignoring the TTL here is what keeps the timer
        # thread from re-probing heavy modules every 5 seconds.
        return _heavy_cache[2]
    # Fast path: no lock on a cache hit.
    if _heavy_cache is not None and _heavy_cache[1] == key and now - _heavy_cache[0] < _HEAVY_TTL:
        return _heavy_cache[2]
    # Slow path: serialise the computation (double-checked locking) so the
    # 400ms refresh thread and the main thread never probe heavy modules /
    # import torch at the same time. detect_vram_gb()/detect_gpu_name() still
    # do a real ``import torch`` -- that is required for VRAM probing, and
    # the lock guarantees only one thread is inside it at a time.
    with _heavy_lock:
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
    light: bool = False,
) -> dict[str, Any]:
    """Build the engine status dict.

    ``light=True`` reuses a warm ``_heavy_snapshot`` cache past its 5s TTL
    (key must still match); an empty or key-mismatched cache recomputes
    under the lock, so the first paint always gets real data. Default
    ``light=False`` keeps the previous behaviour.
    """
    settings = settings or AppSettings()
    heavy = _heavy_snapshot(settings, cfg, light=light)
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
