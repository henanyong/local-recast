"""CUDA VRAM reporting, model teardown, and CPU thread caps for 12600KF."""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

log = logging.getLogger("recast")


def apply_thread_caps(cpu_threads: int = 4, opencv_threads: int = 2) -> None:
    """Leave P-core / E-core headroom so a Gradio UI or desktop stays responsive."""
    cpu_threads = max(1, int(cpu_threads))
    opencv_threads = max(1, int(opencv_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    try:
        import torch

        torch.set_num_threads(cpu_threads)
    except Exception:  # pragma: no cover - torch may be missing in doctor --shallow
        pass
    try:
        import cv2

        cv2.setNumThreads(opencv_threads)
    except Exception:
        pass
    log.info("CPU 线程上限: torch/OMP=%s, OpenCV=%s", cpu_threads, opencv_threads)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def requested_device(preferred: str) -> str:
    if preferred == "cuda" and not cuda_available():
        log.warning("配置要求 CUDA，但当前 PyTorch 看不到 GPU，已回退到 CPU（会慢很多）。")
        return "cpu"
    return preferred


def vram_report(tag: str = "") -> str:
    """Human-readable VRAM line. Safe to call when CUDA is absent."""
    prefix = f"[{tag}] " if tag else ""
    try:
        import torch

        if not torch.cuda.is_available():
            return f"{prefix}CUDA 不可用"
        free_b, total_b = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free = free_b / (1024**3)
        total = total_b / (1024**3)
        name = torch.cuda.get_device_name(0)
        msg = (
            f"{prefix}{name}: 已分配 {allocated:.2f} GB, "
            f"已保留 {reserved:.2f} GB, 空闲 {free:.2f}/{total:.2f} GB"
        )
        log.info(msg)
        return msg
    except Exception as exc:
        msg = f"{prefix}无法读取显存: {exc}"
        log.info(msg)
        return msg


def release_cuda(*objects: Any) -> None:
    """Collect garbage and return already-freed PyTorch CUDA blocks.

    Callers must drop their own references *before* calling this function
    (``self._model = None`` or ``model = None``). ``del obj`` here only
    unbinds the local parameter name, so a live caller reference would keep
    the object (and its VRAM) alive through ``gc.collect()``.

    ``torch.cuda.empty_cache()`` only returns *already freed* PyTorch cache
    blocks to the driver. It has no effect on ONNX Runtime, InsightFace, or
    CTranslate2 (faster-whisper) sessions; those are released when their
    Python objects are destroyed.
    """
    for obj in objects:
        try:
            if hasattr(obj, "cpu"):
                obj.cpu()
        except Exception:
            pass
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    vram_report("释放后")
