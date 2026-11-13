"""Which face backend will run, and whether you are allowed to ship it.

The swap itself already lives in :mod:`recast.stages.swap_preview` — it either
runs InsightFace's ``inswapper_128`` or falls back to an aligned overlay that
is explicitly *not* a swap. This module adds the one thing that was missing:
an inventory with licences, and a gate that a delivery run can close.

Why the gate matters here specifically: every practical open face-swap weight
is non-commercial. ``inswapper_128.onnx`` is InsightFace's, and InsightFace
sells model licences separately from its MIT code. ``hyperswap`` (what
FaceFusion ships by default) is ResearchRAIL. Building the feature and only
discovering the licence at delivery time is the expensive order of events, so
``allow_noncommercial=False`` is a first-class argument, not a footnote.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")


class FaceSwapError(RuntimeError):
    """The requested face backend is unavailable, or not licensed for this run."""


def describe_face_backends(cfg: Any = None) -> dict[str, dict[str, Any]]:
    """Face backends, their availability, and their licence posture.

    ``final`` says whether the backend produces a real swap. The overlay
    backend is listed, not hidden: it is useful for framing checks and it is
    what actually runs on a machine without weights, but calling it a swap
    would make ``is_final`` a lie.
    """
    from recast.stages.swap_preview import inswapper_status

    status = inswapper_status(cfg)
    return {
        "inswapper": {
            "available": bool(status.get("available")),
            "license": "非商用（InsightFace inswapper_128，商用需单独授权）",
            "final": True,
            "reason": str(status.get("reason") or ""),
        },
        "overlay": {
            "available": True,
            "license": "不涉及模型（对齐叠加预览）",
            "final": False,
            "reason": "不是换脸，只是把参考脸对齐叠上去，用来检查构图和位置",
        },
    }


def select_face_backend(
    *, allow_noncommercial: bool = True, cfg: Any = None
) -> tuple[str, str]:
    """(backend, reason). Prefers a real swap, refuses it for delivery runs."""
    table = describe_face_backends(cfg)
    if table["inswapper"]["available"]:
        if allow_noncommercial:
            return "inswapper", ""
        return (
            "overlay",
            "真换脸只有非商用模型（inswapper_128），交付跑已禁用；"
            "要商用换脸需购买 InsightFace 模型授权，或换可商用方案。",
        )
    return "overlay", table["inswapper"]["reason"]


def swap_frame(
    frame_bgr: Any,
    reference_path: str | Path,
    *,
    allow_noncommercial: bool = True,
    cfg: Any = None,
    tmp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Swap the face in one frame and report honestly what happened.

    Returns ``{"image", "backend", "is_final", "note", "license"}``. When only
    the overlay is available the frame still comes back (so a pipeline can
    keep going) with ``is_final=False`` and the reason in ``note``.
    """
    import cv2
    import numpy as np

    from recast.stages.swap_preview import preview_swap

    backend, reason = select_face_backend(allow_noncommercial=allow_noncommercial, cfg=cfg)
    table = describe_face_backends(cfg)
    reference = Path(reference_path)
    if not reference.is_file():
        raise FaceSwapError(f"找不到参考脸：{reference}")

    work = Path(tmp_dir) if tmp_dir else Path.cwd() / ".face_swap_tmp"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "frame_in.png"
    ok, buf = cv2.imencode(".png", np.asarray(frame_bgr))
    if not ok:
        raise FaceSwapError("帧写不进临时文件，换脸中止")
    buf.tofile(str(source))

    result = preview_swap(source, reference, cfg=cfg, out_path=work / "frame_out.png")
    notes = [n for n in (reason, result.get("note")) if n]
    return {
        "image": result["image"],
        "backend": str(result.get("method") or backend),
        "is_final": bool(result.get("is_final")),
        "license": table[backend]["license"],
        "note": "；".join(notes),
    }
