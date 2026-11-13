"""Single-frame swap preview for the character library (C 线：替换预览).

The library picks one captured frame, swaps the character's currently
assembled face onto it, and shows original vs preview side by side.

Real face swap first: the InswapperBackend from ``recast.stages.replace``
is reused when it can actually run. On a machine without InsightFace /
onnxruntime / inswapper weights (the usual CPU-only dev machine) that is
impossible, so the honest fallback is an aligned reference overlay: the
reference face is warped onto the detected face box of the source frame
and blended semi-transparently. The fallback is labelled loudly wherever
it is shown — it is NOT the final swap result, and must never be passed
off as one.

The body is never faked here: the dialog carries a placeholder note
pointing at the body re-render pilot (main window step ④) instead of
drawing a body.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("recast")

#: Shown in the dialog title line and under the preview whenever the
#: fallback ran. Keep the wording: tests and the UI both rely on it.
NOT_FINAL_TEXT = "参考对齐预览，不是最终换脸效果"
#: Body is out of scope until the re-render pilot lands; say so, don't draw one.
BODY_PLACEHOLDER_TEXT = "身体替换走身体重渲染试点，见主窗口第 ④ 步"
INSWAPPER_METHOD_TEXT = "InsightFace 真换脸预览（单帧）"
OVERLAY_METHOD_TEXT = "参考脸对齐叠加预览"
_OVERLAY_ALPHA = 0.62
_MIN_WEIGHT_BYTES = 10_000_000


def inswapper_status(cfg: Any | None = None) -> dict[str, Any]:
    """Can the real swap run here? Never downloads anything.

    Returns ``{"available": bool, "reason": str}``. ``reason`` is empty
    when available, otherwise the exact blocker (missing package or
    missing weights) so the UI can say why it fell back.
    """
    try:
        import insightface  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure blocks it
        return {
            "available": False,
            "reason": f"本机没有安装 InsightFace，跑不了真换脸（{exc}）。",
        }
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure blocks it
        return {
            "available": False,
            "reason": f"本机没有安装 onnxruntime，跑不了真换脸（{exc}）。",
        }
    if cfg is None:
        from recast.config import RecastConfig

        cfg = RecastConfig()
    from recast.config import PROJECT_ROOT

    named = Path(str(cfg.replace.model))
    candidates = [
        named,
        PROJECT_ROOT / "models" / named.name,
        Path.home() / ".insightface" / "models" / named.name,
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > _MIN_WEIGHT_BYTES:
            return {"available": True, "reason": ""}
    return {
        "available": False,
        "reason": (
            f"没有找到 {named.name} 权重（约 550MB），不会自动下载；"
            "装好权重后才能真换脸。"
        ),
    }


def _detect_face_box(image: np.ndarray) -> list[float] | None:
    """Largest frontal-face box (Haar), or None when nothing is found."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(boxes) == 0:
        return None
    x, y, w, h = max((tuple(b) for b in boxes), key=lambda b: int(b[2]) * int(b[3]))
    return [float(x), float(y), float(x + w), float(y + h)]


def _face_box_or_center(image: np.ndarray) -> list[float]:
    """Detected face box, else the central face-sized region (never None)."""
    box = _detect_face_box(image)
    if box is not None:
        return box
    height, width = image.shape[:2]
    return [
        width * 0.30,
        height * 0.20,
        width * 0.70,
        height * 0.70,
    ]


def aligned_overlay_preview(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reference face warped onto the source face box, blended at 62%.

    Pure OpenCV, no model: the reference face region is resized onto the
    source face box (expanded slightly) and composited under a feathered
    elliptical mask. The result reads as "this face, roughly there" — good
    enough to judge framing and identity direction, never the final swap.
    """
    src_box = _face_box_or_center(source)
    ref_box = _face_box_or_center(reference)
    x1, y1, x2, y2 = [float(v) for v in src_box]
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in ref_box]
    height, width = source.shape[:2]
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(width, int(x2)), min(height, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return source.copy()
    # Expand the target box a little so the overlay covers the whole face.
    cx, cy = (x1i + x2i) / 2.0, (y1i + y2i) / 2.0
    bw, bh = (x2i - x1i) * 1.12, (y2i - y1i) * 1.12
    x1i = max(0, int(cx - bw / 2))
    y1i = max(0, int(cy - bh / 2))
    x2i = min(width, int(cx + bw / 2))
    y2i = min(height, int(cy + bh / 2))
    target_w, target_h = x2i - x1i, y2i - y1i
    if target_w <= 0 or target_h <= 0:
        return source.copy()
    ref_crop = reference[max(0, ry1) : max(0, ry2), max(0, rx1) : max(0, rx2)]
    if ref_crop.size == 0:
        ref_crop = reference
    warped = cv2.resize(ref_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (target_w // 2, target_h // 2),
        (int(target_w * 0.46), int(target_h * 0.48)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    blur = max(3, min(target_w, target_h) // 12)
    if blur % 2 == 0:
        blur += 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    alpha = (mask * _OVERLAY_ALPHA)[..., None]
    out = source.copy()
    roi = out[y1i:y2i, x1i:x2i].astype(np.float32)
    out[y1i:y2i, x1i:x2i] = (warped.astype(np.float32) * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return out


def _read_image(path: str | Path) -> np.ndarray:
    """cv2.imread that survives non-ASCII Windows paths.

    ``cv2.imread`` goes through the CRT's narrow-char file API and returns None
    for any path outside the system code page — every character folder named in
    Chinese. Reading the bytes first and decoding from memory is the standard
    workaround and costs nothing.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"读不懂图片：{path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"读不懂图片：{path}")
    return image


_BACKEND_CACHE: dict[str, Any] = {}


def _cached_backend(cfg: Any) -> Any:
    """One loaded inswapper per config, reused across frames.

    ``build_backend`` + ``load`` reads a 554 MB ONNX file. Doing that per frame
    turned a 120-frame clip into six minutes of pure loading. The trade is that
    the weights stay resident — on an 8 GB card that is the right trade, and
    ``release_backend()`` exists for callers that need the memory back.
    """
    key = "|".join(
        [
            str(getattr(getattr(cfg, "replace", None), "backend", "")),
            str(getattr(getattr(cfg, "replace", None), "model", "")),
            str(getattr(getattr(cfg, "identity", None), "model", "")),
        ]
    )
    backend = _BACKEND_CACHE.get(key)
    if backend is None:
        from recast.stages.replace import build_backend

        backend = build_backend(cfg)
        backend.load(cfg, "cpu")
        _BACKEND_CACHE[key] = backend
    return backend


def release_backend() -> None:
    """Drop the cached swapper (and its VRAM) — call at the end of a batch."""
    for backend in list(_BACKEND_CACHE.values()):
        unload = getattr(backend, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:  # noqa: BLE001 - unload must never mask a result
                pass
    _BACKEND_CACHE.clear()


def _try_inswapper(
    frame: np.ndarray,
    source_path: str | Path,
    *,
    cfg: Any | None,
    backend: Any | None,
) -> np.ndarray | None:
    """One real swap via InswapperBackend, or None when it can't run.

    ``backend`` is injectable for tests (a stub with load /
    prepare_source / replace_frame). Any failure — missing packages,
    missing weights, no face found, unchanged frame — falls through to
    the overlay instead of raising.
    """
    try:
        if cfg is None:
            from recast.config import RecastConfig

            cfg = RecastConfig()
        owns_backend = backend is None
        if backend is None:
            backend = _cached_backend(cfg)
        else:
            backend.load(cfg, "cpu")
        try:
            backend.prepare_source(source_path)
            box = _detect_face_box(frame)
            if box is None and getattr(backend, "_app", None) is not None:
                faces = backend._app.get(frame) or []
                if faces:
                    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0) or 0.0))
                    box = [float(v) for v in best.bbox]
            if box is None:
                # No detector hit (synthetic frames, stub backends): hand the
                # backend the central face-sized box and let it decide.
                # A real backend that finds nothing returns the frame
                # unchanged, which the caller below treats as a miss.
                box = _face_box_or_center(frame)
            swapped = backend.replace_frame(frame, [box])
            if swapped is None or np.array_equal(swapped, frame):
                log.info("替换预览：真换脸没有改动画面，改用对齐叠加预览。")
                return None
            return swapped
        finally:
            if not owns_backend:
                # An injected backend belongs to the caller: leave it loaded so
                # a clip run does not reload 554 MB per frame.
                unload = getattr(backend, "unload", None)
                if callable(unload):
                    try:
                        unload()
                    except Exception:  # noqa: BLE001 - unload must not mask the result
                        pass
    except Exception as exc:  # noqa: BLE001 - any swap failure falls back
        log.info("替换预览：真换脸跑不起来（%s），改用对齐叠加预览。", exc)
        return None


def preview_swap(
    source_path: str | Path,
    reference_path: str | Path,
    *,
    cfg: Any | None = None,
    backend: Any | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Swap one frame and report honestly how it was made.

    Returns ``{"image", "method", "is_final", "note", "inswapper_error",
    "source_path", "reference_path"}``. ``method`` is ``"inswapper"``
    only when the real backend actually changed the frame; otherwise
    ``"aligned_overlay"`` with ``is_final=False`` and the NOT_FINAL
    wording in ``note``.
    """
    source = Path(source_path)
    reference = Path(reference_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到采集帧：{source}")
    if not reference.is_file():
        raise FileNotFoundError(f"找不到参考脸：{reference}")
    frame = _read_image(source)
    ref_image = _read_image(reference)

    inswapper_error: str | None = None
    swapped: np.ndarray | None = None
    if backend is not None:
        swapped = _try_inswapper(frame, reference, cfg=cfg, backend=backend)
        if swapped is None:
            inswapper_error = "注入的换脸后端没有产出有效画面。"
    else:
        status = inswapper_status(cfg)
        if status["available"]:
            swapped = _try_inswapper(frame, reference, cfg=cfg, backend=None)
            if swapped is None:
                inswapper_error = "真换脸后端加载或执行失败。"
        else:
            inswapper_error = str(status.get("reason") or "")

    if swapped is not None:
        image = swapped
        method = "inswapper"
        is_final = True
        note = INSWAPPER_METHOD_TEXT
    else:
        image = aligned_overlay_preview(frame, ref_image)
        method = "aligned_overlay"
        is_final = False
        note = f"{OVERLAY_METHOD_TEXT}。{NOT_FINAL_TEXT}"
        if inswapper_error:
            note += f"（{inswapper_error}）"

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # imwrite fails silently on non-ASCII paths: encode, then write bytes.
        ok, buffer = cv2.imencode(out.suffix or ".png", image)
        if not ok:
            raise ValueError(f"写不出图片：{out}")
        buffer.tofile(str(out))
    return {
        "image": image,
        "method": method,
        "is_final": is_final,
        "note": note,
        "inswapper_error": inswapper_error,
        "source_path": str(source),
        "reference_path": str(reference),
    }
