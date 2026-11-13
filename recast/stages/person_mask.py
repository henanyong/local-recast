"""Person silhouette without a download: YOLO boxes refined by GrabCut.

This exists because the segmentation backend the re-render path depends on
disappeared: mediapipe 0.10.35 removed ``mp.solutions``, and its replacement
(``mp.tasks``) needs a ``.tflite`` asset that has to be fetched. The repo
already ships ``yolo11n.pt``, and OpenCV already has GrabCut, so a usable
silhouette is available offline today.

Two stages, because neither alone is good enough:

- YOLO gives a reliable *box* and a person count, but a box is not a
  silhouette — clipping a costume to a rectangle is exactly the "sticker"
  look the 2D route is supposed to avoid.
- GrabCut gives a real outline with hair and shoulder edges, but it needs a
  seed. A YOLO box is a good seed.

Failure is explicit: no person detected raises, and a GrabCut result that
collapses to nothing is reported as ``method="box"`` with a warning rather
than passed off as a silhouette.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

PERSON_CLASS_ID = 0
DEFAULT_WEIGHTS = "yolo11n.pt"
# Where a dropped-in segmentation model is looked for. Both are optional: the
# GrabCut path needs neither, it is just slower and coarser.
MODEL_DIRS = (
    Path(__file__).resolve().parents[2] / "models" / "segmentation",
    Path(__file__).resolve().parents[2] / "models",
    Path(__file__).resolve().parents[2],
)
YOLO_SEG_GLOBS = ("yolo11*-seg.pt", "yolov8*-seg.pt", "yolo*-seg.pt")
BIREFNET_GLOBS = ("birefnet*.onnx", "BiRefNet*.onnx")
MEDIAPIPE_SEG_GLOBS = ("*selfie*segmenter*.tflite", "*selfie*.task", "*segmenter*.task")


class PersonMaskError(RuntimeError):
    """No person could be outlined in this frame."""


def _first_match(patterns: tuple[str, ...]) -> Path | None:
    for directory in MODEL_DIRS:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            found = sorted(directory.glob(pattern))
            if found:
                return found[0]
    return None


def find_birefnet() -> Path | None:
    """BiRefNet ONNX weights. MIT-licensed, so it is the commercial-safe
    counterpart to RMBG-1.4 (which is non-commercial)."""
    for directory in MODEL_DIRS:
        for pattern in BIREFNET_GLOBS:
            found = sorted(directory.glob(pattern))
            if found:
                return found[0]
    return None


def find_rmbg() -> Path | None:
    """ONNX RMBG-1.4 weights, if any are on disk.

    Searched here because the model usually arrives with a 3D/texture
    pipeline rather than with this repo: ``TripoSG/pretrained_weights`` is
    where it actually lives on this machine.
    """
    candidates: list[Path] = []
    for directory in MODEL_DIRS:
        candidates.append(directory / "RMBG-1.4" / "onnx" / "model.onnx")
        candidates.append(directory / "rmbg" / "model.onnx")
    home = Path.home()
    candidates.extend(
        [
            home / "TripoSG" / "pretrained_weights" / "RMBG-1.4" / "onnx" / "model.onnx",
            home / ".cache" / "huggingface" / "hub" / "models--briaai--RMBG-1.4",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    hub_root = home / ".cache" / "huggingface" / "hub" / "models--briaai--RMBG-1.4"
    if hub_root.is_dir():
        found = sorted(hub_root.rglob("model.onnx"))
        if found:
            return found[0]
    return None


_RMBG_SESSION: Any = None
_RMBG_PROVIDER = ""
_RMBG_INPUT_SIZE = 1024
# RECAST_RMBG_CPU=1 forces CPU (e.g. the GPU is busy with another run).
_RMBG_PREFER_CUDA = os.environ.get("RECAST_RMBG_CPU", "") not in {"1", "true", "yes"}


def _register_torch_cuda_dlls() -> None:
    """Put torch's bundled CUDA 12 / cuDNN 9 DLLs on the Windows search path.

    ``onnxruntime-gpu`` needs cuDNN 9 when the session is created and silently
    falls back to CPU without it; the symptom is a 0.86 s/frame mask and one
    log line about providers. torch already ships the exact DLLs
    (``cudnn64_9.dll``, ``cublas64_12.dll``), so registering its ``lib``
    directory is enough — measured 860 ms -> 48 ms per frame, no install.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import torch
    except ImportError:
        return
    lib = Path(torch.__file__).resolve().parent / "lib"
    if not lib.is_dir():
        return
    try:
        os.add_dll_directory(str(lib))
    except (AttributeError, OSError):
        # No add_dll_directory, or already registered: the CPU fallback still
        # works, so this is not fatal.
        pass


def rmbg_provider() -> str:
    """Provider the RMBG session actually runs on: "cuda", "cpu", or ""."""
    return _RMBG_PROVIDER


def _rmbg_session(model_path: str | Path) -> Any:
    global _RMBG_SESSION, _RMBG_PROVIDER
    if _RMBG_SESSION is not None:
        return _RMBG_SESSION
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    providers = ["CPUExecutionProvider"]
    if _RMBG_PREFER_CUDA:
        _register_torch_cuda_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    active = list(session.get_providers())
    _RMBG_PROVIDER = "cuda" if "CUDAExecutionProvider" in active else "cpu"
    if _RMBG_PREFER_CUDA and _RMBG_PROVIDER == "cpu":
        log.warning(
            "RMBG 退回了 CPU（约 0.86 秒/帧）。onnxruntime-gpu 需要 cuDNN 9；"
            "装有 torch 的机器上它通常已在 torch/lib 里，检查该目录是否进了 DLL 搜索路径。"
        )
    _RMBG_SESSION = session
    return _RMBG_SESSION


def rmbg_foreground(bgr: Any, *, model_path: str | Path | None = None) -> Any:
    """Soft foreground alpha (HxW float32, 0..1) from RMBG-1.4.

    The ONNX graph already ends in a sigmoid, so the output is alpha, not
    logits — squashing it again turns a decent mask into "everything".
    """
    import cv2
    import numpy as np

    path = Path(model_path) if model_path else find_rmbg()
    if path is None:
        raise PersonMaskError(
            "没找到 RMBG-1.4 ONNX：把 RMBG-1.4/onnx/model.onnx 放进 models/segmentation/，"
            "或放在 ~/TripoSG/pretrained_weights/RMBG-1.4/onnx/ 下"
        )
    session = _rmbg_session(path)
    frame = np.asarray(bgr)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_RMBG_INPUT_SIZE, _RMBG_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = ((resized.astype(np.float32) / 255.0 - 0.5)).transpose(2, 0, 1)[None]
    name = session.get_inputs()[0].name
    out = session.run(None, {name: tensor.astype(np.float32)})[0]
    alpha = np.asarray(out)[0, 0]
    alpha = cv2.resize(alpha.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)


_BIREFNET_SESSION: Any = None
_BIREFNET_PROVIDER = ""


def _birefnet_session(model_path: str | Path) -> Any:
    global _BIREFNET_SESSION, _BIREFNET_PROVIDER
    if _BIREFNET_SESSION is not None:
        return _BIREFNET_SESSION
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    providers = ["CPUExecutionProvider"]
    if _RMBG_PREFER_CUDA:
        _register_torch_cuda_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    _BIREFNET_PROVIDER = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    _BIREFNET_SESSION = session
    return _BIREFNET_SESSION


def birefnet_foreground(bgr: Any, *, model_path: str | Path | None = None) -> Any:
    """Soft foreground alpha from a BiRefNet ONNX export.

    BiRefNet is trained with ImageNet normalisation (mean 0.485/0.456/0.406,
    std 0.229/0.224/0.225), unlike RMBG's 0.5/1.0 — feeding it the wrong
    statistics yields a plausible-looking but wrong mask, so the two are kept
    as separate functions rather than one behind a flag.
    """
    import cv2
    import numpy as np

    path = Path(model_path) if model_path else find_birefnet()
    if path is None:
        raise PersonMaskError(
            "没找到 BiRefNet ONNX：把 birefnet.onnx 放进 models/segmentation/。"
            "BiRefNet 是 MIT，可商用，是从 RMBG 换出来的首选。"
        )
    session = _birefnet_session(path)
    frame = np.asarray(bgr)
    h, w = frame.shape[:2]
    shape = session.get_inputs()[0].shape
    size = int(shape[2]) if isinstance(shape[2], int) else 1024
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((resized - mean) / std).transpose(2, 0, 1)[None]
    out = session.run(None, {session.get_inputs()[0].name: tensor.astype(np.float32)})[0]
    alpha = np.asarray(out)
    alpha = np.squeeze(alpha)
    if alpha.ndim == 3:
        alpha = alpha[0]
    alpha = cv2.resize(alpha.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if float(alpha.max()) > 1.5:  # some exports skip the sigmoid
        alpha = 1.0 / (1.0 + np.exp(-alpha))
    return np.clip(alpha, 0.0, 1.0)


def segment_with_birefnet(
    bgr: Any,
    *,
    index: int = 0,
    model_path: str | Path | None = None,
    conf: float = 0.25,
    model: Any = None,
    weights: str | Path = DEFAULT_WEIGHTS,
    device: str | None = None,
) -> dict[str, Any]:
    """BiRefNet edges restricted to one person, same ownership trick as RMBG."""
    boxes = detect_person_boxes(bgr, weights=weights, model=model, conf=conf, device=device)
    if not boxes:
        raise PersonMaskError(f"这一帧没有检测到人物（conf={conf}）")
    alpha = birefnet_foreground(bgr, model_path=model_path)
    ranked = rank_boxes_by_owned_foreground(alpha, boxes)
    if index >= len(ranked):
        raise PersonMaskError(f"只分出 {len(ranked)} 个人，取不到第 {index + 1} 个")
    slot, _owned = ranked[index]
    owner = _assign_foreground_to_boxes(alpha, boxes)
    mine = (alpha > 0.5) & (owner == int(slot))
    mask = _clean(mine.astype(np.uint8) * 255)
    if int(mask.sum()) == 0:
        raise PersonMaskError("这个人的框里没有前景像素：可能被完全遮挡，或框选错了人")
    return _mask_to_result(mask, boxes[slot], "birefnet+yolo", len(ranked), "")


def rank_boxes_by_owned_foreground(
    alpha: Any, boxes: list[list[float]], *, min_share: float = 0.02
) -> list[tuple[int, int]]:
    """``[(box_index, owned_pixels), ...]``, biggest real claimant first.

    Ordering by *box area* looks equivalent and is not: YOLO sometimes emits
    one near-full-frame box that swallows the whole two-shot. That box is the
    largest, so it becomes "person 1", while what it actually owns — first
    writer wins uses the smallest box — is a sliver at the frame edge. On a
    measured frame the actor's real body ended up as person 2 and person 1 was
    a strip of wall.

    Counting the foreground pixels each box ends up owning ranks by who is
    actually in the picture. Boxes that own almost nothing are dropped: they
    are the spurious merges.
    """
    import numpy as np

    owner = _assign_foreground_to_boxes(alpha, boxes)
    fg = np.asarray(alpha) > 0.5
    total = int(fg.sum())
    floor = max(200, int(total * min_share))
    ranked: list[tuple[int, int]] = []
    for slot in range(len(boxes)):
        owned = int((fg & (owner == slot)).sum())
        if owned >= floor:
            ranked.append((slot, owned))
    ranked.sort(key=lambda item: -item[1])
    return ranked


def _assign_foreground_to_boxes(alpha: Any, boxes: list[list[float]], *, thr: float = 0.5) -> Any:
    """Owner box index per foreground pixel, -1 where no box contains it.

    A plain intersection is not enough in a two-shot: the close-up subject's
    box covers most of the frame, so intersecting keeps the neighbour too.
    Assigning each pixel to the *smallest* box that contains it uses the one
    reliable cue — a person's own box is tighter around them than the box of
    whoever is standing in front. The cost is a straight cut along the box
    edge between two overlapping people, which is roughly where the real
    boundary is anyway.
    """
    import numpy as np

    h, w = np.asarray(alpha).shape[:2]
    owner = np.full((h, w), -1, dtype=np.int16)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    for slot in np.argsort(areas):  # smallest first: first writer wins
        x0, y0, x1, y1 = boxes[int(slot)]
        ys, ye = max(0, int(y0)), min(h, int(y1))
        xs, xe = max(0, int(x0)), min(w, int(x1))
        if ye <= ys or xe <= xs:
            continue
        window = owner[ys:ye, xs:xe]
        window[window < 0] = int(slot)
    return owner


def segment_with_rmbg(
    bgr: Any,
    *,
    index: int = 0,
    model_path: str | Path | None = None,
    conf: float = 0.25,
    model: Any = None,
    weights: str | Path = DEFAULT_WEIGHTS,
    device: str | None = None,
) -> dict[str, Any]:
    """RMBG edges, restricted to one person by box ownership.

    RMBG segments the salient foreground, which on a two-shot is *both*
    people — recolouring the neighbour's clothes along with the target's is
    exactly the kind of error that survives review. The boxes supply instance
    identity, the model supplies the boundary.
    """
    import numpy as np

    boxes = detect_person_boxes(bgr, weights=weights, model=model, conf=conf, device=device)
    if not boxes:
        raise PersonMaskError(f"这一帧没有检测到人物（conf={conf}）")

    alpha = rmbg_foreground(bgr, model_path=model_path)
    ranked = rank_boxes_by_owned_foreground(alpha, boxes)
    if index >= len(ranked):
        raise PersonMaskError(
            f"只分出 {len(ranked)} 个人，取不到第 {index + 1} 个"
        )
    slot, owned = ranked[index]
    owner = _assign_foreground_to_boxes(alpha, boxes)
    mine = (alpha > 0.5) & (owner == int(slot))
    mask = _clean(mine.astype(np.uint8) * 255)
    if int(mask.sum()) == 0:
        raise PersonMaskError(
            "这个人的框里没有前景像素：可能被完全遮挡，或框选错了人"
        )
    warning = ""
    if len(ranked) < len(boxes):
        warning = (
            f"忽略了 {len(boxes) - len(ranked)} 个空框（几乎没有前景，"
            "多半是合并了多个人的误检）"
        )
    result = _mask_to_result(mask, boxes[slot], "rmbg-1.4+yolo", len(ranked), warning)
    result["owned_pixels"] = owned
    return result


def find_yolo_seg() -> Path | None:
    """A YOLO instance-segmentation checkpoint, if one is on disk."""
    return _first_match(YOLO_SEG_GLOBS)


def find_mediapipe_seg() -> Path | None:
    """A MediaPipe selfie-segmenter asset, if one is on disk."""
    return _first_match(MEDIAPIPE_SEG_GLOBS)


def describe_backends() -> dict[str, dict[str, Any]]:
    """What can segment on this machine, what is missing, and under what licence.

    The UI and the clip runner both need this. Two things it must never hide:
    a silent fall back to GrabCut looks identical to a fast run until the
    clock says otherwise, and three of these backends are **not** free for
    commercial delivery — a licence surprise at delivery time costs more than
    a slower mask.
    """
    rmbg = find_rmbg()
    birefnet = find_birefnet()
    yolo = find_yolo_seg()
    mp_asset = find_mediapipe_seg()
    return {
        "birefnet+yolo": {
            "available": birefnet is not None,
            "path": str(birefnet) if birefnet else "",
            "license": "MIT（可商用）",
            "commercial_ok": True,
            "reason": ""
            if birefnet
            else "没找到 BiRefNet ONNX：把 birefnet.onnx 放进 models/segmentation/（MIT，可商用）",
        },
        "rmbg+yolo": {
            "available": rmbg is not None,
            "path": str(rmbg) if rmbg else "",
            "license": "非商用（BRIA RMBG-1.4），对外交付前必须换掉或买授权",
            "commercial_ok": False,
            "reason": ""
            if rmbg
            else "没找到 RMBG-1.4 ONNX：放到 models/segmentation/ 或 ~/TripoSG/pretrained_weights/RMBG-1.4/onnx/",
        },
        "yolo-seg": {
            "available": yolo is not None,
            "path": str(yolo) if yolo else "",
            "license": "AGPL-3.0（Ultralytics），商用需企业授权",
            "commercial_ok": False,
            "reason": ""
            if yolo
            else "没找到 YOLO 分割权重：把 yolo11n-seg.pt 放进 models/segmentation/ 或项目根目录",
        },
        "mediapipe-tasks": {
            "available": mp_asset is not None,
            "path": str(mp_asset) if mp_asset else "",
            "license": "Apache-2.0（可商用）",
            "commercial_ok": True,
            "reason": ""
            if mp_asset
            else "没找到 MediaPipe 分割模型：把 selfie_segmenter.tflite（或 .task）放进 models/segmentation/",
        },
        "grabcut": {
            "available": True,
            "path": "",
            "license": "OpenCV Apache-2.0（可商用）",
            "commercial_ok": True,
            "reason": "离线兜底：比模型慢（约 3 秒/帧），边缘也更粗",
        },
    }


# Preference order for ``auto``: edge quality first, then licence, then speed.
# RMBG leads because it is the only real model present on most installs, but
# its licence is reported so a delivery run can refuse it.
# yolo-seg first: it is the only backend that gives true per-instance masks, so
# a two-shot does not need box ownership at all. birefnet/rmbg segment the whole
# salient foreground and lean on boxes, which is where the hard box-edge cut
# comes from. Order is by output quality, not by licence; ``allow_noncommercial``
# filters the licence separately.
_BACKEND_ORDER = ("yolo-seg", "birefnet+yolo", "rmbg+yolo", "mediapipe-tasks", "grabcut")


def pick_backend(preferred: str = "auto", *, allow_noncommercial: bool = True) -> tuple[str, str]:
    """(backend, reason) for ``preferred``, falling back in quality order."""
    table = describe_backends()
    if preferred != "auto":
        if preferred not in table:
            raise PersonMaskError(
                f"未知分割后端：{preferred}（可用 auto / {' / '.join(_BACKEND_ORDER)}）"
            )
        if not table[preferred]["available"]:
            raise PersonMaskError(f"{preferred} 不可用：{table[preferred]['reason']}")
        if not allow_noncommercial and not table[preferred]["commercial_ok"]:
            raise PersonMaskError(
                f"{preferred} 是非商用模型（{table[preferred]['license']}），"
                "交付跑不能使用；换 mediapipe-tasks 或 grabcut。"
            )
        return preferred, ""
    for name in _BACKEND_ORDER:
        if not table[name]["available"]:
            continue
        if not allow_noncommercial and not table[name]["commercial_ok"]:
            continue
        if name == "grabcut":
            return name, table[name]["reason"]
        return name, ""
    return "grabcut", table["grabcut"]["reason"]


_YOLO_CACHE: dict[tuple[str, str], Any] = {}


def _load_yolo(weights: str | Path, *, device: str | None = None) -> Any:
    """Load (and cache) a YOLO model.

    Caching matters: box detection runs once per frame, and ``YOLO(...)``
    re-reads the checkpoint every time (~1 s). Without this the clip runner
    spends more time loading weights than processing pixels.
    """
    from ultralytics import YOLO

    key = (str(weights), device or "")
    cached = _YOLO_CACHE.get(key)
    if cached is not None:
        return cached
    model = YOLO(str(weights))
    if device:
        model.to(device)
    _YOLO_CACHE[key] = model
    return model


def detect_person_boxes(
    bgr: Any,
    *,
    weights: str | Path = DEFAULT_WEIGHTS,
    model: Any = None,
    conf: float = 0.25,
    device: str | None = None,
) -> list[list[float]]:
    """Person boxes as [x0, y0, x1, y1], largest first."""
    import numpy as np

    net = model if model is not None else _load_yolo(weights, device=device)
    results = net.predict(bgr, conf=conf, verbose=False, classes=[PERSON_CLASS_ID])
    boxes: list[list[float]] = []
    for result in results:
        if result.boxes is None:
            continue
        for raw in result.boxes.xyxy.cpu().numpy():
            x0, y0, x1, y1 = (float(v) for v in raw[:4])
            boxes.append([x0, y0, x1, y1])
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def grabcut_in_box(
    bgr: Any,
    box: list[float],
    *,
    iterations: int = 3,
    pad: float = 0.04,
    work_size: int = 640,
) -> Any:
    """GrabCut silhouette seeded by ``box``; zeros when it finds nothing.

    GrabCut is run on a downscaled copy. At 1080x1920 a 3-iteration GrabCut
    takes ~28 s per frame, which is useless for a clip; at 640 px it is
    sub-second and the outline is then upscaled and re-thresholded. The
    silhouette loses a couple of pixels of hair detail, which the feather in
    the compositor would have softened anyway.
    """
    import cv2
    import numpy as np

    source = np.asarray(bgr)
    h, w = source.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if work_size and longest > work_size:
        scale = float(work_size) / float(longest)
    small = (
        cv2.resize(source, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else source
    )
    sh, sw = small.shape[:2]

    x0, y0, x1, y1 = (float(v) * scale for v in box)
    pad_x = (x1 - x0) * pad
    pad_y = (y1 - y0) * pad
    rx0 = max(0, int(x0 - pad_x))
    ry0 = max(0, int(y0 - pad_y))
    rx1 = min(sw - 1, int(x1 + pad_x))
    ry1 = min(sh - 1, int(y1 + pad_y))
    rect = (rx0, ry0, rx1 - rx0, ry1 - ry0)
    if rect[2] < 4 or rect[3] < 4:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((sh, sw), dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, int(iterations), cv2.GC_INIT_WITH_RECT)
    except cv2.error as exc:
        log.info("GrabCut 失败: %s", exc)
        return np.zeros((h, w), dtype=np.uint8)
    solid = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if scale < 1.0:
        solid = cv2.resize(solid, (w, h), interpolation=cv2.INTER_LINEAR)
        solid = np.where(solid > 127, 255, 0).astype(np.uint8)
    return solid


def _clean(mask: Any) -> Any:
    import cv2
    import numpy as np

    kernel = np.ones((5, 5), np.uint8)
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    # Keep the largest component: GrabCut sometimes leaks onto a second body.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(out)
    if count <= 1:
        return out
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def segment_person(
    bgr: Any,
    *,
    index: int = 0,
    weights: str | Path = DEFAULT_WEIGHTS,
    model: Any = None,
    conf: float = 0.25,
    device: str | None = None,
    iterations: int = 3,
) -> dict[str, Any]:
    """Silhouette of the ``index``-th largest person in the frame.

    Returns ``{"mask", "bbox", "method", "person_count", "warning"}`` where
    ``method`` is ``"grabcut"`` for a real outline or ``"box"`` when GrabCut
    found nothing and the box is all there is.
    """
    import numpy as np

    boxes = detect_person_boxes(bgr, weights=weights, model=model, conf=conf, device=device)
    if not boxes:
        raise PersonMaskError(
            f"这一帧没有检测到人物（conf={conf}）。"
            "如果画面里确实有人，调低 conf；否则跳过这一帧。"
        )
    if index >= len(boxes):
        raise PersonMaskError(f"只检测到 {len(boxes)} 个人，取不到第 {index + 1} 个")

    box = boxes[index]
    mask = _clean(grabcut_in_box(bgr, box, iterations=iterations))
    warning = ""
    method = "grabcut"
    if int(mask.sum()) == 0:
        h, w = np.asarray(bgr).shape[:2]
        x0, y0, x1, y1 = (int(v) for v in box)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 255
        method = "box"
        warning = "GrabCut 没分出轮廓，退回矩形框（边缘会是硬的）"
        log.warning(warning)
    return {
        "mask": mask,
        "bbox": box,
        "method": method,
        "person_count": len(boxes),
        "warning": warning,
    }


def estimate_neck_y(bgr: Any, *, fallback_ratio: float = 0.42) -> dict[str, Any]:
    """Y of the neck cut, from a detected face when one is available.

    The re-render path derives this from pose keypoints (nose to mid
    shoulder). Without a pose model the chin is the next best anchor, and
    insightface can supply a face box offline. Falls back to a fixed ratio
    and says so, because a guessed neck line shifts every garment edge.
    """
    import numpy as np

    shape = np.asarray(bgr).shape[:2]
    h = int(shape[0])
    try:
        import cv2

        from recast.stages.character_sheet import _face_box

        box = _face_box(np.asarray(bgr))
    except Exception:
        box = None
    if box is None:
        return {
            "neck_y": int(h * fallback_ratio),
            "source": "ratio",
            "note": f"没有检测到人脸，脖子切口用画面高度 {fallback_ratio:.0%} 的估计",
        }
    face_h = max(1.0, box[3] - box[1])
    neck = int(min(shape[0] - 1, box[3] + 0.12 * face_h))
    return {"neck_y": neck, "source": "face", "note": ""}


def _mask_to_result(mask: Any, box: list[float] | None, method: str, count: int, warning: str = "") -> dict[str, Any]:
    import numpy as np

    ys, xs = np.where(mask > 0)
    bbox = box
    if bbox is None and ys.size:
        bbox = [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]
    return {
        "mask": mask,
        "bbox": bbox or [0.0, 0.0, 0.0, 0.0],
        "method": method,
        "person_count": count,
        "warning": warning,
    }


def segment_with_yolo_seg(
    bgr: Any,
    weights: str | Path,
    *,
    index: int = 0,
    conf: float = 0.25,
    model: Any = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Instance segmentation with a YOLO ``-seg`` checkpoint.

    ~30 ms/frame on GPU, which is the difference between 4.6 hours and
    16 minutes per episode. The mask comes back at the model's inference
    resolution and is resized to the frame.
    """
    import cv2
    import numpy as np

    net = model if model is not None else _load_yolo(weights, device=device)
    results = net.predict(bgr, conf=conf, classes=[PERSON_CLASS_ID], verbose=False, retina_masks=True)
    result = results[0] if results else None
    if result is None or result.masks is None or len(result.masks.data) == 0:
        raise PersonMaskError(f"这一帧没有分割出人物（conf={conf}）")

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    order = np.argsort(areas)[::-1]
    if index >= len(order):
        raise PersonMaskError(f"只分割出 {len(order)} 个人，取不到第 {index + 1} 个")

    pick = int(order[index])
    raw = masks[pick]
    h, w = np.asarray(bgr).shape[:2]
    if raw.shape != (h, w):
        raw = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    mask = _clean(((raw > 0.5).astype(np.uint8)) * 255)
    return _mask_to_result(mask, [float(v) for v in boxes[pick]], "yolo-seg", len(order))


def segment_with_mediapipe(bgr: Any, asset: str | Path) -> dict[str, Any]:
    """Foreground mask from a MediaPipe selfie-segmenter asset.

    MediaPipe's segmenter returns one foreground mask, not instances, so
    ``index`` other than 0 is meaningless here — the caller is told instead
    of being handed the wrong person.
    """
    import numpy as np

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise PersonMaskError(f"需要 mediapipe tasks：{exc}") from exc

    asset_path = Path(asset)
    if not asset_path.is_file():
        raise PersonMaskError(f"找不到 MediaPipe 分割模型：{asset_path}")
    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(asset_path)),
        output_category_mask=True,
    )
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        rgb = np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        output = segmenter.segment(image)
    mask = np.asarray(output.category_mask.numpy_view())
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = _clean((mask > 0).astype(np.uint8) * 255)
    return _mask_to_result(mask, None, "mediapipe-tasks", 1)


def segment_person_auto(
    bgr: Any,
    *,
    backend: str = "auto",
    index: int = 0,
    conf: float = 0.25,
    weights: str | Path = DEFAULT_WEIGHTS,
    device: str | None = None,
    allow_noncommercial: bool = True,
    model: Any = None,
) -> dict[str, Any]:
    """Segment with the best backend on this machine, and say which one ran.

    ``backend="auto"`` prefers a real model (rmbg+yolo, yolo-seg, mediapipe)
    and falls back to GrabCut. The chosen backend is in ``method`` and the
    reason for a fallback is in ``warning``, so a slow run is never mistaken
    for a model run. ``allow_noncommercial=False`` is the delivery switch:
    it skips RMBG-1.4 (and any other non-commercial backend) outright.
    """
    name, reason = pick_backend(backend, allow_noncommercial=allow_noncommercial)
    if name == "birefnet+yolo":
        out = segment_with_birefnet(
            bgr, index=index, conf=conf, weights=weights, device=device, model=model
        )
    elif name == "rmbg+yolo":
        out = segment_with_rmbg(bgr, index=index, conf=conf, weights=weights, device=device, model=model)
    elif name == "yolo-seg":
        out = segment_with_yolo_seg(
            bgr, find_yolo_seg() or weights, index=index, conf=conf, device=device
        )
    elif name == "mediapipe-tasks":
        if index != 0:
            raise PersonMaskError("MediaPipe 分割只给一个前景，取不到第 2 个人")
        out = segment_with_mediapipe(bgr, find_mediapipe_seg() or "")
    else:
        out = segment_person(bgr, index=index, conf=conf, weights=weights, device=device)
    if reason:
        out["warning"] = f"{out['warning']}；{reason}".strip("；")
    return out


def mask_fn_from_model(
    model: Any,
    *,
    index: int = 0,
    conf: float = 0.3,
) -> Any:
    """Build a ``mask_fn(frame, index)`` for :func:`replace_clip`.

    The YOLO model is passed in so a clip run loads the weights once instead
    of per frame — at 30 fps the reload would dominate the runtime.
    """

    def _mask(frame: Any, _frame_index: int) -> Any:
        return segment_person(frame, index=index, model=model, conf=conf)["mask"]

    return _mask
