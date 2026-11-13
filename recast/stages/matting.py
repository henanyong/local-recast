"""Person matting: cut a person out of a photo into a transparent PNG.

Uses rembg (default u2net model) locally, one image at a time — no batch
parallelism, so it fits in 8GB VRAM and on CPU. Outputs go to the character
folder's ``cutout/`` directory.

When rembg or its model is missing, nothing fails silently: the error tells
the user exactly what to install and where to put the model file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from PIL import Image

log = logging.getLogger("recast")

DEFAULT_MODEL = "u2net"
MODEL_DOWNLOAD_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

# rembg looks for models under ~/.u2net (or $U2NET_HOME) first.
MODEL_FILE = "u2net.onnx"


class MattingError(RuntimeError):
    """Matting cannot run: rembg or its model is missing, or the run failed."""


def person_mask_available() -> bool:
    """True when the project's own segmentation stack can run.

    ``rembg`` is the nicer cutout (u2net edges) but it is an optional extra, and
    [抠像] refusing to run without it was wrong: the same install already has a
    person-segmentation backend that the replacement pipeline depends on. One
    feature must not need a second, unrelated dependency to work.
    """
    try:
        from recast.stages.person_mask import pick_backend

        return bool(pick_backend("auto")[0])
    except Exception:
        return False


def matte_backend() -> str:
    """Which backend a call would use right now: ``rembg`` | ``person-mask`` | ``""``."""
    if rembg_available():
        return "rembg"
    if person_mask_available():
        return "person-mask"
    return ""


def _model_search_paths(models_dir: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    if models_dir:
        paths.append(Path(models_dir).expanduser() / MODEL_FILE)
    import os

    home = os.environ.get("U2NET_HOME")
    if home:
        paths.append(Path(home).expanduser() / MODEL_FILE)
    paths.append(Path.home() / ".u2net" / MODEL_FILE)
    return paths


def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401
    except ImportError:
        return False
    return True


def _load_rembg() -> Any:
    try:
        import rembg
    except ImportError as exc:
        raise MattingError(
            "抠像需要 rembg，未安装。请先安装：pip install rembg。"
            "（本项目已用 onnxruntime-gpu，加 --no-deps 可避免 CPU 版 onnxruntime"
            " 覆盖 GPU 版：pip install --no-deps rembg。）"
        ) from exc
    return rembg


def _new_session(rembg: Any, model: str, models_dir: str | Path | None) -> Any:
    """Create a rembg session, failing with a clear message when the model is absent."""
    try:
        return rembg.new_session(model)
    except Exception as exc:
        places = "、".join(str(p) for p in _model_search_paths(models_dir))
        raise MattingError(
            f"抠像模型 {model} 不在本机，自动下载也失败了（{exc}）。"
            f"请手动下载模型文件，放进下面任一位置再重试：{places}。"
            f"下载地址：{MODEL_DOWNLOAD_URL}"
        ) from exc


def _remove_image(rembg: Any, image: Image.Image, session: Any) -> Image.Image:
    """Thin wrapper around rembg.remove, kept separate so tests can stub it."""
    out = rembg.remove(image, session=session)
    if isinstance(out, bytes):
        from io import BytesIO

        return Image.open(BytesIO(out)).convert("RGBA")
    return out.convert("RGBA")


def _cutout_with_person_mask(src: Path, dest: Path) -> Path:
    """Cut out with the project's own segmenter, keeping its soft alpha."""
    import numpy as np
    from PIL import Image as _Image

    from recast.stages.person_mask import segment_person_auto

    data = np.fromfile(str(src), dtype=np.uint8)
    bgr = __import__("cv2").imdecode(data, __import__("cv2").IMREAD_COLOR)
    if bgr is None:
        raise MattingError(f"读不出图片：{src}")
    result = segment_person_auto(bgr, index=0, conf=0.25)
    alpha = result.get("alpha")
    if alpha is None:
        alpha = (result["mask"] > 0).astype(np.float32)
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if alpha.shape[:2] != bgr.shape[:2]:
        import cv2

        alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]))
    rgb = np.dstack([bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]])
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    dest.parent.mkdir(parents=True, exist_ok=True)
    _Image.fromarray(rgba, mode="RGBA").save(dest, format="PNG")
    log.info("已抠像（person-mask）: %s -> %s", src.name, dest)
    return dest


def matte_image(
    src: str | Path,
    dest: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    models_dir: str | Path | None = None,
    session: Any = None,
    backend: str = "auto",
) -> Path:
    """Cut one image out to a transparent PNG.

    ``session`` is injectable for tests; normally a fresh session is created
    (one model load per call keeps memory low on 8GB cards). ``backend`` is
    ``auto`` (rembg when installed, otherwise the project's segmenter),
    ``rembg`` or ``person-mask``.
    """
    src = Path(src)
    dest = Path(dest)
    if not src.is_file():
        raise MattingError(f"找不到要抠像的图片：{src}")
    if backend == "auto":
        backend = matte_backend()
    if backend == "":
        raise MattingError(
            "抠像没有可用后端：装 rembg（pip install --no-deps rembg），"
            "或把 yolo11n-seg.pt / rmbg 模型放进 models\\segmentation\\。"
        )
    if backend == "person-mask":
        return _cutout_with_person_mask(src, dest)
    rembg = _load_rembg()
    if session is None:
        session = _new_session(rembg, model, models_dir)
    try:
        image = Image.open(src).convert("RGB")
        cutout = _remove_image(rembg, image, session)
    except MattingError:
        raise
    except Exception as exc:
        raise MattingError(f"抠像失败 {src}：{exc}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    cutout.save(dest, format="PNG")
    log.info("已抠像: %s -> %s", src.name, dest)
    return dest


def matte_character_folder(
    char_folder: str | Path,
    *,
    sources: list[str | Path] | None = None,
    model: str = DEFAULT_MODEL,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Matte a character's photos into its cutout/ directory, one by one.

    ``sources`` defaults to the images in the folder's face/ directory.
    A failure on one photo is recorded and the rest still run. Returns a
    report: outputs, failed (path + reason), skipped (already done).
    """
    from recast.core.characters import load_meta

    folder = Path(char_folder)
    if not folder.is_dir():
        raise MattingError(f"找不到人物文件夹：{folder}")
    meta = load_meta(folder)
    name = (meta or {}).get("name") or folder.name
    if sources is None:
        face_dir = folder / "face"
        sources = (
            sorted(p for p in face_dir.iterdir() if p.is_file())
            if face_dir.is_dir()
            else []
        )
    cutout_dir = folder / "cutout"
    cutout_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"character": name, "outputs": [], "failed": [], "skipped": []}
    if not sources:
        report["failed"].append({"path": "", "reason": "没有可抠像的图片（face/ 为空）"})
        return report
    backend = matte_backend()
    report["backend"] = backend or ""
    if backend == "rembg":
        rembg = _load_rembg()
        session = _new_session(rembg, model, models_dir)
    elif backend == "person-mask":
        session = None
        log.info("rembg 未安装，改用本项目的分割后端抠像（person-mask）")
    else:
        report["failed"].append(
            {"path": "", "reason": "没有可用的抠像后端：装 rembg，或放分割权重到 models/segmentation/"}
        )
        return report
    for raw in sources:
        src = Path(raw)
        dest = cutout_dir / f"{src.stem}.png"
        if dest.is_file():
            report["skipped"].append(str(dest))
            continue
        try:
            matte_image(
                src,
                dest,
                model=model,
                models_dir=models_dir,
                session=session,
                backend=backend,
            )
            report["outputs"].append(str(dest))
        except MattingError as exc:
            report["failed"].append({"path": str(src), "reason": str(exc)})
        except Exception as exc:  # keep going: one bad photo must not stop the rest
            report["failed"].append({"path": str(src), "reason": str(exc)})
    log.info(
        "人物「%s」抠像完成: 成功 %s 张，失败 %s 张，跳过 %s 张。",
        name,
        len(report["outputs"]),
        len(report["failed"]),
        len(report["skipped"]),
    )
    return report
