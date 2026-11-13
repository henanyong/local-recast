"""Costume auto-tagging: crop the body region of a photo and describe it.

Writes a costume screenshot into the character folder's ``costume/`` and a
description (colours, style) into meta.json. Two taggers:

- ``ClipTagger`` (default): local CLIP zero-shot (ViT-B/32) over a small
  Chinese label set. Needs torch + the ``clip`` package and the model
  weights; when any of it is missing the error says exactly what to install
  and where the weights go — never a silent fallback, never invented tags.
- ``HeuristicTagger``: always available, no model. Measures dominant colours
  from the crop (PIL HSV buckets) and reports only colours; style is left
  empty on purpose, because a colour histogram cannot know the style.

The crop is the body region *below* the face: with a face bbox it starts
just under the chin, otherwise the lower 60% of the photo. Face-only crops
are not suitable input — callers must pass full photos/frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

log = logging.getLogger("recast")

DEFAULT_CLIP_MODEL = "ViT-B/32"
CLIP_MODEL_URL = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

STYLE_LABELS = [
    "古装",
    "现代装",
    "正装西装",
    "休闲装",
    "校服",
    "军装",
    "旗袍",
    "运动装",
    "礼服",
    "工作服",
]

COLOR_LABELS = [
    "红色",
    "橙色",
    "黄色",
    "绿色",
    "青色",
    "蓝色",
    "紫色",
    "粉色",
    "棕色",
    "黑色",
    "白色",
    "灰色",
]

# (hue_lo, hue_hi) in degrees, used by the heuristic colour bucketing.
_HUE_BANDS: list[tuple[str, float, float]] = [
    ("红色", 345.0, 15.0),
    ("橙色", 15.0, 40.0),
    ("黄色", 40.0, 70.0),
    ("绿色", 70.0, 160.0),
    ("青色", 160.0, 195.0),
    ("蓝色", 195.0, 250.0),
    ("紫色", 250.0, 290.0),
    ("粉色", 290.0, 345.0),
]


class CostumeError(RuntimeError):
    """Costume tagging cannot run: no usable photo, or the tagger is missing."""


class Tagger(Protocol):
    """Anything that can describe a costume crop."""

    name: str

    def tag(self, image: Image.Image) -> dict[str, Any]:
        """Return {"colors": [...], "style": [...], "description": "..."}."""
        ...


def _hue_name(h_deg: float, s: float, v: float) -> str:
    """Map one HSV pixel to a Chinese colour name (h in degrees, s/v 0..1)."""
    if v < 0.16:
        return "黑色"
    if s < 0.12:
        if v > 0.78:
            return "白色"
        if v > 0.45:
            return "灰色"
        return "黑色"
    if 15.0 <= h_deg < 40.0 and v < 0.6:
        return "棕色"
    for label, lo, hi in _HUE_BANDS:
        if lo <= hi:
            if lo <= h_deg < hi:
                return label
        elif h_deg >= lo or h_deg < hi:
            return label
    return "灰色"


def dominant_colors(
    image: Image.Image, *, top_n: int = 3, mask: Any = None
) -> list[dict[str, Any]]:
    """Top colours in the image as [{"color": ..., "share": 0.0..1.0}].

    ``mask`` (a PIL "L" image or an HxW array) restricts the count to those
    pixels. Without it a full-body portrait reports the studio background as
    its dominant colour — measured: 75% grey for a green cardigan — because
    the person does not fill the frame.
    """
    small = image.convert("RGB").resize((64, 64))
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    h_deg = hsv[..., 0] * 360.0 / 255.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    keep = None
    if mask is not None:
        # ndarray also has .resize (in-place, returns None), so check for the
        # PIL method that actually converts rather than for "resize".
        if hasattr(mask, "convert"):
            keep = np.asarray(mask.convert("L").resize((64, 64))) > 127
        else:
            keep = np.asarray(mask) > 0
            if keep.shape != (64, 64):
                import cv2

                keep = cv2.resize(keep.astype(np.uint8) * 255, (64, 64)) > 127
        keep = keep.ravel()
        if not keep.any():
            keep = None
    flat_h = h_deg.ravel()
    flat_s = s.ravel()
    flat_v = v.ravel()
    if keep is not None:
        flat_h, flat_s, flat_v = flat_h[keep], flat_s[keep], flat_v[keep]
    counts: dict[str, int] = {}
    for h, ss, vv in zip(flat_h, flat_s, flat_v):
        label = _hue_name(float(h), float(ss), float(vv))
        counts[label] = counts.get(label, 0) + 1
    total = max(1, len(flat_h))
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    out: list[dict[str, Any]] = []
    for label, count in ranked[:top_n]:
        share = count / total
        if share < 0.08 and out:
            continue
        out.append({"color": label, "share": round(share, 3)})
    return out or [{"color": "灰色", "share": 1.0}]


class HeuristicTagger:
    """Colour-only tagger. No model, never invents a style."""

    name = "heuristic"

    def tag(self, image: Image.Image) -> dict[str, Any]:
        colors = dominant_colors(image)
        names = [c["color"] for c in colors]
        return {
            "colors": names,
            "style": [],
            "description": f"主要颜色：{'、'.join(names)}。",
        }


class ClipTagger:
    """Local CLIP zero-shot tagger (ViT-B/32). Lazily imports torch/clip."""

    name = "clip"

    def __init__(self, model: str = DEFAULT_CLIP_MODEL, models_dir: str | Path | None = None) -> None:
        self.model = model
        self.models_dir = str(models_dir) if models_dir else ""
        self._model = None
        self._preprocess = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise CostumeError(
                "衣服打标签需要 torch，未安装。请先安装 CPU 版 torch："
                "pip install torch --index-url https://download.pytorch.org/whl/cpu。"
                "装好后第一次运行会自动下载 CLIP 模型。"
            ) from exc
        try:
            import clip as clip_pkg
        except ImportError as exc:
            raise CostumeError(
                "衣服打标签需要 openai-clip，未安装。请先安装："
                "pip install git+https://github.com/openai/CLIP.git。"
                "装好后第一次运行会自动下载 CLIP 模型。"
            ) from exc
        try:
            import os

            if self.models_dir:
                os.environ.setdefault("CLIP_MODEL_DIR", self.models_dir)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip_pkg.load(self.model, device=self._device)
        except Exception as exc:
            raise CostumeError(
                f"CLIP 模型 {self.model} 加载失败（{exc}）。"
                "请检查网络后重试，或手动下载模型放进 "
                f"{self.models_dir or '~/.cache/clip'} 再重试。"
                f"模型地址：{CLIP_MODEL_URL}"
            ) from exc

    def embed(self, image: Image.Image) -> list[float] | None:
        """Normalized CLIP image embedding of one crop (B1 body-appearance vote).

        Returns None when the crop cannot be encoded. Lazily loads the
        model exactly like :meth:`tag`; callers treat None as "body vote
        abstains", never as an error.
        """
        self._load()
        import torch

        assert self._model is not None and self._preprocess is not None
        try:
            image_input = self._preprocess(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(image_input)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return [float(x) for x in feats.cpu().numpy()[0]]
        except Exception:
            return None

    def tag(self, image: Image.Image) -> dict[str, Any]:
        self._load()
        import torch

        assert self._model is not None and self._preprocess is not None
        prompts = [f"一张穿着{label}的人的照片" for label in STYLE_LABELS + COLOR_LABELS]
        image_input = self._preprocess(image).unsqueeze(0).to(self._device)
        import clip as clip_pkg

        text_input = clip_pkg.tokenize(prompts).to(self._device)
        with torch.no_grad():
            logits, _ = self._model(image_input, text_input)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
        n_style = len(STYLE_LABELS)
        style_probs = probs[:n_style]
        color_probs = probs[n_style:]
        style = [STYLE_LABELS[i] for i in np.argsort(-style_probs)[:2] if style_probs[i] > 0.15]
        colors = [COLOR_LABELS[i] for i in np.argsort(-color_probs)[:3] if color_probs[i] > 0.08]
        parts: list[str] = []
        if colors:
            parts.append(f"主要颜色：{'、'.join(colors)}")
        if style:
            parts.append(f"款式：{'、'.join(style)}")
        return {
            "colors": colors,
            "style": style,
            "description": ("；".join(parts) + "。") if parts else "未能识别服装。",
        }


def get_tagger(kind: str, *, model: str = DEFAULT_CLIP_MODEL, models_dir: str | Path | None = None) -> Tagger:
    """Build a tagger by name: "clip" or "heuristic"."""
    kind = (kind or "clip").strip().lower()
    if kind == "heuristic":
        return HeuristicTagger()
    if kind == "clip":
        return ClipTagger(model=model, models_dir=models_dir)
    raise CostumeError(f"未知的衣服打标签方式：{kind}（可用 clip / heuristic）。")


def crop_costume(
    image: Image.Image,
    *,
    face_bbox: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Crop the body region below the face.

    ``face_bbox`` is (x0, y0, x1, y1) in pixels. Without it, the lower 60%
    of the photo is used (portraits put the face in the upper part).
    """
    width, height = image.size
    if face_bbox is not None:
        x0, y0, x1, y1 = (float(v) for v in face_bbox)
        face_w = max(1.0, x1 - x0)
        left = max(0, int(x0 - 0.5 * face_w))
        right = min(width, int(x1 + 0.5 * face_w))
        top = min(height - 1, int(y1 + 0.1 * face_h(face_bbox)))
        return image.crop((left, top, right, height))
    top = int(height * 0.4)
    return image.crop((0, top, width, height))


def face_h(face_bbox: tuple[float, float, float, float]) -> float:
    return max(1.0, face_bbox[3] - face_bbox[1])


def tag_costume_image(
    src: str | Path,
    *,
    tagger: Tagger | None = None,
    face_bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Tag one full photo: crop the costume region and describe it."""
    src = Path(src)
    if not src.is_file():
        raise CostumeError(f"找不到要识别服装的照片：{src}")
    if tagger is None:
        tagger = ClipTagger()
    try:
        image = Image.open(src).convert("RGB")
    except Exception as exc:
        raise CostumeError(f"读不懂照片 {src}：{exc}") from exc
    crop = crop_costume(image, face_bbox=face_bbox)
    if crop.size[0] < 8 or crop.size[1] < 8:
        raise CostumeError(f"服装区域太小，无法识别：{src}")
    result = tagger.tag(crop)
    return {**result, "source": str(src), "tagger": tagger.name}


def tag_character_folder(
    char_folder: str | Path,
    *,
    sources: list[str | Path] | None = None,
    tagger: Tagger | None = None,
    face_bboxes: dict[str, tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    """Tag a character's costume from full photos; write screenshots + meta.json.

    ``sources`` must be full photos/frames (face-only crops are not usable:
    there is no body region in them). Screenshots land in ``costume/`` and
    the description goes into meta.json ``costume``. Failures are recorded
    per photo; nothing is invented when tagging fails.
    """
    from recast.core.characters import load_meta, save_meta

    folder = Path(char_folder)
    if not folder.is_dir():
        raise CostumeError(f"找不到人物文件夹：{folder}")
    meta = load_meta(folder)
    name = (meta or {}).get("name") or folder.name
    if tagger is None:
        tagger = ClipTagger()
    if not sources:
        raise CostumeError(
            "没有可识别服装的全身照片。请传入原始照片/视频帧（人脸小图里没有衣服区域）。"
        )
    costume_dir = folder / "costume"
    costume_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "character": name,
        "tagger": tagger.name,
        "tagged": [],
        "failed": [],
        "description": "",
    }
    descriptions: list[str] = []
    files: list[str] = []
    for raw in sources:
        src = Path(raw)
        dest = costume_dir / f"{src.stem}.jpg"
        try:
            bbox = (face_bboxes or {}).get(src.stem) or (face_bboxes or {}).get(str(src))
            result = tag_costume_image(src, tagger=tagger, face_bbox=bbox)
            crop = crop_costume(Image.open(src).convert("RGB"), face_bbox=bbox)
            crop.save(dest, format="JPEG", quality=90)
            files.append(str(dest))
            descriptions.append(result.get("description") or "")
            report["tagged"].append(
                {
                    "photo": str(src),
                    "screenshot": str(dest),
                    "colors": result.get("colors") or [],
                    "style": result.get("style") or [],
                    "description": result.get("description") or "",
                }
            )
        except CostumeError as exc:
            report["failed"].append({"photo": str(src), "reason": str(exc)})
        except Exception as exc:  # keep going: one bad photo must not stop the rest
            report["failed"].append({"photo": str(src), "reason": str(exc)})
    if descriptions:
        best = max(descriptions, key=len)
        report["description"] = best
        if meta is None:
            meta = {"name": name}
        meta["costume"] = {
            "description": best,
            "source": "auto",
            "files": files,
            "missing": [],
        }
        save_meta(folder, meta)
    log.info(
        "人物「%s」衣服打标签完成: 成功 %s 张，失败 %s 张。",
        name,
        len(report["tagged"]),
        len(report["failed"]),
    )
    return report
