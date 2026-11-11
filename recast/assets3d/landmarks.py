"""Face landmarks + albedo crop. InsightFace if installed, otherwise a center crop fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def extract_landmarks(image: str | Path) -> dict[str, Any]:
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"找不到参考脸: {path}")
    try:
        return _insightface_landmarks(path)
    except Exception:
        return _fallback_box(path)


def _insightface_landmarks(path: Path) -> dict[str, Any]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"无法读取图像: {path}")
    faces = app.get(img)
    if not faces:
        raise RuntimeError("未检测到人脸")
    face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
    kps = face.kps.tolist() if getattr(face, "kps", None) is not None else []
    bbox = [float(x) for x in face.bbox.tolist()]
    embedding = face.normed_embedding.tolist() if getattr(face, "normed_embedding", None) is not None else None
    h, w = img.shape[:2]
    return {
        "backend": "insightface",
        "image": str(path),
        "width": w,
        "height": h,
        "bbox": bbox,
        "landmarks5": kps,
        "embedding": embedding,
        "det_score": float(getattr(face, "det_score", 0) or 0),
    }


def _fallback_box(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        w, h = 512, 512
    # a centered 60% box and 5 guessed points (eyes / nose / mouth)
    x0, y0 = int(w * 0.2), int(h * 0.15)
    x1, y1 = int(w * 0.8), int(h * 0.85)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    kps = [
        [cx - (x1 - x0) * 0.18, cy - (y1 - y0) * 0.12],
        [cx + (x1 - x0) * 0.18, cy - (y1 - y0) * 0.12],
        [cx, cy],
        [cx - (x1 - x0) * 0.15, cy + (y1 - y0) * 0.22],
        [cx + (x1 - x0) * 0.15, cy + (y1 - y0) * 0.22],
    ]
    return {
        "backend": "center-crop-fallback",
        "image": str(path),
        "width": w,
        "height": h,
        "bbox": [x0, y0, x1, y1],
        "landmarks5": kps,
        "embedding": None,
        "det_score": 0.0,
        "note": "未使用 InsightFace，仅按画面中心框估计。质量分会偏低。",
    }


def crop_albedo(image: str | Path, landmarks: dict[str, Any], dest: str | Path, size: int = 512) -> Path:
    from PIL import Image

    path = Path(image)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    bbox = landmarks.get("bbox") or [0, 0, 0, 0]
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        x0, y0, x1, y1 = [float(x) for x in bbox]
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, w, h
        # pad
        pw, ph = (x1 - x0) * 0.25, (y1 - y0) * 0.35
        box = (max(0, x0 - pw), max(0, y0 - ph), min(w, x1 + pw), min(h, y1 + pw * 0.2))
        crop = im.crop(box).resize((size, size), Image.Resampling.LANCZOS)
        crop.save(dest)
    return dest


def quality_score(landmarks: dict[str, Any], n_photos: int = 1) -> float:
    score = 0.4
    if landmarks.get("backend") == "insightface":
        score += 0.3
        score += min(0.2, float(landmarks.get("det_score") or 0) * 0.2)
    if landmarks.get("embedding"):
        score += 0.1
    if n_photos >= 2:
        score += 0.1
    if landmarks.get("backend") == "center-crop-fallback":
        score = min(score, 0.45)
    return round(min(1.0, score), 3)


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
