"""Character sheet: the reference board a 2D pipeline draws from.

The 3D model is not the final pixel anywhere in this product — replacing a
real actor's body with a CG body loses the hair, and the neck seam scores
4x the body's own edge energy no matter how it is composited. What the 3D
asset *is* good for is consistency: one canonical set of angles and
appearance facts, so every episode of every market version draws the same
character.

Outputs live in ``<character>/sheet/``:

    turn_000.png .. turn_270.png   # GLB turntable, if the GLB is readable
    appearance.png                 # swatch strip of the measured palette
    sheet.json                     # angles + appearance + provenance

Every field records where it came from (``"source": "3d" | "photo" |
"model"``). A field that could not be measured is left empty with a reason,
never filled with a guess: a wrong hair colour silently propagates into
every downstream redraw.

Rendering is optional. Without trimesh+pyrender or a GL context the sheet
still builds from photos, and ``model_rendered`` says so.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

TURNTABLE_ANGLES: tuple[int, ...] = (0, 45, 90, 180)
SHEET_DIR = "sheet"
SHEET_FILENAME = "sheet.json"
DEFAULT_SIZE = 512
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SheetError(RuntimeError):
    """A character sheet cannot be built at all (no photos and no model)."""


# ---------------------------------------------------------------------------
# GLB turntable
# ---------------------------------------------------------------------------


def _camera_pose(eye: Any, target: Any, up: Any = (0.0, 1.0, 0.0)) -> Any:
    import numpy as np

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    z = eye - target
    norm = np.linalg.norm(z)
    if norm < 1e-9:
        z = np.array([0.0, 0.0, 1.0])
    else:
        z = z / norm
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:  # camera looking straight down the up axis
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, z
    pose[:3, 3] = eye
    return pose


def _upright_rotation(meshes: list[Any]) -> Any:
    """Rotation that puts the model's longest axis on +Y.

    glTF says Y-up, but plenty of character assets ship Z-up (CesiumMan,
    most Blender exports without the +Y conversion). Rendering a Z-up model
    with a Y-up camera gives a body lying on its back, which looks like a
    renderer bug rather than an asset convention. The longest axis is the
    height for anything human-shaped, so that is the axis to trust.
    """
    import numpy as np
    import trimesh

    bounds = np.vstack([m.bounds for m in meshes])
    extents = bounds[1] - bounds[0]
    axis = int(np.argmax(extents))
    if axis == 1:
        return np.eye(4)
    if axis == 2:  # +Z -> +Y
        return trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0])
    return trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0])  # +X -> +Y


def render_turntable(
    glb_path: str | Path,
    *,
    size: int = DEFAULT_SIZE,
    angles: tuple[int, ...] = TURNTABLE_ANGLES,
) -> dict[str, Any]:
    """Render the GLB from ``angles`` (degrees about the model's up axis).

    Returns ``{"images": [(angle, ndarray), ...], "reason": str|None}``.
    ``images`` is empty and ``reason`` explains why when rendering is not
    possible — this never raises and never fabricates a frame.
    """
    images: list[tuple[int, Any]] = []
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        return {"images": images, "reason": f"需要 trimesh：{exc}"}
    try:
        import pyrender
    except ImportError as exc:
        return {"images": images, "reason": f"需要 pyrender：{exc}"}

    path = Path(glb_path)
    if not path.is_file():
        return {"images": images, "reason": f"找不到模型文件：{path}"}
    try:
        scene_in = trimesh.load(str(path), force="scene")
        meshes = [
            g for g in scene_in.geometry.values() if getattr(g, "faces", None) is not None and len(g.faces)
        ]
    except Exception as exc:
        return {"images": images, "reason": f"GLB 读不出来：{exc}"}
    if not meshes:
        return {"images": images, "reason": "GLB 里没有可渲染的网格"}

    # Stand the model up first, then measure; a Z-up asset would otherwise
    # be framed by its width and overflow the image.
    upright = _upright_rotation(meshes)
    standing = []
    for mesh in meshes:
        copy = mesh.copy()
        copy.apply_transform(upright)
        standing.append(copy)

    bounds = np.vstack([m.bounds for m in standing])
    lo, hi = bounds[0], bounds[1]
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    if radius <= 1e-6:
        return {"images": images, "reason": "网格尺寸为零，无法构图"}

    try:
        renderer = pyrender.OffscreenRenderer(size, size)  # type: ignore[arg-type]
    except Exception as exc:
        return {"images": images, "reason": f"拿不到 GL 上下文，无法离屏渲染：{exc}"}

    try:
        yfov = np.pi / 4.0
        distance = radius / float(np.tan(yfov / 2.0)) * 1.25
        for angle in angles:
            try:
                scene = pyrender.Scene(
                    bg_color=[0.42, 0.42, 0.42, 1.0], ambient_light=[0.55, 0.55, 0.55]
                )
                theta = np.deg2rad(float(angle))
                spin = trimesh.transformations.rotation_matrix(theta, [0.0, 1.0, 0.0])
                spin[:3, 3] = centre - spin[:3, :3] @ centre
                for mesh in standing:
                    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True), pose=spin)
                camera = pyrender.PerspectiveCamera(yfov=yfov)
                eye = centre + np.array([0.0, 0.0, distance])
                scene.add(camera, pose=_camera_pose(eye, centre))
                key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
                scene.add(key, pose=_camera_pose(centre + np.array([1.0, 1.2, distance]), centre))
                fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)
                scene.add(fill, pose=_camera_pose(centre + np.array([-1.0, 0.4, distance]), centre))
                colour, _ = renderer.render(scene)
                images.append((int(angle), np.asarray(colour)))
            except Exception as exc:  # one bad angle must not kill the board
                log.warning("转盘渲染失败 %s°: %s", angle, exc)
    finally:
        try:
            renderer.delete()
        except Exception:
            pass
    if not images:
        return {"images": images, "reason": "所有角度都渲染失败"}
    return {"images": images, "reason": None}


# ---------------------------------------------------------------------------
# Appearance facts (only what can be measured)
# ---------------------------------------------------------------------------


def _palette(bgr: Any, *, top_n: int = 5, mask: Any = None) -> list[dict[str, Any]]:
    """Dominant colours, optionally restricted to the person."""
    from recast.stages.costume import dominant_colors

    from PIL import Image

    image = Image.fromarray(bgr[:, :, ::-1])
    return dominant_colors(image, top_n=top_n, mask=mask)


def _person_mask(bgr: Any) -> Any:
    """Person silhouette for palette counting, or None when no mask is here.

    A full-body portrait is mostly background (measured: 75% studio grey for
    a green cardigan), so the palette is meaningless without this. This goes
    through the same ``segment_person_auto`` / ``pick_backend("auto")`` chain
    as the rest of the pipeline: calling ``rmbg_foreground`` directly made the
    palette silently fall back to full-photo statistics whenever the RMBG
    weights were absent — background pixels passed off as the character's
    palette. None is returned only when no backend produced a usable person
    mask (and ``palette_source`` then honestly says ``full_photo``); a
    rectangle-only GrabCut result ("box") counts as no mask, because a box
    mixes the background inside it into the palette. Lazy import on purpose:
    ``person_mask`` imports this module for face boxes.
    """
    try:
        from recast.stages.person_mask import segment_person_auto

        result = segment_person_auto(bgr, backend="auto")
    except Exception as exc:
        log.info("人物分割不可用，调色板如实回退全图统计: %s", exc)
        return None
    if str(result.get("method") or "") == "box":
        log.info(
            "人物分割只给出矩形框（%s），调色板如实回退全图统计",
            result.get("warning") or "",
        )
        return None
    import numpy as np

    mask = np.asarray(result["mask"]) > 0
    if int(mask.sum()) < 0.02 * mask.size:
        return None
    return mask


def _installed_face_pack() -> str | None:
    """Name of a face pack already on disk, preferred largest-capability first.

    Asking insightface for ``buffalo_l`` when only ``buffalo_s`` is installed
    makes it try to download, which hangs or fails on an offline box. Look at
    what is actually there first.
    """
    root = Path.home() / ".insightface" / "models"
    if not root.is_dir():
        return None
    for name in ("buffalo_l", "antelopev2", "buffalo_m", "buffalo_s", "buffalo_sc"):
        if (root / name).is_dir():
            return name
    return None


def _face_box(bgr: Any) -> list[float] | None:
    """Face bbox from a locally installed insightface pack, or None.

    Detection only: the recognition/gender models are not needed here and
    pulling them in would slow the call down for nothing.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return None
    pack = _installed_face_pack()
    if pack is None:
        return None
    try:
        app = FaceAnalysis(name=pack, allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        faces = app.get(bgr)
    except Exception as exc:
        log.info("人脸检测不可用（%s）: %s", pack, exc)
        return None
    if not faces:
        return None
    box = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))).bbox
    return [float(v) for v in box]


def skin_tone(bgr: Any, box: list[float]) -> dict[str, Any]:
    """Median cheek colour inside a face box, in Lab, plus a human name."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    # Cheek band: below the eyes, inside the face — avoids hair and lips.
    cx0 = int(x0 + 0.22 * w)
    cx1 = int(x0 + 0.78 * w)
    cy0 = int(y0 + 0.58 * h)
    cy1 = int(y0 + 0.80 * h)
    patch = bgr[max(0, cy0) : cy1, max(0, cx0) : cx1]
    if patch.size == 0:
        return {"lab": None, "name": "", "source": "photo", "reason": "脸框太小"}
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    median = np.median(lab, axis=0)
    lightness = float(median[0]) / 255.0
    if lightness > 0.82:
        name = "白皙"
    elif lightness > 0.68:
        name = "自然"
    elif lightness > 0.52:
        name = "小麦"
    else:
        name = "深色"
    return {
        "lab": [round(float(v), 1) for v in median],
        "name": name,
        "source": "photo",
        "reason": "",
    }


def appearance_from_photos(paths: list[Path]) -> dict[str, Any]:
    """Measurable appearance facts from reference photos.

    Only two things are claimed: a dominant palette (every photo has one)
    and a skin tone (only when a face is actually found). Hair colour,
    height and build are left empty on purpose — measuring them from an
    unsegmented photo would be a guess wearing a number.
    """
    import cv2
    import numpy as np

    out: dict[str, Any] = {
        "palette": [],
        "skin": {"lab": None, "name": "", "source": "photo", "reason": "没有可读的参考照"},
        "hair": {"name": "", "source": "unmeasured", "reason": "未做分割，不猜发色"},
        "build": {"name": "", "source": "unmeasured", "reason": "未做人体测量，不猜体型"},
        "photos_used": [],
        "palette_source": "full_photo",
    }
    for path in paths:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            continue
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        out["photos_used"].append(str(path))
        if not out["palette"]:
            person = _person_mask(bgr)
            out["palette"] = _palette(bgr, mask=person)
            out["palette_source"] = "person" if person is not None else "full_photo"
        box = _face_box(bgr)
        if box is not None and not out["skin"]["name"]:
            out["skin"] = skin_tone(bgr, box)
    return out


def _swatch_strip(palette: list[dict[str, Any]], size: int = DEFAULT_SIZE) -> Any:
    """A simple swatch strip so the board shows the palette, not just hexes."""
    import numpy as np

    swatches = max(1, len(palette))
    h = max(48, size // 6)
    strip = np.full((h, size, 3), 32, dtype=np.uint8)
    for i, item in enumerate(palette):
        x0 = int(i * size / swatches)
        x1 = int((i + 1) * size / swatches)
        rgb = _colour_probe(str(item.get("color") or ""))
        strip[:, x0:x1] = rgb
    return strip


_PROBE = {
    "红色": (60, 60, 220),
    "橙色": (60, 140, 240),
    "黄色": (80, 210, 235),
    "绿色": (80, 160, 90),
    "青色": (200, 200, 80),
    "蓝色": (200, 120, 60),
    "紫色": (200, 80, 160),
    "粉色": (200, 140, 220),
    "棕色": (60, 90, 140),
    "黑色": (30, 30, 30),
    "白色": (240, 240, 240),
    "灰色": (140, 140, 140),
}


def _colour_probe(name: str) -> tuple[int, int, int]:
    return _PROBE.get(name, (120, 120, 120))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _photo_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for sub in ("face", "costume"):
        directory = folder / sub
        if directory.is_dir():
            paths.extend(sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
    return paths


def _model_path(folder: Path, meta: dict[str, Any]) -> Path | None:
    declared = ((meta.get("model") or {}).get("path") or "")
    if declared:
        candidate = folder / declared
        if candidate.is_file():
            return candidate
    model_dir = folder / "model"
    if model_dir.is_dir():
        for suffix in (".glb", ".gltf"):
            found = sorted(p for p in model_dir.iterdir() if p.suffix.lower() == suffix)
            if found:
                return found[0]
    return None


def build_character_sheet(folder: str | Path, *, size: int = DEFAULT_SIZE) -> dict[str, Any]:
    """Build the sheet for one character folder and register it in meta.json.

    Returns the report (also written to ``sheet/sheet.json``).
    """
    import cv2

    from recast.core.characters import load_meta, save_meta

    folder = Path(folder)
    if not folder.is_dir():
        raise SheetError(f"找不到人物文件夹：{folder}")
    meta = load_meta(folder) or {}
    name = str(meta.get("name") or folder.name)

    photos = _photo_paths(folder)
    model = _model_path(folder, meta)
    if not photos and model is None:
        raise SheetError(
            f"人物「{name}」既没有参考照片（face/ costume/）也没有 3D 模型（model/*.glb），"
            "建不出设定板。"
        )

    out_dir = folder / SHEET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    turns: list[dict[str, Any]] = []
    render_reason: str | None = None
    if model is not None:
        result = render_turntable(model, size=size)
        render_reason = result["reason"]
        for angle, image in result["images"]:
            dest = out_dir / f"turn_{angle:03d}.png"
            ok, buf = cv2.imencode(".png", image)
            if ok:
                buf.tofile(str(dest))
                turns.append({"angle": angle, "path": f"{SHEET_DIR}/{dest.name}", "source": "3d"})
    else:
        render_reason = "没有 GLB 模型：设定板只用照片，角度图为空"

    appearance = appearance_from_photos(photos)
    palette = appearance.get("palette") or []
    if palette:
        ok, buf = cv2.imencode(".png", _swatch_strip(palette, size=size))
        if ok:
            buf.tofile(str(out_dir / "appearance.png"))

    report: dict[str, Any] = {
        "character": name,
        "folder": str(folder),
        "model": None if model is None else str(model),
        "model_rendered": bool(turns),
        "render_reason": render_reason,
        "turns": turns,
        "appearance": appearance,
        "photos_used": appearance.get("photos_used") or [],
        "built_at": _now(),
    }
    (out_dir / SHEET_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta["appearance"] = {
        "palette": palette,
        "skin": appearance.get("skin"),
        "hair": appearance.get("hair"),
        "build": appearance.get("build"),
    }
    meta["sheet"] = {
        "dir": SHEET_DIR,
        "turns": len(turns),
        "model_rendered": bool(turns),
        "reason": render_reason or "",
        "built_at": report["built_at"],
    }
    save_meta(folder, meta)
    log.info(
        "人物「%s」设定板已建：%s 个角度，%s 张照片。%s",
        name,
        len(turns),
        len(photos),
        render_reason or "",
    )
    return report


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
