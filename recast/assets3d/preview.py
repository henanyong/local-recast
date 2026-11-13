"""Render a GLB body to a front-view preview PNG (best effort, never fake).

The assembly view shows the selected body as a picture, not just a file
name. This module renders it with trimesh + pyrender (both optional:
``pip install trimesh pyrender``). It never raises for a rendering
problem and never writes a fake image:

- trimesh / pyrender missing -> None
- GLB unreadable, empty scene, no GL context -> None
- anything else unexpected during rendering -> None (logged)

The caller (service/UI) then shows a placeholder card with the model info
and "请用 Blender 打开查看" instead of pretending a render exists.

Verified on this headless VM (2026-11-12): rendering does NOT work here —
EGL finds no devices (no GPU), and under Xvfb the PyOpenGL array handlers
are broken with numpy 2.x (``No array-type handler ... registered`` when
calling glGenTextures), even with PyOpenGL-accelerate. So on this machine
``render_model_preview`` returns None. On Windows with a real GPU and a
normal Python environment the same code path should render; if it does
not, the UI still degrades to the placeholder instead of crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

PREVIEW_FILENAME = "preview.png"
DEFAULT_SIZE = 480


def _camera_pose(eye, target, up=(0.0, 1.0, 0.0)):
    """Camera-to-world matrix: camera at ``eye`` looking at ``target``."""
    import numpy as np

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _render_scene(scene: Any, width: int, height: int):
    """Render a trimesh scene to an RGB array with pyrender, or None."""
    import numpy as np
    import pyrender

    bounds = np.asarray(scene.bounds, dtype=float)
    center = bounds.mean(axis=0)
    size = float((bounds[1] - bounds[0]).max())
    if not size or size <= 0:
        return None
    pr = pyrender.Scene.from_trimesh_scene(scene, ambient_light=[0.4, 0.4, 0.4])
    pr.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
        pose=_camera_pose(center + np.array([0.0, 0.0, 3.0 * size]), center),
    )
    pr.add(
        pyrender.PerspectiveCamera(yfov=np.pi / 4),
        pose=_camera_pose(center + np.array([0.0, 0.0, 2.5 * size]), center),
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _depth = renderer.render(pr)
    finally:
        renderer.delete()
    if color is None or color.size == 0:
        return None
    return np.asarray(color[:, :, :3], dtype=np.uint8)


def render_model_preview(
    glb_path: str | Path,
    dest: str | Path,
    *,
    width: int = DEFAULT_SIZE,
    height: int = DEFAULT_SIZE,
    renderer: Any | None = None,
) -> Path | None:
    """Render ``glb_path`` to a front-view PNG at ``dest``.

    Returns the dest path on success, None when rendering is unavailable
    or fails (missing libraries, bad GLB, no GL context). ``renderer`` is
    an injectable ``(trimesh_scene, width, height) -> RGB array`` used by
    tests; the real one is :func:`_render_scene`.
    """
    from recast.assets3d.glb import validate_glb

    glb = Path(glb_path).expanduser()
    dest = Path(dest).expanduser()
    if not glb.is_file():
        log.warning("3D 预览跳过：找不到模型文件：%s", glb)
        return None
    try:
        validate_glb(glb)
    except ValueError as exc:
        log.warning("3D 预览跳过：模型文件已损坏：%s（%s）", glb, exc)
        return None
    try:
        import trimesh
    except ImportError:
        log.info("3D 预览不可用：没有安装 trimesh（pip install trimesh pyrender）。")
        return None
    try:
        scene = trimesh.load(str(glb), force="scene")
        geometries = getattr(scene, "geometry", None)
        if geometries is not None and not geometries:
            log.warning("3D 预览跳过：模型里没有几何体：%s", glb)
            return None
        render = renderer if renderer is not None else _render_scene
        color = render(scene, width, height)
        if color is None:
            return None
        import numpy as np
        from PIL import Image

        arr = np.asarray(color)
        if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr[:, :, :3].astype(np.uint8)).save(dest)
    except Exception as exc:  # noqa: BLE001 - any rendering failure is a fallback
        log.warning("3D 预览渲染失败（已回落到占位卡片）：%s（%s）", glb, exc)
        return None
    log.info("3D 预览已生成：%s", dest)
    return dest
