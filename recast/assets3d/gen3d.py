"""L3 「3D道具生成」pilot.

Pluggable backend (ComfyUI Hunyuan3D / Tripo workflow if configured).
If no backend is available, export a documented job file and a simple
textured card/box placeholder, clearly labelled 试点/占位结果.
Never presented as a finished character model. Characters are not auto-rigged.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PLACEHOLDER_LABEL = "试点/占位结果"
HONESTY = (
    "这是 L3「3D道具生成」的试点/占位结果：一张带贴图的简单盒体或卡片，"
    "不是成品角色模型，也不会自动绑定骨骼。"
)


def _box_obj(dest: Path, texture: str = "albedo.png") -> Path:
    mtl = dest.with_suffix(".mtl")
    mtl.write_text(f"newmtl albedo\nKd 1 1 1\nmap_Kd {texture}\n", encoding="utf-8")
    # unit box
    verts = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (4, 3, 7, 8),
        (1, 4, 8, 5),
        (2, 6, 7, 3),
    ]
    lines = [f"mtllib {mtl.name}", "# Placeholder prop box — 试点/占位结果", "o PropBox", "usemtl albedo"]
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    for f in faces:
        lines.append(f"f {f[0]}/1 {f[1]}/2 {f[2]}/3 {f[3]}/4")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _texture(dest: Path, prompt: str, image: Path | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if image and image.is_file():
        try:
            from PIL import Image

            Image.open(image).convert("RGB").resize((512, 512)).save(dest)
            return dest
        except Exception:
            dest.write_bytes(image.read_bytes())
            return dest
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (512, 512), (90, 78, 62))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        draw.multiline_text((24, 200), f"{PLACEHOLDER_LABEL}\n{prompt[:80]}", fill=(240, 230, 210), font=font, spacing=8)
        img.save(dest)
    except Exception:
        dest.write_bytes(b"")
    return dest


def generate_prop(
    dest_dir: str | Path,
    *,
    prompt: str = "",
    image: str | Path | None = None,
    comfyui_url: str | None = None,
    workflow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    backend = "placeholder"
    outputs: list[str] = []
    if comfyui_url and workflow:
        try:
            from recast.generate.comfy import comfy_available, submit_workflow

            if comfyui_url and comfy_available(comfyui_url):
                files = submit_workflow(
                    comfyui_url,
                    workflow,
                    variables={"prompt": prompt, "image": str(image or "")},
                    dest_dir=dest / "comfy",
                )
                outputs = [str(f) for f in files]
                backend = "comfyui"
        except Exception as exc:
            (dest / "comfy_error.txt").write_text(str(exc), encoding="utf-8")

    albedo = _texture(dest / "albedo.png", prompt or "prop", Path(image) if image else None)
    obj = _box_obj(dest / "placeholder.obj", texture=albedo.name)
    job = {
        "kind": "3d_prop",
        "label": "3D道具生成",
        "result_label": PLACEHOLDER_LABEL,
        "prompt": prompt,
        "image": str(image) if image else None,
        "backend": backend,
        "comfyui_url": comfyui_url or None,
        "outputs": outputs,
        "files": {"obj": str(obj), "albedo": str(albedo)},
        "rigged": False,
        "character_model": False,
        "honesty": HONESTY,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (dest / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (dest / "README.txt").write_text(HONESTY + "\n", encoding="utf-8")
    return {**job, "dir": str(dest)}
