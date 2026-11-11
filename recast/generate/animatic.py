"""Honest offline fallback: Ken Burns / held-frame silent animatic from reference stills.

This is labelled 「分镜预览成片」— not full generative video, and not Wan/AniCrafter.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from recast.ffmpeg import run_ffmpeg, which_ffmpeg

log = logging.getLogger("recast")

ANIMATIC_LABEL = "分镜预览成片"
ANIMATIC_NOTE = (
    "本段是「分镜预览成片」：用参考静帧做缓慢推拉（Ken Burns / 定格），"
    "不是完整的图生视频，也不是默认的 Wan / AniCrafter 生成。"
)


def _write_note(dest: Path) -> None:
    dest.write_text(ANIMATIC_NOTE + "\n", encoding="utf-8")


def _plate(shot: dict[str, Any], width: int, height: int, dest: Path) -> Path:
    """Create a simple color plate with the shot code when no reference image exists."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (width, height), (62, 48, 36))
        draw = ImageDraw.Draw(img)
        text = shot.get("shot_code") or ""
        scene = shot.get("scene_id") or shot.get("scene") or ""
        label = "\n".join(x for x in (text, scene, "环境底板") if x)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
        draw.multiline_text((40, height // 3), label, fill=(230, 214, 180), font=font, spacing=10)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return dest
    except Exception:
        # last resort: tiny PPM then let ffmpeg scale
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_ppm(width, height))
        return dest


def _ppm(width: int, height: int) -> bytes:
    # Not used if PIL works; keep as fallback binary header + gray
    row = bytes([70, 54, 40]) * width
    return f"P6\n{width} {height}\n255\n".encode("ascii") + row * height


def _collage(paths: list[Path], width: int, height: int, dest: Path) -> Path:
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in paths]
    n = max(1, len(images))
    if n == 1:
        img = images[0]
        img = _cover(img, width, height)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return dest
    # side by side
    cell_w = width // n
    canvas = Image.new("RGB", (width, height), (20, 16, 12))
    for i, im in enumerate(images):
        fitted = _cover(im, cell_w, height)
        canvas.paste(fitted, (i * cell_w, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return dest


def _cover(img: Any, width: int, height: int) -> Any:
    from PIL import Image

    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new = img.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.Resampling.LANCZOS)
    left = max(0, (new.size[0] - width) // 2)
    top = max(0, (new.size[1] - height) // 2)
    return new.crop((left, top, left + width, top + height))


def _ffmpeg_kenburns(still: Path, dest: Path, duration: float, width: int, height: int, fps: int) -> Path:
    frames = max(int(round(duration * fps)), fps)
    # Slow zoom 1.0 → 1.08 over the shot. Held-frame if duration is tiny.
    zstep = 0.08 / max(frames, 1)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"zoompan=z='min(zoom+{zstep:.6f},1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d=1:s={width}x{height}:fps={fps},"
        f"fps={fps},format=yuv420p"
    )
    run_ffmpeg(
        [
            "-y",
            "-loop",
            "1",
            "-i",
            str(still),
            "-vf",
            vf,
            "-t",
            f"{duration:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )
    return dest


def build_shot_clip(
    shot: dict[str, Any],
    dest: Path,
    *,
    refs: list[Path] | None = None,
    width: int = 720,
    height: int = 1280,
    fps: int = 24,
) -> Path:
    duration = float(shot.get("duration_sec") or 0)
    if duration <= 0 and shot.get("start_time") is not None and shot.get("end_time") is not None:
        duration = float(shot["end_time"]) - float(shot["start_time"])
    duration = max(duration, 0.4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    still = dest.with_suffix(".still.jpg")
    refs = [p for p in (refs or []) if p and Path(p).is_file()]
    if refs:
        _collage([Path(p) for p in refs], width, height, still)
    else:
        _plate(shot, width, height, still if still.suffix.lower() in {".jpg", ".jpeg", ".png"} else dest.parent / (dest.stem + ".still.jpg"))
        if not still.is_file():
            still = dest.parent / (dest.stem + ".still.jpg")
    try:
        _ffmpeg_kenburns(still, dest, duration, width, height, fps)
    except Exception as exc:
        log.warning("Ken Burns 失败，改为静帧拉长时间: %s", exc)
        # held frame
        run_ffmpeg(
            [
                "-y",
                "-loop",
                "1",
                "-i",
                str(still),
                "-t",
                f"{duration:.3f}",
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps},format=yuv420p",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ]
        )
    return dest


def build_animatic(
    shots: list[dict[str, Any]],
    workdir: str | Path,
    *,
    pack: Any | None = None,
    width: int = 720,
    height: int = 1280,
    fps: int = 24,
) -> dict[str, Any]:
    """Build one silent clip per shot and a concatenated silent picture track."""
    workdir = Path(workdir)
    shots_dir = workdir / "shots_silent"
    shots_dir.mkdir(parents=True, exist_ok=True)
    _write_note(workdir / "ANIMATIC.txt")
    clips: list[Path] = []
    for i, shot in enumerate(shots):
        code = shot.get("shot_code") or f"S{i+1:02d}"
        dest = shots_dir / f"{code}.mp4"
        refs: list[Path] = []
        if pack is not None and hasattr(pack, "resolve_shot_refs"):
            refs = pack.resolve_shot_refs(shot)
        else:
            for item in shot.get("ref_assets") or []:
                token = item.get("file") if isinstance(item, dict) else item
                if token and Path(str(token)).is_file():
                    refs.append(Path(str(token)))
        path = build_shot_clip(shot, dest, refs=refs, width=width, height=height, fps=fps)
        shot["output_path"] = str(path)
        shot["animatic"] = True
        clips.append(path)

    silent = workdir / "silent.mp4"
    if clips:
        _concat(clips, silent)
    return {
        "label": ANIMATIC_LABEL,
        "note": ANIMATIC_NOTE,
        "silent": str(silent) if silent.is_file() else None,
        "clips": [str(c) for c in clips],
        "shots": shots,
    }


def _concat(clips: list[Path], dest: Path) -> Path:
    lst = dest.parent / "silent_concat.txt"
    lines = []
    for c in clips:
        # concat demuxer needs escaped single quotes
        s = str(c.resolve()).replace("'", "'\\''")
        lines.append(f"file '{s}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dest)])
    except Exception:
        run_ffmpeg(
            [
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest),
            ]
        )
    return dest
