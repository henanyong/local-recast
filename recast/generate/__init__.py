"""Generate-from-script path: silent picture first, then dub."""

from recast.generate.animatic import build_animatic, build_shot_clip
from recast.generate.comfy import comfy_available, submit_workflow
from recast.generate.stitch import stitch_and_dub
from recast.generate.tts import synthesize_line, synthesize_table

__all__ = [
    "build_animatic",
    "build_shot_clip",
    "comfy_available",
    "stitch_and_dub",
    "submit_workflow",
    "synthesize_line",
    "synthesize_table",
]
