#!/usr/bin/env bash
# pilot-render-env.sh -- restore the body re-render pilot environment on
# this headless VM after a restart wiped the pip packages.
#
# What runs the renders (verified 2026-11-12):
#   * GL backend: EGL surfaceless + Mesa llvmpipe (software rasteriser).
#     The Mesa DRI stack is already installed system-wide
#     (libegl-mesa0, libgl1-mesa-dri, libglx-mesa0, all 25.2.8) -- no apt
#     install is needed, and OSMesa was NOT needed. Set
#         PYOPENGL_PLATFORM=egl
#     before any python that creates an OffscreenRenderer. Without it,
#     pyrender falls back to pyglet/X11 and dies with
#     "NoSuchDisplayException: Cannot connect to None" (no X server here).
#   * pip packages (user site, system python 3.12):
#         trimesh 5.2.0, pyrender 0.1.45, PyOpenGL 3.1.0
#         mediapipe 0.10.21  (only needed for the composite step:
#         recast/stages/rerender.py composite_clip uses
#         mp.solutions.selfie_segmentation; the render step does not
#         need it)
#     Install with --break-system-packages --no-cache-dir and
#     TMPDIR=~/tmp_pip : /tmp is a 512M tmpfs and fills up during big
#     wheel installs, and the pip cache is what filled it before.
#     NOTE: this pulls numpy down from 2.5.4 to 2.2.6 (a trimesh/pyrender
#     dependency bound) and mediapipe pulls opencv-contrib-python; do NOT
#     install plain `mediapipe` (it resolves to 1.2.3, whose import pulls
#     in the Debian system matplotlib, which is compiled against numpy
#     1.x and crashes under numpy 2.x) -- pin 0.10.21, the version the
#     pilot's pose.json was built with.
#   * ffmpeg/ffprobe are system binaries (/usr/bin/ffmpeg) for the
#     composite writer; Blender is NOT used (there is a Blender 4.2.0
#     tree under ~/workspace/local-recast-audit/blender/ but the pilot
#     renders with pyrender, not bpy).
#
# Render smoke test (should print RENDER OK, ~5 s):
#   PYOPENGL_PLATFORM=egl python3 scripts/render_pilot_frames.py \
#       --frame 60 --force --out-dir /tmp/check60
#
# Full resume run (frames 86-119, ~0.3-1 s/frame at 360x640 on llvmpipe):
#   PYOPENGL_PLATFORM=egl python3 scripts/render_pilot_frames.py \
#       --start 86 --end 119
set -euo pipefail

export TMPDIR="${TMPDIR:-$HOME/tmp_pip}"
mkdir -p "$TMPDIR"

pip3 install --break-system-packages --no-cache-dir --user \
    trimesh pyrender PyOpenGL "mediapipe==0.10.21"

python3 - <<'PY'
import trimesh, pyrender, OpenGL, mediapipe
print("trimesh", trimesh.__version__)
print("pyrender ok")
print("PyOpenGL", OpenGL.__version__)
print("mediapipe", mediapipe.__version__)
PY

echo "render env OK -- remember: PYOPENGL_PLATFORM=egl"
