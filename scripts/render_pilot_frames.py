"""Render body re-render pilot frames: retarget.json + Xbot.glb -> RGBA PNGs.

Rebuilt 2026-11-12 (the previous driver script was lost when the VM
restarted). Pipeline, reverse-engineered from the 86 surviving frames
(output/pilot/render/frame_0000.png .. frame_0085.png) and the retarget
math in recast/stages/rerender.py:

* Pose: per-frame bone rotations from retarget.json are LOCAL quaternions
  (parent_world^-1 * world). Forward kinematics rebuilds the joint world
  matrices from the GLB rest skeleton (recast.stages.rerender.
  load_glb_skeleton); the per-frame root_matrix (body -> camera) is applied
  to the whole model at the Armature level (rotate about the origin -- this
  matches the surviving frames: the head rises ~0.11 m from frame 0 to
  frame 60 exactly as the Armature-root variant predicts, while the
  hips-root variant only rises ~0.03 m).
* Scale: retarget.json "scale" (0.784 for this pilot) multiplies every
  joint offset, i.e. the Armature node scale.
* Skinning is done on the CPU (pyrender has no skinning support): linear
  blend skinning from the GLB JOINTS_0 / WEIGHTS_0 attributes and the skin's
  inverse-bind matrices. Unit trap in this GLB: the stored IBMs are
  authored in cm (rotation x100, e.g. hips translation -103.99) while the
  POSITION accessor is already in metres (accessor min/max: y up to 1.806).
  The IBMs are therefore converted to metres first
  (IBM_m = S(0.01) @ IBM_stored, i.e. scale the 3x4 part by 0.01); with
  that fix, skinning an identity pose reproduces the rest vertices to
  <1e-6 m. Without it the mesh explodes to +/-100 m.
* Render: pyrender OffscreenRenderer (EGL surfaceless + Mesa llvmpipe on
  this headless VM, see scripts/pilot-render-env.sh), 360x640 RGBA,
  PerspectiveCamera, ambient + directional light, GLB PBR materials kept
  (dark-brown Beta_Joints, pink Beta_Surface).

Camera: the original driver's exact camera parameters are not recorded
anywhere, so they were re-fit to the surviving frames (silhouette IoU
over rendered alpha masks, frames 0/60/85; full notes in
output/pilot/RENDER-README.md). Default is the best fixed pinhole that
also renders the CORRECT VIEW (level 3/4 profile): eye (0, 0.7, 1.0),
target (0, 0.7, 0), yfov 45 deg, variant=armature, scale=0.784.
Validation, silhouette IoU vs the surviving frames (held-arm hang on):
frame 0: 0.508, frame 60: 0.711, frame 85: 0.613. A closer looking-up
camera scores higher IoU (0.73-0.75 on frame 60) but renders a front
view from below, which is not what the surviving frames show; no fixed
camera fits all three frames well (best mean ~0.61-0.64 across every
family tried: fixed pinhole, per-frame bbox tracking, weak-perspective
placement, ortho), so the original driver almost certainly used
per-frame camera parameters whose exact rule was lost with its script.
This driver is therefore a best-fit approximation, not an exact
reproduction -- the residuals are documented in RENDER-README.md.
Override the camera with --eye/--target/--yfov to experiment.

Held arms: a held upper-arm bone is rotated to hang down (see HANG_QUAT
below) instead of keeping the T-pose rest direction; without it frames
60/85 render a horizontal arm sticking out of the shoulder that the
original pipeline's frames do not have. Disable with --no-held-arm-hang.

Usage:
    PYOPENGL_PLATFORM=egl python3 scripts/render_pilot_frames.py \
        --start 86 --end 119
    PYOPENGL_PLATFORM=egl python3 scripts/render_pilot_frames.py \
        --frame 60 --force --out-dir /tmp/check   # validation re-render
Existing non-empty frames are skipped (resume); --force re-renders.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from recast.stages.rerender import load_glb_skeleton  # noqa: E402

WIDTH = 360
HEIGHT = 640

# Camera: best fixed-pinhole fit to the surviving frames, chosen for the
# CORRECT VIEW (level, 3/4 profile) as well as the number -- a closer,
# looking-up camera scores a higher silhouette IoU (0.73-0.75) but renders
# a front view from below, which is not what the surviving frames show.
# Validation (silhouette IoU vs the surviving frames, held-arm hang on):
#   frame 0: 0.508, frame 60: 0.711, frame 85: 0.613 (mean 0.61).
# The original driver's exact (probably per-frame) camera parameters were
# lost with its script, so this is a best-fit approximation, not an exact
# reproduction -- see RENDER-README.md for the full reverse-engineering
# notes and the residual differences.
DEFAULT_EYE = (0.0, 0.7, 1.0)
DEFAULT_TARGET = (0.0, 0.7, 0.0)
DEFAULT_YFOV_DEG = 45.0

# Held upper-arm bones point in the T-pose rest direction (+X) when the
# retarget holds them (identity local rotation). In the surviving frames
# the far arm never sticks out sideways -- it hangs down along the body.
# So held upper-arm bones are rotated to hang down (RotZ(-90): +X -> -Y
# in the parent/shoulder frame) instead of keeping the T-pose. Held
# forearms keep identity (they continue the upper arm's direction) and
# held legs already point down at rest. This is a deliberate deviation
# from retarget.json (which holds the far arm at identity); without it
# frames 60/85 render a horizontal arm sticking out of the shoulder that
# the original pipeline's frames do not have. IoU effect: f60 0.687 ->
# 0.711, f85 0.600 -> 0.613, f0 0.510 -> 0.508.
HANG_QUAT = (0.0, 0.0, -0.7071067811865476, 0.7071067811865476)
HANG_BONES = ("mixamorig:LeftArm", "mixamorig:RightArm")


# ---------------------------------------------------------------------------
# GLB reading (raw accessors; trimesh alone does not expose skin weights)
# ---------------------------------------------------------------------------

def read_glb_skin_arrays(glb_path: str | Path):
    """Return (meshes, ibm_metres, joint_names).

    meshes: list of (positions_m, faces, joints_idx, weights, material).
    ibm_metres: (n_joints, 4, 4) inverse-bind matrices converted to metres.
    """
    import trimesh

    data = Path(glb_path).read_bytes()
    jlen, _ = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blen, _ = struct.unpack("<II", data[off:off + 8])
    binchunk = data[off + 8:off + 8 + blen]

    def acc(idx: int) -> np.ndarray:
        a = gltf["accessors"][idx]
        bv = gltf["bufferViews"][a["bufferView"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32,
              5121: np.uint8}[a["componentType"]]
        arr = np.frombuffer(binchunk, dtype=dt, count=a["count"] * ncomp,
                            offset=start)
        return arr.reshape(a["count"], ncomp) if ncomp > 1 else arr

    scene = trimesh.load(str(glb_path), force="scene")
    geoms = list(scene.geometry.values())
    meshes = []
    for mi, mesh in enumerate(gltf["meshes"]):
        p = mesh["primitives"][0]
        pos = acc(p["attributes"]["POSITION"]).astype(np.float64)
        jts = acc(p["attributes"]["JOINTS_0"]).astype(np.int64)
        wts = acc(p["attributes"]["WEIGHTS_0"]).astype(np.float64)
        faces = np.asarray(geoms[mi].faces, dtype=np.int64)
        visual = geoms[mi].visual
        meshes.append((pos, faces, jts, wts, visual))
    ibm = (acc(gltf["skins"][0]["inverseBindMatrices"])
           .astype(np.float64).reshape(-1, 4, 4))
    ibm = np.transpose(ibm, (0, 2, 1))  # glTF MAT4 is column-major
    ibm = ibm.copy()
    ibm[:, :3, :] *= 0.01  # stored IBMs are in cm; mesh is in metres
    joint_names = [gltf["nodes"][i]["name"]
                   for i in gltf["skins"][0]["joints"]]
    return meshes, ibm, joint_names


# ---------------------------------------------------------------------------
# Forward kinematics + linear blend skinning (metres throughout)
# ---------------------------------------------------------------------------

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def fk_world_matrices(frame: dict, skeleton: dict, scale: float,
                      root_variant: str = "armature",
                      held_arm_hang: bool = True) -> dict:
    """Joint name -> 4x4 world matrix (metres, Armature space).

    ``root_variant="armature"`` (default): the per-frame root rotation is
    applied at the Armature level, i.e. the whole model rotates about the
    origin. ``root_variant="hips"``: it is applied to the Hips node only
    (rotate about the hips; the hips stay at their rest height).
    Every driven bone contributes its local quaternion.

    ``held_arm_hang``: a held upper-arm bone (source "held*") is rotated
    to hang down instead of keeping the T-pose rest direction (see
    HANG_QUAT above). Only the upper-arm bones are affected; held
    forearms continue the upper arm's direction, held legs already point
    down at rest.
    """
    joints = skeleton["joints"]
    parents = skeleton["parents"]
    root = np.asarray(frame["root_matrix"], dtype=float)
    rots = {b: _quat_to_mat(np.asarray(q, dtype=float))
            for b, q in frame["bone_rotations"].items()}
    if held_arm_hang:
        sources = frame.get("bone_sources", {})
        for bone in HANG_BONES:
            if sources.get(bone, "").startswith("held"):
                rots[bone] = _quat_to_mat(np.asarray(HANG_QUAT, dtype=float))
    ident = np.eye(3)
    mats: dict[str, np.ndarray] = {}
    if root_variant == "hips":
        arm = np.eye(4)
        arm[:3, :3] = np.eye(3) * scale
        mats["Armature"] = arm

        def build_hips(name: str) -> np.ndarray:
            if name in mats:
                return mats[name]
            parent = parents.get(name)
            off = (joints[name]
                   - (joints[parent] if parent else np.zeros(3))) * scale
            if name == "mixamorig:Hips":
                m = np.eye(4)
                m[:3, :3] = root @ rots.get(name, ident)
                m[:3, 3] = joints[name] * scale
                mats[name] = m
                return m
            local = np.eye(4)
            local[:3, :3] = rots.get(name, ident)
            local[:3, 3] = off
            mats[name] = build_hips(parent) @ local
            return mats[name]

        for name in joints:
            if name != "Armature":
                build_hips(name)
        return mats
    arm = np.eye(4)
    arm[:3, :3] = root * scale
    mats["Armature"] = arm

    def build(name: str) -> np.ndarray:
        if name in mats:
            return mats[name]
        parent = parents.get(name)
        off = (joints[name] - (joints[parent] if parent else np.zeros(3))) * scale
        local = np.eye(4)
        local[:3, :3] = rots.get(name, ident)
        local[:3, 3] = off
        mats[name] = build(parent) @ local
        return mats[name]

    for name in joints:
        if name != "Armature":
            build(name)
    return mats


def skin_frame(frame: dict, skeleton: dict, scale: float,
               meshes, ibm, joint_names,
               root_variant: str = "armature",
               held_arm_hang: bool = True) -> list:
    """Return a list of (skinned_vertices_m, faces, visual) per mesh."""
    mats = fk_world_matrices(frame, skeleton, scale, root_variant,
                             held_arm_hang)
    out = []
    for pos, faces, jts, wts, visual in meshes:
        v_h = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
        acc = np.zeros((len(pos), 3))
        for j in range(4):
            jidx = jts[:, j]
            w = wts[:, j]
            sel = w > 0
            if not sel.any():
                continue
            uniq, inv = np.unique(jidx[sel], return_inverse=True)
            msel = np.stack([mats[joint_names[i]] @ ibm[i] for i in uniq])
            t = np.einsum("nij,nj->ni", msel[inv], v_h[sel])[:, :3]
            acc[sel] += t * w[sel, None]
        out.append((acc, faces, visual))
    return out


# ---------------------------------------------------------------------------
# pyrender
# ---------------------------------------------------------------------------

def _camera_pose(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera-to-world matrix: camera at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = -fwd
    pose[:3, 3] = eye
    return pose


class PilotRenderer:
    """One OffscreenRenderer reused across frames (creation is expensive)."""

    def __init__(self, *, eye=DEFAULT_EYE, target=DEFAULT_TARGET,
                 yfov_deg: float = DEFAULT_YFOV_DEG,
                 width: int = WIDTH, height: int = HEIGHT):
        import pyrender
        import trimesh

        self.width = width
        self.height = height
        self._pyrender = pyrender
        self._trimesh = trimesh
        self._renderer = pyrender.OffscreenRenderer(width, height)
        self._eye = eye
        self._target = target
        self._yfov = np.radians(yfov_deg)

    def render(self, skinned_meshes) -> np.ndarray:
        """Skinned meshes -> RGBA uint8 array (H, W, 4), alpha 0 outside.

        NOTE: with the EGL surfaceless backend the framebuffer has no
        alpha channel (returned alpha is 255 everywhere), so alpha is
        derived from the depth buffer instead: depth == 0 is background
        (pyrender writes exactly 0 where nothing was drawn), depth > 0
        is the model. Verified: depth==0 fraction matches the empty
        background, and the resulting alpha bbox lines up with the
        surviving frames.
        """
        import numpy as np
        import pyrender

        trimesh = self._trimesh
        scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
        for verts, faces, visual in skinned_meshes:
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            tm.visual = visual
            scene.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
            pose=_camera_pose(
                np.asarray(self._target) + np.array([0.0, 0.5, 2.0]),
                self._target),
        )
        scene.add(
            pyrender.PerspectiveCamera(yfov=self._yfov),
            pose=_camera_pose(self._eye, self._target),
        )
        color, depth = self._renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        rgba = np.asarray(color, dtype=np.uint8).copy()
        rgba[:, :, 3] = np.where(np.asarray(depth) > 0, 255, 0).astype(np.uint8)
        return rgba

    def close(self) -> None:
        self._renderer.delete()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retarget", default=REPO / "output/pilot/retarget.json")
    p.add_argument("--glb", default=REPO / "output/pilot/models/Xbot.glb")
    p.add_argument("--out-dir", default=REPO / "output/pilot/render")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=119)
    p.add_argument("--frame", type=int, default=None,
                   help="render exactly one frame (overrides --start/--end)")
    p.add_argument("--force", action="store_true",
                   help="re-render even if the output PNG already exists")
    p.add_argument("--eye", type=float, nargs=3, default=list(DEFAULT_EYE))
    p.add_argument("--target", type=float, nargs=3,
                   default=list(DEFAULT_TARGET))
    p.add_argument("--yfov", type=float, default=DEFAULT_YFOV_DEG)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--root-variant", choices=["armature", "hips"],
                   default="armature",
                   help="where the per-frame root rotation is applied")
    p.add_argument("--scale", type=float, default=None,
                   help="model scale (default: retarget.json scale)")
    p.add_argument("--no-held-arm-hang", action="store_true",
                   help="do NOT rotate held upper arms down; keep the "
                        "T-pose rest direction from retarget.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from PIL import Image

    args = parse_args(argv)
    retarget = json.loads(Path(args.retarget).read_text(encoding="utf-8"))
    frames = {f["frame"]: f for f in retarget["frames"]}
    scale = float(args.scale) if args.scale is not None else float(retarget["scale"])
    skeleton = load_glb_skeleton(args.glb)
    meshes, ibm, joint_names = read_glb_skin_arrays(args.glb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.frame is not None:
        wanted = [args.frame]
    else:
        wanted = list(range(args.start, args.end + 1))
    todo = []
    for fi in wanted:
        dest = out_dir / f"frame_{fi:04d}.png"
        if dest.is_file() and dest.stat().st_size > 0 and not args.force:
            print(f"skip frame {fi} (exists)")
            continue
        if fi not in frames:
            print(f"WARN: no retarget data for frame {fi}, skipping")
            continue
        todo.append(fi)
    if not todo:
        print("nothing to render")
        return 0

    renderer = PilotRenderer(eye=args.eye, target=args.target,
                             yfov_deg=args.yfov, width=args.width,
                             height=args.height)
    try:
        for n, fi in enumerate(todo):
            t0 = time.time()
            skinned = skin_frame(frames[fi], skeleton, scale, meshes, ibm,
                                 joint_names, args.root_variant,
                                 not args.no_held_arm_hang)
            rgba = renderer.render(skinned)
            dest = out_dir / f"frame_{fi:04d}.png"
            Image.fromarray(rgba, mode="RGBA").save(dest)
            dt = time.time() - t0
            alpha_frac = float((rgba[:, :, 3] > 10).mean())
            print(f"frame {fi}: {dest.name} {rgba.shape} "
                  f"alpha_frac={alpha_frac:.3f} {dt:.1f}s "
                  f"({n + 1}/{len(todo)})", flush=True)
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
