"""Blender render job for the body re-render pilot (runs inside Blender).

Usage (from the repo root, Blender 4.2 headless)::

    blender -b --python scripts/rerender_blender.py -- \
        --glb output/pilot/models/Xbot.glb \
        --retarget output/pilot/retarget.json \
        --pose output/pilot/pose.json \
        --frames 60 --out-dir output/pilot/render --width 360 --height 640

Renders the posed GLB body as RGBA PNGs (transparent film, EEVEE) plus a
depth EXR per frame. One invocation renders every frame in ``--frames``
(``60`` or ``0-119``); callers that need a single frame pass one number.

How the framing is matched (weak-perspective, plan SS2.3):
the camera is a fixed orthographic camera in front of the model. Per frame,
the pixels-per-metre scale comes from the measured torso (mid-shoulder to
mid-hip in pose.json, in pixels) divided by the model's torso length after
the retarget scale, and the armature is translated so the hips land on the
measured mid-hip pixel position. No perspective: the shot is a standing
figure ~3 m from the camera, so the error is small, and the overlay check
(frame vs original) shows it directly.

Two frame conversions matter (both verified against test renders):
glTF is Y-up, Blender is Z-up. C maps glTF -> Blender, (x, y, z) ->
(x, -z, y). The retarget root matrix R maps body axes (left, up, facing)
to glTF camera axes; the armature's local axes are Blender axes (left,
-facing, up), so the object rotation is C @ R @ P, where P maps
Blender-local axes back to body axes. Pose-bone quaternions are relative
to each bone's rest orientation, so a retarget local quaternion L is set
as Q_rest^-1 @ (C @ L @ C^-1) @ Q_rest, with Q_rest from the bone's
matrix_local. Setting the quaternions (or the root) without these
conversions renders the figure lying flat -- that failure mode is what the
diagnostic renders in output/pilot/render/diag-*.png pinned down.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

# glTF -> Blender conversion (x, y, z) -> (x, -z, y); P is its inverse,
# mapping Blender-local axes (left, -facing, up) to body axes (left, up, facing).
CONV = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
PERM = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
# Quaternion for the same +90 deg rotation about X (glTF -> Blender).
CONV_QUAT = Quaternion((math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0))


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--retarget", required=True)
    p.add_argument("--pose", required=True)
    p.add_argument("--frames", default="0-119")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--width", type=int, default=360)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--samples", type=int, default=16)
    return p.parse_args(argv)


def _frame_list(spec, count):
    # "60" -> [60]; "0-119" -> 0..119; "0-118:2" -> even frames only.
    # The step form exists for the 12fps pilot cut: render every 2nd
    # frame of the 24fps pose instead of waiting out 120 CPU frames.
    if ":" in spec:
        rng, step = spec.split(":", 1)
        a, b = rng.split("-", 1)
        return list(range(int(a), int(b) + 1, int(step)))
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def _pose_frame(pose, idx):
    return next(f for f in pose["frames"] if f["frame"] == idx)


def _mid(kpts, a, b):
    pa = next(k for k in kpts if k["name"] == a)
    pb = next(k for k in kpts if k["name"] == b)
    return ((pa["x"] + pb["x"]) / 2.0, (pa["y"] + pb["y"]) / 2.0)


def main(argv):
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retarget = json.loads(Path(args.retarget).read_text(encoding="utf-8"))
    pose = json.loads(Path(args.pose).read_text(encoding="utf-8"))
    scale = float(retarget["scale"])

    # --- clean scene, import the model -------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.glb)
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("GLB 里没有找到骨架 (Armature)")
    arm = armatures[0]
    # The imported armature carries the file's own scale (Xbot: 0.01, cm->m);
    # the retarget body scale multiplies it. The scale lives in the matrix
    # composed below, NOT in arm.scale (matrix_basis would overwrite it).
    base_scale = arm.scale[0]
    total_scale = base_scale * scale
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"

    # Hips rest position in armature-local space (for placement math).
    hips_bone = arm.data.bones.get("mixamorig:Hips")
    if hips_bone is None:
        raise RuntimeError("骨架里没有 mixamorig:Hips")
    hips_local = Vector(hips_bone.head_local)
    # Model torso length (metres, after scale) for the pixels-per-metre fit.
    shoulder_l = arm.data.bones["mixamorig:LeftArm"].head_local
    shoulder_r = arm.data.bones["mixamorig:RightArm"].head_local
    mid_shoulder_local = (Vector(shoulder_l) + Vector(shoulder_r)) / 2
    torso_m = (mid_shoulder_local - hips_local).length * total_scale

    # --- camera: fixed orthographic, in front of the model -----------------
    # Blender world is Z-up; glTF (and the retarget root matrix) is Y-up.
    # C converts glTF -> Blender: (x, y, z) -> (x, -z, y). The original
    # camera sat at glTF +Z looking at the origin, which is Blender (0,-5,0)
    # looking along +Y (rotation +90 deg about X, up stays +Z).
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = args.samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(out_dir / "frame_")
    cam_data = bpy.data.cameras.new("PilotCam")
    cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("PilotCam", cam_data)
    cam.location = (0.0, -5.0, 0.0)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # --- lights: warm key from upper-left-front, dim cool fill ------------
    world = bpy.data.worlds.new("PilotWorld")
    world.color = (0.05, 0.05, 0.05)
    scene.world = world
    sun = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "SUN"))
    sun.data.energy = 3.0
    sun.data.color = (1.0, 0.95, 0.85)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(-30))
    scene.collection.objects.link(sun)
    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "SUN"))
    fill.data.energy = 0.8
    fill.data.color = (0.8, 0.85, 1.0)
    fill.rotation_euler = (math.radians(20), 0.0, math.radians(140))
    scene.collection.objects.link(fill)

    # --- depth pass alongside the RGBA output ------------------------------
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    depth_out = tree.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = str(out_dir / "depth")
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_mode = "RGB"  # depth lands in R; BW is not a valid EXR mode
    depth_out.format.color_depth = "32"
    depth_out.file_slots[0].path = "frame_"  # frame number is appended by Blender
    tree.links.new(rl.outputs["Depth"], depth_out.inputs[0])

    # Rest orientation of every bone (armature space), for the pose-bone
    # quaternion conversion (see the module docstring).
    rest_quat = {
        b.name: b.matrix_local.to_quaternion() for b in arm.data.bones
    }

    width = pose["width"]
    height = pose["height"]
    frames = _frame_list(args.frames, pose["frame_count"])
    for idx in frames:
        rframe = next(f for f in retarget["frames"] if f["frame"] == idx)
        pframe = _pose_frame(pose, idx)
        kpts = pframe["keypoints"]
        mid_s_px = _mid(kpts, "left_shoulder", "right_shoulder")
        mid_h_px = _mid(kpts, "left_hip", "right_hip")
        torso_px = math.dist(mid_s_px, mid_h_px)
        if torso_px < 10:
            print(f"frame {idx}: torso too short in pixels ({torso_px:.1f}), skipping")
            continue
        ppm = torso_px / torso_m  # pixels per metre
        # Blender's ortho_scale spans the LONGER sensor side; this render is
        # portrait, so it is the visible HEIGHT in metres (width/ppm would
        # zoom into a 0.4 m slice of the body -- verified the hard way).
        cam_data.ortho_scale = height / ppm
        # Hips target in glTF camera metres, then converted to Blender space.
        target_gltf = Vector(((mid_h_px[0] - width / 2) / ppm,
                              (height / 2 - mid_h_px[1]) / ppm, 0.0))
        target = CONV @ target_gltf
        # Object rotation: C @ R @ P (see module docstring).
        rot = CONV @ Matrix(rframe["root_matrix"]) @ PERM
        rot4 = rot.to_4x4()
        # world hips = location + R @ (S @ hips_local); solve for location.
        scaled_hips = hips_local * total_scale
        loc = target - rot4 @ scaled_hips
        arm.matrix_basis = (
            Matrix.Translation(loc) @ rot4 @ Matrix.Diagonal((total_scale,) * 3).to_4x4()
        )
        for name, q in rframe["bone_rotations"].items():
            pb = arm.pose.bones.get(name)
            if pb is None:
                continue
            # Retarget quats are (x, y, z, w) in glTF model space. Move them
            # to Blender space (conjugate by CONV_QUAT), then into the bone's
            # own rest frame (conjugate by its rest orientation).
            q_local = Quaternion((q[3], q[0], q[1], q[2]))
            q_blender = CONV_QUAT @ q_local @ CONV_QUAT.inverted()
            q_rest = rest_quat.get(name)
            if q_rest is not None:
                q_basis = q_rest.inverted() @ q_blender @ q_rest
            else:
                q_basis = q_blender
            pb.rotation_quaternion = q_basis
        bpy.context.view_layer.update()
        scene.frame_set(idx)
        scene.render.filepath = str(out_dir / f"frame_{idx:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"rendered frame {idx} (ppm={ppm:.0f}, ortho={cam_data.ortho_scale:.3f})")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
