"""Tests for the body re-render pilot (recast/stages/rerender.py).

Only pure logic is tested here (gates, interpolation, renderable summary):
the real MediaPipe run is exercised by the pilot script and its pose.json
artifact, not by the unit suite (it needs the model + clip).
"""

from pathlib import Path

import pytest

from recast.stages.pose import COCO_17_NAMES
from recast.stages.rerender import (
    PilotRejected,
    _summarise,
    interpolate_low_confidence,
    validate_pilot_clip,
)


def _frames(scores_per_joint):
    """Build fake pose frames: {joint_idx: [score, ...]} over time."""
    n_frames = len(next(iter(scores_per_joint.values())))
    frames = []
    for i in range(n_frames):
        kpts = []
        for j, name in enumerate(COCO_17_NAMES):
            score = scores_per_joint.get(j, [1.0] * n_frames)[i]
            kpts.append(
                {
                    "name": name,
                    "x": float(100 + j * 10 + i),
                    "y": float(200 + i),
                    "z": 0.0,
                    "score": float(score),
                    "interpolated": False,
                    "observed": True,
                }
            )
        frames.append({"frame": i, "t": i / 24.0, "keypoints": kpts})
    return frames


def test_validate_rejects_long_clip():
    with pytest.raises(PilotRejected, match="5 秒"):
        validate_pilot_clip(5.04, people=1)
    with pytest.raises(PilotRejected, match="5 秒"):
        validate_pilot_clip(90.0, people=1)


def test_validate_rejects_multi_person():
    with pytest.raises(PilotRejected, match="单人"):
        validate_pilot_clip(5.0, people=2)


def test_validate_rejects_empty_clip():
    with pytest.raises(PilotRejected, match="没有检测到人物"):
        validate_pilot_clip(5.0, people=0)


def test_validate_accepts_pilot_bounds():
    validate_pilot_clip(5.0, people=1)
    validate_pilot_clip(2.5, people=1)


def test_interpolate_bracketed_low_frames():
    # joint 5 (left_shoulder): valid at frame 0 and 3, low in between.
    frames = _frames({5: [0.9, 0.1, 0.2, 0.9]})
    out = interpolate_low_confidence(frames)
    k1, k2 = out[1]["keypoints"][5], out[2]["keypoints"][5]
    assert k1["interpolated"] is True and k2["interpolated"] is True
    # x moves 1 px/frame: frame0 x=150, frame3 x=153 -> interp 151, 152.
    assert k1["x"] == pytest.approx(151.0)
    assert k2["x"] == pytest.approx(152.0)
    # original low score is kept: the data stays honest about being filled in.
    assert k1["score"] == pytest.approx(0.1)


def test_interpolate_holds_leading_run():
    frames = _frames({5: [0.1, 0.1, 0.9]})
    out = interpolate_low_confidence(frames)
    assert out[0]["keypoints"][5]["held"] is True
    assert out[0]["keypoints"][5]["x"] == pytest.approx(152.0)


def test_never_observed_joint_is_flagged_not_invented():
    # joint 16 (right_ankle): low in every frame -> observed False, x untouched.
    frames = _frames({16: [0.05, 0.02, 0.08]})
    out = interpolate_low_confidence(frames)
    for fr in out:
        kp = fr["keypoints"][16]
        assert kp["observed"] is False
        assert kp["interpolated"] is False
        assert kp["x"] == pytest.approx(100 + 16 * 10 + fr["frame"])


def test_out_of_frame_high_score_joint_is_not_observed():
    # MediaPipe reports high visibility for joints it extrapolates below the
    # frame edge; interpolation must never treat them as measured data.
    frames = _frames({13: [0.9, 0.9, 0.9]})
    for fr in frames:
        fr["keypoints"][13]["y"] = 1500.0  # frame height is 1280
        fr["keypoints"][13]["in_frame"] = False
    out = interpolate_low_confidence(frames)
    for fr in out:
        assert fr["keypoints"][13]["observed"] is False
        assert fr["keypoints"][13]["interpolated"] is False


def test_summarise_renderable_when_core_ok():
    frames = _frames({16: [0.05] * 5, 15: [0.02] * 5})  # ankles never seen
    out = interpolate_low_confidence(frames)
    summary = _summarise(out)
    assert summary["renderable"] is True
    assert summary["unobserved_joints"] == ["left_ankle", "right_ankle"]


def test_summarise_not_renderable_when_core_joint_unobserved():
    frames = _frames({5: [0.05] * 5})  # left_shoulder never seen
    out = interpolate_low_confidence(frames)
    assert _summarise(out)["renderable"] is False


def test_summarise_not_renderable_on_long_low_run():
    # left_hip low for 11 straight frames -> over the 10-frame limit.
    frames = _frames({11: [0.1] * 11 + [0.9]})
    out = interpolate_low_confidence(frames)
    summary = _summarise(out)
    assert summary["max_low_run_core_frames"] == 11
    assert summary["renderable"] is False


# --- retarget tests: synthetic mini Mixamo skeleton + synthetic pose frames ---

import json as _json
import math as _math
import struct as _struct

import numpy as _np

from recast.stages.rerender import (
    BONE_PARENTS,
    estimate_body_scale,
    load_glb_skeleton,
    quat_angle_degrees,
    quat_conjugate,
    quat_from_to,
    quat_multiply,
    quat_slerp,
    retarget_frame,
)

# Core of the Xbot skeleton (cm, identity bind rotations), enough for retarget.
# Like the real Xbot, the nodes hang under an Armature root with a 0.01 scale,
# so load_glb_skeleton composes them into metres.
_MINI_NODES = [
    ("Armature", None, ["mixamorig:Hips"]),
    ("mixamorig:Hips", [0, 104, 2], ["mixamorig:Spine", "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg"]),
    ("mixamorig:Spine", [0, 10.2, 0.1], ["mixamorig:Spine1"]),
    ("mixamorig:Spine1", [0, 10.1, -1], ["mixamorig:Spine2"]),
    ("mixamorig:Spine2", [0, 9.1, -1.4], ["mixamorig:Neck", "mixamorig:LeftShoulder", "mixamorig:RightShoulder"]),
    ("mixamorig:Neck", [0, 16.7, -2.5], ["mixamorig:Head"]),
    ("mixamorig:Head", [0, 9.6, 1.7], ["mixamorig:HeadTop_End"]),
    ("mixamorig:HeadTop_End", [0, 17.8, 2.6], []),
    ("mixamorig:LeftShoulder", [4.6, 10.9, -2.6], ["mixamorig:LeftArm"]),
    ("mixamorig:LeftArm", [10.6, -0.5, -2.2], ["mixamorig:LeftForeArm"]),
    ("mixamorig:LeftForeArm", [27.8, 0, 0], ["mixamorig:LeftHand"]),
    ("mixamorig:LeftHand", [28.3, 0, 0], []),
    ("mixamorig:RightShoulder", [-4.6, 10.9, -2.6], ["mixamorig:RightArm"]),
    ("mixamorig:RightArm", [-10.6, -0.5, -2.2], ["mixamorig:RightForeArm"]),
    ("mixamorig:RightForeArm", [-27.8, 0, 0], ["mixamorig:RightHand"]),
    ("mixamorig:RightHand", [-28.3, 0, 0], []),
    ("mixamorig:LeftUpLeg", [8.2, -6.8, -1.6], ["mixamorig:LeftLeg"]),
    ("mixamorig:LeftLeg", [0, -44.4, 0.3], ["mixamorig:LeftFoot"]),
    ("mixamorig:LeftFoot", [0, -44.4, -3.0], ["mixamorig:LeftToeBase"]),
    ("mixamorig:LeftToeBase", [0, -8.7, 10.7], []),
    ("mixamorig:RightUpLeg", [-8.2, -6.8, -1.6], ["mixamorig:RightLeg"]),
    ("mixamorig:RightLeg", [0, -44.4, 0.3], ["mixamorig:RightFoot"]),
    ("mixamorig:RightFoot", [0, -44.4, -3.0], ["mixamorig:RightToeBase"]),
    ("mixamorig:RightToeBase", [0, -8.7, 10.7], []),
]


def _mini_glb_bytes() -> bytes:
    nodes = []
    index = {name: i for i, (name, _, _) in enumerate(_MINI_NODES)}
    for name, t, children in _MINI_NODES:
        node: dict[str, Any] = {
            "name": name,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "children": [index[c] for c in children],
        }
        if t is not None:
            node["translation"] = t
        else:
            node["scale"] = [0.01, 0.01, 0.01]
        nodes.append(node)
    payload = _json.dumps({"asset": {"version": "2.0"}, "nodes": nodes}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = _struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return _struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _mini_skeleton(tmp_path):
    path = tmp_path / "mini.glb"
    path.write_bytes(_mini_glb_bytes())
    return load_glb_skeleton(path)


def _pose_frame(pos_gltf_m, *, unobserved=(), interpolated=()):
    """pos: COCO name -> (x, y, z) metres in glTF camera space, hips-centred.

    Converts back to MediaPipe world coords (wx, wy, z) the way the real
    estimator stores them, so retarget_frame sees realistic input.
    """
    kpts = []
    for name in COCO_17_NAMES:
        p = pos_gltf_m.get(name, pos_gltf_m["nose"])
        kpts.append(
            {
                "name": name,
                "x": 0.0,
                "y": 0.0,
                "z": -float(p[2]),
                "wx": float(p[0]),
                "wy": -float(p[1]),
                "score": 0.0 if name in unobserved else 1.0,
                "in_frame": True,
                "interpolated": name in interpolated,
                "observed": name not in unobserved,
            }
        )
    return {"frame": 0, "keypoints": kpts}


def _t_pose_positions():
    """Rest-pose joint positions (m, hips-centred) from the mini skeleton."""
    hips = _np.array([0.0, 1.04, 0.02])
    rest = {
        "nose": [0, 1.597, -0.011],
        "left_shoulder": [0.152, 1.438, -0.051],
        "right_shoulder": [-0.152, 1.438, -0.051],
        "left_elbow": [0.430, 1.438, -0.051],
        "right_elbow": [-0.430, 1.438, -0.051],
        "left_wrist": [0.713, 1.438, -0.051],
        "right_wrist": [-0.713, 1.438, -0.051],
        "left_hip": [0.082, 0.972, 0.004],
        "right_hip": [-0.082, 0.972, 0.004],
        "left_knee": [0.082, 0.528, 0.007],
        "right_knee": [-0.082, 0.528, 0.007],
        "left_ankle": [0.082, 0.084, -0.023],
        "right_ankle": [-0.082, 0.084, -0.023],
    }
    # The table above is in model space; subtract the hips origin.
    return {k: [v[0] - hips[0], v[1] - hips[1], v[2] - hips[2]] for k, v in rest.items()}


def _quat_apply(q, v):
    q = _np.asarray(q, dtype=float)
    vq = _np.array([v[0], v[1], v[2], 0.0])
    out = quat_multiply(quat_multiply(q, vq), quat_conjugate(q))
    return out[:3]


def _world_dirs(result, skeleton):
    """Bone -> world direction, by composing local rotations down the chain."""
    joints = skeleton["joints"]
    order = ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1",
             "mixamorig:Spine2", "mixamorig:Neck", "mixamorig:Head",
             "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm",
             "mixamorig:LeftHand", "mixamorig:RightShoulder", "mixamorig:RightArm",
             "mixamorig:RightForeArm", "mixamorig:RightHand",
             "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot",
             "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot"]
    child = {
        "mixamorig:Hips": "mixamorig:Spine", "mixamorig:Spine": "mixamorig:Spine1",
        "mixamorig:Spine1": "mixamorig:Spine2", "mixamorig:Spine2": "mixamorig:Neck",
        "mixamorig:Neck": "mixamorig:Head", "mixamorig:Head": "mixamorig:HeadTop_End",
        "mixamorig:LeftShoulder": "mixamorig:LeftArm", "mixamorig:LeftArm": "mixamorig:LeftForeArm",
        "mixamorig:LeftForeArm": "mixamorig:LeftHand",
        "mixamorig:RightShoulder": "mixamorig:RightArm", "mixamorig:RightArm": "mixamorig:RightForeArm",
        "mixamorig:RightForeArm": "mixamorig:RightHand",
        "mixamorig:LeftUpLeg": "mixamorig:LeftLeg", "mixamorig:LeftLeg": "mixamorig:LeftFoot",
        "mixamorig:RightUpLeg": "mixamorig:RightLeg", "mixamorig:RightLeg": "mixamorig:RightFoot",
    }
    local = {b: _np.array(q) for b, q in result["bone_rotations"].items()}
    world = {}
    for bone in order:
        parent = BONE_PARENTS[bone]
        pq = world[parent] if parent else _np.array([0.0, 0.0, 0.0, 1.0])
        world[bone] = quat_multiply(pq, local[bone])
    out = {}
    for bone, ch in child.items():
        rest = joints[ch] - joints[bone]
        rest = rest / _np.linalg.norm(rest)
        out[bone] = _quat_apply(world[bone], rest)
    return out


def test_quat_from_to_rotates_x_to_y():
    q = quat_from_to(_np.array([1.0, 0.0, 0.0]), _np.array([0.0, 1.0, 0.0]))
    assert _np.allclose(_quat_apply(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
    assert quat_angle_degrees(q) == pytest.approx(90.0)


def test_quat_from_to_handles_opposite_vectors():
    q = quat_from_to(_np.array([1.0, 0.0, 0.0]), _np.array([-1.0, 0.0, 0.0]))
    assert _np.allclose(_quat_apply(q, [1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-6)
    assert quat_angle_degrees(q) == pytest.approx(180.0)


def test_quat_slerp_midpoint_is_half_angle():
    q = quat_slerp(
        _np.array([0.0, 0.0, 0.0, 1.0]),
        quat_from_to(_np.array([1.0, 0.0, 0.0]), _np.array([0.0, 1.0, 0.0])),
        0.5,
    )
    assert quat_angle_degrees(q) == pytest.approx(45.0)


def test_load_mini_glb_skeleton(tmp_path):
    skel = _mini_skeleton(tmp_path)
    assert skel["non_identity_bind_rotations"] == []
    assert _np.allclose(skel["joints"]["mixamorig:Hips"], [0, 1.04, 0.02])
    assert _np.allclose(skel["joints"]["mixamorig:LeftHand"], [0.713, 1.438, -0.051], atol=0.002)


def _body_frame(pos):
    """(left, up, facing) of the synthetic frame, per the documented convention."""
    ls = _np.array(pos["left_shoulder"])
    rs = _np.array(pos["right_shoulder"])
    lh = _np.array(pos["left_hip"])
    rh = _np.array(pos["right_hip"])
    mid_s = (ls + rs) / 2
    mid_h = (lh + rh) / 2
    up = mid_s - mid_h
    up = up / _np.linalg.norm(up)
    left = ls - rs
    left = left / _np.linalg.norm(left)
    facing = _np.cross(left, up)
    facing = facing / _np.linalg.norm(facing)
    left = _np.cross(up, facing)
    return left, up, facing


def _expected_dirs(pos):
    """Bone -> expected world direction in model space, from the measurements.

    The invariant under test: the posed bone points where the measured bone
    points. Target directions are taken in body-frame components, which is
    the model's own frame by construction.
    """
    left, up, facing = _body_frame(pos)
    axes = _np.column_stack([left, up, facing])

    def _dir(a, b):
        d = _np.array(pos[b]) - _np.array(pos[a])
        d = d / _np.linalg.norm(d)
        return axes.T @ d  # camera-space direction -> body-frame components

    return {
        "mixamorig:LeftArm": _dir("left_shoulder", "left_elbow"),
        "mixamorig:RightArm": _dir("right_shoulder", "right_elbow"),
        "mixamorig:LeftForeArm": _dir("left_elbow", "left_wrist"),
        "mixamorig:RightForeArm": _dir("right_elbow", "right_wrist"),
        "mixamorig:LeftUpLeg": _dir("left_hip", "left_knee"),
        "mixamorig:RightUpLeg": _dir("right_hip", "right_knee"),
        "mixamorig:LeftLeg": _dir("left_knee", "left_ankle"),
        "mixamorig:RightLeg": _dir("right_knee", "right_ankle"),
    }


def test_retarget_rest_pose_is_identity(tmp_path):
    skel = _mini_skeleton(tmp_path)
    pos = _t_pose_positions()
    result = retarget_frame(_pose_frame(pos), skel)
    # The invariant: every driven bone points along its measured direction.
    dirs = _world_dirs(result, skel)
    for bone, expected in _expected_dirs(pos).items():
        assert _np.allclose(dirs[bone], expected, atol=0.05), bone
    # Root columns are the body frame itself.
    left, up, facing = _body_frame(pos)
    root = _np.array(result["root_matrix"])
    assert _np.allclose(root[:, 0], left, atol=1e-6)
    assert _np.allclose(root[:, 1], up, atol=1e-6)
    assert _np.allclose(root[:, 2], facing, atol=1e-6)


def test_retarget_arm_raised_forward(tmp_path):
    skel = _mini_skeleton(tmp_path)
    pos = _t_pose_positions()
    # Left arm straight forward, forearm hanging down (camera-space deltas;
    # the body frame is tilted ~7 deg by the model's shoulder setback, so the
    # expected directions are computed through the frame, not assumed axes).
    pos["left_elbow"] = [0.152, 0.398, 0.229]  # shoulder + 0.28 forward
    pos["left_wrist"] = [0.152, 0.118, 0.229]  # elbow straight down
    result = retarget_frame(_pose_frame(pos), skel)
    dirs = _world_dirs(result, skel)
    for bone, expected in _expected_dirs(pos).items():
        assert _np.allclose(dirs[bone], expected, atol=0.05), bone
    # And the raised arm is really up out of the rest pose, not ~identity.
    assert quat_angle_degrees(result["bone_rotations"]["mixamorig:LeftArm"]) > 60.0


def test_retarget_legs_held_when_unobserved(tmp_path):
    skel = _mini_skeleton(tmp_path)
    frame = _pose_frame(
        _t_pose_positions(),
        unobserved=("left_knee", "right_knee", "left_ankle", "right_ankle"),
    )
    result = retarget_frame(frame, skel)
    for bone in ("mixamorig:LeftUpLeg", "mixamorig:RightUpLeg",
                 "mixamorig:LeftLeg", "mixamorig:RightLeg"):
        assert quat_angle_degrees(result["bone_rotations"][bone]) == pytest.approx(0.0)
        assert result["bone_sources"][bone] == "held"
    # Arms are still driven.
    assert result["bone_sources"]["mixamorig:LeftArm"] == "observed"


def test_retarget_limb_held_when_endpoint_out_of_frame(tmp_path):
    # A knee MediaPipe extrapolates below the frame edge with a high score
    # (0.80 at f55-60 in the pilot) must not drive the thigh.
    skel = _mini_skeleton(tmp_path)
    frame = _pose_frame(_t_pose_positions())
    for kp in frame["keypoints"]:
        if kp["name"] == "right_knee":
            kp["score"] = 0.8
            kp["in_frame"] = False
            kp["interpolated"] = True
    result = retarget_frame(frame, skel)
    assert result["bone_sources"]["mixamorig:RightUpLeg"] == "held-lowconf"


def test_retarget_limb_held_when_endpoint_low_confidence(tmp_path):
    # The far arm in profile never scores above 0.75: its bones must hold
    # instead of following the occluded elbow/wrist guesses.
    skel = _mini_skeleton(tmp_path)
    frame = _pose_frame(_t_pose_positions())
    for kp in frame["keypoints"]:
        if kp["name"] in ("left_elbow", "left_wrist"):
            kp["score"] = 0.6
    result = retarget_frame(frame, skel)
    assert result["bone_sources"]["mixamorig:LeftArm"] == "held-lowconf"
    assert result["bone_sources"]["mixamorig:LeftForeArm"] == "held-lowconf"
    assert quat_angle_degrees(result["bone_rotations"]["mixamorig:LeftArm"]) == pytest.approx(0.0)
    # The near arm scores 1.0 and stays driven.
    assert result["bone_sources"]["mixamorig:RightArm"] == "observed"


def test_retarget_forearm_held_when_foreshortened(tmp_path):
    # A forearm measuring half the model length (hand occluded behind the
    # back) must hold instead of driving from the bad measurement.
    skel = _mini_skeleton(tmp_path)
    pos = _t_pose_positions()
    # Move the left wrist halfway to the elbow: measured forearm ~0.14 m
    # against a 0.28 m model segment.
    elbow = _np.array(pos["left_elbow"])
    wrist = _np.array(pos["left_wrist"])
    pos["left_wrist"] = list(elbow + 0.5 * (wrist - elbow))
    lengths = {
        "mixamorig:LeftForeArm": 0.283,
        "mixamorig:RightForeArm": 0.283,
        "mixamorig:LeftArm": 0.278,
        "mixamorig:RightArm": 0.278,
    }
    result = retarget_frame(_pose_frame(pos), skel, segment_lengths_m=lengths)
    assert result["bone_sources"]["mixamorig:LeftForeArm"] == "held-short"
    assert quat_angle_degrees(result["bone_rotations"]["mixamorig:LeftForeArm"]) == pytest.approx(0.0)
    # The other forearm measures full length and stays driven.
    assert result["bone_sources"]["mixamorig:RightForeArm"] == "observed"


def test_estimate_body_scale_synthetic(tmp_path):
    skel = _mini_skeleton(tmp_path)
    payload = {"frames": [_pose_frame(_t_pose_positions()) for _ in range(2)]}
    info = estimate_body_scale(payload, skel)
    assert info["scale"] == pytest.approx(1.0, abs=0.02)
    assert info["proportion_warning"] is False


# ----- step 4: composite -----


def _composite_inputs():
    """Synthetic 40x60 scene: flat grey background, a blue 'body' block
    below the neck, and a red render block with full alpha below the neck."""
    import numpy as np

    h, w = 60, 40
    orig = np.full((h, w, 3), 120, dtype=np.uint8)  # BGR grey background
    orig[20:, 10:30] = (200, 30, 30)  # BGR blue body block
    render = np.zeros((h, w, 4), dtype=np.uint8)
    render[20:, 10:30, :3] = (30, 30, 200)  # BGR red render block
    render[20:, 10:30, 3] = 255
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:, 10:30] = 255  # person mask incl. head (rows 10-19)
    return orig, render, mask


def test_composite_keeps_original_head_and_pastes_body():
    import numpy as np

    from recast.stages.rerender import composite_frame

    orig, render, mask = _composite_inputs()
    out = composite_frame(orig, render, mask, neck_y=20, color_match=False)
    # Head rows are untouched.
    assert (out[:20] == orig[:20]).all()
    # Body block is the render where alpha is full.
    assert (out[30, 20] == np.array([30, 30, 200])).all()
    # The original body colour is gone everywhere below the neck.
    assert not (out[20:, 10:30] == np.array([200, 30, 30])).all(axis=2).any()


def test_composite_inpaints_where_render_is_missing():
    import numpy as np

    from recast.stages.rerender import composite_frame

    orig, render, mask = _composite_inputs()
    render[:, :, 3] = 0  # no render at all: the hole must be inpainted
    out = composite_frame(orig, render, mask, neck_y=20, color_match=False)
    # The erased body region is filled with ~the flat background (Telea
    # inpaint drifts a few levels in a large hole), not left as the blue
    # body or black.
    assert np.abs(out[30, 20].astype(int) - 120).max() <= 5
    assert not (out[30, 20] == np.array([200, 30, 30])).all()


def test_composite_colour_match_darkens_bright_render():
    import numpy as np

    from recast.stages.rerender import composite_frame

    orig, render, mask = _composite_inputs()
    orig[20:, 10:30] = (40, 40, 40)  # dark original body
    render[20:, 10:30, :3] = (200, 200, 200)  # bright render
    out = composite_frame(orig, render, mask, neck_y=20, color_match=True)
    assert out[30, 20].mean() < 200  # gain pulled the render down


def test_neck_cut_y_midway_nose_to_shoulders():
    from recast.stages.rerender import neck_cut_y

    frame = {
        "keypoints": [
            {"name": "nose", "y": 300.0},
            {"name": "left_shoulder", "y": 470.0},
            {"name": "right_shoulder", "y": 474.0},
        ]
    }
    assert neck_cut_y(frame) == 386


# ----- step 5: service gate -----


def _write_black_clip(path, seconds: float, fps: int = 10) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for _ in range(int(seconds * fps)):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()


def test_service_rerender_rejects_overlong_clip(tmp_path):
    from recast.core.service import RecastService
    from recast.stages.rerender import PilotRejected

    clip = tmp_path / "long.mp4"
    _write_black_clip(clip, seconds=6.0)
    svc = RecastService(tmp_path / "store.db")
    with pytest.raises(PilotRejected, match="5 秒以内"):
        svc.run_body_rerender(clip, tmp_path / "Xbot.glb", out_dir=tmp_path / "out")


def test_service_find_blender_none_when_absent(tmp_path, monkeypatch):
    from recast.core.service import RecastService

    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir, no blender
    assert RecastService._find_blender(None) is None
    assert RecastService._find_blender(tmp_path / "nope" / "blender") is None


def test_blender_frame_list_step_form():
    """scripts/rerender_blender.py _frame_list: '0-118:2' -> 60 even frames.

    The Blender script imports bpy at module level, so the function is
    extracted from the file with ast and executed standalone instead of
    importing the module.
    """
    import ast

    src = (
        Path(__file__).resolve().parents[1] / "scripts" / "rerender_blender.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_frame_list"
    )
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "frame_list", "exec"), ns)
    frame_list = ns["_frame_list"]
    assert frame_list("60", 120) == [60]
    assert frame_list("0-3", 120) == [0, 1, 2, 3]
    assert frame_list("0-118:2", 120) == list(range(0, 119, 2))
    assert frame_list("86-118:2", 120) == list(range(86, 119, 2))
