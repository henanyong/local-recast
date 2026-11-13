"""Body re-render pilot: pose -> retarget -> render -> composite.

Pilot scope only: ONE person, standing, <= 5 s, small motion. Anything else
is rejected up front (see :func:`validate_pilot_clip`) instead of producing a
bad render. The full four-step plan lives in
``local-recast-audit/BODY-RERENDER-身体重渲染方案.md``.

Step 1 (this module, first cut): run the repo's :class:`MediaPipeEstimator`
(:mod:`recast.stages.pose`) frame by frame over a short clip and write
``pose.json`` — frame number, COCO-17 joints with pixel x/y, MediaPipe world
z and visibility score. Joints below ``LOW_CONF_THRESHOLD`` (0.5, the plan's
threshold) are marked and linearly interpolated between neighbouring valid
frames; joints never observed in the whole clip (e.g. ankles that stay below
the frame edge) are flagged ``unobserved`` and are NOT invented by
interpolation — the retarget step must hold them in a standing pose instead.

Coordinate conventions (written down once, used by every later step):

* Image: origin top-left, x right, y down, pixels. ``z`` is MediaPipe world
  z (metres, origin at the hips centre, +z points *towards* the camera).
* GLB / glTF: right-handed, +Y up, +Z towards the viewer, -Z into the scene.
  Retargeting maps MediaPipe world space to GLB bone space, see
  :mod:`recast.stages.rerender` retarget section.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recast.stages.pose import (
    COCO_17_NAMES,
    KeypointSmoother,
    MediaPipeEstimator,
    draw_skeleton,
)

log = logging.getLogger("recast")

# --- pilot gates (plan §五: single indoor standing shot, 5 s) -----------------
PILOT_MAX_SECONDS = 5.0
PILOT_MAX_PEOPLE = 1
LOW_CONF_THRESHOLD = 0.5
# More than this many consecutive low-confidence frames on a torso joint and
# the shot is declared not renderable (plan §二① suggests ~10 frames).
MAX_LOW_RUN_FRAMES = 10
# Joints that must have at least one valid observation for the shot to be
# renderable at all (head + torso core; arms/legs may be held/interpolated).
CORE_JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")

POSE_FORMAT = "rerender_pilot_pose_v1"


class PilotRejected(RuntimeError):
    """The clip is outside the pilot scope (too long, more than one person)."""


class PoseUnavailable(RuntimeError):
    """No usable pose could be estimated for the clip."""


def validate_pilot_clip(duration_s: float, people: int) -> None:
    """Reject anything outside the pilot scope before any heavy work runs."""
    if float(duration_s) > PILOT_MAX_SECONDS:
        raise PilotRejected(
            f"身体重渲染试点只支持 {PILOT_MAX_SECONDS:.0f} 秒以内的片段，"
            f"当前 {float(duration_s):.1f} 秒。请先截取 5 秒以内的单人片段。"
        )
    if int(people) > PILOT_MAX_PEOPLE:
        raise PilotRejected(
            f"身体重渲染试点只支持单人片段，当前 {int(people)} 人。"
            "多人镜头不在试点范围内。"
        )
    if int(people) < 1:
        raise PilotRejected("片段里没有检测到人物，无法做身体重渲染试点。")


def clip_duration_s(video: str | Path) -> float:
    """Duration in seconds via OpenCV (no ffprobe dependency)."""
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if fps <= 0 or count <= 0:
        raise PoseUnavailable(f"无法读取视频信息: {video}")
    return count / fps


def _joint_entry(
    name: str,
    kp: dict[str, float | None],
    width: int = 0,
    height: int = 0,
    *,
    interpolated: bool = False,
    observed: bool = True,
) -> dict[str, Any]:
    x = None if kp.get("x") is None else float(kp["x"])
    y = None if kp.get("y") is None else float(kp["y"])
    in_frame = (
        x is not None
        and y is not None
        and 0.0 <= x < float(width or 0)
        and 0.0 <= y < float(height or 0)
    ) if width and height else True
    entry: dict[str, Any] = {
        "name": name,
        "x": x,
        "y": y,
        "z": None if kp.get("z") is None else float(kp["z"]),
        "score": float(kp.get("score") or 0.0),
        "in_frame": bool(in_frame),
        "interpolated": bool(interpolated),
        "observed": bool(observed),
    }
    # Metric 3D position from MediaPipe world landmarks (metres, hips centre).
    # May be absent for non-MediaPipe backends; retarget checks for it.
    if kp.get("wx") is not None:
        entry["wx"] = float(kp["wx"])
    if kp.get("wy") is not None:
        entry["wy"] = float(kp["wy"])
    return entry


def interpolate_low_confidence(
    frames: list[dict[str, Any]],
    *,
    threshold: float = LOW_CONF_THRESHOLD,
) -> list[dict[str, Any]]:
    """Linearly interpolate joints with score < threshold between valid frames.

    A joint counts as valid only when its score is above threshold AND it
    lies inside the frame: MediaPipe's ``visibility`` says occluded-or-not,
    not in-frame-or-not, and it happily reports a high-visibility knee that
    sits 200 px below the bottom edge. Such a joint is never treated as
    measured data.
    """
    if not frames:
        return frames
    n_joints = len(frames[0]["keypoints"])
    for j in range(n_joints):
        scores = [float(fr["keypoints"][j].get("score") or 0.0) for fr in frames]
        in_frame = [bool(fr["keypoints"][j].get("in_frame", True)) for fr in frames]
        valid = [i for i, s in enumerate(scores) if s >= threshold and in_frame[i]]
        if not valid:
            for fr in frames:
                fr["keypoints"][j]["observed"] = False
            continue
        for i, fr in enumerate(frames):
            kp = fr["keypoints"][j]
            if float(kp.get("score") or 0.0) >= threshold and in_frame[i]:
                kp["observed"] = True
                continue
            prev_i = max([v for v in valid if v < i], default=None)
            next_i = min([v for v in valid if v > i], default=None)
            if prev_i is not None and next_i is not None:
                t = (i - prev_i) / (next_i - prev_i)
                for axis in ("x", "y", "z", "wx", "wy"):
                    a = frames[prev_i]["keypoints"][j].get(axis)
                    b = frames[next_i]["keypoints"][j].get(axis)
                    if a is None or b is None:
                        continue
                    kp[axis] = float(a) + t * (float(b) - float(a))
                kp["interpolated"] = True
                kp["observed"] = True
            elif prev_i is not None or next_i is not None:
                src = frames[prev_i if prev_i is not None else next_i]["keypoints"][j]
                for axis in ("x", "y", "z", "wx", "wy"):
                    if src.get(axis) is not None:
                        kp[axis] = float(src[axis])
                kp["interpolated"] = True
                kp["held"] = True
                kp["observed"] = True
    return frames


def _frame_bbox(keypoints: list[dict[str, Any]]) -> list[float] | None:
    """Tight bbox (x1, y1, x2, y2) around observed in-frame joints, or None."""
    xs = [
        float(k["x"])
        for k in keypoints
        if k.get("observed") and k.get("in_frame", True) and k.get("x") is not None
    ]
    ys = [
        float(k["y"])
        for k in keypoints
        if k.get("observed") and k.get("in_frame", True) and k.get("y") is not None
    ]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _summarise(frames: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(k.get("score") or 0.0) for fr in frames for k in fr["keypoints"]
    ]
    unobserved: list[str] = []
    for j, name in enumerate(COCO_17_NAMES):
        if not any(fr["keypoints"][j].get("observed") for fr in frames):
            unobserved.append(name)
    max_low_run = 0
    for j, name in enumerate(COCO_17_NAMES):
        if name not in CORE_JOINTS:
            continue
        run = 0
        for fr in frames:
            if float(fr["keypoints"][j].get("score") or 0.0) < LOW_CONF_THRESHOLD:
                run += 1
                max_low_run = max(max_low_run, run)
            else:
                run = 0
    renderable = (
        bool(frames)
        and not any(name in unobserved for name in CORE_JOINTS)
        and max_low_run <= MAX_LOW_RUN_FRAMES
    )
    return {
        "mean_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "unobserved_joints": unobserved,
        "max_low_run_core_frames": int(max_low_run),
        "renderable": bool(renderable),
    }


def estimate_pilot_pose(
    video: str | Path,
    out_dir: str | Path,
    *,
    model_complexity: int = 1,
    source_start_s: float = 0.0,
) -> dict[str, Any]:
    """Run MediaPipe pose frame-by-frame over a pilot clip; write pose.json.

    Uses the full model (``model_complexity=1``) by default: on the pilot's
    profile standing shot it scores mean visibility 0.72-0.74 vs 0.62-0.65
    for the lite model, and the lite model hallucinates the far arm across
    the torso. ``source_start_s`` records where the clip was cut from, so
    frame numbers map back to the source video.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = clip_duration_s(video)
    validate_pilot_clip(duration, people=1)

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    estimator = MediaPipeEstimator(model_complexity=model_complexity)
    smoother = KeypointSmoother(fps, enabled=True)
    frames: list[dict[str, Any]] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            estimated = estimator.estimate(
                frame, [[0.0, 0.0, float(w), float(h)]], [1]
            )
            raw = estimated[0] if estimated else []
            kpts = smoother.smooth(1, raw)
            entries = [
                _joint_entry(name, kp, w, h)
                for name, kp in zip(COCO_17_NAMES, kpts)
            ]
            frames.append(
                {
                    "frame": frame_idx,
                    "t": round(frame_idx / fps, 4) if fps else 0.0,
                    "keypoints": entries,
                }
            )
            frame_idx += 1
    finally:
        cap.release()
        estimator.close()
    if not frames:
        raise PoseUnavailable(f"视频没有可读帧: {video}")

    frames = interpolate_low_confidence(frames)
    for fr in frames:
        fr["low_joints"] = [
            k["name"]
            for k in fr["keypoints"]
            if float(k.get("score") or 0.0) < LOW_CONF_THRESHOLD
        ]
        fr["bbox_xyxy"] = _frame_bbox(fr["keypoints"])
    summary = _summarise(frames)
    payload = {
        "format": POSE_FORMAT,
        "source": video.name,
        "source_start_s": float(source_start_s),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": len(frames),
        "duration_s": round(len(frames) / fps, 4) if fps else 0.0,
        "backend": "mediapipe",
        "model_complexity": int(model_complexity),
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "frames": frames,
        "summary": summary,
    }
    pose_path = out_dir / "pose.json"
    pose_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "试点姿态已写出: %s（%d 帧，平均置信度 %.3f，可渲染=%s，未观测关节=%s）",
        pose_path,
        len(frames),
        summary["mean_score"],
        summary["renderable"],
        summary["unobserved_joints"],
    )
    return payload


def write_pose_overlay(
    video: str | Path,
    pose_payload: dict[str, Any],
    frame_idx: int,
    out_path: str | Path,
) -> Path:
    """Draw the (interpolated) skeleton of one frame onto the source frame."""
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise PoseUnavailable(f"无法读取第 {frame_idx} 帧: {video}")
    entry = next(
        (f for f in pose_payload["frames"] if f["frame"] == int(frame_idx)), None
    )
    if entry is None:
        raise PoseUnavailable(f"pose.json 里没有第 {frame_idx} 帧")
    vis = draw_skeleton(frame, entry["keypoints"], thr=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return out_path


# ---------------------------------------------------------------------------
# Step 2: retargeting (pure numpy) — MediaPipe pose -> GLB bone rotations.
#
# Coordinate conventions (verified against the pilot clip, frame 119):
#
# * MediaPipe world landmarks (metres, hips centre): +x image-right,
#   +y DOWN, +z AWAY from the camera (smaller z = closer). So a man facing
#   image-right has his camera-side (right) shoulder at smaller z, and his
#   nose at wy < 0 (above the hips).
# * glTF / model space: right-handed, +Y up, +Z towards the viewer, model
#   faces +Z, model's left is +X. Conversion: (x, y, z)_gltf = (wx, -wy, -z).
# * Body frame (built per frame, in glTF camera space): up = mid-shoulder
#   minus mid-hip; left = left-shoulder minus right-shoulder; facing =
#   left x up (right-handed in glTF space). All target bone directions are
#   expressed in this frame, so the model's rest pose (facing +Z, left +X,
#   up +Y) lines up with the body frame by construction; the full root
#   rotation (body frame -> camera) is stored per frame for the renderer.
# * Bone orientation: every driven bone's bind rotation must be identity
#   (true for Xbot / Mixamo exports; checked in load_glb_skeleton). Then a
#   bone's world orientation is the shortest-arc quaternion rotating its
#   rest direction to its target direction, and its local rotation is
#   parent_world^-1 * world. Unobserved bones hold: local rotation identity
#   (the bone keeps its rest orientation relative to its parent).
# ---------------------------------------------------------------------------

# COCO joint -> Mixamo bone. Direction joints are the two endpoints whose
# connecting line defines the bone's target direction. Clavicles are NOT
# here: their target direction is the shoulder line (spine-top to shoulder),
# handled separately in retarget_frame.
BONE_MAP: dict[str, tuple[str, str]] = {
    "mixamorig:LeftArm": ("left_shoulder", "left_elbow"),
    "mixamorig:RightArm": ("right_shoulder", "right_elbow"),
    "mixamorig:LeftForeArm": ("left_elbow", "left_wrist"),
    "mixamorig:RightForeArm": ("right_elbow", "right_wrist"),
    "mixamorig:LeftUpLeg": ("left_hip", "left_knee"),
    "mixamorig:RightUpLeg": ("right_hip", "right_knee"),
    "mixamorig:LeftLeg": ("left_knee", "left_ankle"),
    "mixamorig:RightLeg": ("right_knee", "right_ankle"),
}

# Parent bone for each driven bone (Mixamo hierarchy, root = Hips).
BONE_PARENTS: dict[str, str] = {
    "mixamorig:Hips": "",
    "mixamorig:Spine": "mixamorig:Hips",
    "mixamorig:Spine1": "mixamorig:Spine",
    "mixamorig:Spine2": "mixamorig:Spine1",
    "mixamorig:Neck": "mixamorig:Spine2",
    "mixamorig:Head": "mixamorig:Neck",
    "mixamorig:LeftShoulder": "mixamorig:Spine2",
    "mixamorig:RightShoulder": "mixamorig:Spine2",
    "mixamorig:LeftArm": "mixamorig:LeftShoulder",
    "mixamorig:RightArm": "mixamorig:RightShoulder",
    "mixamorig:LeftForeArm": "mixamorig:LeftArm",
    "mixamorig:RightForeArm": "mixamorig:RightArm",
    "mixamorig:LeftHand": "mixamorig:LeftForeArm",
    "mixamorig:RightHand": "mixamorig:RightForeArm",
    "mixamorig:LeftUpLeg": "mixamorig:Hips",
    "mixamorig:RightUpLeg": "mixamorig:Hips",
    "mixamorig:LeftLeg": "mixamorig:LeftUpLeg",
    "mixamorig:RightLeg": "mixamorig:RightUpLeg",
    "mixamorig:LeftFoot": "mixamorig:LeftLeg",
    "mixamorig:RightFoot": "mixamorig:RightLeg",
}

# Child joint (node) of each driven bone, for the rest direction.
BONE_CHILD: dict[str, str] = {
    "mixamorig:Hips": "mixamorig:Spine",
    "mixamorig:Spine": "mixamorig:Spine1",
    "mixamorig:Spine1": "mixamorig:Spine2",
    "mixamorig:Spine2": "mixamorig:Neck",
    "mixamorig:Neck": "mixamorig:Head",
    "mixamorig:Head": "mixamorig:HeadTop_End",
    "mixamorig:LeftShoulder": "mixamorig:LeftArm",
    "mixamorig:RightShoulder": "mixamorig:RightArm",
    "mixamorig:LeftArm": "mixamorig:LeftForeArm",
    "mixamorig:RightArm": "mixamorig:RightForeArm",
    "mixamorig:LeftForeArm": "mixamorig:LeftHand",
    "mixamorig:RightForeArm": "mixamorig:RightHand",
    "mixamorig:LeftUpLeg": "mixamorig:LeftLeg",
    "mixamorig:RightUpLeg": "mixamorig:RightLeg",
    "mixamorig:LeftLeg": "mixamorig:LeftFoot",
    "mixamorig:RightLeg": "mixamorig:RightFoot",
    "mixamorig:LeftFoot": "mixamorig:LeftToeBase",
    "mixamorig:RightFoot": "mixamorig:RightToeBase",
}

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def quat_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion (x, y, z, w) rotating unit vector a to b."""
    a = _norm(np.asarray(a, dtype=float))
    b = _norm(np.asarray(b, dtype=float))
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-9:
        return IDENTITY_QUAT.copy()
    if d < -1.0 + 1e-9:
        # 180 degrees: pick any axis perpendicular to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = _norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    return np.array([axis[0], axis[1], axis[2], 1.0 + d]) / math.sqrt(
        float(np.dot(axis, axis)) + (1.0 + d) ** 2
    )


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product q * r (apply r first, then q). Quaternions are (x,y,z,w)."""
    x1, y1, z1, w1 = [float(v) for v in q]
    x2, y2, z2, w2 = [float(v) for v in r]
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_slerp(q: np.ndarray, r: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between unit quaternions q and r."""
    q = _norm(np.asarray(q, dtype=float))
    r = _norm(np.asarray(r, dtype=float))
    d = float(np.dot(q, r))
    if d < 0.0:
        r = -r
        d = -d
    if d > 1.0 - 1e-9:
        return _norm(q + t * (r - q))
    theta = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(theta)
    return _norm(
        (math.sin((1.0 - t) * theta) / s) * q + (math.sin(t * theta) / s) * r
    )


def quat_angle_degrees(q: np.ndarray) -> float:
    """Rotation magnitude of a unit quaternion, in degrees."""
    w = max(-1.0, min(1.0, float(np.asarray(q, dtype=float)[3])))
    return math.degrees(2.0 * math.acos(abs(w)))


def _read_glb_json(glb_path: str | Path) -> dict[str, Any]:
    import struct

    data = Path(glb_path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"不是 GLB 文件: {glb_path}")
    chunk_len, chunk_type = struct.unpack("<II", data[12:20])
    if chunk_type != 0x4E4F534A:  # "JSON"
        raise ValueError(f"GLB 第一个 chunk 不是 JSON: {glb_path}")
    return json.loads(data[20 : 20 + chunk_len])


def load_glb_skeleton(glb_path: str | Path) -> dict[str, Any]:
    """Rest-pose joint positions (metres) + hierarchy from a GLB's JSON chunk.

    Positions are composed through the node hierarchy, including root scale
    (Xbot is authored in cm with a 0.01 Armature scale, so the composed
    positions come out in metres). Only depends on the JSON chunk (no mesh
    parsing). Warns when a driven bone has a non-identity bind rotation,
    because the retarget math below assumes identity bind rotations.
    """
    gltf = _read_glb_json(glb_path)
    nodes = gltf.get("nodes", [])
    parent: dict[str, str | None] = {}
    for node in nodes:
        name = node.get("name")
        if not name:
            continue
        for child_idx in node.get("children", []):
            child = nodes[child_idx].get("name")
            if child:
                parent[child] = name
    for node in nodes:
        if node.get("name") and node.get("name") not in parent:
            parent[node["name"]] = None

    def _local_matrix(node: dict[str, Any]) -> np.ndarray:
        if "matrix" in node:
            m = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
            return m
        m = np.eye(4)
        t = node.get("translation", [0.0, 0.0, 0.0])
        m[:3, 3] = [float(v) for v in t]
        r = node.get("rotation")
        if r is not None:
            x, y, z, w = [float(v) for v in r]
            rot = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            m[:3, :3] = rot
        s = node.get("scale", [1.0, 1.0, 1.0])
        m[:3, :3] = m[:3, :3] @ np.diag([float(v) for v in s])
        return m

    positions: dict[str, np.ndarray] = {}
    for node in nodes:
        name = node.get("name")
        if not name:
            continue
        m = _local_matrix(node)
        p = parent[name]
        while p is not None:
            pnode = next(n for n in nodes if n.get("name") == p)
            m = _local_matrix(pnode) @ m
            p = parent[p]
        positions[name] = m[:3, 3]

    non_identity: list[str] = []
    for bone in list(BONE_PARENTS) + list(BONE_CHILD.values()):
        node = next((n for n in nodes if n.get("name") == bone), None)
        if node is None:
            continue
        r = node.get("rotation")
        if r is not None and not np.allclose(
            [float(v) for v in r], [0.0, 0.0, 0.0, 1.0], atol=1e-6
        ):
            non_identity.append(bone)
    return {
        "joints": positions,
        "parents": {k: v for k, v in parent.items()},
        "non_identity_bind_rotations": sorted(set(non_identity)),
    }


def _coco_pos(frame: dict[str, Any], name: str) -> np.ndarray | None:
    """COCO joint position in glTF camera space (metres), or None if missing."""
    kp = next((k for k in frame["keypoints"] if k["name"] == name), None)
    if kp is None or kp.get("wx") is None or kp.get("wy") is None or kp.get("z") is None:
        return None
    # MediaPipe: +x right, +y DOWN, +z away. glTF: +x right, +y UP, +z to viewer.
    return np.array([float(kp["wx"]), -float(kp["wy"]), -float(kp["z"])])


def _joint_observed(frame: dict[str, Any], name: str) -> bool:
    kp = next((k for k in frame["keypoints"] if k["name"] == name), None)
    return bool(kp and kp.get("observed"))


def _joint_interpolated(frame: dict[str, Any], name: str) -> bool:
    kp = next((k for k in frame["keypoints"] if k["name"] == name), None)
    return bool(kp and kp.get("interpolated"))


def estimate_body_scale(
    pose_payload: dict[str, Any], skeleton: dict[str, Any]
) -> dict[str, Any]:
    """Model scale (model metres per target metre) from body proportions.

    Ratios from shoulder width, hip width and torso length; the median is
    the scale, and a spread over 20% is flagged (plan §二②: 体型差异大).
    """
    joints = skeleton["joints"]

    def _dist(a: str, b: str) -> float:
        # Skeleton positions are already metres (root scale applied).
        return float(np.linalg.norm(joints[a] - joints[b]))

    model = {
        "shoulder": _dist("mixamorig:LeftArm", "mixamorig:RightArm"),
        "hip": _dist("mixamorig:LeftUpLeg", "mixamorig:RightUpLeg"),
        "torso": float(
            np.linalg.norm(
                joints["mixamorig:Hips"]
                - (joints["mixamorig:LeftArm"] + joints["mixamorig:RightArm"]) / 2
            )
        ),
    }
    ratios: dict[str, list[float]] = {"shoulder": [], "hip": [], "torso": []}
    for frame in pose_payload["frames"]:
        ls, rs = _coco_pos(frame, "left_shoulder"), _coco_pos(frame, "right_shoulder")
        lh, rh = _coco_pos(frame, "left_hip"), _coco_pos(frame, "right_hip")
        if ls is not None and rs is not None and _joint_observed(frame, "left_shoulder") and _joint_observed(frame, "right_shoulder"):
            ratios["shoulder"].append(model["shoulder"] / float(np.linalg.norm(ls - rs)))
        if lh is not None and rh is not None and _joint_observed(frame, "left_hip") and _joint_observed(frame, "right_hip"):
            ratios["hip"].append(model["hip"] / float(np.linalg.norm(lh - rh)))
        if ls is not None and rs is not None and lh is not None and rh is not None:
            mid_s = (ls + rs) / 2
            mid_h = (lh + rh) / 2
            ratios["torso"].append(model["torso"] / float(np.linalg.norm(mid_s - mid_h)))
    medians = {
        k: (float(np.median(v)) if v else None) for k, v in ratios.items()
    }
    used = [v for v in medians.values() if v is not None]
    if not used:
        raise PoseUnavailable("无法从姿态估计身体比例：肩/髋关节点均未观测到")
    scale = float(np.median(used))
    spread = (max(used) - min(used)) / scale if scale else 0.0
    return {
        "scale": scale,
        "ratios": medians,
        "proportion_spread": float(spread),
        "proportion_warning": bool(spread > 0.20),
    }


def retarget_frame(
    frame: dict[str, Any],
    skeleton: dict[str, Any],
    *,
    segment_lengths_m: dict[str, float] | None = None,
    min_length_ratio: float = 0.8,
    drive_min_score: float = 0.75,
) -> dict[str, Any]:
    """One pose frame -> root matrix + per-bone local quaternions.

    ``segment_lengths_m`` maps bone -> model segment length in metres (after
    the body scale). A measured segment much shorter than the model in 3D is
    foreshortened or occluded (e.g. a hand behind the back in profile: the
    wrist lands on the jacket front, so the forearm measures 0.17 m against
    a 0.22 m model) -- the bone is held instead of driven from the bad
    measurement. Pass None to disable the guard (unit tests for pure math).

    ``drive_min_score`` (0.75) is the bar for *driving* a limb bone: both
    endpoints must be clearly visible. The pose stage's 0.5 is the
    "don't crash" floor for interpolation, not a trust bar -- in the pilot
    the far arm never scores above 0.75 in any frame (occluded by the
    torso), while the near arm scores 0.99 throughout, so the gate holds
    exactly the occluded side and drives the visible side.
    """
    joints = skeleton["joints"]
    if skeleton.get("non_identity_bind_rotations"):
        raise ValueError(
            "模型绑定旋转不是单位四元数，重定向假设不成立: "
            + ", ".join(skeleton["non_identity_bind_rotations"])
        )
    pos: dict[str, np.ndarray] = {}
    for name in COCO_17_NAMES:
        p = _coco_pos(frame, name)
        if p is not None:
            pos[name] = p
    for required in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        if required not in pos:
            raise PoseUnavailable(f"第 {frame.get('frame')} 帧缺少 {required} 的 3D 坐标")

    mid_s = (pos["left_shoulder"] + pos["right_shoulder"]) / 2
    mid_h = (pos["left_hip"] + pos["right_hip"]) / 2
    up = _norm(mid_s - mid_h)
    left = _norm(pos["left_shoulder"] - pos["right_shoulder"])
    facing = _norm(np.cross(left, up))
    left = _norm(np.cross(up, facing))  # re-orthogonalise
    root = np.column_stack([left, up, facing])  # body -> camera (glTF space)

    def _body(p: np.ndarray) -> np.ndarray:
        return np.array([float(np.dot(p, left)), float(np.dot(p, up)), float(np.dot(p, facing))])

    body_pos = {name: _body(p - mid_h) for name, p in pos.items()}

    def _rest_dir(bone: str) -> np.ndarray:
        child = BONE_CHILD[bone]
        return _norm(joints[child] - joints[bone])

    def _target_dir(a: str, b: str) -> np.ndarray | None:
        if a not in body_pos or b not in body_pos:
            return None
        d = body_pos[b] - body_pos[a]
        if float(np.linalg.norm(d)) < 1e-6:
            return None
        return _norm(d)

    # World orientation (in body frame) per bone, top-down.
    world_q: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}

    def _set(bone: str, q: np.ndarray, source: str) -> None:
        world_q[bone] = q
        sources[bone] = source

    # Hips: align the pelvis axis (rest +X) to the measured hip line.
    pelvis = _target_dir("right_hip", "left_hip")
    if pelvis is not None:
        _set("mixamorig:Hips", quat_from_to(np.array([1.0, 0.0, 0.0]), pelvis), "observed")
    else:
        _set("mixamorig:Hips", IDENTITY_QUAT.copy(), "held")
    # Spine chain: torso direction is absorbed by the body frame itself.
    for bone in ("mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2"):
        _set(bone, world_q["mixamorig:Hips"], "held")
    # Neck + head: direction from the neck base (mid-shoulder) to the nose,
    # split evenly between the two bones.
    neck_base = (body_pos["left_shoulder"] + body_pos["right_shoulder"]) / 2
    neck_target = None
    if "nose" in body_pos:
        d = body_pos["nose"] - neck_base
        if float(np.linalg.norm(d)) > 1e-6:
            neck_target = _norm(d)
    if neck_target is not None:
        total = quat_from_to(_rest_dir("mixamorig:Neck"), neck_target)
        rel = quat_multiply(quat_conjugate(world_q["mixamorig:Spine2"]), total)
        _set(
            "mixamorig:Neck",
            quat_multiply(world_q["mixamorig:Spine2"], quat_slerp(IDENTITY_QUAT, rel, 0.5)),
            "observed" if not _joint_interpolated(frame, "nose") else "interpolated",
        )
        _set(
            "mixamorig:Head",
            total,
            "observed" if not _joint_interpolated(frame, "nose") else "interpolated",
        )
    else:
        _set("mixamorig:Neck", world_q["mixamorig:Spine2"], "held")
        _set("mixamorig:Head", world_q["mixamorig:Spine2"], "held")

    def _score(frame: dict[str, Any], name: str) -> float:
        kp = next((k for k in frame["keypoints"] if k["name"] == name), None)
        return float(kp.get("score") or 0.0) if kp else 0.0

    def _in_frame(frame: dict[str, Any], name: str) -> bool:
        kp = next((k for k in frame["keypoints"] if k["name"] == name), None)
        return bool(kp and kp.get("in_frame", True))

    def _limb(bone: str, a: str, b: str) -> None:
        """Set one limb bone from its endpoint joints; hold if unobserved."""
        if not (_joint_observed(frame, a) and _joint_observed(frame, b)):
            parent = BONE_PARENTS[bone]
            _set(bone, world_q[parent], "held")
            return
        # A joint MediaPipe extrapolates below the frame edge can carry a
        # high visibility score (the knee at f55-60 scores 0.80 while its y
        # is past the bottom edge). Never drive from an out-of-frame joint.
        if (
            _score(frame, a) < drive_min_score
            or _score(frame, b) < drive_min_score
            or not _in_frame(frame, a)
            or not _in_frame(frame, b)
        ):
            parent = BONE_PARENTS[bone]
            _set(bone, world_q[parent], "held-lowconf")
            return
        d = _target_dir(a, b)
        if d is None:
            parent = BONE_PARENTS[bone]
            _set(bone, world_q[parent], "held")
            return
        if segment_lengths_m is not None and bone in segment_lengths_m:
            measured = float(np.linalg.norm(pos[b] - pos[a]))
            if measured < min_length_ratio * segment_lengths_m[bone]:
                parent = BONE_PARENTS[bone]
                _set(bone, world_q[parent], "held-short")
                return
        _set(
            bone,
            quat_from_to(_rest_dir(bone), d),
            "interpolated"
            if (_joint_interpolated(frame, a) or _joint_interpolated(frame, b))
            else "observed",
        )

    # Clavicles: direction from spine-top to the shoulder joint.
    for side, shoulder in (("Left", "left_shoulder"), ("Right", "right_shoulder")):
        bone = f"mixamorig:{side}Shoulder"
        if _joint_observed(frame, shoulder):
            d = _target_dir("left_shoulder" if side == "Right" else "right_shoulder", shoulder)
            # midpoint of shoulders is the origin of the clavicle line
            if d is not None:
                _set(
                    bone,
                    quat_from_to(_rest_dir(bone), d),
                    "interpolated" if _joint_interpolated(frame, shoulder) else "observed",
                )
            else:
                _set(bone, world_q["mixamorig:Spine2"], "held")
        else:
            _set(bone, world_q["mixamorig:Spine2"], "held")

    for bone, (a, b) in BONE_MAP.items():
        _limb(bone, a, b)
    for bone in ("mixamorig:LeftHand", "mixamorig:RightHand",
                 "mixamorig:LeftFoot", "mixamorig:RightFoot"):
        _set(bone, world_q[BONE_PARENTS[bone]], "held")

    # Local rotations: parent_world^-1 * world.
    local_q: dict[str, list[float]] = {}
    for bone, parent in BONE_PARENTS.items():
        if bone not in world_q:
            continue
        parent_q = world_q[parent] if parent else IDENTITY_QUAT
        local = quat_multiply(quat_conjugate(parent_q), world_q[bone])
        local_q[bone] = [float(v) for v in _norm(local)]
    return {
        "frame": frame.get("frame"),
        "root_matrix": [[float(v) for v in row] for row in root],
        "bone_rotations": local_q,
        "bone_sources": sources,
    }


def retarget_clip(
    pose_payload: dict[str, Any],
    glb_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Retarget every pose frame to the GLB; write retarget.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton = load_glb_skeleton(glb_path)
    scale_info = estimate_body_scale(pose_payload, skeleton)
    scale = scale_info["scale"]
    joints = skeleton["joints"]
    segment_lengths_m = {
        bone: float(np.linalg.norm(joints[BONE_CHILD[bone]] - joints[bone])) * scale
        for bone in BONE_MAP
    }
    frames = [
        retarget_frame(fr, skeleton, segment_lengths_m=segment_lengths_m)
        for fr in pose_payload["frames"]
    ]
    held = sum(
        1
        for fr in frames
        for s in fr["bone_sources"].values()
        if s in ("held", "held-short", "held-lowconf")
    )
    total = sum(len(fr["bone_sources"]) for fr in frames)
    payload = {
        "format": "rerender_pilot_retarget_v1",
        "source_pose": "pose.json",
        "model": Path(glb_path).name,
        "scale": scale_info["scale"],
        "scale_ratios": scale_info["ratios"],
        "proportion_spread": scale_info["proportion_spread"],
        "proportion_warning": scale_info["proportion_warning"],
        "held_bone_fraction": (held / total) if total else 0.0,
        "frames": frames,
    }
    path = out_dir / "retarget.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info(
        "重定向已写出: %s（模型 %s，缩放 %.3f，体型差异警告=%s，hold 骨骼占比 %.2f）",
        path,
        payload["model"],
        payload["scale"],
        payload["proportion_warning"],
        payload["held_bone_fraction"],
    )
    return payload

# ---------------------------------------------------------------------------
# Step 4 — composite: erase the original body below the neck, paste the
# render, feather the seam. The original head is ALWAYS kept: the render's
# head exists only as a geometry check, never as a face replacement.
# ---------------------------------------------------------------------------


def neck_cut_y(frame_pose: dict[str, Any]) -> int:
    """Y (pixels) of the neck cut: halfway between the nose and the
    mid-shoulder line. Everything above stays original; the render is
    pasted below this line."""
    kps = {k["name"]: k for k in frame_pose["keypoints"]}
    nose_y = float(kps["nose"]["y"])
    mid_sh_y = (float(kps["left_shoulder"]["y"]) + float(kps["right_shoulder"]["y"])) / 2
    return int(round(nose_y + 0.5 * (mid_sh_y - nose_y)))


def _alpha_bbox(alpha: Any, thr: int = 8) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of an alpha mask, or None when it is empty."""
    import numpy as np

    ys, xs = np.where(np.asarray(alpha) > thr)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_render_to_person(
    render_rgba: Any,
    person_mask: Any,
    neck_y: int,
    *,
    min_scale: float = 0.6,
    max_scale: float = 1.6,
    min_correction: float = 0.05,
) -> Any:
    """Refit the render so its body spans neck cut -> actor's feet.

    ``estimate_body_scale`` derives "model metres per target metre" from
    MediaPipe world landmarks, whose absolute scale is a body prior, not a
    measurement — on a real clip that is how a 20-30% size error survives
    every parameter tweak. The silhouette, by contrast, is measurable, so
    the render is fitted to it after the fact: the two ends of the body are
    known independently (neck cut at the top, the actor's feet at the
    bottom), so the alpha bbox is mapped onto exactly that span. Aspect is
    untouched, so a deliberately slimmer character stays slimmer.

    Corrections under ``min_correction`` are skipped: resampling every frame
    to win 2% costs sharpness for nothing. Corrections beyond
    ``min_scale..max_scale`` are refused rather than applied — that far off
    means the pose or the mask is wrong, and quietly resizing would hide a
    real fault behind a plausible-looking frame.
    """
    import cv2
    import numpy as np

    h, w = render_rgba.shape[:2]
    body = np.asarray(person_mask) > 0
    body = body.copy()
    body[:neck_y, :] = False
    if not body.any():
        return render_rgba
    ys, xs = np.where(body)
    person_bottom = float(ys.max()) + 1.0
    person_cx = float(xs.min() + xs.max()) / 2.0

    box = _alpha_bbox(render_rgba[:, :, 3])
    if box is None:
        return render_rgba
    x0, y0, x1, y1 = (float(v) for v in box)
    render_h = y1 - y0
    if render_h <= 1.0:
        return render_rgba
    want = person_bottom - float(neck_y)
    if want <= 1.0:
        return render_rgba

    scale = want / render_h
    if not (min_scale <= scale <= max_scale) or abs(scale - 1.0) < min_correction:
        return render_rgba

    # Map render bottom -> actor's feet and render horizontal centre ->
    # actor's centre. The top of the alpha then lands on the neck cut by
    # construction, because the scale was derived from that same span.
    render_cx = (x0 + x1) / 2.0
    tx = person_cx - scale * render_cx
    ty = person_bottom - scale * y1
    m = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float64)
    return cv2.warpAffine(
        render_rgba,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def composite_frame(
    orig_bgr: Any,
    render_rgba: Any,
    erase_mask: Any,
    neck_y: int,
    *,
    feather_px: int = 8,
    color_match: bool = True,
    fit_to_person: bool = True,
) -> Any:
    """Composite one frame.

    ``orig_bgr`` is the original frame (HxWx3 uint8 BGR); ``render_rgba``
    is the rendered body at the same size (HxWx4, alpha 0 outside the
    body); ``erase_mask`` is a 0/255 person mask (head + body) for the
    original frame. Steps: fit the render to the actor's silhouette,
    erase only the body pixels the render does *not* cover, feather the
    render alpha around the neck cut, optionally match the render's mean
    colour to the actor's body, then alpha-blend.

    Erasing only the uncovered part matters. Erasing the whole body and
    inpainting it (the previous behaviour) smears every background pixel
    the actor stood in front of, and a smeared background is a bigger tell
    than the swap it was meant to hide.
    """
    import cv2
    import numpy as np

    h, w = orig_bgr.shape[:2]
    mask = (np.asarray(erase_mask) > 0).astype(np.uint8) * 255
    if fit_to_person and render_rgba.shape[:2] == (h, w):
        render_rgba = fit_render_to_person(render_rgba, mask, neck_y)

    # 1. Erase: only what the render leaves uncovered, below the neck.
    cover = np.zeros((h, w), dtype=bool)
    cover[neck_y:, :] = render_rgba[neck_y:, :, 3] > 8
    # Widen the cover by a hair. The actor's outline peeking out under the
    # render's soft edge would otherwise be inpainted and then covered
    # again, doubling the blur exactly where the eye lands.
    cover = cv2.dilate(cover.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    residual = np.zeros((h, w), dtype=np.uint8)
    residual[neck_y:, :] = (
        (mask[neck_y:, :] > 0) & ~cover[neck_y:, :]
    ).astype(np.uint8) * 255
    # Generous dilation stays: the person mask under-covers a jacket edge by
    # a few pixels, and leftover dark fringe is worse than a wider inpaint.
    erase = cv2.dilate(residual, np.ones((9, 9), np.uint8))
    inpainted = (
        cv2.inpaint(orig_bgr, erase, 5, cv2.INPAINT_TELEA) if erase.any() else orig_bgr.copy()
    )

    # 2. Render alpha with a vertical feather ramp around the neck cut.
    alpha = render_rgba[:, :, 3].astype(np.float32) / 255.0
    alpha[: max(0, neck_y - feather_px), :] = 0.0
    ramp = np.linspace(0.0, 1.0, 2 * feather_px + 1, dtype=np.float32)
    lo = max(0, neck_y - feather_px)
    hi = min(h, neck_y + feather_px + 1)
    if hi > lo:
        alpha[lo:hi, :] *= ramp[: hi - lo, None]
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # 3. Colour match: per-channel gain from the actor's body mean to the
    # render's mean, clamped so a stand-in robot is darkened toward the
    # jacket tones but never crushed or blown out. Sampling the *person
    # mask* (not the erase residual) keeps this working when the render
    # covers the whole body and nothing is inpainted.
    render_rgb = render_rgba[:, :, :3].astype(np.float32)
    if color_match:
        valid = (alpha > 0.5) & (mask > 0)
        if int(valid.sum()) > 100:
            orig_mean = orig_bgr[valid].astype(np.float32).mean(axis=0)
            rend_mean = render_rgb[valid].mean(axis=0)
            gain = np.where(rend_mean > 1, orig_mean / np.maximum(rend_mean, 1.0), 1.0)
            gain = np.clip(gain, 0.6, 1.6)
            render_rgb = np.clip(render_rgb * gain[None, None, :], 0, 255)

    # 4. Blend.
    a = alpha[:, :, None]
    out = (inpainted.astype(np.float32) * (1.0 - a) + render_rgb * a).astype(np.uint8)
    # Above the neck the original is untouched by construction.
    out[:neck_y, :] = orig_bgr[:neck_y, :]
    return out


def comparison_frame(orig_bgr: Any, render_rgba: Any, composited_bgr: Any) -> Any:
    """Side-by-side comparison: original | render (on black) | composite."""
    import numpy as np

    render_on_black = render_rgba[:, :, :3].copy()
    a = (render_rgba[:, :, 3:4].astype(np.float32) / 255.0)
    render_on_black = (render_on_black.astype(np.float32) * a).astype(np.uint8)
    return np.hstack([orig_bgr, render_on_black, composited_bgr])


def composite_clip(
    video_path: str | Path,
    render_dir: str | Path,
    pose_payload: dict[str, Any],
    out_path: str | Path,
    *,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Run the full composite for a pilot clip.

    Reads the original clip and ``render_dir/frame_NNNN.png`` (RGBA) per
    frame, erases the original body below the neck, pastes the render,
    and writes an H.264 mp4 at the original resolution. One capture only:
    each frame is segmented in memory, so mask and frame can never drift
    out of sync. Returns a summary dict with frame count and output path.
    """
    import cv2
    import numpy as np

    from recast.ffmpeg import open_ffmpeg_writer

    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("mediapipe 未安装：pip install mediapipe 后再跑合成。") from exc

    render_dir = Path(render_dir)
    out_path = Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = {f["frame"]: f for f in pose_payload["frames"]}
    writer = open_ffmpeg_writer(out_path, width, height, fps)
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    written = 0
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            res = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
            png = render_dir / f"frame_{idx:04d}.png"
            if not png.is_file():
                raise FileNotFoundError(f"缺少渲染帧：{png}（先跑完渲染再合成）")
            render = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if render is None or render.shape[2] != 4:
                raise ValueError(f"渲染帧不是 RGBA：{png}")
            if (render.shape[1], render.shape[0]) != (width, height):
                render = cv2.resize(render, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            neck_y = neck_cut_y(frames[idx])
            out = composite_frame(frame, render, mask, neck_y)
            writer.stdin.write(out.tobytes())
            written += 1
            idx += 1
    finally:
        seg.close()
        cap.release()
        writer.stdin.close()
        writer.wait()
    if written == 0:
        raise ValueError(f"没有合成任何帧：{video_path}")
    return {
        "frames": written,
        "seconds": written / fps,
        "width": width,
        "height": height,
        "fps": fps,
        "out_path": str(out_path),
    }
