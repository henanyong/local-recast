"""Stage 5: per-track 2D/3D-ish keypoints and optional skeleton overlay.

Default backends are lightweight and do not require licensed SMPL / SMPL-X files.
A later fitting tool can consume the SMPL hook export (OpenPose-style arrays).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np
from tqdm import tqdm

from recast.config import RecastConfig
from recast.device import release_cuda, requested_device, vram_report
from recast.ffmpeg import VideoInfo, open_ffmpeg_writer
from recast.media import open_capture
from recast.runtime import check_cancel

log = logging.getLogger("recast")

COCO_17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# MediaPipe Pose (33) -> COCO 17
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# YOLO-pose used to accept IoU 0.1, which stuck a pose onto the wrong person.
MIN_POSE_MATCH_IOU = 0.3


class PoseEstimator(Protocol):
    name: str
    keypoint_format: str

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[list[float]],
        track_ids: list[int] | None = None,
    ) -> list[list[dict[str, float | None]]]:
        """Return one keypoint list per bbox, each keypoint is {x,y,z,score}."""

    def close(self) -> None:
        ...


def _kpt(x: float, y: float, score: float, z: float | None = None) -> dict[str, float | None]:
    return {"x": float(x), "y": float(y), "z": None if z is None else float(z), "score": float(score)}


def _empty_pose(n: int = 17) -> list[dict[str, float | None]]:
    return [_kpt(0.0, 0.0, 0.0) for _ in range(n)]


def pose_sample_every(quality: str | None, sample_every: int = 2) -> int:
    """Accuracy estimates every frame. Speed keeps one frame out of N."""
    if (quality or "accuracy").strip().lower() == "speed":
        return max(1, int(sample_every or 2))
    return 1


def frame_should_estimate(frame_idx: int, sample_every: int) -> bool:
    return int(frame_idx) % max(1, int(sample_every)) == 0


def assign_pose_detections(
    bboxes: list[list[float]],
    det_boxes: list[list[float]],
    poses: list[list[dict[str, float | None]]],
    *,
    min_iou: float = MIN_POSE_MATCH_IOU,
) -> list[list[dict[str, float | None]]]:
    """Match detected poses to person boxes. Below min_iou the person stays empty."""
    assigned: list[list[dict[str, float | None]]] = []
    used: set[int] = set()
    for bbox in bboxes:
        best_i, best = -1, 0.0
        for i, db in enumerate(det_boxes):
            if i in used:
                continue
            score = _iou(bbox, [float(x) for x in db])
            if score > best:
                best_i, best = i, score
        if best_i >= 0 and best >= float(min_iou) and best_i < len(poses):
            used.add(best_i)
            assigned.append(poses[best_i])
        else:
            assigned.append(_empty_pose())
    return assigned


def mediapipe_pose_kwargs(model_complexity: int = 0) -> dict[str, Any]:
    """Video mode: MediaPipe tracks landmarks across frames instead of restarting each still.

    ``model_complexity`` 0 is the lite model (repo default, fast); 1 is the full
    model — measurably better on profile/standing shots (pilot: mean visibility
    0.72-0.74 vs 0.62-0.65), at roughly 2x CPU cost. The rerender pilot uses 1.
    """
    return {
        "static_image_mode": False,
        "model_complexity": int(model_complexity),
        "enable_segmentation": False,
        "min_detection_confidence": 0.4,
        "min_tracking_confidence": 0.4,
    }


def _smoothing_alpha(cutoff: float, freq: float) -> float:
    freq = max(float(freq), 1e-3)
    cutoff = max(float(cutoff), 1e-3)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


class OneEuroFilter:
    """1€ filter. The first sample passes through; later samples are damped."""

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.freq = max(float(freq), 1.0)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: float | None = None
        self._dx: float | None = None
        self._last: float | None = None

    def filter(self, value: float) -> float:
        value = float(value)
        if self._last is None or self._x is None or self._dx is None:
            self._last = value
            self._x = value
            self._dx = 0.0
            return value
        dx = (value - self._last) * self.freq
        self._last = value
        a_d = _smoothing_alpha(self.d_cutoff, self.freq)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        alpha = _smoothing_alpha(cutoff, self.freq)
        self._x = alpha * value + (1.0 - alpha) * self._x
        return self._x


class KeypointSmoother:
    """OneEuro on x/y/z per track and joint. Missing points are left alone."""

    def __init__(self, fps: float, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.fps = max(1.0, float(fps or 25.0))
        self._filters: dict[tuple[int, int, str], OneEuroFilter] = {}

    def smooth(self, track_id: int, keypoints: list[dict[str, float | None]]) -> list[dict[str, float | None]]:
        if not self.enabled:
            return keypoints
        out: list[dict[str, float | None]] = []
        for joint, kp in enumerate(keypoints):
            score = float(kp.get("score") or 0.0)
            if score <= 0.05:
                out.append(dict(kp))
                continue
            item = dict(kp)
            for axis in ("x", "y", "z", "wx", "wy"):
                raw = kp.get(axis)
                if raw is None:
                    continue
                filt = self._filters.setdefault((int(track_id), joint, axis), OneEuroFilter(self.fps))
                item[axis] = filt.filter(float(raw))
            out.append(item)
        return out


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _padded_crop(frame: np.ndarray, bbox: list[float], pad: float = 0.15) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1i = max(0, int(x1 - bw * pad))
    y1i = max(0, int(y1 - bh * pad))
    x2i = min(w, int(x2 + bw * pad))
    y2i = min(h, int(y2 + bh * pad))
    if x2i <= x1i or y2i <= y1i:
        return frame[0:1, 0:1].copy(), 0, 0
    return frame[y1i:y2i, x1i:x2i], x1i, y1i


class RtmlibEstimator:
    name = "rtmlib"
    keypoint_format = "coco_17"

    def __init__(self, device: str, mode: str) -> None:
        from rtmlib import Body

        self._body = Body(mode=mode, backend="onnxruntime", device=device)
        self._device = device

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[list[float]],
        track_ids: list[int] | None = None,
    ) -> list[list[dict[str, float | None]]]:
        if not bboxes:
            return []
        np_boxes = np.asarray(bboxes, dtype=np.float32)
        keypoints, scores = self._body(frame, bboxes=np_boxes)
        return self._pack(keypoints, scores, len(bboxes))

    def _pack(self, keypoints: Any, scores: Any, n: int) -> list[list[dict[str, float | None]]]:
        kpts = np.asarray(keypoints)
        scs = np.asarray(scores)
        out: list[list[dict[str, float | None]]] = []
        if kpts.size == 0:
            return [_empty_pose() for _ in range(n)]
        if kpts.ndim == 2:
            kpts = kpts[None, ...]
            scs = scs[None, ...]
        for i in range(n):
            if i >= len(kpts):
                out.append(_empty_pose())
                continue
            person = []
            for j in range(min(17, kpts.shape[1])):
                xy = kpts[i, j]
                score = float(scs[i, j]) if scs.ndim >= 2 else float(scs[i])
                person.append(_kpt(float(xy[0]), float(xy[1]), score))
            while len(person) < 17:
                person.append(_kpt(0.0, 0.0, 0.0))
            out.append(person)
        return out

    def close(self) -> None:
        body = getattr(self, "_body", None)
        self._body = None
        release_cuda(body)


class YoloPoseEstimator:
    name = "yolo"
    keypoint_format = "coco_17"

    def __init__(self, device: str, imgsz: int, min_iou: float = MIN_POSE_MATCH_IOU) -> None:
        from ultralytics import YOLO

        log.info("加载 YOLO-pose: yolo11n-pose.pt")
        self._model = YOLO("yolo11n-pose.pt")
        self._device = 0 if device == "cuda" else "cpu"
        self._imgsz = imgsz
        self._min_iou = max(float(min_iou), MIN_POSE_MATCH_IOU)

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[list[float]],
        track_ids: list[int] | None = None,
    ) -> list[list[dict[str, float | None]]]:
        if not bboxes:
            return []
        result = self._model.predict(
            frame,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
            conf=0.25,
        )[0]
        det_xyxy = []
        poses: list[list[dict[str, float | None]]] = []
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            det_xyxy = result.boxes.xyxy.cpu().numpy().tolist()
            xy = result.keypoints.xy.cpu().numpy()
            conf = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(xy.shape[:2], dtype=np.float32)
            )
            for i in range(len(xy)):
                person = []
                for j in range(min(17, xy.shape[1])):
                    person.append(_kpt(float(xy[i, j, 0]), float(xy[i, j, 1]), float(conf[i, j])))
                while len(person) < 17:
                    person.append(_kpt(0.0, 0.0, 0.0))
                poses.append(person)
        return assign_pose_detections(bboxes, det_xyxy, poses, min_iou=self._min_iou)

    def close(self) -> None:
        model = getattr(self, "_model", None)
        self._model = None
        release_cuda(model)


class MediaPipeEstimator:
    name = "mediapipe"
    keypoint_format = "coco_17_from_mediapipe33"

    def __init__(self, model_complexity: int = 0) -> None:
        import mediapipe as mp

        self._mp = mp
        self._model_complexity = int(model_complexity)
        self._poses: dict[str, Any] = {}

    def _pose_for(self, key: str) -> Any:
        pose = self._poses.get(key)
        if pose is None:
            pose = self._mp.solutions.pose.Pose(**mediapipe_pose_kwargs(self._model_complexity))
            self._poses[key] = pose
        return pose

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[list[float]],
        track_ids: list[int] | None = None,
    ) -> list[list[dict[str, float | None]]]:
        out: list[list[dict[str, float | None]]] = []
        for index, bbox in enumerate(bboxes):
            crop, ox, oy = _padded_crop(frame, bbox)
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                out.append(_empty_pose())
                continue
            key = str(track_ids[index]) if track_ids and index < len(track_ids) else f"box{index}"
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._pose_for(key).process(rgb)
            if not result.pose_landmarks:
                out.append(_empty_pose())
                continue
            lms = result.pose_landmarks.landmark
            world = result.pose_world_landmarks.landmark if result.pose_world_landmarks else None
            ch, cw = crop.shape[:2]
            person: list[dict[str, float | None]] = []
            for src_idx in MP_TO_COCO:
                lm = lms[src_idx]
                z = None
                wx = None
                wy = None
                if world is not None:
                    z = float(world[src_idx].z)
                    # Metric 3D position, hips-centred (metres). The rerender
                    # retarget needs full 3D, not just pixel x/y + world z:
                    # pixel x/y carry the camera framing, world x/y carry the
                    # body geometry. Kept as extra keys; other stages ignore
                    # them. World axes: +x image-right, +y up, +z AWAY from
                    # the camera (smaller z = closer), origin at hips centre.
                    wx = float(world[src_idx].x)
                    wy = float(world[src_idx].y)
                else:
                    z = float(lm.z)
                item = _kpt(
                    ox + float(lm.x) * cw,
                    oy + float(lm.y) * ch,
                    float(lm.visibility or 0.0),
                    z=z,
                )
                if wx is not None:
                    item["wx"] = wx
                    item["wy"] = wy
                person.append(item)
            out.append(person)
        return out

    def close(self) -> None:
        for pose in list(self._poses.values()):
            try:
                pose.close()
            except Exception:
                pass
        self._poses.clear()


def _build_estimator(cfg: RecastConfig) -> PoseEstimator:
    device = requested_device(cfg.device)
    requested = (cfg.pose.backend or "auto").lower()
    if requested == "auto":
        order = ["rtmlib", "yolo", "mediapipe"]
    else:
        order = [requested] + [b for b in ("rtmlib", "yolo", "mediapipe") if b != requested]

    errors: list[str] = []
    for name in order:
        try:
            log.info("尝试姿态后端: %s", name)
            if name == "rtmlib":
                est: PoseEstimator = RtmlibEstimator(device=device, mode=cfg.pose.mode or "lightweight")
            elif name in {"yolo", "yolo-pose", "yolo11n-pose"}:
                est = YoloPoseEstimator(
                    device=device,
                    imgsz=int(cfg.detect.imgsz),
                    min_iou=float(getattr(cfg.pose, "min_pose_iou", MIN_POSE_MATCH_IOU) or MIN_POSE_MATCH_IOU),
                )
            elif name == "mediapipe":
                est = MediaPipeEstimator()
            else:
                raise RuntimeError(f"未知姿态后端: {name}")
            log.info("姿态后端已就绪: %s (%s)", est.name, est.keypoint_format)
            return est
        except Exception as exc:
            log.warning("姿态后端 %s 不可用: %s", name, exc)
            errors.append(f"{name}: {exc}")
    raise RuntimeError("没有可用的姿态后端:\n" + "\n".join(errors))


def draw_skeleton(frame: np.ndarray, keypoints: list[dict[str, float | None]], thr: float = 0.3) -> np.ndarray:
    vis = frame
    points: list[tuple[int, int, float]] = []
    for kp in keypoints:
        score = float(kp.get("score") or 0.0)
        points.append((int(round(float(kp["x"] or 0.0))), int(round(float(kp["y"] or 0.0))), score))
    for a, b in COCO_EDGES:
        if a >= len(points) or b >= len(points):
            continue
        if points[a][2] < thr or points[b][2] < thr:
            continue
        cv2.line(vis, points[a][:2], points[b][:2], (40, 220, 120), 2, cv2.LINE_AA)
    for x, y, score in points:
        if score < thr:
            continue
        cv2.circle(vis, (x, y), 3, (20, 80, 255), -1, cv2.LINE_AA)
    return vis


def _export_smpl_hook(track_payload: dict[str, Any], dest: Path, backend: str) -> Path:
    """Write OpenPose-style arrays for later SMPL / SMPL-X fitting. No license files required."""
    dest.mkdir(parents=True, exist_ok=True)
    frames_map: dict[int, list[dict[str, Any]]] = {}
    for track in track_payload.get("tracks") or []:
        tid = int(track["track_id"])
        name = track.get("name")
        for item in track.get("frames") or []:
            kpts = item.get("keypoints") or []
            if not kpts:
                continue
            pose_2d: list[float] = []
            for kp in kpts:
                pose_2d.extend(
                    [
                        float(kp.get("x") or 0.0),
                        float(kp.get("y") or 0.0),
                        float(kp.get("score") or 0.0),
                    ]
                )
            frames_map.setdefault(int(item["frame_idx"]), []).append(
                {
                    "track_id": tid,
                    "name": name,
                    "bbox": item.get("bbox"),
                    "pose_keypoints_2d": pose_2d,
                }
            )
    people = [
        {"frame_idx": idx, "people": frames_map[idx]}
        for idx in sorted(frames_map)
    ]
    payload = {
        "format": "openpose_coco17_like",
        "keypoint_names": COCO_17_NAMES,
        "pose_backend": backend,
        "smpl_x_required": False,
        "note": (
            "这是给后续 SMPL / SMPL-X 拟合用的关键点导出，默认流程不会加载受许可限制的 SMPL 模型。"
            "若要真正导出 SMPL-X 网格，需自行从 https://smpl-x.is.tue.mpg.de/ 申请模型文件，"
            "再接入 vid2smplx / WHAM / SMPLest-X 等工具。本文件可以直接作为那些工具的 2D 关键点输入。"
        ),
        "frames": people,
    }
    json_path = dest / "keypoints_openpose.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (dest / "README.txt").write_text(
        payload["note"] + "\n\n文件: keypoints_openpose.json\n",
        encoding="utf-8",
    )
    log.info("已写出 SMPL 钩子导出（不含 SMPL 模型本身）: %s", json_path)
    return json_path


def run(
    video: str | Path,
    info: VideoInfo,
    track_payload: dict[str, Any],
    workdir: Path,
    cfg: RecastConfig,
) -> dict[str, Any]:
    vram_report("姿态-加载前")
    estimator = _build_estimator(cfg)
    vram_report("姿态-已加载")

    by_id = {int(t["track_id"]): t for t in (track_payload.get("tracks") or [])}
    frame_dets: dict[int, list[dict[str, Any]]] = {}
    for t in by_id.values():
        for item in t.get("frames") or []:
            frame_dets.setdefault(int(item["frame_idx"]), []).append(
                {"track_id": int(t["track_id"]), "item": item}
            )

    cap = open_capture(video)
    writer = None
    skeleton_path: Path | None = None
    if cfg.pose.overlay:
        skeleton_path = workdir / "skeleton.mp4"
        writer = open_ffmpeg_writer(
            skeleton_path,
            info.width,
            info.height,
            info.fps,
            crf=cfg.compose.crf,
            use_nvenc=cfg.compose.use_nvenc,
        )

    sample_every = pose_sample_every(getattr(cfg.pose, "quality", "accuracy"), int(getattr(cfg.pose, "sample_every", 2) or 2))
    smoother = KeypointSmoother(info.fps or 25.0, enabled=bool(getattr(cfg.pose, "smooth", True)))
    held: dict[int, list[dict[str, float | None]]] = {}
    track_payload["pose_quality"] = getattr(cfg.pose, "quality", "accuracy")
    track_payload["pose_sample_every"] = sample_every
    log.info(
        "姿态采样: %s（每 %s 帧估计一次，平滑 %s）",
        "抽帧" if sample_every > 1 else "逐帧",
        sample_every,
        "开" if smoother.enabled else "关",
    )

    total = info.frame_count or None
    pbar = tqdm(total=total, desc="姿态", unit="f")
    frame_idx = 0
    try:
        while True:
            check_cancel()
            ok, frame = cap.read()
            if not ok:
                break
            dets = frame_dets.get(frame_idx, [])
            vis = frame
            if dets:
                bboxes = [d["item"]["bbox"] for d in dets]
                ids = [int(d["track_id"]) for d in dets]
                if frame_should_estimate(frame_idx, sample_every) or any(tid not in held for tid in ids):
                    estimated = estimator.estimate(frame, bboxes, ids)
                    for det, kpts in zip(dets, estimated):
                        held[int(det["track_id"])] = kpts
                poses = []
                for det in dets:
                    raw = held.get(int(det["track_id"])) or _empty_pose()
                    kpts = smoother.smooth(int(det["track_id"]), raw)
                    poses.append(kpts)
                    det["item"]["keypoints"] = kpts
                    det["item"]["keypoint_format"] = estimator.keypoint_format
                    if writer is not None:
                        vis = draw_skeleton(vis, kpts)
            if writer is not None:
                if vis.shape[1] != info.width or vis.shape[0] != info.height:
                    vis = cv2.resize(vis, (info.width, info.height))
                writer.stdin.write(np.ascontiguousarray(vis).tobytes())
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.stdin.close()
            err = writer.communicate()[1]
            if writer.returncode:
                log.warning("骨架预览写出异常: %s", (err or b"")[-1000:])
                skeleton_path = None

    track_payload["keypoint_format"] = estimator.keypoint_format
    track_payload["keypoint_names"] = COCO_17_NAMES
    track_payload["pose_backend"] = estimator.name
    tracks_path = workdir / "tracks.json"
    tracks_path.write_text(json.dumps(track_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log.info("轨迹+姿态已写出: %s", tracks_path)

    smpl_path = None
    if cfg.pose.export_smpl_hook:
        smpl_path = _export_smpl_hook(track_payload, workdir / "smpl_hook", estimator.name)

    estimator.close()
    vram_report("姿态-释放后")
    track_payload["_tracks_path"] = str(tracks_path)
    track_payload["_skeleton_path"] = str(skeleton_path) if skeleton_path else None
    track_payload["_smpl_hook_path"] = str(smpl_path) if smpl_path else None
    return track_payload
