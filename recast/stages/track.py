"""Stage 3: Ultralytics YOLO nano + ByteTrack person tracking."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from recast.config import RecastConfig
from recast.device import release_cuda, requested_device, vram_report
from recast.ffmpeg import VideoInfo
from recast.runtime import check_cancel

log = logging.getLogger("recast")

PERSON_CLASS = 0  # COCO person


def _xyxy_list(box: Any) -> list[float]:
    return [float(x) for x in box]


def run(
    video: str | Path,
    info: VideoInfo,
    workdir: Path,
    cfg: RecastConfig,
) -> dict[str, Any]:
    from ultralytics import YOLO

    device = requested_device(cfg.device)
    log.info(
        "加载检测模型 %s (imgsz=%s, conf=%s, tracker=%s, device=%s)",
        cfg.detect.model,
        cfg.detect.imgsz,
        cfg.detect.conf,
        cfg.detect.tracker,
        device,
    )
    log.info("首次运行会自动下载 YOLO 权重（yolo11n 约 5-6 MB）。")
    vram_report("跟踪-加载前")
    model = YOLO(cfg.detect.model)
    vram_report("跟踪-已加载")

    frame_entries: list[dict[str, Any]] = []
    tracks: dict[int, dict[str, Any]] = {}
    seen_ids: set[int] = set()

    stream = model.track(
        source=str(video),
        persist=True,
        tracker=cfg.detect.tracker,
        classes=[PERSON_CLASS],
        imgsz=int(cfg.detect.imgsz),
        conf=float(cfg.detect.conf),
        iou=float(cfg.detect.iou),
        device=0 if device == "cuda" else "cpu",
        stream=True,
        verbose=False,
        workers=max(1, min(2, cfg.cpu_threads)),
        vid_stride=1,
    )

    frame_idx = -1
    try:
        from tqdm import tqdm

        total = info.frame_count or None
        iterator = tqdm(stream, total=total, desc="跟踪", unit="f")
    except Exception:
        iterator = stream

    for result in iterator:
        check_cancel()
        frame_idx += 1
        boxes = result.boxes
        detections: list[dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [None] * len(xyxy)
            ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)
            for box, conf, tid in zip(xyxy, confs, ids):
                if tid is None:
                    continue
                track_id = int(tid)
                bbox = _xyxy_list(box)
                conf_f = float(conf) if conf is not None else None
                detections.append(
                    {
                        "track_id": track_id,
                        "bbox": bbox,
                        "conf": conf_f,
                    }
                )
                entry = tracks.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "name": None,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "frames": [],
                    },
                )
                entry["last_frame"] = frame_idx
                entry["frames"].append(
                    {
                        "frame_idx": frame_idx,
                        "time": frame_idx / info.fps if info.fps else 0.0,
                        "bbox": bbox,
                        "conf": conf_f,
                    }
                )
                seen_ids.add(track_id)
        frame_entries.append({"frame_idx": frame_idx, "detections": detections})

    payload: dict[str, Any] = {
        "video": info.to_dict(),
        "model": cfg.detect.model,
        "tracker": cfg.detect.tracker,
        "track_ids": sorted(seen_ids),
        "tracks": [tracks[i] for i in sorted(tracks)],
        "frames": frame_entries,
    }
    out = workdir / "tracks_raw.json"
    # frames array can be large; still useful. Also write a compact track list.
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info(
        "跟踪完成: %s 个轨迹, %s 帧。原始数据: %s",
        len(tracks),
        frame_idx + 1,
        out,
    )
    held = model
    model = None
    release_cuda(held)
    return payload


def index_by_frame(track_payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    table: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in track_payload.get("frames") or []:
        table[int(frame["frame_idx"])] = list(frame.get("detections") or [])
    return table
