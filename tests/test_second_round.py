"""Leftover second-round audit behavior: encode, pose, scenes, identity, detect-N, progress, Jianying."""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from recast.compose.jianying import _dur_us, build_audio_segments, export_jianying_draft
from recast.config import RecastConfig, apply_runtime_settings
from recast.core.engine import cached_comfy_available, clear_engine_cache, engine_snapshot
from recast.core.scheduler import Scheduler, should_persist_progress
from recast.core.settings import AppSettings
from recast.extract.scenes import ffmpeg_scene_score, merge_short_shots
from recast.ffmpeg import (
    DEFAULT_CRF,
    H264_SLOT,
    concat_reencode_args,
    expand_h264,
    h264_args,
    run_h264,
    set_nvenc_probe,
)
from recast.stages.pose import (
    MIN_POSE_MATCH_IOU,
    KeypointSmoother,
    OneEuroFilter,
    assign_pose_detections,
    frame_should_estimate,
    mediapipe_pose_kwargs,
    pose_sample_every,
)
from recast.stages.replace import (
    SwapTemporalSmoother,
    collect_faces,
    detection_plan,
    plan_replace_targets,
    shift_detected_face,
    split_tracks_by_identity,
)
from recast.store.db import Store


def test_h264_args_share_one_quality_number_and_nvenc_falls_back(monkeypatch):
    soft = h264_args(crf=18, use_nvenc=False)
    hard = h264_args(crf=18, use_nvenc=True)
    assert soft[soft.index("-crf") + 1] == "18"
    assert "libx264" in soft
    assert "h264_nvenc" not in soft
    assert hard[hard.index("-cq") + 1] == "18"
    assert "h264_nvenc" in hard
    assert "23" not in soft and "23" not in hard
    fallback = expand_h264(concat_reencode_args("list.txt", "out.mp4"), crf=DEFAULT_CRF, use_nvenc=False)
    assert fallback[fallback.index("-crf") + 1] == "18"
    assert "23" not in fallback

    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(list(args))
        if "h264_nvenc" in args:
            from recast.ffmpeg import FFmpegError

            raise FFmpegError("nvenc missing")
        return None

    monkeypatch.setattr("recast.ffmpeg.run_ffmpeg", fake_run)
    monkeypatch.setattr("recast.ffmpeg.nvenc_available", lambda: True)
    set_nvenc_probe(True)
    run_h264(["-i", "in.mp4", H264_SLOT, "out.mp4"], crf=18, use_nvenc=True)
    assert "h264_nvenc" in calls[0]
    assert "libx264" in calls[1]
    assert calls[1][calls[1].index("-crf") + 1] == "18"
    set_nvenc_probe(None)


def test_eight_gig_preset_uses_beam_size_three():
    cfg = RecastConfig.from_yaml(Path("configs/4060ti-8g.yaml"))
    assert cfg.whisper.beam_size == 3
    assert cfg.compose.crf == 18
    assert cfg.compose.use_nvenc is True
    wide = RecastConfig.from_yaml(Path("configs/4060ti-16g.yaml"))
    assert wide.whisper.beam_size == 3
    assert wide.compose.crf == 17


def test_runtime_settings_override_yaml_without_touching_beam():
    cfg = RecastConfig.from_yaml(Path("configs/4060ti-8g.yaml"))
    settings = AppSettings(export_crf=20, use_nvenc=False, scene_threshold=32, min_shot_sec=0.8, detect_every=6, pose_quality="speed")
    apply_runtime_settings(cfg, settings)
    assert cfg.whisper.beam_size == 3
    assert cfg.compose.crf == 20
    assert cfg.compose.use_nvenc is False
    assert cfg.scene.threshold == 32
    assert cfg.scene.min_shot_sec == 0.8
    assert cfg.replace.detect_every == 6
    assert cfg.pose.quality == "speed"


def test_pose_match_threshold_and_smoothing_hooks():
    assert MIN_POSE_MATCH_IOU >= 0.3
    assert mediapipe_pose_kwargs()["static_image_mode"] is False
    loose = [0.0, 0.0, 10.0, 10.0]
    overlap = [6.0, 0.0, 16.0, 10.0]  # IoU 0.25: old 0.1 match, new 0.3 miss
    pose = [{"x": 1.0, "y": 2.0, "z": None, "score": 1.0}]
    missed = assign_pose_detections([loose], [overlap], [pose], min_iou=MIN_POSE_MATCH_IOU)
    assert missed[0][0]["score"] == 0.0
    tight = [1.0, 1.0, 9.0, 9.0]
    hit = assign_pose_detections([loose], [tight], [pose], min_iou=MIN_POSE_MATCH_IOU)
    assert hit[0][0]["x"] == 1.0
    assert pose_sample_every("accuracy", 4) == 1
    assert pose_sample_every("speed", 4) == 4
    assert frame_should_estimate(0, 4) is True
    assert frame_should_estimate(1, 4) is False
    filt = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.0)
    assert filt.filter(0.0) == 0.0
    assert 0.0 < filt.filter(10.0) < 10.0
    smoother = KeypointSmoother(30.0, enabled=True)
    first = smoother.smooth(1, [{"x": 0.0, "y": 0.0, "z": 0.0, "score": 1.0}])
    second = smoother.smooth(1, [{"x": 10.0, "y": 0.0, "z": 0.0, "score": 1.0}])
    assert first[0]["x"] == 0.0
    assert 0.0 < float(second[0]["x"]) < 10.0


def test_scene_backends_share_a_threshold_and_short_shots_merge():
    assert ffmpeg_scene_score(27) == pytest.approx(0.30)
    assert ffmpeg_scene_score(13.5) == pytest.approx(0.15)
    shots = [
        {"start_time": 0.0, "end_time": 2.0, "start_frame": 0, "end_frame": 50},
        {"start_time": 2.0, "end_time": 2.12, "start_frame": 50, "end_frame": 53},
        {"start_time": 2.12, "end_time": 5.0, "start_frame": 53, "end_frame": 125},
    ]
    merged = merge_short_shots(shots, 0.4)
    assert len(merged) == 2
    assert merged[0]["end_time"] == 2.12
    assert merged[0]["index_no"] == 0
    assert merged[1]["start_time"] == 2.12
    leading = [
        {"start_time": 0.0, "end_time": 0.1, "start_frame": 0, "end_frame": 3},
        {"start_time": 0.1, "end_time": 3.0, "start_frame": 3, "end_frame": 75},
    ]
    folded = merge_short_shots(leading, 0.4)
    assert len(folded) == 1
    assert folded[0]["start_time"] == 0.0
    assert folded[0]["end_time"] == 3.0


def test_track_identity_precheck_rejects_a_weak_track_and_keeps_the_cluster_skip():
    kept, rejected = split_tracks_by_identity([1.0, 0.0], {3: [1.0, 0.0], 9: [0.0, 1.0]}, [3, 9], 0.55)
    assert kept == [3]
    assert rejected[0]["track_id"] == 9
    assert "已拒绝替换" in rejected[0]["reason"]
    identity = {
        "cluster_threshold": 0.45,
        "track_embeddings": {"3": [1.0, 0.0], "9": [0.0, 1.0]},
        "persons": [
            {"name": "温念", "cluster_id": 1, "track_ids": [3, 9], "embedding": [1.0, 0.0]},
        ],
    }
    tracks = {
        "tracks": [
            {"track_id": 3, "cluster_id": 1, "frames": [{"frame_idx": 0, "bbox": [0, 0, 1, 1]}]},
            {"track_id": 9, "cluster_id": 1, "frames": [{"frame_idx": 0, "bbox": [2, 2, 3, 3]}]},
        ]
    }
    refs = [{"character_id": "c1", "name": "温念", "face_path": "a.png", "embedding": [1.0, 0.0]}]
    plan = plan_replace_targets(refs, identity, tracks, min_similarity=0.55)
    assert plan["targets"][0]["track_ids"] == [3]
    assert plan["targets"][0]["rejected_tracks"][0]["track_id"] == 9
    # A reference that misses the cluster is still skipped, not swapped.
    stranger = [{"character_id": "c2", "name": "路人", "face_path": "b.png", "embedding": [0.0, 0.0]}]
    skipped = plan_replace_targets(stranger, identity, tracks, min_similarity=0.55)
    assert skipped["targets"] == []
    assert "已跳过" in skipped["skipped"][0]["reason"]


def test_detect_every_n_frames_uses_the_track_crop():
    assert detection_plan(0, 4, True) == "full"
    assert detection_plan(1, 4, True) == "crop"
    assert detection_plan(1, 4, False) == "full"
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    box = [10, 10, 30, 40]

    class Face:
        def __init__(self) -> None:
            self.bbox = np.array([1, 2, 8, 9], dtype=np.float32)
            self.kps = np.array([[1, 2], [3, 4]], dtype=np.float32)

    class App:
        def __init__(self, found: bool) -> None:
            self.found = found
            self.shapes: list[tuple[int, ...]] = []

        def get(self, image):
            self.shapes.append(image.shape)
            if not self.found:
                return []
            if image.shape[0] == frame.shape[0]:
                return []
            return [Face()]

    crop_app = App(True)
    faces = collect_faces(crop_app, frame, [box], full=False)
    assert len(crop_app.shapes) == 1
    assert crop_app.shapes[0][0] < frame.shape[0]
    assert float(faces[0].bbox[0]) > 1
    shifted = shift_detected_face(Face(), 5, 7)
    assert float(shifted.bbox[0]) == 6
    assert float(shifted.kps[1, 1]) == 11
    empty = App(False)
    collect_faces(empty, frame, [box], full=False)
    assert empty.shapes[-1] == frame.shape

    previous = np.zeros((20, 20, 3), dtype=np.uint8)
    previous[:, :] = (0, 0, 10)
    smoother = SwapTemporalSmoother(alpha=0.5)
    first = smoother.apply(previous, [{"track_id": 1, "bbox": [0, 0, 10, 10]}])
    assert int(first[0, 0, 2]) == 10
    current = np.zeros_like(previous)
    current[:, :] = (10, 0, 0)
    blended = smoother.apply(current, [{"track_id": 1, "bbox": [0, 0, 10, 10]}])
    assert int(blended[0, 0, 0]) == 5
    assert int(blended[0, 0, 2]) == 5


def test_progress_throttle_and_future_cleanup(tmp_path: Path):
    assert should_persist_progress(0.0, 1.0, 0.1) is True
    assert should_persist_progress(1.0, 1.1, 0.2) is False
    assert should_persist_progress(1.0, 1.25, 0.4) is True
    assert should_persist_progress(1.0, 1.05, 1.0) is True
    store = Store(tmp_path / "db.db")
    sched = Scheduler(store, AppSettings())
    done = Future()
    done.set_result(None)
    pending = Future()
    sched._futures.extend([done, pending])
    assert sched.reap_futures() == 1
    assert sched._futures == [pending]


def test_comfy_probe_is_cached(monkeypatch):
    clear_engine_cache()
    calls = {"n": 0}

    def probe(url, timeout=1.0):
        calls["n"] += 1
        return url == "http://127.0.0.1:9"

    monkeypatch.setattr("recast.generate.comfy.comfy_available", probe)
    assert cached_comfy_available("http://127.0.0.1:9", now=10.0) is True
    assert cached_comfy_available("http://127.0.0.1:9", now=12.0) is True
    assert calls["n"] == 1
    assert cached_comfy_available("http://127.0.0.1:9", now=20.0) is True
    assert calls["n"] == 2
    clear_engine_cache()
    calls["n"] = 0
    monkeypatch.setattr("recast.core.engine.detect_vram_gb", lambda: None)
    monkeypatch.setattr("recast.core.engine.detect_gpu_name", lambda: None)
    monkeypatch.setattr("recast.core.engine.module_status", lambda: {})
    monkeypatch.setattr("recast.core.engine.status_report", lambda cfg: [])
    monkeypatch.setattr("recast.core.engine.ffmpeg_status", lambda directory="": {"ok": True})
    monkeypatch.setattr("recast.core.engine.disk_free_gb", lambda path: 10.0)
    monkeypatch.setattr("recast.assets3d.drive.liveportrait_available", lambda: False)
    settings = AppSettings(comfyui_url="http://127.0.0.1:9")
    engine_snapshot(settings)
    engine_snapshot(settings)
    # one probe from the two snapshots; the explicit calls above were cleared
    assert calls["n"] == 1
    clear_engine_cache()


def test_jianying_uses_relative_paths_and_per_segment_audio(tmp_path: Path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"v")
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"a")
    shot = tmp_path / "S01.mp4"
    shot.write_bytes(b"s")
    shots = [
        {
            "shot_code": "S01",
            "start_time": 0.0,
            "duration_sec": 4.0,
            "output_path": str(shot),
            "dialogue": "你好",
            "audio_duration": 1.5,
            "audio_path": str(mix),
        },
        {
            "shot_code": "S02",
            "start_time": 4.0,
            "duration_sec": 4.0,
            "dialogue": "再见",
            "audio_duration": 2.0,
            "audio_path": str(mix),
        },
    ]
    built = build_audio_segments(shots)
    assert [row["duration"] for row in built] == [1.5, 2.0]
    assert 8.0 not in [row["duration"] for row in built]
    dest = export_jianying_draft(tmp_path / "draft", shots=shots, output_video=video, audio=mix, title="本地再铸导出")
    project = json.loads((dest / "project.json").read_text(encoding="utf-8"))
    assert project["video"] == "media/out.mp4"
    assert project["paths"] == "relative"
    assert Path(project["video"]).is_absolute() is False
    assert [row["duration"] for row in project["audio_segments"]] == [1.5, 2.0]
    assert project["jianying_verified"] is False
    draft = json.loads((dest / "draft_info.json").read_text(encoding="utf-8"))
    audio = draft["tracks"][1]["segments"]
    assert audio[0]["target_timerange"]["duration"] == _dur_us(1.5)
    assert audio[1]["target_timerange"]["duration"] == _dur_us(2.0)
    assert audio[0]["material"]["path"].startswith("media/")
    content = json.loads((dest / "draft_content.json").read_text(encoding="utf-8"))
    assert content["platform"]["verified"] is False
    assert "5.x" in content["platform"]["target"]
    edl = (dest / "timeline.edl").read_text(encoding="utf-8")
    assert "FROM CLIP NAME: media/" in edl
    assert (dest / "timeline_fcp.xml").is_file()
    readme = (dest / "README.txt").read_text(encoding="utf-8")
    assert "没有打开过剪映" in readme
    assert "已验证能打开" not in readme or "不能把" in readme
    blob = json.dumps(project, ensure_ascii=False) + edl + readme
    assert str(tmp_path) not in blob
