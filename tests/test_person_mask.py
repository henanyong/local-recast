"""Tests for the offline person masker (YOLO box -> GrabCut outline).

The YOLO weights are not needed for most of these: box ordering and the
GrabCut refinement are pure logic, so they are tested with a stub detector
and a synthetic scene. The one test that does need the real weights is
skipped when ``yolo11n.pt`` is absent, so the suite stays runnable offline.
"""

from pathlib import Path

import numpy as np
import pytest

from recast.stages.person_mask import (
    PersonMaskError,
    describe_backends,
    detect_person_boxes,
    estimate_neck_y,
    grabcut_in_box,
    pick_backend,
    segment_person,
    segment_person_auto,
)

WEIGHTS = Path(__file__).resolve().parents[1] / "yolo11n.pt"


class _Boxes:
    def __init__(self, rows):
        self._rows = np.asarray(rows, dtype=np.float32)

    @property
    def xyxy(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._rows


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class _StubModel:
    def __init__(self, rows):
        self._rows = rows

    def predict(self, *_args, **_kwargs):
        return [_Result(self._rows)]


def _scene(h=160, w=160):
    """Grey plate with a distinct rectangle standing in for a body."""
    frame = np.full((h, w, 3), 110, dtype=np.uint8)
    frame[30:150, 60:110] = (40, 60, 170)
    return frame


def test_boxes_are_sorted_by_area_largest_first():
    stub = _StubModel([[0, 0, 10, 10], [0, 0, 80, 90], [100, 100, 140, 150]])
    boxes = detect_person_boxes(_scene(), model=stub)
    assert len(boxes) == 3
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    assert areas == sorted(areas, reverse=True)


def test_grabcut_recovers_a_solid_body_in_a_box():
    frame = _scene()
    mask = grabcut_in_box(frame, [58.0, 28.0, 112.0, 152.0], iterations=2, work_size=0)
    assert mask.shape == frame.shape[:2]
    inside = mask[40:140, 70:100] > 0
    outside = mask[5:25, :] > 0
    assert inside.mean() > 0.9, "前景内部应几乎全被选中"
    assert outside.mean() < 0.02, "上方背景不应被选中"


def test_grabcut_returns_empty_for_a_degenerate_box():
    frame = _scene()
    mask = grabcut_in_box(frame, [10.0, 10.0, 11.0, 11.0])
    assert int(mask.sum()) == 0


def test_grabcut_downscales_before_running():
    """A 1920-tall frame must still come back at full resolution."""
    import cv2

    big = cv2.resize(_scene(), (480, 1920), interpolation=cv2.INTER_NEAREST)
    mask = grabcut_in_box(big, [170.0, 340.0, 340.0, 1820.0], iterations=1, work_size=320)
    assert mask.shape == (1920, 480)
    assert mask.dtype == np.uint8


def test_segment_person_reports_the_box_fallback(monkeypatch):
    """When GrabCut finds nothing the caller must be told, not handed a box
    dressed up as a silhouette."""
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "grabcut_in_box", lambda *a, **k: np.zeros((160, 160), np.uint8))
    stub = _StubModel([[60, 30, 110, 150]])
    result = segment_person(_scene(), model=stub)
    assert result["method"] == "box"
    assert "GrabCut" in result["warning"]
    assert result["mask"][100, 80] == 255


def test_segment_person_rejects_an_index_beyond_the_people_present():
    stub = _StubModel([[0, 0, 50, 50]])
    with pytest.raises(PersonMaskError, match="只检测到"):
        segment_person(_scene(), index=3, model=stub)


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="需要仓库自带的 yolo11n.pt")
def test_real_detector_finds_nothing_in_a_blank_frame():
    with pytest.raises(PersonMaskError, match="没有检测到人物"):
        segment_person(np.full((240, 240, 3), 128, np.uint8), weights=WEIGHTS, conf=0.5)


def test_neck_y_falls_back_to_a_ratio_without_a_face(monkeypatch):
    import recast.stages.character_sheet as sheet

    monkeypatch.setattr(sheet, "_face_box", lambda *_a, **_k: None)
    out = estimate_neck_y(np.zeros((200, 100, 3), np.uint8))
    assert out["source"] == "ratio"
    assert out["neck_y"] == 84  # 42% of 200
    assert "估计" in out["note"]


def test_neck_y_follows_the_chin_when_a_face_is_found(monkeypatch):
    import recast.stages.character_sheet as sheet

    monkeypatch.setattr(sheet, "_face_box", lambda *_a, **_k: [30.0, 20.0, 70.0, 60.0])
    out = estimate_neck_y(np.zeros((200, 100, 3), np.uint8))
    assert out["source"] == "face"
    assert out["neck_y"] == pytest.approx(65, abs=1)  # chin 60 + 12% of 40
    assert out["note"] == ""


# ----- backend selection -----


def test_grabcut_is_always_available_and_models_are_optional():
    table = describe_backends()
    assert table["grabcut"]["available"] is True
    assert set(table) == {
        "birefnet+yolo",
        "rmbg+yolo",
        "yolo-seg",
        "mediapipe-tasks",
        "grabcut",
    }
    for name, row in table.items():
        if not row["available"]:
            assert row["reason"], f"{name} 不可用时必须说明缺什么"
        assert row.get("license"), f"{name} 必须标注许可证"


def test_auto_skips_noncommercial_models_for_a_delivery_run(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_rmbg", lambda: Path("fake/model.onnx"))
    monkeypatch.setattr(module, "find_yolo_seg", lambda: None)
    monkeypatch.setattr(module, "find_mediapipe_seg", lambda: None)
    assert pick_backend("auto")[0] == "rmbg+yolo"
    assert pick_backend("auto", allow_noncommercial=False)[0] == "grabcut"


def test_an_explicit_noncommercial_backend_is_refused_for_delivery(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_rmbg", lambda: Path("fake/model.onnx"))
    with pytest.raises(PersonMaskError, match="非商用"):
        pick_backend("rmbg+yolo", allow_noncommercial=False)


def test_auto_prefers_a_model_over_grabcut(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_birefnet", lambda: None)
    monkeypatch.setattr(module, "find_rmbg", lambda: None)
    monkeypatch.setattr(module, "find_yolo_seg", lambda: Path("fake/yolo11n-seg.pt"))
    name, reason = pick_backend("auto")
    assert name == "yolo-seg"
    assert reason == ""


def test_auto_falls_back_to_grabcut_with_a_reason(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_birefnet", lambda: None)
    monkeypatch.setattr(module, "find_rmbg", lambda: None)
    monkeypatch.setattr(module, "find_yolo_seg", lambda: None)
    monkeypatch.setattr(module, "find_mediapipe_seg", lambda: None)
    name, reason = pick_backend("auto")
    assert name == "grabcut"
    assert "慢" in reason


def test_an_explicit_backend_that_is_missing_fails_loudly(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_yolo_seg", lambda: None)
    with pytest.raises(PersonMaskError, match="yolo-seg 不可用"):
        pick_backend("yolo-seg")


def test_unknown_backend_name_is_rejected():
    with pytest.raises(PersonMaskError, match="未知分割后端"):
        pick_backend("magic")


def test_auto_reports_which_backend_actually_ran(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_birefnet", lambda: None)
    monkeypatch.setattr(module, "find_rmbg", lambda: None)
    monkeypatch.setattr(module, "find_yolo_seg", lambda: None)
    monkeypatch.setattr(module, "find_mediapipe_seg", lambda: None)
    monkeypatch.setattr(
        module,
        "segment_person",
        lambda *a, **k: {
            "mask": np.zeros((10, 10), np.uint8),
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "method": "box",
            "person_count": 1,
            "warning": "",
        },
    )
    out = segment_person_auto(np.zeros((10, 10, 3), np.uint8))
    assert out["method"] == "box"
    assert "慢" in out["warning"]


def test_mediapipe_backend_refuses_a_second_person(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_birefnet", lambda: None)
    monkeypatch.setattr(module, "find_rmbg", lambda: None)
    monkeypatch.setattr(module, "find_yolo_seg", lambda: None)
    monkeypatch.setattr(module, "find_mediapipe_seg", lambda: Path("fake/selfie.tflite"))
    with pytest.raises(PersonMaskError, match="第 2 个人"):
        segment_person_auto(np.zeros((10, 10, 3), np.uint8), index=1)


# ----- onnxruntime provider wiring -----


def test_provider_is_unknown_before_any_session_runs():
    import recast.stages.person_mask as module

    assert module.rmbg_provider() in {"", "cuda", "cpu"}


def test_registering_torch_dlls_never_raises_on_this_platform():
    """The registration is a Windows-only optimisation; on any platform it
    must be a no-op at worst, never an exception that kills a mask run."""
    import recast.stages.person_mask as module

    module._register_torch_cuda_dlls()  # must not raise


def test_cuda_can_be_disabled_by_env(monkeypatch):
    import importlib

    import recast.stages.person_mask as module

    monkeypatch.setenv("RECAST_RMBG_CPU", "1")
    reloaded = importlib.reload(module)
    try:
        assert reloaded._RMBG_PREFER_CUDA is False
    finally:
        monkeypatch.delenv("RECAST_RMBG_CPU", raising=False)
        importlib.reload(reloaded)


# ----- person ordering must follow owned foreground, not box area -----


def test_ranking_prefers_the_tight_box_over_a_mega_box():
    """A near-full-frame box is sometimes emitted on a two-shot. Ranking by
    area made it "person 1" while it owned a sliver of wall; ranking by the
    foreground it actually owns puts the real subject first."""
    import recast.stages.person_mask as module

    alpha = np.zeros((100, 100), np.float32)
    alpha[20:80, 30:70] = 1.0  # one person-sized blob
    mega = [0.0, 0.0, 100.0, 100.0]
    tight = [28.0, 18.0, 72.0, 82.0]
    ranked = module.rank_boxes_by_owned_foreground(alpha, [mega, tight])
    assert ranked[0][0] == 1, "紧凑框应排在前面"
    assert ranked[0][1] > 2000
    assert len(ranked) == 1, "只占背景的大框应被丢掉"


def test_ranking_keeps_two_people_who_each_own_foreground():
    import recast.stages.person_mask as module

    alpha = np.zeros((100, 200), np.float32)
    alpha[20:80, 10:60] = 1.0
    alpha[20:80, 120:190] = 1.0
    boxes = [[0.0, 0.0, 100.0, 100.0], [110.0, 0.0, 200.0, 100.0]]
    ranked = module.rank_boxes_by_owned_foreground(alpha, boxes)
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]
    assert {slot for slot, _ in ranked} == {0, 1}


def test_an_index_beyond_the_ranked_people_is_refused(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "rmbg_foreground", lambda bgr, model_path=None: np.ones((40, 40), np.float32))
    monkeypatch.setattr(module, "detect_person_boxes", lambda *a, **k: [[0.0, 0.0, 40.0, 40.0]])
    # Reference the class through the module: an earlier test reloads it, and a
    # stale imported class would not catch the freshly raised one.
    with pytest.raises(module.PersonMaskError) as excinfo:
        module.segment_with_rmbg(np.zeros((40, 40, 3), np.uint8), index=1)
    assert "取不到第 2 个" in str(excinfo.value)


def test_auto_prefers_instance_segmentation_over_salient_foreground(monkeypatch):
    """``yolo-seg`` gives per-instance masks; rmbg gives the whole foreground and
    needs box ownership, which is what cut a vertical seam through a two-shot.
    Quality order therefore puts yolo-seg first even though its licence is the
    stricter one — the licence is filtered separately."""
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_birefnet", lambda: None)
    monkeypatch.setattr(module, "find_yolo_seg", lambda: Path("fake/yolo11n-seg.pt"))
    monkeypatch.setattr(module, "find_rmbg", lambda: Path("fake/rmbg.onnx"))
    assert module.pick_backend("auto")[0] == "yolo-seg"


def test_a_delivery_run_refuses_agpl_and_noncommercial(monkeypatch):
    import recast.stages.person_mask as module

    monkeypatch.setattr(module, "find_birefnet", lambda: None)
    monkeypatch.setattr(module, "find_yolo_seg", lambda: Path("fake/yolo11n-seg.pt"))
    monkeypatch.setattr(module, "find_rmbg", lambda: Path("fake/rmbg.onnx"))
    name, reason = module.pick_backend("auto", allow_noncommercial=False)
    assert name == "grabcut", "交付模式不能落到 AGPL 或非商用模型"
    assert reason


def test_every_backend_declares_a_commercial_flag():
    for name, row in describe_backends().items():
        assert isinstance(row["commercial_ok"], bool), f"{name} 缺 commercial_ok"
