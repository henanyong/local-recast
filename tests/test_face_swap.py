"""Tests for the face-backend inventory and its licence gate.

No weights are needed: the point under test is that the machine tells the
truth about what will run and whether it may be shipped, not that a swap
looks good. A real inswapper weight is ~550 MB and non-commercial, so the
default on a bare install is the overlay — and the overlay must never be
reported as final.
"""

from pathlib import Path

import numpy as np
import pytest

from recast.stages.face_swap import (
    FaceSwapError,
    describe_face_backends,
    select_face_backend,
    swap_frame,
)


def _write(path, bgr):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", bgr)
    assert ok
    buf.tofile(str(path))


def test_every_backend_declares_a_licence_and_whether_it_is_final():
    table = describe_face_backends()
    assert set(table) == {"inswapper", "overlay"}
    for name, row in table.items():
        assert row["license"], f"{name} 必须标注许可证"
        assert "final" in row
    assert table["overlay"]["available"] is True
    assert table["overlay"]["final"] is False  # an overlay is not a swap


def test_select_prefers_a_real_swap_when_the_weights_are_there(monkeypatch):
    import recast.stages.swap_preview as preview

    monkeypatch.setattr(preview, "inswapper_status", lambda cfg=None: {"available": True, "reason": ""})
    name, reason = select_face_backend()
    assert name == "inswapper"
    assert reason == ""


def test_select_refuses_a_noncommercial_swap_for_a_delivery_run(monkeypatch):
    import recast.stages.swap_preview as preview

    monkeypatch.setattr(preview, "inswapper_status", lambda cfg=None: {"available": True, "reason": ""})
    name, reason = select_face_backend(allow_noncommercial=False)
    assert name == "overlay"
    assert "非商用" in reason


def test_select_explains_why_it_fell_back(monkeypatch):
    import recast.stages.swap_preview as preview

    monkeypatch.setattr(
        preview, "inswapper_status", lambda cfg=None: {"available": False, "reason": "没有权重"}
    )
    name, reason = select_face_backend()
    assert name == "overlay"
    assert reason == "没有权重"


def test_swap_frame_never_claims_final_without_a_real_model(tmp_path, monkeypatch):
    import recast.stages.swap_preview as preview

    monkeypatch.setattr(
        preview, "inswapper_status", lambda cfg=None: {"available": False, "reason": "没有权重"}
    )
    frame = np.full((64, 64, 3), 120, np.uint8)
    reference = tmp_path / "ref.png"
    _write(reference, np.full((64, 64, 3), (120, 150, 190), np.uint8))

    result = swap_frame(frame, reference, tmp_dir=tmp_path / "work")

    assert result["is_final"] is False
    assert result["backend"] == "aligned_overlay"
    # The licence reported is the one that actually ran (no model at all);
    # why the real swap was skipped belongs in the note.
    assert "不涉及模型" in result["license"]
    assert "没有权重" in result["note"]
    assert result["image"].shape == frame.shape


def test_swap_frame_requires_a_reference_file(tmp_path):
    with pytest.raises(FaceSwapError, match="找不到参考脸"):
        swap_frame(np.zeros((16, 16, 3), np.uint8), tmp_path / "missing.png")


def test_delivery_flag_reaches_the_note(tmp_path, monkeypatch):
    """A delivery run must carry the licence reason into the report, so the
    person reading the log knows why the face was only overlaid."""
    import recast.stages.swap_preview as preview

    monkeypatch.setattr(preview, "inswapper_status", lambda cfg=None: {"available": True, "reason": ""})
    frame = np.full((64, 64, 3), 120, np.uint8)
    reference = tmp_path / "ref.png"
    _write(reference, np.full((64, 64, 3), (120, 150, 190), np.uint8))

    result = swap_frame(
        frame, Path(reference), allow_noncommercial=False, tmp_dir=tmp_path / "work"
    )
    assert result["is_final"] is False
    assert "InsightFace" in result["note"] or "非商用" in result["note"]
