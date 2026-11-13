"""Tests for the 2D replacement orchestration.

What is pinned here is the reporting contract, not the pixels: every step
says whether it ran, a step that could not run lands in ``notes``, and
``final`` is False whenever the result is a partial (an overlay instead of a
swap, or no costume at all). A pipeline that silently returns an unchanged
frame and calls it done is the failure mode these tests exist to prevent.
"""

import numpy as np
import pytest

from recast.core.person_replace import rejection_reason, replace_clip, replace_frame

TOP = 20
X0, X1 = 20, 40


def _write(path, bgr):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", bgr)
    assert ok
    buf.tofile(str(path))


def _frame(h=48, w=64, value=120):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _coloured_frame(h=48, w=64):
    """Grey plate with a saturated body: a neutral body is intentionally left
    alone by the recolour guard, so colour assertions need real chroma."""
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    frame[TOP:, X0:X1] = (40, 60, 190)
    return frame


def _mask(h=48, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[TOP:, X0:X1] = 255
    return mask


def _character(tmp_path, with_costume=True, with_face=False):
    folder = tmp_path / "char"
    folder.mkdir()
    if with_costume:
        _write(folder / "costume" / "01.png", np.full((40, 40, 3), (200, 40, 40), np.uint8))
    if with_face:
        _write(folder / "face" / "01.png", np.full((40, 40, 3), (120, 150, 190), np.uint8))
    return folder


def test_costume_step_runs_and_reports(tmp_path):
    frame = _coloured_frame()
    result = replace_frame(
        frame,
        character_folder=_character(tmp_path),
        person_mask=_mask(),
        neck_y=TOP,
        parts=("full",),
    )
    garment = [s for s in result["steps"] if s["step"].startswith("garment")]
    assert garment and garment[0]["ok"] is True
    assert garment[0]["region_pixels"] == int((_mask() > 0).sum())
    assert result["final"] is True
    assert result["audit"] is not None
    # The costume colour actually landed inside the body region.
    inside = result["image"][_mask() > 0]
    assert inside[:, 0].mean() > inside[:, 1].mean() + 15


def test_missing_costume_is_reported_not_silently_skipped(tmp_path):
    frame = _frame()
    result = replace_frame(
        frame,
        character_folder=_character(tmp_path, with_costume=False),
        person_mask=_mask(),
        neck_y=TOP,
        parts=("full",),
    )
    assert result["final"] is False
    assert any("没有服装参考图" in note for note in result["notes"])
    assert np.array_equal(result["image"], frame)


def test_face_step_requires_a_reference_photo(tmp_path):
    result = replace_frame(
        _frame(),
        character_folder=_character(tmp_path, with_face=False),
        person_mask=_mask(),
        neck_y=TOP,
        parts=("full",),
        do_face=True,
    )
    face = [s for s in result["steps"] if s["step"] == "face"]
    assert face and face[0]["ok"] is False
    assert any("参考脸" in note for note in result["notes"])
    assert result["final"] is False  # a partial result is never "final"


def test_face_step_reports_whether_it_was_a_real_swap(tmp_path):
    result = replace_frame(
        _frame(),
        character_folder=_character(tmp_path, with_face=True),
        person_mask=_mask(),
        neck_y=TOP,
        parts=("full",),
        do_face=True,
    )
    face = [s for s in result["steps"] if s["step"] == "face"]
    assert face and face[0]["ok"] is True
    assert face[0]["method"] in {"inswapper", "aligned_overlay"}
    # The machine has no inswapper model, so this must NOT claim to be final.
    if face[0]["method"] == "aligned_overlay":
        assert face[0]["is_final"] is False
        assert result["final"] is False


def test_clip_writes_a_video_and_audits_every_frame(tmp_path):
    import cv2

    clip = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(4):
        writer.write(_frame())
    writer.release()

    result = replace_clip(
        clip,
        tmp_path / "out.mp4",
        character_folder=_character(tmp_path),
        mask_fn=lambda frame, index: _mask(),
        neck_y_fn=lambda frame, index: TOP,
        parts=("full",),
        frame_limit=3,
    )

    assert result["frames"] == 3
    assert result["audited"] == 3
    assert result["seam_ratio_mean"] is not None
    assert len(result["worst_frames"]) <= 10
    assert (tmp_path / "out.mp4").is_file()
    assert (tmp_path / "out.mp4").stat().st_size > 0
    # An explicit neck line was supplied, so no proportional-guess warning.
    assert not any("比例估计" in note for note in result["notes"])


def test_clip_refuses_a_missing_neck_line_upfront(tmp_path):
    """A guessed neck line (42% of frame height) wrecked whole clips: the
    caller must now supply a measured one (estimate_neck_y), and the refusal
    happens before any frame is processed."""
    import cv2

    clip = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(2):
        writer.write(_frame())
    writer.release()

    char = _character(tmp_path)
    # Explicitly passing None gets the guidance message...
    with pytest.raises(TypeError, match=r"neck_y_fn.*estimate_neck_y"):
        replace_clip(
            clip,
            tmp_path / "out.mp4",
            character_folder=char,
            mask_fn=lambda frame, index: _mask(),
            neck_y_fn=None,
            parts=("full",),
            frame_limit=2,
        )
    assert not (tmp_path / "out.mp4").exists()  # refused before any work
    # ...and omitting it is a signature-level error.
    with pytest.raises(TypeError, match="neck_y_fn"):
        replace_clip(
            clip,
            tmp_path / "out.mp4",
            character_folder=char,
            mask_fn=lambda frame, index: _mask(),
            parts=("full",),
            frame_limit=2,
        )


def test_clip_refuses_to_invent_a_segmentation(tmp_path):
    import cv2

    clip = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    writer.write(_frame())
    writer.release()

    with pytest.raises(TypeError):
        replace_clip(clip, tmp_path / "out.mp4", character_folder=_character(tmp_path))  # type: ignore[call-arg]


# ----- per-frame quality gate (a bad frame must not reach the writer) -----


def test_rejection_reason_flags_background_damage_only():
    assert rejection_reason(None) is None
    assert rejection_reason({}) is None
    # Below the 2% limit the composite goes through, whatever the seam says.
    assert rejection_reason({"background_damage_frac": 0.019}) is None
    assert rejection_reason({"background_damage_frac": 0.5}) == "background_damage"


def _bad_audit(background_damage_frac: float = 0.42) -> dict:
    return {
        "background_damage_frac": background_damage_frac,
        "background_damage_mean": 90.0,
        "seam_ratio": 1.0,
        "seam_ratio_before": 1.0,
        "seam_increase": 1.0,
        "render_coverage": None,
        "verdict": "review",
        "problems": [f"背景被改坏 {background_damage_frac:.0%}"],
    }


def test_clip_rejects_damaged_frames_and_writes_the_original(tmp_path, monkeypatch):
    """The gate runs on the final composite and the WRITER only ever sees a
    frame that passed: the rejected result is replaced by the source frame."""
    import cv2

    from recast.core import person_replace as pr

    clip = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(3):
        writer.write(_frame())
    writer.release()

    calls = {"n": 0}

    def fake_replace_frame(frame_bgr, **kwargs):
        calls["n"] += 1
        return {
            # A composite that wrecked the background everywhere.
            "image": np.full_like(frame_bgr, 7),
            "steps": [{"step": "garment:full", "ok": True, "region_pixels": 1}],
            "audit": _bad_audit(),
            "final": True,
            "notes": [],
        }

    monkeypatch.setattr(pr, "replace_frame", fake_replace_frame)
    result = pr.replace_clip(
        clip,
        tmp_path / "out.mp4",
        character_folder=_character(tmp_path),
        mask_fn=lambda frame, index: _mask(),
        neck_y_fn=lambda frame, index: TOP,
        parts=("full",),
        frame_limit=3,
    )

    assert calls["n"] == 3
    assert result["frames"] == 3
    assert result["rejected_frames"] == 3
    assert result["reject_reason_counts"] == {"background_damage": 3}
    assert [item["frame"] for item in result["rejected"]] == [0, 1, 2]
    assert (tmp_path / "out.mp4").stat().st_size > 0
    # The output must carry the ORIGINAL frames: mp4v is lossy, so compare
    # loosely — the wrecked composite would sit ~113 grey levels away.
    cap = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    ok, first = cap.read()
    cap.release()
    assert ok
    diff = float(np.abs(first.astype(np.int16) - _frame().astype(np.int16)).mean())
    # mp4v is lossy; a flat grey frame decodes ~5 levels off. The wrecked
    # composite would sit ~113 levels away, so 15 cleanly separates them.
    assert diff < 15.0, f"被拒绝的帧必须写原帧，实际平均差 {diff:.1f}"


def test_clip_reports_zero_rejections_when_frames_are_clean(tmp_path):
    import cv2

    clip = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(2):
        writer.write(_coloured_frame())
    writer.release()

    result = replace_clip(
        clip,
        tmp_path / "out.mp4",
        character_folder=_character(tmp_path),
        mask_fn=lambda frame, index: _mask(),
        neck_y_fn=lambda frame, index: TOP,
        parts=("full",),
        frame_limit=2,
    )
    assert result["rejected_frames"] == 0
    assert result["reject_reason_counts"] == {}
    assert result["rejected"] == []
