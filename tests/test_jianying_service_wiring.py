"""Service wiring for the Jianying export: pre-validate, build, self-check.

Covers ``recast/core/runner.py::_run_export_stages`` — the only place the
pipeline builds a Jianying draft. Fake 1-byte media files, like
tests/test_jianying_validate.py; no real video, no Jianying, no runner
stages. The two monkeypatch tests prove the service-level pre-validation
and post-write self-check are actually invoked (not just the checks
``export_jianying_draft`` runs internally).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recast.compose.jianying import JianyingCheckResult, JianyingDraftError
from recast.core.runner import _run_export_stages


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _shots(tmp_path: Path) -> tuple[list[dict], Path, Path]:
    shot = _write(tmp_path / "S01.mp4", b"s")
    mix = _write(tmp_path / "mix.wav", b"w")
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
            "output_path": str(shot),
            "dialogue": "再见",
            "audio_duration": 2.0,
            "audio_path": str(mix),
        },
    ]
    return shots, shot, mix


GOOD_SRT = """1
00:00:00,000 --> 00:00:04,000
你好

2
00:00:04,000 --> 00:00:08,000
再见
"""

BAD_SRT_OVERLAP = """1
00:00:00,000 --> 00:00:05,000
你好

2
00:00:04,000 --> 00:00:08,000
再见
"""


def _srt(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "subs.srt"
    p.write_text(text, encoding="utf-8")
    return p


def _dialogue(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dialogue.json"
    p.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
    return p


def _aligned_dialogue() -> list[dict]:
    return [
        {"shot_code": "S01", "speaker": "温念", "text": "你好", "start_time": 0.0, "end_time": 4.0},
        {"shot_code": "S02", "speaker": "路人甲", "text": "再见", "start_time": 4.0, "end_time": 8.0},
    ]


def _export(tmp_path: Path, *, entry_type: str = "import", **overrides):
    shots, output, _mix = _shots(tmp_path)
    kwargs: dict = {
        "workdir": tmp_path,
        "stages": ["export"],
        "store": None,
        "task_id": None,
        "drama_id": None,
        "shots": shots,
        "output": output,
        "entry_type": entry_type,
    }
    # Default SRT only when the caller does not pass one: writing it here
    # would overwrite the caller's file (same subs.srt path).
    if "srt" not in overrides:
        kwargs["srt"] = _srt(tmp_path, GOOD_SRT)
    kwargs.update(overrides)
    return _run_export_stages(**kwargs)


def test_export_stage_builds_draft_when_valid(tmp_path: Path):
    result = _export(tmp_path)

    assert "jianying" in result
    assert "cost_xlsx" in result
    draft = Path(result["jianying"])
    assert (draft / "draft_content.json").is_file()
    assert (draft / "project.json").is_file()


def test_export_stage_refuses_missing_video_and_writes_nothing(tmp_path: Path):
    with pytest.raises(JianyingDraftError) as excinfo:
        _export(tmp_path, output=tmp_path / "nope.mp4")
    assert "成片视频" in str(excinfo.value)
    # Nothing written: no draft dir at all.
    assert not (tmp_path / "delivery" / "jianying_draft").exists()


def test_export_stage_refuses_bad_srt(tmp_path: Path):
    with pytest.raises(JianyingDraftError) as excinfo:
        _export(tmp_path, srt=_srt(tmp_path, BAD_SRT_OVERLAP))
    assert "字幕" in str(excinfo.value)
    assert not (tmp_path / "delivery" / "jianying_draft").exists()


def test_export_stage_runs_prevalidation_before_build(tmp_path: Path, monkeypatch):
    """The explicit pre-check runs: fail it and no draft is written."""
    import recast.compose.jianying as jy

    monkeypatch.setattr(
        jy, "validate_for_jianying",
        lambda *a, **k: JianyingCheckResult(errors=["测试注入的校验失败"]),
    )
    with pytest.raises(JianyingDraftError) as excinfo:
        _export(tmp_path)
    assert "测试注入的校验失败" in str(excinfo.value)
    assert not (tmp_path / "delivery" / "jianying_draft").exists()


def test_export_stage_runs_postwrite_selfcheck(tmp_path: Path, monkeypatch):
    """The draft is written, then the service-level self-check rejects it."""
    import recast.compose.jianying as jy

    monkeypatch.setattr(
        jy, "verify_jianying_draft",
        lambda *a, **k: JianyingCheckResult(errors=["测试注入的自检失败"]),
    )
    with pytest.raises(JianyingDraftError) as excinfo:
        _export(tmp_path)
    assert "测试注入的自检失败" in str(excinfo.value)
    # The draft was written first (then rejected) — the files exist.
    assert (tmp_path / "delivery" / "jianying_draft" / "draft_content.json").is_file()


def test_script_route_checks_dialogue_alignment(tmp_path: Path):
    # dialogue.json has 1 line, SRT has 2: misaligned on the script route.
    _dialogue(tmp_path, _aligned_dialogue()[:1])
    with pytest.raises(JianyingDraftError) as excinfo:
        _export(tmp_path, entry_type="script")
    assert "台词表" in str(excinfo.value)
    assert "字幕" in str(excinfo.value)


def test_script_route_passes_when_dialogue_aligned(tmp_path: Path):
    _dialogue(tmp_path, _aligned_dialogue())
    result = _export(tmp_path, entry_type="script")
    assert "jianying" in result


def test_import_route_ignores_dialogue_file(tmp_path: Path):
    # Same misaligned dialogue.json sitting in the workdir: the recast
    # route must not pass it to validation, so the export still builds.
    _dialogue(tmp_path, _aligned_dialogue()[:1])
    result = _export(tmp_path, entry_type="import")
    assert "jianying" in result
