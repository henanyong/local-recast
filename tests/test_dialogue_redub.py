"""Dialogue editing + selective re-dub (stubbed TTS, real ffmpeg mix)."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from recast.core.characters import character_folder, ensure_character_folder, load_meta
from recast.generate import dialogue


def _sine_wav(path: Path, freq: float = 440.0, secs: float = 0.5, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(secs * rate)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(n)
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return path


def _workdir(tmp_path: Path) -> Path:
    workdir = tmp_path / "ep1"
    workdir.mkdir(parents=True, exist_ok=True)
    shots = [
        {
            "shot_code": "F1-L1",
            "dialogue": "你是谁？",
            "speaker": "温念",
            "start_time": 0.0,
            "end_time": 3.0,
            "duration_sec": 3.0,
        },
        {
            "shot_code": "F1-L2",
            "dialogue": "站住，别动。",
            "speaker": "陈秋微",
            "start_time": 3.0,
            "end_time": 6.0,
            "duration_sec": 3.0,
        },
    ]
    (workdir / "shots_plan.json").write_text(
        json.dumps(shots, ensure_ascii=False), encoding="utf-8"
    )
    return workdir


def _stub_synthesize(calls: list):
    def synth(text, dest, *, voice=None, timbre=None, **kwargs):
        calls.append({"text": text, "voice": voice, "timbre": timbre})
        _sine_wav(Path(dest), secs=0.5)
        return {
            "ok": True,
            "path": str(dest),
            "source": "stub",
            "duration": 0.5,
            "skipped": False,
        }

    return synth


def test_load_builds_dialogue_from_shots_plan(tmp_path: Path):
    workdir = _workdir(tmp_path)
    lines = dialogue.load_dialogue(workdir)
    assert [line["shot_code"] for line in lines] == ["F1-L1", "F1-L2"]
    assert lines[0]["speaker"] == "温念"
    assert lines[0]["text"] == "你是谁？"
    assert lines[0]["dirty"] is False
    assert (workdir / "dialogue.json").is_file()
    # Second load reads the saved file, not the plan.
    assert dialogue.load_dialogue(workdir) == lines


def test_load_needs_a_plan_or_dialogue(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="找不到对白"):
        dialogue.load_dialogue(tmp_path / "empty")


def test_edit_line_marks_dirty_and_rejects_empty(tmp_path: Path):
    workdir = _workdir(tmp_path)
    line = dialogue.edit_line(workdir, "F1-L1", "你到底是谁？")
    assert line["text"] == "你到底是谁？"
    assert line["dirty"] is True
    assert dialogue.load_dialogue(workdir)[0]["dirty"] is True
    assert dialogue.load_dialogue(workdir)[1]["dirty"] is False
    with pytest.raises(ValueError, match="对白文本不能为空"):
        dialogue.edit_line(workdir, "F1-L1", "   ")
    with pytest.raises(FileNotFoundError, match="没有 F9-L9"):
        dialogue.edit_line(workdir, "F9-L9", "你好")


def test_redub_only_dirty_lines_and_registers_in_meta(tmp_path: Path):
    workdir = _workdir(tmp_path)
    out = tmp_path / "out"
    ensure_character_folder(out, "温念")
    ensure_character_folder(out, "陈秋微")
    dialogue.edit_line(workdir, "F1-L1", "你到底是谁？")
    calls: list = []
    report = dialogue.redub_lines(
        workdir,
        output_root=out,
        store=None,
        synthesize=_stub_synthesize(calls),
        remix=False,
    )
    assert [r["shot_code"] for r in report["redubbed"]] == ["F1-L1"]
    assert report["skipped"] == []
    assert len(calls) == 1
    assert calls[0]["text"] == "你到底是谁？"
    lines = dialogue.load_dialogue(workdir)
    assert lines[0]["dirty"] is False
    assert lines[0]["audio_duration"] == 0.5
    # as_posix(): on Windows the stored path uses backslashes, so compare the
    # forward-slash form instead of a forward-slash literal.
    assert Path(lines[0]["audio_path"]).as_posix().endswith("voice/lines/F1-L1.wav")
    assert Path(lines[0]["audio_path"]).is_file()
    meta = load_meta(character_folder(out, "温念"))
    assert meta is not None
    assert Path(meta["voice"]["dubbed_lines"]["F1-L1"]).as_posix().endswith(
        "voice/lines/F1-L1.wav"
    )
    # 陈秋微's line was untouched: no wav, no meta entry.
    assert not (out / "陈秋微" / "voice" / "lines").exists()
    # Nothing dirty any more: a second redub does nothing.
    report = dialogue.redub_lines(
        workdir, output_root=out, store=None, synthesize=_stub_synthesize(calls), remix=False
    )
    assert report["redubbed"] == []
    assert len(calls) == 1


def test_redub_uses_character_timbre(tmp_path: Path):
    workdir = _workdir(tmp_path)
    out = tmp_path / "out"
    folder, _ = ensure_character_folder(out, "温念")
    from recast.core.characters import set_timbre

    set_timbre(folder, pitch=3.0, rate=1.2)
    dialogue.edit_line(workdir, "F1-L1", "你到底是谁？")
    calls: list = []
    dialogue.redub_lines(
        workdir, output_root=out, store=None, synthesize=_stub_synthesize(calls), remix=False
    )
    assert calls[0]["timbre"] == {"pitch": 3.0, "rate": 1.2, "style": ""}


def test_redub_with_explicit_shot_codes_ignores_dirty_flag(tmp_path: Path):
    workdir = _workdir(tmp_path)
    out = tmp_path / "out"
    ensure_character_folder(out, "温念")
    ensure_character_folder(out, "陈秋微")
    calls: list = []
    report = dialogue.redub_lines(
        workdir,
        shot_codes=["F1-L2"],
        output_root=out,
        store=None,
        synthesize=_stub_synthesize(calls),
        remix=False,
    )
    assert [r["shot_code"] for r in report["redubbed"]] == ["F1-L2"]
    assert calls[0]["text"] == "站住，别动。"


def test_remix_writes_mix_and_srt(tmp_path: Path):
    workdir = _workdir(tmp_path)
    out = tmp_path / "out"
    ensure_character_folder(out, "温念")
    ensure_character_folder(out, "陈秋微")
    calls: list = []
    dialogue.redub_lines(
        workdir, output_root=out, store=None, synthesize=_stub_synthesize(calls), remix=False
    )
    dialogue.edit_line(workdir, "F1-L2", "站住，别动。")
    dialogue.redub_lines(
        workdir, output_root=out, store=None, synthesize=_stub_synthesize(calls), remix=False
    )
    result = dialogue.remix_dialogue(workdir)
    assert result["mixed"] is not None
    assert Path(result["mixed"]).is_file()
    assert Path(result["mixed"]).stat().st_size > 0
    assert Path(result["srt"]).is_file()
    srt = Path(result["srt"]).read_text(encoding="utf-8")
    assert "你是谁？" in srt
    assert "站住，别动。" in srt


def test_remix_remuxes_existing_output(tmp_path: Path):
    import subprocess

    workdir = _workdir(tmp_path)
    out = tmp_path / "out"
    ensure_character_folder(out, "温念")
    calls: list = []
    dialogue.edit_line(workdir, "F1-L1", "你到底是谁？")
    dialogue.redub_lines(
        workdir, output_root=out, store=None, synthesize=_stub_synthesize(calls), remix=False
    )
    mp4 = workdir / "output.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=6:size=64x64:rate=10",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        check=True, capture_output=True,
    )
    result = dialogue.remix_dialogue(workdir)
    assert result["remuxed"] == str(mp4)
    assert result["remux_error"] is None
    assert mp4.is_file() and mp4.stat().st_size > 0
