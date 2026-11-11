"""Real audio duration, per-character voice, and an offline TTS path."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from recast.generate.stitch import stitch_and_dub
from recast.generate.tts import synthesize_line, synthesize_table
from recast.templates.duration import extend_timeline_for_audio


def test_real_audio_extends_the_timeline_and_does_not_overlap():
    shots = [
        {
            "shot_code": "A",
            "dialogue": "甲，乙",
            "start_time": 0,
            "end_time": 2,
            "duration_sec": 2,
            "audio_duration": 5,
        },
        {
            "shot_code": "B",
            "dialogue": "丙",
            "start_time": 2,
            "end_time": 4,
            "duration_sec": 2,
            "audio_duration": 1,
        },
    ]
    notes = extend_timeline_for_audio(shots)
    assert shots[0]["end_time"] == 5
    assert shots[0]["duration_sec"] == 5
    assert shots[1]["start_time"] == 5
    assert shots[1]["end_time"] == 7
    assert notes[0]["needs_confirm"] is True
    assert notes[0]["action"] == "extend"
    assert "没有加速" in notes[0]["message"]
    assert "atempo" not in notes[0]["message"].lower()
    assert notes[0]["split_suggestion"]
    assert "甲，" in notes[0]["split_suggestion"]
    # A second pass sees the widened slot and does not shift again.
    assert extend_timeline_for_audio(shots) == []


def test_audio_that_fits_is_left_alone():
    shots = [
        {"shot_code": "A", "dialogue": "短", "start_time": 0, "end_time": 4, "audio_duration": 1.2},
    ]
    assert extend_timeline_for_audio(shots) == []
    assert shots[0]["end_time"] == 4


def test_stitch_surfaces_the_confirmation(tmp_path: Path, monkeypatch):
    silent = tmp_path / "silent.mp4"
    silent.write_bytes(b"picture")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"audio")
    shots = [
        {
            "shot_code": "A",
            "dialogue": "甲，乙",
            "start_time": 0,
            "end_time": 2,
            "duration_sec": 2,
            "audio_duration": 5,
            "audio_path": str(wav),
        }
    ]

    def fake_ffmpeg(args):
        Path(args[-1]).write_bytes(b"out")

    monkeypatch.setattr("recast.generate.stitch.run_ffmpeg", fake_ffmpeg)
    result = stitch_and_dub(shots, tmp_path, silent=silent, precheck_dialogue=False)
    assert result["confirmations"][0]["needs_confirm"] is True
    assert shots[0]["end_time"] == 5


def test_character_voice_is_routed(tmp_path: Path, monkeypatch):
    seen: list[str | None] = []

    def fake_edge(text, dest, voice=None):
        seen.append(voice)
        Path(dest).write_bytes(b"edge")
        return True

    monkeypatch.setattr("recast.generate.tts._try_edge_tts", fake_edge)
    monkeypatch.setattr("recast.generate.tts._try_piper", lambda *args, **kwargs: False)
    monkeypatch.setattr("recast.generate.tts.discover_piper_model", lambda voice=None: None)
    shots = [
        {"shot_code": "A", "dialogue": "你好", "speaker": "温念"},
        {"shot_code": "B", "dialogue": "站住", "speaker": "陈秋微", "voice": "zh-CN-YunxiNeural"},
    ]
    synthesize_table(
        shots,
        tmp_path,
        voices={"温念": "zh-CN-XiaoyiNeural", "陈秋微": "ignored-because-shot-has-voice"},
    )
    assert seen == ["zh-CN-XiaoyiNeural", "zh-CN-YunxiNeural"]


def test_piper_model_is_used_offline(tmp_path: Path, monkeypatch):
    model = tmp_path / "wen.onnx"
    model.write_bytes(b"onnx")
    dest = tmp_path / "line.wav"

    def fake_run(argv, **kwargs):
        assert "--model" in argv
        assert str(model) in argv
        Path(argv[argv.index("--output_file") + 1]).write_bytes(b"local")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("recast.generate.tts.shutil.which", lambda name: "/usr/bin/piper" if name == "piper" else None)
    monkeypatch.setattr("subprocess.run", fake_run)
    called = {"edge": 0}

    def boom(*args, **kwargs):
        called["edge"] += 1
        return False

    monkeypatch.setattr("recast.generate.tts._try_edge_tts", boom)
    rec = synthesize_line("你好", dest, voice=str(model))
    assert rec["source"] == "piper"
    assert rec["skipped"] is False
    assert called["edge"] == 0
    assert dest.read_bytes() == b"local"


def test_reference_wav_uses_local_clone_command(tmp_path: Path, monkeypatch):
    ref = tmp_path / "wen.wav"
    ref.write_bytes(b"ref")
    dest = tmp_path / "out.wav"
    monkeypatch.setenv("RECAST_TTS_CLONE", "clone-bin")

    def fake_run(argv, **kwargs):
        assert argv[:1] == ["clone-bin"]
        assert "--ref" in argv and str(ref) in argv
        assert "--text" in argv and "你好" in argv
        Path(argv[argv.index("--out") + 1]).write_bytes(b"cloned")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    rec = synthesize_line("你好", dest, voice=str(ref))
    assert rec["source"] == "local-clone"
    assert dest.read_bytes() == b"cloned"


def test_pyttsx3_engine_is_created_once(tmp_path: Path, monkeypatch):
    inits = {"n": 0}

    class Engine:
        def save_to_file(self, text, path):
            Path(path).write_bytes(b"spk")

        def runAndWait(self):
            return None

    module = types.ModuleType("pyttsx3")

    def init():
        inits["n"] += 1
        return Engine()

    module.init = init
    monkeypatch.setitem(sys.modules, "pyttsx3", module)
    monkeypatch.setattr("recast.generate.tts._PYTTSX3", None)
    monkeypatch.setattr("recast.generate.tts._try_edge_tts", lambda *args, **kwargs: False)
    monkeypatch.setattr("recast.generate.tts._try_piper", lambda *args, **kwargs: False)
    monkeypatch.setattr("recast.generate.tts.discover_piper_model", lambda voice=None: None)
    synthesize_line("甲", tmp_path / "a.wav")
    synthesize_line("乙", tmp_path / "b.wav")
    assert inits["n"] == 1


def test_imported_wav_is_copied_without_reading_it_all(tmp_path: Path, monkeypatch):
    src = tmp_path / "line.wav"
    src.write_bytes(b"big-wav")
    seen: list[tuple[str, str]] = []
    real = __import__("shutil").copyfile

    def spy(source, dest):
        seen.append((str(source), str(dest)))
        return real(source, dest)

    monkeypatch.setattr("recast.generate.tts.shutil.copyfile", spy)
    rec = synthesize_line("不用念", tmp_path / "out.wav", wav_source=src)
    assert rec["source"] == "import"
    assert seen
    assert Path(seen[0][1]).read_bytes() == b"big-wav"
