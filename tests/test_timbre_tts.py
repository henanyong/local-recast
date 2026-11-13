"""Timbre actually reaches the TTS engines (stubbed, no network)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from recast.generate.tts import synthesize_line, synthesize_table


def _install_fake_edge_tts(monkeypatch, seen: dict):
    module = types.ModuleType("edge_tts")

    class Communicate:
        def __init__(self, text, voice, **kwargs):
            seen["text"] = text
            seen["voice"] = voice
            seen["kwargs"] = kwargs

        async def save(self, path):
            Path(path).write_bytes(b"edge")

    module.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)


def test_edge_tts_gets_rate_and_pitch(tmp_path: Path, monkeypatch):
    seen: dict = {}
    _install_fake_edge_tts(monkeypatch, seen)
    rec = synthesize_line(
        "你好",
        tmp_path / "line.wav",
        voice="zh-CN-XiaoyiNeural",
        timbre={"pitch": 2.0, "rate": 1.2, "style": ""},
    )
    assert rec["source"] == "edge-tts"
    assert seen["kwargs"] == {"rate": "+20%", "pitch": "+2.0st"}
    assert "语速 1.2x" in (rec.get("message") or "")
    assert "音调 +2.0 半音" in (rec.get("message") or "")
    assert rec["timbre"] == {"pitch": 2.0, "rate": 1.2, "style": ""}


def test_edge_tts_default_timbre_passes_no_kwargs(tmp_path: Path, monkeypatch):
    seen: dict = {}
    _install_fake_edge_tts(monkeypatch, seen)
    rec = synthesize_line("你好", tmp_path / "line.wav", voice="zh-CN-XiaoyiNeural")
    assert rec["source"] == "edge-tts"
    assert seen["kwargs"] == {}


def test_piper_gets_length_scale_only_when_rate_changes(tmp_path: Path, monkeypatch):
    model = tmp_path / "wen.onnx"
    model.write_bytes(b"onnx")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        Path(argv[argv.index("--output_file") + 1]).write_bytes(b"local")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("recast.generate.tts.shutil.which", lambda name: "/usr/bin/piper" if name == "piper" else None)
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("recast.generate.tts._try_edge_tts", lambda *a, **k: False)
    rec = synthesize_line("你好", tmp_path / "a.wav", voice=str(model), timbre={"rate": 2.0})
    assert rec["source"] == "piper"
    assert "--length-scale" in calls[0]
    assert calls[0][calls[0].index("--length-scale") + 1] == "0.500"
    rec = synthesize_line("你好", tmp_path / "b.wav", voice=str(model))
    assert "--length-scale" not in calls[1]


def test_clone_keeps_original_timbre_and_says_so(tmp_path: Path, monkeypatch):
    ref = tmp_path / "wen.wav"
    ref.write_bytes(b"ref")
    monkeypatch.setenv("RECAST_TTS_CLONE", "clone-bin")

    def fake_run(argv, **kwargs):
        Path(argv[argv.index("--out") + 1]).write_bytes(b"cloned")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    rec = synthesize_line("你好", tmp_path / "out.wav", voice=str(ref), timbre={"pitch": 3.0, "rate": 0.8})
    assert rec["source"] == "local-clone"
    assert "克隆音色不支持调参" in (rec.get("message") or "")


def test_pyttsx3_rate_is_set_from_timbre(tmp_path: Path, monkeypatch):
    rates: list[int] = []

    class Engine:
        def setProperty(self, name, value):
            if name == "rate":
                rates.append(int(value))

        def save_to_file(self, text, path):
            Path(path).write_bytes(b"spk")

        def runAndWait(self):
            return None

    module = types.ModuleType("pyttsx3")
    module.init = lambda: Engine()
    monkeypatch.setitem(sys.modules, "pyttsx3", module)
    monkeypatch.setattr("recast.generate.tts._PYTTSX3", None)
    monkeypatch.setattr("recast.generate.tts._try_edge_tts", lambda *a, **k: False)
    monkeypatch.setattr("recast.generate.tts._try_piper", lambda *a, **k: False)
    monkeypatch.setattr("recast.generate.tts.discover_piper_model", lambda voice=None: None)
    rec = synthesize_line("你好", tmp_path / "line.wav", timbre={"rate": 1.5})
    assert rec["source"] == "pyttsx3"
    assert rates == [300]


def test_synthesize_table_routes_timbre_per_speaker(tmp_path: Path, monkeypatch):
    seen: dict = {}
    _install_fake_edge_tts(monkeypatch, seen)
    monkeypatch.setattr("recast.generate.tts._try_piper", lambda *a, **k: False)
    monkeypatch.setattr("recast.generate.tts.discover_piper_model", lambda voice=None: None)
    shots = [
        {"shot_code": "A", "dialogue": "你好", "speaker": "温念"},
        {"shot_code": "B", "dialogue": "站住", "speaker": "陈秋微"},
    ]
    synthesize_table(
        shots,
        tmp_path,
        voices={"温念": "zh-CN-XiaoyiNeural", "陈秋微": "zh-CN-YunxiNeural"},
        timbres={"温念": {"pitch": 2.0, "rate": 1.0, "style": ""}},
    )
    # Last call is 陈秋微 (default timbre -> no kwargs); check 温念 got pitch
    # by re-running with only her line.
    seen.clear()
    synthesize_table(
        shots[:1],
        tmp_path / "one",
        voices={"温念": "zh-CN-XiaoyiNeural"},
        timbres={"温念": {"pitch": 2.0, "rate": 1.0, "style": ""}},
    )
    assert seen["kwargs"] == {"pitch": "+2.0st"}
