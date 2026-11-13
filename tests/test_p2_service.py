"""Service wiring for P2: edit dialogue, re-dub, bind costume (stubbed TTS)."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from recast.core.service import RecastService


def _service(tmp_path: Path) -> RecastService:
    svc = RecastService(db_path=str(tmp_path / "db.db"))
    svc.update_settings(output_root=str(tmp_path / "out"))
    return svc


def _sine_wav(path: Path, secs: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 24000
    n = int(secs * rate)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)))
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
        {"shot_code": "F1-L1", "dialogue": "你是谁？", "speaker": "温念",
         "start_time": 0.0, "end_time": 3.0, "duration_sec": 3.0},
        {"shot_code": "F1-L2", "dialogue": "站住。", "speaker": "陈秋微",
         "start_time": 3.0, "end_time": 6.0, "duration_sec": 3.0},
    ]
    (workdir / "shots_plan.json").write_text(
        json.dumps(shots, ensure_ascii=False), encoding="utf-8"
    )
    return workdir


def test_service_edit_and_redub_roundtrip(tmp_path: Path, monkeypatch):
    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    svc.ensure_character_folder("陈秋微")
    workdir = _workdir(tmp_path)
    calls: list = []

    def fake_synthesize(text, dest, *, voice=None, timbre=None, **kwargs):
        calls.append(text)
        _sine_wav(Path(dest))
        return {"ok": True, "path": str(dest), "source": "stub",
                "duration": 0.5, "skipped": False}

    monkeypatch.setattr("recast.generate.tts.synthesize_line", fake_synthesize)
    line = svc.edit_dialogue_line(workdir, "F1-L1", "你到底是谁？")
    assert line["dirty"] is True
    report = svc.redub_dialogue(workdir)
    assert [r["shot_code"] for r in report["redubbed"]] == ["F1-L1"]
    assert calls == ["你到底是谁？"]
    assert report["remix"]["mixed"] is not None
    assert Path(report["remix"]["mixed"]).is_file()
    assert Path(report["remix"]["srt"]).is_file()
    # Second redub with nothing dirty does nothing.
    report = svc.redub_dialogue(workdir)
    assert report["redubbed"] == []
    assert calls == ["你到底是谁？"]


def test_service_redub_with_timbre_change_only(tmp_path: Path, monkeypatch):
    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    workdir = _workdir(tmp_path)
    calls: list = []

    def fake_synthesize(text, dest, *, voice=None, timbre=None, **kwargs):
        calls.append(timbre)
        _sine_wav(Path(dest))
        return {"ok": True, "path": str(dest), "source": "stub",
                "duration": 0.5, "skipped": False}

    monkeypatch.setattr("recast.generate.tts.synthesize_line", fake_synthesize)
    svc.set_voice_timbre("温念", pitch=3.0)
    # Timbre change alone does not mark lines dirty; name the line explicitly.
    report = svc.redub_dialogue(workdir, shot_codes=["F1-L1"], remix=False)
    assert [r["shot_code"] for r in report["redubbed"]] == ["F1-L1"]
    assert calls[0] == {"pitch": 3.0, "rate": 1.0, "style": ""}


def test_service_bind_costume_texture(tmp_path: Path):
    from PIL import Image

    svc = _service(tmp_path)
    folder = svc.ensure_character_folder("温念")["folder"]
    import json as _json
    import struct as _struct

    payload = _json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = _struct.pack("<II", len(payload), 0x4E4F534A) + payload
    total = 12 + len(body)
    glb = tmp_path / "model.glb"
    glb.write_bytes(_struct.pack("<4sII", b"glTF", 2, total) + body)
    svc.import_model_glb("温念", glb)
    tex = Path(folder) / "costume" / "dress.png"
    tex.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), (30, 90, 200)).save(tex)
    report = svc.bind_costume_texture("温念", tex, "全身")
    assert report["part"] == "全身"
    assert report["applied_in_glb"] is False
    assert "Blender" in report["note"]
    from recast.assets3d.binding import list_bindings

    assert len(list_bindings(Path(folder))) == 1


def test_service_bind_needs_a_glb_first(tmp_path: Path):
    from PIL import Image

    svc = _service(tmp_path)
    folder = svc.ensure_character_folder("温念")["folder"]
    tex = Path(folder) / "costume" / "dress.png"
    tex.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), (30, 90, 200)).save(tex)
    with pytest.raises(ValueError, match="还没有可用的 GLB 模型"):
        svc.bind_costume_texture("温念", tex, "全身")
