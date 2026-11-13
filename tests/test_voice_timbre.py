"""Character voice timbre in meta.json + service audition (stubbed TTS)."""

from __future__ import annotations

from pathlib import Path

from recast.core.characters import get_timbre, set_timbre
from recast.core.service import RecastService


def _make_service(tmp_path: Path) -> RecastService:
    svc = RecastService(db_path=str(tmp_path / "recast.db"))
    svc.update_settings(output_root=str(tmp_path / "out"))
    return svc


def test_set_timbre_clamps_and_keeps_other_fields(tmp_path: Path):
    folder = tmp_path / "wen"
    (folder / "voice").mkdir(parents=True)
    assert set_timbre(folder, pitch=99, rate=0.1) == {
        "pitch": 12.0,
        "rate": 0.5,
        "style": "",
    }
    assert set_timbre(folder, style="冷淡") == {
        "pitch": 12.0,
        "rate": 0.5,
        "style": "冷淡",
    }
    assert get_timbre({"voice": {"timbre": {"pitch": 2}}}) == {
        "pitch": 2.0,
        "rate": 1.0,
        "style": "",
    }
    assert get_timbre(None) == {"pitch": 0.0, "rate": 1.0, "style": ""}


def test_set_timbre_needs_a_character_folder(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError, match="找不到人物文件夹"):
        set_timbre(tmp_path / "ghost", pitch=1.0)


def test_service_set_voice_timbre_writes_meta(tmp_path: Path, monkeypatch):
    svc = _make_service(tmp_path)
    created = svc.ensure_character_folder("温念")
    report = svc.set_voice_timbre("温念", pitch=2.0, rate=1.2, style="温柔")
    assert report["timbre"] == {"pitch": 2.0, "rate": 1.2, "style": "温柔"}
    assert report["folder"] == created["folder"]
    from recast.core.characters import load_meta

    meta = load_meta(Path(created["folder"]))
    assert meta["voice"]["timbre"] == {"pitch": 2.0, "rate": 1.2, "style": "温柔"}
    # Second call only changes the given field.
    report = svc.set_voice_timbre("温念", pitch=-1.0)
    assert report["timbre"] == {"pitch": -1.0, "rate": 1.2, "style": "温柔"}


def test_service_audition_uses_character_voice_and_timbre(tmp_path: Path, monkeypatch):
    svc = _make_service(tmp_path)
    svc.ensure_character_folder("温念")
    svc.set_voice_timbre("温念", pitch=2.0, rate=1.5)
    svc.store.insert(
        "characters",
        id="wen",
        display_name="温念",
        name="温念",
        voice_id="zh-CN-XiaoyiNeural",
    )
    seen: dict = {}

    def fake_synthesize(text, dest, *, voice=None, timbre=None, **kwargs):
        seen.update({"text": text, "voice": voice, "timbre": timbre})
        Path(dest).write_bytes(b"fake-wav")
        return {
            "ok": True,
            "path": str(dest),
            "source": "edge-tts",
            "duration": 2.8,
            "skipped": False,
        }

    monkeypatch.setattr("recast.generate.tts.synthesize_line", fake_synthesize)
    rec = svc.audition_voice("wen")
    assert rec["source"] == "edge-tts"
    assert rec["duration"] == 2.8
    assert seen["voice"] == "zh-CN-XiaoyiNeural"
    assert seen["timbre"] == {"pitch": 2.0, "rate": 1.5, "style": ""}
    assert "温念" in seen["text"]
    assert Path(rec["path"]).is_file()
    assert Path(rec["path"]).parent.name == "voice"


def test_service_audition_without_voice_still_returns_a_record(tmp_path: Path, monkeypatch):
    svc = _make_service(tmp_path)
    svc.ensure_character_folder("路人")
    monkeypatch.setattr(
        "recast.generate.tts.synthesize_line",
        lambda text, dest, **kwargs: {
            "ok": True,
            "path": None,
            "source": "skip",
            "duration": 0.0,
            "skipped": True,
            "message": "已跳过配音。",
        },
    )
    rec = svc.audition_voice("路人", text="试一句")
    assert rec["skipped"] is True
    assert rec["text"] == "试一句"
