"""N19: audition is a read-only preview (PM round 4).

Dragging the sliders and pressing [试听] must NOT save the timbre, must
not mark dubbed lines dirty and must not delete dialogue_mix.wav — the
old dub stays valid and step 5 stays done. Only [保存] writes meta.json
(via set_voice_timbre) and invalidates stale dubs. [取消] writes nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from recast.core.constants import ST_DONE
from recast.core.service import RecastService


@pytest.fixture()
def svc(tmp_path: Path):
    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _fake_synthesize(seen: dict):
    def fake(text, dest, *, voice=None, timbre=None, **kwargs):
        seen.update({"text": text, "voice": voice, "timbre": timbre})
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fake-wav")
        return {
            "ok": True,
            "path": str(dest),
            "source": "edge-tts",
            "duration": 2.8,
            "skipped": False,
        }

    return fake


def _meta_path(svc, character: str) -> Path:
    from recast.core.characters import character_folder

    return character_folder(svc.settings.output_root, character) / "meta.json"


def _line(code: str, speaker: str) -> dict:
    return {
        "shot_code": code,
        "speaker": speaker,
        "character": speaker,
        "text": "你好",
        "start_time": 0.0,
        "end_time": 3.0,
        "duration_sec": 3.0,
        "dirty": False,
    }


def _workdir_with_dub(tmp_path: Path, svc, name: str = "ep1") -> Path:
    """A finished episode workdir: dialogue.json (clean) + dialogue_mix.wav."""
    wd = tmp_path / name
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "dialogue.json").write_text(
        json.dumps([_line("F1-L1", "温念")], ensure_ascii=False), encoding="utf-8"
    )
    (wd / "dialogue_mix.wav").write_bytes(b"fake-wav")
    drama = svc.store.insert(
        "dramas", origin_name="剧1", root_path="/tmp/剧1", episode_count=1,
        entry_type="import",
    )
    svc.store.insert(
        "tasks", drama_id=drama["id"], episode_name="第1集",
        video_path="/tmp/ep1.mp4", status=ST_DONE, workdir=str(wd),
    )
    return wd


# ----- (1) service: timbre override is used, meta.json + mix untouched -----


def test_audition_with_timbre_override_uses_preview_not_saved(
    svc, tmp_path: Path, monkeypatch
):
    svc.ensure_character_folder("温念")
    svc.set_voice_timbre("温念", pitch=2.0, rate=1.5, style="温柔")
    wd = _workdir_with_dub(tmp_path, svc)
    meta_before = _meta_path(svc, "温念").read_bytes()
    seen: dict = {}
    monkeypatch.setattr(
        "recast.generate.tts.synthesize_line", _fake_synthesize(seen)
    )

    rec = svc.audition_voice(
        "温念", timbre={"pitch": -3.0, "rate": 0.8, "style": "冷淡"}
    )

    assert rec["timbre"] == {"pitch": -3.0, "rate": 0.8, "style": "冷淡"}
    assert seen["timbre"] == {"pitch": -3.0, "rate": 0.8, "style": "冷淡"}
    # Nothing was written: saved timbre, dubbed lines and mix all intact.
    assert _meta_path(svc, "温念").read_bytes() == meta_before
    assert (wd / "dialogue_mix.wav").is_file()
    lines = json.loads((wd / "dialogue.json").read_text(encoding="utf-8"))
    assert lines[0]["dirty"] is False


# ----- (2) service: no override -> old behaviour, stored timbre is used -----


def test_audition_without_override_still_uses_saved_timbre(
    svc, tmp_path: Path, monkeypatch
):
    svc.ensure_character_folder("温念")
    svc.set_voice_timbre("温念", pitch=2.0, rate=1.5, style="温柔")
    seen: dict = {}
    monkeypatch.setattr(
        "recast.generate.tts.synthesize_line", _fake_synthesize(seen)
    )

    rec = svc.audition_voice("温念")
    assert rec["timbre"] == {"pitch": 2.0, "rate": 1.5, "style": "温柔"}
    assert seen["timbre"] == {"pitch": 2.0, "rate": 1.5, "style": "温柔"}
    # Explicit timbre=None is the same as omitting it.
    rec2 = svc.audition_voice("温念", timbre=None)
    assert rec2["timbre"] == {"pitch": 2.0, "rate": 1.5, "style": "温柔"}


# ----- (3) boundary: override values are clamped, unknown folder still errors -----


def test_audition_override_clamps_out_of_range_values(
    svc, tmp_path: Path, monkeypatch
):
    svc.ensure_character_folder("温念")
    svc.set_voice_timbre("温念", pitch=1.0)
    seen: dict = {}
    monkeypatch.setattr(
        "recast.generate.tts.synthesize_line", _fake_synthesize(seen)
    )

    rec = svc.audition_voice("温念", timbre={"pitch": 99, "rate": 0.1, "style": ""})
    assert rec["timbre"] == {"pitch": 12.0, "rate": 0.5, "style": ""}
    assert seen["timbre"] == {"pitch": 12.0, "rate": 0.5, "style": ""}


def test_audition_override_still_needs_character_folder(svc, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="找不到人物文件夹"):
        svc.audition_voice("不存在的人", timbre={"pitch": 1.0})


# ----- (4) dialog: audition + cancel change nothing; [保存] still saves -----


def test_dialog_audition_then_cancel_writes_nothing(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from recast.desktop.timbre_dialog import TimbreDialog

    svc.ensure_character_folder("温念")
    svc.set_voice_timbre("温念", pitch=1.0, rate=1.0, style="")
    wd = _workdir_with_dub(tmp_path, svc)
    meta_before = _meta_path(svc, "温念").read_bytes()
    seen: dict = {}
    monkeypatch.setattr(
        "recast.generate.tts.synthesize_line", _fake_synthesize(seen)
    )
    # Guard: _audition must never call set_voice_timbre (old N19 behaviour).
    calls: list[dict] = []
    real_set = svc.set_voice_timbre

    def recording_set(character, **kwargs):
        calls.append({"character": character, **kwargs})
        return real_set(character, **kwargs)

    svc.set_voice_timbre = recording_set  # type: ignore[method-assign]

    _app()
    dlg = TimbreDialog(svc, "温念")
    dlg._sync_tasks = True
    try:
        # Drag the sliders away from the saved timbre, then audition.
        dlg.pitch_slider.setValue(-6)   # -3.0 semitones
        dlg.rate_slider.setValue(16)    # 0.8x
        dlg.style_edit.setText("冷淡")
        dlg._audition()

        assert "试听已生成" in dlg.status.text()
        assert seen["timbre"] == {"pitch": -3.0, "rate": 0.8, "style": "冷淡"}
        assert calls == []  # no save happened during audition
        assert _meta_path(svc, "温念").read_bytes() == meta_before
        assert (wd / "dialogue_mix.wav").is_file()
        lines = json.loads((wd / "dialogue.json").read_text(encoding="utf-8"))
        assert lines[0]["dirty"] is False

        # Cancel: still nothing written.
        dlg.reject()
        assert dlg.result() == QDialog.Rejected
        assert _meta_path(svc, "温念").read_bytes() == meta_before
        assert (wd / "dialogue_mix.wav").is_file()
        assert calls == []
    finally:
        dlg.close()


def test_dialog_save_still_writes_and_invalidates(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from recast.desktop.timbre_dialog import TimbreDialog

    svc.ensure_character_folder("温念")
    svc.set_voice_timbre("温念", pitch=1.0, rate=1.0, style="")
    wd = _workdir_with_dub(tmp_path, svc)
    _app()
    dlg = TimbreDialog(svc, "温念")
    dlg._sync_tasks = True
    monkeypatch.setattr(dlg, "_ask_redub_now", lambda a: False)
    try:
        dlg.pitch_slider.setValue(6)  # +3.0 semitones, different from saved 1.0
        dlg._save()
        assert dlg.result() == QDialog.Accepted
        assert "第 ⑤ 步会显示未完成" in dlg.status.text()
        # Save really wrote meta.json and invalidated the old dub.
        meta = json.loads(_meta_path(svc, "温念").read_text(encoding="utf-8"))
        assert meta["voice"]["timbre"]["pitch"] == 3.0
        assert not (wd / "dialogue_mix.wav").exists()
        lines = json.loads((wd / "dialogue.json").read_text(encoding="utf-8"))
        assert lines[0]["dirty"] is True
    finally:
        dlg.close()
