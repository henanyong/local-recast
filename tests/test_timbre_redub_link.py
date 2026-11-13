"""Timbre change linked to re-dub (PM round 3 N7/M4).

Changing a character's voice timbre must invalidate that character's
already-dubbed lines (dirty=True in dialogue.json) and drop the stale
dialogue_mix.wav, so step 5's "mix exists means done" check flips back to
unfinished. The timbre dialog then offers a one-click re-dub of the
affected episodes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

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


def _line(code: str, speaker: str, text: str = "你好") -> dict:
    return {
        "shot_code": code,
        "speaker": speaker,
        "character": speaker,
        "text": text,
        "start_time": 0.0,
        "end_time": 3.0,
        "duration_sec": 3.0,
        "dirty": False,
    }


def _workdir(tmp_path: Path, name: str, lines: list[dict], *, with_mix: bool = True) -> Path:
    wd = tmp_path / name
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "dialogue.json").write_text(
        json.dumps(lines, ensure_ascii=False), encoding="utf-8"
    )
    if with_mix:
        (wd / "dialogue_mix.wav").write_bytes(b"fake-wav")
    return wd


def _add_drama_task(svc, name: str, workdir: Path | None, *, status: str = ST_DONE) -> dict:
    drama = svc.store.insert(
        "dramas", origin_name=name, root_path=f"/tmp/{name}", episode_count=1,
        entry_type="import",
    )
    svc.store.insert(
        "tasks", drama_id=drama["id"], episode_name="第1集",
        video_path="/tmp/ep1.mp4", status=status,
        workdir=str(workdir) if workdir else None,
    )
    return drama


def _read_lines(workdir: Path) -> list[dict]:
    return json.loads((workdir / "dialogue.json").read_text(encoding="utf-8"))


# ----- (a) 改音色后该人物行变 dirty、旧混音被删、affected 列出该集 -----


def test_timbre_change_dirties_lines_and_drops_stale_mix(svc, tmp_path: Path):
    svc.ensure_character_folder("温念")
    wd = _workdir(tmp_path, "ep1", [_line("F1-L1", "温念"), _line("F1-L2", "温念")])
    drama = _add_drama_task(svc, "剧1", wd)

    report = svc.set_voice_timbre("温念", pitch=3.0)

    assert report["affected"] == [
        {"drama_id": drama["id"], "workdir": str(wd), "lines": 2}
    ]
    assert report["stale_mix_removed"] == [str(wd)]
    assert not (wd / "dialogue_mix.wav").exists()
    assert all(line["dirty"] for line in _read_lines(wd))
    # Saving the same timbre again changes nothing: no re-invalidation.
    again = svc.set_voice_timbre("温念", pitch=3.0)
    assert again["affected"] == []
    assert again["stale_mix_removed"] == []


# ----- (b) 该人物没配过音的集不受影响 -----


def test_timbre_change_leaves_other_episodes_alone(svc, tmp_path: Path):
    svc.ensure_character_folder("温念")
    svc.ensure_character_folder("陈秋微")
    wd1 = _workdir(tmp_path, "ep1", [_line("F1-L1", "温念")])
    wd2 = _workdir(tmp_path, "ep2", [_line("F1-L1", "陈秋微")])
    wd3 = tmp_path / "ep3"
    wd3.mkdir()  # workdir without dialogue.json: skipped, no error.
    drama1 = _add_drama_task(svc, "剧1", wd1)
    _add_drama_task(svc, "剧2", wd2)
    _add_drama_task(svc, "剧3", wd3)

    report = svc.set_voice_timbre("温念", pitch=-2.0)

    assert report["affected"] == [
        {"drama_id": drama1["id"], "workdir": str(wd1), "lines": 1}
    ]
    assert report["stale_mix_removed"] == [str(wd1)]
    assert not (wd1 / "dialogue_mix.wav").exists()
    # 陈秋微's episode is untouched: mix still there, line still clean.
    assert (wd2 / "dialogue_mix.wav").is_file()
    assert _read_lines(wd2)[0]["dirty"] is False


# ----- (c) 多集里只有该人物说话的集被标 dirty -----


def test_timbre_change_only_marks_that_speakers_lines(svc, tmp_path: Path):
    svc.ensure_character_folder("温念")
    svc.ensure_character_folder("陈秋微")
    wd1 = _workdir(
        tmp_path, "ep1", [_line("F1-L1", "温念"), _line("F1-L2", "陈秋微")]
    )
    wd2 = _workdir(tmp_path, "ep2", [_line("F1-L1", "温念")])
    wd3 = _workdir(tmp_path, "ep3", [_line("F1-L1", "陈秋微")])
    d1 = _add_drama_task(svc, "剧1", wd1)
    d2 = _add_drama_task(svc, "剧2", wd2)
    _add_drama_task(svc, "剧3", wd3)

    report = svc.set_voice_timbre("温念", pitch=1.0, style="温柔")

    assert report["affected"] == [
        {"drama_id": d1["id"], "workdir": str(wd1), "lines": 1},
        {"drama_id": d2["id"], "workdir": str(wd2), "lines": 1},
    ]
    ep1 = _read_lines(wd1)
    assert ep1[0]["dirty"] is True   # 温念's line
    assert ep1[1]["dirty"] is False  # 陈秋微's line in the same episode
    assert not (wd1 / "dialogue_mix.wav").exists()
    assert not (wd2 / "dialogue_mix.wav").exists()
    assert (wd3 / "dialogue_mix.wav").is_file()
    assert _read_lines(wd3)[0]["dirty"] is False


# ----- (d) service 层：改音色后 _audio_done 的判定会翻转 -----


def test_audio_done_flips_after_timbre_change(svc, tmp_path: Path):
    from recast.desktop.main_window import MainWindow

    svc.ensure_character_folder("温念")
    wd = _workdir(tmp_path, "ep1", [_line("F1-L1", "温念")])
    _add_drama_task(svc, "剧1", wd, status=ST_DONE)
    _app()
    window = MainWindow(svc)
    window.timer.stop()
    try:
        assert window._audio_done() is True
        svc.set_voice_timbre("温念", pitch=2.0)
        assert not (wd / "dialogue_mix.wav").exists()
        assert _read_lines(wd)[0]["dirty"] is True
        assert window._audio_done() is False
    finally:
        window.close()


# ----- 对话框提示 + 一键重配 -----


def test_stale_timbre_message_wording():
    from recast.desktop.timbre_dialog import stale_timbre_message

    assert stale_timbre_message(
        [{"drama_id": "a", "workdir": "/tmp/ep1", "lines": 2},
         {"drama_id": "b", "workdir": "/tmp/ep2", "lines": 3}]
    ) == "音色已改：2 集的旧配音已失效（共 5 句待重配），第 ⑤ 步会显示未完成。"


class _StubService:
    """Minimal stand-in for RecastService: only what TimbreDialog touches."""

    def __init__(self, tmp_path: Path, affected: list[dict]):
        self.settings = SimpleNamespace(output_root=str(tmp_path / "out"))
        self.affected = affected
        self.redub_calls: list[str] = []

    def set_voice_timbre(self, character, **kwargs):
        return {
            "character": character,
            "timbre": {"pitch": 2.0, "rate": 1.0, "style": ""},
            "affected": self.affected,
            "stale_mix_removed": [a["workdir"] for a in self.affected],
        }

    def redub_dialogue(self, workdir):
        self.redub_calls.append(workdir)
        return {"redubbed": [{"shot_code": "F1-L1"}], "skipped": []}


def _dialog(stub, character: str = "温念"):
    from PySide6.QtWidgets import QDialog

    from recast.desktop.timbre_dialog import TimbreDialog

    _app()
    dlg = TimbreDialog(stub, character)
    dlg._sync_tasks = True
    return dlg, QDialog


def test_dialog_offers_one_click_redub(tmp_path: Path, monkeypatch):
    affected = [
        {"drama_id": "d1", "workdir": "/tmp/ep1", "lines": 2},
        {"drama_id": "d2", "workdir": "/tmp/ep2", "lines": 1},
    ]
    stub = _StubService(tmp_path, affected)
    dlg, QDialog = _dialog(stub)
    monkeypatch.setattr(dlg, "_ask_redub_now", lambda a: True)
    try:
        dlg._save()
        assert stub.redub_calls == ["/tmp/ep1", "/tmp/ep2"]
        assert "重配完成" in dlg.status.text()
        assert "2 句" in dlg.status.text()  # stub 每集重配 1 句，共 2 句
        # The dialog stays open so the user can read the result.
        assert dlg.result() == QDialog.Rejected  # not accepted yet
    finally:
        dlg.close()


def test_dialog_later_choice_closes_without_redub(tmp_path: Path, monkeypatch):
    affected = [{"drama_id": "d1", "workdir": "/tmp/ep1", "lines": 2}]
    stub = _StubService(tmp_path, affected)
    dlg, QDialog = _dialog(stub)
    monkeypatch.setattr(dlg, "_ask_redub_now", lambda a: False)
    try:
        dlg._save()
        assert stub.redub_calls == []
        assert dlg.result() == QDialog.Accepted
        assert "第 ⑤ 步会显示未完成" in dlg.status.text()
    finally:
        dlg.close()


def test_dialog_no_prompt_when_nothing_affected(tmp_path: Path, monkeypatch):
    stub = _StubService(tmp_path, affected=[])
    dlg, QDialog = _dialog(stub)

    def _no_prompt(affected):
        raise AssertionError("affected 为空时不应弹重配提示")

    monkeypatch.setattr(dlg, "_ask_redub_now", _no_prompt)
    try:
        dlg._save()
        assert stub.redub_calls == []
        assert dlg.result() == QDialog.Accepted
        assert "下次配音生效" in dlg.status.text()
    finally:
        dlg.close()
