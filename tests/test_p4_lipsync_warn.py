"""P4 唇同步警告 4.3①：合成前拦截（时长检查，不是口型对齐）。

覆盖三层，每项都有正向 + 边界/失败：

* 纯函数 ``check_lipsync_warnings``（recast/quality/review.py）；
* 合成前关口 ``gate_compose_on_lipsync``（recast/core/runner.py）；
* 对话框 ``lipsync_warn_dialog``（recast/desktop/lipsync_warn_dialog.py，
  用假框，不 exec 真模态框）；
* ``run_episode`` 合成阶段集成（假 compose.run，不跑真 ffmpeg）与
  ``dispatch_job`` 经 settings 读回调的转发。

阈值规则（见 check_lipsync_warnings docstring）：配音时长 > 镜头时长
+ 0.05 秒容差。0.05 秒与仓库 ``extend_timeline_for_audio`` 的忽略阈值
一致；不是 >4 字/秒那套文字预估。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from recast.quality.review import (
    LIPSYNC_TOLERANCE_SEC,
    check_lipsync_warnings,
)
from recast.runtime import CancelledError


def _line(code, text, audio, slot, speaker="温念"):
    return {
        "shot_code": code,
        "text": text,
        "speaker": speaker,
        "audio_duration": audio,
        "duration_sec": slot,
    }


# ----- 纯函数：check_lipsync_warnings -----


def test_overrun_detects_correct_row_and_amount():
    lines = [
        _line("F1-L1", "你干什么！", 2.0, 2.5),
        _line("F2-L2", "我没有推她，是她自己摔的", 3.2, 2.5),
        _line("F2-L3", "走开", 1.0, 2.0),
    ]
    warnings = check_lipsync_warnings(lines)
    assert len(warnings) == 1
    w = warnings[0]
    assert w.shot_code == "F2-L2"
    assert w.text == "我没有推她，是她自己摔的"
    assert w.audio_sec == 3.2
    assert w.slot_sec == 2.5
    assert w.over_sec == pytest.approx(0.7)
    assert "F2-L2" in w.message and "超出 0.70 秒" in w.message


def test_within_threshold_passes_without_warning():
    lines = [
        _line("F1-L1", "你干什么！", 2.5, 2.5),  # 配音与镜头一样长
        _line("F1-L2", "走开", 2.0, 2.5),  # 配音更短
        _line("F1-L3", "别碰我", 2.54, 2.5),  # 超 0.04 秒，在容差内
    ]
    assert check_lipsync_warnings(lines) == []


def test_just_over_tolerance_warns():
    assert check_lipsync_warnings([_line("F1-L1", "别碰我", 2.56, 2.5)]) != []
    assert check_lipsync_warnings([_line("F1-L1", "别碰我", 2.5, 2.5)]) == []


def test_custom_tolerance_is_respected():
    line = _line("F1-L1", "别碰我", 2.8, 2.5)
    assert check_lipsync_warnings([line], tolerance_sec=0.5) == []
    assert len(check_lipsync_warnings([line], tolerance_sec=0.2)) == 1


def test_undubbed_lines_are_skipped_not_flagged():
    lines = [
        {"shot_code": "F1-L1", "text": "还没配音", "duration_sec": 1.0},
        _line("F1-L2", "配了但时长为 0", 0, 1.0),
        _line("F1-L3", "正常行", 0.9, 1.0),
    ]
    assert check_lipsync_warnings(lines) == []


def test_slot_falls_back_to_start_end():
    line = {
        "shot_code": "F1-L1",
        "text": "没有 duration_sec，只有起止",
        "audio_duration": 4.0,
        "start_time": 10.0,
        "end_time": 13.0,
    }
    warnings = check_lipsync_warnings([line])
    assert len(warnings) == 1
    assert warnings[0].slot_sec == 3.0
    assert warnings[0].over_sec == pytest.approx(1.0)


def test_zero_slot_with_audio_is_flagged():
    line = {"shot_code": "F1-L1", "text": "没有镜头时长", "audio_duration": 1.2}
    warnings = check_lipsync_warnings([line])
    assert len(warnings) == 1
    assert warnings[0].slot_sec == 0.0
    assert warnings[0].over_sec == pytest.approx(1.2)


def test_multiple_overruns_keep_order_and_as_dict():
    lines = [
        _line("F1-L1", "第一句超", 3.0, 2.0),
        _line("F1-L2", "第二句不超", 1.0, 2.0),
        _line("F1-L3", "第三句超", 5.0, 2.0),
    ]
    warnings = check_lipsync_warnings(lines)
    assert [w.shot_code for w in warnings] == ["F1-L1", "F1-L3"]
    d = warnings[0].as_dict()
    assert d["shot_code"] == "F1-L1"
    assert d["audio_sec"] == 3.0 and d["slot_sec"] == 2.0
    assert d["over_sec"] == pytest.approx(1.0)
    assert d["speaker"] == "温念"
    assert LIPSYNC_TOLERANCE_SEC == 0.05


# ----- 关口：gate_compose_on_lipsync -----


def _write_dialogue(workdir: Path, lines) -> None:
    (workdir / "dialogue.json").write_text(
        json.dumps(lines, ensure_ascii=False), encoding="utf-8"
    )


def test_gate_no_dialogue_file_never_asks(tmp_path: Path):
    from recast.core.runner import gate_compose_on_lipsync

    called = []
    result = gate_compose_on_lipsync(tmp_path, lipsync_confirm=called.append)
    assert result == []
    assert called == []


def test_gate_no_warnings_never_asks(tmp_path: Path):
    from recast.core.runner import gate_compose_on_lipsync

    _write_dialogue(tmp_path, [_line("F1-L1", "正常", 1.0, 2.0)])
    called = []
    result = gate_compose_on_lipsync(tmp_path, lipsync_confirm=called.append)
    assert result == []
    assert called == []


def test_gate_continue_returns_warnings_as_dicts(tmp_path: Path):
    from recast.core.runner import gate_compose_on_lipsync

    _write_dialogue(tmp_path, [_line("F2-L2", "我没有推她", 3.2, 2.5)])
    seen = []
    result = gate_compose_on_lipsync(
        tmp_path, lipsync_confirm=lambda w: seen.extend(w) or True
    )
    assert len(result) == 1
    assert result[0]["shot_code"] == "F2-L2"
    assert result[0]["over_sec"] == pytest.approx(0.7)
    assert len(seen) == 1  # 回调收到的是 LipSyncWarning 对象


def test_gate_cancel_raises_cancelled(tmp_path: Path):
    from recast.core.runner import gate_compose_on_lipsync

    _write_dialogue(tmp_path, [_line("F2-L2", "我没有推她", 3.2, 2.5)])
    with pytest.raises(CancelledError) as exc:
        gate_compose_on_lipsync(tmp_path, lipsync_confirm=lambda w: False)
    assert "F2-L2" in str(exc.value)
    assert "取消" in str(exc.value)


def test_gate_no_callback_proceeds_and_returns_warnings(tmp_path: Path):
    from recast.core.runner import gate_compose_on_lipsync

    _write_dialogue(tmp_path, [_line("F2-L2", "我没有推她", 3.2, 2.5)])
    result = gate_compose_on_lipsync(tmp_path, lipsync_confirm=None)
    assert len(result) == 1
    assert result[0]["shot_code"] == "F2-L2"


def test_gate_corrupt_dialogue_file_does_not_block(tmp_path: Path):
    from recast.core.runner import gate_compose_on_lipsync

    (tmp_path / "dialogue.json").write_text("{不是 json", encoding="utf-8")
    called = []
    assert gate_compose_on_lipsync(tmp_path, lipsync_confirm=called.append) == []
    assert called == []


def test_gate_non_list_dialogue_file_does_not_block(tmp_path: Path):
    from recast.core.runner import gate_compose_on_lipsync

    (tmp_path / "dialogue.json").write_text('{"a": 1}', encoding="utf-8")
    assert gate_compose_on_lipsync(tmp_path) == []


# ----- 对话框：lipsync_warn_dialog（假框，不 exec 真模态框）-----


class _FakeBox:
    """test_main_window_p1._FakeBox 的同款替身：只加 setDefaultButton。"""

    from PySide6.QtWidgets import QMessageBox as _RealBox

    ButtonRole = _RealBox.ButtonRole
    shown: list = []
    click: str | None = None

    def __init__(self, parent=None):
        self.parent = parent
        self._buttons: dict = {}
        self.exec_called = False
        _FakeBox.shown.append(self)

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def setDefaultButton(self, button):
        self.default_button = button

    def addButton(self, text, role):
        btn = object()
        self._buttons[text] = btn
        return btn

    def exec(self):
        self.exec_called = True
        return 1

    def clickedButton(self):
        if type(self).click is None:
            return None
        return self._buttons.get(type(self).click)


@pytest.fixture(autouse=True)
def _reset_fake_box():
    _FakeBox.shown.clear()
    _FakeBox.click = None
    yield
    _FakeBox.shown.clear()
    _FakeBox.click = None


def _warning_dicts():
    from recast.quality.review import LipSyncWarning

    return [
        LipSyncWarning(
            shot_code="F2-L2",
            text="我没有推她，是她自己摔的",
            speaker="温念",
            audio_sec=3.2,
            slot_sec=2.5,
            over_sec=0.7,
        )
    ]


def test_dialog_lists_overruns_and_continue_returns_true():
    from recast.desktop.lipsync_warn_dialog import lipsync_warn_dialog

    _FakeBox.click = "继续合成"
    assert lipsync_warn_dialog(None, _warning_dicts(), _box_cls=_FakeBox) is True
    assert len(_FakeBox.shown) == 1
    box = _FakeBox.shown[0]
    assert box.exec_called
    assert "F2-L2" in box.text
    assert "超出 0.70 秒" in box.text
    assert "我没有推她" in box.text
    assert "口型" in box.text  # 明确这不是口型对齐检查


def test_dialog_cancel_returns_false():
    from recast.desktop.lipsync_warn_dialog import lipsync_warn_dialog

    _FakeBox.click = "取消，回去改台词"
    assert lipsync_warn_dialog(None, _warning_dicts(), _box_cls=_FakeBox) is False


def test_dialog_no_warnings_returns_true_without_box():
    from recast.desktop.lipsync_warn_dialog import lipsync_warn_dialog

    assert lipsync_warn_dialog(None, [], _box_cls=_FakeBox) is True
    assert _FakeBox.shown == []


def test_dialog_truncates_long_lists():
    from recast.desktop.lipsync_warn_dialog import format_warning_lines

    warnings = [
        {
            "shot_code": f"F1-L{i}",
            "text": "超长台词",
            "audio_sec": 3.0,
            "slot_sec": 2.0,
            "over_sec": 1.0,
        }
        for i in range(10)
    ]
    text = format_warning_lines(warnings)
    assert "F1-L7" in text
    assert "F1-L8" not in text
    assert "等 10 行超标" in text


def test_ask_from_worker_thread_marshals_to_gui(monkeypatch):
    """工作线程调 ask_lipsync_on_gui_thread：框在 GUI 线程弹，等到回答才回。

    用假框 + processEvents 泵事件循环；线程 10 秒内不回来即失败，
    不让一次死锁挂住整个测试进程。
    """
    from PySide6.QtWidgets import QApplication

    import recast.desktop.lipsync_warn_dialog as dlg_mod

    monkeypatch.setattr(dlg_mod, "QMessageBox", _FakeBox)
    _FakeBox.click = "继续合成"
    app = QApplication.instance() or QApplication([])
    warnings = _warning_dicts()
    outcome: dict = {}

    def _ask_from_worker():
        outcome["ok"] = dlg_mod.ask_lipsync_on_gui_thread(None, warnings)

    worker = threading.Thread(target=_ask_from_worker, daemon=True)
    worker.start()
    deadline = time.monotonic() + 10
    while worker.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not worker.is_alive(), "工作线程等 GUI 回答时死锁了"
    assert outcome.get("ok") is True
    assert len(_FakeBox.shown) == 1


# ----- run_episode 集成：合成阶段 -----


def _seed_compose_workdir(workdir: Path, video: Path) -> None:
    """让 resolve_plan 只留 compose：extract 产物（meta.json + 字幕）已齐。"""
    (workdir / "meta.json").write_text(
        json.dumps(
            {
                "path": str(video),
                "width": 1280,
                "height": 720,
                "fps": 25.0,
                "duration": 10.0,
                "frame_count": 250,
                "has_audio": True,
                "video_codec": "h264",
                "audio_codec": "aac",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (workdir / "subtitles.srt").write_text("", encoding="utf-8")


def _fake_video(tmp_path: Path) -> Path:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"")
    return video


def test_run_episode_cancel_does_not_compose(tmp_path: Path, monkeypatch):
    from recast.core.runner import run_episode

    video = _fake_video(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _seed_compose_workdir(workdir, video)
    _write_dialogue(workdir, [_line("F2-L2", "我没有推她", 3.2, 2.5)])
    calls = []
    monkeypatch.setattr(
        "recast.stages.compose.run",
        lambda *a, **k: calls.append((a, k)) or str(workdir / "output.mp4"),
    )
    with pytest.raises(CancelledError):
        run_episode(
            video,
            workdir=workdir,
            stages=["compose"],
            lipsync_confirm=lambda w: False,
        )
    assert calls == []  # 取消：compose.run 一次也没跑
    assert not (workdir / "output.mp4").is_file()


def test_run_episode_continue_composes_and_records_summary(tmp_path: Path, monkeypatch):
    from recast.core.runner import run_episode

    video = _fake_video(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _seed_compose_workdir(workdir, video)
    _write_dialogue(
        workdir,
        [
            _line("F2-L2", "我没有推她", 3.2, 2.5),
            _line("F2-L3", "正常行", 1.0, 2.0),
        ],
    )
    calls = []
    monkeypatch.setattr(
        "recast.stages.compose.run",
        lambda *a, **k: calls.append((a, k)) or str(workdir / "output.mp4"),
    )
    summary = run_episode(
        video,
        workdir=workdir,
        stages=["compose"],
        lipsync_confirm=lambda w: True,
    )
    assert len(calls) == 1
    assert summary["lipsync_warnings"][0]["shot_code"] == "F2-L2"
    assert summary["lipsync_warnings"][0]["over_sec"] == pytest.approx(0.7)
    assert len(summary["lipsync_warnings"]) == 1
    on_disk = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["lipsync_warnings"][0]["shot_code"] == "F2-L2"


def test_run_episode_no_dialogue_file_composes_quietly(tmp_path: Path, monkeypatch):
    from recast.core.runner import run_episode

    video = _fake_video(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _seed_compose_workdir(workdir, video)
    called = []
    calls = []
    monkeypatch.setattr(
        "recast.stages.compose.run",
        lambda *a, **k: calls.append((a, k)) or str(workdir / "output.mp4"),
    )
    summary = run_episode(
        video,
        workdir=workdir,
        stages=["compose"],
        lipsync_confirm=lambda w: called.append(w) or True,
    )
    assert len(calls) == 1
    assert called == []  # 没有 dialogue.json：一次也没问
    assert "lipsync_warnings" not in summary


# ----- dispatch_job：经 settings 读回调并转发 -----


def test_dispatch_job_reads_confirm_from_settings(tmp_path: Path, monkeypatch):
    """主窗口把回调挂在 settings 上（_bind_lipsync_confirm），
    dispatch_job 用 getattr 读回并传进 run_episode：取消时不合成。"""
    from types import SimpleNamespace

    from recast.core.runner import dispatch_job

    video = _fake_video(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _seed_compose_workdir(workdir, video)
    _write_dialogue(workdir, [_line("F2-L2", "我没有推她", 3.2, 2.5)])
    calls = []
    monkeypatch.setattr(
        "recast.stages.compose.run",
        lambda *a, **k: calls.append((a, k)) or str(workdir / "output.mp4"),
    )
    settings = SimpleNamespace(
        output_root=str(tmp_path / "outputs"),
        lipsync_confirm=lambda w: False,
    )
    with pytest.raises(CancelledError):
        dispatch_job(
            video=video,
            workdir=workdir,
            stages=["compose"],
            settings=settings,
        )
    assert calls == []


def test_dispatch_job_explicit_confirm_beats_settings(tmp_path: Path, monkeypatch):
    """显式参数优先于 settings 上的回调：settings 说取消，参数说继续。"""
    from types import SimpleNamespace

    from recast.core.runner import dispatch_job

    video = _fake_video(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _seed_compose_workdir(workdir, video)
    _write_dialogue(workdir, [_line("F2-L2", "我没有推她", 3.2, 2.5)])
    calls = []
    monkeypatch.setattr(
        "recast.stages.compose.run",
        lambda *a, **k: calls.append((a, k)) or str(workdir / "output.mp4"),
    )
    settings = SimpleNamespace(
        output_root=str(tmp_path / "outputs"),
        lipsync_confirm=lambda w: False,
    )
    summary = dispatch_job(
        video=video,
        workdir=workdir,
        stages=["compose"],
        settings=settings,
        lipsync_confirm=lambda w: True,
    )
    assert len(calls) == 1
    assert summary["lipsync_warnings"][0]["shot_code"] == "F2-L2"
