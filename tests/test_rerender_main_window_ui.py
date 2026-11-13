"""UI tests for the main-window body re-render (pilot) button.

Offscreen Qt only. File dialogs, message boxes, clip duration and the
service call are all stubbed: no video, no Blender, no real render.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def svc(tmp_path: Path):
    from recast.core.service import RecastService

    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


def _main(svc):
    from recast.desktop.main_window import MainWindow

    _app()
    window = MainWindow(svc)
    # 停掉 400ms 刷新定时器：测试里窗口 close 后对象仍活着，定时器继续
    # 刷会在后续测试的 processEvents() 里触发刷新路径（曾因此挂住全套）。
    window.timer.stop()
    return window


def _button(window):
    for btn in window.findChildren(type(window.rerender_btn)):
        if btn.text().startswith("身体重渲染"):
            return btn
    raise AssertionError("身体重渲染（试点）按钮不存在")


class _FakeThread:
    """Runs the target synchronously so the result signal fires in-test."""

    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def _stub_dialogs(monkeypatch, window, *, files, question_answer):
    """Stub QFileDialog + QMessageBox on the main_window module."""
    import recast.desktop.main_window as mw
    from PySide6.QtWidgets import QMessageBox as RealBox

    answers = iter(files)

    def _get_open(*args, **kwargs):
        return next(answers), ""

    monkeypatch.setattr(mw.QFileDialog, "getOpenFileName", staticmethod(_get_open))

    shown: dict = {"information": [], "warning": []}

    class _Box:
        StandardButton = RealBox.StandardButton

        @staticmethod
        def information(parent, title, text):
            shown["information"].append(text)
            return RealBox.StandardButton.Ok

        @staticmethod
        def warning(parent, title, text):
            shown["warning"].append(text)
            return RealBox.StandardButton.Ok

        @staticmethod
        def question(parent, title, text, buttons=None, default=None):
            return question_answer

    monkeypatch.setattr(mw, "QMessageBox", _Box)
    return shown


def test_rerender_button_exists_and_names_blender(svc):
    window = _main(svc)
    btn = _button(window)
    assert btn.text() == "身体重渲染（试点）"
    assert btn.isEnabled()
    assert "Blender" in btn.toolTip()
    assert "不会直接出片" in btn.toolTip()


def test_long_clip_rejected_before_service(svc, monkeypatch):
    import recast.desktop.main_window as mw
    import recast.stages.rerender as rerender
    from PySide6.QtWidgets import QMessageBox as RealBox

    window = _main(svc)
    shown = _stub_dialogs(
        monkeypatch,
        window,
        files=["/tmp/long.mp4", "/tmp/model.glb"],
        question_answer=RealBox.StandardButton.Yes,
    )
    monkeypatch.setattr(rerender, "clip_duration_s", lambda clip: 6.0)
    called: list = []
    monkeypatch.setattr(svc, "run_body_rerender", lambda *a, **k: called.append((a, k)))

    window.rerender_btn.click()

    assert called == []
    assert shown["information"], "超长片段应弹提示拒绝"
    assert "5 秒" in shown["information"][0]


def test_multi_person_rejected_before_service(svc, monkeypatch):
    import recast.stages.rerender as rerender
    from PySide6.QtWidgets import QMessageBox as RealBox

    window = _main(svc)
    shown = _stub_dialogs(
        monkeypatch,
        window,
        files=["/tmp/clip.mp4", "/tmp/model.glb"],
        question_answer=RealBox.StandardButton.No,
    )
    monkeypatch.setattr(rerender, "clip_duration_s", lambda clip: 5.0)
    called: list = []
    monkeypatch.setattr(svc, "run_body_rerender", lambda *a, **k: called.append((a, k)))

    window.rerender_btn.click()

    assert called == []
    assert shown["information"], "多人片段应弹提示拒绝"
    assert "单人" in shown["information"][0]


def test_valid_clip_calls_service(svc, monkeypatch, tmp_path):
    import recast.desktop.main_window as mw
    import recast.stages.rerender as rerender
    from PySide6.QtWidgets import QMessageBox as RealBox

    window = _main(svc)
    shown = _stub_dialogs(
        monkeypatch,
        window,
        files=["/tmp/clip.mp4", "/tmp/model.glb"],
        question_answer=RealBox.StandardButton.Yes,
    )
    monkeypatch.setattr(rerender, "clip_duration_s", lambda clip: 5.0)
    monkeypatch.setattr(mw.threading, "Thread", _FakeThread)
    out_path = tmp_path / "out.mp4"
    called: list = []

    def _fake_run(clip, glb, **kwargs):
        called.append((clip, glb))
        return {"out_path": str(out_path), "frames": 120}

    monkeypatch.setattr(svc, "run_body_rerender", _fake_run)

    window.rerender_btn.click()

    assert called == [("/tmp/clip.mp4", "/tmp/model.glb")]
    assert shown["information"], "成功后应弹完成提示"
    assert "120" in shown["information"][0]
    assert str(out_path) in shown["information"][0]
    assert window.rerender_btn.isEnabled()
    assert window.rerender_btn.text() == "身体重渲染（试点）"


def test_no_blender_error_is_shown_not_silent(svc, monkeypatch):
    import recast.desktop.main_window as mw
    import recast.stages.rerender as rerender
    from PySide6.QtWidgets import QMessageBox as RealBox

    window = _main(svc)
    shown = _stub_dialogs(
        monkeypatch,
        window,
        files=["/tmp/clip.mp4", "/tmp/model.glb"],
        question_answer=RealBox.StandardButton.Yes,
    )
    monkeypatch.setattr(rerender, "clip_duration_s", lambda clip: 5.0)
    monkeypatch.setattr(mw.threading, "Thread", _FakeThread)

    def _fake_run(clip, glb, **kwargs):
        raise RuntimeError("本机未找到 Blender，无法渲染。")

    monkeypatch.setattr(svc, "run_body_rerender", _fake_run)

    window.rerender_btn.click()

    assert shown["warning"], "无 Blender 时应把 service 的报错弹出来"
    assert "Blender" in shown["warning"][0]
    assert window.rerender_btn.isEnabled()
