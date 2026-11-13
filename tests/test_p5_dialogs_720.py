"""P5/B3 dialogs sweep (group 4): every remaining dialog fits 1280x720.

Spec §5.2: content that can outgrow the screen lives in a QScrollArea
(``dialog_utils.scroll_wrap``); the action bar is pinned at the bottom
outside the scroll area; the initial size is clamped with
``cap_dialog_to_screen`` (860-high -> 696 on 1280x720).

Covered here (dialog_utils itself + the settings dialog were group 1's):

* QDialogs: ShotEnhanceDialog, GenerateDialog, TimbreDialog,
  DialogueDialog, SubtitleDialog, TaskHistoryDialog (+ AuthSkipsDialog),
  QueuePanelDialog (it IS a QDialog, not an embedded panel).
* QMessageBox functions: ``auth_gate_ok`` and ``lipsync_warn_dialog`` —
  no custom layout to wrap, so they are clamped via
  ``_cap_message_box`` (real boxes only; test doubles have no ``resize``
  and are skipped by an isinstance guard against the real class).

The offscreen platform reports an 800x600 primary screen, so the
1280x720 cases monkeypatch ``dialog_utils._available_geometry`` — the
same seam the production code reads (group 1's pattern).
"""

from __future__ import annotations

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


def _screen_1280x720(monkeypatch):
    from PySide6.QtCore import QRect

    import recast.desktop.dialog_utils as du

    monkeypatch.setattr(du, "_available_geometry", lambda: QRect(0, 0, 1280, 720))


def _is_descendant(widget, ancestor) -> bool:
    node = widget
    while node is not None:
        if node is ancestor:
            return True
        node = node.parentWidget()
    return False


def _assert_pinned_bar(dlg, buttons, *, max_h: int = 696):
    """Button bar: visible, inside the dialog, outside the scroll area.

    Shows the dialog and leaves it open; the caller closes it.
    """
    from PySide6.QtWidgets import QApplication, QScrollArea

    dlg.show()
    QApplication.processEvents()
    assert dlg.height() <= max_h, dlg.height()
    assert dlg.width() <= 1256, dlg.width()
    scrolls = dlg.findChildren(QScrollArea)
    assert len(scrolls) == 1, f"want exactly 1 QScrollArea, got {len(scrolls)}"
    assert buttons.isVisible(), "button bar must be visible"
    assert buttons.isEnabled()
    top = buttons.mapTo(dlg, buttons.rect().topLeft()).y()
    bottom = buttons.mapTo(dlg, buttons.rect().bottomLeft()).y()
    assert 0 <= top < bottom <= dlg.height(), (top, bottom, dlg.height())
    assert not _is_descendant(buttons, scrolls[0]), "bar must not scroll away"
    return scrolls[0]


def _scroll_outer_to_bottom(dlg):
    """Drive the dialog's outer scroll area to its maximum."""
    from PySide6.QtWidgets import QApplication, QScrollArea

    scroll = dlg.findChild(QScrollArea)
    bar = scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    QApplication.processEvents()
    return bar.maximum()


def _scroll_table_to_bottom(dlg):
    """Drive the dialog's table to its last row (tables scroll inside)."""
    from PySide6.QtWidgets import QApplication

    bar = dlg.table.verticalScrollBar()
    assert bar.maximum() > 0, "tall table must be scrollable"
    bar.setValue(bar.maximum())
    QApplication.processEvents()
    return bar.maximum()


def _drama(svc, name: str = "演示剧") -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, new_name=name,
        root_path=f"C:/demo/{name}", episode_count=1,
    )


def _task(svc, drama_id: str, *, episode: str, status: str, **extra) -> dict:
    fields: dict = {
        "drama_id": drama_id,
        "episode_name": episode,
        "video_path": f"C:/demo/{episode}.mp4",
        "status": status,
        "progress": "等待执行",
        "progress_pct": 0,
    }
    fields.update(extra)
    return svc.store.insert("tasks", **fields)


# ----- ShotEnhanceDialog -----------------------------------------------------


class _ShotSvc:
    """Minimal stand-in: only what ShotEnhanceDialog touches."""

    def __init__(self):
        self.toggled: list = []

    def shot_enhance_flags(self, task_id):
        return {}

    def set_shot_enhance(self, task_id, code, checked):
        self.toggled.append((task_id, code, checked))


def _shot_rows(n: int):
    return [
        ({"id": f"task-{i // 10}", "episode_name": f"第{i // 10 + 1}集"},
         {"shot_code": f"F{i:02d}-L1"})
        for i in range(n)
    ]


def test_shot_enhance_fits_1280x720_buttons_pinned(monkeypatch):
    from recast.desktop.shot_enhance_dialog import ShotEnhanceDialog

    _app()
    _screen_1280x720(monkeypatch)
    dlg = ShotEnhanceDialog(_ShotSvc(), "drama-1", _shot_rows(3), drama_name="演示剧")
    assert (dlg.width(), dlg.height()) == (480, 480)
    scroll = _assert_pinned_bar(dlg, dlg.close_btn)
    assert dlg.close_btn.isVisible()
    # Hint + all checkboxes live inside the scroll area's content.
    assert _is_descendant(dlg.error_label, scroll.widget())
    for box in dlg.boxes.values():
        assert _is_descendant(box, scroll.widget())
    dlg.close()


def test_shot_enhance_many_shots_scroll_to_bottom_close_still_clickable(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from recast.desktop.shot_enhance_dialog import ShotEnhanceDialog

    _app()
    _screen_1280x720(monkeypatch)
    svc = _ShotSvc()
    dlg = ShotEnhanceDialog(svc, "drama-1", _shot_rows(40), drama_name="演示剧")
    assert len(dlg.boxes) == 40
    dlg.show()
    QApplication.processEvents()
    try:
        assert _scroll_outer_to_bottom(dlg) > 0, "40 rows must overflow the dialog"
        # Bottom reached, [关闭] still visible and inside the dialog.
        assert dlg.close_btn.isVisible()
        bottom = dlg.close_btn.mapTo(dlg, dlg.close_btn.rect().bottomLeft()).y()
        assert bottom <= dlg.height()
        # A toggle still writes through after scrolling.
        box = dlg.boxes[("task-0", "F00-L1")]
        box.setChecked(True)
        assert ("task-0", "F00-L1", True) in svc.toggled
    finally:
        dlg.close()


def test_shot_enhance_empty_rows_still_opens(monkeypatch):
    from recast.desktop.shot_enhance_dialog import ShotEnhanceDialog

    _app()
    _screen_1280x720(monkeypatch)
    dlg = ShotEnhanceDialog(_ShotSvc(), "drama-1", [], drama_name="演示剧")
    try:
        assert dlg.boxes == {}
        assert dlg.height() <= 696
        assert dlg.close_btn.isVisible() is False  # not shown yet
        dlg.show()
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()
        assert dlg.close_btn.isVisible()
    finally:
        dlg.close()


# ----- GenerateDialog --------------------------------------------------------


def test_generate_dialog_fits_1280x720_buttons_pinned(monkeypatch, svc):
    from PySide6.QtWidgets import QDialogButtonBox

    from recast.desktop.generate_dialog import GenerateDialog

    _app()
    _screen_1280x720(monkeypatch)
    dlg = GenerateDialog(svc)
    assert (dlg.width(), dlg.height()) == (640, 320)
    scroll = _assert_pinned_bar(dlg, dlg._buttons)
    buttons = dlg.findChild(QDialogButtonBox)
    assert buttons is dlg._buttons
    assert buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    # Form fields live inside the scroll area's content.
    for field in (dlg.name, dlg.pack, dlg.storyboard, dlg.dialogue):
        assert _is_descendant(field, scroll.widget())
    dlg.close()


def test_generate_dialog_apply_still_works(monkeypatch):
    from types import SimpleNamespace

    from recast.desktop.generate_dialog import GenerateDialog

    _app()
    _screen_1280x720(monkeypatch)
    calls = {}

    def create_script_drama(name, pack, storyboard=None, dialogue=None):
        calls.update(name=name, pack=pack, storyboard=storyboard, dialogue=dialogue)
        return {"id": "new-drama"}

    dlg = GenerateDialog(SimpleNamespace(create_script_drama=create_script_drama))
    try:
        dlg.name.setText("测试剧")
        dlg.pack.setText("/tmp/pack")
        assert dlg.apply() == {"id": "new-drama"}
        assert calls == {
            "name": "测试剧", "pack": "/tmp/pack",
            "storyboard": None, "dialogue": None,
        }
    finally:
        dlg.close()


# ----- TimbreDialog ----------------------------------------------------------


def test_timbre_dialog_fits_1280x720_buttons_pinned(monkeypatch, svc):
    from recast.desktop.timbre_dialog import TimbreDialog

    _app()
    _screen_1280x720(monkeypatch)
    dlg = TimbreDialog(svc, "温念")
    assert (dlg.width(), dlg.height()) == (480, 300)
    scroll = _assert_pinned_bar(dlg, dlg._buttons)
    for widget in (dlg.pitch_slider, dlg.rate_slider, dlg.style_edit, dlg.audition_btn):
        assert _is_descendant(widget, scroll.widget()), widget
    assert dlg.audition_btn.isVisible()
    dlg.close()


def test_timbre_dialog_sliders_and_labels_still_work(monkeypatch, svc):
    from recast.desktop.timbre_dialog import TimbreDialog

    _app()
    _screen_1280x720(monkeypatch)
    dlg = TimbreDialog(svc, "温念")
    try:
        assert dlg.pitch() == 0.0
        assert dlg.rate() == 1.0
        assert dlg.style() == ""
        dlg.pitch_slider.setValue(4)
        assert dlg.pitch() == 2.0
        assert "+2.0 半音" in dlg.pitch_label.text()
        dlg.rate_slider.setValue(30)
        assert dlg.rate() == 1.5
        assert "1.50 倍" in dlg.rate_label.text()
        dlg.style_edit.setText("温柔")
        assert dlg.style() == "温柔"
    finally:
        dlg.close()


# ----- DialogueDialog --------------------------------------------------------


def test_dialogue_dialog_fits_1280x720_buttons_pinned(monkeypatch, svc):
    from recast.desktop.dialogue_dialog import DialogueDialog

    _app()
    _screen_1280x720(monkeypatch)
    dlg = DialogueDialog(svc, "温念")
    assert (dlg.width(), dlg.height()) == (720, 480)
    scroll = _assert_pinned_bar(dlg, dlg.save_btn.parentWidget())
    for widget in (dlg.workdir_edit, dlg.only_mine, dlg.table, dlg.status):
        assert _is_descendant(widget, scroll.widget()), widget
    assert dlg.redub_btn.objectName() == "primary"
    assert not dlg.save_btn.isEnabled()  # no workdir loaded yet
    assert not dlg.redub_btn.isEnabled()
    dlg.close()


def test_dialogue_dialog_loads_and_long_list_scrolls(monkeypatch, svc, tmp_path: Path):
    import json

    from PySide6.QtWidgets import QApplication

    from recast.desktop.dialogue_dialog import DialogueDialog

    _app()
    _screen_1280x720(monkeypatch)
    workdir = tmp_path / "ep1"
    workdir.mkdir()
    lines = [
        {"shot_code": f"F{i:02d}-L1", "speaker": "温念", "text": f"第{i}句台词"}
        for i in range(60)
    ]
    (workdir / "dialogue.json").write_text(
        json.dumps(lines, ensure_ascii=False), encoding="utf-8"
    )
    dlg = DialogueDialog(svc, "温念")
    try:
        dlg.workdir_edit.setText(str(workdir))
        dlg._load()
        assert dlg.table.rowCount() == 60
        assert "已加载 60 句对白" in dlg.status.text()
        assert dlg.save_btn.isEnabled()
        dlg.show()
        QApplication.processEvents()
        _scroll_table_to_bottom(dlg)
        # Table scrolled to the last row; the bottom bar is untouched.
        assert dlg.table.item(59, 2).text() == "第59句台词"
        assert dlg.close_btn.isVisible()
        bottom = dlg.close_btn.mapTo(dlg, dlg.close_btn.rect().bottomLeft()).y()
        assert bottom <= dlg.height()
    finally:
        dlg.close()


# ----- SubtitleDialog --------------------------------------------------------


def _srt_text(n: int) -> str:
    parts = []
    for i in range(n):
        start = i * 2
        end = start + 1
        parts.append(
            f"{i + 1}\n"
            f"00:{start // 60:02d}:{start % 60:02d},000 --> "
            f"00:{end // 60:02d}:{end % 60:02d},000\n"
            f"第{i + 1}行字幕\n"
        )
    return "\n".join(parts)


def test_subtitle_dialog_fits_1280x720_buttons_pinned(monkeypatch, tmp_path: Path):
    from PySide6.QtWidgets import QDialogButtonBox

    from recast.desktop.subtitle_dialog import SubtitleDialog

    _app()
    _screen_1280x720(monkeypatch)
    srt = tmp_path / "subtitles.srt"
    srt.write_text(_srt_text(3), encoding="utf-8")
    dlg = SubtitleDialog(srt)
    assert (dlg.width(), dlg.height()) == (720, 460)
    scroll = _assert_pinned_bar(dlg, dlg._buttons)
    buttons = dlg.findChild(QDialogButtonBox)
    assert buttons is dlg._buttons
    for widget in (dlg.table, dlg.split_btn, dlg.merge_btn, dlg.delete_btn,
                   dlg.insert_btn, dlg.status):
        assert _is_descendant(widget, scroll.widget()), widget
    assert dlg.table.rowCount() == 3
    dlg.close()


def test_subtitle_dialog_long_srt_scrolls_and_split_still_works(monkeypatch, tmp_path: Path):
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from recast.desktop.subtitle_dialog import SubtitleDialog

    _app()
    _screen_1280x720(monkeypatch)
    srt = tmp_path / "subtitles.srt"
    srt.write_text(_srt_text(60), encoding="utf-8")
    dlg = SubtitleDialog(srt)
    try:
        assert dlg.table.rowCount() == 60
        dlg.show()
        QApplication.processEvents()
        _scroll_table_to_bottom(dlg)
        assert dlg.table.item(59, 2).text() == "第60行字幕"
        save = dlg._buttons.button(QDialogButtonBox.StandardButton.Save)
        assert save is not None and save.isVisible()
        bottom = dlg._buttons.mapTo(dlg, dlg._buttons.rect().bottomLeft()).y()
        assert bottom <= dlg.height()
        # Row ops still work after scrolling.
        dlg.table.setCurrentCell(0, 2)
        dlg.split_btn.click()
        assert dlg.table.rowCount() == 61
    finally:
        dlg.close()


# ----- TaskHistoryDialog + AuthSkipsDialog ------------------------------------


def test_task_history_fits_1280x720_buttons_pinned(monkeypatch, svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    _screen_1280x720(monkeypatch)
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="已完成")
    dlg = TaskHistoryDialog(svc)
    assert (dlg.width(), dlg.height()) == (860, 560)
    scroll = _assert_pinned_bar(dlg, dlg.rerun_btn.parentWidget())
    for widget in (dlg.filter_box, dlg.count_label, dlg.table, dlg.detail, dlg.status):
        assert _is_descendant(widget, scroll.widget()), widget
    assert dlg.table.rowCount() == 1
    assert dlg.close_btn.isVisible()
    dlg.close()


def test_task_history_long_list_scrolls_to_bottom(monkeypatch, svc):
    from PySide6.QtWidgets import QApplication

    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    _screen_1280x720(monkeypatch)
    drama = _drama(svc)
    for i in range(60):
        _task(svc, drama["id"], episode=f"第{i + 1}集", status="已完成")
    dlg = TaskHistoryDialog(svc)
    try:
        assert dlg.table.rowCount() == 60
        assert "共 60 条" in dlg.count_label.text()
        dlg.show()
        QApplication.processEvents()
        _scroll_table_to_bottom(dlg)
        assert dlg.rerun_btn.isVisible()
        bottom = dlg.close_btn.mapTo(dlg, dlg.close_btn.rect().bottomLeft()).y()
        assert bottom <= dlg.height()
    finally:
        dlg.close()


def test_auth_skips_dialog_fits_and_long_text_scrolls(monkeypatch):
    from PySide6.QtWidgets import QApplication, QScrollArea

    from recast.desktop.task_history_dialog import AuthSkipsDialog

    _app()
    _screen_1280x720(monkeypatch)
    long_text = "\n".join(f"温念 — 未授权 — 2026-11-13 10:{i % 60:02d}:00" for i in range(200))
    dlg = AuthSkipsDialog("第1集", long_text)
    assert (dlg.width(), dlg.height()) == (520, 300)
    scroll = _assert_pinned_bar(dlg, dlg.close_btn)
    assert _is_descendant(dlg.body, scroll.widget())
    dlg.show()
    QApplication.processEvents()
    try:
        bar = dlg.body.verticalScrollBar()
        assert bar.maximum() > 0, "200 lines must scroll inside the read-only body"
        bar.setValue(bar.maximum())
        QApplication.processEvents()
        assert dlg.close_btn.isVisible()
    finally:
        dlg.close()


# ----- QueuePanelDialog ------------------------------------------------------


def test_queue_panel_fits_1280x720_close_pinned(monkeypatch, svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    _screen_1280x720(monkeypatch)
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第1集", status="处理中", progress="人物替换中")
    dlg = QueuePanelDialog(svc)
    assert (dlg.width(), dlg.height()) == (920, 520)
    scroll = _assert_pinned_bar(dlg, dlg.close_btn)
    for widget in (dlg.count_label, dlg.scheduler_label,
                   dlg.only_in_progress_checkbox, dlg.auto_refresh_checkbox,
                   dlg.empty_label, dlg.table, dlg.status):
        assert _is_descendant(widget, scroll.widget()), widget
    # P3/N33 behaviour untouched: [仅进行中] on, auto-refresh running.
    assert dlg.only_in_progress_checkbox.isChecked() is True
    assert dlg.auto_refresh_checkbox.isChecked() is True
    assert dlg._auto_refresh_timer.isActive()
    dlg.close()
    assert not dlg._auto_refresh_timer.isActive()


def test_queue_panel_long_queue_scrolls_and_timer_stops_on_close(monkeypatch, svc):
    from PySide6.QtWidgets import QApplication

    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    _screen_1280x720(monkeypatch)
    drama = _drama(svc)
    for i in range(60):
        _task(svc, drama["id"], episode=f"第{i + 1}集", status="处理中",
              progress="人物替换中")
    dlg = QueuePanelDialog(svc)
    try:
        assert dlg.table.rowCount() == 60
        dlg.show()
        QApplication.processEvents()
        _scroll_table_to_bottom(dlg)
        assert dlg.close_btn.isVisible()
        bottom = dlg.close_btn.mapTo(dlg, dlg.close_btn.rect().bottomLeft()).y()
        assert bottom <= dlg.height()
    finally:
        dlg.close()
        assert not dlg._auto_refresh_timer.isActive()


# ----- auth_gate_ok (QMessageBox) --------------------------------------------


def _real_recording_box(click: str):
    """Build a real-QMessageBox subclass recording exec + chosen button."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    class RecordingBox(QMessageBox):
        shown: list = []

        def __init__(self, parent=None):
            super().__init__(parent)
            RecordingBox.shown.append(self)

        def exec(self):
            self.show()
            QApplication.processEvents()
            return 1

        def clickedButton(self):
            for btn in self.buttons():
                if btn.text() == click:
                    return btn
            return None

    RecordingBox.shown.clear()
    return RecordingBox


def test_auth_gate_real_box_fits_1280x720_with_40_long_names(monkeypatch):
    from PySide6.QtWidgets import QApplication

    import recast.desktop.auth_gate_dialog as ag

    _app()
    _screen_1280x720(monkeypatch)
    missing = [{"name": f"人物{i:02d}名字很长很长"} for i in range(40)]
    monkeypatch.setattr(
        ag, "check_replace_auth",
        lambda svc, dids: {"unauthorized": missing, "authorized_count": 2},
    )
    seen = {}
    box_cls = _real_recording_box("继续，只替换已授权人物")
    assert ag.auth_gate_ok(
        None, object(), ["d1"], _box_cls=box_cls,
        on_skipped=lambda m, d: seen.setdefault("skipped", (m, d)),
    ) is True
    assert len(box_cls.shown) == 1
    box = box_cls.shown[0]
    assert box.width() <= 1256 and box.height() <= 696, (box.width(), box.height())
    assert "人物39名字很长很长" in box.text()
    assert seen["skipped"][0] == missing
    QApplication.processEvents()


def test_auth_gate_real_box_zero_authorized_disables_continue(monkeypatch):
    import recast.desktop.auth_gate_dialog as ag

    _app()
    _screen_1280x720(monkeypatch)
    missing = [{"name": "陆寒峥"}]
    monkeypatch.setattr(
        ag, "check_replace_auth",
        lambda svc, dids: {"unauthorized": missing, "authorized_count": 0},
    )
    box_cls = _real_recording_box("去人物库补授权")
    assert ag.auth_gate_ok(None, object(), ["d1"], _box_cls=box_cls) is False
    box = box_cls.shown[0]
    assert box.width() <= 1256 and box.height() <= 696
    continue_btn = next(b for b in box.buttons() if b.text() == "继续，只替换已授权人物")
    assert not continue_btn.isEnabled()


def test_auth_gate_fake_box_without_resize_still_works(monkeypatch):
    """Boundary: the isinstance guard must skip doubles with no resize."""
    from PySide6.QtWidgets import QMessageBox as _RealBox

    import recast.desktop.auth_gate_dialog as ag

    _app()

    class NoResizeFakeBox:
        ButtonRole = _RealBox.ButtonRole
        click = "继续，只替换已授权人物"

        def __init__(self, parent=None):
            self._buttons = {}

        def setWindowTitle(self, title):
            self.title = title

        def setText(self, text):
            self.text = text

        def addButton(self, text, role):
            btn = object()
            self._buttons[text] = btn
            return btn

        def exec(self):
            return 1

        def clickedButton(self):
            return self._buttons.get(type(self).click)

    monkeypatch.setattr(
        ag, "check_replace_auth",
        lambda svc, dids: {"unauthorized": [{"name": "陆寒峥"}], "authorized_count": 1},
    )
    assert ag.auth_gate_ok(
        None, object(), ["d1"], _box_cls=NoResizeFakeBox,
        on_skipped=lambda m, d: None,
    ) is True


# ----- lipsync_warn_dialog (QMessageBox) -------------------------------------


def _warning(i: int):
    from recast.quality.review import LipSyncWarning

    return LipSyncWarning(
        shot_code=f"F{i}-L{i}", text="我没有推她，是她自己摔的，你不要再说了",
        speaker="温念", audio_sec=3.2, slot_sec=2.5, over_sec=0.7,
    )


def test_lipsync_warn_real_box_lists_at_most_8_and_fits(monkeypatch):
    from recast.desktop.lipsync_warn_dialog import lipsync_warn_dialog

    _app()
    _screen_1280x720(monkeypatch)
    box_cls = _real_recording_box("继续合成")
    assert lipsync_warn_dialog(None, [_warning(i) for i in range(30)], _box_cls=box_cls) is True
    assert len(box_cls.shown) == 1
    box = box_cls.shown[0]
    assert box.width() <= 1256 and box.height() <= 696, (box.width(), box.height())
    # MAX_LISTED_LINES caps the list; the rest is only a total.
    assert "F7-L7" in box.text()
    assert "F8-L8" not in box.text()
    assert "等 30 行超标" in box.text()


def test_lipsync_warn_real_box_cancel_returns_false_and_fits(monkeypatch):
    from recast.desktop.lipsync_warn_dialog import lipsync_warn_dialog

    _app()
    _screen_1280x720(monkeypatch)
    box_cls = _real_recording_box("取消，回去改台词")
    assert lipsync_warn_dialog(None, [_warning(0)], _box_cls=box_cls) is False
    box = box_cls.shown[0]
    assert box.width() <= 1256 and box.height() <= 696


def test_lipsync_warn_fake_box_without_resize_still_works():
    """Boundary: the isinstance guard must skip doubles with no resize."""
    from PySide6.QtWidgets import QMessageBox as _RealBox

    from recast.desktop.lipsync_warn_dialog import lipsync_warn_dialog

    _app()

    class NoResizeFakeBox:
        ButtonRole = _RealBox.ButtonRole
        click = "取消，回去改台词"

        def __init__(self, parent=None):
            self._buttons = {}

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
            return 1

        def clickedButton(self):
            return self._buttons.get(type(self).click)

    assert lipsync_warn_dialog(None, [_warning(0)], _box_cls=NoResizeFakeBox) is False
