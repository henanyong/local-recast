"""P5/B3 settings dialog: scrollable content with the OK/Cancel bar pinned
at the bottom, sized by cap_dialog_to_screen (spec §5.2).

Daniel's report: on a 1280x720 screen the settings dialog's bottom (the
volcano-engine section + OK/Cancel) fell off-screen and could not be clicked.
The fix: all GroupBox content lives inside a QScrollArea; the button bar
sits outside it, fixed at the bottom; the dialog is clamped to the screen
minus a 24px margin (640x860 -> 640x696 on 1280x720).

The offscreen platform reports an 800x600 primary screen, so the 1280x720
cases monkeypatch ``dialog_utils._available_geometry`` -- the same seam the
production code reads.
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


# ----- dialog_utils.scroll_wrap --------------------------------------------


def test_scroll_wrap_structure_buttons_pinned():
    from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

    from recast.desktop.dialog_utils import scroll_wrap

    _app()
    content = QWidget()
    QVBoxLayout(content).addWidget(QLabel("content"))
    buttons = QWidget()
    QVBoxLayout(buttons).addWidget(QPushButton("OK"))
    wrapper = scroll_wrap(content, buttons)
    scrolls = wrapper.findChildren(QScrollArea)
    assert len(scrolls) == 1
    scroll = scrolls[0]
    assert scroll.widgetResizable()
    assert scroll.widget() is content
    assert scroll.parentWidget() is wrapper
    # The button bar is a direct child of the wrapper layout, NOT inside the
    # scroll area: it stays put while the content scrolls.
    assert buttons.parentWidget() is wrapper
    assert not _is_descendant(buttons, scroll)
    assert wrapper.layout().count() == 2
    assert wrapper.layout().itemAt(0).widget() is scroll
    assert wrapper.layout().itemAt(1).widget() is buttons


def test_scroll_wrap_without_buttons():
    from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

    from recast.desktop.dialog_utils import scroll_wrap

    _app()
    content = QWidget()
    QVBoxLayout(content).addWidget(QLabel("only content"))
    wrapper = scroll_wrap(content)
    assert len(wrapper.findChildren(QScrollArea)) == 1
    assert wrapper.layout().count() == 1


def test_scroll_wrap_rejects_layout_not_widget():
    from PySide6.QtWidgets import QVBoxLayout

    from recast.desktop.dialog_utils import scroll_wrap

    _app()
    with pytest.raises(TypeError):
        scroll_wrap(QVBoxLayout())
    with pytest.raises(TypeError):
        from PySide6.QtWidgets import QWidget

        scroll_wrap(QWidget(), buttons=QVBoxLayout())


def test_scroll_wrap_content_actually_scrolls():
    from PySide6.QtWidgets import QApplication, QLabel, QScrollArea, QVBoxLayout, QWidget

    from recast.desktop.dialog_utils import scroll_wrap

    _app()
    content = QWidget()
    lay = QVBoxLayout(content)
    for i in range(50):
        lay.addWidget(QLabel(f"row {i}"))
    wrapper = scroll_wrap(content)
    wrapper.resize(400, 200)
    wrapper.show()
    QApplication.processEvents()
    scroll = wrapper.findChildren(QScrollArea)[0]
    assert scroll.verticalScrollBar().maximum() > 0  # tall content scrolls, not clipped
    wrapper.close()


# ----- dialog_utils.cap_dialog_to_screen -----------------------------------


def test_cap_dialog_clamps_860_high_to_696_at_1280x720(monkeypatch):
    from PySide6.QtWidgets import QDialog

    from recast.desktop.dialog_utils import cap_dialog_to_screen

    _app()
    _screen_1280x720(monkeypatch)
    dlg = QDialog()
    cap_dialog_to_screen(dlg, 640, 860)
    assert dlg.width() == 640
    assert dlg.height() == 696  # 720 - 24px margin


def test_cap_dialog_clamps_oversize_width_too(monkeypatch):
    from PySide6.QtWidgets import QDialog

    from recast.desktop.dialog_utils import cap_dialog_to_screen

    _app()
    _screen_1280x720(monkeypatch)
    dlg = QDialog()
    cap_dialog_to_screen(dlg, 2000, 2000)
    assert dlg.width() == 1256  # 1280 - 24
    assert dlg.height() == 696


def test_cap_dialog_keeps_small_request(monkeypatch):
    from PySide6.QtWidgets import QDialog

    from recast.desktop.dialog_utils import cap_dialog_to_screen

    _app()
    _screen_1280x720(monkeypatch)
    dlg = QDialog()
    cap_dialog_to_screen(dlg, 400, 300)
    assert (dlg.width(), dlg.height()) == (400, 300)


def test_cap_dialog_never_exceeds_real_screen(monkeypatch):
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QDialog

    import recast.desktop.dialog_utils as du

    _app()
    monkeypatch.setattr(du, "_available_geometry", lambda: QRect(0, 0, 300, 200))
    dlg = QDialog()
    du.cap_dialog_to_screen(dlg, 640, 860)
    assert dlg.width() <= 300 - 24
    assert dlg.height() <= 200 - 24


# ----- SettingsDialog --------------------------------------------------------


def test_settings_dialog_fits_1280x720_with_buttons_visible(svc, monkeypatch):
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QGroupBox, QScrollArea

    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    _screen_1280x720(monkeypatch)
    dlg = SettingsDialog(svc)
    dlg.show()
    QApplication.processEvents()
    try:
        assert dlg.height() <= 696, dlg.height()
        assert dlg.width() == 640
        buttons = dlg.findChild(QDialogButtonBox)
        assert buttons is not None
        # Button bar is visible, clickable, and inside the dialog bounds.
        assert buttons.isVisible()
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok is not None and ok.isEnabled()
        bottom = buttons.mapTo(dlg, buttons.rect().bottomLeft()).y()
        assert bottom <= dlg.height(), (bottom, dlg.height())
        # Button bar is NOT inside the scroll area.
        scroll = dlg.findChild(QScrollArea)
        assert scroll is not None
        assert not _is_descendant(buttons, scroll)
        # Every settings group lives inside the scroll area (火山引擎 section
        # included -- that was the part falling off-screen).
        groups = dlg.findChildren(QGroupBox)
        assert len(groups) >= 6
        for group in groups:
            assert _is_descendant(group, scroll.widget()), group.title()
        volc = next(g for g in groups if "云通道" in g.title())
        assert _is_descendant(volc, scroll.widget())
    finally:
        dlg.close()


def test_settings_dialog_attributes_intact(svc):
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    try:
        for name in (
            "tier", "works", "output", "models", "ffmpeg", "comfy",
            "concurrency", "lang", "burn", "cover_mode", "pose",
            "preset_buttons", "preset_label",
            "named_preset_combo", "named_save_btn", "named_load_btn", "named_delete_btn",
            "nvenc", "crf", "scene_threshold", "min_shot", "pose_quality",
            "pose_every", "detect_every",
            "cloud_enabled", "volc_key", "deep_key", "oai_key", "oai_base",
            "auto_upload", "upload_mode", "upload_folder", "upload_webhook", "upload_command",
        ):
            assert hasattr(dlg, name), name
        assert set(dlg.preset_buttons) == {"high", "balanced", "fast"}
    finally:
        dlg.close()


def test_settings_dialog_apply_still_works(svc):
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    try:
        dlg.crf.setValue(19)
        dlg.cover_mode.setCurrentIndex(dlg.cover_mode.findData("bar"))
        dlg.apply()
        assert svc.settings.export_crf == 19
        assert svc.settings.cover_mode == "bar"
    finally:
        dlg.close()
