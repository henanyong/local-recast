"""UI wiring for the 2D route: [生成设定板] and [换装预览…].

These tests are about the contract the buttons make, not about pixels: a
button must be greyed out with a tooltip that says exactly what is missing,
and a run must land somewhere the operator can look at (a sheet directory, a
before/after pair) rather than only a status line.
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _image(path: Path, bgr=(40, 60, 190), size=(96, 96)) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", np.full((size[1], size[0], 3), bgr, np.uint8))
    assert ok
    buf.tofile(str(path))
    return path


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window(tmp_path: Path, *, with_character: bool = True):
    from recast.core.service import RecastService
    from recast.desktop.library_window import LibraryWindow

    _app()
    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    if with_character:
        svc.store.insert(
            "characters", name="温念", display_name="温念", confirmed=1, authorized=1, cluster_key="c:1"
        )
        svc.ensure_character_folder("温念")
    window = LibraryWindow(svc)
    window._sync_tasks = True
    window.reload()
    return svc, window


def _select(window, row: int = 0) -> None:
    window.list.setCurrentRow(row)
    window._on_select(window.list.currentItem())


def _folder(svc) -> Path:
    return Path(svc.settings.output_root) / "characters" / "温念"


def test_buttons_are_greyed_out_without_a_selection(tmp_path: Path):
    svc, window = _window(tmp_path, with_character=False)
    try:
        assert not window.sheet_btn.isEnabled()
        assert not window.garment_btn.isEnabled()
        assert "选一个人" in window.sheet_btn.toolTip()
        assert "选一个人" in window.garment_btn.toolTip()
    finally:
        window.close()


def test_sheet_button_enables_once_a_folder_exists(tmp_path: Path):
    svc, window = _window(tmp_path)
    try:
        _select(window)
        assert window.sheet_btn.isEnabled()
        assert "设定板" in window.sheet_btn.toolTip()
        assert "sheet/" in window.sheet_btn.toolTip()
    finally:
        window.close()


def test_garment_button_says_costume_is_missing(tmp_path: Path):
    svc, window = _window(tmp_path)
    try:
        _select(window)
        assert not window.garment_btn.isEnabled()
        assert "costume/" in window.garment_btn.toolTip()
    finally:
        window.close()


def test_garment_button_enables_with_a_costume_reference(tmp_path: Path):
    svc, window = _window(tmp_path)
    try:
        _image(_folder(svc) / "costume" / "01.png")
        window.reload()
        _select(window)
        assert window.garment_btn.isEnabled()
        assert "before/after" in window.garment_btn.toolTip()
    finally:
        window.close()


def test_build_sheet_button_writes_the_sheet_and_reports(tmp_path: Path):
    svc, window = _window(tmp_path)
    try:
        _image(_folder(svc) / "face" / "01.png")
        window.reload()
        _select(window)
        window._build_sheet()
        assert "设定板" in window.status_label.text()
        assert (_folder(svc) / "sheet" / "sheet.json").is_file()
        meta = json.loads((_folder(svc) / "meta.json").read_text(encoding="utf-8"))
        assert "sheet" in meta
    finally:
        window.close()


def test_preview_button_writes_before_after_and_opens_the_result(tmp_path: Path, monkeypatch):
    """The whole point of a preview button is that it produces something you
    can look at, so the run must leave a before/after pair on disk."""
    import numpy as np
    import recast.stages.person_mask as person_mask

    monkeypatch.setattr(
        person_mask,
        "segment_person_auto",
        lambda *a, **k: {
            "mask": (lambda h, w: _block_mask(h, w))(96, 96),
            "bbox": [10.0, 10.0, 80.0, 90.0],
            "method": "stub",
            "person_count": 1,
            "warning": "",
        },
    )
    monkeypatch.setattr(
        person_mask, "estimate_neck_y", lambda *a, **k: {"neck_y": 40, "source": "stub", "note": ""}
    )

    svc, window = _window(tmp_path)
    opened: list[str] = []
    try:
        from PySide6.QtGui import QDesktopServices

        monkeypatch.setattr(
            QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile())
        )
        _image(_folder(svc) / "costume" / "01.png")
        window.reload()
        _select(window)
        window._preview_garment()

        out = _folder(svc) / "replace_preview"
        assert (out / "before_upper.png").is_file()
        assert (out / "after_upper.png").is_file()
        assert opened and opened[0].endswith("after_upper.png")
        assert "换装预览" in window.status_label.text()
    finally:
        window.close()


def _block_mask(h: int, w: int):
    import numpy as np

    mask = np.zeros((h, w), np.uint8)
    mask[40:, 20:70] = 255
    return mask
