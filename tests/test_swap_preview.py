"""C 线：替换预览（单帧换脸对照）。

单元测试：真换脸不可用时诚实降级为对齐叠加预览、stub 后端走真换脸
路径、缺文件报错。
UI 测试：人物库中间舞台有「替换预览」按钮；没选人物时灰掉；点了
出左右对照弹窗，降级时弹窗里有“不是最终效果”字样和身体占位说明。
"""

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _face_image(path: Path, bg=(120, 140, 160), skin=(220, 180, 150)) -> Path:
    """A face-like drawing (ellipse + two eyes); Haar may or may not fire,
    the overlay falls back to the centre box either way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 240), bg)
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 30, 140, 150), fill=skin)
    draw.ellipse((80, 80, 95, 95), fill=(30, 30, 30))
    draw.ellipse((105, 80, 120, 95), fill=(30, 30, 30))
    img.save(path)
    return path


def _service(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window_with_faces(tmp_path: Path):
    """Library window with one character, two face samples, face a.jpg assembled."""
    from recast.desktop.library_window import LibraryWindow

    _app()
    svc = _service(tmp_path)
    svc.store.insert(
        "characters",
        name="温念",
        display_name="温念",
        confirmed=1,
        authorized=1,
        cluster_key="cluster:温念",
    )
    folder = Path(svc.ensure_character_folder("温念")["folder"])
    _face_image(folder / "face" / "a.jpg", (200, 30, 30), (220, 180, 150))
    _face_image(folder / "face" / "b.jpg", (30, 30, 200), (200, 150, 120))
    svc.set_character_assembly("温念", face=str(folder / "face" / "a.jpg"))
    window = LibraryWindow(svc)
    window._sync_tasks = True
    window.list.setCurrentRow(0)
    window._on_select(window.list.currentItem())
    return svc, window, folder


def _dialog_labels(window):
    from PySide6.QtWidgets import QLabel

    dialog = window._swap_preview_dialog
    return [w.text() for w in dialog.findChildren(QLabel)]


# ----- 单元：预览逻辑 -----


def test_status_names_the_missing_weight_file(tmp_path: Path):
    """The status must say *what* is missing, not just "unavailable".

    Deliberately points at a path that cannot exist instead of assuming the
    machine has no weights: once inswapper_128.onnx is installed the old
    environment-based assertion silently inverted its meaning.
    """
    from recast.config import RecastConfig
    from recast.stages.swap_preview import inswapper_status

    cfg = RecastConfig()
    cfg.replace.model = str(tmp_path / "definitely-not-here.onnx")
    status = inswapper_status(cfg)
    assert status["available"] is False
    assert "权重" in status["reason"]


def test_overlay_fallback_is_labelled_not_final(tmp_path: Path, monkeypatch):
    import cv2

    import recast.stages.swap_preview as module
    from recast.stages.swap_preview import NOT_FINAL_TEXT

    # Force the "no real backend" condition: the fallback path has to be
    # testable on a machine that does have the weights installed.
    monkeypatch.setattr(
        module, "inswapper_status", lambda cfg=None: {"available": False, "reason": "测试构造"}
    )
    src = _face_image(tmp_path / "src.jpg")
    ref = _face_image(tmp_path / "ref.jpg", (160, 120, 100), (200, 150, 120))
    result = module.preview_swap(src, ref)
    assert result["method"] == "aligned_overlay"
    assert result["is_final"] is False
    assert NOT_FINAL_TEXT in result["note"]
    assert result["image"].shape == cv2.imread(str(src)).shape
    assert not (result["image"] == cv2.imread(str(src))).all()


def test_preview_swap_missing_file(tmp_path: Path):
    from recast.stages.swap_preview import preview_swap

    ref = _face_image(tmp_path / "ref.jpg")
    with pytest.raises(FileNotFoundError):
        preview_swap(tmp_path / "nope.jpg", ref)
    with pytest.raises(FileNotFoundError):
        preview_swap(ref, tmp_path / "nope.jpg")


def test_preview_swap_writes_out_path(tmp_path: Path):
    from recast.stages.swap_preview import preview_swap

    src = _face_image(tmp_path / "src.jpg")
    ref = _face_image(tmp_path / "ref.jpg")
    out = tmp_path / "preview.png"
    preview_swap(src, ref, out_path=out)
    assert out.is_file() and out.stat().st_size > 0


class _StubBackend:
    """Stand-in for InswapperBackend: marks a patch so the swap is visible."""

    def __init__(self, *, change: bool = True) -> None:
        self.change = change

    def load(self, cfg, device):  # noqa: ANN001, ANN202
        return None

    def prepare_source(self, path):  # noqa: ANN001, ANN202
        return None

    def replace_frame(self, frame, boxes, *args, **kwargs):  # noqa: ANN001, ANN202
        out = frame.copy()
        if self.change:
            out[10:20, 10:20] = 255
        return out

    def unload(self):
        return None


def test_preview_swap_uses_injected_backend(tmp_path: Path):
    from recast.stages.swap_preview import preview_swap

    src = _face_image(tmp_path / "src.jpg")
    ref = _face_image(tmp_path / "ref.jpg")
    result = preview_swap(src, ref, backend=_StubBackend())
    assert result["method"] == "inswapper"
    assert result["is_final"] is True


def test_backend_with_no_change_falls_back_to_overlay(tmp_path: Path):
    from recast.stages.swap_preview import NOT_FINAL_TEXT, preview_swap

    src = _face_image(tmp_path / "src.jpg")
    ref = _face_image(tmp_path / "ref.jpg")
    result = preview_swap(src, ref, backend=_StubBackend(change=False))
    assert result["method"] == "aligned_overlay"
    assert result["is_final"] is False
    assert NOT_FINAL_TEXT in result["note"]


# ----- UI：入口与对照弹窗 -----


def test_swap_preview_button_present_and_disabled_without_character(tmp_path: Path):
    from recast.desktop.library_window import LibraryWindow

    _app()
    svc = _service(tmp_path)
    window = LibraryWindow(svc)
    window._sync_tasks = True
    assert window.swap_preview_btn.text() == "替换预览"
    assert window.swap_preview_btn.isEnabled() is False


def test_swap_preview_button_enabled_with_character(tmp_path: Path):
    _svc, window, _folder = _window_with_faces(tmp_path)
    assert window.swap_preview_btn.isEnabled() is True


def test_swap_preview_opens_side_by_side_dialog(tmp_path: Path):
    from recast.stages.swap_preview import (
        BODY_PLACEHOLDER_TEXT,
        NOT_FINAL_TEXT,
    )

    _svc, window, _folder = _window_with_faces(tmp_path)
    window.swap_preview_btn.click()
    dialog = getattr(window, "_swap_preview_dialog", None)
    assert dialog is not None
    assert dialog.isVisible()
    assert "替换预览" in dialog.windowTitle()
    labels = _dialog_labels(window)
    assert any(text == "原图" for text in labels)
    assert any(text == "预览图" for text in labels)
    # Fallback honesty: the red warning and the body placeholder must be visible.
    assert any(NOT_FINAL_TEXT in text for text in labels)
    assert any(BODY_PLACEHOLDER_TEXT in text for text in labels)
    assert window._last_swap_preview["method"] == "aligned_overlay"
    dialog.close()


def test_swap_preview_prefers_selected_sample(tmp_path: Path):
    _svc, window, folder = _window_with_faces(tmp_path)
    # b.jpg is the second sample and NOT the assembled reference (a.jpg):
    # selecting it must make it the source frame.
    window._selected_samples = [str(folder / "face" / "b.jpg")]
    window.swap_preview_btn.click()
    assert window._last_swap_preview["source_path"] == str(folder / "face" / "b.jpg")
    window._swap_preview_dialog.close()
