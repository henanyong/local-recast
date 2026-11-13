"""Layout tests for the reworked LibraryWindow (three-pane studio layout).

Covers: QSplitter with three panes, inspector groups with step titles,
right pane wrapped in a scroll area, step indicator tracking progress,
3D model card, and the main-window library entry point.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _face(path: Path) -> Path:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), (180, 140, 120))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 10, 48, 50), fill=(210, 170, 150))
    img.save(path)
    return path


def _service(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _unconfirmed(svc, name: str) -> dict:
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=0,
        authorized=0,
        cluster_key=f"cluster:{name}",
    )


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window(tmp_path: Path):
    from recast.desktop.library_window import LibraryWindow

    _app()
    svc = _service(tmp_path)
    window = LibraryWindow(svc)
    window._sync_tasks = True
    return svc, window


def _select(window, row: int = 0) -> None:
    window.list.setCurrentRow(row)
    window._on_select(window.list.currentItem())


def _group(window, title: str):
    from PySide6.QtWidgets import QGroupBox

    for group in window.findChildren(QGroupBox):
        if group.title() == title:
            return group
    raise AssertionError(f"no group titled {title!r}")


def _group_button_texts(window, title: str) -> list[str]:
    from PySide6.QtWidgets import QPushButton

    return [b.text() for b in _group(window, title).findChildren(QPushButton)]


def test_three_pane_splitter_layout(tmp_path: Path):
    from PySide6.QtWidgets import QSplitter

    _svc, window = _window(tmp_path)
    assert window.width() == 1280
    assert window.height() == 800
    splitter = window.findChild(QSplitter, "librarySplitter")
    assert splitter is not None
    assert splitter.count() == 3
    names = [splitter.widget(i).objectName() for i in range(3)]
    assert names == ["pipelinePane", "stagePane", "inspectorPane"]
    window.close()


def test_inspector_groups_hold_step_buttons(tmp_path: Path):
    _svc, window = _window(tmp_path)
    assert _group_button_texts(window, "1 建文件夹") == ["建人物文件夹"]
    assert _group_button_texts(window, "2 抠像") == ["抠像"]
    assert _group_button_texts(window, "3 3D 模型") == [
        "生成 3D 模型",
        "导入 3D 模型",
        "切换换装变体…",
        "生成头像卡片（占位）",
        "删除头像卡片",
        "清理未使用的头像卡片",
        "生成换装变体…",
        "生成设定板",
    ]
    assert _group_button_texts(window, "4 声音与服装") == [
        "衣服打标签…",
        "试听声音",
        "调节音色",
        "编辑对话",
    ]
    assert _group_button_texts(window, "5 应用") == ["绑定服装到模型", "换装预览…"]
    assert "删除人物" in _group_button_texts(window, "管理")
    assert "清理失效样本" in _group_button_texts(window, "管理")
    window.close()


def test_right_pane_is_scrollable(tmp_path: Path):
    from PySide6.QtWidgets import QScrollArea

    _svc, window = _window(tmp_path)
    inspector = window.findChild(QScrollArea, "inspectorPane")
    assert inspector is not None
    assert inspector.widgetResizable()
    stage = window.findChild(QScrollArea, "stagePane")
    assert stage is not None
    assert stage.widgetResizable()
    window.close()


def test_step_indicator_tracks_progress(tmp_path: Path, monkeypatch):
    from recast.desktop import library_window

    svc, window = _window(tmp_path)
    assert window.step_labels["import"].text().startswith("○")
    assert window.step_labels["define"].text().startswith("○")

    video = tmp_path / "ep1.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        library_window.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(video), "")),
    )
    window.import_btn.click()
    assert window.step_labels["import"].text().startswith("✓")

    _unconfirmed(svc, "温念")
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window)
    assert window.step_labels["define"].text().startswith("✓")
    assert window.step_labels["folder"].text().startswith("◐")
    assert window.step_labels["model"].text().startswith("○")

    _face(Path(svc.settings.output_root) / "characters" / "温念" / "cutout" / "cut.png")
    window.reload()
    _select(window)
    assert window.step_labels["folder"].text().startswith("✓")
    window.close()


def test_model_card_shows_3d_status(tmp_path: Path):
    from recast.core.characters import load_meta, save_meta

    svc, window = _window(tmp_path)
    _unconfirmed(svc, "温念")
    folder = Path(svc.ensure_character_folder("温念")["folder"])
    meta = load_meta(folder) or {}
    meta["model"] = {"source": "manual", "format": "glb", "path": "model/model.glb", "note": ""}
    save_meta(folder, meta)
    (folder / "model").mkdir(exist_ok=True)
    (folder / "model" / "model.glb").write_bytes(b"glTF")
    window.reload()
    _select(window)
    assert "manual" in window.model_card_label.text()
    assert "glb" in window.model_card_label.text()
    window.close()


def test_main_window_library_entry_is_secondary(tmp_path: Path):
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.main_window import MainWindow

    _app()
    svc = _service(tmp_path)
    svc.update_settings(first_run_done=True)
    window = MainWindow(svc)
    window.timer.stop()  # 停掉 400ms 定时器，避免泄漏的窗口在后续测试里继续刷新。
    by_text = {b.text(): b for b in window.findChildren(QPushButton)}
    assert "人物库" in by_text
    # P5/B3：主窗口唯一 primary 是 [一键成片]；[人物库] 已降级为 secondary。
    assert by_text["人物库"].objectName() == "secondary"
    assert by_text["人物库"].toolTip()
    window.close()
