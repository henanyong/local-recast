"""P5/B3 检查器组：按步骤折叠 + L2 改名 + 3D 组重排 + 禁用按钮 tooltip。

合成图片 + 桩服务数据，不跑真模型、不跑真视频。
"""

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QPushButton

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _face(path: Path) -> Path:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), (180, 140, 120))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 10, 48, 50), fill=(210, 170, 150))
    img.save(path)
    return path


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def svc(tmp_path: Path):
    from recast.core.service import RecastService

    service = RecastService(tmp_path / "db.db")
    service.update_settings(output_root=str(tmp_path / "out"), first_run_done=True)
    return service


def _add_character(svc, name: str, *, confirmed: int = 1):
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=confirmed,
        authorized=0,
        cluster_key=f"test:{name}",
    )


def _library(svc):
    from PySide6.QtWidgets import QApplication

    from recast.desktop.library_window import LibraryWindow

    _app()
    window = LibraryWindow(svc)
    window.resize(1280, 800)
    return window


def _show(window):
    from PySide6.QtWidgets import QApplication

    window.show()
    QApplication.processEvents()


def _select(window, row: int = 0) -> None:
    window.list.setCurrentRow(row)
    window._on_select(window.list.currentItem())


def _group(window, title: str):
    from PySide6.QtWidgets import QGroupBox

    for group in window.findChildren(QGroupBox):
        if group.title() == title:
            return group
    raise AssertionError(f"no group titled {title!r}")


def _expansion(window) -> dict[str, bool]:
    return {
        key: _group(window, title).isChecked()
        for key, title in (
            ("g1", "1 建文件夹"),
            ("g2", "2 抠像"),
            ("g3", "3 3D 模型"),
            ("g4", "4 声音与服装"),
            ("g5", "5 应用"),
        )
    }


def _folder_with(svc, name: str, *, cutout: bool = False, model: bool = False, voice: bool = False) -> Path:
    """建人物文件夹，并按需补抠像 / 真模型 / 声音，返回文件夹路径。"""
    from recast.core.characters import load_meta, save_meta

    folder = Path(svc.ensure_character_folder(name)["folder"])
    if cutout:
        _face(folder / "cutout" / "cut.png")
    if model:
        meta = load_meta(folder) or {}
        meta["model"] = {"source": "manual", "format": "glb", "path": "model/model.glb", "note": ""}
        save_meta(folder, meta)
        (folder / "model").mkdir(exist_ok=True)
        (folder / "model" / "model.glb").write_bytes(b"glTF")
    if voice:
        (folder / "voice").mkdir(exist_ok=True)
        (folder / "voice" / "a.wav").write_bytes(b"fake-wav")
    return folder


def _inspector_buttons(window):
    from PySide6.QtWidgets import QPushButton, QScrollArea

    inspector = window.findChild(QScrollArea, "inspectorPane")
    return inspector.widget().findChildren(QPushButton)


# ----- 1. 检查器按步骤折叠 -----


def test_no_selection_only_g1_expanded(svc):
    # 真·未选人：库里一个人都没有，reload 不会自动选行。
    window = _library(svc)
    assert window._current_id is None
    assert _expansion(window) == {"g1": True, "g2": False, "g3": False, "g4": False, "g5": False}
    assert not window.build_folder_btn.isHidden()
    for btn in _group(window, "4 声音与服装").findChildren(QPushButton):
        assert btn.isHidden(), btn.text()
    window.close()


def test_expansion_folder_only_is_g2(svc):
    _add_character(svc, "温念")
    _folder_with(svc, "温念")
    window = _library(svc)
    window.reload()
    _select(window)
    assert _expansion(window) == {"g1": False, "g2": True, "g3": False, "g4": False, "g5": False}
    assert not window.matte_btn.isHidden()
    assert window.gen3d_btn.isHidden()
    window.close()


def test_expansion_cutout_no_model_is_g3(svc):
    _add_character(svc, "温念")
    _folder_with(svc, "温念", cutout=True)
    window = _library(svc)
    window.reload()
    _select(window)
    assert _expansion(window) == {"g1": False, "g2": False, "g3": True, "g4": False, "g5": False}
    assert not window.gen3d_btn.isHidden()
    assert window.tag_btn.isHidden()
    window.close()


def test_expansion_model_no_voice_is_g4(svc):
    _add_character(svc, "温念")
    _folder_with(svc, "温念", cutout=True, model=True)
    window = _library(svc)
    window.reload()
    _select(window)
    assert _expansion(window) == {"g1": False, "g2": False, "g3": False, "g4": True, "g5": False}
    assert not window.tag_btn.isHidden()
    assert window.bind_btn.isHidden()
    window.close()


def test_expansion_imported_model_without_cutout_is_g4(svc):
    """边界：手动导入的真模型、没抠像时，下一步是声音（g4），不是回头抠像。"""
    _add_character(svc, "温念")
    _folder_with(svc, "温念", model=True)
    window = _library(svc)
    window.reload()
    _select(window)
    assert _expansion(window)["g4"] is True
    assert _expansion(window)["g2"] is False
    window.close()


def test_expansion_model_and_voice_is_g5(svc):
    _add_character(svc, "温念")
    _folder_with(svc, "温念", cutout=True, model=True, voice=True)
    window = _library(svc)
    window.reload()
    _select(window)
    assert _expansion(window) == {"g1": False, "g2": False, "g3": False, "g4": False, "g5": True}
    assert not window.bind_btn.isHidden()
    assert window.tag_btn.isHidden()
    window.close()


def test_manual_toggle_shows_children_and_refresh_resets(svc):
    """边界：用户能手动展开一组；下次刷新按管线状态复位。"""
    from PySide6.QtWidgets import QApplication

    _add_character(svc, "温念")
    window = _library(svc)
    g4 = _group(window, "4 声音与服装")
    assert g4.isChecked() is False
    g4.setChecked(True)
    QApplication.processEvents()
    assert not window.tag_btn.isHidden()
    window._refresh_action_states()
    assert g4.isChecked() is False
    assert window.tag_btn.isHidden()
    window.close()


def test_info_group_never_collapsible(svc):
    window = _library(svc)
    info = _group(window, "基本信息")
    assert info.isCheckable() is False  # 不可折叠的组 isChecked() 恒为 False，不断言它
    save_btn = next(
        b for b in info.findChildren(QPushButton) if b.text() == "保存确认"
    )
    assert not save_btn.isHidden()
    window.close()


def test_inspector_group4_fully_visible_at_1280x720(svc):
    """1280×720 等价测试：有真模型、没声音的状态下 g4 展开，第 4 组首屏可见。"""
    from PySide6.QtWidgets import QGroupBox, QPushButton, QScrollArea

    _add_character(svc, "温念")
    _folder_with(svc, "温念", cutout=True, model=True)
    window = _library(svc)
    window.reload()
    _select(window)
    assert _group(window, "4 声音与服装").isChecked() is True
    window.resize(1280, 720)
    _show(window)
    assert window.width() == 1280 and window.height() == 720
    inspector = window.findChild(QScrollArea, "inspectorPane")
    assert inspector.verticalScrollBar().maximum() == 0  # 不用滚就全看得见
    g4 = next(g for g in window.findChildren(QGroupBox) if g.title() == "4 声音与服装")
    viewport_h = inspector.viewport().height()
    for btn in g4.findChildren(QPushButton):
        assert btn.isVisible(), btn.text()
        top_left = btn.mapTo(inspector.viewport(), btn.rect().topLeft())
        assert 0 <= top_left.y(), btn.text()
        assert top_left.y() + btn.height() <= viewport_h, btn.text()
    window.close()


# ----- 2. L2 改名（只改 UI 文案，不动已存资产 label）-----


def test_l2_buttons_renamed(svc):
    from PySide6.QtWidgets import QPushButton

    window = _library(svc)
    texts = [b.text() for b in window.findChildren(QPushButton)]
    assert "生成头像卡片（占位）" in texts
    assert "删除头像卡片" in texts
    assert "清理未使用的头像卡片" in texts
    assert not any("3D头像资产" in t or "3D头像" in t for t in texts)
    window.close()


def test_hint_text_renamed(svc):
    window = _library(svc)
    text = window.hint_label.text()
    assert "头像卡片（占位）" in text
    assert "3D头像资产" not in text
    assert "3D头像" not in text
    window.close()


# ----- 3. 3D 组重排 -----


def _g3_rows(window):
    from PySide6.QtWidgets import QLabel, QPushButton

    g3 = _group(window, "3 3D 模型")
    layout = g3.layout()
    rows = []
    current_title = None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget()
        if isinstance(widget, QLabel):
            current_title = widget.text()
        elif item.layout() is not None:
            texts = [
                w.text()
                for w in (
                    item.layout().itemAt(j).widget()
                    for j in range(item.layout().count())
                )
                if isinstance(w, QPushButton)
            ]
            rows.append((current_title, texts))
    return rows


def test_g3_rows_grouped_by_kind(svc):
    window = _library(svc)
    assert _g3_rows(window) == [
        ("真模型", ["生成 3D 模型", "导入 3D 模型", "切换换装变体…"]),
        ("占位资产", ["生成头像卡片（占位）", "删除头像卡片", "清理未使用的头像卡片"]),
        ("变体 / 设定板", ["生成换装变体…", "生成设定板"]),
    ]
    window.close()


def test_g3_row_buttons_equal_size_and_full_name_tooltip(svc):
    """边界：每行内按钮等宽等高；最长的文案 tooltip 里有全称。"""
    from PySide6.QtWidgets import QApplication, QPushButton

    window = _library(svc)
    _group(window, "3 3D 模型").setChecked(True)
    _show(window)
    QApplication.processEvents()
    rows = _g3_rows(window)
    widths_row1 = [
        next(b for b in _group(window, "3 3D 模型").findChildren(QPushButton) if b.text() == t).width()
        for t in rows[0][1]
    ]
    assert max(widths_row1) - min(widths_row1) <= 1
    heights_row1 = [
        next(b for b in _group(window, "3 3D 模型").findChildren(QPushButton) if b.text() == t).height()
        for t in rows[0][1]
    ]
    assert max(heights_row1) - min(heights_row1) == 0
    # 变体行只有一个按钮，和上面两行的按钮同宽（行内少一次 3px 间距，
    # 布局取整会差 1–2px，肉眼不可见，容差放宽到 3px）。
    variant_btn = next(
        b for b in _group(window, "3 3D 模型").findChildren(QPushButton)
        if b.text() == "生成换装变体…"
    )
    assert abs(variant_btn.width() - widths_row1[0]) <= 3
    clean_btn = next(
        b for b in _group(window, "3 3D 模型").findChildren(QPushButton)
        if b.text() == "清理未使用的头像卡片"
    )
    assert "清理未使用的头像卡片" in clean_btn.toolTip()
    switch_btn = next(
        b for b in _group(window, "3 3D 模型").findChildren(QPushButton)
        if b.text() == "切换换装变体…"
    )
    assert "切换换装变体…" in switch_btn.toolTip()
    window.close()


# ----- 4. 禁用按钮 tooltip 大扫荡 -----


def test_no_selection_disabled_inspector_buttons_have_tooltips(svc):
    """无选中人物时，检查器里每一个 disabled 按钮的 toolTip() 非空。"""
    window = _library(svc)  # 空库：真·未选人
    assert window._current_id is None
    disabled = [b for b in _inspector_buttons(window) if not b.isEnabled()]
    assert disabled, "无选中人物时检查器里本该有灰掉的按钮"
    for btn in disabled:
        assert (btn.toolTip() or "").strip(), btn.text()
    window.close()


def test_folder_no_cutout_disabled_buttons_name_the_gap(svc):
    """边界：已建文件夹时 [建人物文件夹] 灰掉但写清已建好；没抠像时 [生成 3D 模型] 写清先抠像。"""
    _add_character(svc, "温念")
    _folder_with(svc, "温念")
    window = _library(svc)
    window.reload()
    _select(window)
    assert not window.build_folder_btn.isEnabled()
    assert "已经建好" in window.build_folder_btn.toolTip()
    assert not window.gen3d_btn.isEnabled()
    assert "先抠像" in window.gen3d_btn.toolTip()
    assert not window.switch_variant_btn.isEnabled()
    assert "还没有换装变体" in window.switch_variant_btn.toolTip()
    assert window.matte_btn.isEnabled()
    window.close()


def test_review_buttons_disabled_have_tooltips(svc):
    """审核队列空时，单项按钮灰掉且写清先选一行；一键确认灰掉也有 tooltip。"""
    _add_character(svc, "温念", confirmed=1)
    window = _library(svc)
    for btn in (window.review_assign_btn, window.review_create_btn, window.review_discard_btn):
        assert not btn.isEnabled(), btn.text()
        assert "先在人工审核队列里选一行" in btn.toolTip(), btn.text()
    assert not window.review_confirm_high_btn.isEnabled()
    assert (window.review_confirm_high_btn.toolTip() or "").strip()
    window.close()


def test_no_selection_review_and_delete_buttons_have_tooltips(svc):
    """边界：无选中人物时，[删除人物] 与审核队列按钮灰掉也有 tooltip。"""
    window = _library(svc)  # 空库：真·未选人
    assert window._current_id is None
    assert not window.delete_char_btn.isEnabled()
    assert "先在左侧选一个人" in window.delete_char_btn.toolTip()
    assert not window.review_assign_btn.isEnabled()
    assert (window.review_assign_btn.toolTip() or "").strip()
    window.close()


# ----- 角色：每组最多 1 个 primary -----


def test_button_roles_primary_secondary_danger(svc):
    _add_character(svc, "温念")
    window = _library(svc)
    inspector_primaries = [
        b.text() for b in _inspector_buttons(window) if b.objectName() == "primary"
    ]
    assert inspector_primaries == ["保存确认"]
    for title in ("1 建文件夹", "2 抠像", "3 3D 模型", "4 声音与服装", "5 应用", "管理"):
        primaries = [
            b.text()
            for b in _group(window, title).findChildren(QPushButton)
            if b.objectName() == "primary"
        ]
        assert len(primaries) <= 1, (title, primaries)
    assert window.finish_review_btn.objectName() == "primary"
    assert window.review_confirm_high_btn.objectName() == "secondary"
    assert window.l2_del_btn.objectName() == "danger"
    assert window.delete_char_btn.objectName() == "danger"
    assert window.review_discard_btn.objectName() == "danger"
    window.close()
