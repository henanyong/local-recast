"""UI tests for the review-queue rework (B line): multi-select + batch ops,
thumbnail target list, kind/score/group filters, collapsible groups,
one-click confirm of high-confidence clusters.

Synthetic images + stub service data only. No real video, no detection model.
"""

import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _checker(path: Path, size: int = 140) -> Path:
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if (x // 7 + y // 7) % 2 else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)
    return path


def _solid(path: Path, size: int = 140) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (200, 30, 30)).save(path)
    return path


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


def _add_character(svc, name: str, *, confirmed: int = 0, samples: list[Path] | None = None):
    ch = svc.store.insert(
        "characters", name=name, display_name=name, confirmed=confirmed,
        authorized=0, cluster_key=f"test:{name}",
    )
    for p in samples or []:
        svc.store.insert(
            "character_assets", character_id=ch["id"], kind="face",
            path=str(p), complete=1, notes="聚类样本",
        )
    return svc.store.get_character(ch["id"])


def _library(svc):
    from recast.desktop.library_window import LibraryWindow

    _app()
    return LibraryWindow(svc)


def _data_rows(window):
    """Review rows that are real queue items (group headers excluded)."""
    out = []
    for row in range(window.review_list.count()):
        item = window.review_list.item(row)
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and not data.get("review_header"):
            out.append(item)
    return out


def _headers(window):
    out = []
    for row in range(window.review_list.count()):
        item = window.review_list.item(row)
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and data.get("review_header"):
            out.append(item)
    return out


def _select_all_data(window):
    for item in _data_rows(window):
        item.setSelected(True)


def _yes(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )


# ----- 多选 + 批量操作 -----

def test_review_list_is_extended_selection(tmp_path: Path, svc):
    from PySide6.QtWidgets import QAbstractItemView

    _add_character(svc, "人物1", confirmed=0)
    _add_character(svc, "人物2", confirmed=0)
    window = _library(svc)
    assert window.review_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert not window.review_batch_assign_btn.isEnabled()
    _select_all_data(window)
    assert window.review_batch_assign_btn.isEnabled()
    assert window.review_batch_create_btn.isEnabled()
    assert window.review_batch_discard_btn.isEnabled()
    window.close()


def test_batch_assign_merges_selected_clusters(tmp_path: Path, svc, monkeypatch):
    _yes(monkeypatch)
    c1 = _add_character(svc, "人物1", confirmed=0)
    c2 = _add_character(svc, "人物2", confirmed=0)
    target = _add_character(svc, "温念", confirmed=1)
    window = _library(svc)
    assert len(_data_rows(window)) == 2
    _select_all_data(window)
    window.review_batch_assign_btn.click()
    assert window.review_list.count() == 0
    assert "成功 2 项" in window.status_label.text()
    assert svc.store.get_character(c1["id"]) is None
    assert svc.store.get_character(c2["id"]) is None
    assert svc.store.get_character(target["id"]) is not None
    window.close()


def test_batch_discard_clears_selected_rows(tmp_path: Path, svc, monkeypatch):
    _yes(monkeypatch)
    _add_character(svc, "人物1", confirmed=0)
    _add_character(svc, "人物2", confirmed=0)
    window = _library(svc)
    _select_all_data(window)
    window.review_batch_discard_btn.click()
    assert window.review_list.count() == 0
    assert window.finish_review_btn.isEnabled()
    window.close()


def test_batch_create_names_cluster_and_sample(tmp_path: Path, svc, monkeypatch):
    from recast.desktop import library_window

    _yes(monkeypatch)
    _add_character(svc, "人物甲", confirmed=0)
    bad = _solid(tmp_path / "works" / "bad.png")
    _add_character(svc, "温念", confirmed=1, samples=[bad])
    monkeypatch.setattr(
        library_window.QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: ("新人物", True)),
    )
    window = _library(svc)
    assert len(_data_rows(window)) == 2
    _select_all_data(window)
    window.review_batch_create_btn.click()
    assert window.review_list.count() == 0
    names = [c.get("display_name") for c in svc.characters()]
    assert "人物甲" in names  # 聚类用自己的名字确认
    assert "新人物1" in names  # 样本用前缀+序号
    window.close()


# ----- 目标列表：带缩略图，点谁移给谁 -----

def test_target_list_rows_and_click_assign(tmp_path: Path, svc):
    _add_character(svc, "人物1", confirmed=0)
    t1 = _add_character(svc, "温念", confirmed=1, samples=[_checker(tmp_path / "works" / "a.png")])
    t2 = _add_character(svc, "陈秋微", confirmed=1)
    window = _library(svc)
    assert window.review_target_list.count() == 3
    texts = [window.review_target_list.item(r).text() for r in range(3)]
    assert any("温念" in t and "已确认" in t for t in texts)
    assert any("人物1" in t and "未确认" in t for t in texts)
    # 温念行带脸缩略图图标。
    assert not window.review_target_list.item(1).icon().isNull()
    # 点陈秋微那行，再分：聚类并进陈秋微。
    window.review_list.setCurrentRow(0)
    for row in range(3):
        if window.review_target_list.item(row).data(Qt.ItemDataRole.UserRole) == t2["id"]:
            window.review_target_list.setCurrentRow(row)
    window.review_assign_btn.click()
    assert window.review_list.count() == 0
    assert "陈秋微" in window.status_label.text()
    assert svc.store.get_character(t1["id"]) is not None
    window.close()


# ----- 筛选 + 分组折叠 -----

def test_kind_and_score_filters(tmp_path: Path, svc):
    low = _solid(tmp_path / "works" / "low.png", 40)   # 分数约 0.13（<0.3）
    mid = _solid(tmp_path / "works" / "mid.png", 140)  # 分数约 0.4（0.3–0.5）
    _add_character(svc, "人物1", confirmed=0)
    _add_character(svc, "温念", confirmed=1, samples=[low, mid])
    window = _library(svc)
    assert len(_data_rows(window)) == 3
    # 只看样本：聚类行被筛掉。
    window.review_kind_filter.setCurrentIndex(window.review_kind_filter.findData("sample"))
    assert len(_data_rows(window)) == 2
    assert "已筛出 2 项" in window.review_count_label.text()
    # 分数 <0.3：只剩低分样本（聚类行没有分数，分数筛不过滤它，所以还在）。
    window.review_kind_filter.setCurrentIndex(window.review_kind_filter.findData(None))
    window.review_score_filter.setCurrentIndex(window.review_score_filter.findData("low"))
    assert [i.data(Qt.ItemDataRole.UserRole)["score"] for i in _data_rows(window)] == [None, pytest.approx(0.133, abs=0.02)]
    assert "已筛出 2 项" in window.review_count_label.text()  # 聚类行 + 低分样本
    window.close()


def test_group_filter_shows_one_group(tmp_path: Path, svc):
    import json

    _add_character(svc, "人物1", confirmed=0)
    _add_character(svc, "温念", confirmed=1, samples=[_solid(tmp_path / "works" / "bad.png")])
    window = _library(svc)
    assert len(_headers(window)) == 2  # 聚类组 + 样本组
    key = json.dumps(["sample", "温念"], ensure_ascii=False)
    idx = window.review_group_filter.findData(key)
    assert idx >= 0
    window.review_group_filter.setCurrentIndex(idx)
    assert len(_data_rows(window)) == 1
    assert len(_headers(window)) == 0  # 只剩一组，不画组头
    # 下拉框里别的组还在（选项来自类型/分数筛后的列表，不是分组筛后的）。
    assert window.review_group_filter.count() == 3
    window.close()


def test_many_items_collapse_all_but_first_group(tmp_path: Path, svc):
    for i in range(25):
        _add_character(svc, f"人物{i:02d}", confirmed=0)
    window = _library(svc)
    assert len(_data_rows(window)) == 25
    assert len(_headers(window)) == 25
    assert _headers(window)[0].text().startswith("▼")
    assert _headers(window)[1].text().startswith("▶")
    assert not _data_rows(window)[0].isHidden()
    assert _data_rows(window)[1].isHidden()
    # 点第二组组头：展开，组头变 ▼，行可见。
    window._on_review_item_clicked(_headers(window)[1])
    assert _headers(window)[1].text().startswith("▼")
    assert not _data_rows(window)[1].isHidden()
    window.close()


# ----- 一键确认高置信度 -----

def test_confirm_high_only_clear_clusters(tmp_path: Path, svc, monkeypatch):
    _yes(monkeypatch)
    clear = _checker(tmp_path / "works" / "clear.png")
    blurry = _solid(tmp_path / "works" / "blurry.png")
    c1 = _add_character(svc, "人物甲", confirmed=0, samples=[clear])
    c2 = _add_character(svc, "人物乙", confirmed=0, samples=[blurry])
    c3 = _add_character(svc, "人物丙", confirmed=0)
    window = _library(svc)
    assert "（1 项）" in window.review_confirm_high_btn.text()
    window.review_confirm_high_btn.click()
    assert svc.store.get_character(c1["id"])["confirmed"] == 1
    assert svc.store.get_character(c2["id"])["confirmed"] == 0
    assert svc.store.get_character(c3["id"])["confirmed"] == 0
    # 剩：乙的聚类行、乙的低分样本行、丙的聚类行。
    assert len(_data_rows(window)) == 3
    assert not window.review_confirm_high_btn.isEnabled()
    window.close()


# ----- 遮挡 / 截断修复 + 空占位 -----

def _show(window):
    from PySide6.QtWidgets import QApplication

    window.show()
    QApplication.processEvents()


def test_inspector_group4_fully_visible_at_1280x800(tmp_path: Path, svc):
    from PySide6.QtWidgets import QGroupBox, QPushButton, QScrollArea

    _add_character(svc, "温念", confirmed=1)
    window = _library(svc)
    _show(window)
    assert window.width() == 1280 and window.height() == 800
    inspector = window.findChild(QScrollArea, "inspectorPane")
    assert inspector.verticalScrollBar().maximum() == 0  # 不用滚就全看得见
    g4 = next(g for g in window.findChildren(QGroupBox) if g.title() == "4 声音与服装")
    viewport_h = inspector.viewport().height()
    for btn in g4.findChildren(QPushButton):
        top_left = btn.mapTo(inspector.viewport(), btn.rect().topLeft())
        assert 0 <= top_left.y(), btn.text()
        assert top_left.y() + btn.height() <= viewport_h, btn.text()
    window.close()


def test_hint_toggle_collapses_by_default(tmp_path: Path, svc):
    _add_character(svc, "温念", confirmed=1)
    window = _library(svc)
    assert not window.hint_label.isVisible() or window.hint_label.isHidden()
    assert window.hint_label.isHidden()
    window.hint_toggle.click()
    assert not window.hint_label.isHidden()
    assert "▾" in window.hint_toggle.text()
    window.close()


def test_status_bar_scrolls_long_text(tmp_path: Path, svc):
    from PySide6.QtWidgets import QApplication, QScrollArea

    _add_character(svc, "温念", confirmed=1)
    window = _library(svc)
    _show(window)
    scroll = window.findChild(QScrollArea, "statusScroll")
    assert scroll is not None
    assert scroll.widget() is window.status_label
    window.status_label.setText("这是一段很长很长的状态文字，" * 30)
    QApplication.processEvents()
    assert window.status_label.text().startswith("这是一段很长")
    assert window.status_label.heightForWidth(window.status_label.width()) > 48
    assert scroll.verticalScrollBar().maximum() > 0  # 长文滚得动，不是截掉
    window.close()


def test_character_list_row_tooltip_shows_full_text(tmp_path: Path, svc):
    long_name = "温念非常非常非常非常非常非常非常长的一个名字"
    _add_character(svc, long_name, confirmed=0)
    window = _library(svc)
    item = window.list.item(0)
    assert long_name in item.text()
    assert item.toolTip() == item.text()  # 悬停看全文
    window.close()


def test_assembly_empty_placeholders_with_folder(tmp_path: Path, svc):
    _add_character(svc, "温念", confirmed=1)
    svc.ensure_character_folder("温念")
    window = _library(svc)
    window.list.setCurrentRow(0)
    window._on_select(window.list.currentItem())
    assert window.assembly_body_list.isHidden()
    assert not window.assembly_body_empty.isHidden()
    assert "没有 3D 模型" in window.assembly_body_empty.text()
    assert "生成 3D 模型" in window.assembly_body_empty.text()
    assert window.assembly_costume_list.isHidden()
    assert "衣柜是空的" in window.assembly_costume_empty.text()
    assert window.assembly_face_list.isHidden()
    assert "没有人脸样本" in window.assembly_face_empty.text()
    window.close()


def test_detail_placeholder_when_no_images(tmp_path: Path, svc):
    from PySide6.QtWidgets import QLabel

    _add_character(svc, "温念", confirmed=1)
    svc.ensure_character_folder("温念")
    window = _library(svc)
    window.list.setCurrentRow(0)
    window._on_select(window.list.currentItem())
    texts = [lab.text() for lab in window.detail_images.findChildren(QLabel)]
    assert any("还没有服装截图或抠像" in t for t in texts)
    window.close()


def test_select_all_samples_toggles(tmp_path: Path, svc):
    from PySide6.QtWidgets import QToolButton

    a = _checker(tmp_path / "works" / "a.png")
    b = _checker(tmp_path / "works" / "b.png")
    _add_character(svc, "温念", confirmed=1, samples=[a, b])
    window = _library(svc)
    buttons = window.samples.findChildren(QToolButton)
    assert len(buttons) == 2
    assert not window.correct_move_btn.isEnabled()
    window.select_all_samples_btn.click()
    assert all(btn.isChecked() for btn in buttons)
    assert window.correct_move_btn.isEnabled()
    window.select_all_samples_btn.click()
    assert not any(btn.isChecked() for btn in buttons)
    assert not window.correct_move_btn.isEnabled()
    window.close()
