"""UI tests for the step guide, review queue, correction and main-window rework.

Synthetic images + stub service data only: a sharp checkerboard scores high,
a solid colour scores low. No real video, no detection model.
"""

import os
from pathlib import Path

import pytest
from PIL import Image

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


def _main(svc):
    from recast.desktop.main_window import MainWindow

    _app()
    window = MainWindow(svc)
    # 停掉 400ms 刷新定时器：测试里窗口 close 后对象仍活着，定时器继续
    # 刷会在后续测试的 processEvents() 里触发刷新路径（曾因此挂住全套）。
    window.timer.stop()
    return window


# ----- 人物库：人工审核队列 -----

def test_review_queue_lists_unconfirmed_cluster(tmp_path: Path, svc):
    _add_character(svc, "人物1", confirmed=0)
    _add_character(svc, "温念", confirmed=1, samples=[_checker(tmp_path / "works" / "good.png")])
    window = _library(svc)
    assert window.review_list.count() == 1
    assert "聚类" in window.review_list.item(0).text()
    assert "待审核：1 项" in window.review_count_label.text()
    assert not window.finish_review_btn.isEnabled()
    window.close()


def test_review_assign_cluster_clears_row(tmp_path: Path, svc):
    from PySide6.QtCore import Qt

    _add_character(svc, "人物1", confirmed=0)
    target = _add_character(svc, "温念", confirmed=1)
    window = _library(svc)
    window.review_list.setCurrentRow(0)
    # 目标人物在目标列表里点选（带缩略图的行），不再是小下拉框。
    for row in range(window.review_target_list.count()):
        if window.review_target_list.item(row).data(Qt.ItemDataRole.UserRole) == target["id"]:
            window.review_target_list.setCurrentRow(row)
            break
    window.review_assign_btn.click()
    assert window.review_list.count() == 0
    assert window.finish_review_btn.isEnabled()
    assert svc.store.get_character(target["id"]) is not None
    window.close()


def test_review_discard_cluster_clears_row(tmp_path: Path, svc, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    _add_character(svc, "人物1", confirmed=0)
    window = _library(svc)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    window.review_list.setCurrentRow(0)
    window.review_discard_btn.click()
    assert window.review_list.count() == 0
    assert window.finish_review_btn.isEnabled()
    window.close()


def test_low_score_sample_in_queue(tmp_path: Path, svc):
    bad = _solid(tmp_path / "works" / "bad.png")
    _add_character(svc, "温念", confirmed=1, samples=[bad])
    window = _library(svc)
    assert window.review_list.count() == 1
    assert "样本" in window.review_list.item(0).text()
    assert "分数" in window.review_list.item(0).text()
    window.close()


def test_prev_next_switch_character(tmp_path: Path, svc):
    _add_character(svc, "人物1", confirmed=0)
    _add_character(svc, "人物2", confirmed=0)
    window = _library(svc)
    window.list.setCurrentRow(0)
    window.next_btn.click()
    assert window.list.currentRow() == 1
    window.next_btn.click()  # wraps around
    assert window.list.currentRow() == 0
    window.prev_btn.click()
    assert window.list.currentRow() == 1
    window.close()


# ----- 人物库：纠错 + 审核可见性 -----

def test_sample_check_enables_correction_and_remove_returns_to_queue(tmp_path: Path, svc):
    from PySide6.QtWidgets import QToolButton

    a = _checker(tmp_path / "works" / "a.png")
    b = _checker(tmp_path / "works" / "b.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[a, b])
    window = _library(svc)
    # 只数样本区的打勾按钮：搜索框的内置清除按钮也是 QToolButton，
    # 全窗口数会把它算进来（P2 搜索框，setClearButtonEnabled(True)）。
    buttons = window.samples.findChildren(QToolButton)
    assert len(buttons) == 2
    assert not window.correct_move_btn.isEnabled()
    buttons[0].click()
    assert window.correct_move_btn.isEnabled()
    assert window.correct_remove_btn.isEnabled()
    assert window.correct_split_btn.isEnabled()
    window.correct_remove_btn.click()
    remaining = [x for x in svc.store.get_character(src["id"])["assets"] if x["kind"] == "face"]
    assert len(remaining) == 1
    # 移出的样本回到待审核清单（review_trash/ 根目录进队列）。
    assert window.review_list.count() == 1
    assert "移出" in window.review_list.item(0).text()
    window.close()


def test_audit_label_shows_no_model_no_cutout(tmp_path: Path, svc):
    _add_character(svc, "温念", confirmed=1, samples=[_checker(tmp_path / "works" / "good.png")])
    svc.ensure_character_folder("温念")
    window = _library(svc)
    assert "无模型" in window.audit_label.text()
    assert "未抠像" in window.audit_label.text()
    window.close()


def test_finish_review_closes_when_queue_empty(tmp_path: Path, svc):
    from PySide6.QtWidgets import QDialog

    _add_character(svc, "温念", confirmed=1, samples=[_checker(tmp_path / "works" / "good.png")])
    window = _library(svc)
    assert window.finish_review_btn.isEnabled()
    window.finish_review_btn.click()
    assert window.result() == QDialog.DialogCode.Accepted
    window.close()


# ----- 主窗口：步骤条、锁住、预览、布局 -----

def _paint_main(window, svc):
    """Drive one synchronous paint with a real snapshot (no event loop)."""
    window._paint(svc.snapshot(), full=True)


def test_guide_list_locks_replace_when_queue_nonempty(tmp_path: Path, svc):
    svc.store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "drama"), episode_count=1)
    _add_character(svc, "人物1", confirmed=0)
    window = _main(svc)
    _paint_main(window, svc)
    # 第 ⑤ 步声音与台词已并入引导：步骤条现在是 5 步（原断言 4 步已随规格过期）。
    assert window.guide_list.count() == 5
    assert "下一步" in window.next_btn.text()
    assert "③" in window.next_btn.text() or "审核" in window.next_btn.text()
    assert not window.stage_buttons["replace"].isEnabled()
    assert not window.stage_buttons["generate"].isEnabled()
    assert "先完成人物审核" in window.stage_buttons["replace"].toolTip()
    assert not window.reface_btn.isEnabled()
    assert not window.gen_btn.isEnabled()
    # 阶段清单带状态前缀：锁住的显示 🔒。
    assert window.stage_buttons["replace"].text().startswith("🔒")
    window.close()


def test_guide_unlocks_when_queue_empty(tmp_path: Path, svc):
    svc.store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "drama"), episode_count=1)
    _add_character(svc, "温念", confirmed=1, samples=[_checker(tmp_path / "works" / "good.png")])
    window = _main(svc)
    _paint_main(window, svc)
    assert window.stage_buttons["replace"].isEnabled()
    assert window.reface_btn.isEnabled()
    assert window.gen_btn.isEnabled()
    window.close()


def test_reface_blocked_when_locked(tmp_path: Path, svc, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    svc.store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "drama"), episode_count=1)
    _add_character(svc, "人物1", confirmed=0)
    window = _main(svc)
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    shown: list = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: shown.append(a[1] if len(a) > 1 else "")
    )
    window._drama_ids_checked = lambda: ["d1"]  # type: ignore[method-assign]
    window._reface()
    assert calls == []
    assert shown and "先完成人物审核" in shown[0]
    window.close()


def test_action_bar_groups_and_preview_placeholder(tmp_path: Path, svc):
    from PySide6.QtWidgets import QGroupBox, QSplitter

    svc.store.insert("dramas", origin_name="剧1", root_path=str(tmp_path / "d1"), episode_count=1)
    svc.store.insert("dramas", origin_name="剧2", root_path=str(tmp_path / "d2"), episode_count=2)
    window = _main(svc)
    _paint_main(window, svc)
    titles = [g.title() for g in window.findChildren(QGroupBox)]
    for want in ("执行", "人物", "生成", "导出", "工具"):
        assert want in titles
    assert "先选一部剧" in window.preview_image.text()
    assert "还没有选中的作品" in window.preview_text.text()
    # 2 行作品表不再撑满中间：表高被 cap 住，剩下的位置是预览区。
    assert window.table.maximumHeight() <= 340
    assert window.table.maximumHeight() >= 120
    assert window.overall_text.wordWrap() is True
    splitter = window.findChild(QSplitter, "mainSplitter")
    assert splitter is not None and splitter.count() == 3
    assert window.findChild(QSplitter, "rightSplitter") is not None
    window.close()
