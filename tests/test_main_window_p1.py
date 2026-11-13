"""Main-window P1 batch: review progress + auto-jump, entry cleanup,
review-gate auth pre-check, per-shot enhance toggle, N8 row dots +
batch skip-blocked.

Only synthetic data + stub service calls: no real video, no TTS engines, no
network. Follows the fixture pattern of tests/test_main_window_p0.py.
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
    return MainWindow(svc)


def _paint_main(window, svc):
    """Drive one synchronous paint with a real snapshot (no event loop)."""
    window._paint(svc.snapshot(), full=True)


def _add_drama(svc, name: str = "剧", *, entry_type: str = "import") -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, root_path=f"/tmp/{name}", episode_count=1,
        entry_type=entry_type,
    )


def _add_task(svc, drama_id, *, status: str = "待执行", workdir=None):
    return svc.store.insert(
        "tasks", drama_id=drama_id, episode_name="第1集",
        video_path="/tmp/ep1.mp4", status=status, workdir=workdir,
    )


def _add_character(svc, name: str, *, confirmed: int = 0, authorized: int = 0, auth_type: str | None = None):
    fields: dict = {
        "name": name, "display_name": name, "confirmed": confirmed,
        "authorized": authorized, "cluster_key": f"test:{name}",
    }
    if auth_type:
        fields["auth_type"] = auth_type
    return svc.store.insert("characters", **fields)


def _real_face(tmp_path: Path, name: str = "face.jpg") -> Path:
    """A real readable JPEG (200x200 checkerboard): scores ~1.0, so the
    review queue does not treat it as a low-score sample."""
    from PIL import Image

    img = Image.new("L", (200, 200))
    px = img.load()
    for y in range(200):
        for x in range(200):
            px[x, y] = 255 if (x // 10 + y // 10) % 2 == 0 else 0
    path = tmp_path / name
    img.convert("RGB").save(path, "JPEG")
    return path


def _add_face_asset(svc, character_id: str, path: str):
    return svc.store.insert(
        "character_assets", character_id=character_id, kind="face", path=path,
    )


def _add_identity_checkpoint(svc, task_id: str):
    svc.store.insert(
        "checkpoints", task_id=task_id, step="identity",
        input_hashes="[]", output_paths="[]",
    )


class _FakeBox:
    """Stand-in for QMessageBox with addButton/exec/clickedButton.

    Set ``_FakeBox.click`` (class attribute) to the button text exec()
    should "click". Class-level ``shown`` records every box created
    while the fake is installed.
    """

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

    @staticmethod
    def information(*args, **kwargs):
        return None

    @staticmethod
    def warning(*args, **kwargs):
        return None

    @staticmethod
    def question(*args, **kwargs):
        return None


# ----- 第 1 项：审核进度条 + 自动跳转 -----


def test_review_progress_shows_counts(svc):
    _add_drama(svc, "剧")
    _add_character(svc, "温念", confirmed=1)
    _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    assert "已确认 1" in window.review_progress_label.text()
    assert "待审核 1" in window.review_progress_label.text()
    assert "低分 0" in window.review_progress_label.text()
    assert not window.review_progress.isHidden()
    assert window.review_progress.value() == 50
    window.close()


def test_review_progress_empty_library(svc):
    _add_drama(svc, "剧")
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    assert "还没有人物" in window.review_progress_label.text()
    assert window.review_progress.isHidden()
    window.close()


def test_review_auto_jump_prompts_and_jumps(svc, monkeypatch):
    _add_drama(svc, "剧")
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    window.show()  # 自动跳转只在主窗口可见时提示（隐藏窗口不弹框，见下个测试）。
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    # 队列清空：确认最后一个人，再刷一次就该提示并跳转。
    svc.store.update("characters", ch["id"], confirmed=1)
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.shown.clear()
    monkeypatch.setattr(_FakeBox, "click", "去第 ④ 步")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown, "队列清空后应弹出自动跳转提示"
    assert any("审核已完成" in getattr(b, "title", "") for b in _FakeBox.shown)
    assert jumps == ["replace"]
    assert window._review_was_nonempty is False
    window.close()


def test_review_auto_jump_skipped_while_oneclick(svc, monkeypatch):
    _add_drama(svc, "剧")
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    window.show()  # 可见窗口才会走到一键成片的屏蔽分支（否则在可见性检查就返回了）。
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.store.update("characters", ch["id"], confirmed=1)
    window._oneclick = {"drama_id": "x", "phase": "review", "entry": "import"}
    _FakeBox.shown.clear()
    monkeypatch.setattr(_FakeBox, "click", "去第 ④ 步")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown == [], "一键成片进行中时不重复提示（它自己会在关口继续）"
    window.close()


def test_review_auto_jump_hidden_window_stays_silent(svc, monkeypatch):
    # 回归：隐藏/已关闭的主窗口（测试里定时器没停而泄漏的实例）在之后被
    # 别的测试的 processEvents() 泵到时，绝不能 exec() 出模态框——离屏
    # 环境里没人能点掉它，全套测试会挂在 test_inspector_group4 那里。
    from PySide6.QtWidgets import QApplication

    _add_drama(svc, "剧")
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    assert not window.isVisible()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.store.update("characters", ch["id"], confirmed=1)
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.shown.clear()
    monkeypatch.setattr(_FakeBox, "click", "去第 ④ 步")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    QApplication.processEvents()
    assert _FakeBox.shown == [], "隐藏窗口不得弹自动跳转提示"
    assert window._review_was_nonempty is True, "转变未被消费，显示后仍应提示"
    assert jumps == []
    # 窗口显示出来后，再刷一次就该提示并跳转。
    window.show()
    _paint_main(window, svc)
    assert _FakeBox.shown, "显示后应补弹出自动跳转提示"
    assert jumps == ["replace"]
    assert window._review_was_nonempty is False
    window.close()


# ----- 第 2 项：入口整理 -----


def test_entry_buttons_hierarchy(svc):
    window = _main(svc)
    window.timer.stop()
    assert window.oneclick_btn.text() == "一键成片"
    assert window.oneclick_btn.objectName() == "primary"
    assert window.start_btn.text() == "按勾选阶段执行"
    assert window.start_btn.objectName() != "primary"
    assert window.reface_btn.text() == "只换脸（跳过前面步骤）"
    assert "04" not in window.reface_btn.toolTip()
    assert "09" not in window.export_jy_btn.toolTip()
    assert "成本表" not in window.export_jy_btn.toolTip()
    assert "新建任务行" in window.history_btn.toolTip()
    window.close()


def test_failed_card_retry_label(svc):
    drama = _add_drama(svc, "剧")
    _add_task(svc, drama["id"], status="失败")
    svc.store.update("dramas", drama["id"], status="失败")
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    texts = []
    for card in window.failed_cards:
        from PySide6.QtWidgets import QPushButton

        for btn in card.findChildren(QPushButton):
            texts.append(btn.text())
    assert "从失败步续跑" in texts
    assert "重试" not in texts
    window.close()


# ----- 第 4 项：审核关口授权预检 -----


def test_unauthorized_replace_characters(svc, tmp_path: Path):
    face_ok = _real_face(tmp_path)
    ch1 = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch1["id"], str(face_ok))
    ch2 = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ch2["id"], str(face_ok))
    ch3 = _add_character(svc, "路人", confirmed=1, authorized=0)
    _add_face_asset(svc, ch3["id"], str(tmp_path / "missing.jpg"))
    missing = svc.unauthorized_replace_characters()
    assert [m["name"] for m in missing] == ["温念"]
    window = _main(svc)
    window.close()


def test_one_click_auth_gate_blocks_phase_b(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    _add_identity_checkpoint(svc, task["id"])
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    opened: list = []
    monkeypatch.setattr(window, "_open_library", lambda: opened.append(True))
    _FakeBox.shown.clear()
    monkeypatch.setattr(_FakeBox, "click", "去人物库补授权")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window.oneclick_btn.click()
    assert calls == [], "未授权时不应进后期"
    assert opened == [True]
    assert "未授权" in window.oneclick_label.text()
    window.close()


def test_one_click_auth_gate_continue_replaces_authorized_only(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    _add_identity_checkpoint(svc, task["id"])
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    # N18: “继续，只替换已授权人物”只有在至少一个已授权人物时才有意义，
    # 零已授权时 [继续] 会被灰掉、不会进后期；补一个已授权的温雪。
    ch_ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ch_ok["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(_FakeBox, "click", "继续，只替换已授权人物")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window.oneclick_btn.click()
    assert len(calls) == 1
    assert calls[0][0][1] == ["replace", "compose", "export"]
    window.close()


def test_one_click_auth_gate_silent_when_authorized(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    _add_identity_checkpoint(svc, task["id"])
    ch = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    _FakeBox.shown.clear()
    monkeypatch.setattr(_FakeBox, "click", "继续，只替换已授权人物")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window.oneclick_btn.click()
    assert len(calls) == 1
    assert _FakeBox.shown == [], "全员已授权时关口不应弹框"
    window.close()


# ----- 第 7d 项：N8 行状态点 + 批量跳过被挡剧目 -----


def test_precheck_dot(svc):
    good = _add_drama(svc, "好剧")
    bad = _add_drama(svc, "坏剧")
    warn = _add_drama(svc, "警告剧")
    window = _main(svc)
    window.timer.stop()
    window._precheck[good["id"]] = {
        "video": "/tmp/ep1.mp4", "blocked": None,
        "duration": 10.0, "has_audio": True, "burned": False,
    }
    window._precheck[bad["id"]] = {"video": "/tmp/ep1.mp4", "blocked": "视频时长读不到（0 秒）。"}
    window._precheck[warn["id"]] = {
        "video": "/tmp/ep1.mp4", "blocked": None,
        "duration": 10.0, "has_audio": False, "burned": False,
    }
    _paint_main(window, svc)

    def _status(did):
        from PySide6.QtCore import Qt

        for row in range(window.table.rowCount()):
            item = window.table.item(row, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == did:
                return window.table.item(row, 4).text()
        raise AssertionError("row not found")

    assert _status(good["id"]).startswith("✓ ")
    assert _status(bad["id"]).startswith("✗ ")
    assert _status(warn["id"]).startswith("⚠ ")
    window.close()


def test_start_skips_blocked_drama(svc, monkeypatch):
    good = _add_drama(svc, "好剧")
    bad = _add_drama(svc, "坏剧")
    _add_task(svc, good["id"])
    _add_task(svc, bad["id"])
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    window._precheck_blocked[bad["id"]] = "视频时长读不到（0 秒）。"
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    shown: list = []
    from PySide6.QtWidgets import QMessageBox as _RealBox

    monkeypatch.setattr(_RealBox, "information", lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    window._start()
    assert len(calls) == 1
    assert calls[0][0][0] == [good["id"]]
    assert shown and "坏剧" in shown[0] and "跳过" in shown[0]
    window.close()


def test_start_refuses_when_all_blocked(svc, monkeypatch):
    bad1 = _add_drama(svc, "坏剧1")
    bad2 = _add_drama(svc, "坏剧2")
    _add_task(svc, bad1["id"])
    _add_task(svc, bad2["id"])
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    window._precheck_blocked[bad1["id"]] = "视频时长读不到（0 秒）。"
    window._precheck_blocked[bad2["id"]] = "视频时长读不到（0 秒）。"
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    warned: list = []
    from PySide6.QtWidgets import QMessageBox as _RealBox

    monkeypatch.setattr(_RealBox, "warning", lambda *a, **k: warned.append(a[2] if len(a) > 2 else ""))
    window._start()
    assert calls == []
    assert warned and "预检没通过" in warned[0]
    window.close()


# ----- 第 5 项 UI：终检窗口逐镜头增强开关 -----


def test_review_window_enhance_toggle(svc, tmp_path: Path, monkeypatch):
    from recast.desktop.review_window import ReviewWindow

    drama = _add_drama(svc, "剧")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    task = _add_task(svc, drama["id"], workdir=str(workdir))
    svc.store.insert(
        "quality_marks", task_id=task["id"], shot_code="F1-L1",
        reason="相似度低", detail="参考脸分数 0.42", status="待人工检查",
        score=0.42, thumbnail=None,
    )
    _app()
    window = ReviewWindow(svc, drama_id=drama["id"])
    assert window.table.rowCount() == 1
    assert window.table.columnCount() == 7
    assert window.table.horizontalHeaderItem(6).text() == "增强"
    from PySide6.QtWidgets import QCheckBox

    box = window.table.cellWidget(0, 6)
    assert isinstance(box, QCheckBox)
    assert box.isChecked() is False
    box.setChecked(True)
    assert svc.shot_enhance_flags(task["id"]).get("F1-L1") is True
    box.setChecked(False)
    assert "F1-L1" not in svc.shot_enhance_flags(task["id"])
    window.close()
