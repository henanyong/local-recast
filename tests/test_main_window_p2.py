"""Main-window P2 batch (second wave): N18 auth-gate dead end + skipped
names, N17 review-progress denominator, N23/N25 manual-entry gate +
locked-skip status bar, single-video import (M9/1.1), queue-panel wiring,
"一键复跑（新建任务）" label sync, and the optional N24 per-shot enhance
dialog.

Only synthetic data + stub service calls: no real video, no TTS engines, no
network. Follows the fixture pattern of tests/test_main_window_p1.py.
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


def _low_face(tmp_path: Path, name: str = "low.jpg") -> Path:
    """A tiny solid-colour crop: scores ~0.07, so the review queue treats
    it as a low-score sample."""
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (20, 20), color=(128, 128, 128)).save(path, "JPEG")
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


class _FakeButton:
    def __init__(self, text):
        self.text = text
        self._enabled = True

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled


class _FakeBox:
    """Stand-in for QMessageBox with addButton/exec/clickedButton.

    Set ``_FakeBox.click`` (class attribute) to the button text exec()
    should "click". Class-level ``shown`` records every box created
    while the fake is installed; ``warnings`` / ``infos`` record the
    static warning()/information() calls. Buttons are _FakeButton
    objects, so setEnabled/isEnabled work like the real QPushButton.
    """

    from PySide6.QtWidgets import QMessageBox as _RealBox

    ButtonRole = _RealBox.ButtonRole
    shown: list = []
    warnings: list = []
    infos: list = []
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
        btn = _FakeButton(text)
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
        _FakeBox.infos.append(args[2] if len(args) > 2 else kwargs.get("text", ""))
        return None

    @staticmethod
    def warning(*args, **kwargs):
        _FakeBox.warnings.append(args[2] if len(args) > 2 else kwargs.get("text", ""))
        return None

    @staticmethod
    def question(*args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def _reset_fakebox():
    _FakeBox.shown.clear()
    _FakeBox.warnings.clear()
    _FakeBox.infos.clear()
    _FakeBox.click = None
    yield
    _FakeBox.shown.clear()
    _FakeBox.warnings.clear()
    _FakeBox.infos.clear()
    _FakeBox.click = None


# ----- N18：关口“继续”零已授权死路 + 被跳过人物无记录 -----


def test_auth_gate_zero_authorized_disables_continue(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._authorized_replace_count() == 0
    assert window._auth_gate_ok(prompt=True, drama_ids=[drama["id"]]) is False
    assert len(_FakeBox.shown) == 1
    box = _FakeBox.shown[0]
    assert "没有已授权人物" in box.text
    assert box._buttons["继续，只替换已授权人物"].isEnabled() is False
    assert box._buttons["去人物库补授权"].isEnabled() is True
    # 没人点“继续”，就不该留下“本次未替换”的记录。
    assert "未授权，本次未替换" not in ((svc.store.get("tasks", task["id"]) or {}).get("progress") or "")
    assert not [ln for ln in svc.logs.dump() if "未授权，本次未替换" in ln["message"]]
    window.close()


def test_auth_gate_continue_records_skipped_names(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    ch2 = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch2["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    monkeypatch.setattr(_FakeBox, "click", "继续，只替换已授权人物")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._authorized_replace_count() == 1
    assert window._auth_gate_ok(prompt=True, drama_ids=[drama["id"]]) is True
    box = _FakeBox.shown[0]
    assert box._buttons["继续，只替换已授权人物"].isEnabled() is True
    task_row = svc.store.get("tasks", task["id"]) or {}
    assert "温念未授权，本次未替换" in (task_row.get("progress") or "")
    drama_row = svc.store.get("dramas", drama["id"]) or {}
    assert "温念未授权，本次未替换" in (drama_row.get("progress") or "")
    assert any("温念未授权，本次未替换" in ln["message"] for ln in svc.logs.dump())
    # 已授权的温雪不该出现在被跳过名单里。
    assert "温雪未授权" not in (task_row.get("progress") or "")
    window.close()


def test_one_click_zero_authorized_continue_goes_nowhere(svc, tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(_FakeBox, "click", "继续，只替换已授权人物")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window.oneclick_btn.click()
    assert calls == [], "零已授权时点“继续”也不该进后期（否则任务会被 parked）"
    assert opened == [], "点的是“继续”不是“去人物库”，不应打开人物库"
    assert "没有已授权人物" in _FakeBox.shown[0].text
    assert "未授权" in window.oneclick_label.text()
    window.close()


# ----- N17：进度条分母漏低分样本 -----


def test_review_progress_denominator_includes_low_samples(svc, tmp_path: Path):
    face_ok = _real_face(tmp_path)
    low = _low_face(tmp_path)
    _add_drama(svc, "剧")
    for name in ("温念", "陆寒峥", "孟逸川"):
        ch = _add_character(svc, name, confirmed=1)
        _add_face_asset(svc, ch["id"], str(face_ok))
    ch = _add_character(svc, "路人", confirmed=0)
    _add_face_asset(svc, ch["id"], str(low))
    queue = svc.list_review_queue()
    assert len(queue) == 2, "应有 1 个未确认聚类 + 1 个低分样本"
    assert {item["kind"] for item in queue} == {"cluster", "sample"}
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    # 旧分母是 3 + 1（聚类）= 4 → 75%；新分母是 3 + 2（全部待审核）= 5 → 60%。
    assert window.review_progress.value() == 60
    assert "已确认 3 / 待审核 2（低分 1）" in window.review_progress_label.text()
    assert not window.review_progress.isHidden()
    window.close()


def test_review_progress_zero_confirmed_only_low_samples(svc, tmp_path: Path):
    low = _low_face(tmp_path, "low2.jpg")
    _add_drama(svc, "剧")
    # 只有人物文件夹里的低分样本，没有任何 store 人物：旧分母 total=0，
    # 条会隐藏、标签写“还没有人物”——而队列明明非空。
    folder = tmp_path / "out" / "characters" / "路人" / "face"
    folder.mkdir(parents=True)
    import shutil

    shutil.copy2(low, folder / "low2.jpg")
    queue = svc.list_review_queue()
    assert len(queue) == 1 and queue[0]["kind"] == "sample"
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    assert not window.review_progress.isHidden(), "有待审核项时进度条不应隐藏"
    assert window.review_progress.value() == 0
    assert "待审核 1 项，未确认人物" in window.review_progress_label.text()
    assert "低分 1" in window.review_progress_label.text()
    assert "还没有人物" not in window.review_progress_label.text()
    window.close()


# ----- N23/N25：手动入口关口 + 锁住跳过状态栏 -----


def test_manual_start_auth_gate_blocks_replace(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    _add_task(svc, drama["id"])
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    opened: list = []
    monkeypatch.setattr(window, "_open_library", lambda: opened.append(True))
    monkeypatch.setattr(_FakeBox, "click", "去人物库补授权")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._start()
    assert calls == [], "手动执行勾选替换时也应过授权关口"
    assert opened == [True]
    window.close()


def test_manual_start_auth_gate_continue_starts_and_records(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(_FakeBox, "click", "继续，只替换已授权人物")
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._start()
    assert len(calls) == 1
    assert "replace" in calls[0][0][1]
    task_row = svc.store.get("tasks", task["id"]) or {}
    assert "温念未授权，本次未替换" in (task_row.get("progress") or "")
    window.close()


def test_manual_start_without_replace_skips_gate(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    _add_task(svc, drama["id"])
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    window.all_stages.setChecked(False)
    for sid, btn in window.stage_buttons.items():
        btn.setChecked(sid == "extract")
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._start()
    assert len(calls) == 1, "不勾替换就不该弹授权关口"
    assert calls[0][0][1] == ["extract"]
    assert _FakeBox.shown == [], "不勾替换时关口不应弹框"
    window.close()


def test_manual_start_locked_skip_shows_status_bar(svc, tmp_path: Path, monkeypatch):
    drama = _add_drama(svc, "剧")
    _add_task(svc, drama["id"])
    _add_character(svc, "路人", confirmed=0)  # 未确认 → 审核队列非空 → 第 ④ 步锁住
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    assert getattr(window, "_guide_locked", False) is True
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._start()
    assert len(calls) == 1, "其余阶段照常跑，只跳过锁住的替换/生成"
    assert "replace" not in calls[0][0][1] and "generate" not in calls[0][0][1]
    msg = window.statusBar().currentMessage()
    assert "已跳过锁住的「人物替换" in msg
    assert "第 ③ 步审核" in msg
    assert any("已跳过" in ln["message"] for ln in svc.logs.dump())
    window.close()


# ----- 导入单个视频（M9/1.1） -----


def test_import_single_video_creates_drama(svc, tmp_path: Path, monkeypatch):
    video = tmp_path / "ep1.mp4"
    video.write_bytes(b"\x00\x00\x00\x00")
    window = _main(svc)
    window.timer.stop()
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(video)) is True
    dramas = svc.store.list_dramas()
    assert len(dramas) == 1
    assert dramas[0]["origin_name"] == "ep1"
    tasks = svc.store.tasks_for_drama(dramas[0]["id"])
    assert len(tasks) == 1
    assert tasks[0]["video_path"] == str(video)
    _paint_main(window, svc)
    assert window.table.rowCount() == 1
    window.close()


def test_import_single_video_via_button_dialog(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    video = tmp_path / "单集.mp4"
    video.write_bytes(b"\x00\x00\x00\x00")
    window = _main(svc)
    window.timer.stop()
    assert window.import_single_btn.text() == "导入单个视频…"
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (str(video), "")
    )
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window.import_single_btn.click()
    dramas = svc.store.list_dramas()
    assert len(dramas) == 1 and dramas[0]["origin_name"] == "单集"
    window.close()


def test_import_single_video_rejects_non_video(svc, tmp_path: Path, monkeypatch):
    note = tmp_path / "notes.txt"
    note.write_text("not a video", encoding="utf-8")
    window = _main(svc)
    window.timer.stop()
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(note)) is False
    assert svc.store.list_dramas() == []
    assert _FakeBox.warnings and "不是视频文件" in _FakeBox.warnings[0]
    window.close()


def test_import_single_video_rejects_missing_file(svc, tmp_path: Path, monkeypatch):
    window = _main(svc)
    window.timer.stop()
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(tmp_path / "nope.mp4")) is False
    assert svc.store.list_dramas() == []
    assert _FakeBox.warnings and "找不到视频文件" in _FakeBox.warnings[0]
    window.close()


# ----- 队列面板接线 -----


def test_queue_panel_button_opens_dialog(svc, monkeypatch):
    opened: list = []

    class _StubPanel:
        def __init__(self, service, parent=None):
            opened.append({"service": service, "parent": parent})

        def exec(self):
            opened[0]["exec_called"] = True
            return 1

    window = _main(svc)
    window.timer.stop()
    assert window.queue_btn.text() == "队列面板"
    monkeypatch.setattr("recast.desktop.queue_panel.QueuePanelDialog", _StubPanel)
    window.queue_btn.click()
    assert len(opened) == 1
    assert opened[0]["service"] is svc
    assert opened[0]["parent"] is window
    assert opened[0].get("exec_called") is True
    assert window._queue_panel is not None
    window.close()


def test_queue_panel_button_tooltip_and_real_dialog_class(svc):
    from recast.desktop.queue_panel import QueuePanelDialog

    window = _main(svc)
    window.timer.stop()
    assert "队列面板" in window.queue_btn.toolTip()
    assert "从失败步续跑" in window.queue_btn.toolTip()
    dlg = QueuePanelDialog(svc, parent=window)
    assert dlg.windowTitle() == "队列面板"
    dlg.close()
    window.close()


# ----- 文案同步：一键复跑 → 一键复跑（新建任务） -----


def test_rerun_label_synced(svc, monkeypatch):
    drama = _add_drama(svc, "剧")
    _add_task(svc, drama["id"], status="失败")
    svc.store.update("dramas", drama["id"], status="失败")
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    assert "一键复跑（新建任务）" in window.history_btn.toolTip()
    assert "「一键复跑」" not in window.history_btn.toolTip()
    tooltips = []
    for card in window.failed_cards:
        from PySide6.QtWidgets import QPushButton

        for btn in card.findChildren(QPushButton):
            tooltips.append(btn.toolTip())
    assert tooltips, "失败卡片应已渲染"
    assert any("一键复跑（新建任务）" in t for t in tooltips)
    assert not any("「一键复跑」" in t for t in tooltips)
    window.close()


# ----- N24（可选项）：逐镜头增强对话框 -----


def test_shot_enhance_dialog_lists_shots_and_toggles(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    drama = _add_drama(svc, "剧")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    task = _add_task(svc, drama["id"], workdir=str(workdir))
    for code in ("S1", "S2"):
        svc.store.insert(
            "shots", task_id=task["id"], shot_code=code, index_no=int(code[1]),
            start_time=0.0, end_time=1.0,
        )
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    monkeypatch.setattr(QDialog, "exec", lambda self: 1)
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._open_shot_enhance()
    assert window._shot_enhance_dialog is not None
    from PySide6.QtWidgets import QLabel

    hint_texts = [lbl.text() for lbl in window._shot_enhance_dialog.findChildren(QLabel)]
    assert any("两条路都支持" in t for t in hint_texts)
    assert any("拼接后按标记时段增强" in t for t in hint_texts)
    assert not any("暂不支持" in t for t in hint_texts)
    boxes = window._shot_enhance_boxes
    assert {(tid, code) for tid, code in boxes} == {(task["id"], "S1"), (task["id"], "S2")}
    assert svc.shot_enhance_flags(task["id"]) == {}
    boxes[(task["id"], "S1")].setChecked(True)
    assert svc.shot_enhance_flags(task["id"]).get("S1") is True
    boxes[(task["id"], "S1")].setChecked(False)
    assert "S1" not in svc.shot_enhance_flags(task["id"])
    window.close()


def test_shot_enhance_no_shots_informs(svc, tmp_path: Path, monkeypatch):
    drama = _add_drama(svc, "剧")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    _add_task(svc, drama["id"], workdir=str(workdir))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._open_shot_enhance()
    assert _FakeBox.infos and "还没有镜头记录" in _FakeBox.infos[0]
    window.close()
