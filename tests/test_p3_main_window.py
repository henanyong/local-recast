"""P3 batch, main window: N30 single-video import collision confirm,
N29 enhance-dialog copy, N23 _reface auth gate, N21 auto-jump
"don't prompt again" + confirm-vs-discard.

Only synthetic data + stub service calls: no real video, no TTS engines, no
network. Follows the fixture pattern of tests/test_main_window_p2.py.
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


def _add_drama(svc, name: str = "剧", *, entry_type: str = "import", root_path: str | None = None, source_path: str | None = None) -> dict:
    fields: dict = {
        "origin_name": name, "root_path": root_path or f"/tmp/{name}", "episode_count": 1,
        "entry_type": entry_type,
    }
    if source_path is not None:
        fields["source_path"] = source_path
    return svc.store.insert("dramas", **fields)


def _add_task(svc, drama_id, *, status: str = "待执行", workdir=None, video_path: str = "/tmp/ep1.mp4"):
    return svc.store.insert(
        "tasks", drama_id=drama_id, episode_name="第1集",
        video_path=video_path, status=status, workdir=workdir,
    )


def _add_character(svc, name: str, *, confirmed: int = 1, authorized: int = 0, auth_type: str | None = None):
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


class _FakeButton:
    def __init__(self, text):
        self.text = text
        self._enabled = True

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled


class _FakeBox:
    """Stand-in for QMessageBox with addButton/exec/clickedButton/setCheckBox.

    Set ``_FakeBox.click`` (class attribute) to the button text exec()
    should "click". ``question_answer`` is what the static question()
    returns (default No — safe). ``mute`` makes exec() tick the
    [不再提示] checkbox, if one was attached. Class-level ``shown`` /
    ``questions`` record every box / question while the fake is installed.
    """

    from PySide6.QtWidgets import QMessageBox as _RealBox

    ButtonRole = _RealBox.ButtonRole
    StandardButton = _RealBox.StandardButton
    shown: list = []
    warnings: list = []
    infos: list = []
    questions: list = []
    click: str | None = None
    question_answer = _RealBox.StandardButton.No
    mute: bool = False

    def __init__(self, parent=None):
        self.parent = parent
        self._buttons: dict = {}
        self.exec_called = False
        self.checkbox = None
        _FakeBox.shown.append(self)

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def addButton(self, text, role):
        btn = _FakeButton(text)
        self._buttons[text] = btn
        return btn

    def setCheckBox(self, checkbox):
        self.checkbox = checkbox

    def checkBox(self):
        return self.checkbox

    def exec(self):
        self.exec_called = True
        if type(self).mute and self.checkbox is not None:
            self.checkbox.setChecked(True)
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
        _FakeBox.questions.append(args)
        return _FakeBox.question_answer


@pytest.fixture(autouse=True)
def _reset_fakebox():
    _FakeBox.shown.clear()
    _FakeBox.warnings.clear()
    _FakeBox.infos.clear()
    _FakeBox.questions.clear()
    _FakeBox.click = None
    _FakeBox.mute = False
    _FakeBox.question_answer = _FakeBox.StandardButton.No
    yield
    _FakeBox.shown.clear()
    _FakeBox.warnings.clear()
    _FakeBox.infos.clear()
    _FakeBox.questions.clear()
    _FakeBox.click = None
    _FakeBox.mute = False
    _FakeBox.question_answer = _FakeBox.StandardButton.No


def _check_first_drama_row(window):
    window.table.cellWidget(0, 0).setChecked(True)


# ----- N30：单视频导入碰撞确认框（剧目身份 = 首集视频文件路径） -----


def test_find_import_collision_returns_existing_drama(svc, tmp_path: Path):
    folder = tmp_path / "drama-folder"
    folder.mkdir()
    ep1 = folder / "ep1.mp4"
    ep1.write_bytes(b"")
    drama = _add_drama(svc, "剧", root_path=str(folder), source_path=str(ep1))
    window = _main(svc)
    window.timer.stop()
    # 同一文件路径再次入库：算碰撞。
    assert window._find_import_collision(str(ep1))["id"] == drama["id"]
    # 非规范化路径（带 ..）解析后是同一文件：也算碰撞。
    assert window._find_import_collision(str(folder / "sub" / ".." / "ep1.mp4"))["id"] == drama["id"]
    # 同文件夹、不同视频：不再是碰撞。
    assert window._find_import_collision(str(folder / "ep2.mp4")) is None
    # 不同文件夹：不算碰撞。
    assert window._find_import_collision(str(tmp_path / "other" / "ep9.mp4")) is None
    window.close()


def test_single_video_import_collision_cancel_does_not_import(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox as RealBox

    folder = tmp_path / "drama-folder"
    folder.mkdir()
    ep1 = folder / "ep1.mp4"
    ep1.write_bytes(b"")
    drama = _add_drama(svc, "剧", root_path=str(folder), source_path=str(ep1))
    task = _add_task(svc, drama["id"], video_path=str(ep1))
    window = _main(svc)
    window.timer.stop()
    scans: list = []
    monkeypatch.setattr(svc, "scan", lambda root: scans.append(root))
    _FakeBox.question_answer = RealBox.StandardButton.No
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(ep1)) is False
    assert scans == [], "取消导入时绝不能调 scan、不写脏数据"
    assert _FakeBox.questions, "同一文件再次入库时应先弹确认框"
    assert any("已在库中" in " ".join(str(a) for a in q) for q in _FakeBox.questions)
    assert any("重新导入" in " ".join(str(a) for a in q) for q in _FakeBox.questions)
    tasks = svc.store.tasks_for_drama(drama["id"])
    assert len(tasks) == 1 and tasks[0]["id"] == task["id"]
    assert tasks[0]["video_path"] == str(ep1)
    window.close()


def test_single_video_import_collision_confirmed_reimports_same_file(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox as RealBox

    folder = tmp_path / "drama-folder"
    folder.mkdir()
    ep1 = folder / "ep1.mp4"
    ep1.write_bytes(b"")
    drama = _add_drama(svc, "剧", root_path=str(folder), source_path=str(ep1))
    task = _add_task(svc, drama["id"], video_path=str(ep1))
    window = _main(svc)
    window.timer.stop()
    _FakeBox.question_answer = RealBox.StandardButton.Yes
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(ep1)) is True
    # 同一文件重新导入：不建重复剧目行，分集任务还是原来那一个。
    assert len(svc.store.list_dramas()) == 1
    tasks = svc.store.tasks_for_drama(drama["id"])
    assert len(tasks) == 1 and tasks[0]["id"] == task["id"]
    assert tasks[0]["video_path"] == str(ep1)
    window.close()


def test_single_video_import_same_folder_different_video_is_new_drama(svc, tmp_path: Path, monkeypatch):
    folder = tmp_path / "drama-folder"
    folder.mkdir()
    ep1 = folder / "ep1.mp4"
    ep1.write_bytes(b"")
    drama = _add_drama(svc, "剧", root_path=str(folder), source_path=str(ep1))
    _add_task(svc, drama["id"], video_path=str(ep1))
    ep2 = folder / "ep2.mp4"
    ep2.write_bytes(b"")
    window = _main(svc)
    window.timer.stop()
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(ep2)) is True
    assert _FakeBox.questions == [], "同文件夹、不同视频不再是碰撞，不应弹确认框"
    dramas = svc.store.list_dramas()
    assert len(dramas) == 2, "同文件夹不同视频应新建一部剧目"
    # 原剧目的分集任务原样保留，没被替换。
    tasks = svc.store.tasks_for_drama(drama["id"])
    assert len(tasks) == 1 and tasks[0]["video_path"] == str(ep1)
    new_drama = next(d for d in dramas if d["id"] != drama["id"])
    new_tasks = svc.store.tasks_for_drama(new_drama["id"])
    assert len(new_tasks) == 1 and new_tasks[0]["video_path"] == str(ep2)
    window.close()


# ----- N29：逐镜头增强对话框文案 -----


def test_shot_enhance_hint_says_both_routes_supported(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QLabel

    drama = _add_drama(svc, "剧")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    task = _add_task(svc, drama["id"], workdir=str(workdir))
    svc.store.insert(
        "shots", task_id=task["id"], shot_code="S1", index_no=1,
        start_time=0.0, end_time=1.0,
    )
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    monkeypatch.setattr(QDialog, "exec", lambda self: 1)
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._open_shot_enhance()
    assert window._shot_enhance_dialog is not None
    hint_texts = [lbl.text() for lbl in window._shot_enhance_dialog.findChildren(QLabel)]
    assert any("两条路都支持" in t for t in hint_texts)
    assert any("拼接后按标记时段增强" in t for t in hint_texts)
    assert not any("暂不支持" in t for t in hint_texts)
    window.close()


def test_shot_enhance_docstring_synced():
    from recast.desktop.main_window import MainWindow

    doc = MainWindow._open_shot_enhance.__doc__ or ""
    assert "两条路都支持" in doc
    assert "暂不支持" not in doc


# ----- N23：[只换脸] 走授权关口 -----


def test_reface_blocked_by_auth_gate_zero_authorized(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    _add_task(svc, drama["id"])
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    _check_first_drama_row(window)
    starts: list = []
    monkeypatch.setattr(svc, "start", lambda ids, stages: starts.append((ids, stages)))
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._reface()
    assert starts == [], "零已授权时 [只换脸] 不得进替换"
    assert len(_FakeBox.shown) == 1
    box = _FakeBox.shown[0]
    assert "没有已授权人物" in box.text
    assert box._buttons["继续，只替换已授权人物"].isEnabled() is False
    window.close()


def test_reface_continue_with_authorized_records_skipped_and_starts(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    ch2 = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch2["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    _check_first_drama_row(window)
    starts: list = []
    monkeypatch.setattr(svc, "start", lambda ids, stages: starts.append((ids, stages)))
    _FakeBox.click = "继续，只替换已授权人物"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._reface()
    assert starts == [([drama["id"]], ["replace"])]
    task_row = svc.store.get("tasks", task["id"]) or {}
    assert "温念未授权，本次未替换" in (task_row.get("progress") or "")
    assert svc.store.skipped_character_names(drama_id=drama["id"]) == ["温念"]
    window.close()


def test_reface_all_authorized_starts_without_dialog(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    _add_task(svc, drama["id"])
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    _check_first_drama_row(window)
    starts: list = []
    monkeypatch.setattr(svc, "start", lambda ids, stages: starts.append((ids, stages)))
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._reface()
    assert starts == [([drama["id"]], ["replace"])]
    assert _FakeBox.shown == [], "无人未授权时关口不应弹框"
    window.close()


# ----- N18：关口跳过记录持久化（auth_skips） -----


def test_auth_gate_skip_persists_in_auth_skips(svc, tmp_path: Path, monkeypatch):
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    ch2 = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch2["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _FakeBox.click = "继续，只替换已授权人物"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._auth_gate_ok(prompt=True, drama_ids=[drama["id"]]) is True
    assert svc.store.skipped_character_names(drama_id=drama["id"]) == ["温念"]
    assert svc.store.skipped_character_names(drama_id=drama["id"], task_id=task["id"]) == ["温念"]
    # 关键：runner 重写任务 progress 后，跳过记录还在。
    svc.store.update("tasks", task["id"], progress="01 扫描完成")
    assert svc.store.skipped_character_names(drama_id=drama["id"]) == ["温念"]
    # 再过一次关口、再点一次 [继续]：同一 drama+task+人名只记一次。
    assert window._auth_gate_ok(prompt=True, drama_ids=[drama["id"]]) is True
    assert len(svc.store.list_auth_skips(drama_id=drama["id"])) == 1
    window.close()


def test_auth_gate_skip_persists_without_tasks(svc, tmp_path: Path, monkeypatch):
    """边界：剧目下面还没分集任务，跳过记录也应落下来（drama 级）。"""
    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc, "剧")
    ch2 = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch2["id"], str(face_ok))
    window = _main(svc)
    window.timer.stop()
    _FakeBox.click = "继续，只替换已授权人物"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    # 零已授权时 [继续] 灰掉、关口返回 False，不会走到记录——所以这里
    # 直接测 _record_skipped_characters 的无任务分支。
    window._record_skipped_characters([{"name": "温念"}], [drama["id"]])
    assert svc.store.skipped_character_names(drama_id=drama["id"]) == ["温念"]
    window.close()


# ----- N21：自动跳转“不再提示” + 确认/丢弃区分 -----


def test_review_auto_jump_mute_checkbox_persists(svc, monkeypatch):
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.store.update("characters", ch["id"], confirmed=1)
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "去第 ④ 步"
    _FakeBox.mute = True
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown, "确认清空后应弹自动跳转提示"
    box = _FakeBox.shown[0]
    assert box.checkbox is not None and box.checkbox.text() == "不再提示"
    assert jumps == ["replace"]
    assert svc.settings.review_jump_muted is True
    stored = svc.store.get_setting("app_settings", {}) or {}
    assert stored.get("review_jump_muted") is True
    # 勾过 [不再提示] 后：再来一轮确认清空也不应再弹。
    ch2 = _add_character(svc, "新人", confirmed=0)
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.store.update("characters", ch2["id"], confirmed=1)
    shown_before = len(_FakeBox.shown)
    _paint_main(window, svc)
    assert len(_FakeBox.shown) == shown_before, "已设不再提示，不应再弹"
    assert window._review_was_nonempty is False
    window.close()


def test_review_auto_jump_discard_does_not_prompt(svc, monkeypatch):
    _add_character(svc, "温雪", confirmed=1)
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    # 丢弃最后一个待审核聚类：已确认人数没涨，清空调的是丢弃。
    svc.store.delete("characters", ch["id"])
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown == [], "丢弃清空不应弹自动跳转提示"
    assert window._review_was_nonempty is False
    window.close()


def test_review_auto_jump_zero_confirmed_shows_status_bar_not_dialog(svc, monkeypatch):
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.store.delete("characters", ch["id"])
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown == [], "零确认人物时不应弹自动跳转提示"
    assert "尚未确认人物" in window.statusBar().currentMessage()
    window.close()


def test_review_auto_jump_confirm_still_prompts_with_mute_checkbox(svc, monkeypatch):
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.store.update("characters", ch["id"], confirmed=1)
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "稍后"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown, "确认清空后应弹自动跳转提示"
    box = _FakeBox.shown[0]
    assert box.checkbox is not None and box.checkbox.text() == "不再提示"
    assert not box.checkbox.isChecked(), "没勾 [不再提示] 时不应写设置"
    assert jumps == [], "点 [稍后] 不应跳转"
    assert svc.settings.review_jump_muted is False
    window.close()
