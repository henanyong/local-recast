"""P4 N21 残留：仅低分样本确认导致队列清空时也应弹自动跳转。

PM 第 6 轮 §三：低分样本确认（review_assign / 认领）不改变已确认人数，
旧启发式（清空时已确认人数没涨 = 丢弃）会把它误判成丢弃清空而不弹。
P4 改法（方案 a）：主窗口维护“自队列上次非空以来确认动作次数”计数，
确认动作 = 已确认人数增量 + “已审核”标记的低分样本增量；队列清空时
计数 >0 即判定为确认清空、弹自动跳转。丢弃不打“已审核”标记，所以
丢弃清空计数仍为 0，不弹。

只用合成数据 + 真实 service 方法（review_assign / review_discard /
review_create_character / confirm_character）：不跑真实视频、不调 TTS、
不联网。harness 沿用 tests/test_p3_main_window.py 的写法。
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


def _add_drama(svc, name: str = "剧"):
    return svc.store.insert(
        "dramas", origin_name=name, root_path=f"/tmp/{name}",
        episode_count=1, entry_type="import",
    )


def _add_character(svc, name: str, *, confirmed: int = 1):
    return svc.store.insert(
        "characters", name=name, display_name=name, confirmed=confirmed,
        authorized=0, cluster_key=f"test:{name}",
    )


def _add_face_asset(svc, character_id: str, path: str, *, notes: str = ""):
    fields: dict = {
        "character_id": character_id, "kind": "face", "path": path,
    }
    if notes:
        fields["notes"] = notes
    return svc.store.insert("character_assets", **fields)


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


def _low_face(out_dir: Path, name: str = "low.jpg") -> Path:
    """A tiny solid-colour JPEG: size_term 0.25 x sharpness 0 -> score 0.1,
    well below review_min_score 0.5, so it lands in the review queue as a
    low-score sample row."""
    from PIL import Image

    path = out_dir / name
    Image.new("RGB", (30, 30), (128, 128, 128)).save(path, "JPEG")
    return path


def _queue(svc):
    return svc.list_review_queue()


class _FakeButton:
    def __init__(self, text):
        self.text = text
        self._enabled = True

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled


class _FakeBox:
    """Stand-in for QMessageBox (copied from test_p3_main_window.py)."""

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


def _two_confirmed_with_low_sample(svc, tmp_path: Path, *, low_names=("low.jpg",)):
    """已确认人物 A（带低分样本）+ 已确认人物 B（带清晰参考脸）。

    队列里只有低分样本行：已确认人数恒为 2，旧启发式一定误判为丢弃。
    """
    out = tmp_path / "out"
    _add_drama(svc)
    a = _add_character(svc, "温念", confirmed=1)
    b = _add_character(svc, "温雪", confirmed=1)
    _add_face_asset(svc, b["id"], str(_real_face(tmp_path, "good.jpg")))
    lows = []
    for name in low_names:
        low = _low_face(out, name)
        _add_face_asset(svc, a["id"], str(low))
        lows.append(str(low))
    rows = _queue(svc)
    assert [r["kind"] for r in rows] == ["sample"] * len(low_names), rows
    assert [r["key"] for r in rows] == lows
    return a, b, lows


# ----- 正向：仅低分样本确认清空时弹自动跳转 -----


def test_low_sample_assign_to_other_character_prompts(svc, tmp_path, monkeypatch):
    """核心回归：把唯一的低分样本分给另一已确认人物，清队后应弹框并跳转。

    已确认人数全程是 2、没涨；旧代码（只看已确认人数增量）会在这里沉默。
    """
    a, b, (low,) = _two_confirmed_with_low_sample(svc, tmp_path)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    assert window._review_confirms_since_nonempty == 0
    svc.review_assign(low, b["id"], owner_id=a["id"])
    assert _queue(svc) == [], "分完低分样本后队列应为空"
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert window._review_confirms_since_nonempty == 1, "一次低分样本确认应计 1"
    assert _FakeBox.shown, "仅低分样本确认清空也应弹自动跳转提示"
    assert any("审核已完成" in getattr(box, "title", "") for box in _FakeBox.shown)
    assert jumps == ["replace"]
    assert window._review_was_nonempty is False
    window.close()


def test_low_sample_assign_counts_mid_episode_too(svc, tmp_path, monkeypatch):
    """边界：两个低分样本，先分一个（队列仍非空）、再分一个才清空。

    第一次确认发生在两次非空刷新之间，也要累计进计数，不能只看最后一跳。
    """
    a, b, (low1, low2) = _two_confirmed_with_low_sample(
        svc, tmp_path, low_names=("low1.jpg", "low2.jpg")
    )
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    svc.review_assign(low1, b["id"], owner_id=a["id"])
    _paint_main(window, svc)  # 队列还剩 1 项：这次刷新应把计数累到 1
    assert window._review_was_nonempty is True
    assert window._review_confirms_since_nonempty == 1
    assert _FakeBox.shown == [], "队列没清空时不应弹"
    svc.review_assign(low2, b["id"], owner_id=a["id"])
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert window._review_confirms_since_nonempty == 2
    assert _FakeBox.shown, "两个低分样本都确认完、清空后应弹"
    assert jumps == ["replace"]
    window.close()


def test_low_sample_create_character_prompts(svc, tmp_path, monkeypatch):
    """正向：低分样本新建成人物（已确认 +1，同时打“已审核”标记）清空后应弹。"""
    out = tmp_path / "out"
    _add_drama(svc)
    a = _add_character(svc, "温念", confirmed=1)
    low = _low_face(out, "low.jpg")
    _add_face_asset(svc, a["id"], str(low))
    assert [r["kind"] for r in _queue(svc)] == ["sample"]
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    svc.review_create_character(str(low), "新人", owner_id=a["id"])
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown, "低分样本新建人物清空后应弹"
    assert jumps == ["replace"]
    window.close()


def test_cluster_confirm_still_prompts(svc, tmp_path, monkeypatch):
    """回归：普通聚类确认清空仍弹（旧行为不变）。"""
    _add_drama(svc)
    ch = _add_character(svc, "路人", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.confirm_character(ch["id"])
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown, "聚类确认清空后应弹自动跳转提示"
    assert jumps == ["replace"]
    window.close()


# ----- 边界：丢弃 / 不再提示 / 零确认 / 混合 -----


def test_low_sample_discard_does_not_prompt(svc, tmp_path, monkeypatch):
    """边界：唯一的低分样本是丢弃掉的，清空后仍不弹，计数保持 0。"""
    a, b, (low,) = _two_confirmed_with_low_sample(svc, tmp_path)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    svc.review_discard(low, owner_id=a["id"])
    assert _queue(svc) == []
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert window._review_confirms_since_nonempty == 0, "丢弃不打标记、不应计数"
    assert _FakeBox.shown == [], "丢弃清空不应弹自动跳转提示"
    assert window._review_was_nonempty is False
    window.close()


def test_low_sample_assign_then_discard_last_still_prompts(svc, tmp_path, monkeypatch):
    """边界（方案 a 语义）：本轮先确认了一个低分样本、再丢弃最后一个。

    自队列上次非空以来发生过确认动作，清空应判定为确认清空、弹提示；
    旧基线启发式会在这里沉默（最后一次刷新时已确认人数没涨）。
    """
    a, b, (low1, low2) = _two_confirmed_with_low_sample(
        svc, tmp_path, low_names=("low1.jpg", "low2.jpg")
    )
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    svc.review_assign(low1, b["id"], owner_id=a["id"])
    _paint_main(window, svc)
    assert window._review_confirms_since_nonempty == 1
    svc.review_discard(low2, owner_id=a["id"])
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "稍后"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown, "本轮有过确认动作，清空后应弹"
    assert jumps == [], "点 [稍后] 不应跳转"
    window.close()


def test_low_sample_confirm_muted_does_not_prompt(svc, tmp_path, monkeypatch):
    """边界：勾过 [不再提示] 后，低分样本确认清空也不弹。"""
    a, b, (low,) = _two_confirmed_with_low_sample(svc, tmp_path)
    svc.update_settings(review_jump_muted=True)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    svc.review_assign(low, b["id"], owner_id=a["id"])
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown == [], "已设不再提示，不应再弹"
    assert window._review_was_nonempty is False
    window.close()


def test_low_sample_zero_confirmed_shows_status_bar(svc, tmp_path, monkeypatch):
    """边界：零确认人物时清空（先丢低分样本、再丢唯一聚类），不弹、走状态栏。"""
    out = tmp_path / "out"
    _add_drama(svc)
    a = _add_character(svc, "路人", confirmed=0)
    _add_face_asset(svc, a["id"], str(_real_face(tmp_path, "good.jpg")))
    low = _low_face(out, "low.jpg")
    _add_face_asset(svc, a["id"], str(low))
    assert len(_queue(svc)) == 2  # 聚类行 + 低分样本行
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    svc.review_discard(str(low), owner_id=a["id"])
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    svc.review_discard(a["id"])
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown == [], "零确认人物时不应弹自动跳转提示"
    assert "尚未确认人物" in window.statusBar().currentMessage()
    window.close()


def test_confirm_counter_resets_for_next_episode(svc, tmp_path, monkeypatch):
    """边界：上一轮的确认计数不得漏进下一轮——下一轮纯丢弃清空时不弹。"""
    a, b, (low,) = _two_confirmed_with_low_sample(svc, tmp_path)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    svc.review_assign(low, b["id"], owner_id=a["id"])
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert _FakeBox.shown, "第一轮确认清空应弹"
    assert window._review_confirms_since_nonempty == 1
    _FakeBox.shown.clear()
    # 第二轮：新来一个未确认聚类，确认计数应在变非空时清零。
    c = _add_character(svc, "新人", confirmed=0)
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    assert window._review_confirms_since_nonempty == 0, "新一轮开始计数应清零"
    svc.store.delete("characters", c["id"])  # 丢弃第二轮唯一的待审核项
    _paint_main(window, svc)
    assert _FakeBox.shown == [], "第二轮纯丢弃清空不应弹（上一轮计数不得漏进来）"
    assert window._review_was_nonempty is False
    window.close()


def test_low_sample_jump_deferred_while_hidden(svc, tmp_path, monkeypatch):
    """边界：窗口隐藏时不弹也不消费转变；显示后补弹（计数留到消费时再算）。"""
    from PySide6.QtWidgets import QApplication

    a, b, (low,) = _two_confirmed_with_low_sample(svc, tmp_path)
    window = _main(svc)
    window.timer.stop()
    assert not window.isVisible()
    _paint_main(window, svc)
    svc.review_assign(low, b["id"], owner_id=a["id"])
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    QApplication.processEvents()
    assert _FakeBox.shown == [], "隐藏窗口不得弹自动跳转提示"
    assert window._review_was_nonempty is True, "转变未被消费，显示后仍应提示"
    window.show()
    _paint_main(window, svc)
    assert _FakeBox.shown, "显示后应补弹出自动跳转提示"
    assert jumps == ["replace"]
    assert window._review_was_nonempty is False
    window.close()
