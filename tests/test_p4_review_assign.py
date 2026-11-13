"""P4：低分样本点“确认是同一人”不再抛错。

Bug（N21 worker 顺带发现）：低分样本行在聚类对比视图点 [确认是同一人]
走 cluster_compare._confirm -> review_assign(key, owner)，目标人物就是
样本原来的主人，review_assign 落到 move_face_sample 抛
ValueError“移到另一人物：目标人物和原来是同一个。”，对话框只显示
“确认失败”，样本留在队列里，只能靠分给另一人物 / 新建人物 / 丢弃清掉。

根治在 service 层（review_assign）：目标 = 来源的样本行不移动文件，
直接给资产行打“已审核”标记（与 move_face_sample / _adopt_trash_sample
用的同一标记，队列靠它过滤、主窗口 N21 计数靠它识别），文件原地不动。
目标不同时仍正常移动；聚类行目标=自身仍抛错（聚类确认走
confirm_character，不走 review_assign）。

只用合成数据 + 真实 service 方法 + 真实 ClusterCompareDialog / MainWindow：
不跑真实视频、不调 TTS、不联网。harness 沿用
tests/test_p4_review_jump.py 的写法。
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
    """A real readable JPEG (200x200 checkerboard): scores ~1.0."""
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
    """A tiny solid-colour JPEG: score ~0.1, below review_min_score 0.5,
    so it lands in the review queue as a low-score sample row."""
    from PIL import Image

    path = out_dir / name
    Image.new("RGB", (30, 30), (128, 128, 128)).save(path, "JPEG")
    return path


def _queue(svc):
    return svc.list_review_queue()


def _one_low_sample(svc, tmp_path: Path, *, notes: str = ""):
    """已确认人物 A 带一个低分样本：队列里只有这一行样本行。"""
    out = tmp_path / "out"
    _add_drama(svc)
    a = _add_character(svc, "温念", confirmed=1)
    low = _low_face(out, "low.jpg")
    _add_face_asset(svc, a["id"], str(low), notes=notes)
    rows = _queue(svc)
    assert [r["kind"] for r in rows] == ["sample"], rows
    assert [r["key"] for r in rows] == [str(low)]
    return a, str(low)


def _asset_notes(svc, character_id: str, path: str) -> str | None:
    ch = svc.store.get_character(character_id)
    for a in ch.get("assets") or []:
        if a.get("kind") == "face" and a.get("path") == path:
            return a.get("notes")
    return None


# ----- 正向：同一人物确认不再抛错，打标记、清队列 -----


def test_same_owner_assign_marks_reviewed_and_clears_queue(svc, tmp_path):
    """核心回归：review_assign(低分样本, 它自己的主人) 不再抛错。

    以前这里抛 ValueError“移到另一人物：目标人物和原来是同一个。”。
    """
    a, low = _one_low_sample(svc, tmp_path)
    out = svc.review_assign(low, a["id"], owner_id=a["id"])
    assert out["kind"] == "sample"
    assert out["character_id"] == a["id"]
    assert Path(low).is_file(), "同一人物确认不应移动文件"
    assert "已审核" in (_asset_notes(svc, a["id"], low) or "")
    assert _queue(svc) == [], "打上标记后样本行应从队列消失"


def test_same_owner_assign_resolves_owner_without_owner_id(svc, tmp_path):
    """边界：不传 owner_id 时也一样——这正是 _confirm 的调用方式
    （review_assign(key, owner)，只靠 _sample_owner_id 找主人）。"""
    a, low = _one_low_sample(svc, tmp_path)
    out = svc.review_assign(low, a["id"])
    assert out["kind"] == "sample"
    assert out["path"] == low
    assert "已审核" in (_asset_notes(svc, a["id"], low) or "")
    assert _queue(svc) == []


def test_same_owner_assign_preserves_existing_notes(svc, tmp_path):
    """边界：资产行原有备注（如“聚类样本”）应保留，只追加标记。"""
    a, low = _one_low_sample(svc, tmp_path, notes="聚类样本")
    svc.review_assign(low, a["id"], owner_id=a["id"])
    notes = _asset_notes(svc, a["id"], low) or ""
    assert notes == "聚类样本；已审核", notes
    assert _queue(svc) == []


def test_same_owner_assign_is_idempotent(svc, tmp_path):
    """边界：同一低分样本确认两次也不抛错，标记不重复叠加。"""
    a, low = _one_low_sample(svc, tmp_path)
    svc.review_assign(low, a["id"], owner_id=a["id"])
    svc.review_assign(low, a["id"], owner_id=a["id"])
    notes = _asset_notes(svc, a["id"], low) or ""
    assert notes.count("已审核") == 1, notes
    assert _queue(svc) == []


# ----- 语义不变：目标不同时仍正常移动 -----


def test_assign_to_different_character_still_moves(svc, tmp_path):
    """回归：目标是另一人物时，文件照常移进目标的 face/ 文件夹并打标记。"""
    a, low = _one_low_sample(svc, tmp_path)
    b = _add_character(svc, "温雪", confirmed=1)
    out = svc.review_assign(low, b["id"], owner_id=a["id"])
    assert out["kind"] == "sample"
    dest = Path(out["path"])
    assert dest.is_file() and dest.parent.name == "face"
    assert not Path(low).exists(), "分给另一人物时原文件应被移走"
    assert "已审核" in (_asset_notes(svc, b["id"], str(dest)) or "")
    assert _queue(svc) == []


def test_assign_to_different_character_without_owner_id_still_moves(svc, tmp_path):
    """边界：不传 owner_id、分给另一人物时，同样靠 _sample_owner_id 找到
    主人并正常移动（同一人物分支只在 owner == target 时触发）。"""
    a, low = _one_low_sample(svc, tmp_path)
    b = _add_character(svc, "温雪", confirmed=1)
    out = svc.review_assign(low, b["id"])
    assert Path(out["path"]).is_file()
    assert not Path(low).exists()
    assert _queue(svc) == []


# ----- 失败语义不变 -----


def test_same_owner_assign_unknown_target_still_raises(svc, tmp_path):
    """边界：目标人物不存在时仍抛错，不静默打标记。"""
    a, low = _one_low_sample(svc, tmp_path)
    with pytest.raises(ValueError, match="找不到目标人物"):
        svc.review_assign(low, "no-such-character", owner_id=a["id"])
    assert _queue(svc) != [], "失败后样本行应留在队列里"
    assert "已审核" not in (_asset_notes(svc, a["id"], low) or "")


def test_review_assign_cluster_same_owner_still_raises(svc, tmp_path):
    """边界：聚类行目标=自身仍抛错——聚类确认走 confirm_character，
    review_assign 的聚类分支语义不改。"""
    _add_drama(svc)
    ch = _add_character(svc, "路人", confirmed=0)
    with pytest.raises(ValueError, match="目标人物和这个聚类是同一个"):
        svc.review_assign(ch["id"], ch["id"])
    assert svc.store.get_character(ch["id"]) is not None


# ----- 对比视图：经 _confirm 的同一人物路径 -----


def _dialog(svc, row: dict):
    from recast.desktop.cluster_compare import ClusterCompareDialog

    _app()
    return ClusterCompareDialog(svc, dict(row))


def test_cluster_compare_confirm_same_owner_accepts(svc, tmp_path):
    """核心回归（真实 UI 路径）：低分样本行在对比视图点 [确认是同一人]，
    对话框应 Accepted、队列清空、文件原地不动——以前这里只显示
    “确认失败：移到另一人物：目标人物和原来是同一个。”。"""
    from PySide6.QtWidgets import QDialog

    a, low = _one_low_sample(svc, tmp_path)
    row = _queue(svc)[0]
    assert row["kind"] == "sample" and row["character_id"] == a["id"]
    dlg = _dialog(svc, row)
    dlg.confirm_btn.click()
    assert dlg.result() == QDialog.DialogCode.Accepted, dlg.status_label.text()
    assert dlg.status_label.text() == ""
    assert Path(low).is_file()
    assert "已审核" in (_asset_notes(svc, a["id"], low) or "")
    assert _queue(svc) == []
    dlg.close()


def test_cluster_compare_confirm_sample_without_owner_shows_status(svc, tmp_path):
    """边界：样本行没有归属人物时，[确认是同一人] 不抛、不接受，
    只在状态栏提示（原有行为不变）。"""
    from PySide6.QtWidgets import QDialog

    a, low = _one_low_sample(svc, tmp_path)
    row = _queue(svc)[0]
    row["character_id"] = None
    dlg = _dialog(svc, row)
    dlg.confirm_btn.click()
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert "没有归属人物" in dlg.status_label.text()
    assert _queue(svc) != [], "未确认成功时样本行应留在队列里"
    dlg.close()


def test_cluster_compare_confirm_cluster_still_accepts(svc, tmp_path):
    """回归：聚类行点 [确认是同一人] 仍走 confirm_character、正常接受。"""
    from PySide6.QtWidgets import QDialog

    _add_drama(svc)
    ch = _add_character(svc, "路人", confirmed=0)
    _add_face_asset(svc, ch["id"], str(_real_face(tmp_path, "good.jpg")))
    row = _queue(svc)[0]
    assert row["kind"] == "cluster"
    dlg = _dialog(svc, row)
    dlg.confirm_btn.click()
    assert dlg.result() == QDialog.DialogCode.Accepted, dlg.status_label.text()
    assert svc.store.get_character(ch["id"])["confirmed"] == 1
    assert _queue(svc) == []
    dlg.close()


# ----- N21 端到端：同一人物确认清空后也应弹自动跳转 -----


class _FakeButton:
    def __init__(self, text):
        self.text = text
        self._enabled = True

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled


class _FakeBox:
    """Stand-in for QMessageBox (copied from test_p4_review_jump.py)."""

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


def _main(svc):
    from recast.desktop.main_window import MainWindow

    _app()
    return MainWindow(svc)


def _paint_main(window, svc):
    window._paint(svc.snapshot(), full=True)


def test_low_sample_same_owner_confirm_prompts_jump(svc, tmp_path, monkeypatch):
    """端到端：唯一的低分样本点“确认是同一人”（目标=主人）清空队列后，
    主窗口应把它计为一次确认动作、弹自动跳转提示——以前这里抛错，
    样本清不掉，N21 计数也永远等不到这一笔。"""
    a, low = _one_low_sample(svc, tmp_path)
    window = _main(svc)
    window.timer.stop()
    window.show()
    _paint_main(window, svc)
    assert window._review_was_nonempty is True
    assert window._review_confirms_since_nonempty == 0
    svc.review_assign(low, a["id"], owner_id=a["id"])
    assert _queue(svc) == []
    jumps: list = []
    monkeypatch.setattr(window, "_jump_to_step", lambda sid: jumps.append(sid))
    _FakeBox.click = "去第 ④ 步"
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    _paint_main(window, svc)
    assert window._review_confirms_since_nonempty == 1, "一次同一人物确认应计 1"
    assert _FakeBox.shown, "同一人物确认清空也应弹自动跳转提示"
    assert any("审核已完成" in getattr(box, "title", "") for box in _FakeBox.shown)
    assert jumps == ["replace"]
    assert window._review_was_nonempty is False
    window.close()
