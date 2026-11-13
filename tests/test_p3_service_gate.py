"""P3 batch (second wave): service-side alias linkage (N20) + task
snapshot writes on start/rerun (N11) + shared auth gate for the queue
panel retry (N32).

Only synthetic data + stub scheduler calls: no real video, no TTS
engines, no network. Follows the fixture pattern of
tests/test_main_window_p2.py and tests/test_p3_entry_shared.py.
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


def _add_character(svc, name: str, *, confirmed: int = 1, authorized: int = 0, auth_type: str | None = None):
    fields: dict = {
        "name": name, "display_name": name, "confirmed": confirmed,
        "authorized": authorized, "cluster_key": f"test:{name}",
    }
    if auth_type:
        fields["auth_type"] = auth_type
    return svc.store.insert("characters", **fields)


def _real_face(tmp_path: Path, name: str = "face.jpg") -> Path:
    """A real readable JPEG (200x200 checkerboard)."""
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


def _add_drama(svc, name: str = "剧", *, entry_type: str = "import") -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, root_path=f"/tmp/{name}", episode_count=1,
        entry_type=entry_type,
    )


def _add_task(svc, drama_id, *, status: str = "待执行", workdir=None, **extra):
    fields: dict = {
        "drama_id": drama_id, "episode_name": "第1集",
        "video_path": "/tmp/ep1.mp4", "status": status, "workdir": workdir,
    }
    fields.update(extra)
    return svc.store.insert("tasks", **fields)


def _write_dialogue(workdir: Path, lines: list[dict]) -> None:
    import json

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "dialogue.json").write_text(
        json.dumps(lines, ensure_ascii=False), encoding="utf-8"
    )
    (workdir / "dialogue_mix.wav").write_bytes(b"fake-mix")


def _read_dialogue(workdir: Path) -> list[dict]:
    import json

    return json.loads((workdir / "dialogue.json").read_text(encoding="utf-8"))


# ----- N20：别名表 -----


def test_alias_helpers_roundtrip_and_dedup(svc):
    """(a) 别名读写 helper：写旧名、去重、不把当前名当别名。"""
    ch = _add_character(svc, "温雪")
    svc.ensure_character_folder("温雪")
    assert svc.get_character_aliases(ch["id"]) == []
    assert svc.add_character_alias(ch["id"], "温雪") == []  # 当前名不算别名
    assert svc.add_character_alias(ch["id"], "小雪") == ["小雪"]
    assert svc.add_character_alias(ch["id"], "小雪") == ["小雪"]  # 去重
    assert svc.add_character_alias(ch["id"], "阿雪") == ["小雪", "阿雪"]
    assert svc.get_character_aliases(ch["id"]) == ["小雪", "阿雪"]
    assert svc.get_character_aliases("温雪") == ["小雪", "阿雪"]  # 按显示名也读得到


def test_alias_helpers_without_folder_are_noops(svc):
    """(a) 边界：人物还没有文件夹时，读写别名不建文件夹、不报错。"""
    ch = _add_character(svc, "温雪")
    assert svc.get_character_aliases(ch["id"]) == []
    assert svc.add_character_alias(ch["id"], "小雪") == []
    from recast.core.characters import characters_root

    assert not (characters_root(svc.settings.output_root) / "温雪").exists()


def test_rename_records_old_name_as_alias(svc):
    """(c) 改名入口在 service.confirm_character：旧名进别名表。"""
    ch = _add_character(svc, "温雪")
    svc.ensure_character_folder("温雪")
    renamed = svc.confirm_character(ch["id"], display_name="温雪儿")
    assert renamed is not None
    assert renamed["display_name"] == "温雪儿"
    assert svc.get_character_aliases(ch["id"]) == ["温雪"]
    # 文件夹仍在旧名下，meta 的 name 已同步成新名，aliases 落在旧文件夹里。
    from recast.core.characters import character_folder, load_meta

    old_folder = character_folder(svc.settings.output_root, "温雪")
    meta = load_meta(old_folder)
    assert meta is not None
    assert meta.get("aliases") == ["温雪"]
    assert meta.get("name") == "温雪儿"
    assert not character_folder(svc.settings.output_root, "温雪儿").exists()
    # 再改一次名：旧名追加、不重复；改回已在别名里的名字时，当前名
    # 本身不再重复记，但上一次的旧名（雪儿）仍要追加。
    svc.confirm_character(ch["id"], display_name="雪儿")
    assert svc.get_character_aliases(ch["id"]) == ["温雪", "温雪儿"]
    svc.confirm_character(ch["id"], display_name="温雪")
    assert svc.get_character_aliases(ch["id"]) == ["温雪", "温雪儿", "雪儿"]


def test_rename_same_name_writes_no_alias(svc):
    """(c) 边界：display_name 没变（或只改授权）时不记别名。"""
    ch = _add_character(svc, "温雪")
    svc.ensure_character_folder("温雪")
    svc.confirm_character(ch["id"], display_name="温雪")
    svc.confirm_character(ch["id"], authorized=True, auth_type="自有拍摄")
    assert svc.get_character_aliases(ch["id"]) == []


def test_set_voice_timbre_invalidates_alias_lines(svc, tmp_path: Path):
    """(b) 改名后调音色：旧名说的台词照样标脏、旧混音照样删。"""
    ch = _add_character(svc, "温雪")
    svc.ensure_character_folder("温雪")
    workdir = tmp_path / "works" / "ep1"
    _write_dialogue(workdir, [
        {"speaker": "温雪", "text": "你好", "dirty": False},
        {"speaker": "陆寒峥", "text": "嗯", "dirty": False},
        {"character": "温雪", "text": "再见", "dirty": False},
    ])
    _add_task(svc, _add_drama(svc)["id"], workdir=str(workdir))
    svc.confirm_character(ch["id"], display_name="温雪儿")
    assert svc.get_character_aliases(ch["id"]) == ["温雪"]
    report = svc.set_voice_timbre("温雪儿", pitch=2.0)
    assert len(report["affected"]) == 1
    assert report["affected"][0]["lines"] == 2
    assert report["stale_mix_removed"] == [str(workdir)]
    assert not (workdir / "dialogue_mix.wav").exists()
    lines = _read_dialogue(workdir)
    assert lines[0]["dirty"] is True
    assert lines[1]["dirty"] is False  # 别人的台词不动
    assert lines[2]["dirty"] is True


def test_set_voice_timbre_no_change_invalidates_nothing(svc, tmp_path: Path):
    """(b) 边界：音色没变时不标脏、不删混音（别名也不触发）。"""
    ch = _add_character(svc, "温雪")
    svc.ensure_character_folder("温雪")
    workdir = tmp_path / "works" / "ep1"
    _write_dialogue(workdir, [{"speaker": "温雪", "text": "你好"}])
    _add_task(svc, _add_drama(svc)["id"], workdir=str(workdir))
    svc.confirm_character(ch["id"], display_name="温雪儿")
    report = svc.set_voice_timbre("温雪儿", pitch=0.0, rate=1.0)
    assert report["affected"] == []
    assert report["stale_mix_removed"] == []
    assert (workdir / "dialogue_mix.wav").exists()
    assert _read_dialogue(workdir)[0].get("dirty") is not True


# ----- N11：任务快照写入 -----


def _snapshot_of(svc, task_id: str) -> tuple[list | None, list | None]:
    """Raw stored snapshot fields, JSON-decoded the way the reader does."""
    import json

    row = svc.store.get("tasks", task_id) or {}

    def _decode(value):
        if value is None or value == "":
            return None
        if isinstance(value, list):
            return value
        try:
            parsed = json.loads(value)
        except Exception:
            return value
        return parsed if isinstance(parsed, list) else value

    return _decode(row.get("selected_characters")), _decode(row.get("stage_list"))


def test_start_writes_snapshot_fields(svc, tmp_path: Path, monkeypatch):
    """start() 建 Job 时写选人（已授权人名）与实际阶段清单。"""
    face_ok = _real_face(tmp_path)
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    drama = _add_drama(svc)
    task = _add_task(svc, drama["id"])
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    svc.start([drama["id"]], ["scan", "extract", "replace"])
    assert len(captured) == 1
    assert captured[0].task_id == task["id"]
    selected, stages = _snapshot_of(svc, task["id"])
    assert selected == ["温雪"]
    assert stages == ["scan", "extract", "replace"]


def test_start_without_replace_writes_empty_characters(svc, monkeypatch):
    """没有替换阶段时选人写空列表、阶段照实写，不编造人名。"""
    drama = _add_drama(svc)
    task = _add_task(svc, drama["id"])
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    svc.start([drama["id"]], ["scan"])
    assert len(captured) == 1
    selected, stages = _snapshot_of(svc, task["id"])
    assert selected == []
    assert stages == ["scan"]


def test_start_skips_done_task_without_snapshot(svc, monkeypatch):
    """边界：已完成且不重跑的分集不建 Job、也不写快照。"""
    drama = _add_drama(svc)
    task = _add_task(svc, drama["id"], status="已完成")
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    with pytest.raises(ValueError, match="没有可执行的任务"):
        svc.start([drama["id"]], ["scan"])
    assert captured == []
    selected, stages = _snapshot_of(svc, task["id"])
    assert selected is None
    assert stages is None


def test_rerun_task_refreshes_snapshot(svc, tmp_path: Path, monkeypatch):
    """rerun_task() 用本次实际阶段与选人刷新快照，不沿用旧记录。"""
    face_ok = _real_face(tmp_path)
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    drama = _add_drama(svc)
    old = _add_task(
        svc, drama["id"], status="失败", workdir="/tmp/w",
        selected_characters='["旧人名"]', stage_list='["scan"]',
    )
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    result = svc.rerun_task(old["id"])
    assert result["started"] is True
    assert len(captured) == 1
    new_id = captured[0].task_id
    assert new_id != old["id"]
    selected, stages = _snapshot_of(svc, new_id)
    assert selected == ["温雪"]
    from recast.core.constants import STAGE_GROUP_IDS

    assert stages == list(STAGE_GROUP_IDS)


# ----- N32：队列面板重试关口（三处共用一个入口函数）-----


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

    Same shape as the one in tests/test_main_window_p2.py, but patched
    onto ``recast.desktop.auth_gate_dialog`` (the shared gate's home)
    or ``recast.desktop.main_window`` (the thin wrapper's home).
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


@pytest.fixture(autouse=True)
def _reset_gate_fakebox():
    _FakeBox.shown.clear()
    _FakeBox.click = None
    yield
    _FakeBox.shown.clear()
    _FakeBox.click = None


def _panel(svc, parent=None):
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    dlg = QueuePanelDialog(svc, parent=parent)
    dlg._auto_refresh_timer.stop()
    return dlg


def _failed_task(svc, drama_id, *, episode="第2集", entry_type="import"):
    return _add_task(
        svc, drama_id, episode_name=episode, status="失败",
        progress="人物替换失败", error="显存不足", entry_type=entry_type,
    )


def test_shared_gate_no_missing_passes_without_box(svc, monkeypatch):
    """共用关口正向：没有未授权人物时直接放行，不弹框。"""
    from recast.desktop.auth_gate_dialog import auth_gate_ok

    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    assert auth_gate_ok(None, svc, ["d1"]) is True
    assert _FakeBox.shown == []


def test_panel_retry_blocked_zero_authorized(svc, tmp_path: Path, monkeypatch):
    """零已授权时面板重试被关口拦住：不续跑，[继续] 灰掉。"""
    face_ok = _real_face(tmp_path)
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    drama = _add_drama(svc)
    row = _failed_task(svc, drama["id"])
    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    started: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: started.append((a, k)))
    dlg = _panel(svc)
    dlg._retry_failed(row["id"])
    assert len(_FakeBox.shown) == 1
    box = _FakeBox.shown[0]
    assert "没有已授权人物" in box.text
    assert box._buttons["继续，只替换已授权人物"].isEnabled() is False
    assert started == []
    assert dlg.last_retry is not None and dlg.last_retry["started"] is False
    assert "授权关口" in (dlg.status.text() or "")
    dlg.close()


def test_panel_retry_continue_records_skip_and_starts(svc, tmp_path: Path, monkeypatch):
    """点 [继续]：跳过的人物进持久 auth_skips，续跑照常启动。"""
    face_ok = _real_face(tmp_path)
    ok = _add_character(svc, "温雪", confirmed=1, authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    ch2 = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch2["id"], str(face_ok))
    drama = _add_drama(svc)
    row = _failed_task(svc, drama["id"])
    monkeypatch.setattr(_FakeBox, "click", "继续，只替换已授权人物")
    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    dlg = _panel(svc)
    dlg._retry_failed(row["id"])
    assert len(captured) == 1
    assert captured[0].task_id == row["id"]
    assert dlg.last_retry is not None and dlg.last_retry["started"] is True
    assert "温念" in svc.store.skipped_character_names(drama_id=drama["id"])
    assert "温雪" not in svc.store.skipped_character_names(drama_id=drama["id"])
    dlg.close()


def test_panel_retry_go_library_closes_panel_and_opens_library(
    svc, tmp_path: Path, monkeypatch,
):
    """点 [去人物库补授权]：先关面板，再让主窗口打开人物库，不续跑。"""
    from recast.desktop.main_window import MainWindow

    face_ok = _real_face(tmp_path)
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    drama = _add_drama(svc)
    row = _failed_task(svc, drama["id"])
    monkeypatch.setattr(_FakeBox, "click", "去人物库补授权")
    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    started: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: started.append((a, k)))
    window = MainWindow(svc)
    window.timer.stop()
    opened: list = []
    monkeypatch.setattr(window, "_open_library", lambda: opened.append(True))
    dlg = _panel(svc, parent=window)
    dlg.show()
    assert dlg.isVisible() is True
    dlg._retry_failed(row["id"])
    assert started == []
    assert dlg.last_retry is not None and dlg.last_retry["started"] is False
    assert dlg.isVisible() is False  # 面板已关
    assert opened == [True]  # 主窗口打开了人物库
    dlg.close()
    window.close()


def test_panel_retry_without_main_window_parent_only_closes_panel(
    svc, tmp_path: Path, monkeypatch,
):
    """边界：面板不是主窗口打开的，[去人物库补授权] 只关面板、不炸。"""
    face_ok = _real_face(tmp_path)
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    drama = _add_drama(svc)
    row = _failed_task(svc, drama["id"])
    monkeypatch.setattr(_FakeBox, "click", "去人物库补授权")
    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    started: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: started.append((a, k)))
    dlg = _panel(svc, parent=None)
    dlg.show()
    dlg._retry_failed(row["id"])
    assert started == []
    assert dlg.isVisible() is False
    dlg.close()


def test_panel_retry_skips_gate_for_script_entry(svc, monkeypatch):
    """边界：剧本路重试没有 replace 阶段，不走关口、不弹框、直接续跑。"""
    drama = _add_drama(svc, entry_type="script")
    row = _failed_task(svc, drama["id"], entry_type="script")
    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    captured: list = []
    monkeypatch.setattr(
        svc.scheduler, "enqueue", lambda jobs: captured.extend(jobs) or len(jobs),
    )
    dlg = _panel(svc)
    dlg._retry_failed(row["id"])
    assert _FakeBox.shown == []
    assert len(captured) == 1
    assert dlg.last_retry is not None and dlg.last_retry["started"] is True
    dlg.close()


def test_main_window_gate_still_works_after_delegation(
    svc, tmp_path: Path, monkeypatch,
):
    """主窗口薄壳：旧测试的 patch 点（main_window.QMessageBox）仍生效。"""
    from recast.desktop.main_window import MainWindow

    face_ok = _real_face(tmp_path)
    drama = _add_drama(svc)
    task = _add_task(svc, drama["id"])
    ch = _add_character(svc, "温念", confirmed=1, authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    window = MainWindow(svc)
    window.timer.stop()
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._auth_gate_ok(prompt=True, drama_ids=[drama["id"]]) is False
    assert len(_FakeBox.shown) == 1
    box = _FakeBox.shown[0]
    assert "没有已授权人物" in box.text
    assert box._buttons["继续，只替换已授权人物"].isEnabled() is False
    assert "未授权，本次未替换" not in ((svc.store.get("tasks", task["id"]) or {}).get("progress") or "")
    window.close()
