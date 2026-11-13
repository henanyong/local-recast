"""P4 batch: N39 single-video import collision only matches the first
episode + N30 residual (folder-scan identity is the first episode file).

Only synthetic data + stub message boxes: no real video, no TTS engines, no
network. Follows the fixture pattern of tests/test_p3_main_window.py.
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


class _FakeBox:
    """Minimal stand-in for QMessageBox.question / warning."""

    from PySide6.QtWidgets import QMessageBox as _RealBox

    StandardButton = _RealBox.StandardButton
    questions: list = []
    warnings: list = []
    question_answer = _RealBox.StandardButton.No

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
    _FakeBox.questions.clear()
    _FakeBox.warnings.clear()
    _FakeBox.question_answer = _FakeBox.StandardButton.No
    yield
    _FakeBox.questions.clear()
    _FakeBox.warnings.clear()
    _FakeBox.question_answer = _FakeBox.StandardButton.No


def _two_episode_folder(tmp_path: Path, name: str = "drama") -> tuple[Path, Path, Path]:
    folder = tmp_path / name
    folder.mkdir()
    ep1 = folder / "ep1.mp4"
    ep2 = folder / "ep2.mp4"
    ep1.write_bytes(b"fake")
    ep2.write_bytes(b"fake")
    return folder, ep1, ep2


def _question_texts() -> list[str]:
    return [" ".join(str(a) for a in q) for q in _FakeBox.questions]


# ----- N39：非首集视频单独导入也要撞上 -----


def test_find_import_collision_hits_non_first_episode_task(svc, tmp_path: Path):
    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    drama = svc.store.list_dramas()[0]
    window = _main(svc)
    window.timer.stop()
    # 首集：老规则撞上。
    assert window._find_import_collision(str(ep1))["id"] == drama["id"]
    # 非首集：已是该剧目的分集任务，也必须撞上（N39）。
    assert window._find_import_collision(str(ep2))["id"] == drama["id"]
    # 同文件夹、从未入库的视频：仍不是碰撞。
    ep3 = folder / "ep3.mp4"
    ep3.write_bytes(b"fake")
    assert window._find_import_collision(str(ep3)) is None
    window.close()


def test_single_video_import_non_first_episode_confirmed_merges(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox as RealBox

    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    drama = svc.store.list_dramas()[0]
    before = {t["id"]: t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])}
    window = _main(svc)
    window.timer.stop()
    _FakeBox.question_answer = RealBox.StandardButton.Yes
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(ep2)) is True
    assert any("并入" in text for text in _question_texts()), "非首集碰撞应问是否并入已有剧目"
    # 确认后并入已有剧目行，不建新行、不重复建任务。
    dramas = svc.store.list_dramas()
    assert len(dramas) == 1 and dramas[0]["id"] == drama["id"]
    after = {t["id"]: t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])}
    assert after == before
    window.close()


def test_single_video_import_non_first_episode_cancel_keeps_drama(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox as RealBox

    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    drama = svc.store.list_dramas()[0]
    before = {t["id"]: t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])}
    window = _main(svc)
    window.timer.stop()
    _FakeBox.question_answer = RealBox.StandardButton.No
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(ep2)) is False
    assert _FakeBox.questions, "非首集再次入库时应先弹确认框"
    assert len(svc.store.list_dramas()) == 1
    after = {t["id"]: t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])}
    assert after == before
    window.close()


def test_commit_hit_merges_single_file_by_task_video(svc, tmp_path: Path):
    """Service 层（无 UI）：_commit_hit 按 video_path 命中已有任务即并入。"""
    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    drama = svc.store.list_dramas()[0]
    before = {t["id"]: t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])}
    svc.scan(str(ep2))
    dramas = svc.store.list_dramas()
    assert len(dramas) == 1 and dramas[0]["id"] == drama["id"]
    after = {t["id"]: t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])}
    assert after == before
    # 身份仍是首集，不因并入一集就改指。
    assert dramas[0]["source_path"] == str(ep1)


def test_single_video_import_same_folder_unlisted_video_is_new_drama(svc, tmp_path: Path, monkeypatch):
    """边界：同文件夹、从未入库的视频仍建新剧目（N30 规则不变）。"""
    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    ep3 = folder / "ep3.mp4"
    ep3.write_bytes(b"fake")
    window = _main(svc)
    window.timer.stop()
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    assert window._import_single_video_file(str(ep3)) is True
    assert _FakeBox.questions == [], "从未入库的视频不是碰撞，不应弹确认框"
    assert len(svc.store.list_dramas()) == 2
    window.close()


# ----- N30 残留：删首集再扫文件夹，要认出旧行 -----


def test_folder_rescan_after_first_episode_deleted_merges_into_old_drama(svc, tmp_path: Path):
    """Service 层：删首集再扫，旧行被认出并入，不建新行；已删首集的任务落掉。"""
    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    drama = svc.store.list_dramas()[0]
    ep1.unlink()
    svc.scan(str(folder))
    dramas = svc.store.list_dramas()
    assert len(dramas) == 1 and dramas[0]["id"] == drama["id"]
    tasks = svc.store.tasks_for_drama(drama["id"])
    assert [t["video_path"] for t in tasks] == [str(ep2)]
    # 身份改指现存的首集，不再指向已删文件。
    assert dramas[0]["source_path"] == str(ep2)


def test_folder_rescan_after_first_episode_deleted_prompts_and_merges(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox as RealBox

    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    drama = svc.store.list_dramas()[0]
    ep1.unlink()
    window = _main(svc)
    window.timer.stop()
    window.root_edit.setText(str(folder))
    _FakeBox.question_answer = RealBox.StandardButton.Yes
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._scan()
    assert any("并入" in text for text in _question_texts()), "删首集再扫应提示是否并入/恢复"
    dramas = svc.store.list_dramas()
    assert len(dramas) == 1 and dramas[0]["id"] == drama["id"]
    assert [t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])] == [str(ep2)]
    window.close()


def test_folder_rescan_after_first_episode_deleted_cancel_skips_scan(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox as RealBox

    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    drama = svc.store.list_dramas()[0]
    ep1.unlink()
    window = _main(svc)
    window.timer.stop()
    window.root_edit.setText(str(folder))
    _FakeBox.question_answer = RealBox.StandardButton.No
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._scan()
    # 选“否”：本次不扫描，旧行原样保留（连已删首集的任务都还在，等用户决定）。
    dramas = svc.store.list_dramas()
    assert len(dramas) == 1 and dramas[0]["id"] == drama["id"]
    assert dramas[0]["source_path"] == str(ep1)
    assert sorted(
        t["video_path"] for t in svc.store.tasks_for_drama(drama["id"])
    ) == [str(ep1), str(ep2)]
    window.close()


def test_fresh_folder_scan_creates_drama_without_prompt(svc, tmp_path: Path, monkeypatch):
    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    window = _main(svc)
    window.timer.stop()
    window.root_edit.setText(str(folder))
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._scan()
    assert _FakeBox.questions == [], "全新文件夹扫描不应弹确认框"
    assert len(svc.store.list_dramas()) == 1
    window.close()


def test_folder_rescan_unchanged_does_not_prompt(svc, tmp_path: Path, monkeypatch):
    folder, ep1, ep2 = _two_episode_folder(tmp_path)
    svc.scan(str(folder))
    window = _main(svc)
    window.timer.stop()
    window.root_edit.setText(str(folder))
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._scan()
    assert _FakeBox.questions == [], "首集还在的正常重扫不应弹确认框"
    assert len(svc.store.list_dramas()) == 1
    window.close()
