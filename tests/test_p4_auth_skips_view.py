"""P4 auth_skips viewer: task-history entry point for N18 skip records.

Uses a throwaway SQLite file in tmp_path — never the user's real library.
Synthetic tasks / skip rows only; viewing is read-only and never writes.
"""

from __future__ import annotations

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


def _drama(svc, name: str = "演示剧") -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, new_name=name,
        root_path=f"C:/demo/{name}", episode_count=1,
    )


def _task(svc, drama_id: str, *, episode: str, status: str = "已完成", **extra) -> dict:
    fields: dict = {
        "drama_id": drama_id,
        "episode_name": episode,
        "video_path": f"C:/demo/{episode}.mp4",
        "status": status,
        "progress": "等待执行",
        "progress_pct": 0,
    }
    fields.update(extra)
    return svc.store.insert("tasks", **fields)


def _skip(svc, drama_id: str, name: str, *, task_id: str | None = None, reason: str = "未授权"):
    return svc.store.record_auth_skip(drama_id, name, task_id=task_id, reason=reason)


# ----- core: task-scoped records -----

def test_get_auth_skips_for_task_shows_name_reason_time(svc):
    from recast.core.task_history import format_auth_skips, get_auth_skips_for_task

    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集")
    _skip(svc, drama["id"], "温雪", task_id=task["id"], reason="未授权")
    _skip(svc, drama["id"], "陈秋薇", task_id=task["id"], reason="未授权")
    # A skip belonging to another task must not leak into this one.
    other = _task(svc, drama["id"], episode="第2集")
    _skip(svc, drama["id"], "陆寒峥", task_id=other["id"], reason="未授权")

    records = get_auth_skips_for_task(svc.store, task["id"])
    assert [r["character_name"] for r in records] == ["温雪", "陈秋薇"]
    text = format_auth_skips(records)
    assert "温雪" in text and "陈秋薇" in text
    assert "陆寒峥" not in text
    assert "未授权" in text
    # Every row carries a timestamp (skipped_at).
    assert all(r.get("skipped_at") for r in records)
    assert records[0]["skipped_at"] in text


def test_get_auth_skips_for_task_empty_and_missing_task(svc):
    from recast.core.task_history import NO_AUTH_SKIPS_TEXT, format_auth_skips, get_auth_skips_for_task

    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集")
    assert get_auth_skips_for_task(svc.store, task["id"]) == []
    assert format_auth_skips([]) == NO_AUTH_SKIPS_TEXT
    assert format_auth_skips(None) == "本次无跳过记录"
    assert get_auth_skips_for_task(svc.store, "no-such-task") == []


def test_drama_level_skip_visible_from_task(svc):
    """Drama-level rows (task_id empty, written when no episode tasks
    existed at gate time) are still visible from one of its tasks."""
    from recast.core.task_history import get_auth_skips_for_task

    drama = _drama(svc)
    _skip(svc, drama["id"], "温雪", task_id=None, reason="未授权")
    task = _task(svc, drama["id"], episode="第1集")
    records = get_auth_skips_for_task(svc.store, task["id"])
    assert [r["character_name"] for r in records] == ["温雪"]


def test_drama_level_skip_not_shown_for_other_drama(svc):
    from recast.core.task_history import get_auth_skips_for_task

    drama_a = _drama(svc, "A剧")
    drama_b = _drama(svc, "B剧")
    _skip(svc, drama_a["id"], "温雪", task_id=None, reason="未授权")
    task_b = _task(svc, drama_b["id"], episode="第1集")
    assert get_auth_skips_for_task(svc.store, task_b["id"]) == []


# ----- core: tolerant formatting (N40 annotation fields) -----

def test_format_shows_annotation_fields_generically(svc, monkeypatch):
    """A record carrying an extra scope/annotation field (e.g. what the
    N40 narrowing worker may add) is displayed, without this module
    hard-coding that field's name."""
    from recast.core.task_history import format_auth_skips, get_auth_skips_for_task

    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集")
    annotated = {
        "id": "skip-1",
        "drama_id": drama["id"],
        "task_id": task["id"],
        "character_name": "温雪",
        "reason": "未授权",
        "skipped_at": "2026-11-13 00:00:00",
        "scope": "全局未授权，非本剧人物",
        "note": "待核查",
    }
    monkeypatch.setattr(
        svc.store, "list_auth_skips",
        lambda **kw: [dict(annotated)] if kw.get("task_id") == task["id"] else [],
    )
    text = format_auth_skips(get_auth_skips_for_task(svc.store, task["id"]))
    assert "温雪" in text and "未授权" in text and "2026-11-13 00:00:00" in text
    assert "全局未授权，非本剧人物" in text
    assert "待核查" in text


def test_format_tolerates_missing_name_reason_time():
    from recast.core.task_history import format_auth_skip_record, format_auth_skips

    text = format_auth_skips([{"character_name": "", "reason": None}])
    assert "未记名" in text or "未记录" in text
    assert "原因未记录" in text
    assert "时间未记录" in text
    # A bare empty dict must not crash the viewer.
    assert format_auth_skip_record({})


# ----- core: history entry carries the viewer data -----

def test_history_entry_carries_auth_skips_text(svc):
    from recast.core.task_history import get_history_entry

    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集")
    _skip(svc, drama["id"], "温雪", task_id=task["id"], reason="未授权")
    entry = get_history_entry(svc.store, task["id"])
    assert [r["character_name"] for r in entry["auth_skips"]] == ["温雪"]
    assert "温雪" in entry["auth_skips_text"]
    assert "未授权" in entry["auth_skips_text"]


def test_history_entry_without_skips_reports_none(svc):
    from recast.core.task_history import get_history_entry

    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集")
    entry = get_history_entry(svc.store, task["id"])
    assert entry["auth_skips"] == []
    assert entry["auth_skips_text"] == "本次无跳过记录"
    assert get_history_entry(svc.store, "no-such-task") is None


# ----- dialog: detail line + viewer -----

def test_dialog_detail_counts_skips_and_viewer_lists_them(svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集", status="已完成")
    _skip(svc, drama["id"], "温雪", task_id=task["id"], reason="未授权")
    _skip(svc, drama["id"], "陈秋薇", task_id=task["id"], reason="未授权")
    dlg = TaskHistoryDialog(svc)
    assert dlg.table.rowCount() == 1
    dlg.table.setCurrentCell(0, 0)
    assert "跳过记录：共 2 条" in dlg.detail.toPlainText()
    assert dlg.skips_btn.isEnabled()

    viewer = dlg._build_auth_skips_dialog(task["id"])
    body = viewer.body.toPlainText()
    assert "温雪" in body and "陈秋薇" in body
    assert "未授权" in body
    assert viewer.body.isReadOnly()
    dlg.close()
    viewer.close()


def test_dialog_no_skips_shows_empty_text_not_blank(svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集", status="已完成")
    dlg = TaskHistoryDialog(svc)
    dlg.table.setCurrentCell(0, 0)
    assert "跳过记录：本次无跳过记录" in dlg.detail.toPlainText()
    viewer = dlg._build_auth_skips_dialog(task["id"])
    assert viewer.body.toPlainText() == "本次无跳过记录"
    dlg.close()
    viewer.close()


def test_dialog_skips_button_disabled_without_selection(svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    dlg = TaskHistoryDialog(svc)
    assert dlg.table.rowCount() == 0
    assert not dlg.skips_btn.isEnabled()
    assert not dlg.rerun_btn.isEnabled()
    dlg.close()


def test_dialog_viewer_for_missing_task(svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    dlg = TaskHistoryDialog(svc)
    viewer = dlg._build_auth_skips_dialog("no-such-task")
    assert "已不在库中" in viewer.body.toPlainText()
    dlg.close()
    viewer.close()


def test_clicking_skips_button_opens_viewer(svc, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集", status="已完成")
    _skip(svc, drama["id"], "温雪", task_id=task["id"], reason="未授权")
    dlg = TaskHistoryDialog(svc)
    dlg.table.setCurrentCell(0, 0)
    # Don't run a modal loop in the test: stub exec() to return at once.
    monkeypatch.setattr(QDialog, "exec", lambda self: 0)
    dlg.skips_btn.click()
    assert dlg.last_skips_dialog is not None
    assert "温雪" in dlg.last_skips_dialog.body.toPlainText()
    assert "未授权" in dlg.last_skips_dialog.body.toPlainText()
    dlg.close()


# ----- read-only: viewing must not change the store -----

def test_viewing_skips_does_not_write(svc):
    from recast.core.task_history import get_auth_skips_for_task, get_history_entry
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集", status="已完成")
    _skip(svc, drama["id"], "温雪", task_id=task["id"], reason="未授权")
    before = len(svc.store.list_auth_skips(drama_id=drama["id"]))
    before_tasks = len(svc.store.list_rows("tasks"))

    get_auth_skips_for_task(svc.store, task["id"])
    get_history_entry(svc.store, task["id"])
    dlg = TaskHistoryDialog(svc)
    dlg.table.setCurrentCell(0, 0)
    viewer = dlg._build_auth_skips_dialog(task["id"])
    viewer.body.toPlainText()

    assert len(svc.store.list_auth_skips(drama_id=drama["id"])) == before == 1
    assert len(svc.store.list_rows("tasks")) == before_tasks == 1
    dlg.close()
    viewer.close()


def test_viewing_does_not_duplicate_on_repeat_open(svc):
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    drama = _drama(svc)
    task = _task(svc, drama["id"], episode="第1集", status="已完成")
    _skip(svc, drama["id"], "温雪", task_id=task["id"], reason="未授权")
    dlg = TaskHistoryDialog(svc)
    dlg.table.setCurrentCell(0, 0)
    for _ in range(3):
        viewer = dlg._build_auth_skips_dialog(task["id"])
        assert viewer.body.toPlainText().count("温雪") == 1
        viewer.close()
    assert len(svc.store.list_auth_skips(task_id=task["id"])) == 1
    dlg.close()
