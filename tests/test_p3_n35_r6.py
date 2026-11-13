"""P3 第 3 波（最后一波）：N35 入口–阶段映射去重 + R6 拆分。

* N35：``queue_panel.py`` 的 ``_ENTRY_STAGES`` 拷贝改成 import 共用的
  ``ENTRY_STAGES``（``recast/core/entry_shared.py``）；这里断言同一性、
  值相等，以及按入口过滤阶段的重试行为仍与原来一致。
* R6-A：逐镜头增强对话框搬进 ``recast/desktop/shot_enhance_dialog.py``；
  这里断言新 import 路径可用、主窗口薄调用仍通、N29 文案原样保留。
* R6-B：审核进度纯计算抽进 ``recast/core/review_progress.py``；这里
  断言纯函数的分母规则（N17）与空输入行为，两窗口的集成行为由既有
  测试覆盖（主窗口 test_main_window_p1/p2 的 review_progress 系列、
  人物库 test_library_p2 的 N22 系列与 test_p3_library 的 N37）。

只用合成数据 + tmp sqlite，不跑视频、不跑 runner、不联网。
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


def _task(svc, drama_id: str, *, episode: str = "第1集", status: str = "待执行", **extra) -> dict:
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


def _row_for(dlg, episode: str) -> int:
    for i in range(dlg.table.rowCount()):
        item = dlg.table.item(i, 0)
        if item is not None and item.text().endswith(episode):
            return i
    raise AssertionError(f"no row for {episode!r} in panel")


# ----- N35：去重 -----


def test_n35_queue_panel_uses_shared_entry_stages():
    """queue_panel 不再自带拷贝：模块属性就是共用的 ENTRY_STAGES 本体。"""
    from recast.core import entry_shared
    from recast.desktop import queue_panel

    assert queue_panel.ENTRY_STAGES is entry_shared.ENTRY_STAGES
    assert queue_panel.ENTRY_STAGES == {
        "import": frozenset({"scan", "extract", "library", "replace", "compose", "export"}),
        "script": frozenset({"template", "generate", "dub", "compose", "export"}),
    }
    assert not hasattr(queue_panel, "_ENTRY_STAGES")


def test_n35_retry_import_uses_import_stages(svc, monkeypatch):
    """导入路失败重试：只重跑 import 的 6 个阶段（STAGE_GROUP_IDS 顺序）。"""
    from recast.core.constants import STAGE_GROUP_IDS
    from recast.core.entry_shared import ENTRY_STAGES
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第2集", status="失败",
          progress="人物替换失败", error="显存不足", entry_type="import")
    captured: list = []
    monkeypatch.setattr(svc, "start", lambda ids, stages=None: captured.append((ids, stages)))
    dlg = QueuePanelDialog(svc)
    retry = dlg.table.cellWidget(_row_for(dlg, "第2集"), 5)
    retry.click()
    assert dlg.last_retry is not None and dlg.last_retry["started"] is True
    assert len(captured) == 1
    ids, stages = captured[0]
    assert ids == [drama["id"]]
    assert stages == [sid for sid in STAGE_GROUP_IDS if sid in ENTRY_STAGES["import"]]
    assert "replace" in stages and "generate" not in stages
    dlg.close()


def test_n35_retry_script_uses_script_stages(svc, monkeypatch):
    """剧本路失败重试：只重跑 script 的 5 个阶段，不含 replace。"""
    from recast.core.constants import STAGE_GROUP_IDS
    from recast.core.entry_shared import ENTRY_STAGES
    from recast.desktop.queue_panel import QueuePanelDialog

    _app()
    drama = _drama(svc)
    _task(svc, drama["id"], episode="第2集", status="失败",
          progress="生成失败", error="显存不足", entry_type="script")
    captured: list = []
    monkeypatch.setattr(svc, "start", lambda ids, stages=None: captured.append((ids, stages)))
    dlg = QueuePanelDialog(svc)
    retry = dlg.table.cellWidget(_row_for(dlg, "第2集"), 5)
    retry.click()
    assert dlg.last_retry is not None and dlg.last_retry["started"] is True
    assert len(captured) == 1
    ids, stages = captured[0]
    assert ids == [drama["id"]]
    assert stages == [sid for sid in STAGE_GROUP_IDS if sid in ENTRY_STAGES["script"]]
    assert "replace" not in stages and "generate" in stages
    dlg.close()


# ----- R6-A：逐镜头增强对话框 -----


def _shot_task(svc, drama_id: str, tmp_path: Path):
    workdir = tmp_path / "wd"
    workdir.mkdir(exist_ok=True)
    task = svc.store.insert(
        "tasks", drama_id=drama_id, episode_name="第1集",
        video_path="/tmp/ep1.mp4", status="待执行", workdir=str(workdir),
    )
    for code in ("S1", "S2"):
        svc.store.insert(
            "shots", task_id=task["id"], shot_code=code, index_no=int(code[1]),
            start_time=0.0, end_time=1.0,
        )
    return task


def test_r6a_shot_enhance_dialog_lists_shots_and_toggles(svc, tmp_path: Path):
    """新模块对话框：列出镜头、N29 文案原样、开关写 flags。"""
    from PySide6.QtWidgets import QLabel

    from recast.desktop.shot_enhance_dialog import (
        ShotEnhanceDialog,
        collect_shot_enhance_rows,
    )

    _app()
    drama = _drama(svc)
    task = _shot_task(svc, drama["id"], tmp_path)
    svc.set_shot_enhance(task["id"], "S2", True)  # 预置一个已开的开关
    rows = collect_shot_enhance_rows(svc, drama["id"])
    assert {(t["id"], s.get("shot_code")) for t, s in rows} == {
        (task["id"], "S1"), (task["id"], "S2"),
    }
    dlg = ShotEnhanceDialog(svc, drama["id"], rows, drama_name="演示剧")
    assert dlg.windowTitle() == "逐镜头增强 — 演示剧"
    hint_texts = [lbl.text() for lbl in dlg.findChildren(QLabel)]
    assert any("两条路都支持" in t for t in hint_texts)
    assert any("拼接后按标记时段增强" in t for t in hint_texts)
    assert not any("暂不支持" in t for t in hint_texts)
    assert {(tid, code) for tid, code in dlg.boxes} == {
        (task["id"], "S1"), (task["id"], "S2"),
    }
    assert dlg.boxes[(task["id"], "S2")].isChecked() is True
    assert dlg.boxes[(task["id"], "S1")].isChecked() is False
    dlg.boxes[(task["id"], "S1")].setChecked(True)
    assert svc.shot_enhance_flags(task["id"]).get("S1") is True
    dlg.boxes[(task["id"], "S1")].setChecked(False)
    assert "S1" not in svc.shot_enhance_flags(task["id"])
    assert dlg.error_label.text() == ""
    dlg.close()


def test_r6a_shot_enhance_dialog_toggle_failure_rolls_back(svc, tmp_path: Path, monkeypatch):
    """开关写库失败：勾选回滚、错误行写清镜头号。"""
    from recast.desktop.shot_enhance_dialog import (
        ShotEnhanceDialog,
        collect_shot_enhance_rows,
    )

    _app()
    drama = _drama(svc)
    task = _shot_task(svc, drama["id"], tmp_path)
    rows = collect_shot_enhance_rows(svc, drama["id"])
    dlg = ShotEnhanceDialog(svc, drama["id"], rows, drama_name="演示剧")
    monkeypatch.setattr(
        svc, "set_shot_enhance",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("磁盘满了")),
    )
    dlg.boxes[(task["id"], "S1")].setChecked(True)
    assert dlg.boxes[(task["id"], "S1")].isChecked() is False
    assert "S1" in dlg.error_label.text()
    assert "开关没存上" in dlg.error_label.text()
    dlg.close()


def test_r6a_collect_shot_enhance_rows_skips_blank_codes(svc, tmp_path: Path):
    """空 shot_code 的镜头行跳过，只留有效镜头。"""
    from recast.desktop.shot_enhance_dialog import collect_shot_enhance_rows

    drama = _drama(svc)
    task = _shot_task(svc, drama["id"], tmp_path)
    svc.store.insert(
        "shots", task_id=task["id"], shot_code=" ", index_no=9,
        start_time=0.0, end_time=1.0,
    )
    rows = collect_shot_enhance_rows(svc, drama["id"])
    assert [s.get("shot_code") for _, s in rows] == ["S1", "S2"]


class _FakeBox:
    """只替 information() 静态调用的最小 QMessageBox 替身。"""

    infos: list = []

    @staticmethod
    def information(*args, **kwargs):
        _FakeBox.infos.append(args[2] if len(args) > 2 else kwargs.get("text", ""))
        return None


def test_r6a_main_window_thin_call_still_wires_dialog(svc, tmp_path: Path, monkeypatch):
    """主窗口薄调用仍通：打开后 _shot_enhance_dialog/_shot_enhance_boxes 就位。"""
    from PySide6.QtWidgets import QDialog, QLabel

    from recast.desktop.main_window import MainWindow
    from recast.desktop.shot_enhance_dialog import ShotEnhanceDialog

    _app()
    drama = _drama(svc)
    task = _shot_task(svc, drama["id"], tmp_path)
    window = MainWindow(svc)
    window.timer.stop()
    window._paint(svc.snapshot(), full=True)
    monkeypatch.setattr(QDialog, "exec", lambda self: 1)
    monkeypatch.setattr("recast.desktop.main_window.QMessageBox", _FakeBox)
    window._open_shot_enhance()
    assert isinstance(window._shot_enhance_dialog, ShotEnhanceDialog)
    hint_texts = [lbl.text() for lbl in window._shot_enhance_dialog.findChildren(QLabel)]
    assert any("两条路都支持" in t for t in hint_texts)
    assert any("拼接后按标记时段增强" in t for t in hint_texts)
    assert {(tid, code) for tid, code in window._shot_enhance_boxes} == {
        (task["id"], "S1"), (task["id"], "S2"),
    }
    window._shot_enhance_boxes[(task["id"], "S1")].setChecked(True)
    assert svc.shot_enhance_flags(task["id"]).get("S1") is True
    window.close()


# ----- R6-B：审核进度纯计算 -----


def test_r6b_review_progress_stats_denominator_includes_low_samples():
    """分母 = 已确认 + 待审核全部项（N17）：1 确认 + 聚类 1 + 低分 1 = 3，33%。"""
    from recast.core.review_progress import review_progress_stats

    stats = review_progress_stats(
        [{"confirmed": 1}, {"confirmed": 0}],
        [{"kind": "cluster"}, {"kind": "sample"}],
    )
    assert stats == {
        "confirmed": 1,
        "pending": 2,
        "clusters": 1,
        "low": 1,
        "total": 3,
        "percent": 33,
    }
    assert stats["total"] == stats["confirmed"] + stats["pending"]


def test_r6b_review_progress_stats_empty_and_none_inputs():
    """空输入与 None 输入都返回全零，percent 为 0 不抛错。"""
    from recast.core.review_progress import review_progress_stats

    assert review_progress_stats([], []) == {
        "confirmed": 0, "pending": 0, "clusters": 0,
        "low": 0, "total": 0, "percent": 0,
    }
    assert review_progress_stats(None, None)["total"] == 0
    # 零确认 + 仅低分样本：total 1、percent 0（不报 100%，N22）。
    only_low = review_progress_stats([], [{"kind": "sample"}])
    assert only_low == {
        "confirmed": 0, "pending": 1, "clusters": 0,
        "low": 1, "total": 1, "percent": 0,
    }


def test_r6b_both_windows_import_the_same_function():
    """主窗口与人物库 import 的是同一份纯函数，不是第二份拷贝。"""
    from recast.core import review_progress
    from recast.desktop import library_window, main_window

    assert main_window.review_progress_stats is review_progress.review_progress_stats
    assert library_window.review_progress_stats is review_progress.review_progress_stats
