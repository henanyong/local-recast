"""失败任务卡片的纯计算（R6 拆分，P4）。

主窗口 ``_failed_entries`` / ``_last_checkpoint_stage`` 原来直接读
store、拼失败阶段与原因：这里是唯一来源，主窗口只留取数与刷 UI
的薄委托（``MainWindow._failed_entries`` / ``_last_checkpoint_stage``），
卡片的建 UI 在 ``recast.desktop.failed_cards``。

行为与抽取前一字不差：失败阶段取 runs 表里最新一条失败记录的
step（组标签，去编号）；原因取 run.error → task.error →
drama.error → task.progress；检查点走到哪一步就补一句
“已完成到：X”，重试时引擎按检查点跳过。
"""

from __future__ import annotations

from typing import Any

from recast.core.constants import STAGE_LABELS, ST_FAILED

# 检查点子步骤 → 所属阶段（用于“失败前已完成到哪一步”）。
# 原在 main_window.py 模块级（``_CHECKPOINT_STAGE``），R6 移到这里。
CHECKPOINT_STAGE: dict[str, str] = {
    "scan": "scan",
    "ingest": "extract",
    "scenes": "extract",
    "asr": "extract",
    "ocr": "extract",
    "merge": "extract",
    "track": "library",
    "identity": "library",
    "pose": "pose",
    "replace": "replace",
    "compose": "compose",
}


def _strip_stage_number(label: str) -> str:
    """去掉阶段名前面的数字编号（与 main_window 的同名函数同一正则）。"""
    import re

    return re.sub(r"^\d+\s*", "", label or "").strip()


def _strip_run_step_label(step: str) -> str:
    return " / ".join(_strip_stage_number(part) for part in (step or "").split(" / "))


def last_checkpoint_stage(store: Any, task_id: str) -> str | None:
    """该任务检查点走到的最后一步，换算成阶段名（失败前已完成到哪）。

    纯取数：只读 store 的 checkpoints 表，不碰 UI。读失败返回 None。
    """
    try:
        rows = store.list_rows(
            "checkpoints", "task_id = ?", (task_id,), order="created_at"
        )
    except Exception:
        return None
    if not rows:
        return None
    sid = CHECKPOINT_STAGE.get(rows[-1].get("step") or "")
    if not sid:
        return None
    return _strip_stage_number(STAGE_LABELS.get(sid, sid))


def collect_failed_entries(store: Any) -> list[dict[str, str]]:
    """失败的剧目：失败阶段 + 失败原因。

    只读 store（dramas / tasks / runs / checkpoints），不碰 UI。
    每项 ``{"drama_id", "name", "stage", "reason"}``。
    """
    entries: list[dict[str, str]] = []
    for drama in store.list_dramas():
        tasks = store.tasks_for_drama(drama["id"])
        failed_tasks = [t for t in tasks if t.get("status") == ST_FAILED]
        if not failed_tasks and drama.get("status") != ST_FAILED:
            continue
        task = failed_tasks[0] if failed_tasks else (tasks[0] if tasks else None)
        try:
            runs = store.list_runs(drama_id=drama["id"])
        except Exception:
            runs = []
        failed_runs = [r for r in runs if r.get("status") == "failed"]
        failed_runs.sort(key=lambda r: (r.get("started_at") or "", r.get("created_at") or ""))
        stage = "未知阶段"
        reason = ""
        if failed_runs:
            last = failed_runs[-1]
            stage = _strip_run_step_label(last.get("step") or "") or "未知阶段"
            reason = (last.get("error") or "").strip()
        if not reason and task:
            reason = (task.get("error") or task.get("progress") or "").strip()
        if not reason:
            reason = (drama.get("error") or drama.get("progress") or "失败原因未记录").strip()
        done_to = last_checkpoint_stage(store, task["id"]) if task else None
        if done_to:
            reason = f"{reason}（已完成到：{done_to}）" if reason else f"已完成到：{done_to}"
        name = drama.get("new_name") or drama.get("origin_name") or drama["id"]
        entries.append(
            {"drama_id": drama["id"], "name": name, "stage": stage, "reason": reason}
        )
    return entries
