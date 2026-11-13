"""Task history: read-only queries over the desktop SQLite ``tasks`` table.

What this module does
---------------------
* List past tasks newest-first, optionally filtered by status.
* Build a per-task config snapshot from the fields the ``tasks`` row actually
  carries (template, entry type, source script / dialogue / storyboard,
  pack snapshot). Anything the table does not store is reported as
  "未记录" — never invented.
* Build a re-run task dict from a past task (same config, fresh id, status
  reset to 待执行). It is returned, NOT inserted and NOT executed.
* Delete one history row (the ``tasks`` row only; the output file on disk
  is never touched).
* Read the auth-gate skip records (``auth_skips``) for one task — the
  P4 viewer entry point. Read-only: viewing never writes to the store.

What this module does NOT do (next round)
------------------------------------------
* It does not start execution. The runner/scheduler owns execution.
* It does not insert the re-run row. The caller inserts it and starts it.

Next-round wiring (UI / service)
---------------------------------
The history dialog (``recast/desktop/task_history_dialog.py``) already calls
these functions; the main-window entry button is still to be wired. When it
is, the wiring is one line plus the insert + start calls::

    from recast.desktop.task_history_dialog import TaskHistoryDialog

    dlg = TaskHistoryDialog(self.service, parent=self)  # service.store is read
    dlg.exec()

Re-run from a history entry (service layer, next round)::

    from recast.core.task_history import build_rerun_task

    rerun = build_rerun_task(service.store, task_id)  # pure dict, not inserted
    new_row = service.store.insert("tasks", **rerun)  # now it exists
    # ... then start it through the scheduler/runner (execution entry point),
    # exactly the way "开始执行" starts a normal task. Do NOT call the
    # runner's internal stage functions directly from the dialog.

Status values stored in the table are the Chinese constants from
``recast/core/constants.py`` (待补资料 / 待执行 / 处理中 / 待人工检查 /
已完成 / 失败). This module normalises them to a small canonical kind for
filtering and colouring, and always shows the stored text verbatim in the UI.
"""

from __future__ import annotations

from typing import Any

from recast.core.constants import (
    ST_DONE,
    ST_FAILED,
    ST_NEED_ASSETS,
    ST_PENDING,
    ST_REVIEW,
    ST_RUNNING,
)
from recast.store.db import Store, _now, new_id

NOT_RECORDED = "未记录"

# Stored status text -> canonical kind. Anything not listed here is "unknown"
# and is shown verbatim; the history must never hide a row it cannot classify.
STATUS_KINDS: dict[str, str] = {
    ST_NEED_ASSETS: "need_assets",
    ST_PENDING: "pending",
    ST_RUNNING: "running",
    ST_REVIEW: "review",
    ST_DONE: "done",
    ST_FAILED: "failed",
}

# Canonical kind -> (display label, colour) for the history table.
STATUS_DISPLAY: dict[str, tuple[str, str]] = {
    "need_assets": (ST_NEED_ASSETS, "#ff9800"),
    "pending": (ST_PENDING, "#9e9e9e"),
    "running": (ST_RUNNING, "#2196f3"),
    "review": (ST_REVIEW, "#ff9800"),
    "done": (ST_DONE, "#4caf50"),
    "failed": (ST_FAILED, "#f44336"),
    "unknown": ("未知状态", "#9e9e9e"),
}

# Accepts either a canonical kind ("failed") or the stored text ("失败").
# Both map to the stored text used in the WHERE clause. NOTE: the
# kind -> label overlay must come second — the identity mapping alone
# would make "failed" query the literal text "failed", which no row has.
_STATUS_FILTER_ALIASES: dict[str, str] = {
    **{kind: kind for kind in STATUS_KINDS.values()},
    **{kind: label for label, kind in STATUS_KINDS.items()},
}


def normalize_status(status: str | None) -> tuple[str, str]:
    """Map stored status text to ``(kind, display_label)``.

    Unknown / empty values keep the stored text as the label so the row is
    never hidden or mislabelled.
    """
    text = (status or "").strip()
    kind = STATUS_KINDS.get(text, "unknown")
    if kind == "unknown":
        return kind, text or "未知状态"
    label, _colour = STATUS_DISPLAY[kind]
    return kind, label


def status_colour(status: str | None) -> str:
    """Hex colour for a stored status text (for the history table)."""
    kind, _label = normalize_status(status)
    return STATUS_DISPLAY[kind][1]


def _drama_names(store: Store) -> dict[str, str]:
    """Map drama id -> display name (new name if set, else origin name)."""
    names: dict[str, str] = {}
    for row in store.list_rows("dramas"):
        names[row["id"]] = row.get("new_name") or row.get("origin_name") or row["id"]
    return names


def _present(row: dict[str, Any], drama_names: dict[str, str]) -> dict[str, Any]:
    """Normalise one raw ``tasks`` row for display (no DB access)."""
    kind, label = normalize_status(row.get("status"))
    return {
        "id": row["id"],
        "drama_id": row.get("drama_id"),
        "drama_name": drama_names.get(row.get("drama_id") or "", ""),
        "episode_name": row.get("episode_name") or "",
        "video_path": row.get("video_path") or "",
        "status": row.get("status") or "",
        "status_kind": kind,
        "status_label": label,
        "status_colour": STATUS_DISPLAY[kind][1],
        "progress": row.get("progress") or "",
        "progress_pct": row.get("progress_pct") or 0,
        "workdir": row.get("workdir") or "",
        "output_file": row.get("output_file") or "",
        "retry_count": row.get("retry_count") or 0,
        "error": row.get("error") or "",
        "created_at": row.get("created_at") or "",
        "updated_at": row.get("updated_at") or "",
        "template_pack_id": row.get("template_pack_id") or "",
        "template_version": row.get("template_version") or "",
        "entry_type": row.get("entry_type") or "",
        "source_script": row.get("source_script") or "",
        "source_dialogue": row.get("source_dialogue") or "",
        "source_storyboard": row.get("source_storyboard") or "",
        "pack_snapshot": row.get("pack_snapshot") or "",
        # N11 (P3 schema bundle): snapshot fields stored on the task row.
        # Old rows have NULL here; .get() keeps hand-built dicts working too.
        "selected_characters": row.get("selected_characters") or "",
        "stage_list": row.get("stage_list") or "",
    }


def list_history(
    store: Store,
    *,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List history tasks newest-first.

    ``status`` accepts a canonical kind (``"failed"``) or the stored text
    (``"失败"``); ``None`` means all statuses. ``limit``/``offset`` page
    through the list. Read-only: this never writes to the store.
    """
    where = ""
    args: tuple[Any, ...] = ()
    if status:
        key = (status or "").strip()
        stored = _STATUS_FILTER_ALIASES.get(key, key)
        where = "status = ?"
        args = (stored,)
    rows = store.list_rows(
        "tasks", where, args, order="updated_at DESC, created_at DESC"
    )
    names = _drama_names(store)
    presented = [_present(row, names) for row in rows]
    return presented[offset : offset + limit] if limit > 0 else presented[offset:]


def count_history(store: Store, *, status: str | None = None) -> int:
    """Number of history rows (same filter as :func:`list_history`)."""
    if not status:
        return len(store.list_rows("tasks"))
    key = (status or "").strip()
    stored = _STATUS_FILTER_ALIASES.get(key, key)
    return len(store.list_rows("tasks", "status = ?", (stored,)))


def get_history_entry(store: Store, task_id: str) -> dict[str, Any] | None:
    """One history entry (with config snapshot), or ``None`` if missing."""
    row = store.get("tasks", task_id)
    if row is None:
        return None
    names = _drama_names(store)
    entry = _present(row, names)
    entry["config_snapshot"] = build_config_snapshot(entry)
    # P4 (N18 viewer): the task's auth-gate skip records, read-only.
    entry["auth_skips"] = get_auth_skips_for_task(store, task_id)
    entry["auth_skips_text"] = format_auth_skips(entry["auth_skips"])
    return entry


# ----- auth-gate skip records (P4 viewer, N18) -----

#: Record keys rendered in the fixed columns (name / reason / time).
#: Anything else a record carries is shown generically after them (see
#: :func:`format_auth_skip_record`), so a future annotation field added
#: by another worker (e.g. an N40 scope flag) is displayed without this
#: module having to know its name in advance.
_AUTH_SKIP_CORE_KEYS: frozenset[str] = frozenset(
    {"id", "drama_id", "task_id", "character_name", "reason", "skipped_at", "created_at"}
)

#: Empty-state text for the viewer (shown instead of a blank list).
NO_AUTH_SKIPS_TEXT = "本次无跳过记录"


def format_auth_skip_record(record: dict[str, Any]) -> str:
    """Render one ``auth_skips`` row as a single display line.

    Fixed columns first — 人名、原因、时间 — then any extra field the
    record happens to carry (``scope`` / annotation / …) as ``key：value``,
    in record order. Unknown keys are never dropped and never crash the
    viewer: a missing name/reason/time renders honestly instead.
    """
    record = record or {}
    name = str(record.get("character_name") or record.get("name") or "").strip()
    reason = str(record.get("reason") or "").strip()
    when = str(record.get("skipped_at") or record.get("created_at") or "").strip()
    line = f"• {name or '（未记名）'} — {reason or '原因未记录'}（{when or '时间未记录'}）"
    extras = [
        f"{key}：{value}"
        for key, value in record.items()
        if key not in _AUTH_SKIP_CORE_KEYS and value not in (None, "")
    ]
    if extras:
        line += "；" + "；".join(extras)
    return line


def format_auth_skips(records: list[dict[str, Any]] | None) -> str:
    """Render skip records for the viewer; empty reads as no-skips text."""
    records = records or []
    if not records:
        return NO_AUTH_SKIPS_TEXT
    return "\n".join(format_auth_skip_record(r) for r in records)


def get_auth_skips_for_task(store: Store, task_id: str) -> list[dict[str, Any]]:
    """Skip records for one task, newest first. Read-only.

    Task-scoped rows (``task_id`` = this task) first; drama-level rows
    (same drama, ``task_id`` empty — written when the drama had no
    episode tasks at gate time) are included too, so a drama-level skip
    is still visible from any of its tasks. A missing task, a missing
    table, or any read error reads as “none skipped”, never a crash.
    """
    try:
        records = list(store.list_auth_skips(task_id=task_id))
        row = store.get("tasks", task_id)
        drama_id = (row or {}).get("drama_id")
        if drama_id:
            seen = {str(r.get("id")) for r in records if r.get("id")}
            for extra in store.list_auth_skips(drama_id=drama_id):
                if extra.get("task_id"):
                    continue
                if extra.get("id") and str(extra.get("id")) in seen:
                    continue
                records.append(extra)
                if extra.get("id"):
                    seen.add(str(extra.get("id")))
        records.sort(key=lambda r: str(r.get("skipped_at") or r.get("created_at") or ""), reverse=True)
        return records
    except Exception:
        return []


def _display_list(value: Any) -> str:
    """Render a stored character/stage list for the snapshot.

    Stored as JSON (``Store`` dumps lists) or plain text; old rows store
    NULL/"" and hand-built entries may lack the key entirely — all of
    those honestly render as "未记录", never a crash or an invented value.
    """
    if value is None:
        return NOT_RECORDED
    if isinstance(value, (list, tuple)):
        names = [str(v).strip() for v in value if str(v).strip()]
        return "、".join(names) if names else NOT_RECORDED
    text = str(value).strip()
    if not text:
        return NOT_RECORDED
    if text.startswith("["):
        try:
            import json

            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            names = [str(v).strip() for v in parsed if str(v).strip()]
            return "、".join(names) if names else NOT_RECORDED
    return text


def build_config_snapshot(entry: dict[str, Any]) -> dict[str, str]:
    """Config snapshot for a history entry.

    Only fields the ``tasks`` row actually carries are shown with values.
    Before the P3 schema bundle the table did not store the chosen
    characters or the stage list, so old rows still report those two as
    "未记录" instead of guessed.
    """
    template = entry.get("template_version") or ""
    pack = entry.get("template_pack_id") or ""
    if template and pack:
        template_text = f"{pack}（版本 {template}）"
    elif template:
        template_text = f"版本 {template}"
    elif pack:
        template_text = pack
    else:
        template_text = NOT_RECORDED
    return {
        "模板": template_text,
        "入口类型": entry.get("entry_type") or NOT_RECORDED,
        "剧本": entry.get("source_script") or NOT_RECORDED,
        "台词表": entry.get("source_dialogue") or NOT_RECORDED,
        "分镜表": entry.get("source_storyboard") or NOT_RECORDED,
        "模板快照": entry.get("pack_snapshot") or NOT_RECORDED,
        "选的人物": _display_list(entry.get("selected_characters")),
        "阶段清单": _display_list(entry.get("stage_list")),
        "视频": entry.get("video_path") or NOT_RECORDED,
        "工作目录": entry.get("workdir") or NOT_RECORDED,
        "成片": entry.get("output_file") or NOT_RECORDED,
    }


def build_rerun_task(store: Store, task_id: str) -> dict[str, Any]:
    """Build the re-run task dict for a history entry.

    Same config as the original (drama, episode, video, workdir, output,
    template, entry type, sources); fresh id, status reset to 待执行,
    progress cleared, retry count reset, error cleared, timestamps renewed.

    The dict is returned, NOT inserted and NOT executed. The caller inserts
    it (``store.insert("tasks", **rerun)``) and starts it through the
    scheduler/runner. Raises ``ValueError`` when the task does not exist.
    """
    row = store.get("tasks", task_id)
    if row is None:
        raise ValueError(f"找不到任务：{task_id}")
    now = _now()
    return {
        "id": new_id(),
        "drama_id": row.get("drama_id"),
        "episode_name": row.get("episode_name"),
        "video_path": row.get("video_path"),
        "status": ST_PENDING,
        "progress": "等待执行",
        "progress_pct": 0,
        "workdir": row.get("workdir"),
        "output_file": row.get("output_file"),
        "retry_count": 0,
        "selected": row.get("selected", 1),
        "error": None,
        "created_at": now,
        "updated_at": now,
        "template_pack_id": row.get("template_pack_id"),
        "template_version": row.get("template_version"),
        "entry_type": row.get("entry_type"),
        "source_script": row.get("source_script"),
        "source_dialogue": row.get("source_dialogue"),
        "source_storyboard": row.get("source_storyboard"),
        "pack_snapshot": row.get("pack_snapshot"),
        # N11: a re-run keeps the recorded characters / stages, so its
        # own history snapshot still shows them (NULL on old rows is fine).
        "selected_characters": row.get("selected_characters"),
        "stage_list": row.get("stage_list"),
    }


def delete_history_entry(store: Store, task_id: str) -> bool:
    """Delete one history row.

    Only the ``tasks`` row is deleted (dependent shot / subtitle /
    checkpoint rows go away with the foreign-key cascade). The output file
    on disk is never touched, and the drama row is kept. Returns ``True``
    when a row was deleted, ``False`` when the task did not exist.
    """
    if store.get("tasks", task_id) is None:
        return False
    store.delete("tasks", task_id)
    return True
