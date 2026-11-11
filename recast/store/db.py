"""SQLite store for dramas, tasks, shots, subtitles, characters, runs, checkpoints."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from recast.config import PROJECT_ROOT
from recast.core.constants import ST_PENDING

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dramas (
    id TEXT PRIMARY KEY,
    origin_name TEXT NOT NULL,
    new_name TEXT,
    root_path TEXT NOT NULL UNIQUE,
    episode_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT '待执行',
    progress TEXT,
    progress_pct REAL DEFAULT 0,
    output_dir TEXT,
    selected INTEGER DEFAULT 1,
    missing_files TEXT,
    error TEXT,
    created_at TEXT,
    updated_at TEXT,
    template_pack_id TEXT,
    template_version TEXT,
    entry_type TEXT DEFAULT 'import',
    upload_status TEXT,
    pack_root TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    drama_id TEXT NOT NULL,
    episode_name TEXT,
    video_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT '待执行',
    progress TEXT,
    progress_pct REAL DEFAULT 0,
    workdir TEXT,
    output_file TEXT,
    retry_count INTEGER DEFAULT 0,
    selected INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT,
    updated_at TEXT,
    template_pack_id TEXT,
    template_version TEXT,
    entry_type TEXT,
    source_script TEXT,
    source_dialogue TEXT,
    source_storyboard TEXT,
    pack_snapshot TEXT,
    FOREIGN KEY (drama_id) REFERENCES dramas(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS shots (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    index_no INTEGER,
    start_time REAL,
    end_time REAL,
    start_frame INTEGER,
    end_frame INTEGER,
    status TEXT,
    output_path TEXT,
    shot_code TEXT,
    prompt TEXT,
    ref_assets TEXT,
    dialogue TEXT,
    speaker TEXT,
    duration_sec REAL,
    primary_action TEXT,
    people TEXT,
    is_fight INTEGER DEFAULT 0,
    quality_flags TEXT,
    review_reason TEXT,
    scene_id TEXT,
    kind TEXT,
    filled_prompt TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS subtitles (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    source TEXT,
    start_time REAL,
    end_time REAL,
    text TEXT,
    conf REAL,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    name TEXT,
    display_name TEXT,
    confirmed INTEGER DEFAULT 0,
    authorized INTEGER DEFAULT 0,
    auth_type TEXT,
    cluster_key TEXT,
    notes TEXT,
    source_task_id TEXT,
    created_at TEXT,
    updated_at TEXT,
    aliases TEXT,
    face_embedding TEXT,
    voice_id TEXT,
    source_note TEXT
);

CREATE TABLE IF NOT EXISTS character_assets (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT,
    complete INTEGER DEFAULT 0,
    notes TEXT,
    quality_score REAL,
    source_episodes TEXT,
    source_note TEXT,
    FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    name TEXT,
    level INTEGER,
    payload TEXT,
    created_at TEXT,
    version TEXT,
    json_path TEXT,
    pack_id TEXT,
    updated_at TEXT,
    level_name TEXT
);

CREATE TABLE IF NOT EXISTS template_packs (
    id TEXT PRIMARY KEY,
    name TEXT,
    version TEXT,
    root_path TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    drama_id TEXT,
    step TEXT,
    status TEXT,
    started_at TEXT,
    finished_at TEXT,
    log_path TEXT,
    error TEXT,
    model TEXT,
    duration_sec REAL,
    cost REAL DEFAULT 0,
    message TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    step TEXT NOT NULL,
    input_hashes TEXT,
    output_paths TEXT,
    created_at TEXT,
    UNIQUE(task_id, step)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    level TEXT,
    task_id TEXT,
    message TEXT
);

CREATE TABLE IF NOT EXISTS quality_marks (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    shot_id TEXT,
    shot_code TEXT,
    reason TEXT,
    detail TEXT,
    status TEXT,
    created_at TEXT,
    score REAL,
    thumbnail TEXT
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_drama ON tasks(drama_id);
CREATE INDEX IF NOT EXISTS idx_shots_task ON shots(task_id);
CREATE INDEX IF NOT EXISTS idx_subs_task ON subtitles(task_id);
CREATE INDEX IF NOT EXISTS idx_assets_char ON character_assets(character_id);
CREATE INDEX IF NOT EXISTS idx_ckpt_task ON checkpoints(task_id, step);
CREATE INDEX IF NOT EXISTS idx_quality_task ON quality_marks(task_id);
"""



def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id(n: int = 12) -> str:
    return uuid.uuid4().hex[:n]


def default_db_path() -> Path:
    dest = PROJECT_ROOT / "outputs" / "store"
    dest.mkdir(parents=True, exist_ok=True)
    return dest / "recast.db"


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


class Store:
    """Thread-safe SQLite helper. One connection per thread."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        self._ensure_schema()

    def connect(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(str(self.path), check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA journal_mode = WAL")
            self._local.con = con
        return con

    def _ensure_schema(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            con = self.connect()
            con.executescript(SCHEMA)
            con.commit()
            from recast.store.migrate import migrate

            migrate(con)
            self._initialized = True

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        con = self.connect()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise

    # ----- generic helpers -----

    def get(self, table: str, row_id: str) -> dict[str, Any] | None:
        cur = self.connect().execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        return _row_to_dict(cur.fetchone())

    def list_rows(self, table: str, where: str = "", args: Iterable[Any] = (), order: str = "") -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        if order:
            sql += f" ORDER BY {order}"
        cur = self.connect().execute(sql, tuple(args))
        return [dict(r) for r in cur.fetchall()]

    _HAS_UPDATED = {"dramas", "tasks", "characters"}

    def update(self, table: str, row_id: str, **fields: Any) -> dict[str, Any] | None:
        fields = {k: v for k, v in fields.items() if k != "id"}
        if table in self._HAS_UPDATED and "updated_at" not in fields:
            fields["updated_at"] = _now()
        keys = list(fields)
        if not keys:
            return self.get(table, row_id)
        assignments = ", ".join(f"{k} = ?" for k in keys)
        values = [_dump(fields[k]) for k in keys]
        values.append(row_id)
        with self.tx() as con:
            con.execute(f"UPDATE {table} SET {assignments} WHERE id = ?", values)
        return self.get(table, row_id)

    def insert(self, table: str, **fields: Any) -> dict[str, Any]:
        if "id" not in fields or not fields["id"]:
            fields["id"] = new_id()
        now = _now()
        if table in {"dramas", "tasks", "characters", "templates", "checkpoints", "runs"}:
            # runs uses started_at; still allow created_at only when the table has it
            pass
        if table in {"dramas", "tasks", "characters", "templates", "checkpoints", "template_packs", "quality_marks", "runs"}:
            fields.setdefault("created_at", now)
        if table in {"dramas", "tasks", "characters"}:
            fields.setdefault("updated_at", now)
        keys = list(fields)
        placeholders = ", ".join("?" for _ in keys)
        cols = ", ".join(keys)
        values = [_dump(fields[k]) for k in keys]
        with self.tx() as con:
            con.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", values)
        return self.get(table, fields["id"])  # type: ignore[return-value]

    def delete(self, table: str, row_id: str) -> None:
        with self.tx() as con:
            con.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))

    # ----- dramas / tasks -----

    def upsert_drama_by_path(self, root_path: str, *, origin_name: str, episode_count: int) -> dict[str, Any]:
        cur = self.connect().execute("SELECT * FROM dramas WHERE root_path = ?", (root_path,))
        row = _row_to_dict(cur.fetchone())
        if row:
            return self.update(
                "dramas",
                row["id"],
                origin_name=origin_name,
                episode_count=episode_count,
            ) or row
        return self.insert(
            "dramas",
            origin_name=origin_name,
            new_name=origin_name,
            root_path=root_path,
            episode_count=episode_count,
            status=ST_PENDING,
            progress="等待执行",
            selected=1,
        )

    def replace_tasks_for_drama(self, drama_id: str, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep existing task rows when the video path is unchanged (preserves workdirs)."""
        existing = {t["video_path"]: t for t in self.list_rows("tasks", "drama_id = ?", (drama_id,))}
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ep in episodes:
            path = ep["video_path"]
            seen.add(path)
            old = existing.get(path)
            if old:
                kept.append(
                    self.update(
                        "tasks",
                        old["id"],
                        episode_name=ep.get("episode_name") or old.get("episode_name"),
                    ) or old
                )
            else:
                kept.append(
                    self.insert(
                        "tasks",
                        drama_id=drama_id,
                        episode_name=ep.get("episode_name") or Path(path).stem,
                        video_path=path,
                        status=ST_PENDING,
                        progress="等待执行",
                        selected=1,
                    )
                )
        for path, old in existing.items():
            if path not in seen and old.get("status") != "处理中":
                self.delete("tasks", old["id"])
        self.update("dramas", drama_id, episode_count=len(kept))
        return kept

    def tasks_for_drama(self, drama_id: str) -> list[dict[str, Any]]:
        return self.list_rows("tasks", "drama_id = ?", (drama_id,), order="episode_name COLLATE NOCASE")

    def list_dramas(self) -> list[dict[str, Any]]:
        return self.list_rows("dramas", order="created_at")

    def delete_drama(self, drama_id: str) -> None:
        """Remove the drama and its tasks from the queue. Does not delete output files."""
        with self.tx() as con:
            con.execute("DELETE FROM shots WHERE task_id IN (SELECT id FROM tasks WHERE drama_id = ?)", (drama_id,))
            con.execute("DELETE FROM subtitles WHERE task_id IN (SELECT id FROM tasks WHERE drama_id = ?)", (drama_id,))
            con.execute("DELETE FROM checkpoints WHERE task_id IN (SELECT id FROM tasks WHERE drama_id = ?)", (drama_id,))
            con.execute("DELETE FROM runs WHERE drama_id = ?", (drama_id,))
            con.execute("DELETE FROM tasks WHERE drama_id = ?", (drama_id,))
            con.execute("DELETE FROM dramas WHERE id = ?", (drama_id,))

    def clear_completed_dramas(self) -> int:
        rows = [d for d in self.list_dramas() if d.get("status") == "已完成"]
        for d in rows:
            self.delete_drama(d["id"])
        return len(rows)

    # ----- characters -----

    def list_characters(self) -> list[dict[str, Any]]:
        chars = self.list_rows("characters", order="created_at")
        for ch in chars:
            ch["assets"] = self.list_rows("character_assets", "character_id = ?", (ch["id"],))
        return chars

    def get_character(self, character_id: str) -> dict[str, Any] | None:
        row = self.get("characters", character_id)
        if not row:
            return None
        row["assets"] = self.list_rows("character_assets", "character_id = ?", (character_id,))
        return row

    def find_character_by_cluster(self, cluster_key: str) -> dict[str, Any] | None:
        cur = self.connect().execute("SELECT * FROM characters WHERE cluster_key = ?", (cluster_key,))
        row = _row_to_dict(cur.fetchone())
        if not row:
            return None
        row["assets"] = self.list_rows("character_assets", "character_id = ?", (row["id"],))
        return row

    def upsert_character_candidate(
        self,
        *,
        cluster_key: str,
        auto_name: str,
        sample_paths: list[str],
        source_task_id: str | None = None,
    ) -> dict[str, Any]:
        existing = self.find_character_by_cluster(cluster_key)
        if existing is None:
            # also match by current display name if the user already confirmed a same-named character
            for ch in self.list_characters():
                if (ch.get("display_name") or ch.get("name")) == auto_name and ch.get("confirmed"):
                    existing = ch
                    break
        if existing is None:
            existing = self.insert(
                "characters",
                name=auto_name,
                display_name=auto_name,
                confirmed=0,
                authorized=0,
                cluster_key=cluster_key,
                source_task_id=source_task_id,
            )
        for path in sample_paths:
            if not path:
                continue
            already = any(a.get("path") == path and a.get("kind") == "face" for a in existing.get("assets") or [])
            if already:
                continue
            self.insert(
                "character_assets",
                character_id=existing["id"],
                kind="face",
                path=path,
                complete=1,
                notes="聚类样本",
            )
        return self.get_character(existing["id"])  # type: ignore[return-value]

    def set_character_asset(self, character_id: str, kind: str, path: str | None, *, complete: bool | None = None) -> None:
        rows = self.list_rows("character_assets", "character_id = ? AND kind = ?", (character_id, kind))
        flag = 1 if (complete if complete is not None else bool(path)) else 0
        if rows:
            self.update("character_assets", rows[0]["id"], path=path, complete=flag)
        else:
            self.insert("character_assets", character_id=character_id, kind=kind, path=path, complete=flag)

    # ----- settings / logs -----

    def get_setting(self, key: str, default: Any = None) -> Any:
        cur = self.connect().execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        with self.tx() as con:
            con.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, payload),
            )

    def add_log(self, level: str, message: str, task_id: str | None = None) -> None:
        with self.tx() as con:
            con.execute(
                "INSERT INTO logs(ts, level, task_id, message) VALUES (?, ?, ?, ?)",
                (_now(), level, task_id, message),
            )

    def recent_logs(self, limit: int = 500, task_id: str | None = None, level: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM logs"
        clauses: list[str] = []
        args: list[Any] = []
        if task_id:
            clauses.append("task_id = ?")
            args.append(task_id)
        if level:
            clauses.append("level = ?")
            args.append(level)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        cur = self.connect().execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        rows.reverse()
        return rows

    def replace_shots(self, task_id: str, shots: list[dict[str, Any]]) -> None:
        with self.tx() as con:
            con.execute("DELETE FROM shots WHERE task_id = ?", (task_id,))
        extra_keys = (
            "shot_code", "prompt", "ref_assets", "dialogue", "speaker",
            "duration_sec", "primary_action", "people", "is_fight",
            "quality_flags", "review_reason", "scene_id", "kind", "filled_prompt",
        )
        for i, sh in enumerate(shots):
            fields = {
                "task_id": task_id,
                "index_no": sh.get("index_no", i),
                "start_time": sh.get("start_time"),
                "end_time": sh.get("end_time"),
                "start_frame": sh.get("start_frame"),
                "end_frame": sh.get("end_frame"),
                "status": sh.get("status") or ST_PENDING,
                "output_path": sh.get("output_path"),
            }
            for k in extra_keys:
                if k in sh:
                    fields[k] = sh[k]
            self.insert("shots", **fields)

    def replace_subtitles(self, task_id: str, source: str, segments: list[dict[str, Any]]) -> None:
        with self.tx() as con:
            con.execute("DELETE FROM subtitles WHERE task_id = ? AND source = ?", (task_id, source))
        for seg in segments:
            self.insert(
                "subtitles",
                task_id=task_id,
                source=source,
                start_time=seg.get("start") if "start" in seg else seg.get("start_time"),
                end_time=seg.get("end") if "end" in seg else seg.get("end_time"),
                text=seg.get("text") or "",
                conf=seg.get("conf") or seg.get("avg_logprob"),
            )

    def reset_running_on_boot(self) -> int:
        """Turn leftover 处理中 rows back into 待执行 after a mid-run shutdown."""
        n = 0
        for table in ("tasks", "dramas"):
            rows = self.list_rows(table, "status = ?", ("处理中",))
            for row in rows:
                self.update(table, row["id"], status=ST_PENDING, progress="上次未完成，已恢复为待执行", error=None)
                n += 1
        return n

    def list_runs(self, task_id: str | None = None, drama_id: str | None = None) -> list[dict[str, Any]]:
        if task_id:
            return self.list_rows("runs", "task_id = ?", (task_id,), order="started_at, created_at, id")
        if drama_id:
            return self.list_rows("runs", "drama_id = ?", (drama_id,), order="started_at, created_at, id")
        return self.list_rows("runs", order="started_at, created_at, id")

    def record_run(
        self,
        *,
        task_id: str | None,
        drama_id: str | None,
        step: str,
        status: str,
        duration_sec: float | None = None,
        cost: float | None = 0,
        model: str | None = None,
        message: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        log_path: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        return self.insert(
            "runs",
            task_id=task_id,
            drama_id=drama_id,
            step=step,
            status=status,
            duration_sec=duration_sec,
            cost=0 if cost is None else cost,
            model=model,
            message=message,
            error=error,
            started_at=started_at or now,
            finished_at=finished_at or now,
            log_path=log_path,
            created_at=now,
        )

    def shots_for_task(self, task_id: str) -> list[dict[str, Any]]:
        return self.list_rows("shots", "task_id = ?", (task_id,), order="index_no, shot_code")

    def quality_for_task(self, task_id: str) -> list[dict[str, Any]]:
        return self.list_rows("quality_marks", "task_id = ?", (task_id,), order="created_at")




def _dump(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, bool):
        return int(value)
    return value
