"""SQLite store for dramas, tasks, shots, subtitles, characters, runs, checkpoints."""

from __future__ import annotations

import json
import re
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
    root_path TEXT NOT NULL,
    source_path TEXT UNIQUE,
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
    selected_characters TEXT,
    stage_list TEXT,
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

CREATE TABLE IF NOT EXISTS auth_skips (
    id TEXT PRIMARY KEY,
    drama_id TEXT,
    task_id TEXT,
    character_name TEXT NOT NULL,
    reason TEXT,
    skipped_at TEXT,
    created_at TEXT,
    FOREIGN KEY (drama_id) REFERENCES dramas(id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_auth_skips_drama ON auth_skips(drama_id);
CREATE INDEX IF NOT EXISTS idx_auth_skips_task ON auth_skips(task_id);

CREATE INDEX IF NOT EXISTS idx_tasks_drama ON tasks(drama_id);
CREATE INDEX IF NOT EXISTS idx_shots_task ON shots(task_id);
CREATE INDEX IF NOT EXISTS idx_subs_task ON subtitles(task_id);
CREATE INDEX IF NOT EXISTS idx_assets_char ON character_assets(character_id);
CREATE INDEX IF NOT EXISTS idx_ckpt_task ON checkpoints(task_id, step);
CREATE INDEX IF NOT EXISTS idx_quality_task ON quality_marks(task_id);
"""

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_COLUMNS: dict[str, set[str]] = {}
for _match in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\);", SCHEMA, re.S):
    _cols: set[str] = set()
    for _line in _match.group(2).splitlines():
        _line = _line.strip().rstrip(",")
        if not _line or _line.upper().startswith(("FOREIGN", "UNIQUE", "PRIMARY", "CHECK")):
            continue
        _cols.add(_line.split()[0])
    _TABLE_COLUMNS[_match.group(1)] = _cols


def _require_table(name: str) -> str:
    if name not in _TABLE_COLUMNS:
        raise ValueError(f"不允许的表名: {name}")
    return name


def _require_column(table: str, column: str) -> str:
    if not _IDENT.fullmatch(column) or column not in _TABLE_COLUMNS[table]:
        raise ValueError(f"不允许的列名: {table}.{column}")
    return column


def _require_where(table: str, where: str) -> None:
    if not where.strip():
        return
    parts = re.split(r"\s+AND\s+", where.strip(), flags=re.IGNORECASE)
    for part in parts:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\?", part.strip(), flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"不允许的查询条件: {where}")
        _require_column(table, match.group(1))


def _require_order(table: str, order: str) -> None:
    if not order.strip():
        return
    for piece in order.split(","):
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:ASC|DESC))?(?:\s+COLLATE\s+NOCASE)?",
            piece.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"不允许的排序: {order}")
        _require_column(table, match.group(1))



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
        table = _require_table(table)
        cur = self.connect().execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        return _row_to_dict(cur.fetchone())

    def list_rows(self, table: str, where: str = "", args: Iterable[Any] = (), order: str = "") -> list[dict[str, Any]]:
        table = _require_table(table)
        _require_where(table, where)
        _require_order(table, order)
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
        table = _require_table(table)
        keys = [_require_column(table, key) for key in fields]
        if not keys:
            return self.get(table, row_id)
        assignments = ", ".join(f"{k} = ?" for k in keys)
        values = [_dump(fields[k]) for k in keys]
        values.append(row_id)
        with self.tx() as con:
            con.execute(f"UPDATE {table} SET {assignments} WHERE id = ?", values)
        return self.get(table, row_id)

    def insert(self, table: str, **fields: Any) -> dict[str, Any]:
        table = _require_table(table)
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
        keys = [_require_column(table, key) for key in fields]
        placeholders = ", ".join("?" for _ in keys)
        cols = ", ".join(keys)
        values = [_dump(fields[k]) for k in keys]
        with self.tx() as con:
            con.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", values)
        return self.get(table, fields["id"])  # type: ignore[return-value]

    def delete(self, table: str, row_id: str) -> None:
        table = _require_table(table)
        with self.tx() as con:
            con.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))

    # ----- dramas / tasks -----

    def find_drama_by_source(self, source_path: str) -> dict[str, Any] | None:
        """Drama whose file identity is ``source_path`` (see N30)."""
        cur = self.connect().execute("SELECT * FROM dramas WHERE source_path = ?", (source_path,))
        return _row_to_dict(cur.fetchone())

    def find_task_by_video(self, video_path: str) -> dict[str, Any] | None:
        """Task row for one episode video (N39): any episode, not just the first.

        ``video_path`` is stored as the resolved absolute path the scanner
        produced, so callers must pass a resolved path too.
        """
        cur = self.connect().execute("SELECT * FROM tasks WHERE video_path = ?", (video_path,))
        return _row_to_dict(cur.fetchone())

    def find_drama_by_task_video(self, video_path: str) -> dict[str, Any] | None:
        """Drama that already owns a task for ``video_path`` (N39).

        This is the cross-drama lookup ``find_drama_by_source`` cannot do:
        a drama's identity is its FIRST episode, so a non-first episode
        only shows up here, via its task row. Returns None when no task
        references the video.
        """
        cur = self.connect().execute(
            "SELECT d.* FROM dramas d"
            " JOIN tasks t ON t.drama_id = d.id"
            " WHERE t.video_path = ? LIMIT 1",
            (video_path,),
        )
        return _row_to_dict(cur.fetchone())

    def find_drama_by_root(self, root_path: str) -> dict[str, Any] | None:
        """Drama filed under ``root_path`` (legacy lookup, no uniqueness)."""
        cur = self.connect().execute("SELECT * FROM dramas WHERE root_path = ?", (root_path,))
        return _row_to_dict(cur.fetchone())

    def upsert_drama_by_path(
        self,
        root_path: str,
        *,
        origin_name: str,
        episode_count: int,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        """Insert or refresh a drama row, deduped by FILE identity (N30).

        ``source_path`` is the identity: for an imported drama the first
        episode's video file path, for a script drama its virtual folder.
        Two videos in the same folder therefore get two drama rows, where
        the old ``root_path UNIQUE`` merged them and the second import
        silently replaced the first one's episode tasks. When
        ``source_path`` is given, lookup is by that value ONLY — falling
        back to ``root_path`` would reintroduce the merge. When it is
        omitted (legacy callers), lookup falls back to ``root_path`` and
        the identity is stored as ``root_path`` itself.
        """
        identity = source_path or root_path
        row = self.find_drama_by_source(identity)
        if row is None and source_path is None:
            row = self.find_drama_by_root(root_path)
        if row:
            return self.update(
                "dramas",
                row["id"],
                origin_name=origin_name,
                episode_count=episode_count,
                root_path=root_path,
                source_path=identity,
            ) or row
        return self.insert(
            "dramas",
            origin_name=origin_name,
            new_name=origin_name,
            root_path=root_path,
            source_path=identity,
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
            con.execute("DELETE FROM auth_skips WHERE drama_id = ?", (drama_id,))
            con.execute("DELETE FROM tasks WHERE drama_id = ?", (drama_id,))
            con.execute("DELETE FROM dramas WHERE id = ?", (drama_id,))

    def clear_completed_dramas(self) -> int:
        rows = [d for d in self.list_dramas() if d.get("status") == "已完成"]
        for d in rows:
            self.delete_drama(d["id"])
        return len(rows)

    # ----- auth skips (N18, P3 schema bundle) -----

    # ----- drama <-> character scope (N40, P4) -----
    #
    # 剧目→人物的映射只用库里现成的两处来源，不新增表、不改写旧行：
    # 1. ``characters.source_task_id``——人物首次从哪个分集任务里提取
    #    出来（extract/faces.py 写），经 tasks 表找到所属剧目；
    # 2. ``tasks.selected_characters``——某次启动时该分集任务选中的
    #    人物名快照（N11），名字落在哪个剧目的任务里就属哪个剧目。
    # 两处都对不上的人物（手动添加、旧库行、改名后快照还是旧名）算
    # “归属未知”，关口与读口都按“属于本剧”放行（fail-open）：宁可
    # 多弹一次关口，也不让一个没链路的人物被静默跳过。反过来，只
    # 有明确链到别的剧目的人物才会被收窄出去。
    #
    # 已知缺口：同一人物先在 B 剧出现、后又出现在 A 剧时，
    # ``source_task_id`` 只保留首次提取的任务（upsert 不更新它），
    # 且跨剧合并只认首次聚类——此时该人物仍只链 B，跑 A 时关口
    # 不会弹他；要根治需在提取/合并时补记“出现过的全部任务”。

    def _drama_ids_for_names(self, names: set[str]) -> dict[str, set[str]]:
        """每个名字链到的剧目 id 集合（N40 映射，内部用）。

        空集合 = 归属未知（两处来源都没有这名字），调用方按属于本剧
        处理，绝不据此静默排除。读不到表时同样返回全空，不抛错。
        """
        found: dict[str, set[str]] = {n: set() for n in names}
        if not names:
            return found
        try:
            task_rows = self.list_rows("tasks")
        except Exception:
            return found
        task_to_drama = {
            t.get("id"): t.get("drama_id") for t in task_rows if t.get("id")
        }
        try:
            char_rows = self.list_rows("characters")
        except Exception:
            char_rows = []
        for ch in char_rows:
            for key in (
                (ch.get("display_name") or "").strip(),
                (ch.get("name") or "").strip(),
            ):
                if key in found:
                    did = task_to_drama.get((ch.get("source_task_id") or "").strip())
                    if did:
                        found[key].add(did)
        for t in task_rows:
            did = t.get("drama_id")
            if not did:
                continue
            for n in _decode_name_list(t.get("selected_characters")):
                if n in found:
                    found[n].add(did)
        return found

    def _foreign_names(self, names: set[str], drama_id: str) -> set[str]:
        """名字里明确链到别的剧目、且不链本剧的那些（N40 内部用）。"""
        linked = self._drama_ids_for_names(set(names))
        return {n for n in names if linked.get(n) and drama_id not in linked[n]}

    def linked_drama_ids_for_character_name(self, name: str) -> set[str]:
        """一个人名链到的剧目 id（N40）。

        来源：同名人物行的 ``source_task_id`` 与各任务的
        ``selected_characters`` 快照。空集合表示归属未知，不是“不
        属于任何剧目”——调用方须按属于本剧放行。
        """
        name = (name or "").strip()
        if not name:
            return set()
        return set(self._drama_ids_for_names({name}).get(name) or set())

    def linked_drama_ids_for_character(self, character: dict[str, Any]) -> set[str]:
        """一人物行链到的剧目 id（N40）。

        除名字映射外，行自身的 ``source_task_id`` 是直接证据：改名
        后快照里的旧名对不上新名时，仍靠它链到原剧目。
        """
        character = character or {}
        names = {
            (character.get("display_name") or "").strip(),
            (character.get("name") or "").strip(),
        } - {""}
        out: set[str] = set()
        for linked in self._drama_ids_for_names(names).values():
            out |= linked
        sid = (character.get("source_task_id") or "").strip()
        if sid:
            try:
                row = self.get("tasks", sid)
            except Exception:
                row = None
            if row and row.get("drama_id"):
                out.add(row["drama_id"])
        return out

    def character_in_scope(
        self, character_or_name: Any, drama_ids: list[str] | None = None
    ) -> bool:
        """人物（行或人名）是否属于所选剧目之一（N40）。

        ``drama_ids`` 为空（None/[]）= 不收窄，全算在内；链不到任
        何剧目（归属未知）= 算在内；只链到别的剧目 = 不在范围内。
        读不到映射时同样放行，绝不因一次读失败拦住关口。
        """
        if not drama_ids:
            return True
        wanted = {str(d) for d in drama_ids if str(d).strip()}
        if not wanted:
            return True
        try:
            if isinstance(character_or_name, str):
                linked = self.linked_drama_ids_for_character_name(character_or_name)
            elif isinstance(character_or_name, dict):
                linked = self.linked_drama_ids_for_character(character_or_name)
            else:
                return True
        except Exception:
            return True
        return not linked or bool(linked & wanted)

    def _auth_skip_scope_drama(
        self, drama_id: str | None, task_id: str | None
    ) -> str | None:
        """读口的生效剧目：直接给的 drama_id 优先，否则由 task 反查。"""
        if drama_id:
            return drama_id
        if task_id:
            try:
                row = self.get("tasks", task_id)
            except Exception:
                row = None
            if row and row.get("drama_id"):
                return row["drama_id"]
        return None

    def record_auth_skip(
        self,
        drama_id: str | None,
        character_name: str,
        *,
        task_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Persist one auth-gate skip (N18).

        This is the durable record the gate's ``progress``-text note is
        not: the runner rewrites ``tasks.progress`` / ``dramas.progress``
        as it runs, but it never touches ``auth_skips``, so a skipped
        character stays findable after the run. Same
        (drama, task, character) twice records once (the gate can be
        re-confirmed for the same drama).
        """
        name = (character_name or "").strip()
        if not name:
            raise ValueError("跳过的人物名不能为空。")
        existing = self.list_auth_skips(drama_id=drama_id, task_id=task_id)
        for row in existing:
            if (row.get("character_name") or "") == name:
                return row
        now = _now()
        return self.insert(
            "auth_skips",
            drama_id=drama_id,
            task_id=task_id,
            character_name=name,
            reason=reason,
            skipped_at=now,
            created_at=now,
        )

    def list_auth_skips(
        self,
        *,
        drama_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Skipped characters, newest first. Tolerant: a database that
        somehow has no ``auth_skips`` table yet reads as “none skipped”.

        N40：查某个剧目时，已产生的假记录（人名明确链到别的剧目、
        却记在本剧名下，P3 关口全局算的遗留）不删除——库里原行原
        样返回，只在行上多挂一个 ``scope_note`` 字段：
        ``"全局未授权，非本剧人物"``。任务历史查看器会把这句直接
        显示出来。归属未知的人名（库里没链路）不挂标注。没给剧目
        （也没给能反查剧目的 task）时无法判定，不标注。
        """
        try:
            where, args = "", []
            if drama_id is not None:
                where, args = "drama_id = ?", [drama_id]
            if task_id is not None:
                where = f"{where} AND task_id = ?" if where else "task_id = ?"
                args.append(task_id)
            rows = self.list_rows("auth_skips", where, args, order="skipped_at DESC, character_name")
        except sqlite3.OperationalError:
            return []
        eff = self._auth_skip_scope_drama(drama_id, task_id)
        if eff:
            try:
                names = {
                    (r.get("character_name") or "").strip()
                    for r in rows
                    if (r.get("character_name") or "").strip()
                }
                foreign = self._foreign_names(names, eff)
                for r in rows:
                    if (r.get("character_name") or "").strip() in foreign:
                        r["scope_note"] = "全局未授权，非本剧人物"
            except Exception:
                pass
        return rows

    def skipped_character_names(
        self,
        *,
        drama_id: str | None = None,
        task_id: str | None = None,
    ) -> list[str]:
        """Distinct skipped character names, oldest first, in record order.

        N40：查某个剧目时，只返回属于本剧（或归属未知）的人名；
        明确链到别的剧目的假记录人名会被滤掉——但这只是读口的
        过滤，库里原行还在，用 :meth:`list_auth_skips` 查同一剧
        目仍能看到它（带 ``scope_note`` 标注）。没给剧目（也没给
        能反查剧目的 task）时无法判定，不过滤。
        """
        try:
            where, args = "", []
            if drama_id is not None:
                where, args = "drama_id = ?", [drama_id]
            if task_id is not None:
                where = f"{where} AND task_id = ?" if where else "task_id = ?"
                args.append(task_id)
            rows = self.list_rows("auth_skips", where, args, order="skipped_at ASC")
        except sqlite3.OperationalError:
            return []
        eff = self._auth_skip_scope_drama(drama_id, task_id)
        foreign: set[str] = set()
        if eff:
            try:
                names = {
                    (row.get("character_name") or "").strip()
                    for row in rows
                    if (row.get("character_name") or "").strip()
                }
                foreign = self._foreign_names(names, eff)
            except Exception:
                foreign = set()
        names_out: list[str] = []
        for row in rows:
            name = (row.get("character_name") or "").strip()
            if name and name not in names_out and name not in foreign:
                names_out.append(name)
        return names_out

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

    def delete_character(self, character_id: str) -> None:
        """Remove one character and their asset rows. Does not touch files."""
        with self.tx() as con:
            con.execute("DELETE FROM character_assets WHERE character_id = ?", (character_id,))
            con.execute("DELETE FROM characters WHERE id = ?", (character_id,))

    def delete_asset_rows(self, asset_ids: Iterable[str]) -> int:
        ids = [asset_id for asset_id in asset_ids if asset_id]
        if not ids:
            return 0
        with self.tx() as con:
            con.executemany("DELETE FROM character_assets WHERE id = ?", [(asset_id,) for asset_id in ids])
        return len(ids)

    def delete_characters(self, character_ids: Iterable[str]) -> int:
        ids = [character_id for character_id in character_ids if character_id]
        if not ids:
            return 0
        with self.tx() as con:
            for character_id in ids:
                con.execute("DELETE FROM character_assets WHERE character_id = ?", (character_id,))
                con.execute("DELETE FROM characters WHERE id = ?", (character_id,))
        return len(ids)

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




def _decode_name_list(value: Any) -> list[str]:
    """Decode a stored character-name list (N40 映射用）。

    ``Store`` 存 list 时是 JSON（见 ``_dump``），旧行/手写行可能
    是 ``"、"`` / ``","`` / ``";"`` 分隔的纯文本；NULL/空串读作
    空列表，绝不编造名字。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(v).strip() for v in data if str(v).strip()]
    except Exception:
        pass
    return [p.strip() for p in re.split(r"[、,，;；]", text) if p.strip()]


def _dump(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, bool):
        return int(value)
    return value
