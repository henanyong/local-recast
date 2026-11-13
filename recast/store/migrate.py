"""Database schema versioning and migrations.

V0.1 databases have no schema_meta table (treated as version 1).
Opening them with this code upgrades to SCHEMA_VERSION so old tasks can resume.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 4

# Columns added in the P3 schema bundle (version 4, N11/N18/N30, one change).
# N11: tasks.selected_characters / tasks.stage_list (history snapshot fields).
_V4_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "tasks": [
        ("selected_characters", "TEXT"),
        ("stage_list", "TEXT"),
    ],
}

# Columns added in the V0.5 / V1.0 upgrade (version 2).
_V2_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "dramas": [
        ("template_pack_id", "TEXT"),
        ("template_version", "TEXT"),
        ("entry_type", "TEXT DEFAULT 'import'"),
        ("upload_status", "TEXT"),
        ("pack_root", "TEXT"),
    ],
    "tasks": [
        ("template_pack_id", "TEXT"),
        ("template_version", "TEXT"),
        ("entry_type", "TEXT"),
        ("source_script", "TEXT"),
        ("source_dialogue", "TEXT"),
        ("source_storyboard", "TEXT"),
        ("pack_snapshot", "TEXT"),
    ],
    "shots": [
        ("shot_code", "TEXT"),
        ("prompt", "TEXT"),
        ("ref_assets", "TEXT"),
        ("dialogue", "TEXT"),
        ("speaker", "TEXT"),
        ("duration_sec", "REAL"),
        ("primary_action", "TEXT"),
        ("people", "TEXT"),
        ("is_fight", "INTEGER DEFAULT 0"),
        ("quality_flags", "TEXT"),
        ("review_reason", "TEXT"),
        ("scene_id", "TEXT"),
        ("kind", "TEXT"),
        ("filled_prompt", "TEXT"),
    ],
    "templates": [
        ("version", "TEXT"),
        ("json_path", "TEXT"),
        ("pack_id", "TEXT"),
        ("updated_at", "TEXT"),
        ("level_name", "TEXT"),
    ],
    "runs": [
        ("model", "TEXT"),
        ("duration_sec", "REAL"),
        ("cost", "REAL DEFAULT 0"),
        ("message", "TEXT"),
        ("created_at", "TEXT"),
    ],
    "character_assets": [
        ("quality_score", "REAL"),
        ("source_episodes", "TEXT"),
        ("source_note", "TEXT"),
    ],
    "characters": [
        ("aliases", "TEXT"),
        ("face_embedding", "TEXT"),
        ("voice_id", "TEXT"),
        ("source_note", "TEXT"),
    ],
}


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in con.execute(f"PRAGMA table_info({table})")}


def _add_column(con: sqlite3.Connection, table: str, name: str, decl: str) -> None:
    if name in _columns(con, table):
        return
    con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def current_version(con: sqlite3.Connection) -> int:
    if _table_exists(con, "schema_meta"):
        row = con.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()
        if row is not None:
            try:
                return int(row[0])
            except (TypeError, ValueError):
                return 0
    if _table_exists(con, "dramas"):
        return 1
    return 0


def _set_version(con: sqlite3.Connection, version: int) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)"
    )
    con.execute(
        "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(version),),
    )


def _migrate_to_2(con: sqlite3.Connection) -> None:
    for table, cols in _V2_COLUMNS.items():
        if not _table_exists(con, table):
            continue
        for name, decl in cols:
            _add_column(con, table, name, decl)
    # Extra tables used by later stages (safe to create on a V0.1 file).
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS template_packs (
            id TEXT PRIMARY KEY,
            name TEXT,
            version TEXT,
            root_path TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS quality_marks (
            id TEXT PRIMARY KEY,
            task_id TEXT,
            shot_id TEXT,
            shot_code TEXT,
            reason TEXT,
            detail TEXT,
            status TEXT,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_quality_task ON quality_marks(task_id);
        """
    )


def _dramas_needs_rebuild(con: sqlite3.Connection) -> bool:
    """True when the dramas table still carries the old root_path UNIQUE."""
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='dramas'"
    ).fetchone()
    return bool(row and row[0] and "root_path TEXT NOT NULL UNIQUE" in row[0])


def _backfill_source_path(con: sqlite3.Connection) -> None:
    """Fill NULL dramas.source_path from each drama's first task video.

    Fallback is the drama's own root_path (unique under the old schema, so
    it can never collide); a pathological duplicate video path gets a
    ``#<drama id>`` suffix so the UNIQUE index never fails the migration.
    No row is dropped and no video path is invented.
    """
    rows = con.execute(
        "SELECT id, root_path, source_path FROM dramas"
    ).fetchall()
    used = {r[2] for r in rows if r[2]}
    for drama_id, root_path, existing in rows:
        if existing:
            continue
        task = con.execute(
            "SELECT video_path FROM tasks WHERE drama_id = ? "
            "ORDER BY episode_name COLLATE NOCASE LIMIT 1",
            (drama_id,),
        ).fetchone()
        candidate = task[0] if task and task[0] else root_path
        if candidate in used:
            candidate = f"{candidate}#{drama_id}"
        used.add(candidate)
        con.execute(
            "UPDATE dramas SET source_path = ? WHERE id = ?",
            (candidate, drama_id),
        )


_DRAMAS_V4_COLUMNS = [
    "id", "origin_name", "new_name", "root_path", "source_path",
    "episode_count", "status", "progress", "progress_pct", "output_dir",
    "selected", "missing_files", "error", "created_at", "updated_at",
    "template_pack_id", "template_version", "entry_type", "upload_status",
    "pack_root",
]

# Staging table for the dramas rebuild (P4): the copy + row-count check
# happen here, and only a verified copy is renamed to ``dramas``. Not part
# of the schema — a failed rebuild drops it again, never the old table.
_DRAMAS_STAGING_TABLE = "dramas_new"


def _insert_drama_row(con: sqlite3.Connection, values: list) -> None:
    """Insert one copied row into the staging table.

    Small seam (not a behaviour change): tests monkeypatch this to simulate
    a mid-copy INSERT failure without touching sqlite internals.
    """
    cols_sql = ", ".join(_DRAMAS_V4_COLUMNS)
    placeholders = ", ".join("?" for _ in _DRAMAS_V4_COLUMNS)
    con.execute(
        f"INSERT INTO {_DRAMAS_STAGING_TABLE} ({cols_sql}) "
        f"VALUES ({placeholders})",
        values,
    )


def _verify_rebuilt_dramas(
    con: sqlite3.Connection, old_rows: list[sqlite3.Row | tuple]
) -> None:
    """Row-count / identity check on the staging table, before the swap.

    Raises (and therefore aborts the rebuild, keeping the old table) when
    the copy is incomplete: wrong row count, a missing old id, or a NULL
    source_path (UNIQUE alone would still allow several NULLs).
    """
    expected = len(old_rows)
    actual = con.execute(
        f"SELECT COUNT(*) FROM {_DRAMAS_STAGING_TABLE}"
    ).fetchone()[0]
    if actual != expected:
        raise RuntimeError(
            "重建 dramas 行数不一致：旧表 "
            f"{expected} 行，临时表 {actual} 行，拒绝提交，旧表保留。"
        )
    missing = con.execute(
        "SELECT COUNT(*) FROM dramas "
        f"WHERE id NOT IN (SELECT id FROM {_DRAMAS_STAGING_TABLE})"
    ).fetchone()[0]
    if missing:
        raise RuntimeError(
            f"重建 dramas 缺失 {missing} 个旧行，拒绝提交，旧表保留。"
        )
    null_source = con.execute(
        f"SELECT COUNT(*) FROM {_DRAMAS_STAGING_TABLE} "
        "WHERE source_path IS NULL"
    ).fetchone()[0]
    if null_source:
        raise RuntimeError(
            f"重建 dramas 有 {null_source} 行 source_path 为空，拒绝提交，旧表保留。"
        )


def _rebuild_dramas_table(con: sqlite3.Connection) -> None:
    """Replace the dramas table: drop root_path UNIQUE, add source_path.

    SQLite cannot drop a UNIQUE constraint with ALTER TABLE, so the table
    is rebuilt — into a staging table first (``dramas_new``), never by
    dropping the old table up front:

    1. copy every old row into ``dramas_new`` (new schema);
    2. verify the copy (row count == old row count, every old id present,
       no NULL source_path) — any mismatch raises before the swap;
    3. only then DROP the old table and RENAME the staging table to
       ``dramas``, and commit.

    If any step before the commit fails, the transaction is rolled back:
    the old table is untouched (a failed swap is undone by the rollback,
    and the except-path renames a complete staging table back if the old
    one is somehow gone), and no half-filled ``dramas`` table is left
    behind. The tasks/shots/subtitles rows that reference dramas are
    untouched: foreign keys are switched OFF around the swap — with them
    ON, DROP TABLE would cascade-delete the child rows. The switch only
    takes effect outside a transaction, so commit first and verify it.
    """
    con.commit()
    con.execute("PRAGMA foreign_keys = OFF")
    if con.execute("PRAGMA foreign_keys").fetchone()[0] != 0:
        raise RuntimeError("无法关闭外键，拒绝重建 dramas 表（会误删分集任务）。")
    try:
        old_rows = con.execute("SELECT * FROM dramas").fetchall()
        old_cols = [r[1] for r in con.execute("PRAGMA table_info(dramas)")]
        identities: dict[str, str] = {}
        used: set[str] = set()
        for row in old_rows:
            d = dict(zip(old_cols, row))
            task = con.execute(
                "SELECT video_path FROM tasks WHERE drama_id = ? "
                "ORDER BY episode_name COLLATE NOCASE LIMIT 1",
                (d["id"],),
            ).fetchone()
            candidate = task[0] if task and task[0] else d["root_path"]
            if candidate in used:
                candidate = f"{candidate}#{d['id']}"
            used.add(candidate)
            identities[d["id"]] = candidate
        con.execute(f"DROP TABLE IF EXISTS {_DRAMAS_STAGING_TABLE}")
        con.execute(
            f"""
            CREATE TABLE {_DRAMAS_STAGING_TABLE} (
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
            )
            """
        )
        for row in old_rows:
            d = dict(zip(old_cols, row))
            values = [d.get(c) for c in _DRAMAS_V4_COLUMNS]
            values[_DRAMAS_V4_COLUMNS.index("source_path")] = identities[d["id"]]
            _insert_drama_row(con, values)
        _verify_rebuilt_dramas(con, old_rows)
        con.execute("DROP TABLE dramas")
        con.execute(f"ALTER TABLE {_DRAMAS_STAGING_TABLE} RENAME TO dramas")
        con.commit()
    except Exception:
        con.rollback()
        # Rollback undoes the DROP + RENAME if the swap already ran, so
        # the old table is normally back at this point. Belt and braces:
        # if the old table is somehow gone but a complete staging table
        # survived, rename it back instead of leaving no dramas table.
        try:
            if not _table_exists(con, "dramas") and _table_exists(
                con, _DRAMAS_STAGING_TABLE
            ):
                con.execute(
                    f"ALTER TABLE {_DRAMAS_STAGING_TABLE} RENAME TO dramas"
                )
                con.commit()
            elif _table_exists(con, _DRAMAS_STAGING_TABLE):
                con.execute(f"DROP TABLE {_DRAMAS_STAGING_TABLE}")
                con.commit()
        except sqlite3.Error:
            # Cleanup must never mask the original rebuild failure.
            try:
                con.rollback()
            except sqlite3.Error:
                pass
        raise
    finally:
        con.execute("PRAGMA foreign_keys = ON")


def _migrate_to_4(con: sqlite3.Connection) -> None:
    """P3 schema bundle, additive only — old databases keep opening.

    N11: add the two history-snapshot columns to ``tasks`` (NULL for old
    rows; the read path shows "未记录" for NULL). N18/N30 steps are added
    by the later steps of the same bundle (see below).
    """
    for table, cols in _V4_COLUMNS.items():
        if not _table_exists(con, table):
            continue
        for name, decl in cols:
            _add_column(con, table, name, decl)
    # N30: drama identity moves from folder (root_path UNIQUE) to file
    # (source_path UNIQUE). Old tables must be rebuilt — SQLite cannot
    # drop a UNIQUE constraint — and old rows get their identity backfilled.
    if _table_exists(con, "dramas"):
        if _dramas_needs_rebuild(con):
            _rebuild_dramas_table(con)
        else:
            _add_column(con, "dramas", "source_path", "TEXT")
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='dramas'"
            ).fetchone()
            if not (row and row[0] and "source_path TEXT UNIQUE" in row[0]):
                con.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dramas_source_path "
                    "ON dramas(source_path)"
                )
            _backfill_source_path(con)
    # N18: durable auth-skip record (the runner never rewrites this table).
    # SCHEMA also declares it; creating it here keeps migrate() alone working
    # on a raw connection that never ran the full schema script.
    con.executescript(
        """
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
        CREATE INDEX IF NOT EXISTS idx_auth_skips_drama ON auth_skips(drama_id);
        CREATE INDEX IF NOT EXISTS idx_auth_skips_task ON auth_skips(task_id);
        """
    )


def migrate(con: sqlite3.Connection) -> int:
    """Bring `con` up to SCHEMA_VERSION. Returns the version after migration."""
    con.execute("PRAGMA foreign_keys = ON")
    version = current_version(con)
    if version < 2:
        _migrate_to_2(con)
        version = 2
    if version < 3:
        if _table_exists(con, "quality_marks"):
            _add_column(con, "quality_marks", "score", "REAL")
            _add_column(con, "quality_marks", "thumbnail", "TEXT")
        version = 3
    if version < 4:
        _migrate_to_4(con)
        version = 4
    _set_version(con, SCHEMA_VERSION)
    con.commit()
    return SCHEMA_VERSION
