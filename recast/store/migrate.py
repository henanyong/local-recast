"""Database schema versioning and migrations.

V0.1 databases have no schema_meta table (treated as version 1).
Opening them with this code upgrades to SCHEMA_VERSION so old tasks can resume.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 3

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
    _set_version(con, SCHEMA_VERSION)
    con.commit()
    return SCHEMA_VERSION
