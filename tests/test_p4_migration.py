"""P4: dramas rebuild must not lose data when the copy fails (PM6 §三补充项).

P3's ``_rebuild_dramas_table`` dropped the old table first and then copied
row by row with no row-count check: a mid-copy INSERT failure left a
half-filled ``dramas`` table and no rollback (the old table was gone).
P4 rebuilds into a staging table (``dramas_new``), verifies the copy
(row count == old row count, every old id present, no NULL source_path),
and only then swaps it in by rename. Any failure before the commit must
leave the old table byte-for-byte intact and no half table behind.

Uses throwaway SQLite files in tmp_path — never the user's real library.
Old-library construction is reused from the P3 migration tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import recast.store.migrate as migrate_mod
from recast.store.db import Store
from recast.store.migrate import (
    SCHEMA_VERSION,
    _rebuild_dramas_table,
    current_version,
)
from tests.test_p3_migration import _make_old_db


def _snapshot(con: sqlite3.Connection) -> dict:
    """Full old-table contents, for before/after equality checks."""
    dramas = con.execute("SELECT * FROM dramas ORDER BY id").fetchall()
    tasks = con.execute("SELECT * FROM tasks ORDER BY id").fetchall()
    return {
        "dramas": [tuple(r) for r in dramas],
        "tasks": [tuple(r) for r in tasks],
    }


def _table_names(con: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _snapshot_core(con: sqlite3.Connection) -> dict:
    """Old-schema columns only: a failed Store() open may already have
    committed the additive v2/v4 ALTERs (extra columns), which must not
    count as data loss — only the old columns' values matter here."""
    dramas = con.execute(
        "SELECT id, origin_name, new_name, root_path, status FROM dramas"
        " ORDER BY id"
    ).fetchall()
    tasks = con.execute(
        "SELECT id, drama_id, episode_name, video_path, status FROM tasks"
        " ORDER BY id"
    ).fetchall()
    return {
        "dramas": [tuple(r) for r in dramas],
        "tasks": [tuple(r) for r in tasks],
    }


def _dramas_sql(con: sqlite3.Connection) -> str:
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='dramas'"
    ).fetchone()
    return row[0] if row else ""


def _make_three_drama_db(path: Path) -> Path:
    """Old DB: d1 (2 tasks), d2 (no tasks), d3 (1 task sharing d1's
    first-episode video path, so the identity dedup ``#<id>`` runs)."""
    _make_old_db(path)  # d1 + t1/t2
    con = sqlite3.connect(str(path))
    con.execute(
        "INSERT INTO dramas(id, origin_name, new_name, root_path, status)"
        " VALUES (?,?,?,?,?)",
        ("d2", "空剧", "空剧", str(path.parent / "empty-drama"), "待执行"),
    )
    con.execute(
        "INSERT INTO dramas(id, origin_name, new_name, root_path, status)"
        " VALUES (?,?,?,?,?)",
        ("d3", "撞路剧", "撞路剧", str(path.parent / "other-drama"), "待执行"),
    )
    con.execute(
        "INSERT INTO tasks(id, drama_id, episode_name, video_path, status)"
        " VALUES (?,?,?,?,?)",
        ("t3", "d3", "第01集", str(path.parent / "drama" / "第01集.mp4"), "待执行"),
    )
    con.commit()
    con.close()
    return path


def _assert_old_table_intact(path: Path, before: dict) -> None:
    """The old dramas table must survive a failed rebuild unchanged."""
    con = sqlite3.connect(str(path))
    try:
        assert _snapshot(con) == before
        assert "root_path TEXT NOT NULL UNIQUE" in _dramas_sql(con)
        assert "dramas_new" not in _table_names(con)
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        con.close()


# ----- Positive: normal rebuild, row count matches -----


def test_p4_rebuild_row_count_matches_and_migration_succeeds(tmp_path: Path):
    path = _make_three_drama_db(tmp_path / "old.db")
    store = Store(path)
    assert current_version(store.connect()) == SCHEMA_VERSION
    assert len(store.list_dramas()) == 3
    d1 = store.get("dramas", "d1")
    assert d1["source_path"] == str(tmp_path / "drama" / "第01集.mp4")
    d2 = store.get("dramas", "d2")
    assert d2["source_path"] == str(tmp_path / "empty-drama")  # no tasks: folder
    d3 = store.get("dramas", "d3")
    # Same first-episode video as d1: dedup suffix, no row dropped.
    assert d3["source_path"] == str(tmp_path / "drama" / "第01集.mp4") + "#d3"
    # No data lost, still linked; new schema in place; no staging left.
    assert sorted(t["episode_name"] for t in store.tasks_for_drama("d1")) == [
        "第01集", "第02集",
    ]
    assert len(store.tasks_for_drama("d3")) == 1
    con = store.connect()
    assert "source_path TEXT UNIQUE" in _dramas_sql(con)
    assert "root_path TEXT NOT NULL UNIQUE" not in _dramas_sql(con)
    assert "dramas_new" not in _table_names(con)
    # Rename swap kept the child FKs working (no dangling references).
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    # Re-opening is a no-op (idempotent migration).
    store2 = Store(path)
    assert len(store2.list_dramas()) == 3
    assert store2.get("dramas", "d3")["source_path"] == d3["source_path"]


def test_p4_rebuild_empty_dramas_table(tmp_path: Path):
    """Boundary: an old library with zero dramas rebuilds 0 == 0 rows."""
    from tests.test_p3_migration import OLD_SCHEMA

    path = tmp_path / "empty.db"
    con = sqlite3.connect(str(path))
    con.executescript(OLD_SCHEMA)
    con.commit()
    con.close()
    store = Store(path)
    assert current_version(store.connect()) == SCHEMA_VERSION
    assert store.list_dramas() == []
    assert "dramas_new" not in _table_names(store.connect())
    assert "source_path TEXT UNIQUE" in _dramas_sql(store.connect())


# ----- Failure: mid-copy INSERT error, old table must survive -----


def test_p4_mid_insert_failure_keeps_old_table_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_three_drama_db(tmp_path / "old.db")
    con = sqlite3.connect(str(path))
    before = _snapshot(con)
    original = migrate_mod._insert_drama_row
    calls = {"n": 0}

    def flaky(c, values):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("模拟 INSERT 中途失败（磁盘/约束）")
        return original(c, values)

    monkeypatch.setattr(migrate_mod, "_insert_drama_row", flaky)
    with pytest.raises(RuntimeError, match="模拟 INSERT 中途失败"):
        _rebuild_dramas_table(con)
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1  # restored
    con.close()
    _assert_old_table_intact(path, before)
    # Retry after the fault is gone: the same library migrates cleanly.
    monkeypatch.undo()
    con = sqlite3.connect(str(path))
    _rebuild_dramas_table(con)
    con.close()
    store = Store(path)
    assert current_version(store.connect()) == SCHEMA_VERSION
    assert len(store.list_dramas()) == 3


def test_p4_first_insert_failure_keeps_old_table_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Boundary: failure before a single row is copied is equally safe."""
    path = _make_three_drama_db(tmp_path / "old.db")
    con = sqlite3.connect(str(path))
    before = _snapshot(con)

    def always_fail(c, values):
        raise RuntimeError("模拟首次 INSERT 即失败")

    monkeypatch.setattr(migrate_mod, "_insert_drama_row", always_fail)
    with pytest.raises(RuntimeError, match="模拟首次 INSERT 即失败"):
        _rebuild_dramas_table(con)
    con.close()
    _assert_old_table_intact(path, before)


def test_p4_silent_row_drop_detected_by_count_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A copy that silently drops a row (no exception from INSERT) must
    still be refused by the row-count check — never committed half-full."""
    path = _make_three_drama_db(tmp_path / "old.db")
    con = sqlite3.connect(str(path))
    before = _snapshot(con)
    original = migrate_mod._insert_drama_row
    calls = {"n": 0}

    def drop_one(c, values):
        calls["n"] += 1
        if calls["n"] == 2:
            return  # silently skip: staging ends up 2 rows, old has 3
        return original(c, values)

    monkeypatch.setattr(migrate_mod, "_insert_drama_row", drop_one)
    with pytest.raises(RuntimeError, match="行数不一致"):
        _rebuild_dramas_table(con)
    con.close()
    _assert_old_table_intact(path, before)


def test_p4_store_open_failure_leaves_library_openable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """End to end through Store(): a failed migration must not poison the
    file — the version is not bumped and a later open retries cleanly."""
    path = _make_three_drama_db(tmp_path / "old.db")
    con = sqlite3.connect(str(path))
    before = _snapshot_core(con)
    con.close()
    original = migrate_mod._insert_drama_row
    calls = {"n": 0}

    def flaky(c, values):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("模拟重建中途失败")
        return original(c, values)

    monkeypatch.setattr(migrate_mod, "_insert_drama_row", flaky)
    with pytest.raises(RuntimeError, match="模拟重建中途失败"):
        Store(path)
    monkeypatch.undo()
    # Old rows (old-schema columns) intact; additive ALTER columns from the
    # aborted attempt are fine, the old table itself was never dropped.
    con = sqlite3.connect(str(path))
    try:
        assert _snapshot_core(con) == before
        assert "root_path TEXT NOT NULL UNIQUE" in _dramas_sql(con)
        assert "dramas_new" not in _table_names(con)
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        con.close()
    raw = sqlite3.connect(str(path))
    assert current_version(raw) < SCHEMA_VERSION  # not marked migrated
    raw.close()
    store = Store(path)  # retry: old library opens and migrates normally
    assert current_version(store.connect()) == SCHEMA_VERSION
    assert len(store.list_dramas()) == 3
    assert len(store.tasks_for_drama("d1")) == 2
