"""P3 schema bundle (N11 + N18 + N30, one migration to version 4).

Additive-only migration: old databases must keep opening, old rows must
keep reading, and no data is dropped. Uses throwaway SQLite files in
tmp_path — never the user's real library. Synthetic rows only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from recast.store.db import Store
from recast.store.migrate import SCHEMA_VERSION, current_version

# Faithful pre-P3 (v1-era) schema: dramas.root_path UNIQUE, no source_path,
# tasks without the N11 columns, no auth_skips table. The v2/v3 steps of the
# migration chain add their own columns on the way to v4.
OLD_SCHEMA = """
CREATE TABLE dramas (
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
    updated_at TEXT
);
CREATE TABLE tasks (
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
    FOREIGN KEY (drama_id) REFERENCES dramas(id) ON DELETE CASCADE
);
CREATE TABLE shots (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    index_no INTEGER,
    start_time REAL,
    end_time REAL,
    start_frame INTEGER,
    end_frame INTEGER,
    status TEXT,
    output_path TEXT
);
CREATE TABLE templates (
    id TEXT PRIMARY KEY,
    name TEXT,
    level INTEGER,
    payload TEXT,
    created_at TEXT
);
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    drama_id TEXT,
    step TEXT,
    status TEXT,
    started_at TEXT,
    finished_at TEXT,
    log_path TEXT,
    error TEXT
);
CREATE TABLE characters (
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
    updated_at TEXT
);
CREATE TABLE character_assets (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT,
    complete INTEGER DEFAULT 0,
    notes TEXT
);
CREATE TABLE checkpoints (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    step TEXT NOT NULL,
    input_hashes TEXT,
    output_paths TEXT,
    created_at TEXT,
    UNIQUE(task_id, step)
);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE logs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, task_id TEXT, message TEXT);
CREATE TABLE subtitles (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    source TEXT,
    start_time REAL,
    end_time REAL,
    text TEXT,
    conf REAL
);
"""


def _make_old_db(path: Path, *, with_tasks: bool = True) -> Path:
    """Build a pre-P3 database file on disk (old schema, old rows)."""
    con = sqlite3.connect(str(path))
    con.executescript(OLD_SCHEMA)
    con.execute(
        "INSERT INTO dramas(id, origin_name, new_name, root_path, status, progress)"
        " VALUES (?,?,?,?,?,?)",
        ("d1", "老剧", "老剧", str(path.parent / "drama"), "待执行", "等待执行"),
    )
    if with_tasks:
        con.execute(
            "INSERT INTO tasks(id, drama_id, episode_name, video_path, status)"
            " VALUES (?,?,?,?,?)",
            ("t1", "d1", "第01集", str(path.parent / "drama" / "第01集.mp4"), "已完成"),
        )
        con.execute(
            "INSERT INTO tasks(id, drama_id, episode_name, video_path, status)"
            " VALUES (?,?,?,?,?)",
            ("t2", "d1", "第02集", str(path.parent / "drama" / "第02集.mp4"), "待执行"),
        )
    con.commit()
    con.close()
    return path


def _columns(store: Store, table: str) -> set[str]:
    return {r[1] for r in store.connect().execute(f"PRAGMA table_info({table})")}


# ----- N11: history snapshot fields -----


def test_n11_fresh_db_has_snapshot_columns(tmp_path: Path):
    store = Store(tmp_path / "new.db")
    assert current_version(store.connect()) == SCHEMA_VERSION
    cols = _columns(store, "tasks")
    assert "selected_characters" in cols
    assert "stage_list" in cols


def test_n11_snapshot_shows_stored_characters_and_stages(tmp_path: Path):
    from recast.core.task_history import get_history_entry

    store = Store(tmp_path / "new.db")
    drama = store.insert(
        "dramas", origin_name="剧", new_name="剧",
        root_path=str(tmp_path / "drama"), episode_count=1,
    )
    row = store.insert(
        "tasks", drama_id=drama["id"], episode_name="第1集",
        video_path=str(tmp_path / "第1集.mp4"), status="已完成",
        selected_characters=["温念", "陆寒峥"],
        stage_list=["scan", "extract", "replace"],
    )
    snap = get_history_entry(store, row["id"])["config_snapshot"]
    assert snap["选的人物"] == "温念、陆寒峥"
    assert snap["阶段清单"] == "scan、extract、replace"


def test_n11_snapshot_old_row_without_fields_shows_unrecorded(tmp_path: Path):
    """Old rows (NULL in the new columns) read as “未记录”, never crash."""
    from recast.core.task_history import build_config_snapshot, get_history_entry

    path = _make_old_db(tmp_path / "old.db")
    store = Store(path)
    assert current_version(store.connect()) == SCHEMA_VERSION
    snap = get_history_entry(store, "t1")["config_snapshot"]
    assert snap["选的人物"] == "未记录"
    assert snap["阶段清单"] == "未记录"
    assert snap["视频"].endswith("第01集.mp4")
    # Hand-built entry dicts without the keys must not crash either.
    assert build_config_snapshot({"video_path": "x.mp4"})["选的人物"] == "未记录"
    assert build_config_snapshot({"video_path": "x.mp4"})["阶段清单"] == "未记录"
    # ... and a plain-text (non-JSON) stored value is shown as-is.
    assert build_config_snapshot(
        {"selected_characters": "温念、陆寒峥", "stage_list": ""}
    )["选的人物"] == "温念、陆寒峥"


def test_n11_rerun_carries_snapshot_fields(tmp_path: Path):
    from recast.core.task_history import build_rerun_task

    store = Store(tmp_path / "new.db")
    drama = store.insert(
        "dramas", origin_name="剧", new_name="剧",
        root_path=str(tmp_path / "drama"), episode_count=1,
    )
    row = store.insert(
        "tasks", drama_id=drama["id"], episode_name="第1集",
        video_path=str(tmp_path / "第1集.mp4"), status="失败",
        selected_characters=["温念"], stage_list=["replace"],
    )
    rerun = build_rerun_task(store, row["id"])
    assert rerun["selected_characters"] is not None
    assert rerun["stage_list"] is not None
    assert store.get("tasks", rerun["id"]) is None  # pure builder: not inserted


# ----- N18: durable auth-skip record -----


def test_n18_skip_survives_runner_progress_rewrite(tmp_path: Path):
    """The runner rewrites tasks/dramas progress; auth_skips must survive."""
    store = Store(tmp_path / "new.db")
    drama = store.insert(
        "dramas", origin_name="剧", new_name="剧",
        root_path=str(tmp_path / "drama"), episode_count=1,
    )
    task = store.insert(
        "tasks", drama_id=drama["id"], episode_name="第1集",
        video_path=str(tmp_path / "第1集.mp4"), status="待执行",
        progress="温雪未授权，本次未替换",
    )
    store.record_auth_skip(drama["id"], "温雪", task_id=task["id"], reason="未授权")
    store.record_auth_skip(drama["id"], "陈秋薇", task_id=task["id"], reason="未授权")
    # Same skip recorded again (gate re-confirmed) must not duplicate.
    store.record_auth_skip(drama["id"], "温雪", task_id=task["id"], reason="未授权")
    assert store.skipped_character_names(drama_id=drama["id"]) == ["温雪", "陈秋薇"]
    assert len(store.list_auth_skips(drama_id=drama["id"])) == 2
    assert len(store.list_auth_skips(task_id=task["id"])) == 2
    assert store.list_auth_skips(drama_id="no-such-drama") == []
    # Simulate the runner starting: progress text is overwritten wholesale.
    store.update("tasks", task["id"], progress="换脸进行中", progress_pct=30)
    store.update("dramas", drama["id"], progress="换脸进行中", progress_pct=30)
    assert store.get("tasks", task["id"])["progress"] == "换脸进行中"
    assert store.skipped_character_names(drama_id=drama["id"]) == ["温雪", "陈秋薇"]


def test_n18_old_db_has_empty_skip_table_and_accepts_new_skips(tmp_path: Path):
    path = _make_old_db(tmp_path / "old.db")
    store = Store(path)
    assert current_version(store.connect()) == SCHEMA_VERSION
    assert "auth_skips" in {
        r[0] for r in store.connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert store.list_auth_skips(drama_id="d1") == []
    assert store.skipped_character_names(drama_id="d1") == []
    store.record_auth_skip("d1", "温雪", task_id="t1")
    assert store.skipped_character_names(drama_id="d1") == ["温雪"]
    with pytest.raises(ValueError, match="人物名不能为空"):
        store.record_auth_skip("d1", "  ")


def test_n18_delete_drama_cleans_its_skips(tmp_path: Path):
    store = Store(tmp_path / "new.db")
    drama = store.insert(
        "dramas", origin_name="剧", new_name="剧",
        root_path=str(tmp_path / "drama"), episode_count=1,
    )
    store.record_auth_skip(drama["id"], "温雪")
    assert store.list_auth_skips(drama_id=drama["id"])
    store.delete_drama(drama["id"])
    assert store.list_auth_skips(drama_id=drama["id"]) == []


# ----- N30: drama identity is the video file, not the folder -----


def test_n30_old_db_migrates_source_backfilled_and_tasks_kept(tmp_path: Path):
    """Old library: identity = first task video; task-less row keeps folder."""
    path = _make_old_db(tmp_path / "old.db")
    con = sqlite3.connect(str(path))
    con.execute(
        "INSERT INTO dramas(id, origin_name, new_name, root_path, status)"
        " VALUES (?,?,?,?,?)",
        ("d2", "空剧", "空剧", str(tmp_path / "empty-drama"), "待执行"),
    )
    con.commit()
    con.close()
    store = Store(path)
    assert current_version(store.connect()) == SCHEMA_VERSION
    d1 = store.get("dramas", "d1")
    assert d1["source_path"] == str(tmp_path / "drama" / "第01集.mp4")
    assert d1["root_path"] == str(tmp_path / "drama")
    d2 = store.get("dramas", "d2")
    assert d2["source_path"] == str(tmp_path / "empty-drama")  # fallback: folder
    # No data lost: both old tasks still there, still linked to d1.
    assert sorted(t["episode_name"] for t in store.tasks_for_drama("d1")) == [
        "第01集", "第02集",
    ]
    # Same folder, second drama, different file identity: now allowed.
    d3 = store.upsert_drama_by_path(
        str(tmp_path / "drama"), origin_name="第二集",
        episode_count=1,
        source_path=str(tmp_path / "drama" / "第02集.mp4"),
    )
    assert d3["id"] != "d1"
    assert len(store.list_dramas()) == 3
    # Same file identity twice: still one row, not a duplicate.
    again = store.upsert_drama_by_path(
        str(tmp_path / "drama"), origin_name="第二集",
        episode_count=1,
        source_path=str(tmp_path / "drama" / "第02集.mp4"),
    )
    assert again["id"] == d3["id"]
    assert len(store.list_dramas()) == 3
    # The UNIQUE is real: a raw duplicate insert is refused.
    with pytest.raises(sqlite3.IntegrityError):
        store.insert(
            "dramas", origin_name="重复", root_path=str(tmp_path / "drama"),
            source_path=str(tmp_path / "drama" / "第01集.mp4"),
        )
    # Re-opening the migrated library is a no-op (idempotent migration).
    store2 = Store(path)
    assert current_version(store2.connect()) == SCHEMA_VERSION
    assert store2.get("dramas", "d1")["source_path"] == str(
        tmp_path / "drama" / "第01集.mp4"
    )
    assert len(store2.tasks_for_drama("d1")) == 2


def _service(tmp_path: Path):
    from recast.core.service import RecastService

    out, works = tmp_path / "out", tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


def test_n30_same_folder_two_single_imports_no_longer_replace(tmp_path: Path):
    """The reported N30 scenario: ep1 then ep2 from one folder."""
    folder = tmp_path / "drama"
    folder.mkdir()
    ep1, ep2 = folder / "第一集.mp4", folder / "第二集.mp4"
    ep1.write_bytes(b"fake")
    ep2.write_bytes(b"fake")
    svc = _service(tmp_path)
    svc.scan(str(ep1))
    svc.scan(str(ep2))
    dramas = svc.store.list_dramas()
    assert len(dramas) == 2, "two videos in one folder must be two dramas now"
    videos = sorted(
        t["video_path"]
        for d in dramas
        for t in svc.store.tasks_for_drama(d["id"])
    )
    assert videos == [str(ep1), str(ep2)]
    # Re-importing the first video finds its own row; nothing is replaced.
    svc.scan(str(ep1))
    assert len(svc.store.list_dramas()) == 2
    assert sorted(
        t["video_path"]
        for d in svc.store.list_dramas()
        for t in svc.store.tasks_for_drama(d["id"])
    ) == [str(ep1), str(ep2)]


def test_n30_folder_scan_merges_into_first_episode_drama(tmp_path: Path):
    """Folder scan after single imports: no third drama, no task lost."""
    folder = tmp_path / "drama"
    folder.mkdir()
    ep1, ep2 = folder / "第一集.mp4", folder / "第二集.mp4"
    ep1.write_bytes(b"fake")
    ep2.write_bytes(b"fake")
    svc = _service(tmp_path)
    svc.scan(str(ep1))
    svc.scan(str(ep2))
    svc.scan(str(folder))  # full folder scan: identity = 第一集.mp4
    dramas = svc.store.list_dramas()
    assert len(dramas) == 2, "folder scan must merge into the ep1 drama, not add one"
    first = svc.store.find_drama_by_source(str(ep1))
    assert first is not None
    assert sorted(
        t["video_path"] for t in svc.store.tasks_for_drama(first["id"])
    ) == [str(ep1), str(ep2)]
    # And a later single re-import of ep1 must not drop the ep2 task.
    svc.scan(str(ep1))
    assert sorted(
        t["video_path"] for t in svc.store.tasks_for_drama(first["id"])
    ) == [str(ep1), str(ep2)]


def test_n30_upsert_without_source_falls_back_to_root(tmp_path: Path):
    """Legacy callers (no file identity, e.g. script dramas) keep working."""
    store = Store(tmp_path / "new.db")
    virtual = str(tmp_path / "script-jobs" / "剧本剧")
    row = store.upsert_drama_by_path(
        virtual, origin_name="剧本剧", episode_count=1, source_path=virtual,
    )
    assert row["source_path"] == virtual
    again = store.upsert_drama_by_path(virtual, origin_name="剧本剧", episode_count=1)
    assert again["id"] == row["id"]
    assert len(store.list_dramas()) == 1
