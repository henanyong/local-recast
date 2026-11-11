import sqlite3
from pathlib import Path

from recast.core.constants import ST_PENDING
from recast.store.db import Store
from recast.store.migrate import SCHEMA_VERSION, current_version


V01_SCHEMA = """
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


def test_v01_db_migrates_and_old_task_resumes(tmp_path: Path):
    path = tmp_path / "recast.db"
    con = sqlite3.connect(str(path))
    con.executescript(V01_SCHEMA)
    con.execute(
        "INSERT INTO dramas(id, origin_name, new_name, root_path, status, progress) VALUES (?,?,?,?,?,?)",
        ("d1", "老剧", "老剧", str(tmp_path / "drama"), "待执行", "上次未完成，已恢复为待执行"),
    )
    con.execute(
        "INSERT INTO tasks(id, drama_id, episode_name, video_path, status, workdir, retry_count) VALUES (?,?,?,?,?,?,?)",
        ("t1", "d1", "第01集", str(tmp_path / "ep.mp4"), "待执行", str(tmp_path / "work"), 0),
    )
    con.execute(
        "INSERT INTO checkpoints(id, task_id, step, input_hashes, output_paths, created_at) VALUES (?,?,?,?,?,?)",
        ("c1", "t1", "extract", "{}", "[]", "2026-01-01 00:00:00"),
    )
    con.commit()
    con.close()

    store = Store(path)
    assert current_version(store.connect()) == SCHEMA_VERSION
    # new columns exist
    cols = {r[1] for r in store.connect().execute("PRAGMA table_info(tasks)")}
    assert "template_version" in cols
    assert "entry_type" in cols
    run_cols = {r[1] for r in store.connect().execute("PRAGMA table_info(runs)")}
    assert "duration_sec" in run_cols and "cost" in run_cols
    task = store.get("tasks", "t1")
    assert task["status"] == ST_PENDING
    assert task["episode_name"] == "第01集"
    ckpt = store.list_rows("checkpoints", "task_id = ?", ("t1",))
    assert ckpt and ckpt[0]["step"] == "extract"
    # can write new fields on the old row
    store.update("tasks", "t1", template_version="1.0.0", entry_type="import")
    assert store.get("tasks", "t1")["template_version"] == "1.0.0"
