"""Local dark batch workbench: FastAPI + static dashboard.

Runs on the user's machine, scans local folders, queues episode clips, and
drives the existing V1 recast pipeline (including single-stage reruns).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from recast import __version__
from recast.config import DEFAULT_CONFIG, PROJECT_ROOT
from recast.pipeline import STAGE_GROUP_IDS, group_labels
from recast.runtime import CancelledError, bind_job, clear_stop, request_stop, stopping

log = logging.getLogger("recast")

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", "recast-runs", ".insightface"}

STATE_PATH = PROJECT_ROOT / "outputs" / "workbench-state.json"
ASSETS_DIR = PROJECT_ROOT / "outputs" / "workbench-assets"
RUNS_DIR = PROJECT_ROOT / "outputs" / "workbench"
STATIC_DIR = Path(__file__).resolve().parent / "workbench_static"

STATUS_PENDING = "待执行"
STATUS_RUNNING = "处理中"
STATUS_NEED_ASSETS = "待补资料"
STATUS_REVIEW = "待人工检查"
STATUS_DONE = "已完成"
STATUS_FAILED = "失败"

INCOMPLETE = {STATUS_PENDING, STATUS_NEED_ASSETS, STATUS_FAILED}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class LogHub:
    def __init__(self, maxlen: int = 2000) -> None:
        self.lines: deque[str] = deque(maxlen=maxlen)
        self._subs: list[Queue[str]] = []
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        text = line.rstrip("\n")
        if not text:
            return
        with self._lock:
            self.lines.append(text)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(text)
            except Exception:
                pass

    def subscribe(self) -> Queue[str]:
        q: Queue[str] = Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: Queue[str]) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def dump(self) -> list[str]:
        with self._lock:
            return list(self.lines)


class HubHandler(logging.Handler):
    def __init__(self, hub: LogHub) -> None:
        super().__init__(level=logging.INFO)
        self.hub = hub
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.hub.append(self.format(record))
        except Exception:
            pass


class Board:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tasks: list[dict[str, Any]] = []
        self.scan_root: str | None = None
        self.current_id: str | None = None
        self.running: bool = False
        self.run_total: int = 0
        self.run_done: int = 0
        self.job_queue: list[tuple[str, list[str]]] = []
        self.hub = LogHub()
        self._last_save = 0.0
        self._save_pending = False
        self._save_timer: threading.Timer | None = None
        self._load()
        handler = HubHandler(self.hub)
        recast_log = logging.getLogger("recast")
        if not any(isinstance(h, HubHandler) for h in recast_log.handlers):
            recast_log.addHandler(handler)
            recast_log.setLevel(logging.INFO)
            recast_log.propagate = False

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            current_pct = 0.0
            if self.current_id:
                for t in self.tasks:
                    if t["id"] == self.current_id:
                        current_pct = float(t.get("progress_pct") or 0)
                        break
            overall = 0.0
            if self.run_total:
                overall = (self.run_done + current_pct) / self.run_total
            return {
                "tasks": copy.deepcopy(self.tasks),
                "scan_root": self.scan_root,
                "current_id": self.current_id,
                "running": self.running,
                "run_total": self.run_total,
                "run_done": self.run_done,
                "overall": min(1.0, overall),
                "version": __version__,
            }

    def _load(self) -> None:
        if not STATE_PATH.is_file():
            return
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        self.scan_root = data.get("scan_root")
        tasks = data.get("tasks") or []
        for t in tasks:
            if t.get("status") == STATUS_RUNNING:
                t["status"] = STATUS_PENDING
                t["progress"] = "上次未完成，已恢复为待执行"
                t["progress_pct"] = 0
        self.tasks = tasks

    def save(self, *, flush: bool = False) -> None:
        """Persist state. Progress updates are coalesced to about twice a second."""
        if not flush:
            now = time.monotonic()
            with self.lock:
                due = now - self._last_save >= 0.5
                if not due:
                    self._save_pending = True
                    if self._save_timer is None or not self._save_timer.is_alive():
                        delay = max(0.05, 0.5 - (now - self._last_save))
                        self._save_timer = threading.Timer(delay, self._flush_save)
                        self._save_timer.daemon = True
                        self._save_timer.start()
                    return
        self._flush_save()

    def _flush_save(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            payload = {"scan_root": self.scan_root, "tasks": self.tasks, "saved_at": _now()}
            self._save_pending = False
            self._last_save = time.monotonic()
        STATE_PATH.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        return None

    def update_task(self, task_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.lock:
            task = self.get_task(task_id)
            if not task:
                return None
            task.update(fields)
            task["updated_at"] = _now()
            copy = dict(task)
        self.save()
        return copy


BOARD = Board()


def _list_videos(folder: Path, *, recursive: bool) -> list[Path]:
    out: list[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            if any(part in SKIP_DIR_NAMES or part == "workbench-assets" for part in p.parts):
                continue
            out.append(p)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                out.append(p)
    return sorted(out, key=lambda x: x.as_posix().lower())


def _new_task(name: str, path: Path, clips: list[Path], kind: str) -> dict[str, Any]:
    now = _now()
    return {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "path": str(path),
        "kind": kind,
        "clips": [str(c) for c in clips],
        "episode_count": len(clips),
        "status": STATUS_PENDING,
        "progress": "等待执行",
        "progress_pct": 0.0,
        "created_at": now,
        "updated_at": now,
        "output_dir": None,
        "output_file": None,
        "error": None,
        "workdir_map": {},
        "settings": {
            "face": None,
            "names": None,
            "config": "configs/4060ti-8g.yaml",
            "lang": "auto",
            "burn_subs": None,
            "track_id": None,
        },
    }


def _belongs_to_root(path: str | Path, root: str | Path) -> bool:
    try:
        child = Path(path).expanduser().resolve()
        base = Path(root).expanduser().resolve()
    except OSError:
        return str(path).startswith(str(root))
    if child == base:
        return True
    try:
        child.relative_to(base)
        return True
    except ValueError:
        return False


def scan_directory(root: str | Path, *, merge: bool = True) -> dict[str, Any]:
    folder = Path(root).expanduser()
    file_scan = False
    if folder.is_file() and folder.suffix.lower() in VIDEO_EXTS:
        found = [_new_task(folder.stem, folder, [folder], "video")]
        root_s = str(folder.parent.resolve())
        file_scan = True
    else:
        folder = folder.resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到目录: {folder}")
        found: list[dict[str, Any]] = []
        for video in _list_videos(folder, recursive=False):
            found.append(_new_task(video.stem, video, [video], "video"))
        for sub in sorted([p for p in folder.iterdir() if p.is_dir() and p.name not in SKIP_DIR_NAMES and not p.name.startswith(".")], key=lambda x: x.name.lower()):
            clips = _list_videos(sub, recursive=True)
            if clips:
                found.append(_new_task(sub.name, sub, clips, "folder"))
        root_s = str(folder)

    with BOARD.lock:
        existing_by_path = {t["path"]: t for t in BOARD.tasks} if merge else {}
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for task in found:
            old = existing_by_path.get(task["path"])
            if old:
                old["name"] = task["name"]
                old["clips"] = task["clips"]
                old["episode_count"] = task["episode_count"]
                old["kind"] = task["kind"]
                old["updated_at"] = _now()
                merged.append(old)
                seen.add(old["path"])
            else:
                merged.append(task)
                seen.add(task["path"])
        if merge:
            # Keep tasks from other scan roots, and running jobs. Same-root
            # entries that disappeared are dropped; use "移除勾选" to delete others.
            for old in BOARD.tasks:
                if old["path"] in seen:
                    continue
                if old.get("status") == STATUS_RUNNING or file_scan or not _belongs_to_root(old["path"], root_s):
                    merged.append(old)
        BOARD.tasks = merged
        BOARD.scan_root = root_s
    BOARD.save(flush=True)
    BOARD.hub.append(f"{_now()[11:]} [INFO] 已扫描 {root_s}，队列 {len(merged)} 项")
    return BOARD.snapshot()


def refresh_need_assets(task: dict[str, Any], stages: list[str] | None = None) -> str:
    """If replace is requested and no face is set, mark 待补资料; names are optional."""
    face = (task.get("settings") or {}).get("face")
    has_face = bool(face and Path(str(face)).is_file())
    wants_replace = not stages or "compose" in stages or "replace" in stages
    if wants_replace and not has_face:
        return STATUS_NEED_ASSETS
    if task["status"] == STATUS_NEED_ASSETS and (has_face or not wants_replace):
        return STATUS_PENDING
    return task["status"]


def update_settings(task_id: str, settings: dict[str, Any], *, apply_ids: list[str] | None = None) -> dict[str, Any]:
    targets = apply_ids or [task_id]
    with BOARD.lock:
        for tid in targets:
            task = BOARD.get_task(tid)
            if not task:
                continue
            cur = dict(task.get("settings") or {})
            for key in ("face", "names", "config", "lang", "burn_subs", "track_id"):
                if key in settings:
                    cur[key] = settings[key]
            task["settings"] = cur
            # Re-evaluate 待补资料 when a face is added or removed
            if task["status"] in {STATUS_PENDING, STATUS_NEED_ASSETS, STATUS_FAILED}:
                previous = task["status"]
                task["status"] = refresh_need_assets(task)
                if previous == STATUS_NEED_ASSETS and task["status"] == STATUS_PENDING:
                    task["progress"] = "资料已补齐，等待执行"
                elif previous != STATUS_NEED_ASSETS and task["status"] == STATUS_NEED_ASSETS:
                    task["progress"] = "缺少参考脸图片，请在右侧任务设置中补充"
                task["updated_at"] = _now()
    BOARD.save(flush=True)
    return BOARD.snapshot()


def save_upload(task_id: str, field: str, filename: str, data: bytes) -> dict[str, Any]:
    task = BOARD.get_task(task_id)
    if not task:
        raise FileNotFoundError(f"找不到任务 {task_id}")
    dest_dir = ASSETS_DIR / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name or field
    dest = dest_dir / safe
    dest.write_bytes(data)
    settings = {field: str(dest)}
    return update_settings(task_id, settings)


def browse(path: str | None = None, *, kind: str = "dir") -> dict[str, Any]:
    if not path:
        entries = []
        roots: list[Path] = []
        if os.name == "nt":
            import string

            for letter in string.ascii_uppercase:
                drive = Path(f"{letter}:/")
                if drive.exists():
                    roots.append(drive)
        else:
            roots.append(Path("/"))
        home = Path.home()
        extras = [home, home / "Desktop", home / "Videos", home / "Desktop", PROJECT_ROOT, Path("/workspace")]
        for extra in extras:
            if extra.exists() and extra not in roots:
                roots.insert(0, extra) if extra == home or extra == PROJECT_ROOT else roots.append(extra)
        # unique, home first
        uniq: list[Path] = []
        for r in [home, PROJECT_ROOT, *roots]:
            rp = r.resolve() if r.exists() else r
            if r.exists() and rp not in uniq:
                uniq.append(rp)
        for r in uniq:
            entries.append(_browse_entry(r, as_self=True))
        return {"path": "", "parent": None, "entries": entries}

    folder = Path(path).expanduser()
    if folder.is_file():
        folder = folder.parent
    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"找不到目录: {folder}")
    entries: list[dict[str, Any]] = []
    try:
        children = list(folder.iterdir())
    except PermissionError as exc:
        raise PermissionError(f"没有权限读取: {folder}") from exc
    for child in sorted(children, key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith(".") or child.name in SKIP_DIR_NAMES:
            continue
        if child.is_dir():
            entries.append(_browse_entry(child))
        elif kind in {"all", "video"} and child.suffix.lower() in VIDEO_EXTS:
            entries.append(_browse_entry(child))
        elif kind in {"all", "image"} and child.suffix.lower() in IMAGE_EXTS:
            entries.append(_browse_entry(child))
        elif kind == "names" and child.suffix.lower() in {".yaml", ".yml", ".txt", ".json"}:
            entries.append(_browse_entry(child))
    parent = str(folder.parent) if folder.parent != folder else None
    return {"path": str(folder), "parent": parent, "entries": entries}


def _browse_entry(path: Path, *, as_self: bool = False) -> dict[str, Any]:
    is_dir = path.is_dir()
    video_count = 0
    if is_dir:
        try:
            video_count = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
        except Exception:
            video_count = 0
    return {
        "name": path.name or str(path),
        "path": str(path),
        "is_dir": is_dir,
        "is_video": (not is_dir) and path.suffix.lower() in VIDEO_EXTS,
        "is_image": (not is_dir) and path.suffix.lower() in IMAGE_EXTS,
        "video_count": video_count,
    }


def open_path(path: str | Path) -> None:
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"找不到路径: {target}")
    if target.is_file():
        target = target.parent
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def clear_completed() -> dict[str, Any]:
    with BOARD.lock:
        BOARD.tasks = [t for t in BOARD.tasks if t.get("status") != STATUS_DONE]
    BOARD.save(flush=True)
    BOARD.hub.append(f"{_now()[11:]} [INFO] 已清除完成任务")
    return BOARD.snapshot()


def remove_tasks(ids: list[str]) -> dict[str, Any]:
    drop = set(ids)
    with BOARD.lock:
        BOARD.tasks = [t for t in BOARD.tasks if t["id"] not in drop or t["status"] == STATUS_RUNNING]
    BOARD.save(flush=True)
    return BOARD.snapshot()


def _clip_workdir(task: dict[str, Any], clip: Path, idx: int | None = None) -> Path:
    from recast.media import workdir_leaf, write_source_record

    mapping = task.setdefault("workdir_map", {})
    try:
        key = str(clip.resolve())
    except OSError:
        key = str(clip)
    if key in mapping and Path(mapping[key]).is_dir():
        return Path(mapping[key])
    # also reuse a mapping stored under the original (unresolved) path
    if str(clip) in mapping and Path(mapping[str(clip)]).is_dir():
        mapping[key] = mapping[str(clip)]
        return Path(mapping[key])
    if idx is None:
        clips = [str(c) for c in (task.get("clips") or [])]
        try:
            idx = clips.index(str(clip)) + 1
        except ValueError:
            idx = len(mapping) + 1
    leaf = workdir_leaf(clip, idx)
    dest = RUNS_DIR / task["id"] / leaf
    if dest.exists():
        # collision with a different source: add a unique suffix
        n = 2
        while dest.exists() and str(dest) not in mapping.values():
            dest = RUNS_DIR / task["id"] / f"{leaf}_{n}"
            n += 1
    dest.mkdir(parents=True, exist_ok=True)
    write_source_record(dest, clip)
    mapping[key] = str(dest)
    if str(clip) != key:
        mapping[str(clip)] = str(dest)
    return dest


def _run_one_clip(task: dict[str, Any], clip: Path, stages: list[str], idx: int, total: int, *, force_rerun: bool = False) -> dict[str, Any]:
    settings = task.get("settings") or {}
    workdir = _clip_workdir(task, clip, idx)
    BOARD.update_task(
        task["id"],
        output_dir=str(workdir.parent if total > 1 else workdir),
        progress=f"{clip.stem}  ·  {idx}/{total}  开始",
    )

    from recast.pipeline import process

    def on_progress(stage: str, message: str, fraction: float) -> None:
        prefix = f"{idx}/{total} · " if total > 1 else ""
        BOARD.update_task(
            task["id"],
            progress=f"{prefix}{stage} · {message}",
            progress_pct=fraction,
            status=STATUS_RUNNING,
        )

    face = settings.get("face")
    face_path = face if face and Path(str(face)).is_file() else None
    skip_replace = face_path is None
    return process(
        clip,
        face=face_path,
        out=workdir / "output.mp4",
        config=settings.get("config") or "configs/4060ti-8g.yaml",
        track_id=settings.get("track_id"),
        names=settings.get("names") or None,
        skip_replace=skip_replace,
        lang=settings.get("lang") or "auto",
        burn_subs=settings.get("burn_subs"),
        progress=on_progress,
        stages=stages,
        workdir=workdir,
        force_rerun=force_rerun,
    )


def _worker() -> None:
    try:
        while True:
            with BOARD.lock:
                if not BOARD.job_queue:
                    break
                item = BOARD.job_queue.pop(0)
                if len(item) >= 3:
                    task_id, stages, force_rerun = item[0], item[1], bool(item[2])
                else:
                    task_id, stages = item[0], item[1]
                    force_rerun = False
                task = BOARD.get_task(task_id)
                BOARD.current_id = task_id
                BOARD.running = True
            bind_job()
            if not task:
                with BOARD.lock:
                    BOARD.run_done += 1
                continue

            requested = list(stages or STAGE_GROUP_IDS)
            status_guess = refresh_need_assets(task, requested)
            wanted = list(requested)
            if status_guess == STATUS_NEED_ASSETS and "compose" in wanted:
                BOARD.hub.append(f"任务「{task['name']}」缺少参考脸，先跑前置阶段，替换合成将等待补资料。")
                pre = [s for s in wanted if s != "compose"]
                if not pre:
                    BOARD.update_task(
                        task_id,
                        status=STATUS_NEED_ASSETS,
                        progress="缺少参考脸图片，请在右侧任务设置中补充",
                    )
                    with BOARD.lock:
                        BOARD.run_done += 1
                    continue
                wanted = pre

            BOARD.update_task(task_id, status=STATUS_RUNNING, progress="开始处理", progress_pct=0, error=None)
            clips = [Path(c) for c in (task.get("clips") or [task["path"]])]
            last_output = None
            last_dir = None
            try:
                for i, clip in enumerate(clips, start=1):
                    if stopping():
                        raise CancelledError("已停止当前任务")
                    if not clip.is_file():
                        raise FileNotFoundError(f"找不到视频: {clip}")
                    summary = _run_one_clip(task, clip, wanted, i, len(clips), force_rerun=force_rerun)
                    last_output = summary.get("output") or last_output
                    last_dir = summary.get("workdir") or last_dir
                end_status = STATUS_DONE
                progress_text = "全部完成"
                if status_guess == STATUS_NEED_ASSETS and "compose" in requested:
                    end_status = STATUS_NEED_ASSETS
                    progress_text = "前置阶段已完成，请补充参考脸后执行「05 本地替换合成」"
                out_dir = None
                if last_dir:
                    out_dir = str(Path(last_dir).parent if len(clips) > 1 else last_dir)
                BOARD.update_task(
                    task_id,
                    status=end_status,
                    progress=progress_text,
                    progress_pct=1.0 if end_status == STATUS_DONE else 0.9,
                    output_file=last_output,
                    output_dir=out_dir or task.get("output_dir"),
                    error=None,
                )
            except CancelledError:
                BOARD.hub.append(f"任务「{task['name']}」已停止，产物保留，可继续执行。")
                BOARD.update_task(task_id, status=STATUS_PENDING, progress="已停止，可继续", error=None)
                # Only the current job is cancelled; queued jobs keep running.
            except Exception as exc:
                log.exception("任务失败: %s", task["name"])
                BOARD.update_task(task_id, status=STATUS_FAILED, progress=f"失败: {exc}", error=str(exc))
            with BOARD.lock:
                BOARD.run_done += 1
    finally:
        with BOARD.lock:
            BOARD.current_id = None
            BOARD.running = False
        BOARD.save(flush=True)
        BOARD.hub.append("队列空闲。")



_worker_thread: threading.Thread | None = None
_worker_guard = threading.Lock()


def start_jobs(ids: list[str], stages: list[str] | None, *, include_done: bool = False, force_rerun: bool = False) -> dict[str, Any]:
    if not ids:
        raise ValueError("请先勾选要执行的任务。")
    stages = list(stages or STAGE_GROUP_IDS)
    queued = 0
    skipped_assets = 0
    with BOARD.lock:
        if BOARD.running:
            # append to queue
            existing = {item[0] for item in BOARD.job_queue}
            if BOARD.current_id:
                existing.add(BOARD.current_id)
        else:
            existing = set()
            BOARD.job_queue = []
            BOARD.run_done = 0
            BOARD.run_total = 0
        for tid in ids:
            task = BOARD.get_task(tid)
            if not task:
                continue
            if task["status"] == STATUS_RUNNING or tid in existing:
                continue
            if task["status"] == STATUS_DONE and not include_done:
                continue
            # If only compose is requested and no face, park as 待补资料
            if stages == ["compose"] and not ((task.get("settings") or {}).get("face") and Path(str((task.get("settings") or {}).get("face"))).is_file()):
                task["status"] = STATUS_NEED_ASSETS
                task["progress"] = "缺少参考脸图片，请在右侧补充后再执行替换合成"
                skipped_assets += 1
                continue
            BOARD.job_queue.append((tid, stages, force_rerun))
            queued += 1
        BOARD.run_total = max(BOARD.run_total, BOARD.run_done + len(BOARD.job_queue))
        if BOARD.running:
            total = BOARD.run_total
        else:
            total = len(BOARD.job_queue)
            BOARD.run_total = total
            BOARD.run_done = 0
    BOARD.save(flush=True)
    if queued == 0 and skipped_assets and not BOARD.running:
        BOARD.hub.append("勾选的任务都缺少参考脸，已标记为「待补资料」。")
        return BOARD.snapshot()
    if queued == 0:
        raise ValueError("没有可执行的任务。请勾选待处理、失败或待补资料的项目。")
    BOARD.hub.append(f"已加入 {queued} 个任务到执行队列（阶段: {', '.join(stages)}）。")
    _ensure_worker()
    return BOARD.snapshot()


def _ensure_worker() -> None:
    global _worker_thread
    with _worker_guard:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        bind_job()
        clear_stop()
        _worker_thread = threading.Thread(target=_safe_worker, name="recast-worker", daemon=True)
        _worker_thread.start()


def _safe_worker() -> None:
    try:
        _worker()
    except Exception:
        log.exception("工作线程异常退出")
    finally:
        with BOARD.lock:
            BOARD.running = False
            BOARD.current_id = None


def stop_current() -> dict[str, Any]:
    request_stop()
    BOARD.hub.append("已请求停止当前任务，正在等待当前阶段安全退出…")
    return BOARD.snapshot()


def launch(host: str = "127.0.0.1", port: int = 7860, *, open_browser: bool = True) -> None:
    import uvicorn
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    if not STATIC_DIR.is_dir():
        raise RuntimeError(f"找不到工作台静态文件目录: {STATIC_DIR}")

    app = FastAPI(title="本地再铸工作台", version=__version__)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    def api_state() -> dict[str, Any]:
        data = BOARD.snapshot()
        data["stages"] = group_labels()
        data["configs"] = _config_list()
        return data

    @app.get("/api/logs")
    def api_logs() -> dict[str, Any]:
        return {"lines": BOARD.hub.dump()}

    @app.get("/api/events")
    def api_events() -> StreamingResponse:
        def stream():
            q = BOARD.hub.subscribe()
            try:
                payload = json.dumps({"type": "state", "data": BOARD.snapshot()}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                for line in BOARD.hub.dump()[-200:]:
                    yield f"data: {json.dumps({'type': 'log', 'line': line}, ensure_ascii=False)}\n\n"
                last_state = time.time()
                while True:
                    try:
                        line = q.get(timeout=0.5)
                        yield f"data: {json.dumps({'type': 'log', 'line': line}, ensure_ascii=False)}\n\n"
                    except Empty:
                        pass
                    if time.time() - last_state >= 0.6:
                        yield f"data: {json.dumps({'type': 'state', 'data': BOARD.snapshot()}, ensure_ascii=False)}\n\n"
                        last_state = time.time()
            finally:
                BOARD.hub.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/browse")
    def api_browse(path: str = "", kind: str = "dir") -> dict[str, Any]:
        try:
            return browse(path or None, kind=kind)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.post("/api/scan")
    def api_scan(body: dict[str, Any]) -> dict[str, Any]:
        path = (body or {}).get("path")
        if not path:
            raise HTTPException(400, "请选择要扫描的目录。")
        try:
            return scan_directory(path, merge=bool((body or {}).get("merge", True)))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/tasks/{task_id}/settings")
    def api_settings(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if not BOARD.get_task(task_id):
            raise HTTPException(404, "找不到该任务。")
        settings = body.get("settings") or body
        apply_ids = body.get("apply_ids")
        return update_settings(task_id, settings, apply_ids=apply_ids)

    @app.post("/api/tasks/{task_id}/upload")
    async def api_upload(task_id: str, field: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
        if field not in {"face", "names"}:
            raise HTTPException(400, "只能上传 face 或 names")
        data = await file.read()
        if not data:
            raise HTTPException(400, "空文件")
        try:
            return save_upload(task_id, field, file.filename or field, data)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/start")
    def api_start(body: dict[str, Any]) -> dict[str, Any]:
        ids = list(body.get("ids") or [])
        stages = body.get("stages")
        include_done = bool(body.get("include_done"))
        force_rerun = bool(body.get("force_rerun"))
        try:
            return start_jobs(ids, stages, include_done=include_done, force_rerun=force_rerun)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/stop")
    def api_stop() -> dict[str, Any]:
        return stop_current()

    @app.post("/api/clear-completed")
    def api_clear() -> dict[str, Any]:
        return clear_completed()

    @app.post("/api/remove")
    def api_remove(body: dict[str, Any]) -> dict[str, Any]:
        return remove_tasks(list(body.get("ids") or []))

    @app.post("/api/open")
    def api_open(body: dict[str, Any]) -> dict[str, Any]:
        path = (body or {}).get("path")
        if not path:
            raise HTTPException(400, "缺少路径")
        try:
            open_path(path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"无法打开文件夹: {exc}") from exc
        return {"ok": True}

    @app.exception_handler(Exception)
    async def on_error(request, exc):  # type: ignore[no-untyped-def]
        if isinstance(exc, HTTPException):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        log.exception("工作台接口异常")
        return JSONResponse({"detail": str(exc)}, status_code=500)

    if host not in {"127.0.0.1", "localhost", "::1"}:
        warning = (
            f"工作台正监听 {host}:{port}。/api/browse 与 /api/open 没有鉴权，"
            "局域网内任何人都可以浏览本机目录并打开路径。请只在受信任的网络上使用，或改为监听 127.0.0.1。"
        )
        log.warning(warning)
        print(warning, flush=True)
        BOARD.hub.append(warning)
    BOARD.hub.append(f"本地再铸工作台 v{__version__}  已启动  http://{host}:{port}")
    log.info("工作台: http://%s:%s", host, port)
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(f"http://{host}:{port}")
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")


def _config_list() -> list[dict[str, str]]:
    items = []
    for p in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        rel = str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
        label = p.stem
        if "8g" in p.stem:
            label = "4060 Ti 8GB（推荐）"
        elif "16g" in p.stem:
            label = "4060 Ti 16GB"
        items.append({"id": rel, "label": label})
    if not items:
        items.append({"id": str(DEFAULT_CONFIG.name), "label": DEFAULT_CONFIG.name})
    return items
