"""Queue + worker for dramas / episodes. Default concurrency is 1 (full GPU)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from recast.config import RecastConfig, resolve_config_path
from recast.core.constants import DEFAULT_CONCURRENCY, MAX_CONCURRENCY, ST_DONE, ST_FAILED, ST_NEED_ASSETS, ST_PENDING, ST_REVIEW, ST_RUNNING
from recast.core.runner import classify_and_message, run_episode
from recast.core.settings import AppSettings
from recast.core.state_machine import apply
from recast.runtime import CancelledError, clear_stop, request_stop, stopping
from recast.store.db import Store

log = logging.getLogger("recast")

ProgressFn = Callable[[str, str, float], None]
PROGRESS_WRITE_INTERVAL = 0.2


def should_persist_progress(last_write: float, now: float, fraction: float, interval: float = PROGRESS_WRITE_INTERVAL) -> bool:
    """Database progress writes are coalesced. The first write and the final 100% always land."""
    if last_write <= 0:
        return True
    if fraction >= 1.0:
        return True
    return (now - last_write) >= interval


@dataclass
class Job:
    drama_id: str
    task_id: str
    stages: list[str]
    character_id: str | None = None
    character_ids: list[str] | None = None
    face: str | None = None
    track_id: int | None = None
    names: str | None = None
    require_auth: bool = True
    entry_type: str = "import"
    pack_root: str | None = None
    storyboard: str | None = None
    dialogue: str | None = None
    wav_table: str | None = None
    rerun_shot: str | None = None
    comfyui_url: str | None = None
    max_cps: float = 4.0


class Scheduler:
    def __init__(
        self,
        store: Store,
        settings: AppSettings,
        *,
        on_log: Callable[[str, str, str | None], None] | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.on_log = on_log
        self.on_change = on_change
        self._lock = threading.Lock()
        self._queue: deque[Job] = deque()
        self._running_ids: set[str] = set()
        # Running jobs by task id. _running_ids alone cannot say which person a
        # running job uses, and delete_character needs that answer.
        self._running_jobs: dict[str, Job] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[Any]] = []
        self._last_progress_at = 0.0
        self.current_task_id: str | None = None
        self.current_drama_id: str | None = None
        self.run_total = 0
        self.run_done = 0
        self.current_pct = 0.0
        self.current_progress = ""
        self._gpu_sema = threading.Semaphore(1)
        self._cpu_sema = threading.Semaphore(2)
        self._gpu_running: set[str] = set()
        self._cpu_running: set[str] = set()

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._running_ids) or bool(self._queue)

    def jobs_using_character(self, character_id: str) -> list[dict[str, Any]]:
        """Queued or running jobs that reference one character.

        Only live scheduler jobs count: a task row merely marked 待执行 in the
        database has no character binding yet (the binding is made when the
        user starts it), so it cannot be broken by a delete.
        """
        cid = str(character_id)
        with self._lock:
            pending = [(job, "排队中") for job in list(self._queue)]
            active = [(job, "处理中") for job in list(self._running_jobs.values())]
        found: list[dict[str, Any]] = []
        for job, status in pending + active:
            ids = {str(x) for x in (job.character_ids or []) if x}
            if job.character_id:
                ids.add(str(job.character_id))
            if cid not in ids:
                continue
            task = self.store.get("tasks", job.task_id) or {}
            drama = self.store.get("dramas", job.drama_id) or {}
            label = (
                f"{drama.get('new_name') or drama.get('origin_name') or job.drama_id}"
                f" / {task.get('episode_name') or job.task_id}"
            )
            found.append({"task_id": job.task_id, "drama_id": job.drama_id, "status": status, "label": label})
        return found

    def _emit(self, level: str, message: str, task_id: str | None = None) -> None:
        log.log(getattr(logging, level, logging.INFO), message)
        try:
            self.store.add_log(level, message, task_id)
        except Exception:
            pass
        if self.on_log:
            self.on_log(level, message, task_id)

    def _changed(self) -> None:
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                pass

    def _cfg(self) -> RecastConfig:
        from recast.config import apply_runtime_settings

        cfg = RecastConfig.from_yaml(resolve_config_path(self.settings.config_file()))
        return apply_runtime_settings(cfg, self.settings)

    def reap_futures(self) -> int:
        """Drop completed worker futures so the list does not grow for the life of the process."""
        with self._lock:
            before = len(self._futures)
            self._futures = [fut for fut in self._futures if not fut.done()]
            return before - len(self._futures)

    def enqueue(self, jobs: list[Job]) -> int:
        n = 0
        with self._lock:
            existing = {j.task_id for j in self._queue} | set(self._running_ids)
            if not self._running_ids and not self._queue:
                self.run_done = 0
                self.run_total = 0
            for job in jobs:
                if job.task_id in existing:
                    continue
                self._queue.append(job)
                existing.add(job.task_id)
                n += 1
            self.run_total = max(self.run_total, self.run_done + len(self._queue) + len(self._running_ids))
        if n:
            self._emit("INFO", f"已加入 {n} 个分集任务到队列（并发 {self.settings.clamped_concurrency()}）。")
            self._ensure_workers()
        self._changed()
        return n

    def stop(self) -> None:
        request_stop()
        with self._lock:
            self._queue.clear()
        self._emit("INFO", "已请求停止当前任务，正在等待当前步骤安全退出…")
        self._changed()

    def _ensure_workers(self) -> None:
        clear_stop()
        workers = self.settings.clamped_concurrency()
        with self._lock:
            if self._executor is None:
                # Extra workers so a job waiting on the GPU lock can still let
                # CPU preprocess of episode N+1 run (default still 1 GPU job).
                self._executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY + 2, thread_name_prefix="recast-job")
        # Pump GPU slots + one extra CPU-only start
        for _ in range(workers + 1):
            self._pump()

    def _pump(self) -> None:
        with self._lock:
            if stopping():
                return
            workers = self.settings.clamped_concurrency()  # GPU jobs
            if not self._queue:
                return
            # Default: at most `workers` jobs may hold the GPU. CPU preprocess of
            # the next episode can start while GPU is busy.
            gpu_busy = len(self._gpu_running) >= workers
            pick = None
            for job in list(self._queue):
                from recast.core.runner import split_resource_stages

                pre, gpu, post = split_resource_stages(job.stages)
                needs_gpu_now = bool(gpu) and not pre
                # If GPU is saturated, only start jobs that still have CPU-pre work
                # (they will wait on the GPU semaphore later).
                if gpu_busy and needs_gpu_now and not pre:
                    continue
                if job.task_id in self._running_ids:
                    continue
                pick = job
                break
            if pick is None:
                return
            self._queue.remove(pick)
            self._running_ids.add(pick.task_id)
            self._running_jobs[pick.task_id] = pick
            assert self._executor is not None
            fut = self._executor.submit(self._run_job, pick)
            self._futures = [item for item in self._futures if not item.done()]
            self._futures.append(fut)
            fut.add_done_callback(lambda _f: self._after())

    def _after(self) -> None:
        self.reap_futures()
        if not stopping():
            self._pump()
        with self._lock:
            idle = not self._queue and not self._running_ids
        if idle:
            self._emit("INFO", "队列空闲。")
            self.current_task_id = None
            self.current_drama_id = None
        self._changed()

    def _run_job(self, job: Job) -> None:
        task = self.store.get("tasks", job.task_id)
        drama = self.store.get("dramas", job.drama_id)
        if not task or not drama:
            with self._lock:
                self._running_ids.discard(job.task_id)
                self._running_jobs.pop(job.task_id, None)
                self.run_done += 1
            return
        self.current_task_id = job.task_id
        self.current_drama_id = job.drama_id
        trans = apply(task.get("status") or ST_PENDING, "start")
        self.store.update("tasks", job.task_id, status=trans.state, progress=trans.progress, error=None, progress_pct=0, retry_count=0)
        self._rollup_drama(job.drama_id)
        self._emit("INFO", f"开始处理「{drama.get('new_name') or drama.get('origin_name')} / {task.get('episode_name')}」", job.task_id)

        output_root = Path(self.settings.output_root or Path.cwd() / "outputs")
        workdir = Path(task["workdir"]) if task.get("workdir") else (output_root / "workbench" / job.drama_id / Path(task["video_path"]).stem)
        workdir.mkdir(parents=True, exist_ok=True)
        self.store.update("tasks", job.task_id, workdir=str(workdir))
        self.store.update("dramas", job.drama_id, output_dir=str(workdir.parent))

        def on_progress(stage: str, message: str, fraction: float) -> None:
            self.current_pct = fraction
            self.current_progress = f"{stage} · {message}"
            now = time.monotonic()
            if not should_persist_progress(self._last_progress_at, now, fraction):
                return
            self._last_progress_at = now
            self.store.update(
                "tasks",
                job.task_id,
                status=ST_RUNNING,
                progress=self.current_progress,
                progress_pct=fraction,
            )
            self._rollup_drama(job.drama_id, running=True)
            self._changed()

        retry_count = 0
        last_exc: BaseException | None = None
        while True:
            try:
                summary = self._dispatch(job, task, workdir, on_progress)

                last = apply(ST_RUNNING, "all_done", reason="全部完成")
                self.store.update(
                    "tasks",
                    job.task_id,
                    status=last.state,
                    progress=last.progress,
                    progress_pct=1.0,
                    output_file=summary.get("output"),
                    error=None,
                    retry_count=0,
                )
                self._emit("INFO", f"分集完成: {summary.get('output')}", job.task_id)
                last_exc = None
                break
            except CancelledError:
                last = apply(ST_RUNNING, "stop")
                self.store.update("tasks", job.task_id, status=last.state, progress=last.progress, error=None)
                self._emit("INFO", f"任务已停止，产物保留，可继续。状态已回到「{ST_PENDING}」。", job.task_id)
                last_exc = None
                break
            except (PermissionError, RuntimeError) as exc:
                text = str(exc)
                # Missing named materials / unauthorized character → 待补资料, never 失败
                if isinstance(exc, PermissionError) or text.startswith(ST_NEED_ASSETS) or "缺少已授权" in text or text.startswith("缺少 "):
                    msg = text
                    if ST_NEED_ASSETS in msg and "：" in msg:
                        msg = msg.split("：", 1)[-1]
                    if not msg.startswith("缺少"):
                        msg = "缺少已授权角色的参考脸（" + msg + "）"
                    self.store.update("tasks", job.task_id, status=ST_NEED_ASSETS, progress=msg, error=msg)
                    self._emit("WARNING", msg, job.task_id)
                    last_exc = None
                    break
                if not isinstance(exc, RuntimeError):
                    raise
                # other RuntimeError falls through to generic handler via re-raise as Exception path
                last_exc = exc
                kind, err_text = classify_and_message(exc)
                log.exception("分集失败: %s", task.get("episode_name"))
                if kind == "fatal":
                    last = apply(ST_RUNNING, "unrecoverable", reason=err_text)
                    self.store.update("tasks", job.task_id, status=last.state, progress=last.progress, error=err_text)
                    self._emit("ERROR", f"不可恢复错误，任务失败: {err_text}", job.task_id)
                    break
                trans = apply(ST_RUNNING, "step_fail", reason=err_text, retry_count=retry_count, step="当前步骤")
                retry_count = trans.retry_count
                if trans.state == ST_REVIEW:
                    self.store.update("tasks", job.task_id, status=ST_REVIEW, progress=trans.progress, error=err_text, retry_count=retry_count)
                    self._emit("ERROR", trans.progress, job.task_id)
                    break
                self._emit("WARNING", trans.progress, job.task_id)
                self.store.update("tasks", job.task_id, progress=trans.progress, retry_count=retry_count, status=ST_RUNNING)
                continue
            except Exception as exc:
                last_exc = exc
                kind, text = classify_and_message(exc)
                log.exception("分集失败: %s", task.get("episode_name"))
                if kind == "fatal":
                    last = apply(ST_RUNNING, "unrecoverable", reason=text)
                    self.store.update("tasks", job.task_id, status=last.state, progress=last.progress, error=text)
                    self._emit("ERROR", f"不可恢复错误，任务失败: {text}", job.task_id)
                    break
                trans = apply(ST_RUNNING, "step_fail", reason=text, retry_count=retry_count, step="当前步骤")
                retry_count = trans.retry_count
                if trans.state == ST_REVIEW:
                    self.store.update("tasks", job.task_id, status=ST_REVIEW, progress=trans.progress, error=text, retry_count=retry_count)
                    self._emit("ERROR", trans.progress, job.task_id)
                    break
                self._emit("WARNING", trans.progress, job.task_id)
                self.store.update("tasks", job.task_id, progress=trans.progress, retry_count=retry_count, status=ST_RUNNING)
                continue

        self._rollup_drama(job.drama_id)
        with self._lock:
            self._running_ids.discard(job.task_id)
            self._running_jobs.pop(job.task_id, None)
            self.run_done += 1
        self._changed()


    def _dispatch(self, job: Job, task: dict[str, Any], workdir: Path, on_progress: ProgressFn) -> dict[str, Any]:
        from recast.core.runner import dispatch_job, split_resource_stages

        pre, gpu, post = split_resource_stages(job.stages)
        settings = self.settings
        common = dict(
            video=task.get("video_path") or None,
            workdir=workdir,
            config=self._cfg(),
            face=job.face,
            character_id=job.character_id,
            character_ids=job.character_ids,
            track_id=job.track_id,
            names=job.names,
            lang=settings.lang,
            burn_subs=settings.burn_subs,
            require_auth=job.require_auth,
            with_pose=settings.with_pose,
            store=self.store,
            task_id=job.task_id,
            drama_key=job.drama_id,
            progress=on_progress,
            output=workdir / "output.mp4",
            entry_type=job.entry_type or task.get("entry_type") or "import",
            pack_root=job.pack_root or task.get("pack_snapshot") or task.get("template_pack_id") or (self.store.get("dramas", job.drama_id) or {}).get("pack_root"),
            storyboard=job.storyboard or task.get("source_storyboard"),
            dialogue=job.dialogue or task.get("source_dialogue"),
            wav_table=job.wav_table,
            comfyui_url=job.comfyui_url or getattr(settings, "comfyui_url", "") or None,
            max_cps=job.max_cps or getattr(settings, "max_cps", 4.0),
            rerun_shot=job.rerun_shot,
            settings=settings,
        )
        # Honour basic-tier caps: no heavy generate / L2 drive
        if settings.is_basic_tier() and "generate" in gpu:
            # generate still runs, but dispatch will use animatic (comfy skipped)
            pass
        summary: dict[str, Any] = {}
        if pre:
            with self._cpu_sema:
                self._cpu_running.add(job.task_id)
                try:
                    summary.update(dispatch_job(stages=pre, skip_replace=True, **common))
                finally:
                    self._cpu_running.discard(job.task_id)
        if stopping():
            from recast.runtime import CancelledError
            raise CancelledError()
        if gpu:
            with self._gpu_sema:
                self._gpu_running.add(job.task_id)
                self._changed()
                try:
                    if stopping():
                        from recast.runtime import CancelledError
                        raise CancelledError()
                    summary.update(dispatch_job(stages=gpu, skip_replace="replace" not in gpu, **common))
                finally:
                    self._gpu_running.discard(job.task_id)
        if stopping():
            from recast.runtime import CancelledError
            raise CancelledError()
        if post:
            with self._cpu_sema:
                self._cpu_running.add(job.task_id)
                try:
                    summary.update(dispatch_job(stages=post, skip_replace=True, **common))
                finally:
                    self._cpu_running.discard(job.task_id)
        return summary

    def _rollup_drama(self, drama_id: str, *, running: bool = False) -> None:
        tasks = self.store.tasks_for_drama(drama_id)
        if not tasks:
            return
        statuses = [t.get("status") for t in tasks]
        if any(s == ST_RUNNING for s in statuses) or running:
            status = ST_RUNNING
        elif any(s == ST_FAILED for s in statuses):
            # unrecoverable on one episode → drama 失败
            status = ST_FAILED
        elif any(s == ST_REVIEW for s in statuses):
            status = ST_REVIEW
        elif any(s == ST_NEED_ASSETS for s in statuses) and all(s in {ST_NEED_ASSETS, ST_DONE} for s in statuses):
            status = ST_NEED_ASSETS
        elif all(s == ST_DONE for s in statuses):
            status = ST_DONE
        elif any(s == ST_NEED_ASSETS for s in statuses):
            # mix of pending/done/need assets
            status = ST_NEED_ASSETS if not any(s == ST_PENDING for s in statuses) else ST_PENDING
        else:
            status = ST_PENDING
        # progress text: prefer a concrete missing-file line, else current step
        progress = ""
        pct = 0.0
        if status == ST_NEED_ASSETS:
            for t in tasks:
                if t.get("status") == ST_NEED_ASSETS and t.get("progress"):
                    progress = t["progress"]
                    break
        elif status == ST_FAILED:
            for t in tasks:
                if t.get("status") == ST_FAILED:
                    progress = t.get("progress") or t.get("error") or "失败"
                    break
        elif status == ST_REVIEW:
            for t in tasks:
                if t.get("status") == ST_REVIEW:
                    progress = t.get("progress") or "待人工检查"
                    break
        elif status == ST_DONE:
            progress = "全部完成"
            pct = 1.0
        else:
            running_task = next((t for t in tasks if t.get("status") == ST_RUNNING), None)
            if running_task:
                progress = running_task.get("progress") or "处理中"
                pct = float(running_task.get("progress_pct") or 0)
            else:
                progress = "等待执行"
        outputs = [t.get("output_file") for t in tasks if t.get("output_file")]
        out_dir = tasks[0].get("workdir")
        if out_dir:
            out_dir = str(Path(out_dir).parent if len(tasks) > 1 else out_dir)
        self.store.update(
            "dramas",
            drama_id,
            status=status,
            progress=progress,
            progress_pct=pct,
            output_dir=out_dir,
            error=None if status not in {ST_FAILED} else (progress or None),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            overall = 0.0
            if self.run_total:
                overall = (self.run_done + self.current_pct) / self.run_total
            return {
                "running": bool(self._running_ids) or bool(self._queue),
                "current_task_id": self.current_task_id,
                "current_drama_id": self.current_drama_id,
                "run_total": self.run_total,
                "run_done": self.run_done,
                "current_pct": self.current_pct,
                "current_progress": self.current_progress,
                "queued": len(self._queue),
                "overall": min(1.0, overall),
            }
