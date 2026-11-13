"""UI-agnostic facade: scan, queue, character library, settings, engine status."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from recast import __version__
from recast.core.checker import check_drama, check_replace_ready
from recast.core.constants import (
    AUTH_TYPES,
    PRODUCT_NAME,
    STAGE_GROUP_IDS,
    STAGE_GROUPS,
    ST_DONE,
    ST_FAILED,
    ST_NEED_ASSETS,
    ST_PENDING,
    ST_REVIEW,
    ST_RUNNING,
)
from recast.core.engine import engine_snapshot, first_run_report
from recast.core.scanner import DramaHit, scan_root
from recast.core.scheduler import Job, Scheduler
from recast.core.settings import AppSettings, SettingsManager
from recast.core.state_machine import apply, format_missing
from recast.store.db import Store, default_db_path

log = logging.getLogger("recast")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class LogBuffer:
    def __init__(self, maxlen: int = 2000) -> None:
        self.lines: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, level: str, message: str, task_id: str | None = None) -> dict[str, Any]:
        item = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "level": level.upper(),
            "message": message,
            "task_id": task_id,
        }
        with self._lock:
            self.lines.append(item)
        return item

    def dump(self, task_id: str | None = None, level: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self.lines)
        if task_id:
            rows = [r for r in rows if r.get("task_id") == task_id]
        if level:
            rows = [r for r in rows if r.get("level") == level.upper()]
        return rows


class RecastService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.store = Store(db_path or default_db_path())
        recovered = self.store.reset_running_on_boot()
        self.settings_mgr = SettingsManager(self.store)
        self.logs = LogBuffer()
        self.scheduler = Scheduler(
            self.store,
            self.settings_mgr.current,
            on_log=self._on_log,
        )
        if recovered:
            self._on_log("INFO", f"检测到 {recovered} 条上次未完成的任务，已恢复为「{ST_PENDING}」。")

    # ----- logging / snapshot -----

    def _on_log(self, level: str, message: str, task_id: str | None = None) -> None:
        self.logs.append(level, message, task_id)

    @property
    def settings(self) -> AppSettings:
        # keep scheduler settings object in sync
        self.scheduler.settings = self.settings_mgr.current
        return self.settings_mgr.current

    def update_settings(self, **fields: Any) -> AppSettings:
        s = self.settings_mgr.update(**fields)
        self.scheduler.settings = s
        if s.ffmpeg_dir:
            from recast.ffmpeg import set_ffmpeg_dir

            set_ffmpeg_dir(s.ffmpeg_dir)
        from recast.core.engine import clear_engine_cache

        clear_engine_cache()
        return s

    def engine(self, *, light: bool = False) -> dict[str, Any]:
        return engine_snapshot(self.settings, running=self.scheduler.running, light=light)

    def _scene_run(self, task_id: str | None) -> dict[str, Any]:
        if not task_id:
            return {}
        task = self.store.get("tasks", task_id)
        work = (task or {}).get("workdir")
        if not work:
            return {}
        path = Path(work) / "shots.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            "backend": data.get("backend"),
            "backend_label": data.get("backend_label"),
            "threshold": data.get("threshold"),
            "min_shot_sec": data.get("min_shot_sec"),
        }

    def first_run(self) -> dict[str, Any]:
        return first_run_report(self.settings)

    def snapshot(self, status_filter: str | None = None, *, light_engine: bool = False) -> dict[str, Any]:
        """Full UI snapshot.

        ``light_engine=True`` lets the engine section reuse a warm cache
        past its TTL (see ``engine_snapshot(light=True)``); this is what
        the 400ms timer tick uses. All other callers (``scan()``,
        explicit refreshes, first-run report) keep the default full
        recompute.
        """
        dramas = self.store.list_dramas()
        if status_filter and status_filter not in {"", "全部"}:
            dramas = [d for d in dramas if d.get("status") == status_filter]
        sched = self.scheduler.snapshot()
        return {
            "product": PRODUCT_NAME,
            "version": __version__,
            "dramas": dramas,
            "engine": self.engine(light=light_engine),
            "scene_run": self._scene_run(sched.get("current_task_id")),
            "scheduler": sched,
            "settings": self.settings.to_dict(),
            "stages": [{"id": i, "label": l} for i, l, _ in STAGE_GROUPS],
            "works_root": self.settings.works_root,
            "guide": self.guide_state(),
        }

    # ----- scan / dramas -----

    def scan(self, root: str | Path) -> dict[str, Any]:
        hits = scan_root(root)
        if not hits:
            self._on_log("WARNING", f"在 {root} 没有找到视频。")
        for hit in hits:
            self._commit_hit(hit)
        if not self.settings.works_root:
            self.update_settings(works_root=str(Path(root).expanduser().resolve()))
        self._on_log("INFO", f"已扫描 {root}，共 {len(hits)} 部剧目。")
        return self.snapshot()

    def find_drama_for_hit(self, hit: DramaHit) -> dict[str, Any] | None:
        """Existing drama a scan hit belongs to, if any (N39 + N30 残留).

        Shared by ``_commit_hit`` and the main window's folder-rescan
        prompt — the dedup rule lives here, not in two places.
        1. First episode matches a drama's file identity (``source_path``).
        2. Otherwise, any episode already owned as a task by a drama
           (``tasks.video_path``). When episodes point at different
           dramas, the one owning the most episodes wins; ties go to the
           earliest episode in scan order.
        Returns None for a genuinely new drama. Deliberately NO
        ``root_path`` fallback: same folder, different videos are
        different dramas (N30) — a folder match alone must never merge.
        Consequence: a folder whose videos were ALL replaced is
        indistinguishable from a brand-new show in the same folder and
        still creates a new row (path-string identity has no fingerprint).
        """
        if not hit.episodes:
            return None
        source_path = str(hit.episodes[0].video_path)
        by_source = self.store.find_drama_by_source(source_path)
        if by_source is not None:
            return by_source
        counts: dict[str, int] = {}
        order: dict[str, int] = {}
        for idx, ep in enumerate(hit.episodes):
            owner = self.store.find_drama_by_task_video(str(ep.video_path))
            if owner is None:
                continue
            did = owner["id"]
            counts[did] = counts.get(did, 0) + 1
            if did not in order:
                order[did] = idx
        if not counts:
            return None
        winner = max(counts, key=lambda did: (counts[did], -order[did]))
        return self.store.get("dramas", winner)

    def _commit_hit(self, hit: DramaHit) -> dict[str, Any]:
        # N30/N39: drama identity is the FIRST episode's video file, not the
        # folder. A folder scan and a single-file import of its first
        # episode therefore land on the same row; a second video from the
        # same folder gets its own row instead of silently replacing the
        # first one's episode tasks.
        # N39: the target is NOT only the source match — a single-file
        # import of a NON-first episode must merge into the drama that
        # already owns it as a task (``find_drama_for_hit``), not create
        # a second row for the same file. N30 residual: same for a folder
        # rescan after the first episode was deleted — the remaining
        # episodes still point at the old row via their task rows.
        episodes = [{"episode_name": ep.name, "video_path": str(ep.video_path)} for ep in hit.episodes]
        source_path = (
            str(hit.episodes[0].video_path) if hit.episodes else str(hit.root)
        )
        target = self.find_drama_for_hit(hit)
        pre_tasks = (
            self.store.tasks_for_drama(target["id"]) if target else []
        )
        if target is not None:
            # Merge into the existing row (never upsert by source_path here:
            # the target may have been found via a task video while the
            # hit's first episode is a different file, and upsert would
            # create a second row). Keep the old identity while it still
            # names an episode of the merged set; otherwise re-point it at
            # the new first episode (e.g. the old first episode was
            # deleted from disk before this rescan).
            merged_paths = {ep["video_path"] for ep in episodes}
            for old in pre_tasks:
                old_path = str(old.get("video_path"))
                if old_path not in merged_paths and Path(old_path).is_file():
                    episodes.append(
                        {
                            "episode_name": old.get("episode_name")
                            or Path(old_path).stem,
                            "video_path": old_path,
                        }
                    )
                    merged_paths.add(old_path)
            identity = (
                target.get("source_path")
                if target.get("source_path") in merged_paths
                else source_path
            )
            # Another drama could already own the new identity only when
            # it is the hit's first episode — but then rule 1 above would
            # have returned that drama as the target. Guard anyway so a
            # UNIQUE violation can never lose the merge.
            clash = self.store.find_drama_by_source(identity) if identity else None
            if clash is not None and clash["id"] != target["id"]:
                identity = target.get("source_path") or identity
            drama = self.store.update(
                "dramas",
                target["id"],
                origin_name=hit.origin_name,
                episode_count=hit.episode_count,
                root_path=str(hit.root),
                source_path=identity,
            ) or target
        else:
            drama = self.store.upsert_drama_by_path(
                str(hit.root),
                origin_name=hit.origin_name,
                episode_count=hit.episode_count,
                source_path=source_path,
            )
        self.store.replace_tasks_for_drama(drama["id"], episodes)
        # material check (named files); extract can still run later
        result = check_drama(hit, require_faces=bool(hit.roster.entries))
        if hit.roster.entries and not result.ok:
            trans = apply(drama.get("status") or ST_PENDING, "materials_missing", missing=result.missing_names)
            self.store.update(
                "dramas",
                drama["id"],
                status=trans.state,
                progress=trans.progress,
                missing_files=result.missing_names,
            )
            for t in self.store.tasks_for_drama(drama["id"]):
                if t.get("status") in {ST_PENDING, ST_NEED_ASSETS}:
                    self.store.update("tasks", t["id"], status=ST_NEED_ASSETS, progress=trans.progress)
        else:
            # don't downgrade a completed/review/failed drama just because materials exist
            if drama.get("status") == ST_NEED_ASSETS:
                self.store.update("dramas", drama["id"], status=ST_PENDING, progress="资料已补齐，等待执行", missing_files=[])
        return self.store.get("dramas", drama["id"])  # type: ignore[return-value]

    def recheck_materials(self, drama_ids: list[str] | None = None) -> dict[str, Any]:
        ids = drama_ids or [d["id"] for d in self.store.list_dramas()]
        for did in ids:
            drama = self.store.get("dramas", did)
            if not drama:
                continue
            hits = scan_root(drama["root_path"])
            hit = next((h for h in hits if str(h.root) == drama["root_path"]), None)
            if hit is None and hits:
                hit = hits[0]
            if hit is None:
                continue
            # refresh episodes then recheck
            self._commit_hit(hit)
        self._on_log("INFO", "已重新检查资料。")
        return self.snapshot()

    def rename_drama(self, drama_id: str, new_name: str) -> dict[str, Any] | None:
        return self.store.update("dramas", drama_id, new_name=new_name.strip() or None)

    def set_selected(self, drama_id: str, selected: bool) -> None:
        self.store.update("dramas", drama_id, selected=1 if selected else 0)
        # Unchecking must not delete output files — we only flip the flag.

    def delete_dramas(self, drama_ids: list[str]) -> dict[str, Any]:
        for did in drama_ids:
            drama = self.store.get("dramas", did)
            if not drama:
                continue
            if drama.get("status") == ST_RUNNING:
                self._on_log("WARNING", f"「{drama.get('new_name') or drama.get('origin_name')}」正在处理中，已跳过删除。")
                continue
            self.store.delete_drama(did)
            self._on_log("INFO", f"已从队列删除「{drama.get('new_name') or drama.get('origin_name')}」（成片文件保留）。")
        return self.snapshot()

    def clear_completed(self) -> dict[str, Any]:
        n = self.store.clear_completed_dramas()
        self._on_log("INFO", f"已从队列清除 {n} 部已完成剧目（成片文件保留）。")
        return self.snapshot()

    def open_output(self, drama_id: str) -> Path:
        drama = self.store.get("dramas", drama_id)
        if not drama:
            raise FileNotFoundError("找不到该剧目。")
        path = drama.get("output_dir") or self.settings.output_root
        if not path:
            raise FileNotFoundError("还没有成片目录。")
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"找不到路径: {target}")
        open_folder(target)
        return target

    # ----- start / stop -----

    def start(
        self,
        drama_ids: list[str],
        stages: list[str] | None = None,
        *,
        character_id: str | None = None,
        include_done: bool = False,
    ) -> dict[str, Any]:
        if not drama_ids:
            raise ValueError("请先勾选要执行的剧目。")
        stages = list(stages or STAGE_GROUP_IDS)
        jobs: list[Job] = []
        parked = 0
        for did in drama_ids:
            drama = self.store.get("dramas", did)
            if not drama:
                continue
            if drama.get("status") == ST_RUNNING:
                continue
            if drama.get("status") == ST_DONE and not include_done:
                continue
            tasks = self.store.tasks_for_drama(did)
            wants_replace = "replace" in stages
            only_replace = stages == ["replace"] or (wants_replace and not any(s in stages for s in ("scan", "extract", "library")))
            if wants_replace:
                ready = check_replace_ready(
                    authorized_faces=[
                        a["path"]
                        for ch in self.store.list_characters()
                        if ch.get("authorized") and ch.get("auth_type")
                        for a in ch.get("assets") or []
                        if a.get("kind") == "face" and a.get("path")
                    ],
                    declared_missing=_missing_list(drama),
                )
                if not ready.ok and only_replace:
                    msg = ready.message or format_missing(ready.missing_names)
                    self.store.update("dramas", did, status=ST_NEED_ASSETS, progress=msg)
                    for t in tasks:
                        if t.get("status") != ST_DONE:
                            self.store.update("tasks", t["id"], status=ST_NEED_ASSETS, progress=msg)
                    parked += 1
                    continue
                if not ready.ok:
                    # run leading stages, leave replace for later
                    lead = [s for s in stages if s != "replace"]
                    if not lead:
                        parked += 1
                        continue
                    self._on_log("INFO", f"「{drama.get('new_name') or drama.get('origin_name')}」{ready.message}，先跑前置阶段。")
                    stages_for = lead
                else:
                    stages_for = stages
            else:
                stages_for = stages
            # One replace pass covers every authorized reference face.
            # An explicit character_id still limits the job to that person.
            cid = character_id
            cids: list[str] | None = None
            if "replace" in stages_for and not cid:
                cids = _authorized_swap_ids(self.store)
            for t in tasks:
                if t.get("status") == ST_RUNNING:
                    continue
                if t.get("status") == ST_DONE and not include_done:
                    continue
                drama_pack = drama.get("pack_root") or drama.get("template_pack_id")
                entry = t.get("entry_type") or drama.get("entry_type") or "import"
                jobs.append(
                    Job(
                        drama_id=did,
                        task_id=t["id"],
                        stages=list(stages_for),
                        character_id=cid,
                        character_ids=cids,
                        names=None,
                        require_auth=True,
                        entry_type=entry,
                        pack_root=t.get("pack_snapshot") or t.get("template_pack_id") or drama_pack,
                        storyboard=t.get("source_storyboard"),
                        dialogue=t.get("source_dialogue"),
                        comfyui_url=self.settings.comfyui_url or None,
                        max_cps=getattr(self.settings, "max_cps", 4.0),
                    )
                )
                # N11：建 Job 的同时把本次快照写进分集任务行——选的人物
                # （只在替换阶段有）与实际排队的阶段清单。没有选人时写
                # 空列表，不写脏数据；写失败只记日志，绝不能拦住一次启动。
                try:
                    selected = (
                        _selected_character_names(self.store, cid, cids)
                        if "replace" in stages_for
                        else []
                    )
                    self.store.update(
                        "tasks",
                        t["id"],
                        selected_characters=selected,
                        stage_list=list(stages_for),
                    )
                except Exception as exc:  # noqa: BLE001 - snapshot only
                    self._on_log("WARNING", f"任务 {t['id']} 快照字段写入失败：{exc}")
        queued = self.scheduler.enqueue(jobs)
        if queued == 0 and parked:
            self._on_log("WARNING", "勾选的任务因缺少已授权参考脸已标记为「待补资料」。")
            return self.snapshot()
        if queued == 0:
            raise ValueError("没有可执行的任务。请勾选待执行、失败、待补资料或待人工检查的项目。")
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        self.scheduler.stop()
        return self.snapshot()

    def rerun_task(self, task_id: str) -> dict[str, Any]:
        """Re-run one history task: insert the re-run row, start it.

        Builds the re-run dict with ``build_rerun_task`` (same config,
        fresh id, status reset to 待执行), inserts it with
        ``store.insert("tasks", **rerun)``, then starts it through the
        scheduler with the same Job construction as 开始执行 (default full
        stage list ``STAGE_GROUP_IDS`` — the original task's stage list is
        not stored, so a re-run is a full re-run from its checkpoints;
        already-done stages are skipped by the runner). The replace-auth
        gate is the same as in ``start``: without an authorized reference
        face the replace stage is dropped (or the task is parked as
        待补资料 when replace is the only stage — unreachable with the
        default stage list, kept for parity with ``start``).

        Returns the inserted task row plus ``started`` (bool), ``reason``
        (Chinese message when not started, else None) and ``stages`` (the
        stage list actually queued). Raises ValueError when the task or
        its drama does not exist.
        """
        from recast.core.task_history import build_rerun_task

        rerun = build_rerun_task(self.store, task_id)  # ValueError when missing
        drama_id = rerun.get("drama_id") or ""
        drama = self.store.get("dramas", drama_id) if drama_id else None
        if drama is None:
            raise ValueError(f"复跑的任务没有所属剧目，无法启动：{task_id}")
        row = self.store.insert("tasks", **rerun)
        new_id = row["id"]
        stages = list(STAGE_GROUP_IDS)
        entry = rerun.get("entry_type") or drama.get("entry_type") or "import"
        pack_root = (
            rerun.get("pack_snapshot")
            or rerun.get("template_pack_id")
            or drama.get("pack_root")
            or drama.get("template_pack_id")
        )
        stages_for = stages
        if "replace" in stages:
            ready = check_replace_ready(
                authorized_faces=[
                    a["path"]
                    for ch in self.store.list_characters()
                    if ch.get("authorized") and ch.get("auth_type")
                    for a in ch.get("assets") or []
                    if a.get("kind") == "face" and a.get("path")
                ],
                declared_missing=_missing_list(drama),
            )
            if not ready.ok:
                only_replace = stages == ["replace"] or not any(
                    s in stages for s in ("scan", "extract", "library")
                )
                if only_replace:
                    msg = ready.message or format_missing(ready.missing_names)
                    self.store.update("tasks", new_id, status=ST_NEED_ASSETS, progress=msg)
                    self._on_log("WARNING", f"复跑任务已标记为「{ST_NEED_ASSETS}」：{msg}")
                    return {**row, "started": False, "reason": msg, "stages": []}
                stages_for = [s for s in stages if s != "replace"]
                self._on_log("INFO", f"复跑任务{ready.message}，先跑前置阶段。")
        # N11：建 Job 的同时用本次实际排队的阶段与选人刷新快照（复跑行
        # 继承了原任务的旧快照，不刷新就会把上一次的记录当成这一次的）。
        # 替换被 gate 拿掉时选人写空列表；写失败只记日志，不拦复跑。
        try:
            selected = (
                _selected_character_names(
                    self.store,
                    None,
                    _authorized_swap_ids(self.store),
                )
                if "replace" in stages_for
                else []
            )
            self.store.update(
                "tasks",
                new_id,
                selected_characters=selected,
                stage_list=list(stages_for),
            )
        except Exception as exc:  # noqa: BLE001 - snapshot only
            self._on_log("WARNING", f"复跑任务 {new_id} 快照字段写入失败：{exc}")
        job = Job(
            drama_id=drama_id,
            task_id=new_id,
            stages=list(stages_for),
            character_id=None,
            character_ids=_authorized_swap_ids(self.store) if "replace" in stages_for else None,
            names=None,
            require_auth=True,
            entry_type=entry,
            pack_root=pack_root,
            storyboard=rerun.get("source_storyboard"),
            dialogue=rerun.get("source_dialogue"),
            comfyui_url=self.settings.comfyui_url or None,
            max_cps=getattr(self.settings, "max_cps", 4.0),
        )
        queued = self.scheduler.enqueue([job])
        if queued == 0:
            reason = "复跑任务未能排队（调度器未接受）。"
            self._on_log("WARNING", f"复跑任务 {new_id} {reason}")
            return {**row, "started": False, "reason": reason, "stages": list(stages_for)}
        self._on_log("INFO", f"已复跑任务 {new_id}（原任务 {task_id}），已排队执行。")
        return {**row, "started": True, "reason": None, "stages": list(stages_for)}

    # ----- character library -----

    def characters(self) -> list[dict[str, Any]]:
        rows = self.store.list_characters()
        for ch in rows:
            ch["asset_flags"] = asset_flags(ch)
        return rows

    def confirm_character(
        self,
        character_id: str,
        *,
        display_name: str | None = None,
        authorized: bool | None = None,
        auth_type: str | None = None,
        face_path: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any] | None:
        ch = self.store.get_character(character_id)
        if not ch:
            return None
        # N20：改名前先记下旧显示名——改名后它要进人物 meta.json 的
        # aliases（旧台词仍用旧名说话，音色联动靠别名找到它们）。
        old_name = (ch.get("display_name") or ch.get("name") or "").strip()
        fields: dict[str, Any] = {"confirmed": 1}
        if display_name is not None:
            fields["display_name"] = display_name.strip()
            fields["name"] = display_name.strip()
        if authorized is not None:
            fields["authorized"] = 1 if authorized else 0
            if authorized and not (auth_type or ch.get("auth_type")):
                raise ValueError("勾选授权时请选择授权类型：自有拍摄 / 已授权 / 自有生成角色。")
        if auth_type is not None:
            if auth_type and auth_type not in AUTH_TYPES:
                raise ValueError(f"未知授权类型: {auth_type}")
            fields["auth_type"] = auth_type
            if not auth_type:
                fields["authorized"] = 0
        if notes is not None:
            fields["notes"] = notes
        self.store.update("characters", character_id, **fields)
        if face_path:
            if not Path(face_path).is_file():
                raise FileNotFoundError(f"找不到参考脸: {face_path}")
            self.store.set_character_asset(character_id, "face", face_path, complete=True)
        # N20：真改名了（旧名非空且与新名不同）才记别名。人物文件夹不随
        # 改名移动，所以别名写进旧名对应的文件夹（add_character_alias
        # 会按旧名找文件夹），同时把该文件夹 meta 的 name 同步成新名，
        # 这样以后按新名找文件夹还能通过 meta 认出它。别名记账绝不能
        # 搞坏一次改名：任何异常只记日志，不抛回调用方。
        if (
            display_name is not None
            and old_name
            and old_name != display_name.strip()
        ):
            try:
                self.add_character_alias(
                    character_id, old_name, folder_name=old_name
                )
                folder = self._character_meta_folder(
                    character_id, folder_name=old_name
                )
                if folder is not None:
                    from recast.core.characters import load_meta, save_meta

                    meta = load_meta(folder) or {}
                    if str(meta.get("name") or "").strip() != display_name.strip():
                        meta["name"] = display_name.strip()
                        save_meta(folder, meta)
            except Exception as exc:  # noqa: BLE001 - bookkeeping only
                self._on_log("WARNING", f"人物「{old_name}」改名别名记录失败：{exc}")
        return self.store.get_character(character_id)

    def add_character_manual(self, name: str, face_path: str | None = None) -> dict[str, Any]:
        row = self.store.insert(
            "characters",
            name=name,
            display_name=name,
            confirmed=1 if name else 0,
            authorized=0,
            cluster_key=f"manual:{name}",
        )
        if face_path:
            self.store.set_character_asset(row["id"], "face", face_path, complete=True)
        return self.store.get_character(row["id"])  # type: ignore[return-value]

    # ----- manual review queue + correction (step 3: 人物审核确认) -----

    def _review_min_score(self) -> float:
        try:
            cfg = RecastConfig.from_yaml(self.settings.config_path)
            return float(cfg.identity.review_min_score)
        except Exception:
            from recast.core.review import DEFAULT_REVIEW_MIN_SCORE

            return DEFAULT_REVIEW_MIN_SCORE

    def list_review_queue(
        self, min_score: float | None = None, *, include_rejected: bool = False
    ) -> list[dict[str, Any]]:
        """Manual review queue (see recast/core/review.py). Empty = step 3 done.

        ``include_rejected`` passes through to build_review_queue: False
        (default) hides extraction-held-back samples, True shows them.
        """
        from recast.core.review import build_review_queue

        items = build_review_queue(
            self.store,
            self.settings.output_root,
            min_score=self._review_min_score() if min_score is None else min_score,
            include_rejected=include_rejected,
        )
        return [item.to_dict() for item in items]

    def guide_state(self) -> list[dict[str, Any]]:
        """Five-step guide state (see recast/core/guide.py)."""
        from recast.core.guide import get_guide_state

        return [step.to_dict() for step in get_guide_state(self)]

    def replace_locked(self) -> tuple[bool, str]:
        """Whether replace/generate must stay locked, and why."""
        from recast.core.guide import replace_locked as _locked

        return _locked(self)

    def unauthorized_replace_characters(
        self, drama_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Characters with a reference face but no authorization (see N13).

        These are the people the replace stage would skip: they have a
        face asset in the library, but no authorization type is set, so
        ``check_replace_ready`` (used by ``start`` / ``rerun_task``)
        cannot use them. The one-click review gate calls this before
        phase B so the user is told up front instead of finding out at
        the replace stage, when the task would be parked as 待补资料.

        N40：``drama_ids`` 给出时只返回属于所选剧目的人物——映射
        见 ``Store.character_in_scope``（``source_task_id`` 或
        ``selected_characters`` 快照链到的剧目；两处都链不到的
        人物算归属未知，照样返回，宁可多弹一次关口也不静默漏
        掉）。``drama_ids`` 为空（None/[]）时保持老行为：返回全
        库未授权人物。只链到别的剧目的人物不再出现在这里，关口
        也就不会再为他们弹框、记账。
        """
        out: list[dict[str, Any]] = []
        for ch in self.store.list_characters():
            if ch.get("authorized") and ch.get("auth_type"):
                continue
            faces = [
                a
                for a in (ch.get("assets") or [])
                if a.get("kind") == "face"
                and a.get("path")
                and Path(str(a["path"])).is_file()
            ]
            if not faces:
                continue
            if drama_ids:
                in_scope = getattr(self.store, "character_in_scope", None)
                try:
                    if callable(in_scope) and not in_scope(ch, drama_ids):
                        continue
                except Exception:
                    pass
            out.append(
                {
                    "id": ch.get("id"),
                    "name": ch.get("display_name") or ch.get("name") or ch.get("id"),
                    "face_paths": [str(a.get("path")) for a in faces],
                }
            )
        return out

    def run_body_rerender(
        self,
        clip_path: str | Path,
        glb_path: str | Path,
        out_dir: str | Path | None = None,
        *,
        blender_path: str | Path | None = None,
        render_width: int = 360,
        render_height: int = 640,
        render_samples: int = 16,
    ) -> dict[str, Any]:
        """Body re-render pilot chain for one short single-person clip.

        Pose → retarget → Blender render → composite. The pilot gate
        (<=5s, exactly one person) is enforced by
        ``recast.stages.rerender.validate_pilot_clip`` and raises
        ``PilotRejected`` for longer or multi-person clips.

        If no Blender binary is found (``blender_path``, ``blender`` on
        PATH, or the Windows default install path), the pose + retarget
        artifacts and a render-job file are still written to ``out_dir``
        and a RuntimeError is raised telling the user to run the render
        step on Windows/Blender — never a fake frame.
        """
        from recast.stages.rerender import (
            composite_clip,
            estimate_pilot_pose,
            retarget_clip,
        )

        clip = Path(clip_path)
        glb = Path(glb_path)
        out = Path(out_dir) if out_dir else clip.parent / f"{clip.stem}-rerender"
        out.mkdir(parents=True, exist_ok=True)
        self.logs.append("INFO", f"身体重渲染试点：{clip.name} + {glb.name}")

        # Duration gate is enforced inside estimate_pilot_pose (raises
        # PilotRejected for >5s). The single-person precondition is NOT
        # auto-checked: HOG was tested on the pilot and is useless here
        # (0 detections on a frame with a clear person, 4 false positives
        # on another), and YOLO needs ultralytics (not installed). The
        # operator confirms single-person by viewing the clip first.
        pose = estimate_pilot_pose(clip, out)
        retarget = retarget_clip(pose, glb, out)

        blender = self._find_blender(blender_path)
        render_dir = out / "render"
        if blender is None:
            job = {
                "clip": str(clip),
                "glb": str(glb),
                "pose": str(out / "pose.json"),
                "retarget": str(out / "retarget.json"),
                "render_dir": str(render_dir),
                "command": (
                    "blender -b --python scripts/rerender_blender.py -- "
                    f"--glb {glb} --retarget {out / 'retarget.json'} "
                    f"--pose {out / 'pose.json'} --frames 0-{pose['frame_count'] - 1} "
                    f"--out-dir {render_dir} --width {render_width} "
                    f"--height {render_height} --samples {render_samples}"
                ),
            }
            (out / "render-job.json").write_text(
                json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.logs.append(
                "WARNING",
                "本机未找到 Blender：姿态 + 重定向已写好，渲染步需在 Windows/Blender 上跑（见 render-job.json）。",
            )
            raise RuntimeError(
                "本机未找到 Blender，无法渲染。姿态估计和重定向已完成并保存在 "
                f"{out}，按 render-job.json 里的命令在 Windows/Blender 上跑渲染，"
                "再用合成步出片。"
            )

        script = Path(__file__).resolve().parents[2] / "scripts" / "rerender_blender.py"
        cmd = [
            str(blender),
            "-b",
            "--python",
            str(script),
            "--",
            "--glb",
            str(glb),
            "--retarget",
            str(out / "retarget.json"),
            "--pose",
            str(out / "pose.json"),
            "--frames",
            f"0-{pose['frame_count'] - 1}",
            "--out-dir",
            str(render_dir),
            "--width",
            str(render_width),
            "--height",
            str(render_height),
            "--samples",
            str(render_samples),
        ]
        if sys.platform.startswith("linux") and shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] + cmd
        self.logs.append("INFO", f"渲染：{' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"Blender 渲染失败（exit {proc.returncode}）：{proc.stderr[-2000:]}")
        result = composite_clip(clip, render_dir, pose, out / f"{clip.stem}-rerendered.mp4")
        result.update({"pose": pose, "retarget": retarget, "out_dir": str(out)})
        self.logs.append("INFO", f"身体重渲染完成：{result['out_path']}")
        return result

    @staticmethod
    def _find_blender(blender_path: str | Path | None) -> str | None:
        """Locate a Blender binary, or None (render step then defers)."""
        if blender_path:
            p = Path(blender_path)
            if p.is_file():
                return str(p)
        found = shutil.which("blender")
        if found:
            return found
        for candidate in (
            Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
        return None


    def _sample_owner_id(self, sample_path: str) -> str | None:
        """Which character owns a face sample file (store asset rows)."""
        for ch in self.store.list_characters():
            for a in ch.get("assets") or []:
                if a.get("kind") == "face" and a.get("path") in (sample_path, str(Path(sample_path).resolve())):
                    return ch["id"]
        return None

    def _check_sample_movable(self, sample_path: str | Path) -> Path:
        """Validate a face sample file can be moved (correction safety).

        Refuses symlinks, missing files, and anything outside the works
        root / output root — the same "inside the project" rule as the
        costume binding check. Returns the resolved path.
        """
        from recast.core.characters import _is_within as _within

        src = Path(str(sample_path)).expanduser()
        if src.is_symlink():
            raise ValueError(f"纠错拒绝符号链接：{sample_path}")
        try:
            resolved = src.resolve()
        except OSError as exc:
            raise ValueError(f"纠错解析失败：{sample_path}（{exc}）") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"找不到人脸样本：{sample_path}")
        roots = [
            Path(str(p)).expanduser().resolve()
            for p in (self.settings.works_root, self.settings.output_root)
            if p
        ]
        if not any(_within(resolved, root) for root in roots):
            raise ValueError(f"人脸样本必须在作品目录或输出目录里面，文件夹外的文件不接受：{sample_path}")
        return resolved

    def _face_folder_for(self, character_id: str) -> Path:
        from recast.core.characters import ensure_character_folder

        ch = self.store.get_character(character_id)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character_id
        folder, _meta = ensure_character_folder(self.settings.output_root, str(name))
        return folder

    def _unique_dest(self, folder: Path, name: str) -> Path:
        dest = folder / name
        if not dest.exists():
            return dest
        stem, suffix = dest.stem, dest.suffix
        n = 1
        while (folder / f"{stem}_{n}{suffix}").exists():
            n += 1
        return folder / f"{stem}_{n}{suffix}"

    def move_face_sample(
        self, character_id: str, sample_path: str, target_character_id: str
    ) -> dict[str, Any]:
        """Move one face sample file to another character (correction).

        The file moves into the target's ``face/`` folder; the store asset
        rows and both folders' meta.json are updated; if the source's
        assembly face pointed at this file it is cleared. Returns the new
        path. Raises when the file is a symlink, missing, or outside the
        project folders.
        """
        from recast.core.characters import (
            add_face_file,
            character_folder,
            get_assembly,
            load_meta,
            remove_face_file,
            set_assembly,
        )

        if target_character_id == character_id:
            raise ValueError("移到另一人物：目标人物和原来是同一个。")
        target = self.store.get_character(target_character_id)
        if not target:
            raise ValueError(f"找不到目标人物：{target_character_id}")
        resolved = self._check_sample_movable(sample_path)
        target_folder = self._face_folder_for(target_character_id)
        dest = self._unique_dest(target_folder / "face", resolved.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(dest))
        for key in (str(sample_path), str(resolved)):
            for row in self.store.list_rows(
                "character_assets", "character_id = ? AND kind = ? AND path = ?", (character_id, "face", key)
            ):
                self.store.delete_asset_rows([row["id"]])
        self.store.insert(
            "character_assets",
            character_id=target_character_id,
            kind="face",
            path=str(dest),
            complete=1,
            notes="纠错移入；已审核",
        )
        add_face_file(target_folder, str(dest))
        source = self.store.get_character(character_id)
        if source:
            name = source.get("display_name") or source.get("name") or character_id
            src_folder = character_folder(self.settings.output_root, str(name))
            if src_folder.is_dir():
                remove_face_file(src_folder, str(sample_path))
                remove_face_file(src_folder, str(resolved))
                meta = load_meta(src_folder)
                if meta and get_assembly(meta).get("face"):
                    from recast.core.characters import resolve_assembly_path

                    if resolve_assembly_path(src_folder, get_assembly(meta)["face"]) in (
                        resolved,
                        Path(str(sample_path)),
                    ):
                        set_assembly(src_folder, face="")
        self._on_log("INFO", f"人脸样本已从「{source.get('display_name') if source else character_id}」移到「{target.get('display_name') or target_character_id}」：{dest.name}")
        return {"path": str(dest), "character_id": target_character_id}

    def remove_face_sample(
        self, character_id: str, sample_path: str, *, discarded: bool = False
    ) -> dict[str, Any]:
        """Take one face sample out of a character (correction).

        The file is NOT deleted: it moves to ``<output_root>/review_trash/``
        and its asset row is removed, so a wrong move can be undone. A
        correction move (``discarded=False``) shows up in the review queue
        again from the trash folder; a discard (``discarded=True``) goes to
        ``review_trash/discarded/`` and leaves the queue. Returns the trash path.
        """
        from recast.core.characters import (
            character_folder,
            get_assembly,
            load_meta,
            remove_face_file,
            resolve_assembly_path,
            set_assembly,
        )
        from recast.core.review import review_trash_path

        resolved = self._check_sample_movable(sample_path)
        dest = review_trash_path(self.settings.output_root, resolved.name, discarded=discarded)
        shutil.move(str(resolved), str(dest))
        for key in (str(sample_path), str(resolved)):
            for row in self.store.list_rows(
                "character_assets", "character_id = ? AND kind = ? AND path = ?", (character_id, "face", key)
            ):
                self.store.delete_asset_rows([row["id"]])
        source = self.store.get_character(character_id)
        if source:
            name = source.get("display_name") or source.get("name") or character_id
            src_folder = character_folder(self.settings.output_root, str(name))
            if src_folder.is_dir():
                remove_face_file(src_folder, str(sample_path))
                remove_face_file(src_folder, str(resolved))
                meta = load_meta(src_folder)
                if meta and get_assembly(meta).get("face"):
                    if resolve_assembly_path(src_folder, get_assembly(meta)["face"]) in (
                        resolved,
                        Path(str(sample_path)),
                    ):
                        set_assembly(src_folder, face="")
        self._on_log(
            "INFO",
            f"人脸样本已{'丢弃到 review_trash/discarded/' if discarded else '移出到待审核'}（未删除）：{dest.name}",
        )
        return {"trash": str(dest), "character_id": character_id}

    def split_character(
        self, character_id: str, sample_paths: list[str], new_name: str
    ) -> dict[str, Any]:
        """Split the selected face samples off into a new character.

        For a character folder that mixed in a different person's face:
        pick the wrong samples, give the new person a name, and they move
        over (same rules as move_face_sample). Returns the new character.
        """
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("拆分需要给新人物起个名字。")
        if not sample_paths:
            raise ValueError("拆分需要先选至少一张人脸样本。")
        new_ch = self.add_character_manual(new_name)
        moved: list[str] = []
        for path in sample_paths:
            moved.append(self.move_face_sample(character_id, path, new_ch["id"])["path"])
        return {"character": new_ch, "moved": moved}

    def _adopt_trash_sample(self, item_key: str, target_character_id: str) -> dict[str, Any]:
        """Adopt a file waiting in review_trash/ into a character.

        Used when a corrected sample (moved out of its character for review)
        is assigned to a character from the queue, or made into a new one.
        The file moves into the target's face/ folder and gets a store asset
        row + meta.json entry. Raises when the file is not in the trash folder.
        """
        from recast.core.characters import add_face_file
        from recast.core.review import REVIEW_TRASH_DIR

        target = self.store.get_character(target_character_id)
        if not target:
            raise ValueError(f"找不到目标人物：{target_character_id}")
        trash = Path(self.settings.output_root).expanduser().resolve() / REVIEW_TRASH_DIR
        try:
            resolved = Path(item_key).resolve()
            resolved.relative_to(trash)
        except (OSError, ValueError):
            raise ValueError(f"这个样本不在待审核区：{item_key}")
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(f"找不到待审核样本：{item_key}")
        target_folder = self._face_folder_for(target_character_id)
        dest = self._unique_dest(target_folder / "face", resolved.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(dest))
        self.store.insert(
            "character_assets",
            character_id=target_character_id,
            kind="face",
            path=str(dest),
            complete=1,
            notes="待审核区认领；已审核",
        )
        add_face_file(target_folder, str(dest))
        self._on_log("INFO", f"待审核样本已认领到「{target.get('display_name') or target_character_id}」：{dest.name}")
        return {"path": str(dest), "character_id": target_character_id}

    def _mark_sample_reviewed(self, owner_id: str, sample_path: str) -> dict[str, Any]:
        """Mark one face sample as reviewed in place (no file move).

        Used when a low-score sample is confirmed as the same person as its
        owner: the file already belongs to that character, so there is
        nothing to move — only the store asset row's notes gain "已审核",
        the same marker move_face_sample / _adopt_trash_sample set, which
        is what drops the row from the review queue (recast/core/review.py)
        and what the main window counts as a confirm action (N21).
        Idempotent: confirming twice does not duplicate the marker.
        """
        try:
            resolved = str(Path(sample_path).resolve())
        except OSError:
            resolved = None
        marked: list[str] = []
        for key in (str(sample_path), resolved):
            if not key:
                continue
            for row in self.store.list_rows(
                "character_assets", "character_id = ? AND kind = ? AND path = ?", (owner_id, "face", key)
            ):
                notes = str(row.get("notes") or "")
                if "已审核" not in notes:
                    notes = f"{notes}；已审核" if notes else "已审核"
                    self.store.update("character_assets", row["id"], notes=notes)
                marked.append(str(row.get("path") or key))
        if not marked:
            raise ValueError(f"找不到这个人脸样本的资产行：{sample_path}")
        self._on_log("INFO", f"人脸样本已确认为同一人（已审核）：{Path(marked[0]).name}")
        return {"path": marked[0], "character_id": owner_id}

    def review_assign(self, item_key: str, character_id: str, *, owner_id: str | None = None) -> dict[str, Any]:
        """Handle one review row: assign it to an existing character.

        - Cluster row (key is a character id): merge the whole cluster into
          the target character (its samples move over, then the cluster row
          is deleted). Files stay on disk; only the store rows change hands.
        - Sample row (key is a file path): move that one sample to the
          target character (see move_face_sample).
        - Sample row whose target IS its owner ("确认是同一人" on a
          low-score sample): nothing to move — mark the asset row
          "已审核" in place (see _mark_sample_reviewed). Moving it to
          itself used to raise ValueError from move_face_sample and left
          the row stuck in the queue.
        """
        target = self.store.get_character(character_id)
        if not target:
            raise ValueError(f"找不到目标人物：{character_id}")
        cluster = self.store.get_character(item_key)
        if cluster is not None:
            if cluster["id"] == character_id:
                raise ValueError("分给已有的人物：目标人物和这个聚类是同一个。")
            moved: list[str] = []
            for a in list(cluster.get("assets") or []):
                if a.get("kind") == "face" and a.get("path"):
                    try:
                        moved.append(self.move_face_sample(cluster["id"], str(a["path"]), character_id)["path"])
                    except (ValueError, FileNotFoundError, OSError) as exc:
                        self._on_log("WARNING", f"合并聚类时跳过一个样本：{a.get('path')}（{exc}）")
            self.store.delete_character(cluster["id"])
            return {"kind": "cluster", "merged_into": character_id, "moved": moved}
        owner = owner_id or self._sample_owner_id(item_key)
        if not owner:
            # The sample waits in review_trash/ (moved out by correction):
            # adopt the file straight into the target character.
            return {"kind": "sample", **self._adopt_trash_sample(item_key, character_id)}
        if owner == character_id:
            # Same-person confirm: the file already belongs to the target.
            return {"kind": "sample", **self._mark_sample_reviewed(owner, item_key)}
        return {"kind": "sample", **self.move_face_sample(owner, item_key, character_id)}

    def review_create_character(self, item_key: str, name: str, *, owner_id: str | None = None) -> dict[str, Any]:
        """Handle one review row: make a new character out of it.

        - Cluster row: confirm the cluster under the given name.
        - Sample row: create the character and move the sample into it.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("新建人物需要一个名字。")
        cluster = self.store.get_character(item_key)
        if cluster is not None:
            return {"kind": "cluster", "character": self.confirm_character(cluster["id"], display_name=name)}
        owner = owner_id or self._sample_owner_id(item_key)
        new_ch = self.add_character_manual(name)
        if not owner:
            # The sample waits in review_trash/: adopt it into the new character.
            self._adopt_trash_sample(item_key, new_ch["id"])
        else:
            self.move_face_sample(owner, item_key, new_ch["id"])
        return {"kind": "sample", "character": new_ch}

    def review_discard(self, item_key: str, *, owner_id: str | None = None) -> dict[str, Any]:
        """Handle one review row: discard it.

        - Cluster row: delete the unconfirmed character (files stay on disk).
        - Sample row: move the file to ``review_trash/discarded/`` (never
          deleted outright, and out of the queue for good).
        - Trash row (a corrected sample waiting in ``review_trash/``):
          move it to ``review_trash/discarded/`` as well.
        """
        cluster = self.store.get_character(item_key)
        if cluster is not None:
            self.store.delete_character(cluster["id"])
            return {"kind": "cluster", "discarded": cluster["id"]}
        owner = owner_id or self._sample_owner_id(item_key)
        if not owner:
            # Already waiting in review_trash/ (a corrected sample): discarding
            # moves it to the "discarded" subfolder, still never deleted outright.
            from recast.core.review import REVIEW_TRASH_DIR, review_trash_path

            trash = Path(self.settings.output_root).expanduser().resolve() / REVIEW_TRASH_DIR
            try:
                resolved = Path(item_key).resolve()
                resolved.relative_to(trash)
            except (OSError, ValueError):
                raise ValueError(f"找不到这个样本属于哪个人物：{item_key}")
            if not resolved.is_file():
                raise FileNotFoundError(f"找不到待审核样本：{item_key}")
            dest = review_trash_path(self.settings.output_root, resolved.name, discarded=True)
            shutil.move(str(resolved), str(dest))
            return {"kind": "sample", "trash": str(dest), "discarded": str(dest)}
        return {"kind": "sample", **self.remove_face_sample(owner, item_key, discarded=True)}

    # ----- character folders (P0: 人物库为核心) -----

    def ensure_character_folder(
        self, name: str, *, name_source: str = "manual"
    ) -> dict[str, Any]:
        """Create output/characters/<name>/ with meta.json (UI: 人物库入口)."""
        from recast.core.characters import ensure_character_folder as _ensure

        folder, meta = _ensure(self.settings.output_root, name, name_source=name_source)
        return {"name": meta.get("name") or name, "folder": str(folder), "meta": meta}

    def migrate_character_folders(
        self,
        *,
        face_samples: dict[str, list[str]] | None = None,
        workdir: str | Path | None = None,
        roster: Any | None = None,
    ) -> dict[str, Any]:
        """Migrate old-layout assets into per-character folders.

        face_samples comes from workdir/face_samples/index.json when workdir
        is given. Sources are only read, never deleted; re-running is safe.
        """
        from recast.core.characters import migrate_to_character_folders

        if face_samples is None and workdir is not None:
            index = Path(workdir) / "face_samples" / "index.json"
            if index.is_file():
                try:
                    face_samples = json.loads(index.read_text(encoding="utf-8") or "{}")
                except ValueError as exc:
                    raise ValueError(f"读不懂人脸样本索引 {index}：{exc}") from exc
        return migrate_to_character_folders(
            self.settings.output_root,
            face_samples=face_samples,
            roster=roster,
        )

    def run_matting(
        self, character: str, *, sources: list[str | Path] | None = None
    ) -> dict[str, Any]:
        """Cut one character's photos into its cutout/ folder (UI: [抠像]).

        ``character`` is a store character id or a character folder name.
        """
        from recast.config import RecastConfig
        from recast.core.characters import character_folder
        from recast.stages.matting import matte_character_folder

        cfg = RecastConfig.from_yaml(self.settings.config_path)
        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        report = matte_character_folder(
            folder,
            sources=sources,
            model=cfg.matting.model,
            models_dir=cfg.matting.models_dir or None,
        )
        self._on_log(
            "INFO",
            f"人物「{name}」抠像完成：成功 {len(report['outputs'])} 张，"
            f"失败 {len(report['failed'])} 张。",
        )
        return report

    def run_diarization(
        self,
        source: str | Path,
        *,
        character_names: list[str] | None = None,
        n_speakers: int | None = None,
    ) -> dict[str, Any]:
        """Split a video/wav track into per-speaker clips (UI: [自动定义人物] 的音频部分).

        Clips land in each character folder's voice/. Speakers map to
        ``character_names`` in order (SPK_1 -> first name, ...). Extra
        speakers are skipped and reported.
        """
        from recast.config import RecastConfig
        from recast.core.characters import ensure_character_folder
        from recast.ffmpeg import extract_audio, which_ffmpeg

        cfg = RecastConfig.from_yaml(self.settings.config_path)
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(f"找不到音视频文件：{src}")
        if src.suffix.lower() == ".wav":
            wav_path = src
        else:
            which_ffmpeg()  # clear error when ffmpeg is missing
            tmp = Path(self.settings.output_root) / "_diarization" / f"{src.stem}.wav"
            wav_path = extract_audio(src, tmp, sample_rate=cfg.diarization.sample_rate)
            if wav_path is None:
                raise ValueError(f"源视频没有音轨，无法做说话人分离：{src}")
        names = [str(n).strip() for n in (character_names or []) if str(n).strip()]
        if n_speakers is None and names:
            n_speakers = len(names)
        if n_speakers is not None:
            n_speakers = max(1, min(int(n_speakers), cfg.diarization.max_speakers))
        # Map SPK_1.. to character folders in order.
        from recast.stages.diarization import diarize

        found = diarize(
            wav_path,
            n_speakers=n_speakers,
            min_speech_sec=cfg.diarization.min_speech_sec,
            min_silence_sec=cfg.diarization.min_silence_sec,
            merge_gap_sec=cfg.diarization.merge_gap_sec,
        )
        speakers = sorted({seg["speaker"] for seg in found})
        mapping: dict[str, Path] = {}
        for speaker, display in zip(speakers, names):
            folder, _meta = ensure_character_folder(
                self.settings.output_root, display, name_source="auto"
            )
            mapping[speaker] = folder
        # Reuse the splitter with a precomputed mapping: write clips directly.
        from recast.stages.diarization import read_wav, write_wav
        from recast.core.characters import load_meta, save_meta

        samples, sr = read_wav(wav_path)
        report: dict[str, Any] = {"segments": len(found), "clips": {}, "skipped": []}
        for seg in found:
            folder = mapping.get(seg["speaker"])
            if folder is None:
                report["skipped"].append(seg)
                continue
            voice_dir = folder / "voice"
            voice_dir.mkdir(parents=True, exist_ok=True)
            clip = voice_dir / f"{seg['speaker']}_{seg['start']:.2f}_{seg['end']:.2f}.wav"
            write_wav(clip, samples[int(seg["start"] * sr):int(seg["end"] * sr)], sr)
            report["clips"].setdefault(seg["speaker"], []).append(str(clip))
            meta = load_meta(folder) or {}
            voice_meta = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
            known = list(voice_meta.get("samples", []) or [])
            if str(clip) not in known:
                known.append(str(clip))
            meta["voice"] = {**voice_meta, "samples": known}
            save_meta(folder, meta)
        self._on_log(
            "INFO",
            f"说话人分离完成：{len(found)} 段，{len(report['clips'])} 个人物有声音样本，"
            f"跳过 {len(report['skipped'])} 段（没有对应人物）。",
        )
        return {**report, "wav": str(wav_path)}

    def tag_costume(
        self,
        character: str,
        *,
        sources: list[str | Path] | None = None,
        tagger: str | None = None,
    ) -> dict[str, Any]:
        """Auto-tag one character's costume (UI: [衣服打标签] 逻辑层).

        ``character`` is a store character id or a character folder name.
        ``sources`` must be full photos/frames (face crops have no body
        region). Screenshots land in costume/, the description in meta.json.
        """
        from recast.config import RecastConfig
        from recast.core.characters import character_folder
        from recast.stages.costume import get_tagger, tag_character_folder

        cfg = RecastConfig.from_yaml(self.settings.config_path)
        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        tagger_obj = get_tagger(
            tagger or cfg.costume.tagger,
            model=cfg.costume.clip_model,
            models_dir=cfg.costume.models_dir or None,
        )
        report = tag_character_folder(folder, sources=sources, tagger=tagger_obj)
        self._on_log(
            "INFO",
            f"人物「{name}」衣服打标签完成：成功 {len(report['tagged'])} 张，"
            f"失败 {len(report['failed'])} 张。",
        )
        return report

    def suggest_names(
        self,
        *,
        workdir: str | Path | None = None,
        persons: list[dict[str, Any]] | None = None,
        tracks: list[dict[str, Any]] | None = None,
        dialogue_shots: list[dict[str, Any]] | None = None,
        ocr_hits: list[dict[str, Any]] | None = None,
        fps: float = 25.0,
        write_meta: bool = True,
    ) -> list[dict[str, Any]]:
        """Suggest names for unnamed face clusters (UI: [自动定义人物] 的名字部分).

        Loads identity.json + tracks.json from ``workdir`` when persons/tracks
        are not passed directly. Suggestions are written to each folder's
        meta.json ``name_suggestion`` for the user to confirm — nobody is
        renamed here.
        """
        from recast.stages.name_suggest import suggest_names as _suggest
        from recast.stages.name_suggest import write_name_suggestions

        if persons is None and workdir is not None:
            ident_path = Path(workdir) / "identity.json"
            if not ident_path.is_file():
                raise FileNotFoundError(f"找不到身份聚类结果：{ident_path}")
            try:
                payload = json.loads(ident_path.read_text(encoding="utf-8") or "{}")
            except ValueError as exc:
                raise ValueError(f"读不懂身份聚类结果 {ident_path}：{exc}") from exc
            persons = list(payload.get("persons") or [])
        if tracks is None and workdir is not None:
            tracks_path = Path(workdir) / "tracks.json"
            if tracks_path.is_file():
                try:
                    raw_tracks = json.loads(tracks_path.read_text(encoding="utf-8") or "{}")
                except ValueError:
                    raw_tracks = {}
                tracks = raw_tracks.get("tracks") if isinstance(raw_tracks, dict) else raw_tracks
                tracks = list(tracks or [])
        suggestions = _suggest(
            persons or [],
            tracks or [],
            dialogue_shots=dialogue_shots,
            ocr_hits=ocr_hits,
            fps=fps,
        )
        if write_meta and suggestions:
            written = write_name_suggestions(self.settings.output_root, suggestions)
            self._on_log("INFO", f"名字自动建议完成：{len(suggestions)} 条建议已写入 {len(written)} 个人物文件夹。")
        return suggestions

    def import_model_glb(self, character: str, glb_path: str | Path) -> dict[str, Any]:
        """Import a GLB file as one character's 3D model (UI: [导入 3D 模型]).

        ``character`` is a store character id or a character folder name.
        """
        from recast.assets3d.glb import import_glb
        from recast.core.characters import character_folder

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        report = import_glb(glb_path, folder)
        self._on_log("INFO", f"人物「{name}」已导入 GLB 模型：{report['path']}")
        return report

    def generate_model_3d(
        self,
        character: str,
        *,
        image: str | Path | None = None,
    ) -> dict[str, Any]:
        """Auto-generate one character's 3D model from an image (UI: [生成 3D 模型]).

        ``character`` is a store character id or a character folder name.
        ``image`` defaults to the first cutout PNG (transparent background
        helps the generator). Needs a CUDA GPU >= 8GB VRAM and a TripoSG
        checkout; see model3d.triposg_repo in the config. The generated
        GLB is registered in meta.json and ``assembly.body`` is pointed at
        it, so the assembly view can use it right away. The GLB is shape
        only (no texture, no skeleton) — it cannot be used for body
        re-rendering.
        """
        from recast.assets3d.generate3d import generate_model_glb
        from recast.config import RecastConfig
        from recast.core.characters import character_folder

        cfg = RecastConfig.from_yaml(self.settings.config_path)
        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        if image is None:
            cutout_dir = folder / "cutout"
            pngs = sorted(cutout_dir.glob("*.png")) if cutout_dir.is_dir() else []
            if not pngs:
                raise FileNotFoundError(
                    f"人物「{name}」没有抠像图片（cutout/ 为空），请先抠像再生成 3D 模型。"
                )
            image = pngs[0]
        report = generate_model_glb(
            image,
            folder,
            repo_dir=cfg.model3d.triposg_repo or None,
            python=cfg.model3d.triposg_python or None,
        )
        self._on_log("INFO", f"人物「{name}」3D 模型自动生成完成：{report['path']}")
        return report

    def check_model_generation(self) -> dict[str, Any]:
        """Pre-flight check for [生成 3D 模型] (UI: show before starting).

        Returns {"ok": bool, "reason": str, "setup_steps": str}. ``reason``
        names what is missing and the next step; ``setup_steps`` is the
        full Windows preparation list (clone -> conda -> torch -> example
        run -> config). The UI shows this instead of starting a generation
        that is bound to fail.
        """
        from recast.assets3d.generate3d import TRIPOSG_SETUP_TEXT, TripoSGBackend
        from recast.config import RecastConfig

        cfg = RecastConfig.from_yaml(self.settings.config_path)
        ok, reason = TripoSGBackend(
            cfg.model3d.triposg_repo or None,
            python=cfg.model3d.triposg_python or None,
        ).check()
        return {"ok": ok, "reason": reason, "setup_steps": TRIPOSG_SETUP_TEXT}

    def generate_costume_variant(
        self,
        character: str,
        *,
        image: str | Path,
        variant_name: str,
        backend: Any | None = None,
    ) -> dict[str, Any]:
        """Generate one dressed-body variant for a character (UI: [生成换装变体…]).

        ``image`` is a full-body photo of the character wearing the outfit
        (a different costume reference, not the main cutout). The variant
        is a whole-body GLB — clothes are outfit variants, not separate
        garment meshes. ``backend`` is only for tests (a stub backend);
        the UI always uses the configured TripoSG backend.
        """
        from recast.assets3d.variants import generate_costume_variant as _generate_variant
        from recast.config import RecastConfig
        from recast.core.characters import character_folder

        cfg = RecastConfig.from_yaml(self.settings.config_path)
        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        report = _generate_variant(
            image,
            folder,
            variant_name=variant_name,
            backend=backend,
            repo_dir=cfg.model3d.triposg_repo or None,
            python=cfg.model3d.triposg_python or None,
        )
        self._on_log(
            "INFO",
            f"人物「{name}」换装变体「{variant_name}」已生成：{report['path']}",
        )
        return report

    def list_costume_variants(self, character: str) -> list[dict[str, Any]]:
        """Every registered costume variant for one character (UI: 换装变体列表)."""
        from recast.assets3d.variants import list_costume_variants as _list_variants
        from recast.core.characters import character_folder

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        return _list_variants(folder)

    def switch_costume_variant(
        self, character: str, *, variant_name: str | None
    ) -> dict[str, Any]:
        """Wear a costume variant, or switch back to the main model (UI: 装配区）.

        ``variant_name`` "" / None switches back to ``model/model.glb``.
        The switch is ``assembly.body``, so the assembly view shows the
        variant as the body right away.
        """
        from recast.assets3d.variants import switch_costume_variant as _switch_variant
        from recast.core.characters import character_folder

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        report = _switch_variant(folder, variant_name)
        worn = report["variant"] or "主模型"
        self._on_log("INFO", f"人物「{name}」已换成{worn}：{report['path']}")
        return report

    # ----- character aliases (N20) -----

    def describe_replace_backends(self) -> dict[str, Any]:
        """Which segmentation / face backends exist here, and their licences.

        The UI needs this before a run, not after: it is what lets a button
        say "会退到 GrabCut，约 3 秒/帧" or "真换脸是非商用模型" instead of
        letting the operator discover it from a slow run or a delivery review.
        """
        from recast.stages.face_swap import describe_face_backends
        from recast.stages.person_mask import describe_backends

        try:
            segmentation = describe_backends()
        except Exception as exc:  # a broken optional dep must not blank the panel
            segmentation = {"grabcut": {"available": True, "license": "", "reason": str(exc)}}
        try:
            face = describe_face_backends()
        except Exception as exc:
            face = {"overlay": {"available": True, "license": "", "final": False, "reason": str(exc)}}
        return {"segmentation": segmentation, "face": face}

    def build_sheet_for_character(self, character: str) -> dict[str, Any]:
        """Build the character sheet (3D turntable + measured appearance)."""
        from recast.stages.character_sheet import build_character_sheet

        folder = self._character_meta_folder(character)
        if folder is None:
            raise ValueError(f"人物「{character}」还没有人物文件夹，先点[建人物文件夹]。")
        report = build_character_sheet(folder)
        appearance = report.get("appearance") or {}
        palette = "、".join(str(p.get("color")) for p in (appearance.get("palette") or [])[:3])
        skin = (appearance.get("skin") or {}).get("name") or "—"
        if report.get("model_rendered"):
            head = f"设定板已建：{len(report.get('turns') or [])} 个角度"
        else:
            head = f"设定板已建（只有照片）：{report.get('render_reason') or ''}"
        return {
            **report,
            "message": f"{head}；主色 {palette or '—'}；肤色 {skin}。位置：{folder / 'sheet'}",
        }

    def preview_garment_swap(
        self,
        character: str,
        *,
        frame_path: str | Path | None = None,
        part: str = "upper",
        cluster: int | None = None,
        allow_noncommercial: bool = True,
    ) -> dict[str, Any]:
        """Recolour one frame's costume and write a before/after pair.

        Deliberately a *preview*: it runs one frame, writes two PNGs next to
        the character, and returns the audit numbers. A full clip is
        ``replace_clip``'s job, and it needs a mask policy decision that a
        button press should not make silently.
        """
        from recast.stages.garment_transfer import (
            IMAGE_SUFFIXES,
            costume_reference,
            read_image,
            split_by_colour,
            transfer_garment,
            _region_mask,
        )
        from recast.stages.person_mask import estimate_neck_y, segment_person_auto

        folder = self._character_meta_folder(character)
        if folder is None:
            raise ValueError(f"人物「{character}」还没有人物文件夹，先点[建人物文件夹]。")
        reference = costume_reference(folder)
        if reference is None:
            raise ValueError(
                f"没有服装参考图：{folder / 'costume'}。先跑[衣服打标签…]，或把全身截图放进 costume/。"
            )

        source = Path(frame_path) if frame_path else None
        if source is None:
            for sub in ("costume", "face"):
                directory = folder / sub
                if not directory.is_dir():
                    continue
                found = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                if found:
                    source = found[0]
                    break
        if source is None or not source.is_file():
            raise ValueError(
                f"没有可用的一帧：把要试的截图放进 {folder / 'costume'}，或调用时传 frame_path。"
            )

        frame = read_image(source)
        seg = segment_person_auto(
            frame, backend="auto", index=0, allow_noncommercial=allow_noncommercial
        )
        neck = estimate_neck_y(frame)
        result = transfer_garment(
            frame,
            seg["mask"],
            int(neck["neck_y"]),
            read_image(reference),
            part=part,
            cluster=cluster,
        )
        # Report the colour parts so a caller can offer "只换这一件" next time.
        try:
            region = _region_mask(seg["mask"], int(neck["neck_y"]), part, None)
            parts = [
                {"order": p["order"], "share": p["share"], "chroma": p["chroma"]}
                for p in split_by_colour(frame, region)
            ]
        except Exception:
            parts = []

        from recast.quality.replace_audit import audit_composite

        audit = audit_composite(
            frame, result["image"], seg["mask"], int(neck["neck_y"])
        )
        out_dir = folder / "replace_preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        import cv2

        def _save(path: Path, image) -> None:
            ok, buf = cv2.imencode(".png", image)
            if ok:
                buf.tofile(str(path))

        before = out_dir / f"before_{part}.png"
        after = out_dir / f"after_{part}.png"
        _save(before, frame)
        _save(after, result["image"])

        licence_note = ""
        if seg["method"].startswith("rmbg"):
            licence_note = "；注意：分割用的是非商用模型 RMBG-1.4，交付前要换后端"
        return {
            "character": character,
            "part": part,
            "method": seg["method"],
            "neck_y": int(neck["neck_y"]),
            "neck_source": neck["source"],
            "region_pixels": result["region_pixels"],
            "resolved_part": result["part"],
            "colour_parts": parts,
            "audit": audit,
            "before": str(before),
            "after": str(after),
            "warning": seg.get("warning") or "",
            "message": (
                f"换装预览（{part}）：分割 {seg['method']}，改动 {result['region_pixels']} 像素，"
                f"背景损伤 {audit['background_damage_frac']:.3%}，"
                f"接缝增量 {audit['seam_increase']:.2f}× → {audit['verdict']}。"
                f"对比图：{after}{licence_note}"
            ),
        }


    def _character_meta_folder(
        self, character: str, *, folder_name: str | None = None
    ) -> Path | None:
        """Find the character folder whose meta.json belongs to ``character``.

        ``character`` may be a character id or a display name. Tries, in
        order: the folder named by the current display name, the folder
        named by ``folder_name`` (the pre-rename name — folders are not
        moved on rename), then a scan for a folder whose meta ``name``
        matches the current display name. Returns None when no folder
        exists yet (e.g. a manually added character with no folder).
        """
        from recast.core.characters import (
            character_folder,
            list_character_folders,
        )

        ch = self.store.get_character(character) if character else None
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        name = str(name or "").strip()
        candidates: list[str] = []
        if name:
            candidates.append(name)
        if folder_name and str(folder_name).strip() not in candidates:
            candidates.append(str(folder_name).strip())
        for cand in candidates:
            folder = character_folder(self.settings.output_root, cand)
            if folder.is_dir():
                return folder
        if not candidates:
            return None
        try:
            for entry in list_character_folders(self.settings.output_root):
                meta = entry.get("meta") or {}
                # meta 的 name 在上一次改名时已同步成当时的新名；多次改名后
                # 它等于候选名里的某一个（旧显示名），而不是最新的显示名。
                if str(meta.get("name") or "").strip() in candidates:
                    return Path(entry["folder"])
        except Exception:
            pass
        return None

    @staticmethod
    def _clean_aliases(raw: Any) -> list[str]:
        """Normalise a stored ``aliases`` value to a deduped str list."""
        out: list[str] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                text = str(item or "").strip()
                if text and text not in out:
                    out.append(text)
        elif isinstance(raw, str) and raw.strip():
            out.append(raw.strip())
        return out

    def get_character_aliases(self, character: str) -> list[str]:
        """Read one character's ``aliases`` from their meta.json (N20).

        Missing folder / missing meta / missing key all read as [] — a
        character simply has no recorded old names yet. A corrupt meta
        also reads as [] rather than breaking the caller.
        """
        from recast.core.characters import load_meta

        folder = self._character_meta_folder(character)
        if folder is None:
            return []
        try:
            meta = load_meta(folder) or {}
        except Exception:
            return []
        return self._clean_aliases(meta.get("aliases"))

    def add_character_alias(
        self, character: str, old_name: str, *, folder_name: str | None = None
    ) -> list[str]:
        """Record ``old_name`` as an alias of ``character`` in meta.json.

        Appends to the ``aliases`` string list (deduped, order kept);
        the current display name itself is never stored as its own alias.
        Returns the alias list afterwards. When the character has no
        folder yet, nothing is written and [] is returned — no stray
        folder is created just to hold an alias.
        """
        from recast.core.characters import load_meta, save_meta

        old = str(old_name or "").strip()
        if not old:
            return self.get_character_aliases(character)
        folder = self._character_meta_folder(character, folder_name=folder_name)
        if folder is None:
            return []
        meta = load_meta(folder) or {}
        ch = self.store.get_character(character) if character else None
        display = (
            (ch or {}).get("display_name") or (ch or {}).get("name") or ""
        )
        display = str(display or "").strip()
        aliases = self._clean_aliases(meta.get("aliases"))
        # 只排除当前显示名本身：meta["name"] 在改名同步前仍是旧名，
        # 不能拿它当“当前名”来挡——否则第一次记别名就永远加不进去。
        if old != display and old not in aliases:
            aliases.append(old)
            meta["aliases"] = aliases
            save_meta(folder, meta)
        return aliases

    def set_voice_timbre(
        self,
        character: str,
        *,
        pitch: float | None = None,
        rate: float | None = None,
        style: str | None = None,
    ) -> dict[str, Any]:
        """Change one character's voice timbre (UI: [调节音色]).

        Only the given fields change. Values are clamped (pitch -12..+12
        semitones, rate 0.5..2.0). The timbre is stored in the character
        folder's meta.json and applied at the next dub/audition.

        A real change also invalidates that character's already-dubbed
        lines: every task workdir whose dialogue.json has lines spoken by
        this character gets those lines marked dirty, and its stale
        dialogue_mix.wav is removed (otherwise step 5's "mix exists means
        done" check would keep showing ✓ on a mix made with the old
        timbre). No change in timbre means nothing is invalidated.
        """
        from recast.core.characters import (
            character_folder,
            get_timbre,
            load_meta,
            set_timbre,
        )

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        # N20：改名后文件夹仍留在旧名下，_character_meta_folder 会按
        # meta 的 name 认出它；真没有文件夹时报错文案与原来一致。
        folder = self._character_meta_folder(character)
        if folder is None:
            expected = character_folder(self.settings.output_root, str(name))
            raise FileNotFoundError(f"找不到人物文件夹：{expected}。请先建人物文件夹。")
        old_timbre = get_timbre(load_meta(folder) or {})
        timbre = set_timbre(folder, pitch=pitch, rate=rate, style=style)
        self._on_log("INFO", f"人物「{name}」音色已调为：{timbre}")
        report: dict[str, Any] = {
            "character": str(name),
            "folder": str(folder),
            "timbre": timbre,
        }
        if timbre == old_timbre:
            return {**report, "affected": [], "stale_mix_removed": []}
        # N20：联动时传入该人物的别名集合——旧名说的台词也要标脏，
        # 否则改名后第 ⑤ 步还对旧混音显示 ✓。
        aliases = set(self.get_character_aliases(character))
        affected, stale_mix_removed = self._invalidate_timbre_dubs(
            str(name), {str(character)} | aliases
        )
        if affected:
            total = sum(a["lines"] for a in affected)
            self._on_log(
                "INFO",
                f"人物「{name}」音色已改：{len(affected)} 集的旧配音已失效"
                f"（共 {total} 句待重配），旧混音已删除。",
            )
        return {**report, "affected": affected, "stale_mix_removed": stale_mix_removed}

    def _invalidate_timbre_dubs(
        self, name: str, aliases: set[str] | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Mark one character's dubbed lines dirty + drop their stale mixes.

        Scans every task's workdir for dialogue.json; a line whose speaker
        or character is this person becomes dirty, and that episode's
        dialogue_mix.wav is removed so step 5 no longer counts the old mix
        as done. Workdirs without (or with an unreadable) dialogue.json are
        skipped. Returns (affected episodes, workdirs whose mix was removed).
        """
        from recast.generate.dialogue import DIALOGUE_FILENAME, load_dialogue, save_dialogue

        wanted = {name} | {str(a) for a in (aliases or set())}
        affected: list[dict[str, Any]] = []
        stale_mix_removed: list[str] = []
        seen: set[str] = set()
        try:
            tasks = self.store.list_rows("tasks")
        except Exception:
            return affected, stale_mix_removed
        for task in tasks:
            workdir = str(task.get("workdir") or "").strip()
            if not workdir or workdir in seen:
                continue
            seen.add(workdir)
            wd = Path(workdir)
            if not (wd / DIALOGUE_FILENAME).is_file():
                continue
            try:
                lines = load_dialogue(wd)
            except Exception:
                continue
            n = 0
            for line in lines:
                if str(line.get("speaker") or "") in wanted or str(
                    line.get("character") or ""
                ) in wanted:
                    line["dirty"] = True
                    n += 1
            if not n:
                continue
            save_dialogue(wd, lines)
            mix = wd / "dialogue_mix.wav"
            if mix.is_file():
                try:
                    mix.unlink()
                    stale_mix_removed.append(str(wd))
                except OSError:
                    pass
            affected.append(
                {"drama_id": task.get("drama_id"), "workdir": str(wd), "lines": n}
            )
        return affected, stale_mix_removed

    def audition_voice(
        self,
        character: str,
        *,
        text: str | None = None,
        timbre: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesize one short line with the character's voice + timbre (UI: [试听声音]).

        About 3 seconds; does not touch the episode dub. The sample is
        written to the character folder's voice/audition.wav.

        When ``timbre`` is given (N19 preview route), it is used as-is
        (normalized/clamped) and the stored timbre in meta.json is neither
        read nor written, so previewing a timbre never invalidates an
        existing dub. When it is omitted, the stored timbre is used, as
        before. Preview mode only reads the voice id from the character
        record, not from meta.json.
        """
        from recast.core.characters import character_folder, get_timbre, load_meta
        from recast.generate.timbre import normalize_timbre
        from recast.generate.tts import synthesize_line

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        if timbre is not None:
            timbre = normalize_timbre(timbre)
            voice = str((ch or {}).get("voice") or (ch or {}).get("voice_id") or "").strip() or None
        else:
            meta = load_meta(folder) or {}
            timbre = get_timbre(meta)
            voice = (
                str((ch or {}).get("voice") or (ch or {}).get("voice_id") or "").strip()
                or str((meta.get("voice") or {}).get("tts_voice") or "").strip()
                or None
            )
        sample = (text or "").strip() or f"你好，我是{name}。"
        dest = folder / "voice" / "audition.wav"
        rec = synthesize_line(sample, dest, voice=voice, timbre=timbre)
        self._on_log(
            "INFO",
            f"人物「{name}」试听完成：{rec.get('source')}，{rec.get('duration') or 0:.1f} 秒。",
        )
        return {"character": str(name), "text": sample, "timbre": timbre, **rec}

    def get_character_assembly(self, character: str) -> dict[str, Any]:
        """Current assembly for one character: stored picks + resolved paths."""
        from recast.core.character_assets import resolve_character_assets
        from recast.core.characters import character_folder, get_assembly

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        assets = resolve_character_assets(self.settings.output_root, str(name), store=self.store)
        stored = get_assembly((assets.meta if assets is not None else {}) or {})
        resolved = {
            "face": str(assets.face_path) if assets is not None and assets.face_path else None,
            "body": str(assets.body_path) if assets is not None and assets.body_path else None,
            "costume": str(assets.costume_path) if assets is not None and assets.costume_path else None,
        }
        return {
            "character": str(name),
            "folder": str(folder),
            "assembly": stored,
            "face": resolved["face"],
            "body": resolved["body"],
            "costume": resolved["costume"],
        }

    def set_character_assembly(
        self,
        character: str,
        *,
        face: str | Path | None = None,
        body: str | Path | None = None,
        costume: str | Path | None = None,
    ) -> dict[str, Any]:
        """Change one character's assembly (UI: 装配视图，点选即应用）.

        Only the given slots change. A costume pick is also registered as
        a binding (``bind_costume`` part "全身") when the character has a
        GLB body: a costume from the shared wardrobe is first copied into
        the character's costume/ (the binding only accepts in-folder
        files). When there is no GLB yet the pick is still saved and the
        binding is reported as skipped, not an error.
        """
        from recast.core.characters import character_folder, set_assembly

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        assembly = set_assembly(
            folder,
            face=face,
            body=body,
            costume=costume,
            output_root=self.settings.output_root,
        )
        binding: dict[str, Any] = {"applied": False, "skipped": True, "reason": "未选衣服"}
        if costume is not None and str(costume or "").strip():
            binding = self._bind_assembly_costume(folder, costume)
        self._on_log("INFO", f"人物「{name}」装配已更新：{assembly}")
        return {
            "character": str(name),
            "folder": str(folder),
            "assembly": assembly,
            "binding": binding,
        }

    def _bind_assembly_costume(
        self, folder: Path, costume: str | Path
    ) -> dict[str, Any]:
        """Register the picked costume on the character's GLB body.

        Copies a shared-wardrobe costume into costume/ first (bindings only
        accept in-folder files). Returns {"applied": bool, ...}; a missing
        GLB is a skip, not an error — the assembly pick is already saved.
        """
        from recast.assets3d.binding import BindingError, bind_costume

        raw = Path(str(costume)).expanduser()
        if not raw.is_absolute():
            candidate = Path(self.settings.output_root).expanduser() / raw
            raw = candidate if candidate.is_file() else raw
        if not raw.is_file():
            return {"applied": False, "skipped": True, "reason": f"找不到衣服文件：{costume}"}
        try:
            in_folder = raw.resolve().relative_to(folder.resolve())
        except (OSError, ValueError):
            in_folder = None
        target = folder / in_folder if in_folder is not None else self._copy_into_costume(folder, raw)
        if in_folder is not None:
            source = ""
        else:
            # Record the wardrobe path as stored in meta.json
            # ("wardrobe/red.png"), not the absolute pick path.
            try:
                source = str(
                    raw.resolve().relative_to(
                        Path(self.settings.output_root).expanduser().resolve()
                    )
                ).replace("\\", "/")
            except (OSError, ValueError):
                source = str(costume).replace("\\", "/")
        try:
            report = bind_costume(folder, costume_file=target, part="全身", source=source)
        except BindingError as exc:
            return {"applied": False, "skipped": True, "reason": str(exc)}
        return {"applied": True, "skipped": False, "reason": "", **report}

    @staticmethod
    def _copy_into_costume(folder: Path, src: Path) -> Path:
        """Copy a shared costume into the character's costume/ (no overwrite)."""
        dest_dir = folder / "costume"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.is_file() and dest.stat().st_size == src.stat().st_size:
            return dest
        if dest.exists():
            index = 2
            while True:
                candidate = dest_dir / f"{src.stem}-{index}{src.suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                if candidate.stat().st_size == src.stat().st_size:
                    return candidate
                index += 1
        shutil.copy2(src, dest)
        return dest

    def list_wardrobe(self) -> list[dict[str, Any]]:
        """Local wardrobe: every costume image in reach (UI: 装配视图）."""
        from recast.core.wardrobe import list_wardrobe as _list_wardrobe

        return _list_wardrobe(self.settings.output_root)

    def list_bodies(self) -> list[dict[str, Any]]:
        """Local body library: every GLB in reach (UI: 装配视图）."""
        from recast.core.wardrobe import list_bodies as _list_bodies

        return _list_bodies(self.settings.output_root)

    def render_character_preview(self, character: str) -> dict[str, Any]:
        """Render the assembled body's front-view preview to model/preview.png.

        Reuses the cached preview when it is newer than the GLB. When
        rendering is unavailable on this machine the result reports
        rendered=False instead of raising — the UI shows a placeholder.
        """
        from recast.assets3d.preview import render_model_preview
        from recast.core.character_assets import resolve_body_model
        from recast.core.characters import character_folder

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        body = resolve_body_model(self.settings.output_root, str(name), store=self.store)
        if body is None:
            return {
                "character": str(name),
                "rendered": False,
                "path": None,
                "message": "人物还没有可用的 GLB 身体模型，先导入或生成 3D 模型。",
            }
        dest = folder / "model" / "preview.png"
        if dest.is_file() and dest.stat().st_mtime >= body.stat().st_mtime:
            return {
                "character": str(name),
                "rendered": False,
                "path": str(dest),
                "message": "预览图已存在。",
            }
        out = render_model_preview(body, dest)
        if out is None:
            return {
                "character": str(name),
                "rendered": False,
                "path": None,
                "message": "本机渲染不了 3D 预览（没有可用的渲染器），请用 Blender 打开模型查看。",
            }
        return {"character": str(name), "rendered": True, "path": str(out), "message": "3D 预览已生成。"}

    def edit_dialogue_line(
        self, workdir: str | Path, shot_code: str, new_text: str
    ) -> dict[str, Any]:
        """Change one dialogue line's text (UI: [编辑对话]).

        Only marks the line dirty; call redub_dialogue to re-synthesize it.
        """
        from recast.generate.dialogue import edit_line

        return edit_line(workdir, shot_code, new_text)

    def redub_dialogue(
        self,
        workdir: str | Path,
        *,
        shot_codes: list[str] | None = None,
        remix: bool = True,
    ) -> dict[str, Any]:
        """Re-dub only the changed lines, then re-mix the episode track (UI: [编辑对话] 后重配).

        Each new wav goes to the speaker's ``voice/lines/<shot_code>.wav``
        and is registered in that character folder's meta.json. Changing
        text or timbre re-runs independently; untouched lines are never
        re-synthesized. Audio is never sped up.
        """
        from recast.generate.dialogue import redub_lines

        report = redub_lines(
            workdir,
            shot_codes=shot_codes,
            output_root=self.settings.output_root,
            store=self.store,
            remix=remix,
        )
        self._on_log(
            "INFO",
            f"重配音完成：重配 {len(report['redubbed'])} 句，跳过 {len(report['skipped'])} 句。",
        )
        return report

    def bind_costume_texture(
        self,
        character: str,
        costume_file: str | Path,
        part: str,
        *,
        texture_slot: str | None = None,
    ) -> dict[str, Any]:
        """Bind a costume image to the character's GLB model (UI: [绑定服装]).

        Registers the binding in meta.json and validates the texture; the
        texture is NOT written into the GLB here (see the returned note for
        the Blender / pygltflib steps on Windows).
        """
        from recast.assets3d.binding import bind_costume
        from recast.core.characters import character_folder

        ch = self.store.get_character(character)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or character
        folder = character_folder(self.settings.output_root, str(name))
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
        report = bind_costume(
            folder, costume_file=costume_file, part=part, texture_slot=texture_slot
        )
        self._on_log(
            "INFO",
            f"人物「{name}」服装已绑定：{report['costume']} -> {part}。",
        )
        return report

    # ----- templates / generate-from-script -----

    def create_script_drama(
        self,
        name: str,
        pack_root: str | Path,
        *,
        storyboard: str | Path | None = None,
        dialogue: str | Path | None = None,
        episode_name: str = "第01集",
    ) -> dict[str, Any]:
        from recast.templates.pack import load_pack, upsert_pack_row
        from recast.templates.validate import validate_pack
        from recast.core.script_pipeline import load_or_parse_shots, material_errors

        pack = load_pack(pack_root)
        upsert_pack_row(self.store, pack)
        root = Path(pack.root)
        # Use a unique root_path so it can live in the queue without a video file
        virtual = str((Path(self.settings.works_root or self.settings.output_root) / "script-jobs" / name).resolve())
        Path(virtual).mkdir(parents=True, exist_ok=True)
        drama = self.store.upsert_drama_by_path(
            virtual,
            origin_name=name,
            episode_count=1,
            source_path=virtual,
        )
        self.store.update(
            "dramas",
            drama["id"],
            new_name=name,
            entry_type="script",
            template_pack_id=pack.name,
            template_version=pack.version,
            pack_root=str(pack.root),
        )
        storyboard = str(storyboard) if storyboard else (str(root / "storyboard.txt") if (root / "storyboard.txt").is_file() else None)
        dialogue = str(dialogue) if dialogue else (str(root / "dialogue.txt") if (root / "dialogue.txt").is_file() else None)
        # placeholder video_path so the unique constraint / NOT NULL holds
        placeholder = str(root / "storyboard.txt") if (root / "storyboard.txt").is_file() else str(root / "pack.json")
        tasks = self.store.replace_tasks_for_drama(
            drama["id"],
            [{"episode_name": episode_name, "video_path": placeholder}],
        )
        task = tasks[0]
        self.store.update(
            "tasks",
            task["id"],
            entry_type="script",
            template_pack_id=str(pack.root),
            template_version=pack.version,
            source_storyboard=storyboard,
            source_dialogue=dialogue,
            source_script=str(root / "script.json"),
        )
        shots = load_or_parse_shots(pack, storyboard=storyboard, dialogue=dialogue)
        files, others, issues = material_errors(pack, shots)
        if files:
            msg = format_missing(files)
            self.store.update("dramas", drama["id"], status=ST_NEED_ASSETS, progress=msg, missing_files=files)
            self.store.update("tasks", task["id"], status=ST_NEED_ASSETS, progress=msg)
        self._on_log("INFO", f"已从模板「{pack.name}」v{pack.version} 创建按剧本生成任务「{name}」。")
        return self.store.get("dramas", drama["id"])  # type: ignore[return-value]

    def assign_pack(self, drama_id: str, pack_root: str | Path) -> dict[str, Any] | None:
        from recast.templates.pack import load_pack, upsert_pack_row

        pack = load_pack(pack_root)
        upsert_pack_row(self.store, pack)
        row = self.store.update(
            "dramas",
            drama_id,
            pack_root=str(pack.root),
            template_pack_id=pack.name,
            template_version=pack.version,
        )
        for t in self.store.tasks_for_drama(drama_id):
            if not t.get("template_version"):
                self.store.update("tasks", t["id"], template_pack_id=str(pack.root), template_version=pack.version)
        self._on_log("INFO", f"剧目已绑定模板「{pack.name}」版本 {pack.version}。")
        return row

    def list_packs(self) -> list[dict[str, Any]]:
        return self.store.list_rows("template_packs", order="updated_at DESC, created_at")

    def register_pack(self, pack_root: str | Path) -> dict[str, Any]:
        from recast.templates.pack import load_pack, upsert_pack_row

        pack = load_pack(pack_root)
        return upsert_pack_row(self.store, pack)

    def cost_runs(self, drama_id: str | None = None, task_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.list_runs(task_id=task_id, drama_id=drama_id)

    def write_cost(self, drama_id: str, dest: str | Path | None = None) -> Path:
        from recast.compose.cost import write_cost_xlsx

        drama = self.store.get("dramas", drama_id)
        if not drama:
            raise FileNotFoundError("找不到该剧目。")
        dest = Path(dest) if dest else Path(drama.get("output_dir") or self.settings.output_root) / "cost.xlsx"
        return write_cost_xlsx(self.store.list_runs(drama_id=drama_id), dest, title=f"{drama.get('new_name') or drama.get('origin_name')} 成本表")

    def review_items(self, drama_id: str | None = None, task_id: str | None = None) -> list[dict[str, Any]]:
        if task_id:
            return self.store.quality_for_task(task_id)
        items: list[dict[str, Any]] = []
        ids = [drama_id] if drama_id else [d["id"] for d in self.store.list_dramas()]
        for did in ids:
            for t in self.store.tasks_for_drama(did):
                items.extend(self.store.quality_for_task(t["id"]))
        return items

    def set_review_status(self, mark_id: str, status: str) -> dict[str, Any] | None:
        if status not in {"待人工检查", "已通过", "已忽略"}:
            raise ValueError("终检状态只能是待人工检查、已通过或已忽略。")
        return self.store.update("quality_marks", mark_id, status=status)

    def rerun_shot(self, task_id: str, shot_code: str) -> dict[str, Any]:
        task = self.store.get("tasks", task_id)
        if not task:
            raise FileNotFoundError("找不到该分集任务。")
        job = Job(
            drama_id=task["drama_id"],
            task_id=task_id,
            stages=["generate", "dub", "export"],
            entry_type=task.get("entry_type") or "script",
            pack_root=task.get("pack_snapshot") or task.get("template_pack_id"),
            storyboard=task.get("source_storyboard"),
            dialogue=task.get("source_dialogue"),
            rerun_shot=shot_code,
            require_auth=True,
            comfyui_url=self.settings.comfyui_url or None,
        )
        self.scheduler.enqueue([job])
        self._on_log("INFO", f"已排队重跑镜头 {shot_code}。")
        return self.snapshot()

    # -- 逐镜头后处理增强（设计 4.2）：终检窗口 / 主窗口的 [增强] 开关调这里 --
    # 持久化不在 shots 表（表里没有 enhance 列），而在任务 workdir 下的
    # shots_enhance.json，只记开着的镜头 {shot_code: true}；默认全关。
    # 合成时 runner.enhance_shots_for_task 读同一文件，接线在
    # recast/stages/compose.py（时间段滤镜，整段视频上只处理开着的镜头）。

    SHOT_ENHANCE_FILENAME = "shots_enhance.json"

    def _shot_enhance_path(self, task_id: str) -> Path:
        task = self.store.get("tasks", task_id) if task_id else None
        workdir = (task or {}).get("workdir")
        if not task or not workdir:
            raise FileNotFoundError("找不到该分集任务。")
        return Path(str(workdir)) / self.SHOT_ENHANCE_FILENAME

    def shot_enhance_flags(self, task_id: str | None) -> dict[str, bool]:
        """Per-shot enhance switches for a task ({shot_code: True}, default {}).

        Unknown / missing task returns {} instead of raising, so the review
        window can render its switches before any task exists.
        """
        if not task_id:
            return {}
        try:
            path = self._shot_enhance_path(task_id)
        except FileNotFoundError:
            return {}
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): True for k, v in data.items() if v}

    def set_shot_enhance(
        self, task_id: str, shot_code: str, enabled: bool
    ) -> dict[str, bool]:
        """Switch one shot's enhance on/off and persist it. Returns all flags.

        Raises FileNotFoundError for an unknown task and ValueError for an
        unknown shot_code (when the task has shots recorded in the store;
        with no shots recorded yet any non-empty code is accepted).
        """
        if not shot_code or not str(shot_code).strip():
            raise ValueError("镜头编号不能为空。")
        code = str(shot_code)
        path = self._shot_enhance_path(task_id)
        shots = self.store.shots_for_task(task_id)
        if shots and code not in {str(s.get("shot_code") or "") for s in shots}:
            raise ValueError(f"找不到镜头 {code}。")
        flags = self.shot_enhance_flags(task_id)
        if enabled:
            flags[code] = True
        else:
            flags.pop(code, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return dict(flags)

    def build_l2(self, character_id: str, photos: list[str] | None = None) -> dict[str, Any]:
        from recast.assets3d.avatar import bind_path_for, build_avatar

        ch = self.store.get_character(character_id)
        if not ch:
            raise FileNotFoundError("找不到该人物。")
        if not (ch.get("authorized") and ch.get("auth_type")):
            raise PermissionError("请先勾选授权，再生成 3D头像资产。")
        faces = photos or [a["path"] for a in (ch.get("assets") or []) if a.get("kind") == "face" and a.get("path") and Path(str(a["path"])).is_file()]
        # bind_path_for appends l2/<id>. Passing output_root/l2 landed files in l2/l2/<id>.
        dest = bind_path_for(character_id, self.settings.output_root)
        asset = build_avatar(
            character_id,
            faces,
            dest,
            display_name=ch.get("display_name") or ch.get("name") or "",
            models_dir=self.settings.models_dir,
        )
        self.store.set_character_asset(character_id, "3d", asset["path"], complete=True)
        rows = self.store.list_rows("character_assets", "character_id = ? AND kind = ?", (character_id, "3d"))
        if rows:
            self.store.update(
                "character_assets",
                rows[0]["id"],
                quality_score=asset.get("quality_score"),
                source_note=asset.get("source_note"),
                notes=asset.get("honesty"),
            )
        self._on_log("INFO", f"已为「{ch.get('display_name') or ch.get('name')}」生成 3D头像资产（可跨集复用，不是生产级完整网格）。")
        return asset

    def delete_l2(self, character_id: str) -> dict[str, Any]:
        """Delete one character's 3D头像资产 files and clear the database row.

        Face, costume, and voice assets are left untouched. The path must be a
        character folder inside the configured L2 output tree (canonical
        ``output_root/l2/<id>`` or the legacy ``output_root/l2/l2/<id>``).
        """
        from recast.assets3d.avatar import L2PathError, remove_avatar_dir, require_l2_asset_dir

        ch = self.store.get_character(character_id)
        if not ch:
            raise FileNotFoundError("找不到该人物。")
        rows = [a for a in (ch.get("assets") or []) if a.get("kind") in {"3d", "3d_avatar"}]
        if not rows:
            raise FileNotFoundError("该人物没有 3D头像资产。")

        planned: list[tuple[dict[str, Any], Path | None]] = []
        for row in rows:
            raw = row.get("path")
            if not raw:
                planned.append((row, None))
                continue
            try:
                dest = require_l2_asset_dir(raw, self.settings.output_root)
            except L2PathError as exc:
                raise PermissionError(str(exc)) from exc
            planned.append((row, dest))

        protected = self._protected_asset_paths()
        removed: list[str] = []
        failed: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for row, dest in planned:
            if dest is not None and dest not in seen:
                try:
                    remove_avatar_dir(dest, protected=protected)
                except (OSError, L2PathError) as exc:
                    # 单个目录删失败（文件被占用等）不掀翻整个操作：记下来，
                    # 对应的数据库行保留，等用户重试。已删掉的不受影响。
                    failed.append({"path": str(dest), "reason": str(exc)})
                    continue
                removed.append(str(dest))
                seen.add(dest)
            self.store.delete("character_assets", row["id"])
        name = ch.get("display_name") or ch.get("name") or character_id
        if failed:
            self._on_log(
                "WARNING",
                f"「{name}」的 3D头像资产只删了一部分，{len(failed)} 个目录没删掉："
                + "、".join(item["path"] for item in failed)
                + "。请关闭占用文件的程序后重试。",
            )
        else:
            self._on_log("INFO", f"已删除「{name}」的 3D头像资产。人脸照片仍保留，3D 状态为「缺失」。")
        return {
            "character_id": character_id,
            "display_name": name,
            "removed": removed,
            "failed": failed,
            "partial": bool(failed),
            "status": "缺失" if not failed else "部分删除",
        }

    def list_unused_l2(self) -> list[dict[str, Any]]:
        """List 3D头像 folders under the L2 tree that nothing still references.

        Includes unreferenced folders, leftovers on the doubled ``l2/l2`` path,
        and folders whose character row is gone. Referenced assets, and
        anything outside the L2 tree, are omitted.
        """
        from recast.assets3d.avatar import L2PathError, iter_l2_asset_dirs, load_avatar, require_l2_asset_dir

        output_root = self.settings.output_root
        characters = {c["id"]: c for c in self.store.list_characters()}
        referenced: set[Path] = set()
        dangling: list[dict[str, Any]] = []
        for row in self.store.list_rows("character_assets"):
            if row.get("kind") not in {"3d", "3d_avatar"}:
                continue
            if row.get("character_id") not in characters:
                dangling.append(row)
                continue
            raw = row.get("path")
            if not raw:
                continue
            try:
                dest = require_l2_asset_dir(raw, output_root)
            except L2PathError:
                continue
            referenced.add(dest)

        items: list[dict[str, Any]] = []
        by_path: dict[Path, dict[str, Any]] = {}

        def _add(folder: Path, *, legacy: bool, row_id: str | None = None) -> None:
            try:
                resolved = require_l2_asset_dir(folder, output_root)
            except L2PathError:
                return
            existing = by_path.get(resolved)
            if existing is not None:
                if row_id and row_id not in existing["asset_row_ids"]:
                    existing["asset_row_ids"].append(row_id)
                return
            if resolved in referenced:
                return
            meta = load_avatar(resolved) or {}
            meta_id = str(meta.get("character_id") or "")
            cid = meta_id or folder.name
            display = str(meta.get("display_name") or "")
            character = characters.get(cid)
            if character and not display:
                display = str(character.get("display_name") or character.get("name") or "")
            if legacy:
                reason = "旧路径残留（l2/l2），未被引用"
            elif meta_id and meta_id not in characters:
                reason = "对应人物已不存在"
            else:
                reason = "未被任何人物引用"
            item = {
                "character_id": cid,
                "display_name": display,
                "name": display or cid,
                "path": str(resolved),
                "reason": reason,
                "legacy": legacy,
                "asset_row_ids": [row_id] if row_id else [],
            }
            items.append(item)
            by_path[resolved] = item

        for folder, legacy in iter_l2_asset_dirs(output_root):
            _add(folder, legacy=legacy)

        for row in dangling:
            raw = row.get("path") or ""
            if not raw:
                items.append(
                    {
                        "character_id": str(row.get("character_id") or ""),
                        "display_name": "",
                        "name": str(row.get("character_id") or "未知人物"),
                        "path": "",
                        "reason": "对应人物已不存在",
                        "legacy": False,
                        "asset_row_ids": [row["id"]],
                    }
                )
                continue
            try:
                dest = require_l2_asset_dir(raw, output_root)
            except L2PathError:
                continue
            if dest in referenced:
                continue
            _add(dest, legacy=len(dest.relative_to(Path(output_root).expanduser().resolve() / "l2").parts) == 2, row_id=row["id"])

        items.sort(key=lambda item: (item.get("name") or "", item.get("path") or ""))
        return items

    def delete_unused_l2(self, only: list[str] | None = None) -> dict[str, Any]:
        """Delete unused 3D头像资产 currently listed by :meth:`list_unused_l2`.

        ``only`` limits the delete to paths the user already confirmed. Paths
        outside the L2 tree are ignored. Still-referenced assets are skipped
        even if a caller names them.
        """
        from recast.assets3d.avatar import L2PathError, l2_tree_root, remove_avatar_dir, require_l2_asset_dir

        items = self.list_unused_l2()
        if only is not None:
            allowed: set[str] = set()
            include_pathless = False
            for raw in only:
                if not raw:
                    include_pathless = True
                    continue
                try:
                    allowed.add(str(require_l2_asset_dir(raw, self.settings.output_root)))
                except L2PathError:
                    continue
            selected: list[dict[str, Any]] = []
            for item in items:
                path = item.get("path") or ""
                if path and str(Path(path).resolve()) in allowed:
                    selected.append(item)
                elif not path and include_pathless:
                    selected.append(item)
            items = selected

        planned: list[tuple[dict[str, Any], Path | None]] = []
        for item in items:
            raw = item.get("path") or ""
            dest: Path | None = None
            if raw:
                try:
                    dest = require_l2_asset_dir(raw, self.settings.output_root)
                except L2PathError:
                    continue
            planned.append((item, dest))

        protected = self._protected_asset_paths()
        removed: list[dict[str, Any]] = []
        for item, dest in planned:
            if dest is not None:
                remove_avatar_dir(dest, protected=protected)
            for row_id in item.get("asset_row_ids") or []:
                self.store.delete("character_assets", row_id)
            removed.append(
                {
                    "character_id": item.get("character_id") or "",
                    "display_name": item.get("display_name") or "",
                    "name": item.get("name") or item.get("character_id") or "",
                    "path": item.get("path") or "",
                    "reason": item.get("reason") or "",
                }
            )
        doubled = l2_tree_root(self.settings.output_root) / "l2"
        try:
            if doubled.is_dir() and not doubled.is_symlink() and not any(doubled.iterdir()):
                doubled.rmdir()
        except OSError:
            pass
        self._on_log("INFO", f"已清理 {len(removed)} 个未使用的 3D头像资产。")
        return {"count": len(removed), "removed": removed}

    def delete_character(self, character_id: str) -> dict[str, Any]:
        """Remove one person from the library, plus safe generated files.

        The ``characters`` row and every ``character_assets`` row are deleted.
        A 3D头像资产 is removed only when it passes the same path checks as
        :meth:`delete_l2`. Generated face samples under ``face_samples`` inside
        the output tree may be deleted; a reference photo outside that tree is
        left on disk. Drama, task, and output videos are not touched.
        """
        from recast.assets3d.avatar import L2PathError, remove_avatar_dir, require_l2_asset_dir
        from recast.core.sample_paths import plan_sample_removal, prune_empty_sample_dirs

        ch = self.store.get_character(character_id)
        if not ch:
            raise FileNotFoundError("找不到该人物。")
        # 正在排队或处理中的任务已经拿到了这个人物的绑定，删掉会让任务在换脸
        # 那步找不到人。先拒绝，等任务停了或跑完再删。
        name = ch.get("display_name") or ch.get("name") or character_id
        blocking = self.scheduler.jobs_using_character(character_id)
        if blocking:
            detail = "、".join(f"{item['label']}（{item['status']}）" for item in blocking)
            raise RuntimeError(
                f"人物「{name}」正被任务使用（{detail}），请先停止或等任务完成后再删除。"
            )
        assets = list(ch.get("assets") or [])
        output_root = self.settings.output_root

        planned_l2: list[tuple[dict[str, Any], Path | None]] = []
        for row in assets:
            if row.get("kind") not in {"3d", "3d_avatar"}:
                continue
            raw = str(row.get("path") or "").strip()
            if not raw:
                planned_l2.append((row, None))
                continue
            try:
                dest = require_l2_asset_dir(raw, output_root)
            except L2PathError as exc:
                if _l2_target_present(raw):
                    raise PermissionError(str(exc)) from exc
                planned_l2.append((row, None))
                continue
            planned_l2.append((row, dest))

        sample_targets: list[Path] = []
        sample_rows_by_path: dict[Path, list[dict[str, Any]]] = {}
        kept_files: list[str] = []
        prune_from: list[Path] = []
        shared = self._asset_paths_used_by_others(character_id)
        seen_samples: set[Path] = set()
        for row in assets:
            if row.get("kind") != "face":
                continue
            raw = str(row.get("path") or "").strip()
            if not raw:
                continue
            decision = plan_sample_removal(raw, output_root)
            if decision.action == "delete" and decision.path is not None:
                if decision.path in shared or decision.path in seen_samples:
                    kept_files.append(str(decision.path))
                    continue
                sample_targets.append(decision.path)
                sample_rows_by_path.setdefault(decision.path, []).append(row)
                seen_samples.add(decision.path)
            elif decision.action == "missing":
                prune_from.append(Path(raw).expanduser().parent)
            elif decision.action == "skip" and _path_present(raw):
                kept_files.append(str(decision.path or raw))

        removed_samples: list[str] = []
        failed_samples: list[dict[str, Any]] = []
        for dest in sample_targets:
            if dest.is_symlink() or not dest.is_file():
                continue
            parent = dest.parent
            try:
                dest.unlink()
            except OSError as exc:
                # 单个样本删失败（文件被占用等）不掀翻整个操作：记下来继续删别的。
                failed_samples.append({"path": str(dest), "reason": str(exc)})
                continue
            removed_samples.append(str(dest))
            prune_from.append(parent)

        pruned: list[str] = []
        seen_dirs: set[Path] = set()
        for directory in prune_from:
            for removed in prune_empty_sample_dirs(directory, output_root):
                if removed in seen_dirs:
                    continue
                seen_dirs.add(removed)
                pruned.append(str(removed))

        protected = self._protected_asset_paths()
        removed_l2: list[str] = []
        failed_l2: list[dict[str, Any]] = []
        seen_l2: set[Path] = set()
        for _row, dest in planned_l2:
            if dest is None or dest in seen_l2:
                continue
            try:
                remove_avatar_dir(dest, protected=protected)
            except (OSError, L2PathError) as exc:
                failed_l2.append({"path": str(dest), "reason": str(exc)})
                continue
            removed_l2.append(str(dest))
            seen_l2.add(dest)

        failed = failed_samples + failed_l2
        if failed:
            # 有文件没删掉：人物留在库里，只删掉那些文件确实已删的资料行，
            # 让用户关掉占用文件的程序后重试，而不是留一个“半残”的人物。
            removed_sample_paths = {Path(p) for p in removed_samples}
            for path, rows in sample_rows_by_path.items():
                if path in removed_sample_paths:
                    for row in rows:
                        self.store.delete("character_assets", row["id"])
            removed_l2_paths = set(removed_l2)
            for row, dest in planned_l2:
                if dest is None or str(dest) in removed_l2_paths:
                    self.store.delete("character_assets", row["id"])
            self._on_log(
                "WARNING",
                f"「{name}」只删了一部分，{len(failed)} 个文件没删掉："
                + "、".join(item["path"] for item in failed)
                + "。人物仍在人物库中，请关闭占用文件的程序后重试。",
            )
            return {
                "character_id": character_id,
                "display_name": name,
                "removed_samples": removed_samples,
                "pruned_dirs": pruned,
                "kept_files": kept_files,
                "removed_l2": removed_l2,
                "failed": failed,
                "partial": True,
            }

        self.store.delete_character(character_id)
        self._on_log("INFO", f"已将「{name}」移出人物库。")
        return {
            "character_id": character_id,
            "display_name": name,
            "removed_samples": removed_samples,
            "pruned_dirs": pruned,
            "kept_files": kept_files,
            "removed_l2": removed_l2,
            "failed": [],
            "partial": False,
        }

    def list_invalid_samples(self) -> dict[str, Any]:
        """Describe stale asset rows and people who have no valid face file.

        A row is stale when ``path`` is set and the file is gone. For a 3D头像
        asset, the row is stale only when the asset directory itself is gone;
        a directory that is still there is kept. Every character with no
        remaining valid face file is listed for removal, including confirmed
        and authorized people. Characters who still have a face file stay;
        those who also have a stale row are listed under ``keep_characters``.
        """
        from recast.assets3d.avatar import asset_directory

        characters = {c["id"]: c for c in self.store.list_characters()}
        stale: list[dict[str, Any]] = []
        stale_by_character: dict[str, list[dict[str, Any]]] = {}
        for row in self.store.list_rows("character_assets"):
            if not _asset_row_is_stale(row, asset_directory):
                continue
            ch = characters.get(row.get("character_id")) or {}
            item = {
                "id": row.get("id"),
                "character_id": row.get("character_id") or "",
                "kind": row.get("kind") or "",
                "kind_label": _asset_kind_label(row.get("kind")),
                "path": row.get("path") or "",
                "name": ch.get("display_name") or ch.get("name") or row.get("character_id") or "未知人物",
                "display_name": ch.get("display_name") or ch.get("name") or "",
                "label": _stale_asset_label(ch, row),
            }
            stale.append(item)
            stale_by_character.setdefault(item["character_id"], []).append(item)

        remove_characters: list[dict[str, Any]] = []
        keep_characters: list[dict[str, Any]] = []
        for ch in characters.values():
            valid_face = _has_valid_face(ch)
            has_stale = ch["id"] in stale_by_character
            if not valid_face:
                remove_characters.append(_character_brief(ch, kept=False))
            elif has_stale:
                keep_characters.append(_character_brief(ch, kept=True))

        kind_counts: dict[str, int] = {}
        for item in stale:
            kind_counts[item["kind_label"]] = kind_counts.get(item["kind_label"], 0) + 1
        remove_characters.sort(key=lambda item: (item.get("name") or "", item.get("id") or ""))
        keep_characters.sort(key=lambda item: (item.get("name") or "", item.get("id") or ""))
        stale.sort(key=lambda item: (item.get("name") or "", item.get("kind") or "", item.get("path") or ""))
        return {
            "stale_assets": stale,
            "stale_count": len(stale),
            "kind_counts": kind_counts,
            "remove_characters": remove_characters,
            "remove_count": len(remove_characters),
            "keep_characters": keep_characters,
            "keep_count": len(keep_characters),
        }

    def clean_invalid_samples(self) -> dict[str, Any]:
        """Drop stale asset rows, then drop anyone with no remaining valid face.

        Generated sample directories that are already empty may be removed.
        Files that still exist, including a valid L2 directory, are not deleted.
        A person who still has a face file stays, confirmed or not. A person
        with no remaining face file leaves the library, including when they
        are confirmed or authorized.
        """
        from recast.core.sample_paths import plan_sample_removal, prune_empty_sample_dirs

        # 失效判断全靠 is_file()：输出盘没挂载、输出路径被移动时，所有人脸
        # 都会被判成“不存在”。先确认输出根目录本身可达，否则直接拒绝，
        # 绝不能在这种状态下把整库人物都删了。
        output_root = self.settings.output_root
        root = Path(output_root).expanduser()
        if not root.is_dir():
            raise RuntimeError(
                f"输出目录不可用（{output_root}），无法判断哪些样本真的失效。"
                "请检查外接硬盘是否已挂载、输出路径是否被移动或改名，确认后再清理。"
            )

        plan = self.list_invalid_samples()
        pruned: list[str] = []
        seen_dirs: set[Path] = set()
        for item in plan["stale_assets"]:
            raw = str(item.get("path") or "").strip()
            if not raw or item.get("kind") in {"3d", "3d_avatar"}:
                continue
            decision = plan_sample_removal(raw, output_root)
            if decision.action != "missing":
                continue
            start = Path(raw).expanduser()
            start = start if start.is_dir() else start.parent
            for removed in prune_empty_sample_dirs(start, output_root):
                if removed in seen_dirs:
                    continue
                seen_dirs.add(removed)
                pruned.append(str(removed))

        self.store.delete_asset_rows([item["id"] for item in plan["stale_assets"] if item.get("id")])
        self.store.delete_characters([item["id"] for item in plan["remove_characters"] if item.get("id")])
        self._on_log(
            "INFO",
            f"已清理 {plan['stale_count']} 条失效资料，移出 {plan['remove_count']} 个人物。",
        )
        return {**plan, "pruned_dirs": pruned}

    def _asset_paths_used_by_others(self, character_id: str) -> set[Path]:
        used: set[Path] = set()
        for row in self.store.list_rows("character_assets"):
            if str(row.get("character_id") or "") == str(character_id):
                continue
            raw = row.get("path")
            if not raw:
                continue
            path = Path(str(raw)).expanduser()
            try:
                if path.is_symlink() or not path.exists():
                    continue
                used.add(path.resolve())
            except OSError:
                continue
        return used

    def _protected_asset_paths(self) -> set[Path]:
        """Face, costume, and voice files that cleanup must not unlink."""
        kept: set[Path] = set()
        for row in self.store.list_rows("character_assets"):
            if row.get("kind") not in {"face", "costume", "voice"}:
                continue
            raw = row.get("path")
            if not raw:
                continue
            path = Path(str(raw)).expanduser()
            try:
                if path.exists():
                    kept.add(path.resolve())
            except OSError:
                continue
        return kept

    def generate_prop(self, dest_dir: str | Path, prompt: str = "", image: str | None = None) -> dict[str, Any]:
        from recast.assets3d.gen3d import generate_prop

        return generate_prop(
            dest_dir,
            prompt=prompt,
            image=image,
            comfyui_url=self.settings.comfyui_url or None,
        )



def _flag_on(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no"}
    return bool(value)


def _path_present(raw: str) -> bool:
    path = Path(raw).expanduser()
    try:
        return path.exists() or path.is_symlink()
    except OSError:
        return True


def _l2_target_present(raw: str) -> bool:
    from recast.assets3d.avatar import asset_directory

    path = Path(raw).expanduser()
    directory = asset_directory(path)
    try:
        return path.exists() or path.is_symlink() or directory.exists() or directory.is_symlink()
    except OSError:
        return True


def _has_valid_face(character: dict[str, Any]) -> bool:
    for asset in character.get("assets") or []:
        if asset.get("kind") != "face" or not asset.get("path"):
            continue
        try:
            if Path(str(asset["path"])).is_file():
                return True
        except OSError:
            continue
    return False


def _person_label(character: dict[str, Any]) -> str:
    name = str(character.get("display_name") or character.get("name") or "未命名")
    cid = str(character.get("id") or "")
    if cid and cid != name:
        return f"{name}（{cid}）"
    return name or cid or "未命名"


def _asset_kind_label(kind: str | None) -> str:
    if kind in {"3d", "3d_avatar"}:
        return "3D"
    return {"face": "脸", "costume": "服装", "voice": "声音"}.get(kind or "", kind or "资料")


def _stale_asset_label(character: dict[str, Any], row: dict[str, Any]) -> str:
    base = _person_label(character) if character else str(row.get("character_id") or "未知人物")
    kind = _asset_kind_label(str(row.get("kind") or ""))
    filename = Path(str(row.get("path") or "")).name
    if filename:
        return f"{base}  {kind}  {filename}"
    return f"{base}  {kind}"


def _character_brief(character: dict[str, Any], *, kept: bool) -> dict[str, Any]:
    confirmed = _flag_on(character.get("confirmed"))
    authorized = _flag_on(character.get("authorized"))
    if authorized and confirmed:
        status = "已确认 / 已授权"
    elif authorized:
        status = "已授权"
    elif confirmed:
        status = "已确认"
    else:
        status = "未确认"
    return {
        "id": character.get("id") or "",
        "name": character.get("display_name") or character.get("name") or character.get("id") or "未命名",
        "display_name": character.get("display_name") or character.get("name") or "",
        "label": _person_label(character),
        "confirmed": confirmed,
        "authorized": authorized,
        "status": status,
        "kept": kept,
    }


def _asset_row_is_stale(row: dict[str, Any], asset_directory: Any) -> bool:
    raw = str(row.get("path") or "").strip()
    if not raw:
        return False
    path = Path(raw)
    try:
        if row.get("kind") in {"3d", "3d_avatar"}:
            return not asset_directory(path).exists()
        return not path.is_file()
    except OSError:
        return True


_DISK_ASSET_SUFFIXES = {
    "face": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
    "costume": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
    "voice": {".wav", ".mp3", ".flac", ".m4a"},
}


def _asset_on_disk(output_root: str | Path, character: dict[str, Any], kind: str) -> bool:
    """Is this kind present as a file in the character folder?

    The row is the index, the folder is the data. Importing a character by
    copying its folder (or seeding a demo pack) creates no asset rows, and the
    panel then said "脸：缺失" while the face photo was displayed right next to
    it. The folder is what every pipeline stage actually reads, so it wins.
    """
    suffixes = _DISK_ASSET_SUFFIXES.get(kind)
    if not suffixes:
        return False
    name = str(character.get("display_name") or character.get("name") or "").strip()
    if not name:
        return False
    from recast.core.characters import character_folder

    directory = character_folder(output_root, name) / kind
    if not directory.is_dir():
        return False
    try:
        return any(p.is_file() and p.suffix.lower() in suffixes for p in directory.iterdir())
    except OSError:
        return False


def asset_flags(
    character: dict[str, Any], *, output_root: str | Path | None = None
) -> dict[str, str]:
    assets = {a.get("kind"): a for a in (character.get("assets") or [])}
    # also accept kind aliases used by L2
    if "3d_avatar" in {a.get("kind") for a in (character.get("assets") or [])} and "3d" not in assets:
        assets["3d"] = next(a for a in character["assets"] if a.get("kind") in {"3d", "3d_avatar"})
    flags = {}
    labels = {"face": "脸", "costume": "服装", "voice": "声音", "3d": "3D"}
    for kind, label in labels.items():
        a = assets.get(kind)
        if kind == "3d":
            from recast.assets3d import asset_status

            flags[kind] = asset_status(a.get("path") if a else None)
            continue
        present = bool(a and a.get("path") and Path(str(a["path"])).is_file())
        if not present and output_root is not None:
            present = _asset_on_disk(output_root, character, kind)
        flags[kind] = "已齐" if present else "缺失"
    return flags


def _authorized_swap_ids(store: Store) -> list[str]:
    ids: list[str] = []
    for ch in store.list_characters():
        if not (ch.get("authorized") and ch.get("auth_type")):
            continue
        if any(
            a.get("kind") == "face" and a.get("path") and Path(str(a["path"])).is_file()
            for a in (ch.get("assets") or [])
        ):
            ids.append(ch["id"])
    return ids


def _selected_character_names(
    store: Store,
    character_id: str | None,
    character_ids: list[str] | None,
) -> list[str]:
    """Display names of the characters selected for this run (N11).

    Only the replace pass selects characters: an explicit ``character_id``
    limits the run to one person, otherwise the authorised swap ids.
    Anything else (no replace stage, unknown id) is [] — never a guessed
    name, so the history snapshot shows "未记录" instead of dirty data.
    """
    names: list[str] = []
    ids = [character_id] if character_id else list(character_ids or [])
    for cid in ids:
        try:
            ch = store.get_character(cid)
        except Exception:
            ch = None
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or ""
        if str(name).strip():
            names.append(str(name).strip())
    return names


def _missing_list(drama: dict[str, Any]) -> list[str]:
    raw = drama.get("missing_files")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        import json

        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return []


def open_folder(path: str | Path) -> None:
    target = Path(path)
    if target.is_file():
        target = target.parent
    if not target.exists():
        raise FileNotFoundError(f"找不到路径: {target}")
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
