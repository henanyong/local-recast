"""UI-agnostic facade: scan, queue, character library, settings, engine status."""

from __future__ import annotations

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
        return s

    def engine(self) -> dict[str, Any]:
        return engine_snapshot(self.settings, running=self.scheduler.running)

    def first_run(self) -> dict[str, Any]:
        return first_run_report(self.settings)

    def snapshot(self, status_filter: str | None = None) -> dict[str, Any]:
        dramas = self.store.list_dramas()
        if status_filter and status_filter not in {"", "全部"}:
            dramas = [d for d in dramas if d.get("status") == status_filter]
        sched = self.scheduler.snapshot()
        return {
            "product": PRODUCT_NAME,
            "version": __version__,
            "dramas": dramas,
            "engine": self.engine(),
            "scheduler": sched,
            "settings": self.settings.to_dict(),
            "stages": [{"id": i, "label": l} for i, l, _ in STAGE_GROUPS],
            "works_root": self.settings.works_root,
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

    def _commit_hit(self, hit: DramaHit) -> dict[str, Any]:
        drama = self.store.upsert_drama_by_path(
            str(hit.root),
            origin_name=hit.origin_name,
            episode_count=hit.episode_count,
        )
        episodes = [{"episode_name": ep.name, "video_path": str(ep.video_path)} for ep in hit.episodes]
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
            # pick default authorized character if none specified
            cid = character_id
            if "replace" in stages_for and not cid:
                for ch in self.store.list_characters():
                    if ch.get("authorized") and ch.get("auth_type"):
                        cid = ch["id"]
                        break
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

    def build_l2(self, character_id: str, photos: list[str] | None = None) -> dict[str, Any]:
        from recast.assets3d.avatar import bind_path_for, build_avatar

        ch = self.store.get_character(character_id)
        if not ch:
            raise FileNotFoundError("找不到该人物。")
        if not (ch.get("authorized") and ch.get("auth_type")):
            raise PermissionError("请先勾选授权，再生成 3D头像资产。")
        faces = photos or [a["path"] for a in (ch.get("assets") or []) if a.get("kind") == "face" and a.get("path") and Path(str(a["path"])).is_file()]
        dest = bind_path_for(character_id, Path(self.settings.output_root) / "l2")
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

    def generate_prop(self, dest_dir: str | Path, prompt: str = "", image: str | None = None) -> dict[str, Any]:
        from recast.assets3d.gen3d import generate_prop

        return generate_prop(
            dest_dir,
            prompt=prompt,
            image=image,
            comfyui_url=self.settings.comfyui_url or None,
        )



def asset_flags(character: dict[str, Any]) -> dict[str, str]:
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
        if a and a.get("path") and Path(str(a["path"])).is_file():
            flags[kind] = "已齐"
        else:
            flags[kind] = "缺失"
    return flags


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
