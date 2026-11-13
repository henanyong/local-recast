"""Character folders: one folder per person under output/characters/<name>/.

This is the unit the realignment plan is built on. A character folder holds
everything that belongs to one person, instead of scattering assets by type:

    characters/<名字>/
      meta.json        # 名字及来源、衣服描述、声音样本、音色、3D 绑定
      face/            # 人脸样本
      costume/         # 服装截图 + 描述
      voice/           # 按人切出的声音样本
      cutout/          # 抠像 PNG（透明底）
      model/           # 3D 模型（自动生成或手动选，GLB）
        model.glb
        textures/

Migration from the old layout is idempotent and never deletes sources:
face_samples/<auto_name>/ -> face/, l2/<id>/ -> model/ (marked as a face-card
placeholder, not a full mesh), roster costume/voice paths -> costume//voice/.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

CHARACTERS_DIR_NAME = "characters"
SUBDIRS = ("face", "costume", "voice", "cutout", "model")
META_FILENAME = "meta.json"
META_VERSION = 1

# Windows 不允许出现在文件夹名里的字符。
_FORBIDDEN_CHARS = '<>:"/\\|?*'


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_folder_name(name: str) -> str:
    """Make a display name safe to use as a folder name (Windows-safe)."""
    text = "".join("_" if ch in _FORBIDDEN_CHARS else ch for ch in str(name or "").strip())
    text = " ".join(text.split())
    return text or "未命名"


def characters_root(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / CHARACTERS_DIR_NAME


def character_folder(output_root: str | Path, name: str) -> Path:
    return characters_root(output_root) / safe_folder_name(name)


def default_meta(name: str, *, name_source: str = "manual") -> dict[str, Any]:
    """Blank meta.json for a character folder (schema v1)."""
    now = _now()
    return {
        "version": META_VERSION,
        "name": name,
        "name_source": name_source,  # auto | roster | manual
        "costume": {"description": "", "source": "", "files": [], "missing": []},
        "voice": {
            "samples": [],
            "missing": [],
            "timbre": {"pitch": 0.0, "rate": 1.0, "style": ""},
        },
        "model": {"source": "none", "format": "", "path": "", "note": ""},
        # 当前装配：这个人物现在用哪张脸、哪个身体（GLB）、哪件衣服。
        # 值是相对路径（人物文件夹内相对文件夹，共享库内相对 output_root），
        # "" 表示没选（回落到各截面的首个可用项）。选脸选身体选衣服都写这里，
        # 换脸、生成、预览都认这一个地方。
        "assembly": {"face": "", "body": "", "costume": ""},
        "created_at": now,
        "updated_at": now,
        "source": {"drama": "", "task_id": ""},
    }


def get_timbre(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Read the voice timbre from a meta dict (defaults when absent)."""
    from recast.generate.timbre import normalize_timbre

    voice = (meta or {}).get("voice") if isinstance((meta or {}).get("voice"), dict) else {}
    return normalize_timbre(voice.get("timbre"))


def set_timbre(
    folder: str | Path,
    *,
    pitch: float | None = None,
    rate: float | None = None,
    style: str | None = None,
) -> dict[str, Any]:
    """Update one character's voice timbre in meta.json.

    Only the given fields change; the rest keep their values. Values are
    clamped to the supported ranges (pitch -12..+12 semitones,
    rate 0.5..2.0). Returns the effective timbre.
    """
    from recast.generate.timbre import normalize_timbre

    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
    meta = load_meta(folder) or default_meta(folder.name)
    voice = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
    current = normalize_timbre(voice.get("timbre"))
    patch: dict[str, Any] = {}
    if pitch is not None:
        patch["pitch"] = pitch
    if rate is not None:
        patch["rate"] = rate
    if style is not None:
        patch["style"] = style
    meta["voice"] = {**voice, "timbre": normalize_timbre({**current, **patch})}
    save_meta(folder, meta)
    log.info("人物「%s」音色已更新: %s", meta.get("name") or folder.name, meta["voice"]["timbre"])
    return dict(meta["voice"]["timbre"])


ASSEMBLY_KEYS = ("face", "body", "costume")


def get_assembly(meta: dict[str, Any] | None) -> dict[str, str]:
    """Read the current assembly from a meta dict ("" when nothing chosen)."""
    raw = (meta or {}).get("assembly") if isinstance((meta or {}).get("assembly"), dict) else {}
    return {key: str(raw.get(key) or "").strip() for key in ASSEMBLY_KEYS}


def _assembly_allowed_roots(
    folder: str | Path, output_root: str | Path | None = None
) -> tuple[tuple[Path, Path, Path], Path]:
    """Directories an assembly pick may live in.

    The character folder itself, plus the shared wardrobe/ and models/
    libraries under the output root. ``folder`` is
    ``<output_root>/characters/<name>``, so the root is its grandparent
    unless given explicitly.
    """
    folder = Path(folder).expanduser().resolve()
    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else folder.parent.parent
    )
    return (folder, root / "wardrobe", root / "models"), root


def resolve_assembly_path(
    folder: str | Path, value: str, *, output_root: str | Path | None = None
) -> Path | None:
    """Absolute path for one stored assembly value, or None.

    "" means nothing chosen. Relative values are tried against the
    character folder first, then the output root (shared libraries).
    A value whose file is gone resolves to None, never to a guess.
    """
    value = str(value or "").strip()
    if not value:
        return None
    folder = Path(folder).expanduser()
    _roots, root = _assembly_allowed_roots(folder, output_root)
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in (folder, root):
        trial = base / value
        if trial.is_file():
            return trial
    return None


def set_assembly(
    folder: str | Path,
    *,
    face: str | Path | None = None,
    body: str | Path | None = None,
    costume: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, str]:
    """Update one character's assembly in meta.json.

    Only the given fields change; the rest keep their values. Pass "" to
    clear a slot. A pick must be an existing file inside the character
    folder or inside the shared wardrobe/ / models/ libraries; anything
    outside (or a symlink) is refused. Face/costume must be images, body
    must be a valid GLB. Returns the effective assembly (stored relative
    paths, forward slashes).
    """
    from recast.assets3d.glb import validate_glb
    from recast.core.constants import IMAGE_EXTS

    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"找不到人物文件夹：{folder}。请先建人物文件夹。")
    roots, root = _assembly_allowed_roots(folder, output_root)
    meta = load_meta(folder) or default_meta(folder.name)
    current = get_assembly(meta)
    patch: dict[str, str] = {}
    for key, raw in (("face", face), ("body", body), ("costume", costume)):
        if raw is None:
            continue
        value = str(raw or "").strip()
        if not value:
            patch[key] = ""
            continue
        src = Path(value).expanduser()
        if src.is_symlink():
            raise ValueError(f"装配选择拒绝符号链接：{value}")
        try:
            resolved = src.resolve()
        except OSError as exc:
            raise ValueError(f"装配选择解析失败：{value}（{exc}）") from exc
        if not resolved.is_file():
            raise ValueError(f"装配选择不是有效文件：{value}")
        allowed = next((base for base in roots if _is_within(resolved, base)), None)
        if allowed is None:
            raise ValueError(
                f"装配选择必须在人物文件夹里面，或在共享衣柜（wardrobe/）/ 模型库（models/）里面，"
                f"文件夹外的文件不接受：{value}"
            )
        if key in ("face", "costume"):
            if resolved.suffix.lower() not in IMAGE_EXTS:
                raise ValueError(f"装配的{key}必须是图片（jpg/png/webp/bmp）：{value}")
        else:
            if resolved.suffix.lower() != ".glb":
                raise ValueError(f"装配的身体必须是 GLB 模型文件：{value}")
            try:
                validate_glb(resolved)
            except ValueError as exc:
                raise ValueError(f"装配的身体模型已损坏，读不懂：{value}（{exc}）") from exc
        if allowed == folder:
            patch[key] = str(resolved.relative_to(folder)).replace("\\", "/")
        else:
            # Shared library: store relative to the output root so the
            # reference survives a folder rename (e.g. "models/fox.glb").
            patch[key] = str(resolved.relative_to(root)).replace("\\", "/")
    meta["assembly"] = {**current, **patch}
    save_meta(folder, meta)
    log.info("人物「%s」装配已更新: %s", meta.get("name") or folder.name, meta["assembly"])
    return dict(meta["assembly"])


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def add_face_file(folder: str | Path, path: str | Path) -> None:
    """Register a face sample file in meta.json (no duplicates)."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    meta = load_meta(folder) or default_meta(folder.name)
    face = meta.get("face") if isinstance(meta.get("face"), dict) else {}
    files = [str(f) for f in (face.get("files") or [])]
    if str(path) not in files:
        files.append(str(path))
    meta["face"] = {**face, "files": files}
    save_meta(folder, meta)


def remove_face_file(folder: str | Path, path: str | Path) -> bool:
    """Unregister a face sample file from meta.json. True when removed."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    meta = load_meta(folder)
    if meta is None:
        return False
    face = meta.get("face") if isinstance(meta.get("face"), dict) else {}
    files = [str(f) for f in (face.get("files") or [])]
    if str(path) not in files:
        return False
    files = [f for f in files if f != str(path)]
    meta["face"] = {**face, "files": files}
    save_meta(folder, meta)
    return True


def load_meta(folder: str | Path) -> dict[str, Any] | None:
    """Read meta.json from a character folder, or None when it is absent."""
    path = Path(folder) / META_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "null")
    except (OSError, ValueError) as exc:
        raise ValueError(f"人物信息损坏，读不懂 {path}：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"人物信息损坏，{path} 不是一个对象。")
    return data


def save_meta(folder: str | Path, meta: dict[str, Any]) -> Path:
    """Write meta.json (refreshes updated_at).

    N36: every meta.json write takes the shared ``META_WRITE_LOCK``
    (``recast.core.write_lock``) — the GUI thread and the 3D-generation
    worker thread use the same lock, so two concurrent writes can never
    interleave into a torn file. The write itself is atomic (temp file
    in the same directory + ``os.replace``), so a reader never sees a
    half-written file and a failed serialisation leaves the old file
    untouched.

    The write is also a shallow top-level merge under the lock: the
    current on-disk content is read first and the incoming dict wins
    per top-level key, so keys the incoming dict does not carry at all
    (e.g. the ``{"name": ...}`` fallback in ``generate_model_glb``)
    are preserved instead of wiped. It does **not** rescue a stale
    full dict: a caller that loaded meta before another thread's save
    and then saves the whole stale dict will still overwrite the other
    thread's section with its old value. Full read-modify-write safety
    needs the caller to hold ``meta_write_lock`` across the whole
    load -> mutate -> save span (see ``write_lock``); ``save_meta``
    alone guarantees the write itself is serialised and never torn.
    A corrupt on-disk file falls back to writing the incoming dict
    as-is rather than losing the save.
    """
    import os
    import tempfile

    from recast.core.write_lock import meta_write_lock

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / META_FILENAME
    with meta_write_lock():
        merged = dict(meta)
        try:
            raw = path.read_text(encoding="utf-8") if path.is_file() else ""
            current = json.loads(raw) if raw.strip() else None
        except (OSError, ValueError) as exc:
            log.warning("人物信息 %s 读不懂（%s），本次写入不再合并旧内容。", path, exc)
            current = None
        if isinstance(current, dict):
            merged = {**current, **merged}
        merged = {**merged, "updated_at": _now()}
        payload = json.dumps(merged, ensure_ascii=False, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(folder), prefix=".meta-", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return path


def ensure_character_folder(
    output_root: str | Path,
    name: str,
    *,
    name_source: str = "manual",
    source: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create the folder tree and meta.json for one person.

    Running it twice is safe: existing folders and meta.json are kept, and
    only missing subdirectories are created.
    """
    name = str(name or "").strip()
    if not name:
        raise ValueError("人物名字不能为空。")
    folder = character_folder(output_root, name)
    created = not folder.is_dir()
    for sub in SUBDIRS:
        (folder / sub).mkdir(parents=True, exist_ok=True)
    meta = load_meta(folder)
    if meta is None:
        meta = default_meta(name, name_source=name_source)
        if source:
            meta["source"] = {**meta["source"], **source}
        save_meta(folder, meta)
    elif created:
        # Folder existed without meta (e.g. copied in by hand): adopt it.
        save_meta(folder, meta)
    if created:
        log.info("已创建人物文件夹: %s", folder)
    return folder, meta


def list_character_folders(output_root: str | Path) -> list[dict[str, Any]]:
    """Every character folder under output_root, with its meta if present."""
    root = characters_root(output_root)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        try:
            meta = load_meta(child)
        except ValueError:
            meta = None
        out.append({"name": child.name, "folder": str(child), "meta": meta})
    return out


def _copy_file_if_needed(src: Path, dest: Path) -> str:
    """Copy one file unless the destination already has identical content.

    Returns "copied" or "skipped". Never overwrites a different file.
    """
    if dest.is_file() and dest.stat().st_size == src.stat().st_size:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return "copied"


def _copy_tree_if_needed(src: Path, dest: Path) -> tuple[int, int]:
    """Copy a directory tree, skipping files whose destination already matches.

    Returns (copied, skipped) file counts.
    """
    copied = skipped = 0
    for item in sorted(src.rglob("*")):
        if item.is_dir() or item.is_symlink():
            continue
        rel = item.relative_to(src)
        if _copy_file_if_needed(item, dest / rel) == "copied":
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def migrate_to_character_folders(
    output_root: str | Path,
    *,
    face_samples: dict[str, list[str]] | None = None,
    l2_root: str | Path | None = None,
    roster: Any | None = None,
    embeddings: dict[str, list[float]] | None = None,
    merge_similarity: float | None = None,
    store: Any | None = None,
) -> dict[str, Any]:
    """Migrate existing assets into per-character folders.

    Sources are only read, never deleted. Running it twice gives the same
    result: the second run copies nothing (all files skipped).

    ``face_samples`` maps auto_name -> [sample paths] (the face_samples
    index.json content). ``l2_root`` defaults to ``<output_root>/l2``.
    ``roster`` is a Roster object (or a path/dict/list for parse_roster).

    Folder-level merge: before creating a folder for an auto_name, the
    samples are resolved to an existing character folder when possible,
    instead of always creating a ``Person_N`` folder:

    1. Store ownership — when the store is available (passed in, or
       auto-opened from the default ``<output_root>/store/recast.db``),
       a sample file already registered as a face asset of a character
       row belongs to that row's folder (the ingest step already ran the
       store-level merge, so the row is the merged identity).
    2. Embedding comparison — when ``embeddings`` maps auto_name to the
       cluster embedding (identity.json ``persons[].embedding``), it is
       compared against existing folders' embeddings; at or above
       ``merge_similarity`` (default ``IdentityConfig.merge_similarity``)
       the samples merge into that folder.
    3. Otherwise the old path: roster name_map, else the auto_name itself.

    Returns a report: created folders, copied/skipped file counts,
    ``merged`` (samples that went into an already-existing folder),
    ``resolved`` (every auto_name whose folder came from the store or an
    embedding comparison), ``merge_note`` (why no comparison ran, when
    it did not), errors.
    """
    from recast.core.roster import parse_roster

    output_root = Path(output_root).expanduser().resolve()
    name_map: dict[str, str] = {}
    roster_entries: list[Any] = []
    if roster is not None:
        parsed = roster if hasattr(roster, "entries") else parse_roster(roster)
        roster_entries = list(getattr(parsed, "entries", []) or [])
        name_map = parsed.name_map() if hasattr(parsed, "name_map") else {}

    report: dict[str, Any] = {
        "created": [],
        "copied": 0,
        "skipped": 0,
        "merged": [],
        "resolved": [],
        "merge_note": "",
        "errors": [],
    }
    created: set[str] = set()

    # ---- 0. 归并准备：store 资产归属 + 文件夹 embedding 候选 ----
    # 只读；打不开 store 或没有 embedding 时回落到老路，并在报告里写明
    # 原因——不建错文件夹，也不静默丢归并。
    from recast.core.merge import (
        find_merge_target,
        load_existing_embeddings,
        merge_embeddings,
        parse_embedding,
    )

    if merge_similarity is None:
        try:
            from recast.config import IdentityConfig

            threshold = float(IdentityConfig().merge_similarity)
        except Exception:
            threshold = 0.6
    else:
        threshold = float(merge_similarity)

    resolved_store = store
    if resolved_store is None:
        db_path = output_root / "store" / "recast.db"
        if db_path.is_file():
            try:
                from recast.store.db import Store as _Store

                resolved_store = _Store(db_path)
            except Exception as exc:  # noqa: BLE001 - migration must not fail on the store
                report["merge_note"] = f"人物库打不开（{exc}），未做 store 归并。"
                resolved_store = None
    owner_by_path: dict[str, dict[str, Any]] = {}
    if resolved_store is not None:
        try:
            for ch in resolved_store.list_characters():
                owner = str(ch.get("display_name") or ch.get("name") or "")
                if not owner:
                    continue
                for asset in ch.get("assets") or []:
                    if asset.get("kind") == "face" and asset.get("path"):
                        owner_by_path.setdefault(str(asset["path"]), ch)
        except Exception as exc:  # noqa: BLE001 - same fallback as above
            report["merge_note"] = f"读人物库失败（{exc}），未做 store 归并。"
            owner_by_path = {}

    candidates: list[dict[str, Any]] = []
    if embeddings:
        try:
            candidates = load_existing_embeddings(None, output_root)
        except Exception as exc:  # noqa: BLE001 - no candidates, no comparison
            report["merge_note"] = f"读人物文件夹 embedding 失败（{exc}），未做 embedding 比对。"
            candidates = []
    if resolved_store is None and not embeddings and not report["merge_note"]:
        report["merge_note"] = (
            "未提供 embedding 且打不开人物库，未做归并比对；如需归并，"
            "请从 identity.json 传入人物 embedding，或确认人物库可用。"
        )

    def _resolve_folder(auto_name: str, paths: list[str]) -> tuple[str, str, str]:
        """Folder name + how it was chosen + detail, for one auto_name.

        via is "store资产归属" / "embedding比对" / "花名册" / "auto_name".
        """
        if owner_by_path:
            owners: dict[str, int] = {}
            for raw in paths:
                row = owner_by_path.get(str(Path(str(raw)).expanduser()))
                if row is None:
                    continue
                key = str(row.get("display_name") or row.get("name") or "")
                if key:
                    owners[key] = owners.get(key, 0) + 1
            if owners:
                best_name = max(sorted(owners), key=lambda k: owners[k])
                return best_name, "store资产归属", f"样本已属于「{best_name}」"
        emb = (embeddings or {}).get(auto_name)
        if emb is None:
            emb = (embeddings or {}).get(str(auto_name))
        if emb is not None and candidates:
            target, sim = find_merge_target(emb, candidates, threshold)
            if target is not None:
                return str(target["name"]), "embedding比对", f"相似度 {sim:.2f}"
        display = name_map.get(auto_name) or name_map.get(str(auto_name)) or auto_name
        via = "花名册" if display != auto_name else "auto_name"
        return str(display), via, ""

    def _ensure(name: str, *, name_source: str) -> tuple[Path, dict[str, Any]]:
        folder = character_folder(output_root, name)
        existed = folder.is_dir()
        folder, meta = ensure_character_folder(output_root, name, name_source=name_source)
        if not existed:
            created.add(folder.name)
        return folder, meta

    # ---- 1. face_samples -> face/ ----
    for auto_name, paths in (face_samples or {}).items():
        try:
            folder_name, via, detail = _resolve_folder(
                str(auto_name), [str(p) for p in (paths or [])]
            )
            source = "roster" if via == "花名册" else "auto"
            existed = character_folder(output_root, folder_name).is_dir()
            folder, meta = _ensure(folder_name, name_source=source)
            files: list[str] = list(meta.get("face", {}).get("files", []) or []) if isinstance(
                meta.get("face"), dict
            ) else []
            files_before = len(files)
            added = 0
            for raw in paths or []:
                src = Path(str(raw)).expanduser()
                if not src.is_file():
                    report["errors"].append(f"人脸样本不存在，跳过：{raw}")
                    continue
                dest = folder / "face" / src.name
                if _copy_file_if_needed(src, dest) == "copied":
                    report["copied"] += 1
                    added += 1
                else:
                    report["skipped"] += 1
                if str(dest) not in files:
                    files.append(str(dest))
            meta["face"] = {"files": files}
            # 归并进来的 embedding 写进文件夹镜像，供下一次比对用；
            # 只在真的加了新样本时更新，重复跑不改变它（幂等）。
            emb = (embeddings or {}).get(str(auto_name))
            if emb is not None and added > 0:
                merged_emb = merge_embeddings(
                    meta.get("face_embedding"),
                    emb,
                    old_count=max(1, files_before),
                    new_count=added,
                )
                if merged_emb is not None:
                    meta["face_embedding"] = merged_emb
                    parsed = parse_embedding(merged_emb)
                    for cand in candidates:
                        if cand["name"] == folder_name:
                            cand["embedding"] = parsed
                            break
                    else:
                        candidates.append(
                            {
                                "character_id": None,
                                "name": folder_name,
                                "embedding": parsed,
                                "source": "meta",
                                "cluster_key": None,
                            }
                        )
            save_meta(folder, meta)
            if via in ("store资产归属", "embedding比对"):
                report["resolved"].append(
                    {"from": str(auto_name), "into": folder_name, "via": via, "detail": detail}
                )
                if existed and added > 0:
                    report["merged"].append(
                        {
                            "from": str(auto_name),
                            "into": folder_name,
                            "via": via,
                            "detail": detail,
                        }
                    )
        except (OSError, ValueError) as exc:
            report["errors"].append(f"迁移人脸样本 {auto_name} 失败：{exc}")

    # ---- 2. l2/<id>/ -> model/ (face-card placeholder, not a full mesh) ----
    l2 = Path(l2_root).expanduser().resolve() if l2_root else output_root / "l2"
    try:
        from recast.assets3d.avatar import iter_l2_asset_dirs, load_avatar

        for asset_dir, _legacy in iter_l2_asset_dirs(l2.parent):
            try:
                info = load_avatar(asset_dir) or {}
                display = (
                    info.get("display_name")
                    or info.get("character_id")
                    or asset_dir.name
                )
                display = name_map.get(str(display)) or str(display)
                folder, meta = _ensure(display, name_source="roster")
                copied, skipped = _copy_tree_if_needed(asset_dir, folder / "model")
                report["copied"] += copied
                report["skipped"] += skipped
                meta["model"] = {
                    "source": "face_card",
                    "format": "obj",
                    "path": "model/asset.json",
                    "note": "脸卡占位（关键点 + 脸部贴图 + 驱动配置），不是完整网格。",
                }
                save_meta(folder, meta)
            except (OSError, ValueError) as exc:
                report["errors"].append(f"迁移 3D 资产 {asset_dir} 失败：{exc}")
    except (OSError, ValueError) as exc:
        report["errors"].append(f"扫描 L2 目录失败：{exc}")

    # ---- 3. roster costume/voice -> costume//voice/ (copy, or register) ----
    for entry in roster_entries:
        key = str(getattr(entry, "key", "") or "")
        display = str(getattr(entry, "name", "") or key)
        if not display:
            continue
        for kind, attr in (("costume", "costume"), ("voice", "voice")):
            raw = getattr(entry, attr, None)
            if not raw:
                continue
            try:
                folder, meta = _ensure(display, name_source="roster")
                src = Path(str(raw)).expanduser()
                section = meta.get(kind) if isinstance(meta.get(kind), dict) else {}
                if kind == "costume":
                    files = list(section.get("files", []) or [])
                    missing = list(section.get("missing", []) or [])
                else:
                    files = list(section.get("samples", []) or [])
                    missing = list(section.get("missing", []) or [])
                if src.is_file():
                    dest = folder / kind / src.name
                    if _copy_file_if_needed(src, dest) == "copied":
                        report["copied"] += 1
                    else:
                        report["skipped"] += 1
                    if str(dest) not in files:
                        files.append(str(dest))
                else:
                    # 复制不了就登记原路径，不丢信息。
                    if str(raw) not in missing:
                        missing.append(str(raw))
                    report["errors"].append(f"{kind} 文件不存在，已登记原路径：{raw}")
                if kind == "costume":
                    meta["costume"] = {**section, "files": files, "missing": missing}
                else:
                    meta["voice"] = {**section, "samples": files, "missing": missing}
                save_meta(folder, meta)
            except (OSError, ValueError) as exc:
                report["errors"].append(f"迁移 {display} 的 {kind} 失败：{exc}")

    report["created"] = sorted(created)
    log.info(
        "人物文件夹迁移完成: 新建 %s 个，复制 %s 个文件，跳过 %s 个，问题 %s 条。",
        len(report["created"]),
        report["copied"],
        report["skipped"],
        len(report["errors"]),
    )
    return report
