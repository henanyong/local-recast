"""Template pack I/O: folder of JSON + a SQLite row, with pinned versions."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from recast.templates.fill import apply_pack_to_shots
from recast.templates.validate import ValidationIssue, validate_pack

LEVELS = ("script", "character", "scene", "shot")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_pack_dict(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    data.setdefault("root", str(root))
    return data


@dataclass
class TemplatePack:
    root: Path
    name: str
    version: str
    script: dict[str, Any] = field(default_factory=dict)
    characters: list[dict[str, Any]] = field(default_factory=list)
    scenes: list[dict[str, Any]] = field(default_factory=list)
    shots: list[dict[str, Any]] = field(default_factory=list)
    workflows: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "root": str(self.root),
            "script": self.script,
            "characters": self.characters,
            "scenes": self.scenes,
            "shots": self.shots,
            "workflows": self.workflows,
            "meta": self.meta,
        }

    def validate(self, *, require_files: bool = True) -> list[ValidationIssue]:
        return validate_pack(
            self.to_dict(),
            pack_root=self.root if require_files else None,
            fight_max_two=bool(self.script.get("fight_max_two", True)),
        )

    def errors(self, *, require_files: bool = True) -> list[ValidationIssue]:
        return [i for i in self.validate(require_files=require_files) if i.is_error]

    def character(self, key: str) -> dict[str, Any] | None:
        for ch in self.characters:
            if key in {ch.get("id"), ch.get("name"), ch.get("display_name")}:
                return ch
        return None

    def ref_path(self, token: str) -> Path | None:
        if not token:
            return None
        p = Path(token)
        candidates = [
            p if p.is_absolute() else self.root / p,
            self.root / "refs" / p.name,
            self.root / "refs" / token,
        ]
        for c in candidates:
            if c.is_file():
                return c
        # stem search
        stem = p.stem
        refs = self.root / "refs"
        if refs.is_dir():
            for f in refs.iterdir():
                if f.is_file() and (f.stem == stem or stem in f.stem or f.stem.startswith(stem)):
                    return f
        return None

    def resolve_shot_refs(
        self, shot: dict[str, Any], *, output_root: str | Path | None = None
    ) -> list[Path]:
        """Reference stills for one shot, folder face first for person tokens.

        A token that names a character (e.g. “温念”) resolves to that person's
        folder face/ when a character folder exists; anything else resolves
        against the pack's refs directory exactly as before. Without
        ``output_root`` the behaviour is unchanged.
        """
        paths: list[Path] = []
        for item in shot.get("ref_assets") or []:
            token = item.get("file") if isinstance(item, dict) else item
            if not token or str(token).startswith("无"):
                continue
            found: Path | None = None
            if output_root is not None:
                from recast.core.character_assets import resolve_generation_ref

                found = resolve_generation_ref(output_root, str(token))
            if found is None:
                found = self.ref_path(str(token))
            if found:
                paths.append(found)
        return paths


def load_pack(root: str | Path) -> TemplatePack:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到模板包目录: {root}")
    pack_file = root / "pack.json"
    meta: dict[str, Any] = _load_json(pack_file) if pack_file.is_file() else {}
    script_path = root / (meta.get("script") or "script.json")
    script = _load_json(script_path) if script_path.is_file() else (meta.get("script_data") or {})

    characters: list[dict[str, Any]] = []
    cdir = root / "characters"
    if cdir.is_dir():
        for f in sorted(cdir.glob("*.json")):
            characters.append(_load_json(f))
    elif (root / "characters.json").is_file():
        characters = _load_json(root / "characters.json")

    scenes: list[dict[str, Any]] = []
    sdir = root / "scenes"
    if sdir.is_dir():
        for f in sorted(sdir.glob("*.json")):
            scenes.append(_load_json(f))
    elif (root / "scenes.json").is_file():
        scenes = _load_json(root / "scenes.json")

    shots: list[dict[str, Any]] = []
    shdir = root / "shots"
    if shdir.is_dir():
        files = sorted(shdir.glob("*.json"), key=lambda p: p.name)
        for f in files:
            shots.append(_load_json(f))
    elif (root / "shots.json").is_file():
        shots = _load_json(root / "shots.json")

    workflows: dict[str, Any] = {}
    wdir = root / "workflows"
    if wdir.is_dir():
        for f in wdir.glob("*.json"):
            workflows[f.stem] = _load_json(f)

    name = meta.get("name") or script.get("name") or root.name
    version = str(meta.get("version") or script.get("version") or "1.0.0")
    return TemplatePack(
        root=root,
        name=name,
        version=version,
        script=script if isinstance(script, dict) else {},
        characters=characters,
        scenes=scenes,
        shots=shots,
        workflows=workflows,
        meta=meta,
    )


def save_pack(pack: TemplatePack | dict[str, Any], root: str | Path | None = None) -> Path:
    if isinstance(pack, TemplatePack):
        data = pack.to_dict()
        dest = Path(root or pack.root)
        obj = pack
    else:
        data = dict(pack)
        dest = Path(root or data.get("root") or "")
        obj = None
    if not dest:
        raise ValueError("保存模板包需要目标目录")
    dest.mkdir(parents=True, exist_ok=True)
    script = data.get("script") or {}
    version = str(data.get("version") or script.get("version") or "1.0.0")
    name = data.get("name") or script.get("name") or dest.name
    _dump_json(
        dest / "pack.json",
        {
            "name": name,
            "version": version,
            "level": "pack",
            "script": "script.json",
            "characters": "characters/",
            "scenes": "scenes/",
            "shots": "shots/",
            "workflows": "workflows/",
        },
    )
    _dump_json(dest / "script.json", script)
    cdir = dest / "characters"
    if cdir.exists():
        for f in cdir.glob("*.json"):
            f.unlink()
    for ch in data.get("characters") or []:
        fname = f"{ch.get('id') or ch.get('name') or 'character'}.json"
        _dump_json(cdir / fname, ch)
    sdir = dest / "scenes"
    if sdir.exists():
        for f in sdir.glob("*.json"):
            f.unlink()
    for sc in data.get("scenes") or []:
        fname = f"{sc.get('id') or sc.get('name') or 'scene'}.json"
        _dump_json(sdir / fname, sc)
    shdir = dest / "shots"
    if shdir.exists():
        for f in shdir.glob("*.json"):
            f.unlink()
    for sh in data.get("shots") or []:
        code = sh.get("shot_code") or sh.get("code") or sh.get("id") or "shot"
        _dump_json(shdir / f"{code}.json", sh)
    wdir = dest / "workflows"
    for key, wf in (data.get("workflows") or {}).items():
        _dump_json(wdir / f"{key}.json", wf)
    (dest / "refs").mkdir(exist_ok=True)
    if obj is not None:
        obj.root = dest
        obj.version = version
        obj.name = name
    return dest


def export_pack(pack: TemplatePack, dest_zip_or_dir: str | Path) -> Path:
    dest = Path(dest_zip_or_dir)
    if dest.suffix.lower() in {".zip"}:
        base = dest.with_suffix("")
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        shutil.copytree(pack.root, base, ignore=_ignore_heavy)
        archive = shutil.make_archive(str(base), "zip", root_dir=base)
        return Path(archive)
    if dest.exists() and dest.is_dir():
        target = dest / pack.root.name
    else:
        target = dest
        target.mkdir(parents=True, exist_ok=True)
    if target.resolve() != pack.root.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(pack.root, target, ignore=_ignore_heavy)
    return target


def _ignore_heavy(_dir: str, names: list[str]) -> set[str]:
    skip = {".venv", "__pycache__", ".git", "outputs"}
    return {n for n in names if n in skip}


def import_pack(src: str | Path, dest_dir: str | Path) -> TemplatePack:
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".zip":
        shutil.unpack_archive(str(src), str(dest_dir))
        # find pack.json
        found = list(dest_dir.rglob("pack.json"))
        if not found:
            raise FileNotFoundError("压缩包里没有 pack.json")
        return load_pack(found[0].parent)
    target = dest_dir / src.name
    if src.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
        return load_pack(target)
    return load_pack(src)


def snapshot_pack(pack: TemplatePack, dest: str | Path) -> Path:
    """Copy the exact pack used by a task so later edits cannot pollute reruns."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(pack.root, dest, ignore=_ignore_heavy)
    pin = {
        "name": pack.name,
        "version": pack.version,
        "pinned_at": _now(),
        "source": str(pack.root),
    }
    _dump_json(dest / "pin.json", pin)
    return dest


def pin_pack_for_task(pack: TemplatePack, workdir: str | Path) -> Path:
    dest = Path(workdir) / "templates_used"
    return snapshot_pack(pack, dest)


def upsert_pack_row(store: Any, pack: TemplatePack) -> dict[str, Any]:
    """Write a templates / template_packs SQLite row (level + version)."""
    existing = store.list_rows("template_packs", "root_path = ?", (str(pack.root),))
    now = _now()
    if existing:
        row = store.update(
            "template_packs",
            existing[0]["id"],
            name=pack.name,
            version=pack.version,
            root_path=str(pack.root),
            updated_at=now,
        )
        pack_id = existing[0]["id"]
    else:
        row = store.insert(
            "template_packs",
            name=pack.name,
            version=pack.version,
            root_path=str(pack.root),
            created_at=now,
            updated_at=now,
        )
        pack_id = row["id"]
    # per-level rows
    def _upsert_level(level: str, name: str, payload: Any, json_path: str) -> None:
        found = store.list_rows(
            "templates", "pack_id = ? AND (level_name = ? OR name = ?)", (pack_id, level, name)
        )
        # also match by level column as text
        if not found:
            found = [
                r
                for r in store.list_rows("templates", "pack_id = ?", (pack_id,))
                if (r.get("level_name") or r.get("level")) == level and r.get("name") == name
            ]
        fields = {
            "name": name,
            "level": level,  # stored as text; SQLite affinity is fine
            "level_name": level,
            "version": pack.version,
            "json_path": json_path,
            "pack_id": pack_id,
            "payload": payload,
            "updated_at": now,
        }
        if found:
            store.update("templates", found[0]["id"], **fields)
        else:
            store.insert("templates", **fields)

    _upsert_level("script", pack.script.get("name") or pack.name, pack.script, str(pack.root / "script.json"))
    for ch in pack.characters:
        _upsert_level("character", ch.get("name") or ch.get("id"), ch, str(pack.root / "characters"))
    for sc in pack.scenes:
        _upsert_level("scene", sc.get("name") or sc.get("id"), sc, str(pack.root / "scenes"))
    for sh in pack.shots:
        _upsert_level("shot", sh.get("shot_code") or sh.get("id"), sh, str(pack.root / "shots"))
    return store.get("template_packs", pack_id)


def apply_and_record_version(shots: list[dict[str, Any]], pack: TemplatePack) -> list[dict[str, Any]]:
    return apply_pack_to_shots(shots, pack.to_dict())
