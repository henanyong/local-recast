"""Hard template validation (save + before execute)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from recast.core.state_machine import format_missing


@dataclass
class ValidationIssue:
    level: str  # error | warning
    code: str
    message: str
    shot_code: str | None = None
    missing_files: list[str] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.level == "error"


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def primary_actions_of(shot: dict[str, Any]) -> list[str]:
    """Return the list of declared primary actions on a shot."""
    if "primary_actions" in shot and shot["primary_actions"] is not None:
        items = [str(x).strip() for x in _as_list(shot["primary_actions"]) if str(x).strip()]
        return items
    raw = shot.get("primary_action")
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    # A single field may accidentally hold two actions split by ； or newline
    # or the pattern "动作A，然后动作B" when the author used a list-like separator.
    if "\n" in text or "；" in text or ";" in text:
        parts = [p.strip() for p in re_split(text) if p.strip()]
        if len(parts) >= 2:
            return parts
    return [text] if text else []


def re_split(text: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for ch in text:
        if ch in "；;\n":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf.strip())
    return out


def _people(shot: dict[str, Any]) -> list[str]:
    people = shot.get("people") or shot.get("characters") or shot.get("character_ids") or []
    names = [str(x).strip() for x in _as_list(people) if str(x).strip()]
    return names


def _ref_names(shot: dict[str, Any]) -> list[str]:
    refs = shot.get("ref_assets") or shot.get("refs") or shot.get("reference_images") or []
    out: list[str] = []
    for item in _as_list(refs):
        if isinstance(item, dict):
            name = item.get("file") or item.get("path") or item.get("name") or item.get("character")
            if name:
                out.append(str(name))
        elif item:
            out.append(str(item))
    return [x for x in out if x and x not in {"无", "无（环境）", "环境", "none", "None"}]


def _missing_ref_filename(person: str, shot: dict[str, Any], pack: dict[str, Any] | None) -> str:
    """Name the file the user should add, e.g. 温念.png or 01-温念.jpg."""
    refs = shot.get("ref_assets") or []
    for item in _as_list(refs):
        if isinstance(item, dict):
            who = str(item.get("character") or item.get("name") or "")
            if who == person:
                path = item.get("file") or item.get("path")
                if path:
                    return Path(str(path)).name
    # character template may declare a default face filename
    if pack:
        for ch in pack.get("characters") or []:
            if (ch.get("id") or ch.get("name") or ch.get("display_name")) == person:
                face = ch.get("face") or ch.get("face_file") or ""
                if face:
                    return Path(str(face)).name
                # conventional name
                return f"{ch.get('name') or person}.png"
    # fall back to "person.png"
    return f"{person}.png"


def validate_shot(
    shot: dict[str, Any],
    *,
    pack: dict[str, Any] | None = None,
    pack_root: str | Path | None = None,
    fight_max_two: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    code = str(shot.get("shot_code") or shot.get("code") or "")
    actions = primary_actions_of(shot)
    if len(actions) >= 2:
        issues.append(
            ValidationIssue(
                "error",
                "one_action",
                f"镜头 {code or '（未编号）'} 填写了 {len(actions)} 个主动作（{ '、'.join(actions) }）。一镜只能有一个主动作，请拆镜后再保存。",
                shot_code=code or None,
            )
        )

    people = _people(shot)
    refs = _ref_names(shot)
    # Map refs that are already bound to a person
    bound: set[str] = set()
    raw_refs = shot.get("ref_assets") or []
    for item in _as_list(raw_refs):
        if isinstance(item, dict):
            who = str(item.get("character") or item.get("name") or "").strip()
            path = item.get("file") or item.get("path")
            if who and path:
                bound.add(who)

    if len(people) >= 2:
        missing: list[str] = []
        for person in people:
            has = person in bound
            if not has:
                # accept a ref filename that contains the person name
                if any(person in Path(r).stem or Path(r).stem in person for r in refs):
                    has = True
            if not has:
                missing.append(_missing_ref_filename(person, shot, pack))
        if missing:
            # unique, keep order
            seen: set[str] = set()
            named: list[str] = []
            for m in missing:
                if m not in seen:
                    seen.add(m)
                    named.append(m)
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_refs",
                    format_missing(named) + f"（镜头 {code or '（未编号）'} 为多人镜头，必须每人绑一张参考图）",
                    shot_code=code or None,
                    missing_files=named,
                )
            )

    # If refs are declared, the actual files should exist when a pack root is given.
    if pack_root:
        root = Path(pack_root)
        file_missing: list[str] = []
        for item in _as_list(raw_refs):
            path = None
            if isinstance(item, dict):
                path = item.get("file") or item.get("path")
            else:
                path = item
            if not path or str(path) in {"无", "无（环境）"}:
                continue
            p = Path(str(path))
            candidates = [p, root / p, root / "refs" / p.name, root / "characters" / p.name]
            if not any(c.is_file() for c in candidates):
                file_missing.append(p.name)
        if file_missing:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_ref_files",
                    format_missing(file_missing),
                    shot_code=code or None,
                    missing_files=file_missing,
                )
            )

    is_fight = bool(shot.get("is_fight") or shot.get("fight"))
    pack_flag = True if pack is None else bool((pack.get("script") or pack).get("fight_max_two", fight_max_two))
    if fight_max_two and pack_flag and is_fight and len(people) > 2:
        issues.append(
            ValidationIssue(
                "error",
                "fight_two_person",
                f"镜头 {code or '（未编号）'} 是打架类镜头，同框人数为 {len(people)}，硬约束最多两人。请拆镜后再保存或执行。",
                shot_code=code or None,
            )
        )
    return issues


def validate_pack(
    pack: dict[str, Any],
    *,
    pack_root: str | Path | None = None,
    fight_max_two: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    script = pack.get("script") or {}
    # Style lock must live only on the script level.
    for ch in pack.get("characters") or []:
        if ch.get("style_lock") and script.get("style_lock") and ch.get("style_lock") != script.get("style_lock"):
            issues.append(
                ValidationIssue(
                    "warning",
                    "style_lock_scope",
                    f"角色「{ch.get('name') or ch.get('id')}」含有独立风格锁。全局风格锁只应写在剧本模板里，将以剧本为准。",
                )
            )
    for shot in pack.get("shots") or []:
        issues.extend(validate_shot(shot, pack=pack, pack_root=pack_root, fight_max_two=fight_max_two))
    errors = [i for i in issues if i.is_error]
    return issues
