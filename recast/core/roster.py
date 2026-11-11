"""Parse character rosters (名单.yaml and friends) into structured entries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from recast.core.constants import IMAGE_EXTS, ROSTER_FILENAMES


@dataclass
class RosterEntry:
    key: str
    name: str
    face: str | None = None
    costume: str | None = None
    voice: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def implied_face_names(self) -> list[str]:
        """Filenames we will look for if a face path was not given explicitly."""
        names: list[str] = []
        if self.face:
            names.append(Path(self.face).name)
        stem_candidates = [self.key, self.name]
        for stem in stem_candidates:
            stem = str(stem).strip()
            if not stem:
                continue
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                names.append(f"{stem}{ext}")
        # de-dupe, keep order
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            key = n.lower()
            if key not in seen:
                seen.add(key)
                out.append(n)
        return out


@dataclass
class Roster:
    path: Path | None
    entries: list[RosterEntry] = field(default_factory=list)

    def by_key(self) -> dict[str, RosterEntry]:
        return {e.key: e for e in self.entries}

    def name_map(self) -> dict[str, str]:
        """Compatibility mapping for --names / identity.apply_names."""
        out: dict[str, str] = {}
        for e in self.entries:
            if e.name:
                out[e.key] = e.name
                out[e.name] = e.name
        return out

    def expected_face_files(self) -> list[tuple[RosterEntry, str]]:
        """(entry, preferred filename) for material checking."""
        items: list[tuple[RosterEntry, str]] = []
        for e in self.entries:
            if e.face:
                items.append((e, Path(e.face).name))
            else:
                # Prefer {key}.png as the canonical missing name (e.g. 5.png)
                items.append((e, f"{e.key}.png" if _looks_like_id(e.key) else f"{e.name or e.key}.png"))
        return items


def _looks_like_id(key: str) -> bool:
    text = str(key).strip()
    return text.isdigit() or (len(text) <= 8 and text.replace("-", "").isalnum())


def find_roster_file(folder: Path) -> Path | None:
    for name in ROSTER_FILENAMES:
        cand = folder / name
        if cand.is_file():
            return cand
    # also accept a roster sitting in 参考图/
    for sub in folder.iterdir() if folder.is_dir() else []:
        if sub.is_dir():
            for name in ROSTER_FILENAMES:
                cand = sub / name
                if cand.is_file():
                    return cand
    # any yaml with 'name'/'face' maybe too loose; skip
    return None


def parse_roster(source: str | Path | dict | list | None) -> Roster:
    if source is None:
        return Roster(path=None, entries=[])
    if isinstance(source, (dict, list)):
        return Roster(path=None, entries=_from_data(source))
    path = Path(source)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text or "null")
        else:
            data = yaml.safe_load(text) or {}
        return Roster(path=path, entries=_from_data(data))
    # inline "1=张三,2=李四"
    entries: list[RosterEntry] = []
    for part in str(source).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, name = part.split("=", 1)
        elif ":" in part:
            key, name = part.split(":", 1)
        else:
            continue
        entries.append(RosterEntry(key=key.strip(), name=name.strip()))
    return Roster(path=None, entries=entries)


def _from_data(data: Any) -> list[RosterEntry]:
    entries: list[RosterEntry] = []
    if data is None:
        return entries
    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            entries.extend(_entry_from_item(str(i), item))
        return entries
    if not isinstance(data, dict):
        return entries
    # {"characters": [...]} or {"人物": [...]}
    for key in ("characters", "人物", "cast", "names", "roster"):
        if key in data and isinstance(data[key], (list, dict)):
            return _from_data(data[key])
    for key, value in data.items():
        entries.extend(_entry_from_item(str(key), value))
    return entries


def _entry_from_item(key: str, value: Any) -> list[RosterEntry]:
    if value is None:
        return [RosterEntry(key=key, name=key)]
    if isinstance(value, str):
        return [RosterEntry(key=key, name=value)]
    if not isinstance(value, dict):
        return [RosterEntry(key=key, name=str(value))]
    name = str(value.get("name") or value.get("名称") or value.get("display_name") or key)
    face = value.get("face") or value.get("脸") or value.get("ref") or value.get("image")
    costume = value.get("costume") or value.get("服装") or value.get("clothes")
    voice = value.get("voice") or value.get("声音")
    ident = str(value.get("id") or value.get("key") or key)
    return [
        RosterEntry(
            key=ident,
            name=name,
            face=str(face) if face else None,
            costume=str(costume) if costume else None,
            voice=str(voice) if voice else None,
            extra=dict(value),
        )
    ]


def iter_image_files(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
