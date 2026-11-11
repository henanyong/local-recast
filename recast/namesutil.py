"""Parse a roster file or --names CLI string into track/person labels."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


def parse_names(value: str | Path | Mapping[str, str] | None) -> dict[str, str]:
    """Return a mapping that accepts both track ids ('1') and auto names ('Person_1')."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(v).strip()}
    text = str(value).strip()
    if not text:
        return {}
    path = Path(text)
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"人名文件必须是 YAML 字典: {path}")
        return {str(k).strip(): str(v).strip() for k, v in loaded.items() if v is not None}

    mapping: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, name = part.split("=", 1)
        elif ":" in part:
            key, name = part.split(":", 1)
        else:
            raise ValueError(f"无法解析人名项 '{part}'，请用 1=张三 或 Person_1=张三")
        mapping[key.strip()] = name.strip()
    return mapping


def apply_names(
    person_name: str,
    track_id: int,
    roster: Mapping[str, str],
) -> str:
    if not roster:
        return person_name
    for key in (str(track_id), person_name, person_name.lower(), f"track_{track_id}"):
        if key in roster and roster[key]:
            return roster[key]
    return person_name
