"""Fill shot prompts from four template levels. Style lock lives ONLY on the script."""

from __future__ import annotations

from typing import Any


def style_lock_of(pack: dict[str, Any]) -> tuple[str, str]:
    script = pack.get("script") or {}
    return (script.get("style_lock") or "").strip(), (script.get("negative") or "").strip()


def filled_prompt(shot: dict[str, Any], pack: dict[str, Any]) -> str:
    """Build the executable prompt: shot body + script-level style lock + negatives.

    Character/scene 'style_lock' words are ignored for the global look; only
    the script template is appended, so later pack edits of shot body still
    inherit one shared lock.
    """
    body = (shot.get("prompt") or "").strip()
    style, negative = style_lock_of(pack)
    parts = [body] if body else []
    # Append lock only if the body does not already end with it (source packs
    # often baked the lock into the copied prompt).
    if style and style not in body:
        parts.append(style)
    if negative:
        tag = "避免：" + negative if not negative.startswith("避免") else negative
        if tag not in body and negative not in body:
            parts.append(tag)
    return "\n".join(parts).strip()


def apply_pack_to_shots(shots: list[dict[str, Any]], pack: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sh in shots:
        row = dict(sh)
        row["filled_prompt"] = filled_prompt(row, pack)
        row["template_version"] = pack.get("version") or (pack.get("script") or {}).get("version")
        out.append(row)
    return out
