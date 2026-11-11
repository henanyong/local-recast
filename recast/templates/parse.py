"""Parse storyboard text / JSON and dialogue tables into shot rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SHOT_HEAD = re.compile(
    r"^={3,}\s*(?P<code>F\d+(?:-L\d+)?)\s*(?:[|｜]\s*(?P<span>\d+\s*[-~～到至]\s*\d+\s*s?))?\s*={0,}",
    re.IGNORECASE,
)
SPAN = re.compile(r"(?P<a>\d+)\s*[-~～到至]\s*(?P<b>\d+)")
REF_LINE = re.compile(r"^参考图\s*[:：]\s*(.+)$")
ACTION = re.compile(r"只有一个动作[：:]\s*([^。.\n]+)")
DIALOGUE_HEAD = re.compile(
    r"^\[(?P<code>F\d+(?:-L\d+)?)(?P<rest>[^\]]*)\]\s*(?P<body>.*)$"
)
SPEAKER = re.compile(r"^(?P<who>.+?)(?:·(?P<style>.+?))?[：:]\s*(?P<line>.*)$")

FIGHT_WORDS = ("打", "扇", "踹", "掐", "抽", "勒", "顶", "推", "拧", "耳光", "打架", "搏")


def _read(path_or_text: str | Path) -> str:
    p = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return str(path_or_text)


def _parse_span(text: str | None) -> tuple[float | None, float | None]:
    if not text:
        return None, None
    m = SPAN.search(text)
    if not m:
        return None, None
    return float(m.group("a")), float(m.group("b"))


def _split_refs(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text or text.startswith("无"):
        return []
    parts = re.split(r"\s*[+＋、,，]+\s*", text)
    out: list[dict[str, str]] = []
    for part in parts:
        token = part.strip()
        if not token or token.startswith("无"):
            continue
        # "01-温念" / "01-温念-冷笑" / "温念.png"
        name = token
        stem = Path(token).stem
        # character name is usually after the first hyphen if prefixed with digits
        who = stem
        m = re.match(r"^\d+[-_](.+)$", stem)
        if m:
            who = re.sub(r"-(冷笑|怒斥|特写|侧面)$", "", m.group(1))
        out.append({"file": token if "." in Path(token).name else token, "character": who, "raw": token})
    return out


def _guess_shot_size(prompt: str) -> str:
    for key in ("远景", "全景", "中景", "近景", "特写", "过肩"):
        if key in (prompt or ""):
            return key
    return ""


def _is_fight(prompt: str, action: str) -> bool:
    # Only the declared primary action (not the style-lock prompt, which may
    # contain words like「窗光侧打」).
    blob = action or ""
    if not blob:
        return False
    return any(w in blob for w in FIGHT_WORDS)


def parse_storyboard(path_or_text: str | Path) -> list[dict[str, Any]]:
    """Parse the episode-1 style 分镜提示词 text, or a JSON list/object."""
    raw = _read(path_or_text).lstrip("\ufeff")
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("shots") or data.get("storyboard") or []
        return [dict(x) for x in data]

    shots: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    buf: list[str] = []
    expecting_prompt = False

    def flush() -> None:
        nonlocal current, buf, expecting_prompt
        if current is None:
            buf = []
            expecting_prompt = False
            return
        prompt = "\n".join(buf).strip()
        # drop the "完整提示词（直接复制）：" heading if it leaked in
        prompt = re.sub(r"^完整提示词[^：:]*[：:]\s*", "", prompt).strip()
        action = current.get("primary_action") or ""
        if not action:
            m = ACTION.search(prompt)
            if m:
                action = m.group(1).strip()
        start, end = current.get("start_time"), current.get("end_time")
        duration = (end - start) if start is not None and end is not None else current.get("duration_sec")
        people = []
        for ref in current.get("ref_assets") or []:
            who = ref.get("character")
            if who and who not in people:
                people.append(who)
        current.update(
            {
                "prompt": prompt,
                "primary_action": action,
                "duration_sec": duration,
                "people": people,
                "shot_size": current.get("shot_size") or _guess_shot_size(prompt),
                "is_fight": bool(current.get("is_fight")) or _is_fight(prompt, action),
            }
        )
        shots.append(current)
        current = None
        buf = []
        expecting_prompt = False

    for line in raw.splitlines():
        m = SHOT_HEAD.match(line.strip())
        if m:
            flush()
            start, end = _parse_span(m.group("span"))
            current = {
                "shot_code": m.group("code"),
                "start_time": start,
                "end_time": end,
                "ref_assets": [],
                "kind": "picture",
            }
            continue
        if current is None:
            continue
        ref_m = REF_LINE.match(line.strip())
        if ref_m:
            current["ref_assets"] = _split_refs(ref_m.group(1))
            continue
        if "完整提示词" in line:
            expecting_prompt = True
            # remainder after colon may be the prompt itself
            if "：" in line or ":" in line:
                rest = re.split(r"[：:]", line, 1)[-1].strip()
                if rest:
                    buf.append(rest)
            continue
        if line.strip().startswith("========"):
            continue
        buf.append(line)
    flush()

    for i, sh in enumerate(shots):
        sh["index_no"] = i
        # A "standing / 定格 / 环境" shot with 3 people is usually not a fight close-up.
        if sh.get("is_fight") and len(sh.get("people") or []) > 2:
            # keep fight only when the action itself is a hit
            action = sh.get("primary_action") or ""
            if not any(w in action for w in ("打", "扇", "踹", "掐", "抽", "勒", "顶", "拧", "耳光")):
                sh["is_fight"] = False
    return shots


def parse_dialogue_table(path_or_text: str | Path) -> list[dict[str, Any]]:
    """Parse the episode-1 style 对白表, or a JSON list."""
    raw = _read(path_or_text).lstrip("\ufeff")
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("dialogue") or data.get("lines") or []
        return [dict(x) for x in data]

    lines: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal current, body
        if current is None:
            body = []
            return
        text = "\n".join(body).strip()
        if current.get("text"):
            text = (str(current["text"]) + ("\n" + text if text else "")).strip()
        current["text"] = text
        current["dialogue"] = text
        lines.append(current)
        current = None
        body = []

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = DIALOGUE_HEAD.match(line)
        if m:
            flush()
            code = m.group("code")
            rest = (m.group("rest") or "").strip()
            body_part = (m.group("body") or "").strip()
            start, end = _parse_span(rest)
            kind = "dialogue"
            speaker = ""
            style = ""
            text = ""
            if "旁白" in rest or (body_part.startswith("旁白")):
                kind = "narration"
            if "无对白" in body_part or "无对白" in rest:
                kind = "none"
            sp = SPEAKER.match(body_part)
            if sp and kind != "none":
                speaker = sp.group("who").strip()
                style = (sp.group("style") or "").strip()
                text = (sp.group("line") or "").strip()
            elif kind == "narration" and not text:
                # body may be empty; text comes on following lines
                text = ""
            current = {
                "shot_code": code,
                "kind": kind,
                "speaker": speaker,
                "style": style,
                "start_time": start,
                "end_time": end,
                "text": text,
                "raw_head": line,
            }
            continue
        if current is None:
            continue
        body.append(line)
    flush()
    return lines


def attach_dialogue(
    shots: list[dict[str, Any]],
    dialogue: list[dict[str, Any]],
    *,
    narration_segments: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Bind dialogue/narration lines onto shots.

    Narration spanning a section (e.g. F1 0-15s) is attached to the opening
    segment, not copied onto every child shot. Later shots only receive dialogue.
    """
    from typing import Iterable as _Iter  # noqa: F401

    nar_segs = set(narration_segments or ["F1"])
    by_code: dict[str, dict[str, Any]] = {}
    section_lines: dict[str, list[dict[str, Any]]] = {}
    for line in dialogue:
        code = str(line.get("shot_code") or "")
        if re.fullmatch(r"F\d+", code):
            section_lines.setdefault(code, []).append(line)
        else:
            by_code[code] = line

    for sh in shots:
        code = str(sh.get("shot_code") or "")
        section = code.split("-")[0] if code else ""
        line = by_code.get(code)
        if line:
            sh["dialogue"] = line.get("dialogue") or line.get("text") or ""
            sh["speaker"] = line.get("speaker") or ""
            sh["kind"] = line.get("kind") or "dialogue"
            if line.get("start_time") is not None and sh.get("start_time") is None:
                sh["start_time"] = line["start_time"]
            if line.get("end_time") is not None and sh.get("end_time") is None:
                sh["end_time"] = line["end_time"]
            continue
        # section-level narration: only the first shot of that section
        if section in section_lines:
            is_first = all(
                not str(other.get("shot_code") or "").startswith(section + "-")
                or str(other.get("shot_code") or "") >= code
                for other in shots
                if other is not sh
            )
            # simpler: first shot whose code starts with section-
            pass
        sh.setdefault("dialogue", "")
        sh.setdefault("speaker", "")
        sh.setdefault("kind", "picture")

    # Attach section narration to the first shot of that section only.
    first_of: dict[str, dict[str, Any]] = {}
    for sh in shots:
        section = str(sh.get("shot_code") or "").split("-")[0]
        first_of.setdefault(section, sh)
    for section, lines in section_lines.items():
        target = first_of.get(section)
        if not target:
            continue
        # If the pack says narration only in opening segments, still attach
        # F1 narration to the first F1 shot (covers 0-15s across F1-L1..L3).
        text = "\n".join((ln.get("dialogue") or ln.get("text") or "") for ln in lines).strip()
        kind = lines[0].get("kind") or "narration"
        if kind == "none":
            target.setdefault("dialogue", "")
            continue
        if kind == "narration" and nar_segs and section not in nar_segs:
            continue
        target["dialogue"] = text
        target["kind"] = kind
        target["speaker"] = lines[0].get("speaker") or ("旁白" if kind == "narration" else "")
        # narration timing may cover the whole section
        if lines[0].get("start_time") is not None:
            target["narration_start"] = lines[0]["start_time"]
        if lines[0].get("end_time") is not None:
            target["narration_end"] = lines[0]["end_time"]
    return shots


# leftover import used in type hint above — keep Iterable imported at top
from typing import Iterable  # noqa: E402  (used by attach_dialogue signature)
