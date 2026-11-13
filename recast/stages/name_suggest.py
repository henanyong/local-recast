"""Name auto-suggestion: match face clusters to person names.

Names come from two sources only:

- the dialogue table (``speaker`` per shot, with a time span), and
- subtitle OCR lines that start with a ``名字：`` style prefix.

A name is suggested for a cluster only when the evidence points at exactly
one cluster *and* exactly one name (mutual-best, like face matching). When
the evidence is ambiguous the suggestion is left empty — never guessed.
Suggestions are suggestions only: they are written to meta.json
``name_suggestion`` for the user to confirm in the library UI. Nothing here
renames anyone.

Matching uses time overlap: a cluster's span is [first_frame, last_frame]
over its tracks, converted with fps; a dialogue shot or OCR hit at time t
counts for the clusters visible at that time.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

# "温念：..." at the start of an OCR line, same shape as dialogue tables.
_NAME_PREFIX = re.compile(r"^\s*(?P<who>[\u4e00-\u9fa5]{2,4})\s*[：:]\s*(?P<line>.*)$")
# "温念说/道/喊/问 ..." at the start of a line.
_NAME_VERB = re.compile(r"^\s*(?P<who>[\u4e00-\u9fa5]{2,4})\s*(?:说|道|喊|问|笑|哭)")

_EXCLUDED = {"旁白", "无对白", "画外音", "众人", "大家", "路人", "群众"}

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def _clean_name(raw: str) -> str | None:
    name = (raw or "").strip()
    if not (2 <= len(name) <= 4):
        return None
    if not all("\u4e00" <= ch <= "\u9fa5" for ch in name):
        return None
    if name in _EXCLUDED:
        return None
    return name


def extract_names_from_text(text: str) -> list[str]:
    """Person names mentioned at the start of OCR/subtitle lines."""
    found: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name = None
        m = _NAME_PREFIX.match(line)
        if m:
            name = _clean_name(m.group("who"))
        else:
            m = _NAME_VERB.match(line)
            if m:
                name = _clean_name(m.group("who"))
        if name and name not in found:
            found.append(name)
    return found


def _cluster_spans(
    persons: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    fps: float,
) -> dict[str, tuple[float, float]]:
    """auto_name -> (start_sec, end_sec) from its tracks' frame ranges."""
    by_id = {int(t.get("track_id")): t for t in tracks or []}
    spans: dict[str, tuple[float, float]] = {}
    for person in persons or []:
        auto = str(person.get("auto_name") or person.get("name") or "")
        if not auto:
            continue
        firsts: list[int] = []
        lasts: list[int] = []
        for tid in person.get("track_ids") or []:
            track = by_id.get(int(tid))
            if track is None:
                continue
            firsts.append(int(track.get("first_frame") or 0))
            lasts.append(int(track.get("last_frame") or firsts[-1]))
        if firsts and lasts:
            spans[auto] = (min(firsts) / fps, max(lasts) / fps)
        else:
            first = int(person.get("first_frame") or 0)
            spans[auto] = (first / fps, first / fps)
    return spans


def _visible_at(spans: dict[str, tuple[float, float]], when: float) -> list[str]:
    return [auto for auto, (start, end) in spans.items() if start <= when <= end]


def _overlapping(spans: dict[str, tuple[float, float]], start: float, end: float) -> list[str]:
    return [
        auto
        for auto, (c_start, c_end) in spans.items()
        if c_start <= end and start <= c_end
    ]


def _bump(confidence: str) -> str:
    order = ["low", "medium", "high"]
    idx = min(order.index(confidence) + 1, len(order) - 1)
    return order[idx]


def suggest_names(
    persons: list[dict[str, Any]],
    tracks: list[dict[str, Any]] | None = None,
    *,
    dialogue_shots: list[dict[str, Any]] | None = None,
    ocr_hits: list[dict[str, Any]] | None = None,
    fps: float = 25.0,
) -> list[dict[str, Any]]:
    """Suggest a name per unnamed cluster.

    ``persons`` are identity ``persons`` (only those still carrying their
    auto_name are considered). Returns one entry per cluster that got a
    suggestion: {cluster_id, auto_name, suggested_name, source, confidence,
    evidence}. Clusters with no clean evidence get no entry.
    """
    fps = float(fps) if fps else 25.0
    unnamed = [
        p for p in (persons or []) if str(p.get("name") or p.get("auto_name")) == str(p.get("auto_name"))
    ]
    if not unnamed:
        return []
    spans = _cluster_spans(unnamed, tracks or [], fps)

    # speaker -> {cluster auto_name: overlapping shot count}
    dialogue_votes: dict[str, dict[str, int]] = {}
    all_names: set[str] = set()
    for shot in dialogue_shots or []:
        speaker = _clean_name(str(shot.get("speaker") or ""))
        if not speaker:
            continue
        all_names.add(speaker)
        start = float(shot.get("start_time") or 0.0)
        end = float(shot.get("end_time") or start)
        for auto in _overlapping(spans, start, end):
            dialogue_votes.setdefault(speaker, {}).setdefault(auto, 0)
            dialogue_votes[speaker][auto] += 1

    # name -> {cluster auto_name: hit count} from OCR lines.
    ocr_votes: dict[str, dict[str, int]] = {}
    for hit in ocr_hits or []:
        when = float(hit.get("time") or hit.get("start_time") or 0.0)
        for name in extract_names_from_text(str(hit.get("text") or "")):
            all_names.add(name)
            for auto in _visible_at(spans, when):
                ocr_votes.setdefault(name, {}).setdefault(auto, 0)
                ocr_votes[name][auto] += 1

    def _mutual_best(votes: dict[str, dict[str, int]]) -> dict[str, tuple[str, int]]:
        """name -> (auto_name, votes) when the link is mutual-best, else skip."""
        cluster_to_names: dict[str, list[str]] = {}
        for name, per_cluster in votes.items():
            for auto in per_cluster:
                cluster_to_names.setdefault(auto, []).append(name)
        out: dict[str, tuple[str, int]] = {}
        for name, per_cluster in votes.items():
            if len(per_cluster) != 1:
                continue
            auto, count = next(iter(per_cluster.items()))
            if len(cluster_to_names.get(auto, [])) != 1:
                continue
            out[name] = (auto, count)
        return out

    dialogue_best = _mutual_best(dialogue_votes)
    ocr_best = _mutual_best(ocr_votes)

    by_auto: dict[str, dict[str, Any]] = {}
    for name, (auto, count) in dialogue_best.items():
        by_auto[auto] = {
            "suggested_name": name,
            "source": "对白表",
            "confidence": "high" if count >= 2 else "medium",
            "evidence": {"dialogue_shots": count},
        }
    for name, (auto, count) in ocr_best.items():
        existing = by_auto.get(auto)
        if existing is not None:
            if existing["suggested_name"] == name:
                existing["confidence"] = _bump(existing["confidence"])
                existing["evidence"]["ocr_hits"] = count
            continue
        by_auto[auto] = {
            "suggested_name": name,
            "source": "字幕OCR",
            "confidence": "medium" if count >= 2 else "low",
            "evidence": {"ocr_hits": count},
        }

    # Last resort: exactly one unnamed cluster and exactly one candidate name
    # in the whole material (even if its shots never overlapped this cluster).
    if not by_auto:
        if len(unnamed) == 1 and len(all_names) == 1:
            auto = str(unnamed[0].get("auto_name"))
            name = next(iter(all_names))
            source = "对白表" if name in dialogue_votes or any(
                _clean_name(str(s.get("speaker") or "")) == name for s in (dialogue_shots or [])
            ) else "字幕OCR"
            by_auto[auto] = {
                "suggested_name": name,
                "source": f"唯一候选（{source}）",
                "confidence": "low",
                "evidence": {"single_candidate": True},
            }

    out: list[dict[str, Any]] = []
    for person in unnamed:
        auto = str(person.get("auto_name"))
        item = by_auto.get(auto)
        if item is None:
            continue
        out.append(
            {
                "cluster_id": person.get("cluster_id"),
                "auto_name": auto,
                "suggested_name": item["suggested_name"],
                "source": item["source"],
                "confidence": item["confidence"],
                "evidence": item["evidence"],
            }
        )
    log.info("名字自动建议完成: %s 个聚类得到建议，共 %s 个未命名聚类。", len(out), len(unnamed))
    return out


def write_name_suggestions(
    output_root: str | Path,
    suggestions: list[dict[str, Any]],
) -> list[str]:
    """Write suggestions into each folder's meta.json ``name_suggestion``.

    Only folders whose name is the cluster auto_name are touched. The folder
    is not renamed; the user confirms the suggestion in the library UI.
    """
    from recast.core.characters import character_folder, load_meta, save_meta

    written: list[str] = []
    for item in suggestions or []:
        auto = str(item.get("auto_name") or "")
        if not auto:
            continue
        folder = character_folder(output_root, auto)
        if not folder.is_dir():
            continue
        meta = load_meta(folder) or {"name": auto}
        meta["name_suggestion"] = {
            "name": item.get("suggested_name"),
            "source": item.get("source"),
            "confidence": item.get("confidence"),
        }
        save_meta(folder, meta)
        written.append(str(folder))
    return written
