"""Scan a local drama root into dramas + episode videos (no side effects until committed)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from recast.core.constants import (
    REF_DIR_NAMES,
    SKIP_DIR_NAMES,
    VIDEO_DIR_NAMES,
    VIDEO_EXTS,
)
from recast.core.roster import Roster, find_roster_file, parse_roster


@dataclass
class EpisodeHit:
    name: str
    video_path: Path


@dataclass
class DramaHit:
    origin_name: str
    root: Path
    episodes: list[EpisodeHit] = field(default_factory=list)
    roster_path: Path | None = None
    ref_dir: Path | None = None
    roster: Roster = field(default_factory=lambda: Roster(path=None, entries=[]))

    @property
    def episode_count(self) -> int:
        return len(self.episodes)


def is_skipped_dir(path: Path) -> bool:
    return path.name in SKIP_DIR_NAMES or path.name.startswith(".")


_NUM = re.compile(r"(\d+)")


def natural_sort_key(path: Path) -> tuple:
    """Order 第2集 before 第10集 by comparing numeric segments as integers."""
    pieces: list[tuple] = []
    for part in _NUM.split(path.as_posix().lower()):
        if part.isdigit():
            pieces.append((0, int(part)))
        else:
            pieces.append((1, part))
    return tuple(pieces)


def list_videos(folder: Path, *, recursive: bool = False) -> list[Path]:
    if not folder.exists():
        return []
    if folder.is_file():
        return [folder] if folder.suffix.lower() in VIDEO_EXTS else []
    out: list[Path] = []
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if any(part in SKIP_DIR_NAMES or part in REF_DIR_NAMES for part in p.parts):
            continue
        out.append(p)
    return sorted(out, key=natural_sort_key)


def find_ref_dir(folder: Path) -> Path | None:
    for name in REF_DIR_NAMES:
        cand = folder / name
        if cand.is_dir():
            return cand
    return None


def find_video_dir(folder: Path) -> Path | None:
    for name in VIDEO_DIR_NAMES:
        cand = folder / name
        if cand.is_dir():
            return cand
    return None


def _episodes_in(folder: Path, *, nested: bool = False) -> list[EpisodeHit]:
    """Videos in this folder, its 视频/ directory, and optionally season subfolders.

    Nested mode is for one drama (第一季 / 第二季). The scan root stays shallow so
    sibling shows are not merged into a single drama. Reference-image folders and
    tool directories are skipped.
    """
    found: list[Path] = []
    video_dir = find_video_dir(folder)
    if video_dir is not None:
        found.extend(list_videos(video_dir, recursive=True))
    found.extend(p for p in list_videos(folder, recursive=False) if p not in found)
    if nested:
        try:
            children = list(folder.iterdir())
        except OSError:
            children = []
        for sub in children:
            if not sub.is_dir() or is_skipped_dir(sub):
                continue
            if sub.name in REF_DIR_NAMES or sub.name in VIDEO_DIR_NAMES:
                continue
            for video in list_videos(sub, recursive=True):
                if video not in found:
                    found.append(video)
    found = sorted(found, key=natural_sort_key)
    return [EpisodeHit(name=p.stem, video_path=p) for p in found]


def _as_drama(folder: Path, episodes: list[EpisodeHit] | None = None) -> DramaHit | None:
    eps = episodes if episodes is not None else _episodes_in(folder, nested=True)
    if not eps:
        return None
    roster_path = find_roster_file(folder)
    roster = parse_roster(roster_path) if roster_path else Roster(path=None, entries=[])
    return DramaHit(
        origin_name=folder.name,
        root=folder.resolve(),
        episodes=eps,
        roster_path=roster_path,
        ref_dir=find_ref_dir(folder),
        roster=roster,
    )


def scan_root(root: str | Path) -> list[DramaHit]:
    """Return dramas found under a user-chosen root.

    Rules:
    - A single video file → one drama with one episode.
    - Videos sitting directly in the root → one drama named after the root.
    - Each immediate subfolder that contains videos → its own drama.
    - A mix of both is allowed.
    """
    target = Path(root).expanduser()
    if target.is_file() and target.suffix.lower() in VIDEO_EXTS:
        hit = DramaHit(
            origin_name=target.stem,
            root=target.parent.resolve(),
            episodes=[EpisodeHit(name=target.stem, video_path=target.resolve())],
            roster_path=find_roster_file(target.parent),
            ref_dir=find_ref_dir(target.parent),
        )
        if hit.roster_path:
            hit.roster = parse_roster(hit.roster_path)
        return [hit]
    if not target.exists():
        raise FileNotFoundError(f"找不到目录: {target}")
    folder = target.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"找不到目录: {folder}")

    dramas: list[DramaHit] = []
    seen_roots: set[str] = set()

    top_eps = _episodes_in(folder)
    if top_eps:
        hit = _as_drama(folder, top_eps)
        if hit:
            dramas.append(hit)
            seen_roots.add(str(hit.root))

    for sub in sorted([p for p in folder.iterdir() if p.is_dir() and not is_skipped_dir(p)], key=lambda p: p.name.lower()):
        if sub.name in REF_DIR_NAMES or sub.name in VIDEO_DIR_NAMES:
            continue
        hit = _as_drama(sub)
        if hit and str(hit.root) not in seen_roots:
            dramas.append(hit)
            seen_roots.add(str(hit.root))

    return dramas
