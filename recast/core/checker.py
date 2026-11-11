"""Material-completeness checker. Missing items are always named, never「资料不全」."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from recast.core.constants import IMAGE_EXTS
from recast.core.roster import Roster, RosterEntry, iter_image_files, parse_roster
from recast.core.scanner import DramaHit, find_ref_dir
from recast.core.state_machine import format_missing


@dataclass
class MissingItem:
    name: str
    reason: str = ""


@dataclass
class CheckResult:
    ok: bool
    missing: list[MissingItem] = field(default_factory=list)
    message: str = ""
    ref_dir: Path | None = None
    roster_path: Path | None = None
    found_faces: list[str] = field(default_factory=list)

    @property
    def missing_names(self) -> list[str]:
        return [m.name for m in self.missing]


def _index_images(folder: Path | None) -> dict[str, Path]:
    if folder is None or not folder.is_dir():
        return {}
    index: dict[str, Path] = {}
    for img in iter_image_files(folder):
        index[img.name.lower()] = img
        index[img.stem.lower()] = img
    # also search one level of nested folders
    for sub in folder.iterdir():
        if sub.is_dir():
            for img in iter_image_files(sub):
                index.setdefault(img.name.lower(), img)
                index.setdefault(img.stem.lower(), img)
    return index


def _face_found(entry: RosterEntry, index: dict[str, Path]) -> Path | None:
    for name in entry.implied_face_names:
        hit = index.get(name.lower()) or index.get(Path(name).stem.lower())
        if hit is not None:
            return hit
    return None


def check_drama(
    drama: DramaHit | Path,
    *,
    roster: Roster | None = None,
    require_faces: bool = True,
) -> CheckResult:
    """Check a drama folder / hit.

    If a roster lists characters, each missing reference file is named
    (e.g. 「缺少 5.png、6.png」). When there is no roster, ``require_faces``
    only reports a missing 参考图 directory or empty one if replace is planned
    and the caller asked for it; extract-only checks can pass ``require_faces=False``.
    """
    if isinstance(drama, Path):
        root = drama
        ref_dir = find_ref_dir(root)
        roster_path = None
        roster_obj = roster or Roster(path=None, entries=[])
    else:
        root = drama.root
        ref_dir = drama.ref_dir or find_ref_dir(root)
        roster_path = drama.roster_path
        roster_obj = roster or drama.roster

    missing: list[MissingItem] = []
    index = _index_images(ref_dir)
    # also allow reference images sitting next to the videos
    index.update(_index_images(root))

    found: list[str] = []
    if roster_obj.entries:
        for entry, preferred in roster_obj.expected_face_files():
            hit = _face_found(entry, index)
            if hit is None:
                missing.append(MissingItem(name=preferred, reason=f"角色「{entry.name}」缺少参考脸"))
            else:
                found.append(str(hit))
        # costume / voice if explicitly named
        for entry in roster_obj.entries:
            if entry.costume:
                name = Path(entry.costume).name
                if name.lower() not in index and not (root / entry.costume).is_file():
                    missing.append(MissingItem(name=name, reason=f"角色「{entry.name}」缺少服装参考"))
            if entry.voice:
                voice_path = Path(entry.voice)
                candidates = [root / voice_path, (ref_dir / voice_path.name) if ref_dir else None]
                if not any(c is not None and c.is_file() for c in candidates):
                    missing.append(MissingItem(name=voice_path.name, reason=f"角色「{entry.name}」缺少声音参考"))
    elif require_faces:
        if not index:
            # No roster and no images: name the conventional files / folder so the
            # user knows exactly what to add, never a vague「资料不全」.
            if ref_dir is None:
                missing.append(MissingItem(name="参考图/", reason="未找到参考图目录，请在剧目文件夹下创建「参考图」并放入已授权角色的人脸图"))
            else:
                missing.append(MissingItem(name="参考脸图片", reason="「参考图」目录是空的，请放入已授权角色的人脸图（如 5.png）"))

    # De-dupe names, keep order
    uniq: list[MissingItem] = []
    seen: set[str] = set()
    for item in missing:
        if item.name in seen:
            continue
        seen.add(item.name)
        uniq.append(item)

    message = format_missing(i.name for i in uniq)
    return CheckResult(
        ok=not uniq,
        missing=uniq,
        message=message,
        ref_dir=ref_dir,
        roster_path=roster_path if isinstance(drama, DramaHit) else None,
        found_faces=found,
    )


def check_replace_ready(
    *,
    authorized_faces: list[str],
    declared_missing: list[str] | None = None,
) -> CheckResult:
    """Used right before 04 人物替换: need at least one authorized face asset."""
    missing: list[MissingItem] = []
    for name in declared_missing or []:
        missing.append(MissingItem(name=name, reason="名单中的参考文件仍缺失"))
    usable = [p for p in authorized_faces if p and Path(p).is_file()]
    if not usable:
        missing.append(
            MissingItem(
                name="已授权角色的参考脸",
                reason="人物替换仅能用于已勾选授权（自有拍摄 / 已授权 / 自有生成角色）且有参考脸的人物",
            )
        )
    # unique
    uniq: list[MissingItem] = []
    seen: set[str] = set()
    for item in missing:
        if item.name in seen:
            continue
        seen.add(item.name)
        uniq.append(item)
    return CheckResult(ok=not uniq, missing=uniq, message=format_missing(i.name for i in uniq), found_faces=usable)


def load_roster_from_path(path: str | Path | None) -> Roster:
    return parse_roster(path)
