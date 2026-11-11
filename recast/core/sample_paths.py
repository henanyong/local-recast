"""Safe deletion of generated face-sample files.

Only image files that live under a generated sample directory inside the
configured output tree may be removed. User-picked reference photos, videos,
model weights, virtualenvs, and anything outside that tree are refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recast.core.constants import IMAGE_EXTS, VIDEO_EXTS

SAMPLE_DIR_NAMES = {"face_samples"}
_BLOCKED_DIR_NAMES = {"models", ".venv", "venv", ".git"}


class SamplePathError(ValueError):
    """Path is not a generated face sample inside the output tree."""


@dataclass(frozen=True)
class SampleDecision:
    """What a caller may do with one stored sample path.

    ``action`` is ``delete`` (regular image inside a sample dir), ``missing``
    (nothing on disk), or ``skip`` (present, but not safe to delete).
    """

    action: str
    path: Path | None
    reason: str = ""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _output_root(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve()


def _blocked(parts: tuple[str, ...]) -> bool:
    return any(part in _BLOCKED_DIR_NAMES or part in {".", ".."} for part in parts)


def _inside_sample_dir(parts: tuple[str, ...]) -> bool:
    """True when ``parts`` is the sample dir itself or a path inside it."""
    return any(part in SAMPLE_DIR_NAMES for part in parts)


def plan_sample_removal(path: str | Path, output_root: str | Path) -> SampleDecision:
    """Classify ``path`` without deleting it.

    A symlink is never a deletion target, even when it sits inside
    ``face_samples``. Resolving it and unlinking the target could remove a
    reference photo that lives outside the output tree.
    """
    root = _output_root(output_root)
    raw = Path(path).expanduser()
    if not str(path).strip():
        return SampleDecision("missing", None, "空路径")
    try:
        if raw.is_symlink():
            return SampleDecision("skip", raw, f"拒绝删除符号链接：{raw}")
        if not raw.exists():
            return SampleDecision("missing", raw, "文件已不存在")
        resolved = raw.resolve()
    except OSError:
        return SampleDecision("skip", raw, f"无法解析路径：{raw}")
    if resolved.is_symlink():
        return SampleDecision("skip", resolved, f"拒绝删除符号链接：{resolved}")
    if resolved == root or not _is_relative_to(resolved, root):
        return SampleDecision("skip", resolved, f"拒绝删除输出目录以外的路径：{resolved}")
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        return SampleDecision("skip", resolved, f"拒绝删除输出目录以外的路径：{resolved}")
    if _blocked(parts) or (parts and parts[0] == "l2"):
        return SampleDecision("skip", resolved, f"拒绝删除该路径：{resolved}")
    if not resolved.is_file():
        return SampleDecision("skip", resolved, f"拒绝删除目录：{resolved}")
    if "face_samples" not in parts[:-1]:
        return SampleDecision("skip", resolved, f"拒绝删除非生成样本目录中的文件：{resolved}")
    suffix = resolved.suffix.lower()
    if suffix in VIDEO_EXTS or suffix not in IMAGE_EXTS:
        return SampleDecision("skip", resolved, f"拒绝删除非人脸样本图片：{resolved}")
    return SampleDecision("delete", resolved, "")


def require_generated_sample_file(path: str | Path, output_root: str | Path) -> Path:
    """Return the image file to delete, or raise if the path is not allowed."""
    decision = plan_sample_removal(path, output_root)
    if decision.action != "delete" or decision.path is None:
        raise SamplePathError(decision.reason or f"拒绝删除该路径：{path}")
    return decision.path


def prune_empty_sample_dirs(start: str | Path, output_root: str | Path) -> list[Path]:
    """Remove empty sample directories, and stop above ``face_samples``.

    Parents such as the workbench job folder, the output root, and any
    directory that is not a generated sample dir are left in place.
    """
    root = _output_root(output_root)
    current = Path(start).expanduser()
    removed: list[Path] = []
    # A missing sample file's parent may already be gone. Climb to the nearest
    # existing directory, and stop if that climb leaves the sample tree.
    while not current.exists():
        parent = current.parent
        if parent == current:
            return removed
        try:
            if parent.is_symlink():
                return removed
        except OSError:
            return removed
        current = parent
        try:
            if not current.exists():
                continue
            if current.is_symlink():
                return removed
            resolved = current.resolve()
        except OSError:
            return removed
        if resolved == root or not _is_relative_to(resolved, root):
            return removed
        parts = resolved.relative_to(root).parts
        if _blocked(parts) or not _inside_sample_dir(parts):
            return removed
    while True:
        try:
            if current.is_symlink() or not current.exists() or not current.is_dir():
                break
            resolved = current.resolve()
        except OSError:
            break
        if resolved == root or not _is_relative_to(resolved, root):
            break
        parts = resolved.relative_to(root).parts
        if _blocked(parts) or not _inside_sample_dir(parts):
            break
        try:
            if any(current.iterdir()):
                break
        except OSError:
            break
        try:
            current.rmdir()
        except OSError:
            break
        removed.append(resolved)
        if resolved.name in SAMPLE_DIR_NAMES:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return removed


def remove_generated_sample_file(path: str | Path, output_root: str | Path) -> Path:
    """Delete one generated sample image. Refuses anything outside that set."""
    dest = require_generated_sample_file(path, output_root)
    if dest.is_symlink() or not dest.is_file():
        raise SamplePathError(f"拒绝删除该路径：{dest}")
    dest.unlink()
    prune_empty_sample_dirs(dest.parent, output_root)
    return dest
