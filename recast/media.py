"""Image/video path helpers, including Windows non-ASCII path workarounds."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("recast")

SOURCE_RECORD = "input.src"
SOURCE_META = "source_meta.json"


def imread(path: str | Path) -> np.ndarray | None:
    """Read a BGR image; falls back to np.fromfile for Unicode Windows paths."""
    import cv2
    import numpy as np

    src = Path(path)
    img = cv2.imread(str(src))
    if img is not None:
        return img
    try:
        data = np.fromfile(str(src), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)  # noqa: imported above


def open_capture(path: str | Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(str(Path(path).resolve()))
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频: {path}。若路径含中文或空格，流水线会尝试在工作目录创建 ASCII 别名；"
            "仍失败时请把视频拷到纯英文路径后重试。"
        )
    return cap


def is_ascii_path(path: str | Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_stem(path: str | Path, *, length: int = 12) -> str:
    """Return an ASCII directory/file stem; hash the path when the name is not ASCII."""
    src = Path(path)
    stem = src.stem
    if stem and stem.isascii() and all(c.isalnum() or c in "-_." for c in stem):
        return stem
    try:
        key = str(src.resolve())
    except OSError:
        key = str(src)
    return hashlib.md5(key.encode("utf-8", "surrogatepass")).hexdigest()[:length]


def workdir_leaf(path: str | Path, idx: int) -> str:
    """ASCII work-directory name unique among sibling clips, even when stems collide."""
    return f"{int(idx):03d}_{ascii_stem(path)}"


def same_media_file(a: str | Path, b: str | Path) -> bool:
    """True when both paths exist and refer to the same file, including hard links."""
    left, right = Path(a), Path(b)
    try:
        if left.exists() and right.exists() and os.path.samefile(left, right):
            return True
    except OSError:
        pass
    try:
        return left.resolve() == right.resolve() and left.exists()
    except OSError:
        return False


def write_source_record(workdir: Path, src: str | Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    try:
        resolved = str(src.resolve())
    except OSError:
        resolved = str(src)
    (workdir / SOURCE_RECORD).write_text(resolved, encoding="utf-8")
    payload = {
        "original_path": resolved,
        "original_name": src.name,
        "display_name": src.stem,
    }
    (workdir / SOURCE_META).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_source_record(workdir: Path) -> Path | None:
    record = workdir / SOURCE_RECORD
    if not record.is_file():
        return None
    text = record.read_text(encoding="utf-8").strip()
    return Path(text) if text else None


def load_source_meta(workdir: Path) -> dict:
    path = workdir / SOURCE_META
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def alias_sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + ".src")


def write_alias_source(dest: Path, src: str | Path) -> None:
    try:
        resolved = str(Path(src).resolve())
    except OSError:
        resolved = str(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    alias_sidecar(dest).write_text(resolved, encoding="utf-8")
    write_source_record(dest.parent, src)


def read_alias_source(dest: Path) -> Path | None:
    sidecar = alias_sidecar(dest)
    if not sidecar.is_file():
        return None
    text = sidecar.read_text(encoding="utf-8").strip()
    return Path(text) if text else None


def _paths_equal(left: Path, right: Path) -> bool:
    try:
        if left.exists() and right.exists() and os.path.samefile(left, right):
            return True
    except OSError:
        pass
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def alias_matches_source(dest: Path, src: Path, workdir: Path | None = None) -> bool:
    """Whether an existing alias still points at the given source file."""
    if dest.exists() and same_media_file(dest, src):
        return True
    recorded = read_alias_source(dest)
    if recorded is None:
        return False
    return _paths_equal(recorded, Path(src))


def _unique_alias(workdir: Path, suffix: str, src: Path) -> Path:
    dest = workdir / f"input{suffix}"
    if not dest.exists() or alias_matches_source(dest, src, workdir):
        return dest
    digest = ascii_stem(src, length=8)
    alt = workdir / f"input_{digest}{suffix}"
    if not alt.exists() or alias_matches_source(alt, src, workdir):
        log.info("工作目录里已有来自其他源文件的别名，改用 %s", alt.name)
        return alt
    n = 2
    while True:
        cand = workdir / f"input_{digest}_{n}{suffix}"
        if not cand.exists() or alias_matches_source(cand, src, workdir):
            return cand
        n += 1


def alias_input(src: str | Path, workdir: Path) -> Path:
    """Give the pipeline an ASCII path (hardlink/symlink/copy) so OpenCV/YOLO work on Windows."""
    src = Path(src).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"找不到视频: {src}")
    workdir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() if src.suffix else ".mp4"
    dest = _unique_alias(workdir, suffix, src)
    if dest.exists() and alias_matches_source(dest, src, workdir):
        write_alias_source(dest, src)
        return dest
    try:
        dest.hardlink_to(src)
        log.info("已为输入视频创建硬链接: %s", dest)
        write_alias_source(dest, src)
        return dest
    except OSError:
        pass
    try:
        dest.symlink_to(src)
        log.info("已为输入视频创建符号链接: %s", dest)
        write_alias_source(dest, src)
        return dest
    except OSError:
        pass
    log.info("正在把输入视频复制到工作目录（可能需要一点时间）: %s", dest)
    shutil.copy2(src, dest)
    write_alias_source(dest, src)
    return dest


def find_input_alias(workdir: Path, src: Path) -> Path | None:
    """Return an existing ASCII alias only when it still belongs to *src*."""
    src = Path(src)
    suffix = src.suffix.lower() if src.suffix else ".mp4"
    candidates = [workdir / f"input{suffix}", workdir / "input.mp4"]
    digest = ascii_stem(src, length=8)
    candidates.extend(sorted(workdir.glob(f"input_{digest}*{suffix}")))
    seen: set[Path] = set()
    for cand in candidates:
        try:
            key = cand.resolve() if cand.exists() else cand
        except OSError:
            key = cand
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file() and alias_matches_source(cand, src, workdir):
            return cand
    return None
