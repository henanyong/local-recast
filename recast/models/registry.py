"""Named model registry. Large weights are not bundled; they download on demand."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from recast.config import PROJECT_ROOT, RecastConfig

log = logging.getLogger("recast")


@dataclass
class ModelSpec:
    key: str
    label: str
    kind: str
    filename: str | None = None
    url: str | None = None
    sha256: str | None = None
    min_size: int = 1_000_000
    local_dirs: tuple[str, ...] = ()
    required_for: tuple[str, ...] = ()
    notes: str = ""


REGISTRY: list[ModelSpec] = [
    ModelSpec(
        key="whisper",
        label="faster-whisper 语音转写",
        kind="asr",
        required_for=("extract",),
        notes="由 faster-whisper 在首次转写时自动下载，不随安装包分发。",
    ),
    ModelSpec(
        key="yolo11n",
        label="YOLO11n 行人检测",
        kind="detect",
        filename="yolo11n.pt",
        local_dirs=("models",),
        required_for=("library",),
        notes="由 ultralytics 在首次检测时自动下载。",
    ),
    ModelSpec(
        key="buffalo_s",
        label="InsightFace buffalo_s",
        kind="face",
        required_for=("library", "replace"),
        notes="首次运行下载到用户目录 .insightface/models/ 。",
    ),
    ModelSpec(
        key="inswapper_128",
        label="InsightFace inswapper_128（人物替换）",
        kind="replace",
        filename="inswapper_128.onnx",
        url="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        min_size=10_000_000,
        local_dirs=("models",),
        required_for=("replace",),
        notes="首次执行「04 人物替换」时下载。请自行阅读 InsightFace / inswapper 许可证。",
    ),
]


def _candidates(spec: ModelSpec, cfg: RecastConfig | None = None) -> list[Path]:
    paths: list[Path] = []
    if spec.filename:
        named = Path(spec.filename)
        paths.append(named)
        paths.append(PROJECT_ROOT / "models" / named.name)
        paths.append(Path.home() / ".insightface" / "models" / named.name)
        if cfg is not None and getattr(cfg, "replace", None) and spec.key.startswith("inswapper"):
            paths.insert(0, Path(cfg.replace.model))
            paths.insert(1, PROJECT_ROOT / "models" / Path(cfg.replace.model).name)
    for d in spec.local_dirs:
        if spec.filename:
            paths.append(PROJECT_ROOT / d / spec.filename)
    # unique
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def locate(spec: ModelSpec, cfg: RecastConfig | None = None) -> Path | None:
    for path in _candidates(spec, cfg):
        if path.is_file() and path.stat().st_size >= spec.min_size:
            return path
    return None


def verify_hash(path: Path, sha256: str | None) -> bool:
    if not sha256:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == sha256.lower()


def download_spec(spec: ModelSpec, dest_dir: Path | None = None) -> Path:
    if not spec.url or not spec.filename:
        raise FileNotFoundError(f"模型「{spec.label}」没有登记下载地址，需由对应库自动获取。")
    dest_dir = dest_dir or (PROJECT_ROOT / "models")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.filename
    if dest.is_file() and dest.stat().st_size >= spec.min_size and verify_hash(dest, spec.sha256):
        return dest
    log.info("正在下载模型 %s → %s", spec.label, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(spec.url, timeout=60) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    if tmp.stat().st_size < spec.min_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"下载的模型过小，可能不完整: {spec.label}")
    if not verify_hash(tmp, spec.sha256):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"模型哈希校验失败: {spec.label}")
    tmp.replace(dest)
    return dest


def status_report(cfg: RecastConfig | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in REGISTRY:
        path = locate(spec, cfg)
        present = path is not None
        hash_ok = verify_hash(path, spec.sha256) if present and spec.sha256 else True
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "present": present and hash_ok,
                "path": str(path) if path else None,
                "hash_ok": hash_ok,
                "notes": spec.notes,
            }
        )
    return rows
