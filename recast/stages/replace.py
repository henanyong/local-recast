"""Stage 6: default face/character swap (InsightFace inswapper).

`ReplacementBackend` is the extension point for a later 3D-mesh or LivePortrait
backend. v1 only ships InswapperBackend. Wan2.2 / AniCrafter are not wired in
because they will not reliably fit in 8GB VRAM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from tqdm import tqdm

from recast.config import PROJECT_ROOT, RecastConfig
from recast.device import release_cuda, requested_device, vram_report
from recast.ffmpeg import VideoInfo, open_ffmpeg_writer
from recast.media import imread, open_capture
from recast.runtime import check_cancel

log = logging.getLogger("recast")


class ReplacementBackend(Protocol):
    """Common interface for the default inswapper path and future 3D / LivePortrait backends."""

    name: str

    def load(self, cfg: RecastConfig, device: str) -> None:
        ...

    def prepare_source(self, face_image_path: str | Path) -> None:
        ...

    def replace_frame(self, frame: np.ndarray, target_bboxes: list[list[float]]) -> np.ndarray:
        ...

    def unload(self) -> None:
        ...


def _providers(device: str) -> list[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _center_in(person: list[float], face: list[float]) -> bool:
    cx = (face[0] + face[2]) / 2.0
    cy = (face[1] + face[3]) / 2.0
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def ensure_inswapper_weights(cfg: RecastConfig) -> Path:
    """Locate or download inswapper_128.onnx. Downloads only happen at runtime."""
    named = Path(cfg.replace.model)
    candidates = [
        named,
        PROJECT_ROOT / "models" / named.name,
        Path.home() / ".insightface" / "models" / named.name,
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 10_000_000:
            log.info("使用已有 inswapper 模型: %s", path)
            return path

    dest = PROJECT_ROOT / "models" / named.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "首次替换需要下载 %s（约 550 MB）到 %s 。请保持网络畅通。",
        named.name,
        dest,
    )
    url = cfg.replace.model_url
    try:
        from huggingface_hub import hf_hub_download

        # repo ezioruan/inswapper_128.onnx stores the onnx at repo root
        downloaded = hf_hub_download(
            repo_id="ezioruan/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            local_dir=str(dest.parent),
        )
        path = Path(downloaded)
        if path.is_file():
            if path.resolve() != dest.resolve():
                dest.write_bytes(path.read_bytes())
            return dest
    except Exception as exc:
        log.warning("Hugging Face 下载失败，改用直接 URL: %s", exc)

    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"无法下载 inswapper 模型。请手动下载 {url} 并保存为 {dest}"
        ) from None
    if not dest.is_file() or dest.stat().st_size < 10_000_000:
        raise RuntimeError(f"inswapper 模型下载不完整: {dest}")
    log.info("inswapper 模型已就绪: %s", dest)
    return dest


class InswapperBackend:
    """Default v1 replacement: paste a reference face onto the target track."""

    name = "inswapper"

    def __init__(self) -> None:
        self._app: Any = None
        self._swapper: Any = None
        self._source_face: Any = None
        self._device = "cpu"

    def load(self, cfg: RecastConfig, device: str) -> None:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        self._device = device
        providers = _providers(device)
        log.info("加载 InsightFace 检测器 %s 与 inswapper", cfg.identity.model)
        self._app = FaceAnalysis(name=cfg.identity.model, providers=providers)
        ctx_id = 0 if device == "cuda" else -1
        size = int(cfg.identity.det_size)
        self._app.prepare(ctx_id=ctx_id, det_size=(size, size))
        model_path = ensure_inswapper_weights(cfg)
        self._swapper = get_model(str(model_path), providers=providers)
        if hasattr(self._swapper, "prepare"):
            self._swapper.prepare(ctx_id=ctx_id)
        vram_report("替换-已加载")

    def prepare_source(self, face_image_path: str | Path) -> None:
        if self._app is None:
            raise RuntimeError("InswapperBackend.load() 必须先于 prepare_source() 调用")
        img = imread(face_image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取参考脸图片: {face_image_path}")
        faces = self._app.get(img) or []
        if not faces:
            raise RuntimeError(f"参考图里没有检测到人脸: {face_image_path}")
        faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
        self._source_face = faces[0]
        log.info(
            "参考脸已就绪（检测分 %.3f，框 %s）",
            float(getattr(self._source_face, "det_score", 0.0) or 0.0),
            [round(float(x), 1) for x in self._source_face.bbox],
        )

    def replace_frame(self, frame: np.ndarray, target_bboxes: list[list[float]]) -> np.ndarray:
        if self._app is None or self._swapper is None or self._source_face is None:
            return frame
        if not target_bboxes:
            return frame
        faces = self._app.get(frame) or []
        if not faces:
            return frame
        out = frame
        used: set[int] = set()
        for bbox in target_bboxes:
            best_i, best = -1, 0.0
            for i, face in enumerate(faces):
                if i in used:
                    continue
                fb = [float(x) for x in face.bbox]
                score = _iou(bbox, fb)
                if score < 0.05 and _center_in(bbox, fb):
                    score = max(score, 0.12)
                if score > best:
                    best_i, best = i, score
            if best_i < 0 or best < 0.05:
                continue
            used.add(best_i)
            try:
                out = self._swapper.get(out, faces[best_i], self._source_face, paste_back=True)
            except Exception as exc:
                log.debug("单帧替换失败: %s", exc)
        return out

    def unload(self) -> None:
        app, swapper, face = self._app, self._swapper, self._source_face
        self._app = None
        self._swapper = None
        self._source_face = None
        release_cuda(app, swapper, face)


def build_backend(cfg: RecastConfig) -> ReplacementBackend:
    name = (cfg.replace.backend or "inswapper").lower()
    if name != "inswapper":
        raise RuntimeError(
            f"v1 默认替换后端只有 inswapper，收到的是 {name}。"
            "Wan2.2 / AniCrafter / LivePortrait 属于后续可选工作，尚未接入。"
        )
    return InswapperBackend()


def select_target_tracks(
    track_payload: dict[str, Any],
    track_id: int | None,
) -> list[int]:
    tracks = list(track_payload.get("tracks") or [])
    if not tracks:
        raise RuntimeError("没有可用于替换的人物轨迹。")
    if track_id is not None:
        ids = {int(t["track_id"]) for t in tracks}
        if int(track_id) not in ids:
            raise RuntimeError(f"找不到 track-id={track_id}。可用: {sorted(ids)}")
        return [int(track_id)]

    def _length(t: dict[str, Any]) -> int:
        return len(t.get("frames") or [])

    # Prefer the person who appears the most; break ties by earlier first_frame.
    tracks = sorted(tracks, key=lambda t: (-_length(t), int(t.get("first_frame") or 0), int(t["track_id"])))
    chosen = int(tracks[0]["track_id"])
    log.info(
        "未指定 --track-id，默认替换出现帧数最多的轨迹 %s（%s，%s 帧）",
        chosen,
        tracks[0].get("name"),
        _length(tracks[0]),
    )
    return [chosen]


def run(
    video: str | Path,
    info: VideoInfo,
    track_payload: dict[str, Any],
    workdir: Path,
    cfg: RecastConfig,
    face_image: str | Path,
    track_id: int | None = None,
) -> Path:
    targets = set(select_target_tracks(track_payload, track_id))
    frame_boxes: dict[int, list[list[float]]] = {}
    for track in track_payload.get("tracks") or []:
        if int(track["track_id"]) not in targets:
            continue
        for item in track.get("frames") or []:
            frame_boxes.setdefault(int(item["frame_idx"]), []).append(item["bbox"])

    device = requested_device(cfg.device)
    vram_report("替换-加载前")
    backend = build_backend(cfg)
    backend.load(cfg, device)
    backend.prepare_source(face_image)

    out_path = workdir / "replaced.mp4"
    cap = open_capture(video)
    writer = open_ffmpeg_writer(out_path, info.width, info.height, info.fps, crf=cfg.compose.crf)
    total = info.frame_count or None
    pbar = tqdm(total=total, desc="替换", unit="f")
    frame_idx = 0
    swapped = 0
    try:
        while True:
            check_cancel()
            ok, frame = cap.read()
            if not ok:
                break
            boxes = frame_boxes.get(frame_idx, [])
            if boxes:
                new_frame = backend.replace_frame(frame, boxes)
                if new_frame is not frame:
                    swapped += 1
                frame = new_frame
            if frame.shape[1] != info.width or frame.shape[0] != info.height:
                frame = cv2.resize(frame, (info.width, info.height))
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.stdin.close()
        err = writer.communicate()[1]
        if writer.returncode:
            raise RuntimeError(f"写出替换视频失败: {(err or b'')[-2000:]}")
        backend.unload()

    log.info("替换完成: %s 帧中有 %s 帧执行了换脸，输出 %s", frame_idx, swapped, out_path)
    vram_report("替换-释放后")
    return out_path
