"""Stage 6: default face/character swap (InsightFace inswapper).

`ReplacementBackend` is the extension point for a later 3D-mesh or LivePortrait
backend. v1 only ships InswapperBackend. Wan2.2 / AniCrafter are not wired in
because they will not reliably fit in 8GB VRAM.
"""

from __future__ import annotations

import json
import logging
import shutil
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

# Identity clustering uses cosine distance (default 0.45), i.e. similarity about 0.55,
# to treat tracks as the same person. A reference still has to clear that bar, and
# beat the next cluster, before we swap. Below that we skip instead of guessing.
DEFAULT_MATCH_SIMILARITY = 0.55
MATCH_MARGIN = 0.08


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
                shutil.copyfile(path, dest)
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
        self._min_face_iou = 0.2

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
        self._min_face_iou = float(getattr(cfg.identity, "min_face_iou", 0.2))
        vram_report("替换-已加载")

    def _best_face(self, face_image_path: str | Path) -> Any | None:
        if self._app is None:
            raise RuntimeError("InswapperBackend.load() 必须先于检测调用")
        img = imread(face_image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取参考脸图片: {face_image_path}")
        faces = self._app.get(img) or []
        if not faces:
            return None
        faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
        return faces[0]

    def prepare_source(self, face_image_path: str | Path) -> Any:
        face = self._best_face(face_image_path)
        if face is None:
            raise RuntimeError(f"参考图里没有检测到人脸: {face_image_path}")
        self._source_face = face
        log.info(
            "参考脸已就绪（检测分 %.3f，框 %s）",
            float(getattr(self._source_face, "det_score", 0.0) or 0.0),
            [round(float(x), 1) for x in self._source_face.bbox],
        )
        return face

    def embed_image(self, face_image_path: str | Path) -> list[float] | None:
        """Embedding of the clearest face in an image, or None when none is detected."""
        try:
            face = self._best_face(face_image_path)
        except Exception as exc:
            log.debug("读取人脸特征失败 %s: %s", face_image_path, exc)
            return None
        if face is None:
            return None
        return embedding_of(face)

    def replace_frame(
        self,
        frame: np.ndarray,
        target_bboxes: list[list[float]],
        source_faces: list[Any] | None = None,
    ) -> np.ndarray:
        if self._app is None or self._swapper is None:
            return frame
        if not target_bboxes:
            return frame
        if source_faces is None:
            if self._source_face is None:
                return frame
            source_faces = [self._source_face] * len(target_bboxes)
        faces = self._app.get(frame) or []
        if not faces:
            return frame
        from recast.stages.face_match import match_boxes

        face_boxes = [[float(x) for x in face.bbox] for face in faces]
        pairs = match_boxes(target_bboxes, face_boxes, min_iou=self._min_face_iou)
        out = frame
        for person_i, face_i in pairs:
            source = source_faces[person_i] if person_i < len(source_faces) else None
            if source is None:
                continue
            try:
                out = self._swapper.get(out, faces[face_i], source, paste_back=True)
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


def embedding_of(face: Any) -> list[float] | None:
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return [float(x) for x in (vec / norm)]


def as_embedding(value: Any) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        vec = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return vec or None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def match_similarity(identity: dict[str, Any] | None, override: float | None = None) -> float:
    """Minimum cosine similarity for a confident reference-to-cluster match."""
    if override is not None:
        return float(override)
    raw = (identity or {}).get("cluster_threshold")
    if raw is None:
        return DEFAULT_MATCH_SIMILARITY
    try:
        dist = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MATCH_SIMILARITY
    if 0.0 < dist < 1.0:
        return 1.0 - dist
    return DEFAULT_MATCH_SIMILARITY


def _known_track_ids(track_payload: dict[str, Any] | None, identity: dict[str, Any] | None) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []

    def _add(raw: Any) -> None:
        if raw is None:
            return
        tid = int(raw)
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)

    for track in (track_payload or {}).get("tracks") or []:
        _add(track.get("track_id"))
    for person in (identity or {}).get("persons") or []:
        for tid in person.get("track_ids") or []:
            _add(tid)
    return ordered


def _tracks_in_cluster(
    identity: dict[str, Any] | None,
    track_payload: dict[str, Any] | None,
    cluster_id: int,
) -> list[int]:
    if identity:
        for person in identity.get("persons") or []:
            if person.get("cluster_id") is None:
                continue
            if int(person["cluster_id"]) != int(cluster_id):
                continue
            ids: list[int] = []
            seen: set[int] = set()
            for tid in person.get("track_ids") or []:
                tid = int(tid)
                if tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
            if ids:
                return ids
    ids = []
    seen = set()
    for track in (track_payload or {}).get("tracks") or []:
        if track.get("cluster_id") is None or track.get("track_id") is None:
            continue
        if int(track["cluster_id"]) != int(cluster_id):
            continue
        tid = int(track["track_id"])
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids


def select_target_tracks(
    track_payload: dict[str, Any],
    track_id: int | None = None,
    *,
    cluster_id: int | None = None,
    identity: dict[str, Any] | None = None,
) -> list[int]:
    """Tracks to swap for one person.

    A cluster contributes every track id that belongs to it. An explicit
    ``track_id`` narrows that one person to the given track when the track is
    part of the cluster. There is no longest-track fallback.
    """
    known = _known_track_ids(track_payload, identity)
    known_set = set(known)
    if not known_set:
        raise RuntimeError("没有可用于替换的人物轨迹。")

    if cluster_id is None:
        if track_id is None:
            raise RuntimeError(
                "未指定人物聚类。人物替换按 identity.json 的聚类选择全部轨迹，不再默认只换出现最多的一条。"
            )
        if int(track_id) not in known_set:
            raise RuntimeError(f"找不到 track-id={track_id}。可用: {sorted(known_set)}")
        return [int(track_id)]

    ids = _tracks_in_cluster(identity, track_payload, int(cluster_id))
    if known_set:
        present = [i for i in ids if i in known_set]
        if present:
            ids = present
    if not ids:
        raise RuntimeError(f"聚类 {cluster_id} 没有可替换的轨迹。")
    if track_id is not None and int(track_id) in set(ids):
        return [int(track_id)]
    return ids


def _person_name(person: dict[str, Any]) -> str:
    return str(person.get("name") or person.get("auto_name") or person.get("cluster_id") or "未命名")


def _owns_track(person: dict[str, Any], track_id: int | None) -> bool:
    if track_id is None:
        return False
    return int(track_id) in {int(t) for t in (person.get("track_ids") or [])}


def _owner_name(identity: dict[str, Any] | None, track_id: int | None) -> str | None:
    if track_id is None or not identity:
        return None
    for person in identity.get("persons") or []:
        if _owns_track(person, track_id):
            return _person_name(person)
    return None


def _ref_brief(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "character_id": ref.get("character_id"),
        "name": ref.get("name") or Path(str(ref.get("face_path") or "")).stem,
        "face_path": str(ref.get("face_path") or ""),
    }


def plan_replace_targets(
    references: list[dict[str, Any]],
    identity: dict[str, Any] | None,
    track_payload: dict[str, Any] | None = None,
    *,
    track_id: int | None = None,
    min_similarity: float | None = None,
    margin: float = MATCH_MARGIN,
) -> dict[str, Any]:
    """Map each reference face onto one identity cluster, or skip it.

    Assignment is one-to-one. A reference is skipped when it has no embedding,
    its best cluster is below ``min_similarity``, or the top two clusters are
    too close to tell apart. An explicit ``track_id`` only narrows the matched
    person who owns that track.
    """
    identity = identity or {}
    track_payload = track_payload or {"tracks": []}
    threshold = match_similarity(identity, min_similarity)
    if track_id is not None:
        known = _known_track_ids(track_payload, identity)
        if int(track_id) not in set(known):
            raise RuntimeError(f"找不到 track-id={track_id}。可用: {sorted(known)}")

    persons = []
    for person in identity.get("persons") or []:
        emb = as_embedding(person.get("embedding"))
        tracks = [int(t) for t in (person.get("track_ids") or [])]
        if emb is None or not tracks:
            continue
        persons.append({**person, "embedding": emb, "track_ids": tracks})

    any_embedding = any(as_embedding(p.get("embedding")) for p in (identity.get("persons") or []))
    skipped: list[dict[str, Any]] = []
    scored: list[list[tuple[float, int]]] = []
    handled: set[int] = set()

    for ri, ref in enumerate(references):
        brief = _ref_brief(ref)
        emb = as_embedding(ref.get("embedding"))
        if emb is None:
            reason = ref.get("embed_error") or "参考脸没有人脸特征，无法与视频中的人物对照，已跳过。"
            skipped.append({**brief, "reason": str(reason), "score": None})
            handled.add(ri)
            scored.append([])
            continue
        if not persons:
            if not any_embedding:
                reason = "身份聚类没有人脸特征，无法确认参考脸对应哪个人，已跳过以免换错人。"
            else:
                reason = "身份聚类里没有带轨迹的人物，已跳过以免换错人。"
            skipped.append({**brief, "reason": reason, "score": None})
            handled.add(ri)
            scored.append([])
            continue
        pairs = [(cosine_similarity(emb, person["embedding"]), pi) for pi, person in enumerate(persons)]
        pairs.sort(key=lambda item: (-item[0], item[1]))
        scored.append(pairs)

    candidates: list[tuple[float, int, int]] = []
    for ri, pairs in enumerate(scored):
        for sim, pi in pairs:
            candidates.append((sim, ri, pi))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    assigned: dict[int, tuple[int, float]] = {}
    used_person: set[int] = set()
    for sim, ri, pi in candidates:
        if ri in handled or ri in assigned or pi in used_person:
            continue
        pairs = scored[ri]
        if not pairs or pi != pairs[0][1]:
            continue
        best_sim = pairs[0][0]
        if best_sim < threshold:
            continue
        second = pairs[1][0] if len(pairs) > 1 else -1.0
        if second >= 0 and best_sim - second < margin:
            person_a = persons[pairs[0][1]]
            person_b = persons[pairs[1][1]]
            skipped.append(
                {
                    **_ref_brief(references[ri]),
                    "reason": (
                        f"与「{_person_name(person_a)}」相似度 {best_sim:.3f}，"
                        f"与「{_person_name(person_b)}」相似度 {second:.3f}，"
                        f"差距小于 {margin:.2f}，无法确认是谁，已跳过以免换错人。"
                    ),
                    "score": best_sim,
                }
            )
            handled.add(ri)
            continue
        assigned[ri] = (pi, best_sim)
        used_person.add(pi)

    targets: list[dict[str, Any]] = []
    for ri, (pi, sim) in sorted(assigned.items()):
        person = persons[pi]
        ref = references[ri]
        override = int(track_id) if _owns_track(person, track_id) else None
        cluster_raw = person.get("cluster_id")
        cluster_id = int(cluster_raw) if cluster_raw is not None else None
        track_ids = select_target_tracks(
            track_payload,
            override,
            cluster_id=cluster_id,
            identity=identity,
        )
        targets.append(
            {
                **_ref_brief(ref),
                "cluster_id": cluster_id,
                "person_name": _person_name(person),
                "track_ids": track_ids,
                "score": sim,
            }
        )

    for ri, ref in enumerate(references):
        if ri in handled or ri in assigned:
            continue
        pairs = scored[ri]
        brief = _ref_brief(ref)
        if not pairs:
            continue
        best_sim, best_pi = pairs[0]
        person = persons[best_pi]
        if best_pi in used_person:
            reason = (
                f"最相似的人物「{_person_name(person)}」已由另一张参考脸匹配"
                f"（相似度 {best_sim:.3f}），已跳过以免把两张脸换到同一个人。"
            )
        else:
            reason = (
                f"与最相似的人物「{_person_name(person)}」相似度 {best_sim:.3f}，"
                f"低于 {threshold:.2f}，已跳过以免换错人。"
            )
        skipped.append({**brief, "reason": reason, "score": best_sim})

    track_note = None
    if track_id is not None:
        owner = _owner_name(identity, track_id)
        narrowed = [t for t in targets if t.get("track_ids") == [int(track_id)]]
        if narrowed:
            who = narrowed[0].get("person_name") or narrowed[0].get("name")
            track_note = (
                f"track-id={int(track_id)} 只收窄「{who}」的轨迹，"
                "其他已匹配人物仍替换聚类内的全部轨迹。"
            )
        elif owner:
            track_note = (
                f"指定的 track-id={int(track_id)} 属于「{owner}」，"
                "但没有参考脸被确认匹配到该人物，因此没有替换这条轨迹。"
            )
        else:
            track_note = f"已指定 track-id={int(track_id)}。"

    return {
        "targets": targets,
        "skipped": skipped,
        "track_note": track_note,
        "min_similarity": threshold,
    }


def index_swap_frames(
    track_payload: dict[str, Any],
    targets: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """One frame index → every box that should be swapped, across people."""
    source_of: dict[int, str] = {}
    for target in targets:
        key = str(target.get("source_key") or target.get("face_path") or "")
        for tid in target.get("track_ids") or []:
            source_of[int(tid)] = key
    frames: dict[int, list[dict[str, Any]]] = {}
    for track in track_payload.get("tracks") or []:
        if track.get("track_id") is None:
            continue
        tid = int(track["track_id"])
        if tid not in source_of:
            continue
        for item in track.get("frames") or []:
            frames.setdefault(int(item["frame_idx"]), []).append(
                {
                    "bbox": item["bbox"],
                    "source_key": source_of[tid],
                    "track_id": tid,
                }
            )
    return frames


def replace_checkpoint_inputs(
    video: str | Path,
    workdir: str | Path,
    face_paths: list[str | Path],
) -> list[Path]:
    """Fingerprint inputs for the replace step: video, identity, tracks, every face."""
    work = Path(workdir)
    inputs = [Path(video), work / "identity.json"]
    for name in ("tracks_raw.json", "tracks.json"):
        path = work / name
        if path.is_file():
            inputs.append(path)
    for face in face_paths:
        inputs.append(Path(face))
    return inputs


def _load_identity(workdir: Path, track_payload: dict[str, Any], identity: dict[str, Any] | None) -> dict[str, Any]:
    if identity:
        return identity
    cached = track_payload.get("_identity")
    if isinstance(cached, dict) and cached:
        return cached
    path = Path(workdir) / "identity.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _fill_cluster_embeddings(backend: InswapperBackend, identity: dict[str, Any], workdir: Path) -> None:
    persons = list(identity.get("persons") or [])
    if not persons or all(as_embedding(p.get("embedding")) for p in persons):
        return
    samples: dict[str, list[str]] = {}
    index = Path(workdir) / "face_samples" / "index.json"
    if index.is_file():
        try:
            raw = json.loads(index.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                samples = {str(k): [str(p) for p in (v or [])] for k, v in raw.items()}
        except Exception as exc:
            log.debug("读取人脸样本索引失败: %s", exc)
    for person in persons:
        if as_embedding(person.get("embedding")):
            continue
        auto = str(person.get("auto_name") or person.get("name") or "")
        for raw in samples.get(auto) or []:
            emb = backend.embed_image(raw)
            if emb:
                person["embedding"] = emb
                log.info("聚类「%s」使用人脸样本补齐了特征。", _person_name(person))
                break


def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    def _target(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "character_id": row.get("character_id"),
            "name": row.get("name"),
            "face_path": row.get("face_path"),
            "cluster_id": row.get("cluster_id"),
            "person_name": row.get("person_name"),
            "track_ids": list(row.get("track_ids") or []),
            "score": row.get("score"),
        }

    def _skip(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "character_id": row.get("character_id"),
            "name": row.get("name"),
            "face_path": row.get("face_path"),
            "reason": row.get("reason"),
            "score": row.get("score"),
        }

    return {
        "targets": [_target(row) for row in plan.get("targets") or []],
        "skipped": [_skip(row) for row in plan.get("skipped") or []],
        "track_note": plan.get("track_note"),
        "min_similarity": plan.get("min_similarity"),
        "note": "L1 人物替换是本地换脸，不是提取了3D模型。",
    }


def _write_plan(workdir: Path, plan: dict[str, Any]) -> Path:
    path = Path(workdir) / "replace_plan.json"
    path.write_text(json.dumps(_public_plan(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _collect_references(
    face_image: str | Path | None,
    references: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in references or []:
        path = ref.get("face_path")
        if not path:
            continue
        key = str(Path(path))
        if key in seen:
            continue
        seen.add(key)
        rows.append({**ref, "face_path": key})
    if face_image:
        path = Path(face_image)
        key = str(path)
        if path.is_file() and key not in seen:
            rows.append(
                {
                    "character_id": None,
                    "name": path.stem,
                    "face_path": key,
                }
            )
    return rows


def run(
    video: str | Path,
    info: VideoInfo,
    track_payload: dict[str, Any],
    workdir: Path,
    cfg: RecastConfig,
    face_image: str | Path | None = None,
    track_id: int | None = None,
    *,
    references: list[dict[str, Any]] | None = None,
    identity: dict[str, Any] | None = None,
    min_similarity: float | None = None,
) -> Path:
    """Swap every matched reference in one pass. Output is always replaced.mp4."""
    workdir = Path(workdir)
    refs = _collect_references(face_image, references)
    if not refs:
        raise RuntimeError("缺少参考脸，无法进行人物替换。")
    ident = _load_identity(workdir, track_payload, identity)
    if not ident:
        raise RuntimeError("人物替换需要 identity.json 中的人物聚类，请先完成「03 人物提取与建库」。")

    device = requested_device(cfg.device)
    vram_report("替换-加载前")
    backend = build_backend(cfg)
    backend.load(cfg, device)
    prepared: dict[str, Any] = {}
    try:
        _fill_cluster_embeddings(backend, ident, workdir)
        for ref in refs:
            try:
                prepared[str(ref["face_path"])] = backend.prepare_source(ref["face_path"])
            except Exception as exc:
                ref["embed_error"] = str(exc)
                continue
            if as_embedding(ref.get("embedding")) is None:
                ref["embedding"] = embedding_of(prepared[str(ref["face_path"])])
        plan = plan_replace_targets(
            refs,
            ident,
            track_payload,
            track_id=track_id,
            min_similarity=min_similarity,
        )
        usable: list[dict[str, Any]] = []
        for target in plan["targets"]:
            source = prepared.get(str(target.get("face_path")))
            if source is None:
                plan["skipped"].append(
                    {
                        **_ref_brief(target),
                        "reason": "参考脸未能载入，无法换脸。",
                        "score": target.get("score"),
                    }
                )
                continue
            target["source_key"] = str(target["face_path"])
            usable.append(target)
        plan["targets"] = usable
        _write_plan(workdir, plan)
        for row in plan["skipped"]:
            log.warning("已跳过参考脸「%s」：%s", row.get("name") or row.get("face_path"), row.get("reason"))
        if plan.get("track_note"):
            log.info("%s", plan["track_note"])
        if not usable:
            reasons = "；".join(
                f"{row.get('name') or row.get('face_path')}：{row.get('reason')}" for row in plan["skipped"]
            )
            raise RuntimeError("人物替换没有可确认的目标，已跳过以免换错人。" + (reasons or ""))

        names = "、".join(str(t.get("person_name") or t.get("name") or "") for t in usable)
        log.info(
            "执行 L1 人物替换（本地换脸，不是提取了3D模型）。一次处理 %s 个已匹配人物：%s",
            len(usable),
            names,
        )
        from recast.stages.track import fill_track_gaps

        track_payload = fill_track_gaps(
            track_payload,
            max_gap=int(getattr(cfg.detect, "max_gap_frames", 10) or 10),
            fps=float(info.fps) if info.fps else None,
        )
        frames = index_swap_frames(track_payload, usable)
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
                items = frames.get(frame_idx, [])
                if items:
                    boxes = [item["bbox"] for item in items]
                    sources = [prepared.get(item["source_key"]) for item in items]
                    new_frame = backend.replace_frame(frame, boxes, sources)
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
    finally:
        backend.unload()

    log.info("替换完成: %s 帧中有 %s 帧执行了换脸，输出 %s", frame_idx, swapped, out_path)
    vram_report("替换-释放后")
    return out_path
