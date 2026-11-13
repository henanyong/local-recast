"""Person replacement, 2D route: costume + face, with a cost per frame.

Why 2D and not a 3D stand-in: replacing the body means rebuilding the hair,
and hair is where a real person's identity actually lives. Measured on a
real frame, a pasted body scores a neck seam ~4x the body's own edge
energy, and it deletes shoulder-length hair outright. Keeping the actor's
body, hair and lighting removes both failure modes by construction — the
only pixels that change are inside the clothing mask and (optionally) the
face.

Two entry points:

- :func:`replace_frame` — one frame in, one frame out, plus what actually
  ran and an audit of the result.
- :func:`replace_clip` — a whole clip, with a per-frame audit so a batch
  run reports which frames need eyes instead of asking a human to watch
  eight minutes per episode.

Honesty rules carried over from the rest of the codebase:

- a step that could not run says so in ``notes`` and leaves ``final``
  False; it never silently does nothing;
- the face step is delegated to ``stages.swap_preview``, which already
  distinguishes a real inswapper result from an aligned overlay;
- the audit numbers are reported raw, and a frame that cannot be audited
  is reported as not audited rather than as clean.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from recast.quality.replace_audit import BG_DAMAGE_LIMIT

log = logging.getLogger("recast")

# A frame whose composite damaged the background beyond this share must not
# reach the output video: the damage is visible in every later viewing, while
# a missing frame of effect is not. Same number as the audit's review flag,
# but here it gates the writer instead of a report.
REJECT_BG_DAMAGE_LIMIT = BG_DAMAGE_LIMIT


def rejection_reason(audit: dict[str, Any] | None) -> str | None:
    """Why this frame's composite must not go into the output, or None.

    The gate runs on the FINAL composite — face swap and garment transfer are
    both already applied — against the source frame, so a face swap that
    smears the background is caught exactly like a bad garment pass. Every
    criterion returns a stable reason string so callers can aggregate counts.
    """
    if not audit:
        return None
    damage = audit.get("background_damage_frac")
    if damage is not None and float(damage) > REJECT_BG_DAMAGE_LIMIT:
        return "background_damage"
    return None


def _read(path: Path) -> Any:
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write(path: Path, image: Any) -> bool:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _face_reference(folder: Path) -> Path | None:
    """First face sample in ``face/``.

    Deliberately does NOT fall back to ``costume/``: a costume crop is a body
    shot, and feeding it to a face swapper yields a face-shaped smear rather
    than an error. Missing is better than wrong here.
    """
    from recast.stages.garment_transfer import IMAGE_SUFFIXES

    directory = folder / "face"
    if not directory.is_dir():
        return None
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def replace_frame(
    frame_bgr: Any,
    *,
    character_folder: str | Path,
    person_mask: Any,
    neck_y: int,
    parts: tuple[str, ...] = ("upper",),
    split_y: int | None = None,
    auto_garment: bool = False,
    face_source: Path | None = None,
    do_face: bool = False,
    cfg: Any = None,
    work_dir: str | Path | None = None,
    allow_noncommercial: bool = True,
) -> dict[str, Any]:
    """Apply costume (and optionally face) to one frame.

    ``do_face`` is off by default on purpose: the available face backend is
    either an aligned overlay (not a swap) or a non-commercial model, so the
    caller has to ask for it and read ``final`` to know what it got.

    ``allow_noncommercial=False`` is the delivery switch: it also refuses the
    non-commercial face model, not just the segmentation model.
    """
    import numpy as np

    from recast.quality.replace_audit import audit_composite
    from recast.stages.garment_transfer import (
        GarmentError,
        costume_reference,
        read_image,
        transfer_garment,
    )

    folder = Path(character_folder)
    notes: list[str] = []
    steps: list[dict[str, Any]] = []
    image = np.asarray(frame_bgr).copy()

    reference_path = costume_reference(folder)
    if reference_path is None:
        notes.append(f"没有服装参考图（{folder / 'costume'}），跳过换装")
    else:
        reference = read_image(reference_path)
        for part in parts:
            try:
                result = transfer_garment(
                    image,
                    person_mask,
                    neck_y,
                    reference,
                    part=part,
                    split_y=split_y,
                    auto_garment=auto_garment,
                )
                image = result["image"]
                steps.append(
                    {
                        "step": f"garment:{part}",
                        "ok": True,
                        "reference": str(reference_path),
                        "region_pixels": result["region_pixels"],
                    }
                )
            except GarmentError as exc:
                steps.append({"step": f"garment:{part}", "ok": False, "reason": str(exc)})
                notes.append(str(exc))

    face_final = False
    if do_face:
        reference = Path(face_source) if face_source is not None else _face_reference(folder)
        if reference is None or not reference.is_file():
            notes.append("人脸步骤要求提供参考脸，但人物文件夹里没有可用图片")
            steps.append({"step": "face", "ok": False, "reason": "没有参考脸"})
        else:
            from recast.stages.face_swap import swap_frame

            tmp_dir = Path(work_dir) if work_dir else folder / ".tmp"
            try:
                result = swap_frame(
                    image,
                    reference,
                    allow_noncommercial=allow_noncommercial,
                    cfg=cfg,
                    tmp_dir=tmp_dir,
                )
                image = result["image"]
                face_final = bool(result["is_final"])
                steps.append(
                    {
                        "step": "face",
                        "ok": True,
                        "method": result["backend"],
                        "is_final": face_final,
                        "license": result.get("license"),
                        "reference": str(reference),
                    }
                )
                if result.get("note"):
                    notes.append(str(result["note"]))
            except Exception as exc:
                steps.append({"step": "face", "ok": False, "reason": str(exc)})
                notes.append(f"换脸失败：{exc}")

    audit: dict[str, Any] | None
    try:
        audit = audit_composite(
            np.asarray(frame_bgr),
            image,
            person_mask,
            neck_y,
            render_alpha=None,
        )
    except Exception as exc:
        audit = None
        notes.append(f"穿帮审计失败：{exc}")

    garment_ok = any(s["ok"] for s in steps if str(s.get("step", "")).startswith("garment"))
    final = bool(garment_ok and (face_final if do_face else True))
    return {
        "image": image,
        "steps": steps,
        "audit": audit,
        "final": final,
        "notes": notes,
    }


def replace_clip(
    video_path: str | Path,
    out_path: str | Path,
    *,
    character_folder: str | Path,
    mask_fn: Callable[[Any, int], Any],
    neck_y_fn: Callable[[Any, int], int],
    parts: tuple[str, ...] = ("upper",),
    auto_garment: bool = False,
    do_face: bool = False,
    cfg: Any = None,
    frame_limit: int | None = None,
    allow_noncommercial: bool = True,
) -> dict[str, Any]:
    """Run :func:`replace_frame` over a clip and report per-frame quality.

    ``mask_fn(frame_bgr, index) -> mask`` is required: the segmentation
    backend is a policy decision (mediapipe's legacy API was removed in
    0.10.35, SCHP is not wired yet), so this function refuses to invent one.
    ``neck_y_fn(frame_bgr, index) -> int`` is required too: a guessed neck
    line (the old 42%-of-frame-height default) shifts every upper/lower
    garment edge, so a wrong guess silently wrecked whole clips. Pass
    ``estimate_neck_y`` (:mod:`recast.stages.person_mask`) — it returns
    ``{"neck_y": ...}`` measured from a detected face. Missing either one
    raises before any frame is processed.

    Frames are gated BEFORE they reach the writer: when :func:`rejection_reason`
    flags the final composite (face + garment, audited against the source
    frame) the replaced result is discarded and the ORIGINAL frame is written
    instead — a bad frame must never land in the deliverable. Every rejected
    frame is listed in ``rejected`` with its reason, and ``reject_reason_counts``
    aggregates them for the run report.

    Returns a summary with the audit distribution and the worst frames —
    the list a human should actually look at.
    """
    import cv2
    import numpy as np

    from recast.ffmpeg import open_ffmpeg_writer

    video_path = Path(video_path)
    out_path = Path(out_path)
    if neck_y_fn is None:
        raise TypeError(
            "neck_y_fn 是必填项：脖子切口必须来自实测（例如用 "
            "recast.stages.person_mask.estimate_neck_y 的结果包装成 "
            "neck_y_fn），不再接受画面高度 42% 的比例猜测——猜错一次，"
            "整段上下身分界全错。"
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    writer = open_ffmpeg_writer(out_path, width, height, fps)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    idx = 0
    notes: list[str] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_limit is not None and idx >= frame_limit:
                break
            mask = mask_fn(frame, idx)
            neck_y = int(neck_y_fn(frame, idx))
            result = replace_frame(
                frame,
                character_folder=character_folder,
                person_mask=mask,
                neck_y=int(neck_y),
                parts=parts,
                auto_garment=auto_garment,
                do_face=do_face,
                cfg=cfg,
                allow_noncommercial=allow_noncommercial,
            )
            audit = result["audit"] or {}
            reason = rejection_reason(audit)
            if reason is not None:
                # The composite failed the gate: write the untouched source
                # frame so the deliverable never carries the damage.
                out_frame = np.ascontiguousarray(frame)
                rejected.append(
                    {
                        "frame": idx,
                        "reason": reason,
                        "background_damage_frac": audit.get("background_damage_frac"),
                    }
                )
                log.info("帧 %d 替换结果被拦截（%s），写入原帧", idx, reason)
            else:
                out_frame = np.ascontiguousarray(result["image"])
            writer.stdin.write(out_frame.tobytes())
            rows.append(
                {
                    "frame": idx,
                    "seam_ratio": audit.get("seam_ratio"),
                    "background_damage_frac": audit.get("background_damage_frac"),
                    "verdict": audit.get("verdict"),
                    "written": "original" if reason is not None else "replaced",
                    "reject_reason": reason,
                }
            )
            if result["notes"] and len(notes) < 5:
                notes.extend(result["notes"][: 5 - len(notes)])
            idx += 1
    finally:
        cap.release()
        writer.stdin.close()
        writer.wait()

    if idx == 0:
        raise RuntimeError(f"没有处理任何帧：{video_path}")

    scored = [r for r in rows if r["seam_ratio"] is not None]
    seams = [r["seam_ratio"] for r in scored]
    worst = sorted(scored, key=lambda r: r["seam_ratio"], reverse=True)[:10]
    review = sum(1 for r in rows if r["verdict"] == "review")
    reason_counts: dict[str, int] = {}
    for item in rejected:
        reason_counts[item["reason"]] = reason_counts.get(item["reason"], 0) + 1
    return {
        "frames": idx,
        "seconds": idx / fps,
        "width": width,
        "height": height,
        "fps": fps,
        "out_path": str(out_path),
        "audited": len(scored),
        "seam_ratio_mean": round(float(np.mean(seams)), 3) if seams else None,
        "seam_ratio_max": round(float(np.max(seams)), 3) if seams else None,
        "review_frames": review,
        "worst_frames": worst,
        "rejected_frames": len(rejected),
        "reject_reason_counts": reason_counts,
        "rejected": rejected,
        "notes": notes,
    }
