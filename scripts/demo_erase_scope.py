"""A/B the erase-scope fix on one real frame, and print the audit numbers.

The old compositor erased the actor's whole body below the neck and
inpainted the hole, then pasted the render on top. When the render covers
less than the actor (a slimmer CG body, or arms the model got wrong), the
inpainted smear is exactly what stays visible — and because the hole was
body-sized, Telea's diffusion had nothing good to pull from.

The new compositor erases only the pixels the render does not cover, so the
strips that stay visible are inpainted from the real background on both
sides.

Run:
    python scripts/demo_erase_scope.py [image] [out_dir]

Nothing here is a product path: it exists to keep the fix honest and to
make the numbers reproducible after any later change to composite_frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recast.quality.replace_audit import audit_composite  # noqa: E402
from recast.stages.rerender import composite_frame, neck_cut_y  # noqa: E402

DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "开局即打-示例剧目"
    / "参考图"
    / "01-温念.jpg"
)


def person_mask(bgr: np.ndarray) -> np.ndarray:
    """Person silhouette for the demo.

    The product path (``composite_clip``) calls
    ``mp.solutions.selfie_segmentation``, which mediapipe 0.10.35 removed
    outright — that call now raises AttributeError. Rather than pull a
    .tflite in a demo, this falls back to background differencing on a
    studio-plate portrait, which is good enough for an erase-scope A/B and
    is labelled as a fallback so it is never mistaken for the real masker.
    """
    try:
        import mediapipe as mp

        seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        try:
            res = seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            seg.close()
        mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    except (AttributeError, ImportError):
        print("（mediapipe solutions 不可用，改用背景差分取 mask）")
        return _mask_by_background(bgr)


def _mask_by_background(bgr: np.ndarray) -> np.ndarray:
    border = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    m = (diff > 28).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == keep).astype(np.uint8) * 255
    # fill interior holes (a jacket gap would otherwise be "background")
    flood = m.copy()
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    return m | cv2.bitwise_not(flood)


def neck_line(bgr: np.ndarray, fallback_ratio: float = 0.42) -> int:
    """Neck cut from pose landmarks, or a proportional guess if none."""
    h = bgr.shape[0]
    try:
        import mediapipe as mp

        pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
        try:
            res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            pose.close()
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            nose_y = lm[mp.solutions.pose.PoseLandmark.NOSE].y * h
            ls = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y * h
            rs = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y * h
            frame = {
                "keypoints": [
                    {"name": "nose", "y": nose_y},
                    {"name": "left_shoulder", "y": ls},
                    {"name": "right_shoulder", "y": rs},
                ]
            }
            return neck_cut_y(frame)
    except (AttributeError, ImportError):
        pass
    return int(h * fallback_ratio)


def slim_standin(mask: np.ndarray, neck_y: int, shrink: int = 14) -> np.ndarray:
    """A narrower 'CG body' silhouette in flat grey — a stand-in, not a model.

    Deliberately not photoreal: the point of the comparison is the
    background behind it, and a realistic body would distract from that.
    """
    h, w = mask.shape
    body = mask.copy()
    body[:neck_y, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1, 3))
    body = cv2.erode(body, kernel)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = (150, 150, 150)
    noise = np.random.default_rng(7).integers(-6, 7, size=(h, w, 1))
    rgba[:, :, :3] = np.clip(
        rgba[:, :, :3].astype(int) + noise, 0, 255
    ).astype(np.uint8)
    rgba[:, :, 3] = body
    return rgba


def composite_old(orig, render, mask, neck_y, feather_px=8):
    """The previous behaviour, kept here only to measure it."""
    h, w = orig.shape[:2]
    erase = np.zeros((h, w), dtype=np.uint8)
    erase[neck_y:, :] = mask[neck_y:, :]
    erase = cv2.dilate(erase, np.ones((9, 9), np.uint8))
    inpainted = cv2.inpaint(orig, erase, 5, cv2.INPAINT_TELEA)

    alpha = render[:, :, 3].astype(np.float32) / 255.0
    alpha[: max(0, neck_y - feather_px), :] = 0.0
    ramp = np.linspace(0.0, 1.0, 2 * feather_px + 1, dtype=np.float32)
    lo, hi = max(0, neck_y - feather_px), min(h, neck_y + feather_px + 1)
    if hi > lo:
        alpha[lo:hi, :] *= ramp[: hi - lo, None]
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    a = alpha[:, :, None]
    out = (inpainted.astype(np.float32) * (1.0 - a) + render[:, :, :3].astype(np.float32) * a)
    out = out.astype(np.uint8)
    out[:neck_y, :] = orig[:neck_y, :]
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (24, 24, 24), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread fails on non-ASCII Windows paths; go through a buffer."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent.parent / "_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        print(f"找不到图片：{src}")
        return 2

    orig = imread_unicode(src)
    if orig is None:
        print(f"读不懂图片：{src}")
        return 2
    mask = person_mask(orig)
    ny = neck_line(orig)
    render = slim_standin(mask, ny)

    old = composite_old(orig, render, mask, ny)
    new = composite_frame(orig, render, mask, ny, color_match=False)

    print(f"图片 {src.name}  {orig.shape[1]}x{orig.shape[0]}  脖子切口 y={ny}")
    for name, img in (("旧做法（擦整身+inpaint）", old), ("新做法（只擦残差）", new)):
        rep = audit_composite(orig, img, mask, ny, render_alpha=render[:, :, 3])
        print(
            f"  {name}: 背景损伤 {rep['background_damage_frac']:.2%} "
            f"(均值 {rep['background_damage_mean']:.1f})  "
            f"接缝 {rep['seam_ratio']:.2f}  "
            f"覆盖 {rep['render_coverage']:.0%}  -> {rep['verdict']}"
        )

    strip = np.hstack(
        [
            label(orig, "original"),
            label(old, "old: erase whole body"),
            label(new, "new: erase uncovered only"),
        ]
    )
    dest = out_dir / "erase_scope_ab.png"
    imwrite_unicode(dest, strip)
    print(f"对比图：{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
