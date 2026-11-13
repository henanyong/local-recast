"""Garment transfer: put the character's costume on the actor, in 2D.

This is the workhorse of the 2D route, and the reason it wins on
show-through: the actor's body, hair and silhouette are never touched, so
there is nothing to seam against and no hair to rebuild. Only the pixels
inside the clothing region change.

The naive version — paste the reference garment over the region — reads as
a sticker: flat, wrong lighting, no folds. So the transfer is split by
frequency:

- **Colour** comes from the reference (that is the point: a different
  costume).
- **Low-frequency luminance** comes from the frame. This is the scene's
  key light, shadow under the chin, falloff toward the frame edge. Without
  it the garment floats.
- **High-frequency luminance** comes from the frame. These are the folds,
  creases and fabric texture that follow *this* body. Without them the
  garment looks painted on.

Alpha comes from the person mask, feathered, so the edge lands on the
actor's own outline rather than a rectangle.

Nothing here needs a model, a GPU, or a download: it runs today, in
82 ms/frame at 864x1152, which is what "日产多集" requires. The upgrade
path (SCHP per-garment masks, so upper/lower/coat are transferred
separately) plugs into :func:`split_body` without touching the transfer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("recast")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_SHADE_STRENGTH = 1.0
DEFAULT_DETAIL_STRENGTH = 0.85
DEFAULT_FEATHER = 9
# The horizontal cut at the neck (and at the waist for ``upper``) is a step in
# the mask. Left hard it reads as a rectangle drawn across the garment —
# measured on a cardigan: the shoulders stayed wine while the chest turned
# green, which looks like a bug even though every pixel is where it should be.
# Ramping the outermost rows turns the step into a falloff the eye accepts.
DEFAULT_EDGE_FEATHER = 20
# Chroma (|a-128|+|b-128| in Lab) below FLOOR is treated as neutral and left
# alone; above FULL it is recoloured at full strength. Between the two the
# guard ramps, so a collar does not get a visible outline. These only apply
# when the region itself has no dominant hue (see NEUTRAL_MAIN_CHROMA): on a
# chromatic garment the relative guard below replaces them.
CHROMA_FLOOR = 5.0
CHROMA_FULL = 14.0
# --- relative neutral guard -------------------------------------------------
# The absolute ramp above cannot tell "a shadow on the garment" from "a white
# collar": knitwear measures chroma 22.8 in the light and 5-11 in the shadow,
# so an absolute floor left every shadow un-recoloured and the cardigan came
# out mottled. What actually separates garment-body pixels from foreign ones
# is the chroma *direction relative to the region's own main colour*: a shadow
# keeps the garment's hue (measured wine cardigan: lit (a+49, b+23), shadow
# (a+27, b+11) — within a few degrees), while a collar or a tie sits far off
# it. So the guard ramps over hue angle instead of chroma magnitude.
REL_HUE_FULL = 22.0  # degrees from the main hue: full recolour weight
REL_HUE_ZERO = 50.0  # degrees: a genuinely different colour, zero weight
# Below this chroma the hue is pure noise (a white collar measures ~0), so
# the pixel is protected regardless of angle; the ramp back to full weight is
# ABS_NEUTRAL_RAMP wide. Knit shadows (chroma >= 5) clear it.
ABS_NEUTRAL_FLOOR = 1.0
ABS_NEUTRAL_RAMP = 4.0
# If the region's own main chroma is below this there is no hue to compare
# against (grey suit, white shirt) and the old absolute ramp is used.
NEUTRAL_MAIN_CHROMA = 4.0
# A bright near-neutral pixel is a foreign white (collar, shirt highlight),
# NOT garment shadow: a fold is always darker than the region's main
# luminance, a collar is always lighter. Measured on the real frame: the
# dyed-collar streak sat at chroma 2-5 / L 130-175 while the knit's main
# luminance is ~55 — the L margin separates them cleanly. The chroma cap is
# scaled by the main chroma so a low-chroma garment never protects its own
# bright folds.
NEUTRAL_CHROMA_MAX = 10.0
NEUTRAL_L_MARGIN = 30.0
# --- print exemption (connected components) ---------------------------------
# A red flower printed on a cream blouse sits at the SAME hue as a wine
# cardigan, so no per-pixel colour rule can spare it — and a neighbourhood
# share test fails too: inside a dense print the protected pixels never reach
# a majority. What identifies a print is structure, not share: the printed
# flower is a SMALL connected blob of hue-aligned pixels surrounded by a low
# -chroma / foreign-hue base, while the garment body is one huge blob (or
# surrounded by its own shadow, which is hue-aligned and thus not "base").
# So: find hue-aligned blobs, and exempt the small ones whose surrounding
# ring reads as printed-fabric base.
PRINT_BLOB_MAX_AREA = 20000   # px; real flowers measure 0.5-3k, the cardigan blob 100k+
PRINT_SURROUND_PX = 21        # ring width around the blob
PRINT_SURROUND_MIN = 0.30     # base share in the ring that triggers exemption
# A ring pixel counts as "printed-fabric base" when it is either foreign-hued
# (cream ~+40 deg off a wine main, blue flowers ~-100) or nearly neutral
# (white ground). Foreign-but-saturated neighbours (another garment of a
# different colour, chroma 60+) do NOT count: that is how a real cardigan
# pressed against something else stays exempt.
PRINT_FOREIGN_HUE_DEG = 30.0
PRINT_BASE_MAG_MAX = 28.0
PRINT_NEUTRAL_MAG = 4.0
# --- skin guard -------------------------------------------------------------
# skin_mask's YCrCb rule also matches mid-dark reds: a wine cardigan pixel
# measured Y=83 / Cr=162 / Cb=126, comfortably inside the rule, so every such
# garment pixel was skipped and the recolour went patchy. Real skin is LIT
# skin — in the footage hands and necks measure Y 130-230, the garment
# misdetects sit below ~110. The skip therefore only fires on bright pixels;
# dark "skin" is treated as garment (shadow) and recoloured.
SKIN_SKIP_MIN_Y = 115.0
# Shadowed skin escapes that Y gate too: the chest V under the chin measured
# Y 90-125 with hue +29..+50 degrees, and got dyed pale green. Skin is always
# WARMER than a red/wine main hue, so skin-flagged pixels inside this warm
# band off the main hue are skipped as well — the garment misdetects sit ON
# the main hue (offset ~0) and stay recoloured.
SKIN_WARM_OFFSET_MIN = 20.0
SKIN_WARM_OFFSET_MAX = 80.0
# Auto garment pick: a colour cluster only counts as "the garment" if its own
# chroma is above this. Measured on a grey blazer over a wine top: blazer
# clusters sit at chroma 1.6 / 5.2 / 11.2, the top at 18.4 — 14 separates them
# with margin on both sides.
GARMENT_MIN_CHROMA = 14.0
GARMENT_MIN_SHARE = 0.06
GARMENT_SPLIT_CLUSTERS = 5
# k-means is O(pixels x k x iters). On a 1080x1920 torso that is ~470k samples
# per frame and it doubled a clip run from 20 to 42 minutes. Fitting on a
# bounded subsample and then assigning every pixel to its nearest centre gives
# the same clusters — the centres are what matter, not how many points voted.
SPLIT_SAMPLE_CAP = 60000
# A person silhouette never fills its own bounding box. Above this the mask is
# a rectangle, which means segmentation leaked into the background — the
# recolour then follows the rectangle, and that is the single most confusing
# failure mode to debug from the output image alone.
RECTANGLE_FILL = 0.95


class GarmentError(RuntimeError):
    """The costume cannot be transferred (no reference, no region, bad input)."""


def _as_mask(mask: Any) -> Any:
    import numpy as np

    arr = np.asarray(mask)
    return arr if arr.dtype == bool else arr > 0


def split_body(mask: Any, neck_y: int, *, split_ratio: float = 0.55) -> dict[str, Any]:
    """Split the clothed body into upper and lower parts.

    ``split_ratio`` is where the waist sits between the neck cut and the
    actor's feet. A hip landmark would be better; without one this ratio is
    wrong for a low-angle shot, so the caller can pass an explicit
    ``split_y`` through :func:`transfer_garment` instead.
    """
    import numpy as np

    person = _as_mask(mask)
    h = person.shape[0]
    body = person.copy()
    body[:neck_y, :] = False
    if not body.any():
        raise GarmentError("脖子以下没有人物像素，无法换装")
    ys, _ = np.where(body)
    bottom = int(ys.max())
    split_y = int(neck_y + (bottom - neck_y) * float(split_ratio))
    upper = body.copy()
    upper[split_y:, :] = False
    lower = body.copy()
    lower[:split_y, :] = False
    return {"upper": upper, "lower": lower, "split_y": split_y, "bottom": bottom}


def _region_mask(mask: Any, neck_y: int, part: str, split_y: int | None) -> Any:
    import numpy as np

    if part == "full":
        region = _as_mask(mask).copy()
        region[:neck_y, :] = False
        return region
    parts = split_body(mask, neck_y)
    if part == "upper":
        region = parts["upper"].copy()
    elif part == "lower":
        region = parts["lower"].copy()
    else:
        raise GarmentError(f"未知部位：{part}（可用 full / upper / lower）")
    if split_y is not None:
        # Honour an explicit waist line (hip landmark) instead of the ratio.
        region = _as_mask(mask).copy()
        region[:neck_y, :] = False
        if part == "upper":
            region[int(split_y) :, :] = False
        else:
            region[: int(split_y), :] = False
    return region


def split_by_colour(
    frame_bgr: Any,
    region: Any,
    *,
    clusters: int = 3,
    min_share: float = 0.06,
) -> list[dict[str, Any]]:
    """Split a clothed region into its garments by colour.

    SCHP would give named parts (upper-clothes / dress / coat) but needs a
    model download. In the footage this runs on, a cardigan and a floral
    blouse are simply *different colours*, so k-means on Lab chroma separates
    them well enough to "recolour only the jacket and leave the shirt alone" —
    which is the actual operation people ask for.

    It is not a parser: same-coloured layers merge, and a busy print can split
    into several clusters. Parts are returned largest-first with their share,
    and clusters below ``min_share`` are dropped as noise.
    """
    import cv2
    import numpy as np

    frame = np.asarray(frame_bgr)
    mask = _as_mask(region)
    if not mask.any():
        raise GarmentError("区域里没有像素，无法分件")
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[mask][:, 1:3]  # chroma only: lighting must not drive the split
    if samples.shape[0] < clusters * 8:
        raise GarmentError("区域太小，分不出件")

    k = int(max(1, min(clusters, samples.shape[0] // 8)))
    import numpy as _np

    rng = _np.random.default_rng(0)  # deterministic, so a preview is repeatable
    if samples.shape[0] > SPLIT_SAMPLE_CAP:
        fit = samples[rng.choice(samples.shape[0], SPLIT_SAMPLE_CAP, replace=False)]
    else:
        fit = samples
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(
        fit.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = _assign_to_centres(samples, centres)

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    box_h, box_w = y1 - y0, x1 - x0
    local_y = ys - y0
    local_x = xs - x0
    kernel = np.ones((9, 9), np.uint8)
    parts: list[dict[str, Any]] = []
    total = float(samples.shape[0])
    for index in range(k):
        chosen = labels == index
        share = float(chosen.sum()) / total
        if share < min_share:
            continue
        # Build and close inside the region's own box: the closing is the
        # expensive step and it must not run over 2M background pixels five
        # times per frame.
        part_box = np.zeros((box_h, box_w), dtype=np.uint8)
        part_box[local_y[chosen], local_x[chosen]] = 255
        # Fill the speckle inside a garment: a print leaves holes that would
        # otherwise turn the recolour into confetti.
        part_box = cv2.morphologyEx(part_box, cv2.MORPH_CLOSE, kernel)
        part_mask = np.zeros(mask.shape, dtype=np.uint8)
        part_mask[y0:y1, x0:x1] = part_box
        parts.append(
            {
                "index": int(index),
                "mask": part_mask > 0,
                "share": round(share, 4),
                "chroma": [round(float(centres[index][0]), 1), round(float(centres[index][1]), 1)],
                "pixels": int(chosen.sum()),
            }
        )
    if not parts:
        raise GarmentError("分件结果都是噪声（每块占比都低于下限）")
    parts.sort(key=lambda item: -item["pixels"])
    for order, part in enumerate(parts):
        part["order"] = order
    return parts


def _assign_to_centres(samples: Any, centres: Any) -> Any:
    """Nearest centre for every sample, via a 256x256 chroma lookup table.

    Chroma is integer-valued in Lab here, so a table is exact and turns the
    per-sample distance loop into 65k comparisons computed once. Measured on a
    1080x1920 torso: 470k samples went from ~90 ms to ~4 ms, which is what made
    per-frame garment selection affordable for a clip.
    """
    import numpy as np

    pts = np.asarray(samples, dtype=np.float32)
    ctr = np.asarray(centres, dtype=np.float32)
    grid_a, grid_b = np.meshgrid(
        np.arange(256, dtype=np.float32), np.arange(256, dtype=np.float32), indexing="ij"
    )
    flat = np.stack([grid_a.ravel(), grid_b.ravel()], axis=1)
    d = ((flat[:, None, :] - ctr[None, :, :]) ** 2).sum(axis=2)
    lut = np.argmin(d, axis=1).astype(np.int32).reshape(256, 256)
    a = np.clip(pts[:, 0], 0, 255).astype(np.int32)
    b = np.clip(pts[:, 1], 0, 255).astype(np.int32)
    return lut[a, b]


def _pick_part(
    frame_bgr: Any,
    mask: Any,
    neck_y: int,
    part: str,
    split_y: int | None,
    cluster: int | None,
    auto_garment: bool = False,
) -> tuple[Any, str]:
    """Resolve ``part`` / ``cluster`` / auto-pick into one boolean region."""
    if cluster is not None:
        if part not in {"full", "upper", "lower"}:
            raise GarmentError(f"未知部位：{part}（可用 full / upper / lower）")
        base = _region_mask(mask, neck_y, part, split_y)
        parts = split_by_colour(frame_bgr, base)
        if not 0 <= int(cluster) < len(parts):
            raise GarmentError(
                f"这件衣服只分出 {len(parts)} 块，取不到第 {int(cluster) + 1} 块"
            )
        picked = parts[int(cluster)]
        return picked["mask"], f"{part}:cluster{picked['order']}"
    base = _region_mask(mask, neck_y, part, split_y)
    if auto_garment:
        picked = _auto_garment(frame_bgr, base)
        if picked is not None:
            return picked["mask"], f"{part}:auto{picked['order']}"
        # Nothing chromatic to isolate (an all-grey outfit): fall back to the
        # whole region and let the neutral guard decide pixel by pixel.
        return base, f"{part}:auto-all"
    return base, part


def skin_mask(bgr: Any) -> Any:
    """True where the pixel looks like skin (YCrCb rule, luminance-gated).

    Without this the recolour dyes hands, neck and ears along with the
    garment. On a frame where the actor wears a dark red top the hand measured
    the same Lab chroma as the fabric (149/130), so no chroma threshold can
    separate them — but skin is far brighter, and the YCrCb rule separates the
    two cleanly at that brightness. Measured on a real frame: 50 426 skin
    pixels in the upper body, the wine top untouched.
    """
    import cv2
    import numpy as np

    ycc = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0].astype(np.int16)
    cr = ycc[:, :, 1].astype(np.int16)
    cb = ycc[:, :, 2].astype(np.int16)
    return (y > 70) & (y < 250) & (cr >= 137) & (cr <= 175) & (cb >= 85) & (cb <= 128)


def _auto_garment(frame_bgr: Any, region: Any) -> dict[str, Any] | None:
    """The colour cluster most likely to be the garment, or None.

    ``part="upper"`` covers the blazer, the blouse, the arms and the hands;
    recolouring all of it turns a navy suit green *and* the actor's hands
    green. What a caller means by "change the costume" is the chromatic,
    compact, non-skin part — this picks it by colour, which needs no human
    parsing model and no extra download.
    """
    import numpy as np

    parts = split_by_colour(
        frame_bgr, region, clusters=GARMENT_SPLIT_CLUSTERS, min_share=0.02
    )
    skin = skin_mask(frame_bgr)
    best: dict[str, Any] | None = None
    for part in parts:
        chroma = abs(part["chroma"][0] - 128.0) + abs(part["chroma"][1] - 128.0)
        if chroma < GARMENT_MIN_CHROMA:
            continue
        if part["share"] < GARMENT_MIN_SHARE:
            continue
        mask = part["mask"]
        overlap = float((mask & skin).sum()) / max(1, int(mask.sum()))
        if overlap > 0.5:  # mostly skin: an arm, not a sleeve
            continue
        if best is None or part["pixels"] > best["pixels"]:
            best = part
    return best


def _lowpass(channel: Any, sigma: float) -> Any:
    """Gaussian blur used by the texture mode to split shade from detail."""
    import cv2

    return cv2.GaussianBlur(channel, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def _reference_chroma(reference_bgr: Any) -> tuple[float, float, dict[str, Any]]:
    """Median garment chroma of a reference crop, ignoring background-ish pixels.

    The reference is whatever the caller has: usually a body crop from a
    portrait, which contains background, hands and a bag. Taking the median
    of the middle band and dropping near-white/near-grey pixels gives the
    garment's own colour instead of the average of the photo.
    """
    import cv2
    import numpy as np

    lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    band = lab[int(h * 0.25) : int(h * 0.9), int(w * 0.15) : int(w * 0.85)]
    if band.size == 0:
        band = lab
    flat = band.reshape(-1, 3)
    a = flat[:, 1]
    b = flat[:, 2]
    # Low-chroma pixels are background wall, white shirt collar or shadow.
    keep = (np.abs(a - 128.0) + np.abs(b - 128.0)) > 6.0
    if int(keep.sum()) >= 64:
        a, b = a[keep], b[keep]
    return float(np.median(a)), float(np.median(b)), {"kept": int(a.size)}


def transfer_garment(
    frame_bgr: Any,
    person_mask: Any,
    neck_y: int,
    reference_bgr: Any,
    *,
    part: str = "upper",
    split_y: int | None = None,
    cluster: int | None = None,
    auto_garment: bool = False,
    mode: str = "recolor",
    chroma_strength: float = 0.9,
    neutral_guard: bool = True,
    skin_guard: bool = True,
    shade_strength: float = DEFAULT_SHADE_STRENGTH,
    detail_strength: float = DEFAULT_DETAIL_STRENGTH,
    feather: int = DEFAULT_FEATHER,
    edge_feather: int = DEFAULT_EDGE_FEATHER,
) -> dict[str, Any]:
    """Return the frame with the costume applied inside the body region.

    Two modes, because they make very different assumptions:

    ``recolor`` (default) — keep every pixel's lightness and the garment's
    own pattern, and move the colour centre onto the reference garment's
    colour. Lightness carries the folds, the shadows and the light
    direction, so leaving it untouched is what keeps the result from
    looking pasted on. This is the mode that works with the references a
    character pack actually has: a crop of a different costume photo.

    ``cluster`` — recolour only one colour-cluster of the region (see
    :func:`split_by_colour`), so the jacket can change while the blouse
    underneath stays. ``cluster=None`` means the whole region.

    ``auto_garment`` — pick the cluster for you: the largest chromatic,
    non-skin one. This is what makes ``part="upper"`` mean "the garment"
    instead of "everything from the neck down", which otherwise dyes the
    blazer, the arms and the hands.

    ``texture`` — stretch the reference over the region and keep the
    frame's shade and detail. This only makes sense when the reference is a
    real garment texture already aligned to a body, which a portrait crop is
    NOT (it contains background, hands and bags, and stretching it produces
    a collage rather than a costume). Kept for the case where a proper
    garment plate exists.
    """
    import cv2
    import numpy as np

    if frame_bgr is None or reference_bgr is None:
        raise GarmentError("换装需要原始帧和服装参考图")
    if reference_bgr.size == 0:
        raise GarmentError("服装参考图是空的")
    if mode not in {"recolor", "texture"}:
        raise GarmentError(f"未知换装方式：{mode}（可用 recolor / texture）")
    region, resolved = _pick_part(
        frame_bgr, person_mask, neck_y, part, split_y, cluster, auto_garment
    )
    if not region.any():
        raise GarmentError(f"「{part}」区域里没有人物像素，无法换装")

    if mode == "recolor":
        image, extra = _recolor(
            frame_bgr,
            region,
            reference_bgr,
            chroma_strength,
            neutral_guard=neutral_guard,
            skin_guard=skin_guard,
        )
    else:
        image, extra = _texture(
            frame_bgr, region, reference_bgr, shade_strength, detail_strength
        )

    # --- feathered alpha on the actor's own outline ---
    # A plain Gaussian on the mask tints a band ~3 sigma wide *outside* the
    # body, which puts the costume colour onto the background — a halo, and
    # a halo is show-through. Confine the ramp to a narrow ring so the
    # background stays bit-identical beyond it.
    alpha = region.astype(np.float32)
    if edge_feather > 0:
        # Soften the horizontal cuts (neck above, waist/feet below). The sides
        # are handled by the outline feather; these two are a hard step in the
        # mask and are what makes the result read as a pasted rectangle.
        rows = np.arange(alpha.shape[0], dtype=np.float32)[:, None]
        span = float(edge_feather)
        ys_any = np.where(region.any(axis=1))[0]
        if ys_any.size:
            top = float(ys_any.min())
            bottom = float(ys_any.max())
            ramp_top = np.clip((rows - top) / span, 0.0, 1.0)
            ramp_bot = np.clip((bottom - rows) / span, 0.0, 1.0)
            if part != "lower":
                alpha *= ramp_top
            if part != "upper":
                alpha *= ramp_bot
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(feather) / 3.0)
        spill = max(1, int(feather) // 2)
        ring = cv2.dilate(
            region.astype(np.uint8),
            np.ones((2 * spill + 1, 2 * spill + 1), np.uint8),
        ).astype(np.float32)
        alpha = alpha * ring
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

    # Blend only where alpha is non-zero (region + feather ring): the rest of
    # the frame is bit-identical and needs no float round trip.
    touched = alpha[:, :, 0] > 0
    out = np.asarray(frame_bgr).copy()
    if touched.any():
        ty, tx = np.where(touched)
        by0, by1 = int(ty.min()), int(ty.max()) + 1
        bx0, bx1 = int(tx.min()), int(tx.max()) + 1
        sub_a = alpha[by0:by1, bx0:bx1]
        sub = (
            np.asarray(frame_bgr)[by0:by1, bx0:bx1].astype(np.float32) * (1.0 - sub_a)
            + image[by0:by1, bx0:bx1].astype(np.float32) * sub_a
        )
        out[by0:by1, bx0:bx1] = np.clip(sub, 0, 255).astype(np.uint8)
    ys, xs = np.where(region)
    box_w = int(xs.max() - xs.min() + 1)
    box_h = int(ys.max() - ys.min() + 1)
    fill = float(region.sum()) / max(1, box_w * box_h)
    warnings: list[str] = []
    if fill >= RECTANGLE_FILL:
        warnings.append(
            f"区域填充率 {fill:.0%}，几乎是一个矩形：分割没有贴合人物，"
            "换装会沿矩形边界生效。换一个分割模型或换一帧再试。"
        )
    return {
        "image": out,
        "region_pixels": int(region.sum()),
        "part": resolved,
        "mode": mode,
        "shade_strength": shade_strength,
        "detail_strength": detail_strength,
        "chroma_strength": chroma_strength,
        "neutral_guard": neutral_guard,
        "skin_guard": skin_guard,
        "auto_garment": auto_garment,
        "reference_chroma": extra,
        "region_fill": round(fill, 4),
        "warnings": warnings,
        "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
    }


def _region_main_chroma(a: Any, b: Any, box_region: Any) -> tuple[float, float]:
    """The garment's own hue direction, as a chroma vector.

    Component-wise medians fail on a multi-hue region: a 50/50 red/green body
    has median (a+3, b+44) — a phantom hue matching NEITHER half, and the
    guard then starves both. So the main colour is the *magnitude-weighted
    peak of the hue histogram*: real garments are one coherent hue mass, so
    the peak lands on the garment even with a print or skin in the region.
    Near-neutral pixels (mag <= 2) carry no direction and are excluded; a
    region too small or too neutral to have a peak falls back to the median.
    """
    import numpy as np

    if not bool(box_region.any()):
        return 0.0, 0.0
    va = (a - 128.0)[box_region]
    vb = (b - 128.0)[box_region]
    mag = np.sqrt(va * va + vb * vb)
    keep = mag > 2.0
    if int(keep.sum()) < 64:
        # An almost-neutral region: no hue to peak on, keep the old median.
        return float(np.median(va)), float(np.median(vb))
    va, vb, mag = va[keep], vb[keep], mag[keep]
    # The peak is a histogram mode: a bounded deterministic subsample gives
    # the same peak at a fraction of the atan2/histogram cost (measured on a
    # 1080x1920 torso: 46 ms -> 9 ms, peak within one bin).
    stride = max(1, int(va.size) // 100000)
    if stride > 1:
        va, vb, mag = va[::stride], vb[::stride], mag[::stride]
    ang = np.degrees(np.arctan2(vb, va))  # -180..180
    bins = 144  # 2.5 degrees per bin: fine enough for the guard's ramp scale
    hist, edges = np.histogram(ang, bins=bins, range=(-180.0, 180.0), weights=mag)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel = kernel / kernel.sum()
    wrapped = np.concatenate([hist[-2:], hist, hist[:2]])
    smooth = np.convolve(wrapped, kernel, mode="same")[2:-2]
    peak = float(edges[int(np.argmax(smooth))]) + (edges[1] - edges[0]) / 2.0
    # The main vector is the mean chroma of the pixels around the peak, so it
    # carries the peak family's own saturation, not a unit direction.
    offset = (ang - peak + 180.0) % 360.0 - 180.0
    sel = np.abs(offset) <= 30.0
    return float(np.mean(va[sel])), float(np.mean(vb[sel]))


_HUE_GUARD_LUT: Any = None
_HUE_ANGLE_LUT: Any = None


def _relative_guard(a: Any, b: Any, main_da: float, main_db: float) -> tuple[Any, Any]:
    """Per-pixel recolour weight and signed hue offset, from the main colour.

    A pixel counts as garment body when its chroma vector points the same way
    as the region's main colour — at ANY magnitude, so shadows and highlights
    recolour together. Foreign colours (collar, tie, a printed flower of a
    different hue) ramp to zero by ``REL_HUE_ZERO`` degrees. Chroma below
    ``ABS_NEUTRAL_FLOOR`` has no stable direction, so it is protected by an
    absolute ramp: that is what keeps a white collar white.

    Returns ``(guard, off)`` where ``off`` is the pixel's signed hue offset
    from the main hue in degrees (positive = warmer than the main hue). The
    print-exemption pass needs it to tell a cream/white base from a garment
    shadow without recomputing arctan2. The angle is read from a cosine
    lookup table: arccos on a 1080x1920 torso costs ~15 ms per frame, the
    LUT the same math in ~3 ms with < 0.3 degrees of error.
    """
    import numpy as np

    global _HUE_GUARD_LUT, _HUE_ANGLE_LUT
    va = a - 128.0
    vb = b - 128.0
    mag = np.sqrt(va * va + vb * vb)
    main_mag = float(np.sqrt(main_da * main_da + main_db * main_db))
    if main_mag < 1e-6:
        zeros = np.zeros(mag.shape, dtype=np.float32)
        return zeros, zeros.copy()
    dot = va * main_da + vb * main_db
    cosang = np.clip(dot / np.maximum(mag * main_mag, 1e-6), -1.0, 1.0)
    if _HUE_GUARD_LUT is None:
        cos_grid = np.linspace(-1.0, 1.0, 2048)
        angle_grid = np.degrees(np.arccos(cos_grid))
        span = max(1e-6, REL_HUE_ZERO - REL_HUE_FULL)
        _HUE_GUARD_LUT = np.clip(
            (REL_HUE_ZERO - angle_grid) / span, 0.0, 1.0
        ).astype(np.float32)
        _HUE_ANGLE_LUT = angle_grid.astype(np.float32)
    idx = np.clip(((cosang + 1.0) * 1023.5).astype(np.int32), 0, 2047)
    hue_ramp = _HUE_GUARD_LUT[idx]
    off = _HUE_ANGLE_LUT[idx] * np.sign(dot)
    abs_ramp = np.clip((mag - ABS_NEUTRAL_FLOOR) / ABS_NEUTRAL_RAMP, 0.0, 1.0)
    return (hue_ramp * abs_ramp).astype(np.float32), off.astype(np.float32)


def _print_blobs(guard: Any, off: Any, mag: Any, box_region: Any) -> Any:
    """Hue-aligned small blobs that float in a printed-fabric base, as a mask.

    Structure, not neighbourhood share, identifies a print: a flower is a
    small connected component of hue-aligned pixels ringed by low-chroma /
    foreign-hue ground, while the garment body is one component far above
    ``PRINT_BLOB_MAX_AREA`` — or pressed against other SATURATED garments,
    whose chroma disqualifies them as base (a wine cardigan leaning on a
    green couch must NOT be exempted just because the couch is foreign-hued).

    Returns a boolean mask of the exempted blobs (their recolour weight
    becomes zero).
    """
    import cv2
    import numpy as np

    aligned = ((guard > 0.5) & box_region).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(aligned, connectivity=8)
    # Base = foreign-hued low-chroma ground (cream, blue flowers) or true
    # neutral (white). A foreign-hued but saturated neighbour does not count.
    base = (
        ((np.abs(off) > PRINT_FOREIGN_HUE_DEG) & (mag < PRINT_BASE_MAG_MAX))
        | (mag < PRINT_NEUTRAL_MAG)
    ) & box_region
    kernel = np.ones((PRINT_SURROUND_PX, PRINT_SURROUND_PX), np.uint8)
    blob_mask = np.zeros(box_region.shape, dtype=bool)
    small_ids = [
        i
        for i in range(1, num)
        if stats[i, cv2.CC_STAT_AREA] <= PRINT_BLOB_MAX_AREA
    ]
    if not small_ids:
        return blob_mask
    if len(small_ids) > 200:
        # Degenerate fragmentation (noisy guard): fall back to one combined
        # ring test instead of per-blob passes.
        small = np.isin(labels, np.array(small_ids, dtype=np.int32))
        ring = (
            cv2.dilate(small.astype(np.uint8), kernel).astype(bool)
            & ~small
            & box_region
        )
        if ring.any() and float(base[ring].mean()) >= PRINT_SURROUND_MIN:
            blob_mask = small
        return blob_mask
    for i in small_ids:
        blob = labels == i
        ring = (
            cv2.dilate(blob.astype(np.uint8), kernel).astype(bool)
            & ~blob
            & box_region
        )
        if not ring.any():
            continue  # blob fills its surroundings: treat as garment body
        if float(base[ring].mean()) >= PRINT_SURROUND_MIN:
            blob_mask |= blob
    return blob_mask


def _recolor(
    frame_bgr: Any,
    region: Any,
    reference_bgr: Any,
    strength: float,
    *,
    neutral_guard: bool = True,
    skin_guard: bool = True,
    chroma_floor: float = CHROMA_FLOOR,
    chroma_full: float = CHROMA_FULL,
) -> tuple[Any, dict]:
    """Shift the garment's chroma centre; leave lightness and pattern alone.

    ``neutral_guard`` keeps foreign colours where they are. The guard is
    *relative*: when the region has a dominant hue (a wine cardigan, a navy
    suit) the weight ramps over each pixel's hue angle to that main colour,
    so a shadow on the knit — chroma 5-11, same hue as the lit band —
    recolours at full strength instead of staying behind and mottling the
    garment. Only pixels far off the main hue (a white collar, a black tie,
    a contrasting print) ramp down. When the region is neutral overall there
    is no hue to compare against and the old absolute chroma ramp applies.

    ``skin_guard`` skips pixels that look like skin — but only *lit* skin
    (Y > SKIN_SKIP_MIN_Y). The YCrCb skin rule also matches mid-dark reds
    (a cardigan measured Y=83/Cr=162, inside the rule); skipping those dyed
    the garment patchy. Dark "skin" is garment shadow and is recoloured.
    """
    import cv2
    import numpy as np

    # Work inside the region's bounding box: a full-frame LAB round trip on
    # 1080x1920 is ~2M pixels of float work for a garment that covers a third
    # of it. Cropping first is the single largest speed win in this path.
    ys, xs = np.where(region)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    lab = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    a_ref, b_ref, extra = _reference_chroma(reference_bgr)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    box_region = region[y0:y1, x0:x1]
    a_mean = float(a[box_region].mean())
    b_mean = float(b[box_region].mean())
    # Re-centre on the reference while keeping the frame's own spread, so the
    # floral print survives as a pattern and only its palette changes.
    a_target = (a - a_mean) + a_ref
    b_target = (b - b_mean) + b_ref
    s = float(np.clip(strength, 0.0, 1.0))

    main_da, main_db = _region_main_chroma(a, b, box_region)
    main_chroma = float(np.hypot(main_da, main_db))
    relative = main_chroma >= NEUTRAL_MAIN_CHROMA
    if neutral_guard:
        if relative:
            guard, off = _relative_guard(a, b, main_da, main_db)
            extra["guard_mode"] = "relative_hue"
            va = a - 128.0
            vb = b - 128.0
            mag = np.sqrt(va * va + vb * vb)
            # Bright near-neutral pixels are foreign whites, not garment
            # shadow (see NEUTRAL_L_MARGIN): kill their weight outright.
            l_plane = lab[:, :, 0]
            l_main = float(np.median(l_plane[box_region]))
            chroma_cap = min(NEUTRAL_CHROMA_MAX, 0.5 * main_chroma)
            bright_neutral = (mag < chroma_cap) & (
                l_plane > l_main + NEUTRAL_L_MARGIN
            )
            guard = np.where(bright_neutral, 0.0, guard)
            # Print exemption: a hue-aligned SMALL blob whose surrounding ring
            # reads as printed-fabric base (cream ground, white, blue flowers)
            # is a flower of the shirt, not of the garment. The garment body is
            # either one blob far above the size cap, or pressed against other
            # SATURATED garments whose chroma disqualifies them as "base".
            blob_mask = _print_blobs(guard, off, mag, box_region)
            weight = s * guard * (~blob_mask).astype(np.float32)
            extra["print_protected_frac"] = round(
                float(blob_mask.sum()) / max(1, int(box_region.sum())), 4
            )
        else:
            # No dominant hue in the region: fall back to the absolute ramp.
            chroma = np.abs(a - 128.0) + np.abs(b - 128.0)
            span = max(1e-6, float(chroma_full) - float(chroma_floor))
            guard = np.clip((chroma - float(chroma_floor)) / span, 0.0, 1.0).astype(
                np.float32
            )
            extra["guard_mode"] = "absolute_chroma"
            weight = s * guard
        extra["neutral_pixels_protected"] = int(((guard < 0.5) & box_region).sum())
        extra["main_chroma"] = round(main_chroma, 1)
    else:
        weight = np.full(a.shape, s, dtype=np.float32)
        extra["neutral_pixels_protected"] = 0

    if skin_guard:
        box = frame_bgr[y0:y1, x0:x1]
        skin = skin_mask(box)
        y = cv2.cvtColor(box, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        # Dyeing an arm is worse than not changing the garment at all: the eye
        # reads a green hand instantly. But only LIT pixels are plausibly
        # skin — dark skin-rule hits on a red garment are garment shadow, and
        # skipping them is what mottled the knit. The boundary follows the
        # mask rather than a blur, which would reintroduce a coloured halo.
        skip = skin & (y > SKIN_SKIP_MIN_Y)
        if relative:
            # Shadowed skin escapes the Y gate too: the chest V under the
            # chin measured Y 90-125 and got dyed pale green. Skin is always
            # WARMER than a red/wine main hue, so skin-flagged pixels inside
            # a warm band off the main hue are skipped as well — the garment
            # misdetects sit ON the main hue (offset ~0) and stay recoloured.
            # NOTE: a/b are the raw 0-255 planes; the hue needs the 128 centre
            # removed, and the difference wraps into [-180, 180).
            sy, sx = np.where(skin)
            if sy.size:
                main_hue_deg = float(np.degrees(np.arctan2(main_db, main_da)))
                off = (
                    np.degrees(np.arctan2(b[sy, sx] - 128.0, a[sy, sx] - 128.0))
                    - main_hue_deg
                    + 180.0
                ) % 360.0 - 180.0
                warm = (off > SKIN_WARM_OFFSET_MIN) & (off < SKIN_WARM_OFFSET_MAX)
                skip[sy, sx] |= warm
        weight = weight * (1.0 - skip.astype(np.float32))
        extra["skin_pixels_protected"] = int((skip & box_region).sum())
    else:
        extra["skin_pixels_protected"] = 0

    # Fast path: nothing to recolour anywhere (a fully neutral region with no
    # lit skin in it). Returning the frame untouched avoids a Lab round trip
    # whose 8-bit quantisation would otherwise smear ±1 noise over pixels the
    # guard deliberately left alone.
    active = weight > 0
    if skin_guard:
        active = active | skip
    if not bool(active.any()):
        return np.asarray(frame_bgr).copy(), {
            "reference_a": round(a_ref, 1),
            "reference_b": round(b_ref, 1),
            "neutral_guard": neutral_guard,
            **extra,
        }

    lab[:, :, 1] = np.where(box_region, a + weight * (a_target - a), a)
    lab[:, :, 2] = np.where(box_region, b + weight * (b_target - b), b)
    patched = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = np.asarray(frame_bgr).copy()
    out[y0:y1, x0:x1] = patched
    return out, {
        "reference_a": round(a_ref, 1),
        "reference_b": round(b_ref, 1),
        "neutral_guard": neutral_guard,
        **extra,
    }


def _texture(
    frame_bgr: Any, region: Any, reference_bgr: Any, shade_strength: float, detail_strength: float
) -> tuple[Any, dict]:
    """Stretch the reference over the region, keeping the frame's shading."""
    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]
    ys, xs = np.where(region)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    ref = cv2.resize(reference_bgr, (box_w, box_h), interpolation=cv2.INTER_AREA)

    frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    frame_box = frame_lab[y0:y1, x0:x1]
    ref_box = ref_lab
    box_mask = region[y0:y1, x0:x1]

    l_ref = ref_box[:, :, 0]
    l_frame = frame_box[:, :, 0]
    sigma = max(3.0, min(box_w, box_h) / 12.0)
    l_frame_low = _lowpass(l_frame, sigma)
    shade = (
        l_frame_low - float(l_frame_low[box_mask].mean()) if box_mask.any() else l_frame_low * 0.0
    )
    detail = l_frame - l_frame_low

    l_out = l_ref + shade_strength * shade + detail_strength * detail
    lo = float(np.percentile(l_frame[box_mask], 2)) if box_mask.any() else 0.0
    hi = float(np.percentile(l_frame[box_mask], 98)) if box_mask.any() else 255.0
    l_out = np.clip(l_out, max(0.0, lo - 24.0), min(255.0, hi + 24.0))

    out_box = ref_box.copy()
    out_box[:, :, 0] = l_out
    out_lab = frame_lab.copy()
    out_lab[y0:y1, x0:x1] = out_box
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, {"box": [x0, y0, x1, y1]}


def costume_reference(folder: str | Path) -> Path | None:
    """First usable costume screenshot in a character folder, if any."""
    directory = Path(folder) / "costume"
    if not directory.is_dir():
        return None
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


_IMAGE_CACHE: dict[tuple[str, int, int], Any] = {}
_IMAGE_CACHE_LIMIT = 8


def read_image(path: str | Path) -> Any:
    """cv2.imread that survives non-ASCII Windows paths.

    Cached by (path, mtime, size): a clip run asks for the same costume
    reference once per frame, and re-decoding it 12000 times would cost more
    than the transfer itself. Callers must treat the result as read-only.
    """
    import cv2
    import numpy as np

    file_path = Path(path)
    try:
        stat = file_path.stat()
        key = (str(file_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        key = None
    if key is not None and key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key]

    data = np.fromfile(str(file_path), dtype=np.uint8)
    if data.size == 0:
        raise GarmentError(f"找不到或读不出图片：{path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise GarmentError(f"解码失败：{path}")
    if key is not None:
        if len(_IMAGE_CACHE) >= _IMAGE_CACHE_LIMIT:
            _IMAGE_CACHE.pop(next(iter(_IMAGE_CACHE)))
        _IMAGE_CACHE[key] = image
    return image


def transfer_character_costume(
    frame_bgr: Any,
    person_mask: Any,
    neck_y: int,
    character_folder: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Transfer the character's costume/ reference onto the frame.

    Raises :class:`GarmentError` with the folder in the message when the
    character has no costume screenshot — a silent no-op here would look
    like a working run that changed nothing.
    """
    folder = Path(character_folder)
    reference = costume_reference(folder)
    if reference is None:
        raise GarmentError(
            f"人物文件夹没有服装参考图：{folder / 'costume'}。"
            "先跑「衣服打标签」，或把全身截图放进 costume/。"
        )
    ref = read_image(reference)
    result = transfer_garment(frame_bgr, person_mask, neck_y, ref, **kwargs)
    result["reference"] = str(reference)
    log.info("换装完成：%s（%s 像素）", reference.name, result["region_pixels"])
    return result
