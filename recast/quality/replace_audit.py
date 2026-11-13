"""Show-through audit for a composited replacement frame.

A replacement betrays itself in exactly three ways, and each one is
measurable without a human looking at the frame:

- **Background damage.** Erasing the actor and inpainting the hole smears
  whatever stood behind them. This counts how much of the untouched
  background actually changed.
- **Seam.** A pasted body leaves edge energy along the neck cut. Real
  bodies have texture everywhere, so a band of edge energy at a fixed
  height that matches no anatomy is the machine signature.
- **Coverage.** If the replacement covers less than the actor's
  silhouette, the leftover original pixels next to a CG body read as a
  double outline.

Pure numpy/cv2, no models, no GPU: this runs per frame and can gate a
batch, which is what "日产多集" needs — a human cannot review 8 minutes of
video per episode to find the one bad shot.

Numbers are for *ranking and regression*, not for certification. A frame
that scores clean can still look wrong for reasons no gradient can see
(wrong identity, wrong costume), so ``verdict`` is "needs eyes" at best.
"""

from __future__ import annotations

from typing import Any

# Thresholds are deliberately loose. Tightening them turns the audit into a
# taste panel; the point is to catch the obviously-broken frames.
BG_DIFF_LEVEL = 6
BG_DAMAGE_LIMIT = 0.02
# Judged on the *increase* over the source frame, not on the absolute value.
# Measured on a real drama frame: the untouched original already scores 3.2,
# because a neck with a collar, a necklace and a jaw edge is genuinely edgy.
# An absolute limit would fail every real shot; a limit on the increase asks
# the only question that matters — did the composite add a seam?
SEAM_INCREASE_LIMIT = 1.15
COVERAGE_FLOOR = 0.85


def _as_mask(mask: Any) -> Any:
    import numpy as np

    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    return arr > 0


def _sobel_magnitude(bgr: Any) -> Any:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def background_damage(orig_bgr: Any, out_bgr: Any, person_mask: Any) -> dict[str, float]:
    """How much of the non-person area changed. Should be ~0.

    Returns ``mean`` (average absolute change, 0-255) and ``frac`` (share of
    background pixels that moved more than ``BG_DIFF_LEVEL``).
    """
    import numpy as np

    person = _as_mask(person_mask)
    bg = ~person
    if not bg.any():
        return {"mean": 0.0, "frac": 0.0}
    diff = np.abs(
        out_bgr.astype(np.int16) - orig_bgr.astype(np.int16)
    ).max(axis=2)
    return {
        "mean": float(diff[bg].mean()),
        "frac": float((diff[bg] > BG_DIFF_LEVEL).mean()),
    }


def seam_ratio(out_bgr: Any, person_mask: Any, neck_y: int, *, band_px: int = 10) -> float:
    """Edge energy in the neck band, relative to the body's own texture.

    Above 1 means the cut is edgier than the body it joins. This number is
    NOT comparable across footage: a real neck with a collar and a necklace
    measures ~3 on its own, before any compositing. Use
    :func:`audit_composite`, which judges the increase over the source
    frame instead of this absolute value.

    Returns 0.0 when there is nothing to compare against.
    """
    import numpy as np

    person = _as_mask(person_mask)
    h = person.shape[0]
    mag = _sobel_magnitude(out_bgr)
    lo = max(0, neck_y - band_px)
    hi = min(h, neck_y + band_px + 1)
    if hi <= lo or not person[lo:hi].any():
        return 0.0
    seam = float(mag[lo:hi][person[lo:hi]].mean())
    ref_lo = min(h, neck_y + 3 * band_px)
    ref_px = mag[ref_lo:][person[ref_lo:]]
    if ref_px.size < 64:
        return 0.0
    ref = float(np.median(ref_px))
    if ref <= 1e-6:
        return 0.0 if seam < 1e-6 else 99.0
    return float(min(seam / ref, 99.0))


def render_coverage(render_alpha: Any, person_mask: Any, neck_y: int, *, thr: int = 8) -> float:
    """Share of the actor's silhouette (below the neck) the render covers."""
    import numpy as np

    person = _as_mask(person_mask)
    body = person.copy()
    body[:neck_y, :] = False
    if not body.any():
        return 0.0
    alpha = np.asarray(render_alpha)
    if alpha.ndim == 3:
        alpha = alpha[:, :, -1]
    return float((alpha[body] > thr).mean())


def audit_composite(
    orig_bgr: Any,
    out_bgr: Any,
    person_mask: Any,
    neck_y: int,
    *,
    render_alpha: Any | None = None,
    band_px: int = 10,
) -> dict[str, Any]:
    """All three numbers plus a verdict for one composited frame.

    ``render_alpha`` is optional; without it coverage is reported as None
    and the verdict only weighs damage and the added seam. The source frame
    is required because the seam is judged against its own neck band — see
    :data:`SEAM_INCREASE_LIMIT`.
    """
    damage = background_damage(orig_bgr, out_bgr, person_mask)
    seam_before = seam_ratio(orig_bgr, person_mask, neck_y, band_px=band_px)
    seam_after = seam_ratio(out_bgr, person_mask, neck_y, band_px=band_px)
    if seam_before > 1e-6:
        increase = seam_after / seam_before
    else:
        increase = 1.0 if seam_after < 1e-6 else 99.0
    coverage = (
        render_coverage(render_alpha, person_mask, neck_y)
        if render_alpha is not None
        else None
    )

    problems: list[str] = []
    if damage["frac"] > BG_DAMAGE_LIMIT:
        problems.append(f"背景被改坏 {damage['frac']:.1%}（上限 {BG_DAMAGE_LIMIT:.0%}）")
    if increase > SEAM_INCREASE_LIMIT:
        problems.append(
            f"新增接缝，颈部梯度比原帧高 {increase:.2f} 倍（上限 {SEAM_INCREASE_LIMIT}）"
        )
    if coverage is not None and coverage < COVERAGE_FLOOR:
        problems.append(f"覆盖不足 {coverage:.0%}（下限 {COVERAGE_FLOOR:.0%}），会露出原人物")

    return {
        "background_damage_mean": round(damage["mean"], 3),
        "background_damage_frac": round(damage["frac"], 5),
        "seam_ratio": round(seam_after, 3),
        "seam_ratio_before": round(seam_before, 3),
        "seam_increase": round(increase, 3),
        "render_coverage": None if coverage is None else round(coverage, 4),
        "verdict": "ok" if not problems else "review",
        "problems": problems,
    }
