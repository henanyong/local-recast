"""Temporal masks: keep the silhouette steady across a clip.

Per-frame segmentation flickers. Even a good model moves the boundary by a
pixel or two between frames, and after a recolour those pixels become a
visible shimmer along the collar and the sleeves. GrabCut is far worse: it
re-seeds from a fresh box every frame and the result can jump by tens of
pixels when the box moves slightly.

So the mask is treated as a track, not a per-frame answer:

- **Detect on a schedule.** Every ``redetect_every`` frames the backend runs
  and re-seeds the track; in between, the previous mask is reused.
- **Compensate global motion.** Between detections the frame-to-frame shift
  is estimated with phase correlation (cheap, no model) and the mask is
  translated by it, so a camera pan or a walking actor stays covered.
- **Smooth with an exponential moving average.** Boundary noise averages out
  while a real movement of a few pixels still comes through within a couple
  of frames.

The trade-off is stated up front: a fast limb movement between two
detections will lag by up to ``redetect_every`` frames. That is why the
schedule is configurable and a real segmentation model is preferred — with
one, ``redetect_every=1`` is affordable and the track only does smoothing.

``allow_noncommercial=False`` is the delivery switch: RMBG-1.4 is the only
strong mask on a typical install and it is non-commercial, so a run meant
for delivery must be able to refuse it rather than discover the problem at
delivery time.
"""

from __future__ import annotations

import logging
from typing import Any

from recast.stages.person_mask import (
    DEFAULT_WEIGHTS,
    PersonMaskError,
    segment_person_auto,
)

log = logging.getLogger("recast")

MOTION_WORK_SIZE = 240


class TemporalMasker:
    """A callable ``(frame, index) -> mask`` that holds a track across frames.

    Drop-in for :func:`recast.core.person_replace.replace_clip`'s ``mask_fn``.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        index: int = 0,
        redetect_every: int = 5,
        smooth: float = 0.6,
        conf: float = 0.25,
        weights: str | Path = DEFAULT_WEIGHTS,  # type: ignore[name-defined]
        device: str | None = None,
        allow_noncommercial: bool = True,
        adaptive: bool = True,
        max_drift_ratio: float = 0.10,
        fast_motion_ratio: float = 0.03,
    ) -> None:
        if redetect_every < 1:
            raise ValueError("redetect_every 至少为 1")
        if not (0.0 <= smooth < 1.0):
            raise ValueError("smooth 必须在 [0, 1) 内；1.0 会把掩码冻死")
        self.backend = backend
        self.index = index
        self.redetect_every = int(redetect_every)
        self.smooth = float(smooth)
        self.conf = conf
        self.weights = weights
        self.device = device
        self.allow_noncommercial = allow_noncommercial
        # Adaptive: a fixed schedule is right for a talking head and wrong for
        # a fight scene. Drift is measured in fractions of the working image
        # width, so the same settings behave the same at 720p and 1080p.
        self.adaptive = adaptive
        self.max_drift_ratio = float(max_drift_ratio)
        self.fast_motion_ratio = float(fast_motion_ratio)

        self._mask: Any = None
        self._prev_small: Any = None
        self._last_detect = -10**9
        self._drift = 0.0
        self.method = ""
        self.warnings: list[str] = []
        self.detections = 0
        self.reuses = 0
        self.early_detections = 0

    # -- helpers ---------------------------------------------------------

    def _small_gray(self, frame: Any) -> Any:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        longest = max(h, w)
        scale = MOTION_WORK_SIZE / float(longest) if longest > MOTION_WORK_SIZE else 1.0
        if scale < 1.0:
            gray = cv2.resize(
                gray, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        return gray.astype("float32")

    @staticmethod
    def _estimate_shift(prev: Any, cur: Any) -> tuple[float, float]:
        """(dx, dy) that moves ``prev`` onto ``cur``, in working pixels."""
        import cv2

        if prev is None or cur is None or prev.shape != cur.shape:
            return 0.0, 0.0
        try:
            (dx, dy), response = cv2.phaseCorrelate(prev, cur)
        except cv2.error:
            return 0.0, 0.0
        if response < 0.05:  # no trustworthy peak (cut, flash, black frame)
            return 0.0, 0.0
        return float(dx), float(dy)

    @staticmethod
    def smooth_mask(prev: Any, new: Any, alpha: float) -> Any:
        """EMA of two boolean masks. ``alpha`` is the weight of ``prev``."""
        import numpy as np

        if prev is None:
            return new.astype(bool)
        blended = alpha * prev.astype(np.float32) + (1.0 - alpha) * new.astype(np.float32)
        return blended >= 0.5

    def _shift_mask(self, mask: Any, dx: float, dy: float) -> Any:
        import cv2
        import numpy as np

        if dx == 0.0 and dy == 0.0:
            return mask
        h, w = mask.shape[:2]
        # The shift was measured on the downscaled working image.
        small_longest = max(self._prev_small.shape[:2]) if self._prev_small is not None else max(h, w)
        scale = float(max(h, w)) / float(small_longest) if small_longest else 1.0
        m = np.array([[1.0, 0.0, dx * scale], [0.0, 1.0, dy * scale]], dtype=np.float64)
        shifted = cv2.warpAffine(
            mask.astype(np.uint8) * 255,
            m,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return shifted > 127

    # -- main ------------------------------------------------------------

    def __call__(self, frame: Any, frame_index: int) -> Any:
        import numpy as np

        small = self._small_gray(frame)
        dx, dy = self._estimate_shift(self._prev_small, small)
        shift_mag = abs(dx) + abs(dy)
        working_w = float(small.shape[1]) if small.ndim >= 2 else 1.0

        scheduled = (frame_index - self._last_detect) >= self.redetect_every
        early = False
        if self.adaptive and self._mask is not None and not scheduled:
            drifted = self._drift + shift_mag >= self.max_drift_ratio * working_w
            jolted = shift_mag >= self.fast_motion_ratio * working_w
            early = drifted or jolted
        due = scheduled or early

        candidate = None
        if self._mask is None or due:
            try:
                result = segment_person_auto(
                    frame,
                    backend=self.backend,
                    index=self.index,
                    conf=self.conf,
                    weights=self.weights,
                    device=self.device,
                    allow_noncommercial=self.allow_noncommercial,
                )
            except PersonMaskError:
                if self._mask is None:
                    raise
                # Detection lost the person for one pass: keep the track
                # rather than punching a hole through the costume.
                self.reuses += 1
                self._mask = self._shift_mask(self._mask, dx, dy)
                self._drift += shift_mag
                self._prev_small = small
                return self._to_uint8(self._mask)
            self.method = result["method"]
            if result.get("warning") and result["warning"] not in self.warnings:
                self.warnings.append(result["warning"])
            candidate = result["mask"] > 0
            self.detections += 1
            if early:
                self.early_detections += 1
            self._last_detect = frame_index
            self._drift = 0.0
        else:
            self.reuses += 1
            candidate = self._shift_mask(self._mask, dx, dy)
            self._drift += shift_mag

        self._mask = self.smooth_mask(self._mask, candidate, self.smooth)
        self._prev_small = small
        return self._to_uint8(self._mask)

    @staticmethod
    def _to_uint8(mask: Any) -> Any:
        import numpy as np

        arr = np.asarray(mask)
        return arr.astype(np.uint8) * 255

    def report(self) -> dict[str, Any]:
        """What the track did, for the clip summary."""
        total = self.detections + self.reuses
        return {
            "backend": self.method or self.backend,
            "frames": total,
            "detections": self.detections,
            "reuses": self.reuses,
            "redetect_every": self.redetect_every,
            "smooth": self.smooth,
            "adaptive": self.adaptive,
            "early_detections": self.early_detections,
            "warnings": list(self.warnings),
        }


def build_mask_fn(**kwargs: Any) -> TemporalMasker:
    """A ``mask_fn`` for :func:`replace_clip` that also keeps its own report.

    The returned object is callable as ``mask_fn(frame, index)``; read
    ``.report()`` afterwards to see how many detections it actually ran.
    """
    return TemporalMasker(**kwargs)
