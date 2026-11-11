"""Shared face-to-person association for identity sampling and L1 replace.

Identity and replace must use this module so a fight-shot close-up cannot
bind a face in one stage and a different face in the other.
"""

from __future__ import annotations

from typing import Any

# Raw IoU against the head zone. The old 0.05 floor plus a 0.15 center boost
# let a distant speck beat the face that actually sits on the person.
MIN_FACE_IOU = 0.2
# A full-body box is much taller than it is wide; the head is the top portion.
_TALL_ASPECT = 1.4
_HEAD_FRACTION = 0.42
# Face area / zone area below this is a distant speck, not the tracked person.
_MIN_AREA_RATIO = 0.05


def box_area(box: list[float]) -> float:
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return width * height


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def center_inside(outer: list[float], inner: list[float]) -> bool:
    cx = (float(inner[0]) + float(inner[2])) / 2.0
    cy = (float(inner[1]) + float(inner[3])) / 2.0
    return float(outer[0]) <= cx <= float(outer[2]) and float(outer[1]) <= cy <= float(outer[3])


def head_zone(person_box: list[float]) -> list[float]:
    """Region where the face should sit. Tall body boxes use the top; close-ups use the whole box."""
    x1, y1, x2, y2 = (float(v) for v in person_box)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return [x1, y1, x2, y2]
    if height > width * _TALL_ASPECT:
        return [x1, y1, x2, y1 + height * _HEAD_FRACTION]
    return [x1, y1, x2, y2]


def pair_score(person_box: list[float], face_box: list[float], *, min_iou: float = MIN_FACE_IOU) -> float:
    """Score a face against a person. 0 means do not associate them.

    IoU is measured against the head zone, so a face on the shoulders still
    clears the floor while a speck lower in the body does not. A face that is
    tiny relative to that zone is rejected even if the boxes overlap.
    """
    zone = head_zone(person_box)
    overlap = iou(zone, face_box)
    if overlap < min_iou:
        return 0.0
    zone_area = box_area(zone)
    face_area = box_area(face_box)
    if zone_area <= 0 or face_area <= 0:
        return 0.0
    ratio = face_area / zone_area
    if ratio < _MIN_AREA_RATIO:
        return 0.0
    if ratio < 0.2:
        size_prior = ratio / 0.2
    elif ratio > 1.8:
        size_prior = max(0.0, 1.8 / ratio)
    else:
        size_prior = 1.0
    if not center_inside(zone, face_box):
        size_prior *= 0.5
    return overlap * size_prior


def match_boxes(
    person_boxes: list[list[float]],
    face_boxes: list[list[float]],
    *,
    min_iou: float = MIN_FACE_IOU,
) -> list[tuple[int, int]]:
    """Mutual best match. A pair is kept only when each side prefers the other."""
    if not person_boxes or not face_boxes:
        return []
    scores = [
        [pair_score(person, face, min_iou=min_iou) for face in face_boxes]
        for person in person_boxes
    ]
    best_face: list[int] = []
    for row in scores:
        best_j = max(range(len(row)), key=lambda j: row[j])
        best_face.append(best_j if row[best_j] > 0 else -1)
    best_person: list[int] = []
    for j in range(len(face_boxes)):
        best_i = max(range(len(person_boxes)), key=lambda i: scores[i][j])
        best_person.append(best_i if scores[best_i][j] > 0 else -1)
    pairs: list[tuple[int, int]] = []
    for i, j in enumerate(best_face):
        if j >= 0 and best_person[j] == i:
            pairs.append((i, j))
    return pairs


def match_detections(
    faces: list[Any],
    detections: list[dict[str, Any]],
    *,
    min_iou: float = MIN_FACE_IOU,
) -> list[tuple[int, Any]]:
    """Return (track_id, face) pairs using the same rule as replace."""
    face_boxes = [[float(x) for x in face.bbox] for face in faces]
    person_boxes = [list(det["bbox"]) for det in detections]
    pairs = match_boxes(person_boxes, face_boxes, min_iou=min_iou)
    return [(int(detections[pi]["track_id"]), faces[fi]) for pi, fi in pairs]
