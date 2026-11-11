"""Tight two-person face binding. The old IoU-0.05 rule assigns the wrong face."""

from recast.stages.face_match import iou, match_boxes, pair_score


def _old_score(person: list[float], face: list[float]) -> float:
    """Previous replace/identity rule: IoU, or 0.15 when a face center sits in the body."""
    overlap = iou(person, face)
    cx = (face[0] + face[2]) / 2.0
    cy = (face[1] + face[3]) / 2.0
    inside = person[0] <= cx <= person[2] and person[1] <= cy <= person[3]
    if overlap < 0.05 and inside:
        return max(overlap, 0.15)
    return overlap


def test_close_fighters_are_not_bound_to_a_distant_speck():
    person_a = [0, 0, 200, 480]
    person_b = [190, 0, 400, 480]
    face_a = [45, 25, 145, 155]
    face_b = [230, 30, 340, 160]
    speck = [80, 300, 100, 320]
    faces = [face_a, face_b, speck]

    old_best_a = max(range(len(faces)), key=lambda i: _old_score(person_a, faces[i]))
    assert old_best_a == 2
    assert _old_score(person_a, speck) > _old_score(person_a, face_a)

    pairs = match_boxes([person_a, person_b], faces)
    assert pairs == [(0, 0), (1, 1)]
    assert pair_score(person_a, speck) == 0.0
    assert pair_score(person_a, face_a) >= 0.2
    assert pair_score(person_a, face_b) == 0.0
