"""Tests for name auto-suggestion (recast/stages/name_suggest.py).

Synthetic clusters, tracks and dialogue shots only. The rule under test:
a name is suggested only when the evidence points at exactly one cluster
and exactly one name (mutual-best); ambiguous evidence leaves it blank.
"""

from pathlib import Path

from recast.core.characters import ensure_character_folder, load_meta
from recast.stages.name_suggest import (
    extract_names_from_text,
    suggest_names,
    write_name_suggestions,
)


def _persons():
    return [
        {"cluster_id": 0, "auto_name": "Person_1", "name": "Person_1", "track_ids": [1], "first_frame": 0},
        {"cluster_id": 1, "auto_name": "Person_2", "name": "Person_2", "track_ids": [2], "first_frame": 250},
    ]


def _tracks():
    return [
        {"track_id": 1, "first_frame": 0, "last_frame": 250},
        {"track_id": 2, "first_frame": 250, "last_frame": 500},
    ]


def test_extract_names_from_text():
    assert extract_names_from_text("温念：你干什么") == ["温念"]
    assert extract_names_from_text("温念说走吧") == ["温念"]
    assert extract_names_from_text("旁白：1980年的春天") == []
    assert extract_names_from_text("无对白") == []
    assert extract_names_from_text("念：一个字不算名") == []
    assert extract_names_from_text("温念：你好\n陆寒铮：你好") == ["温念", "陆寒铮"]


def test_dialogue_mutual_best_suggests_both():
    shots = [
        {"shot_code": "F1-L1", "speaker": "温念", "start_time": 1.0, "end_time": 3.0},
        {"shot_code": "F1-L2", "speaker": "温念", "start_time": 4.0, "end_time": 6.0},
        {"shot_code": "F2-L1", "speaker": "陆寒铮", "start_time": 12.0, "end_time": 14.0},
    ]
    out = suggest_names(_persons(), _tracks(), dialogue_shots=shots, fps=25.0)
    by_auto = {item["auto_name"]: item for item in out}
    assert by_auto["Person_1"]["suggested_name"] == "温念"
    assert by_auto["Person_1"]["source"] == "对白表"
    assert by_auto["Person_1"]["confidence"] == "high"  # two shots
    assert by_auto["Person_2"]["suggested_name"] == "陆寒铮"
    assert by_auto["Person_2"]["confidence"] == "medium"  # one shot


def test_ambiguous_speaker_gets_no_suggestion():
    # 温念 speaks in shots overlapping BOTH clusters: no clean match.
    shots = [
        {"shot_code": "F1-L1", "speaker": "温念", "start_time": 1.0, "end_time": 3.0},
        {"shot_code": "F2-L1", "speaker": "温念", "start_time": 12.0, "end_time": 14.0},
    ]
    out = suggest_names(_persons(), _tracks(), dialogue_shots=shots, fps=25.0)
    assert out == []


def test_ocr_suggests_with_low_confidence_on_single_hit():
    hits = [{"text": "温念：你干什么", "time": 2.0}]
    out = suggest_names(_persons(), _tracks(), ocr_hits=hits, fps=25.0)
    assert len(out) == 1
    assert out[0]["suggested_name"] == "温念"
    assert out[0]["source"] == "字幕OCR"
    assert out[0]["confidence"] == "low"


def test_dialogue_and_ocr_agreeing_bumps_confidence():
    shots = [{"shot_code": "F1-L1", "speaker": "温念", "start_time": 1.0, "end_time": 3.0}]
    hits = [{"text": "温念：你干什么", "time": 2.0}]
    out = suggest_names(_persons(), _tracks(), dialogue_shots=shots, ocr_hits=hits, fps=25.0)
    assert len(out) == 1
    assert out[0]["suggested_name"] == "温念"
    assert out[0]["confidence"] == "high"  # medium (dialogue) + OCR agreement


def test_single_candidate_fallback():
    # One unnamed cluster, one candidate name, but the shot does not overlap
    # the cluster's time span: no mutual-best evidence, so the weak fallback
    # (single candidate) is what suggests it.
    persons = [_persons()[0]]
    shots = [{"shot_code": "F1-L1", "speaker": "温念", "start_time": 30.0, "end_time": 32.0}]
    out = suggest_names(persons, _tracks()[:1], dialogue_shots=shots, fps=25.0)
    assert len(out) == 1
    assert out[0]["confidence"] == "low"
    assert "唯一候选" in out[0]["source"]


def test_named_persons_are_skipped():
    persons = [
        {"cluster_id": 0, "auto_name": "Person_1", "name": "温念", "track_ids": [1], "first_frame": 0},
    ]
    shots = [{"shot_code": "F1-L1", "speaker": "温念", "start_time": 1.0, "end_time": 3.0}]
    assert suggest_names(persons, _tracks()[:1], dialogue_shots=shots, fps=25.0) == []


def test_write_name_suggestions_only_touches_matching_folder(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "Person_1")
    suggestions = [
        {
            "cluster_id": 0,
            "auto_name": "Person_1",
            "suggested_name": "温念",
            "source": "对白表",
            "confidence": "high",
            "evidence": {},
        },
        {
            "cluster_id": 1,
            "auto_name": "Person_2",
            "suggested_name": "陆寒铮",
            "source": "对白表",
            "confidence": "medium",
            "evidence": {},
        },
    ]
    written = write_name_suggestions(tmp_path, suggestions)
    assert written == [str(folder)]
    meta = load_meta(folder)
    assert meta is not None
    assert meta["name_suggestion"]["name"] == "温念"
    assert meta["name"] == "Person_1", "suggestion must not rename the folder"
    assert not (tmp_path / "characters" / "Person_2").exists()
