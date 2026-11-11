from __future__ import annotations

from recast.extract.merge_subs import merge_subtitles


def test_asr_is_base_track() -> None:
    asr = [{"start": 0.0, "end": 1.2, "text": "你好，世界"}]
    merged = merge_subtitles(asr, [])
    assert len(merged) == 1
    assert merged[0]["source"] == "asr"
    assert merged[0]["text"] == "你好，世界"


def test_duplicate_ocr_is_dropped() -> None:
    asr = [{"start": 1.0, "end": 3.0, "text": "快起来别装死"}]
    ocr = [{"start": 1.1, "end": 2.8, "text": "快起来！别装死！"}]
    merged = merge_subtitles(asr, ocr)
    assert [s["source"] for s in merged] == ["asr"]


def test_ocr_gap_or_title_is_kept() -> None:
    asr = [{"start": 5.0, "end": 7.0, "text": "谁允许你动手"}]
    ocr = [
        {"start": 0.2, "end": 1.5, "text": "第一集"},
        {"start": 5.1, "end": 6.8, "text": "谁允许你动手？"},
    ]
    merged = merge_subtitles(asr, ocr)
    texts = [s["text"] for s in merged]
    assert "第一集" in texts
    assert any("谁允许你动手" in t for t in texts)
    assert len(merged) == 2
    assert merged[0]["start"] <= merged[1]["start"]


def test_empty_inputs() -> None:
    assert merge_subtitles([], []) == []
    assert merge_subtitles(None, None) == []
