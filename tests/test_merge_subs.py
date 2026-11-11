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


def test_shared_characters_are_not_the_same_line() -> None:
    """Bag-of-characters ratio is 1.0 here; the lines are not the same sequence."""
    asr_text = "你好我是温念"
    ocr_text = "念温是我好你"
    shorter, longer = asr_text, ocr_text
    bag = sum(1 for ch in shorter if ch in longer) / len(shorter)
    assert bag >= 0.72
    from recast.extract.merge_subs import text_similarity

    assert text_similarity(asr_text, ocr_text) < 0.72
    merged = merge_subtitles(
        [{"start": 1.0, "end": 3.0, "text": asr_text, "avg_logprob": -0.2}],
        [{"start": 1.0, "end": 3.0, "text": ocr_text, "conf": 0.99}],
    )
    texts = [s["text"] for s in merged]
    assert asr_text in texts
    assert ocr_text in texts


def test_overlapping_line_keeps_the_higher_confidence() -> None:
    asr = [{"start": 1.0, "end": 3.0, "text": "快起来，别装死", "avg_logprob": -1.6}]
    ocr = [{"start": 1.05, "end": 2.9, "text": "快起来别装死", "conf": 0.93}]
    merged = merge_subtitles(asr, ocr)
    assert len(merged) == 1
    assert merged[0]["source"] == "ocr"
    assert merged[0]["text"] == "快起来别装死"

    asr_better = [{"start": 1.0, "end": 3.0, "text": "快起来，别装死", "avg_logprob": -0.05}]
    ocr_weaker = [{"start": 1.05, "end": 2.9, "text": "快起来别装死", "conf": 0.4}]
    merged = merge_subtitles(asr_better, ocr_weaker)
    assert len(merged) == 1
    assert merged[0]["source"] == "asr"
    assert "别装死" in merged[0]["text"]


def test_equal_confidence_keeps_asr() -> None:
    merged = merge_subtitles(
        [{"start": 0.0, "end": 2.0, "text": "谁允许你动手", "conf": 0.8}],
        [{"start": 0.1, "end": 1.9, "text": "谁允许你动手", "conf": 0.8}],
    )
    assert len(merged) == 1
    assert merged[0]["source"] == "asr"
