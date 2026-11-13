"""Timbre normalization and per-engine parameter mapping."""

from __future__ import annotations

from recast.generate.timbre import (
    edge_tts_kwargs,
    is_default,
    normalize_timbre,
    piper_length_scale,
    pyttsx3_wpm,
    timbre_note,
)


def test_normalize_clamps_ranges():
    t = normalize_timbre({"pitch": 99, "rate": 9, "style": "  冷淡  "})
    assert t == {"pitch": 12.0, "rate": 2.0, "style": "冷淡"}
    t = normalize_timbre({"pitch": -99, "rate": 0.01})
    assert t["pitch"] == -12.0
    assert t["rate"] == 0.5
    assert t["style"] == ""


def test_normalize_bad_input_becomes_default():
    assert normalize_timbre(None) == {"pitch": 0.0, "rate": 1.0, "style": ""}
    assert normalize_timbre("高") == {"pitch": 0.0, "rate": 1.0, "style": ""}
    assert normalize_timbre({"pitch": "高", "rate": "快"}) == {
        "pitch": 0.0,
        "rate": 1.0,
        "style": "",
    }
    long_style = "x" * 100
    assert len(normalize_timbre({"style": long_style})["style"]) == 40


def test_is_default():
    assert is_default(None)
    assert is_default({"pitch": 0.0, "rate": 1.0, "style": ""})
    assert not is_default({"pitch": 1.0})
    assert not is_default({"rate": 1.2})
    assert not is_default({"style": "冷淡"})


def test_edge_tts_kwargs():
    assert edge_tts_kwargs(None) == {}
    assert edge_tts_kwargs({"pitch": 0.0, "rate": 1.0}) == {}
    assert edge_tts_kwargs({"rate": 1.2}) == {"rate": "+20%"}
    assert edge_tts_kwargs({"rate": 0.8}) == {"rate": "-20%"}
    assert edge_tts_kwargs({"pitch": 2.5}) == {"pitch": "+2.5st"}
    assert edge_tts_kwargs({"pitch": -3, "rate": 1.5}) == {
        "rate": "+50%",
        "pitch": "-3.0st",
    }


def test_piper_length_scale_is_inverse_of_rate():
    assert piper_length_scale(1.0) == 1.0
    assert piper_length_scale(2.0) == 0.5
    assert piper_length_scale(0.5) == 2.0
    assert piper_length_scale(99) == 0.5
    assert piper_length_scale(0.01) == 2.0


def test_pyttsx3_wpm():
    assert pyttsx3_wpm(1.0) == 200
    assert pyttsx3_wpm(1.5) == 300
    assert pyttsx3_wpm(0.5) == 100
    assert pyttsx3_wpm(99) == 400
    assert pyttsx3_wpm(0.01) == 100


def test_timbre_note_names_unsupported_parts():
    assert timbre_note(None, "edge-tts") == ""
    note = timbre_note({"pitch": 2, "rate": 1.2, "style": "冷淡"}, "edge-tts")
    assert "语速 1.2x" in note
    assert "音调 +2.0 半音" in note
    assert "冷淡" in note and "不支持" in note
    note = timbre_note({"pitch": 2, "rate": 1.0}, "piper")
    assert "音调未应用" in note
    assert "克隆音色不支持调参" in timbre_note({"pitch": 2}, "local-clone")
    assert "音调未应用" in timbre_note({"pitch": 2}, "pyttsx3")
