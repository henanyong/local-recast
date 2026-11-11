"""ASR prompt, no-speech filter, word-edge snap, and local Whisper cache."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from recast.config import PROJECT_ROOT, RecastConfig
from recast.stages.subtitles import (
    build_initial_prompt,
    filter_no_speech,
    snap_segments_to_shots,
)


def test_initial_prompt_lists_roster_names_once():
    prompt = build_initial_prompt(["温念", "温念", " 陈秋微 ", ""])
    assert prompt is not None
    assert "温念" in prompt
    assert "陈秋微" in prompt
    assert prompt.count("温念") == 1
    assert build_initial_prompt([]) is None
    assert build_initial_prompt(None) is None


def test_no_speech_filter_drops_only_above_threshold():
    segments = [
        {"text": "留下", "no_speech_prob": 0.6},
        {"text": "丢掉", "no_speech_prob": 0.61},
        {"text": "安静", "no_speech_prob": 0.9},
    ]
    kept = filter_no_speech(segments, max_no_speech_prob=0.6)
    assert [s["text"] for s in kept] == ["留下"]
    looser = filter_no_speech(segments, max_no_speech_prob=0.8)
    assert [s["text"] for s in looser] == ["留下", "丢掉"]


def test_word_times_snap_toward_shot_cuts_and_tighten_duration():
    segment = {
        "start": 0.0,
        "end": 10.0,
        "text": "快起来",
        "words": [
            {"word": "快", "start": 1.2, "end": 1.6},
            {"word": "起来", "start": 1.6, "end": 3.1},
        ],
    }
    shots = [{"start_time": 1.0, "end_time": 3.2}]
    snapped = snap_segments_to_shots([segment], shots, max_shift=0.35)
    assert snapped[0]["start"] == 1.0
    assert snapped[0]["end"] == 3.2
    assert snapped[0]["duration_note"] == "已按词级时间收紧"

    far = {
        "start": 5.0,
        "end": 6.0,
        "text": "远处",
        "words": [{"word": "远处", "start": 5.0, "end": 6.0}],
    }
    held = snap_segments_to_shots([far], [{"start_time": 0.0, "end_time": 1.0}], max_shift=0.35)
    assert held[0]["start"] == 5.0
    assert held[0]["end"] == 6.0
    assert "duration_note" not in held[0]

    plain = {"start": 2.0, "end": 3.0, "text": "无词", "words": []}
    untouched = snap_segments_to_shots([plain], shots, max_shift=0.35)
    assert untouched[0]["start"] == 2.0
    assert untouched[0]["end"] == 3.0


def test_yaml_beam_sizes_stay_put():
    eight = (PROJECT_ROOT / "configs" / "4060ti-8g.yaml").read_text(encoding="utf-8")
    sixteen = (PROJECT_ROOT / "configs" / "4060ti-16g.yaml").read_text(encoding="utf-8")
    assert "beam_size: 1" in eight
    assert "beam_size: 3" in sixteen
    assert RecastConfig().whisper.beam_size == 1
    assert RecastConfig().whisper.no_speech_prob == 0.6


class _Word:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word
        self.probability = 0.9


class _Seg:
    def __init__(self, text, no_speech):
        self.id = 0
        self.start = 0.0
        self.end = 1.0
        self.text = text
        self.avg_logprob = -0.2
        self.no_speech_prob = no_speech
        self.words = [_Word(0.2, 0.8, text.strip())]


class _Info:
    language = "zh"
    language_probability = 0.99
    duration = 1.0


def test_transcribe_uses_local_cache_then_prompt_and_words(tmp_path: Path, monkeypatch):
    calls: list[tuple] = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append(("init", name, kwargs))

        def transcribe(self, path, **kwargs):
            calls.append(("transcribe", kwargs))
            return iter([_Seg(" 温念 ", 0.1), _Seg(" 噪声 ", 0.95)]), _Info()

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    from recast.stages.subtitles import run

    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    _srt, _js, segments = run(wav, tmp_path, RecastConfig(), names=["温念", "陈秋微"])
    assert calls[0][0] == "init"
    assert calls[0][2]["local_files_only"] is True
    assert len([c for c in calls if c[0] == "init"]) == 1
    kwargs = calls[1][1]
    assert kwargs["word_timestamps"] is True
    assert kwargs["beam_size"] == 1
    assert "温念" in kwargs["initial_prompt"]
    assert "陈秋微" in kwargs["initial_prompt"]
    assert [s["text"] for s in segments] == ["温念"]
    assert segments[0]["words"][0]["word"] == "温念"


def test_cache_miss_falls_back_and_double_failure_is_explicit(tmp_path: Path, monkeypatch):
    calls: list[dict] = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("local_files_only"):
                raise RuntimeError("no cache")

        def transcribe(self, path, **kwargs):
            return iter([]), _Info()

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    from recast.stages.subtitles import run

    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    run(wav, tmp_path, RecastConfig())
    assert calls[0].get("local_files_only") is True
    assert "local_files_only" not in calls[1]

    class AlwaysFail:
        def __init__(self, name, **kwargs):
            raise RuntimeError("offline")

    fake.WhisperModel = AlwaysFail
    try:
        run(wav, tmp_path, RecastConfig())
    except RuntimeError as exc:
        assert "本地缓存" in str(exc)
    else:
        raise AssertionError("expected a cache-and-download failure")
