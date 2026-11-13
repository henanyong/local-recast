"""Diarization: energy segmentation, spectral speaker split, clip writing.

All audio here is generated in the test (sine tones + silence) — no real
video, photo, or speaker model is needed.
"""

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from recast.core.characters import ensure_character_folder
from recast.stages.diarization import (
    DiarizationError,
    diarize,
    read_wav,
    segment_speech,
    split_to_character_folders,
    write_wav,
)

SR = 16000


def _tone(path: Path, parts: list[tuple[float, float]]) -> Path:
    """Write a wav from (frequency, seconds) parts; frequency 0 = silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for freq, secs in parts:
        n = int(secs * SR)
        t = np.arange(n) / SR
        wave_data = np.zeros(n) if freq == 0 else 0.5 * np.sin(2 * math.pi * freq * t)
        frames += (np.clip(wave_data, -1, 1) * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(bytes(frames))
    return path


def test_wav_round_trip(tmp_path: Path):
    src = _tone(tmp_path / "a.wav", [(440.0, 0.5)])
    samples, sr = read_wav(src)
    assert sr == SR
    assert samples.shape[0] == SR // 2
    assert float(np.max(np.abs(samples))) > 0.4
    out = write_wav(tmp_path / "b.wav", samples, sr)
    again, _ = read_wav(out)
    assert np.allclose(samples, again, atol=1e-3)


def test_read_missing_wav(tmp_path: Path):
    with pytest.raises(DiarizationError, match="找不到音频文件"):
        read_wav(tmp_path / "nope.wav")


def test_segment_speech_finds_three_parts(tmp_path: Path):
    src = _tone(tmp_path / "talk.wav", [(220.0, 1.0), (0, 0.6), (440.0, 1.0), (0, 0.6), (220.0, 1.0)])
    samples, sr = read_wav(src)
    segs = segment_speech(samples, sr)
    assert len(segs) == 3, segs
    assert abs(segs[0][0] - 0.0) < 0.1
    assert abs(segs[0][1] - 1.0) < 0.1
    assert abs(segs[1][0] - 1.6) < 0.1
    assert abs(segs[2][0] - 3.2) < 0.1


def test_segment_silence_is_empty(tmp_path: Path):
    src = _tone(tmp_path / "quiet.wav", [(0, 2.0)])
    samples, sr = read_wav(src)
    assert segment_speech(samples, sr) == []


def test_diarize_groups_same_tone_together(tmp_path: Path):
    # 220Hz, 440Hz, 220Hz: the two 220Hz parts must share a speaker.
    src = _tone(tmp_path / "talk.wav", [(220.0, 1.0), (0, 0.6), (440.0, 1.0), (0, 0.6), (220.0, 1.0)])
    found = diarize(src, n_speakers=2)
    assert len(found) == 3, found
    assert found[0]["speaker"] == found[2]["speaker"]
    assert found[0]["speaker"] != found[1]["speaker"]
    assert {f["speaker"] for f in found} == {"SPK_1", "SPK_2"}


def test_split_writes_clips_into_character_folders(tmp_path: Path):
    src = _tone(tmp_path / "talk.wav", [(220.0, 1.0), (0, 0.6), (440.0, 1.0)])
    folder_a, _ = ensure_character_folder(tmp_path / "outputs", "温念")
    folder_b, _ = ensure_character_folder(tmp_path / "outputs", "陈秋微")
    found = diarize(src, n_speakers=2)
    assert len(found) == 2, found
    mapping = {found[0]["speaker"]: folder_a, found[1]["speaker"]: folder_b}
    report = split_to_character_folders(src, mapping, n_speakers=2)
    assert report["segments"] == 2, report
    assert report["skipped"] == []
    clips_a = list((folder_a / "voice").glob("*.wav"))
    clips_b = list((folder_b / "voice").glob("*.wav"))
    assert len(clips_a) == 1 and len(clips_b) == 1
    samples, sr = read_wav(clips_a[0])
    assert abs(samples.shape[0] / sr - 1.0) < 0.1
    # The clip is recorded in the character meta.
    from recast.core.characters import load_meta

    assert len(load_meta(folder_a)["voice"]["samples"]) == 1


def test_split_skips_unmapped_speaker(tmp_path: Path):
    src = _tone(tmp_path / "talk.wav", [(220.0, 1.0), (0, 0.6), (440.0, 1.0)])
    folder_a, _ = ensure_character_folder(tmp_path / "outputs", "温念")
    report = split_to_character_folders(src, {"SPK_1": folder_a}, n_speakers=2)
    assert report["segments"] == 2
    assert len(report["skipped"]) == 1
    assert len(report["clips"].get("SPK_1", [])) == 1
