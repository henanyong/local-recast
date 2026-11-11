"""Dialogue mix must not attenuate by the clip count, and must not speed audio."""

from pathlib import Path

from recast.generate.stitch import _mix_audio, mix_filter_graph


def test_mix_graph_disables_normalize_and_loudnorm():
    graph = mix_filter_graph([0, 1500, 3000, 4200], 12.0)
    assert "amix=inputs=4:" in graph
    assert "normalize=0" in graph
    assert "loudnorm=" in graph
    assert "atempo" not in graph


def test_mix_audio_passes_that_graph_to_ffmpeg(tmp_path: Path, monkeypatch):
    wavs = []
    for i in range(3):
        path = tmp_path / f"{i}.wav"
        path.write_bytes(b"RIFF")
        wavs.append(path)
    shots = [
        {"audio_path": str(wavs[0]), "start_time": 0.0},
        {"audio_path": str(wavs[1]), "start_time": 1.5},
        {"audio_path": str(wavs[2]), "narration_start": 3.0},
    ]
    captured: dict[str, list[str]] = {}

    def fake_ffmpeg(args):
        captured["args"] = list(args)
        Path(args[-1]).write_bytes(b"wav")

    monkeypatch.setattr("recast.generate.stitch.run_ffmpeg", fake_ffmpeg)
    out = _mix_audio(shots, tmp_path / "mix.wav", 8.0)
    assert out is not None
    args = captured["args"]
    graph = args[args.index("-filter_complex") + 1]
    assert "normalize=0" in graph
    assert "loudnorm=" in graph
    assert "amix=inputs=3:" in graph
    assert "atempo" not in graph
    # Picture encode flags are not part of the audio mix command.
    assert "libx264" not in args
    assert "-c:v" not in args
