"""Speaker diarization: split an audio track into per-speaker voice samples.

Lightweight and offline on purpose. pyannote-audio needs a Hugging Face
token and a gated licence, which does not fit the local-first default, so
this module is built from two replaceable pieces instead:

1. Segmentation by frame energy (silence / volume). Implemented here with
   the stdlib ``wave`` module and numpy — no model download.
2. Speaker assignment by a pluggable voiceprint embedder. The built-in
   :class:`SpectralEmbedder` is a heuristic (band energies + pitch +
   centroid): it separates clearly different voices, but it is NOT a real
   speaker-embedding model, and overlapping speech will be imprecise.
   Swap in a real embedder (e.g. a local ECAPA model) through the
   ``VoiceEmbedder`` protocol when accuracy matters.

Outputs are plain wav clips written into each character folder's ``voice/``.
"""

from __future__ import annotations

import logging
import math
import wave
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger("recast")

DEFAULT_SAMPLE_RATE = 16000


class DiarizationError(RuntimeError):
    """Audio cannot be read, split, or attributed to speakers."""


# ---------------------------------------------------------------------------
# WAV I/O (stdlib only)
# ---------------------------------------------------------------------------

def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM wav file into float32 mono samples in [-1, 1]."""
    path = Path(path)
    if not path.is_file():
        raise DiarizationError(f"找不到音频文件：{path}")
    try:
        with wave.open(str(path), "rb") as fh:
            n_channels = fh.getnchannels()
            width = fh.getsampwidth()
            sr = fh.getframerate()
            frames = fh.readframes(fh.getnframes())
    except (wave.Error, OSError) as exc:
        raise DiarizationError(f"读不懂音频文件 {path}（只支持 PCM wav）：{exc}") from exc
    if width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        data = (raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)).astype(np.float32)
        data = np.where(data >= 1 << 23, data - (1 << 24), data) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise DiarizationError(f"不支持的采样位宽：{width} 字节")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    """Write float32 mono samples as 16-bit PCM wav."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sr))
        fh.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------------------
# Segmentation by frame energy
# ---------------------------------------------------------------------------

def segment_speech(
    samples: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 20.0,
    min_speech_sec: float = 0.3,
    min_silence_sec: float = 0.3,
    merge_gap_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """Split samples into (start, end) speech segments in seconds.

    Two-threshold energy VAD: frames above the high threshold are speech,
    neighbours above the low threshold are kept, then close segments merge
    and short ones drop. Returns [] for silence.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return []
    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = int(math.ceil(samples.size / frame))
    padded = np.pad(samples, (0, n_frames * frame - samples.size))
    frames = padded.reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    # Noise floor from a low percentile, NOT the median: when speech is most
    # of the track, the median is a speech level and nothing would trigger.
    peak = float(np.max(rms)) if rms.size else 0.0
    floor = float(np.percentile(rms, 10)) if rms.size else 0.0
    high = max(floor * 4.0, 0.01)
    if peak > 0:
        high = min(high, peak * 0.5)
    low = high * 0.5
    is_speech = rms >= high
    # hysteresis: keep low-threshold frames next to speech
    active = is_speech.copy()
    for i in range(n_frames):
        if rms[i] >= low and (
            (i > 0 and active[i - 1]) or (i + 1 < n_frames and is_speech[i + 1])
        ):
            active[i] = True
    segments: list[tuple[float, float]] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segments.append((start * frame_ms / 1000.0, i * frame_ms / 1000.0))
            start = None
    if start is not None:
        segments.append((start * frame_ms / 1000.0, n_frames * frame_ms / 1000.0))
    # merge segments separated by a short silence
    merged: list[tuple[float, float]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap_sec:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    # drop segments shorter than the minimum, and clip to the audio length
    total = samples.size / sr
    kept = [
        (max(0.0, s), min(total, e))
        for s, e in merged
        if e - s >= min_speech_sec
    ]
    return kept


# ---------------------------------------------------------------------------
# Voiceprint embedding (pluggable)
# ---------------------------------------------------------------------------

class VoiceEmbedder(Protocol):
    """Anything that turns a speech segment into a speaker vector."""

    @property
    def name(self) -> str: ...

    def embed(self, samples: np.ndarray, sr: int, start: float, end: float) -> np.ndarray | None:
        ...


class SpectralEmbedder:
    """Heuristic embedder: band energies + pitch + centroid. No model.

    Separates clearly different voices (e.g. different pitch ranges). It is
    not a speaker-embedding model: same-pitch speakers and overlapping
    speech will be confused. Use it for a first usable version; replace it
    through the VoiceEmbedder protocol for production accuracy.
    """

    name = "spectral-heuristic"

    def embed(self, samples: np.ndarray, sr: int, start: float, end: float) -> np.ndarray | None:
        seg = np.asarray(samples, dtype=np.float32)[max(0, int(start * sr)): int(end * sr)]
        if seg.size < sr // 10:
            return None
        seg = seg - seg.mean()
        if float(np.max(np.abs(seg))) < 1e-4:
            return None
        # 8 log band energies over 0..(sr/2), averaged over full windows.
        win = min(seg.size, 2048)
        step = max(1, win // 2)
        spectra = []
        if seg.size >= win:
            for off in range(0, seg.size - win + 1, step):
                chunk = seg[off:off + win] * np.hanning(win)
                spectra.append(np.abs(np.fft.rfft(chunk)))
        else:
            spectra.append(np.abs(np.fft.rfft(seg * np.hanning(seg.size))))
        if not spectra:
            return None
        spec = np.mean(spectra, axis=0)
        edges = np.linspace(0, len(spec), 9).astype(int)
        bands = np.array(
            [np.log1p(float(spec[edges[i]:edges[i + 1]].mean()) + 1e-6) for i in range(8)],
            dtype=np.float64,
        )
        # pitch estimate by autocorrelation on the loudest window
        loudest = int(np.argmax([float(np.mean(c ** 2)) for c in spectra])) if spectra else 0
        chunk = seg[loudest * step:loudest * step + win] if seg.size >= win else seg
        pitch = _estimate_pitch(chunk, sr)
        # spectral centroid + zero-crossing rate
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        centroid = float((freqs * spec).sum() / (spec.sum() + 1e-9)) / (sr / 2.0)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(seg)))) if seg.size > 1 else 0.0)
        vec = np.concatenate([bands, [math.log1p(pitch) / math.log1p(500.0), centroid, zcr]])
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return (vec / norm).astype(np.float32)


def _estimate_pitch(chunk: np.ndarray, sr: int) -> float:
    """Median pitch (Hz) by autocorrelation, 0 when no clear periodicity."""
    if chunk.size < sr // 20:
        return 0.0
    x = chunk - chunk.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    corr[: max(1, sr // 500)] = 0  # ignore lags above 500 Hz
    limit = min(len(corr), sr // 50)  # ignore lags below 50 Hz
    if limit <= 0:
        return 0.0
    peak = int(np.argmax(corr[:limit]))
    if corr[peak] <= 0:
        return 0.0
    return float(sr / max(peak, 1))


def cluster_speakers(
    embeddings: list[np.ndarray | None],
    *,
    n_speakers: int | None = None,
) -> list[int]:
    """Assign speaker labels (0-based) to segments by embedding distance.

    Returns -1 for segments without an embedding. With n_speakers given,
    uses k-means; otherwise average-linkage merging with a distance cut.
    Pure numpy on purpose: the repo's sklearn import pulls pandas, which is
    binary-incompatible with pip numpy on some Linux installs, and this
    clustering is small enough not to need it.
    """
    labels = [-1] * len(embeddings)
    idx = [i for i, e in enumerate(embeddings) if e is not None]
    if not idx:
        return labels
    if len(idx) == 1:
        labels[idx[0]] = 0
        return labels
    mat = np.stack([np.asarray(embeddings[i], dtype=np.float64) for i in idx])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-9)
    if n_speakers is not None and 1 < n_speakers <= len(idx):
        found = _kmeans(mat, n_speakers)
    else:
        found = _agglomerative(mat, threshold=0.6)
    for pos, i in enumerate(idx):
        labels[i] = int(found[pos])
    return labels


def _kmeans(mat: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> np.ndarray:
    """Deterministic k-means on L2-normalized rows (cosine distance)."""
    rng = np.random.default_rng(seed)
    n = len(mat)
    centers = [mat[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min(
            np.stack([((mat - c) ** 2).sum(axis=1) for c in centers]), axis=0
        )
        total = float(d2.sum())
        centers.append(mat[int(rng.integers(n))] if total <= 0 else mat[int(rng.choice(n, p=d2 / total))])
    centers = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((mat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = dists.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                c = mat[labels == j].mean(axis=0)
                centers[j] = c / max(float(np.linalg.norm(c)), 1e-9)
    return labels


def _agglomerative(mat: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Average-linkage merging while the closest pair is within threshold."""
    sims = mat @ mat.T
    dists = 1.0 - sims
    np.fill_diagonal(dists, 0.0)
    clusters: list[list[int]] = [[i] for i in range(len(mat))]

    def cluster_dist(a: list[int], b: list[int]) -> float:
        return float(dists[np.ix_(a, b)].mean())

    while True:
        best: tuple[int, int] | None = None
        best_d = threshold
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_dist(clusters[i], clusters[j])
                if d <= best_d:
                    best_d = d
                    best = (i, j)
        if best is None:
            break
        i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    labels = np.zeros(len(mat), dtype=int)
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = ci
    return labels


# ---------------------------------------------------------------------------
# End-to-end: diarize + write clips into character folders
# ---------------------------------------------------------------------------

def diarize(
    wav_path: str | Path,
    *,
    n_speakers: int | None = None,
    embedder: VoiceEmbedder | None = None,
    min_speech_sec: float = 0.3,
    min_silence_sec: float = 0.3,
    merge_gap_sec: float = 0.5,
) -> list[dict[str, Any]]:
    """Split a wav into speaker-labelled segments.

    Returns [{"start", "end", "speaker", "duration"}] with speaker ids
    SPK_1.. in order of first appearance. Empty list means no speech found.
    """
    samples, sr = read_wav(wav_path)
    segments = segment_speech(
        samples,
        sr,
        min_speech_sec=min_speech_sec,
        min_silence_sec=min_silence_sec,
        merge_gap_sec=merge_gap_sec,
    )
    if not segments:
        log.info("说话人分离：%s 里没有检测到语音。", wav_path)
        return []
    emb = embedder or SpectralEmbedder()
    vectors = [emb.embed(samples, sr, s, e) for s, e in segments]
    labels = cluster_speakers(vectors, n_speakers=n_speakers)
    order: dict[int, str] = {}
    out: list[dict[str, Any]] = []
    for (start, end), label in zip(segments, labels):
        if label < 0:
            continue
        if label not in order:
            order[label] = f"SPK_{len(order) + 1}"
        out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": order[label],
                "duration": round(end - start, 3),
            }
        )
    log.info(
        "说话人分离：%s 切出 %s 段，分成 %s 个说话人（%s）。",
        wav_path,
        len(out),
        len(order),
        emb.name,
    )
    return out


def split_to_character_folders(
    wav_path: str | Path,
    speaker_to_folder: dict[str, str | Path],
    *,
    n_speakers: int | None = None,
    embedder: VoiceEmbedder | None = None,
    min_speech_sec: float = 0.3,
    min_silence_sec: float = 0.3,
    merge_gap_sec: float = 0.5,
) -> dict[str, Any]:
    """Write one wav clip per speech segment into the mapped character folder.

    ``speaker_to_folder`` maps speaker ids (SPK_1..) to character folder
    paths; clips land in each folder's ``voice/`` as
    ``<speaker>_<start>_<end>.wav``. Speakers with no mapping are skipped
    and reported. Returns a report with clips per character and skips.
    """
    from recast.core.characters import load_meta, save_meta

    samples, sr = read_wav(wav_path)
    found = diarize(
        wav_path,
        n_speakers=n_speakers,
        embedder=embedder,
        min_speech_sec=min_speech_sec,
        min_silence_sec=min_silence_sec,
        merge_gap_sec=merge_gap_sec,
    )
    report: dict[str, Any] = {"segments": len(found), "clips": {}, "skipped": []}
    for seg in found:
        folder = speaker_to_folder.get(seg["speaker"])
        if folder is None:
            report["skipped"].append(seg)
            continue
        folder = Path(folder)
        voice_dir = folder / "voice"
        voice_dir.mkdir(parents=True, exist_ok=True)
        clip = voice_dir / f"{seg['speaker']}_{seg['start']:.2f}_{seg['end']:.2f}.wav"
        s0, s1 = int(seg["start"] * sr), int(seg["end"] * sr)
        write_wav(clip, samples[s0:s1], sr)
        report["clips"].setdefault(seg["speaker"], []).append(str(clip))
        meta = load_meta(folder) or {}
        voice_meta = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
        known = list(voice_meta.get("samples", []) or [])
        if str(clip) not in known:
            known.append(str(clip))
        meta["voice"] = {**voice_meta, "samples": known}
        save_meta(folder, meta)
    return report
