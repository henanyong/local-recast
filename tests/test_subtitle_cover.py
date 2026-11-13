"""Cover burned-in subtitles: blur / black bar (inpaint is a stub)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from recast.ffmpeg import (
    FFmpegError,
    cover_subtitle_band,
    inpaint_subtitle_band,
    probe_video,
)


def _make_video(path: Path, *, seconds: float = 3.0) -> Path:
    """testsrc + a bright white box in the bottom 15% (fake burned-in subs)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=10:duration={seconds}",
            "-vf", "drawbox=x=30:y=ih*0.86:w=260:h=ih*0.12:color=white:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


def _frame_at(path: Path, t: float):
    import cv2

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, f"读不到 {path} 在 {t}s 的帧"
    return frame


def _lap_var(gray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def test_blur_covers_band_and_keeps_rest(tmp_path):
    import cv2
    import numpy as np

    src = _make_video(tmp_path / "src.mp4")
    out = cover_subtitle_band(src, tmp_path / "blur.mp4", mode="blur", use_nvenc=False)
    assert out.is_file()
    assert probe_video(out).duration == pytest.approx(probe_video(src).duration, abs=0.2)
    assert probe_video(out).has_audio is False

    orig = _frame_at(src, 1.5)
    new = _frame_at(out, 1.5)
    assert orig.shape == new.shape
    h = orig.shape[0]
    band = slice(int(h * 0.85), h)
    top = slice(0, int(h * 0.7))
    # Bottom band lost its sharp edges (white box smeared).
    orig_var = _lap_var(cv2.cvtColor(orig[band], cv2.COLOR_BGR2GRAY))
    new_var = _lap_var(cv2.cvtColor(new[band], cv2.COLOR_BGR2GRAY))
    assert new_var < orig_var * 0.5, f"模糊后底部清晰度应大幅下降：{orig_var:.1f} -> {new_var:.1f}"
    # Top of the frame barely changed (re-encode drift only).
    drift = float(np.mean(cv2.absdiff(orig[top], new[top])))
    assert drift < 10.0, f"画面上部不应被动：平均像素差 {drift:.2f}"


def test_bar_blacks_band_and_keeps_rest(tmp_path):
    import cv2
    import numpy as np

    src = _make_video(tmp_path / "src.mp4")
    out = cover_subtitle_band(src, tmp_path / "bar.mp4", mode="bar", use_nvenc=False)
    assert out.is_file()
    orig = _frame_at(src, 1.5)
    new = _frame_at(out, 1.5)
    h = orig.shape[0]
    band = slice(int(h * 0.85), h)
    top = slice(0, int(h * 0.7))
    assert float(np.mean(new[band])) < 10.0, "黑条模式下底部应接近全黑"
    drift = float(np.mean(cv2.absdiff(orig[top], new[top])))
    assert drift < 10.0, f"画面上部不应被动：平均像素差 {drift:.2f}"


def test_inpaint_is_an_explicit_stub(tmp_path):
    src = _make_video(tmp_path / "src.mp4")
    with pytest.raises(NotImplementedError, match="修复重绘.*未实现"):
        inpaint_subtitle_band(src, tmp_path / "x.mp4")
    with pytest.raises(NotImplementedError, match="修复重绘.*未实现"):
        cover_subtitle_band(src, tmp_path / "x.mp4", mode="inpaint", use_nvenc=False)


def test_cover_bad_args(tmp_path):
    src = _make_video(tmp_path / "src.mp4")
    with pytest.raises(ValueError, match="未知的遮盖模式"):
        cover_subtitle_band(src, tmp_path / "x.mp4", mode="erase", use_nvenc=False)
    with pytest.raises(ValueError, match="字幕带顶部位置不合法"):
        cover_subtitle_band(src, tmp_path / "x.mp4", band_top=0.2, use_nvenc=False)
    with pytest.raises(FFmpegError, match="视频不存在"):
        cover_subtitle_band(tmp_path / "nope.mp4", tmp_path / "x.mp4", use_nvenc=False)


# --- compose.run() integration ------------------------------------------------

_SRT = """1
00:00:00,500 --> 00:00:01,500
新字幕第一行

2
00:00:02,000 --> 00:00:02,800
新字幕第二行
"""


def _compose_setup(tmp_path):
    from recast.config import RecastConfig

    video = _make_video(tmp_path / "src.mp4")
    srt = tmp_path / "new.srt"
    srt.write_text(_SRT, encoding="utf-8")
    return video, srt, RecastConfig()


def test_compose_covers_before_burning(tmp_path):
    import cv2

    from recast.stages.compose import run as compose_run

    video, srt, cfg = _compose_setup(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = compose_run(
        video, video, workdir, cfg, tmp_path / "out.mp4",
        subtitle=srt, burn_subs=True, cover_burned_subs=True, cover_mode="blur",
    )
    assert out.is_file()
    covered = workdir / "covered_subs.mp4"
    assert covered.is_file(), "遮盖后应留下中间文件 covered_subs.mp4"
    # The covered intermediate (before new subs are burned) is blurred at the bottom.
    orig = _frame_at(video, 1.5)
    cov = _frame_at(covered, 1.5)
    h = orig.shape[0]
    band = slice(int(h * 0.85), h)
    orig_var = _lap_var(cv2.cvtColor(orig[band], cv2.COLOR_BGR2GRAY))
    cov_var = _lap_var(cv2.cvtColor(cov[band], cv2.COLOR_BGR2GRAY))
    assert cov_var < orig_var * 0.5
    # Sidecar SRT still written.
    assert (tmp_path / "out.srt").is_file()


def test_compose_cover_off_and_bar(tmp_path):
    from recast.stages.compose import run as compose_run

    video, srt, cfg = _compose_setup(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = compose_run(
        video, video, workdir, cfg, tmp_path / "out.mp4",
        subtitle=srt, burn_subs=True, cover_burned_subs=False,
    )
    assert out.is_file()
    assert not (workdir / "covered_subs.mp4").exists()

    workdir2 = tmp_path / "work2"
    workdir2.mkdir()
    out2 = compose_run(
        video, video, workdir2, cfg, tmp_path / "out2.mp4",
        subtitle=srt, burn_subs=True, cover_burned_subs=True, cover_mode="bar",
    )
    assert out2.is_file()
    assert (workdir2 / "covered_subs.mp4").is_file()


def test_compose_cover_auto_without_ocr_skips_cover(tmp_path):
    # RapidOCR is not installed here: auto mode must degrade to "no cover"
    # and still produce the output, not crash.
    from recast.stages.compose import run as compose_run

    video, srt, cfg = _compose_setup(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = compose_run(
        video, video, workdir, cfg, tmp_path / "out.mp4",
        subtitle=srt, burn_subs=True,
    )
    assert out.is_file()
    assert not (workdir / "covered_subs.mp4").exists()


def test_compose_cover_bad_mode(tmp_path):
    from recast.stages.compose import run as compose_run

    video, srt, cfg = _compose_setup(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    with pytest.raises(ValueError, match="未知的遮盖模式"):
        compose_run(
            video, video, workdir, cfg, tmp_path / "out.mp4",
            subtitle=srt, burn_subs=True, cover_burned_subs=True, cover_mode="erase",
        )
    with pytest.raises(NotImplementedError, match="修复重绘.*未实现"):
        compose_run(
            video, video, workdir, cfg, tmp_path / "out.mp4",
            subtitle=srt, burn_subs=True, cover_burned_subs=True, cover_mode="inpaint",
        )
