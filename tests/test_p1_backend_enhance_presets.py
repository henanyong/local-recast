"""P1 backend: per-shot enhance (4.2), export presets (8.5/10.5),
named setting presets, subtitle-dialog cover wiring (7a).

Enhance is a real ffmpeg filter (light denoise + sharpen), verified on a
generated clip: output decodes, bytes differ, and the pixel change is
concentrated inside the enhanced time range.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def svc(tmp_path: Path):
    from recast.core.service import RecastService

    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


@pytest.fixture()
def task_with_shots(svc, tmp_path: Path):
    drama = svc.store.insert(
        "dramas", origin_name="剧", root_path=str(tmp_path / "show"), episode_count=1
    )
    workdir = tmp_path / "w1"
    task = svc.store.insert(
        "tasks",
        drama_id=drama["id"],
        episode_name="e1",
        video_path=str(tmp_path / "a.mp4"),
        workdir=str(workdir),
    )
    svc.store.replace_shots(
        task["id"],
        [
            {"shot_code": "S01", "start_time": 0.0, "end_time": 1.0},
            {"shot_code": "S02", "start_time": 1.0, "end_time": 2.0},
        ],
    )
    return svc, task, workdir


def _make_clip(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:d={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
    )
    return path


def _frame_md5(video: Path, at: float) -> bytes:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(at), "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return hashlib.md5(raw).digest()


def _mean_abs_diff(a: Path, b: Path, at: float) -> float:
    import numpy as np

    def frame(video: Path) -> "np.ndarray":
        raw = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", str(at), "-i", str(video), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True,
        ).stdout
        return np.frombuffer(raw, dtype=np.uint8).astype(int)

    return float(abs(frame(a) - frame(b)).mean())


# ---------------------------------------------------------------------------
# 1. Per-shot enhance
# ---------------------------------------------------------------------------


def test_enhance_filter_ranges():
    from recast.ffmpeg import enhance_filter_for_shots

    assert enhance_filter_for_shots([]) is None
    assert enhance_filter_for_shots(None) is None
    assert enhance_filter_for_shots([{"start": 2.0, "end": 1.0}]) is None
    vf = enhance_filter_for_shots([{"start": 0.5, "end": 1.5}])
    assert vf is not None
    assert "hqdn3d=4:3:6:4.5" in vf and "unsharp=5:5:0.8" in vf
    assert "between(t,0.5,1.5)" in vf
    # Two shots -> two enabled ranges, not one global filter.
    vf2 = enhance_filter_for_shots(
        [{"start": 0.0, "end": 1.0}, {"start": 3.0, "end": 4.0}]
    )
    assert vf2 is not None
    assert "between(t,0,1)" in vf2 and "between(t,3,4)" in vf2


def test_enhance_really_processes_video(tmp_path: Path):
    from recast.ffmpeg import apply_shot_enhance, probe_video

    src = _make_clip(tmp_path / "src.mp4")
    out = apply_shot_enhance(
        src, tmp_path / "enh.mp4", [{"start": 0.5, "end": 1.5}],
        crf=18, use_nvenc=False,
    )
    assert out.is_file()
    info = probe_video(out)
    assert info.duration == pytest.approx(2.0, abs=0.1)
    assert info.has_audio  # audio stream-copied, not dropped
    assert (tmp_path / "enh.mp4").read_bytes() != src.read_bytes()
    # Pixels move much more inside the enhanced range than outside it
    # (outside movement is only re-encode noise).
    outside = _mean_abs_diff(src, out, 0.2)
    inside = _mean_abs_diff(src, out, 1.0)
    assert inside > outside * 2


def test_enhance_rejects_empty_and_missing(tmp_path: Path):
    from recast.ffmpeg import apply_shot_enhance

    src = _make_clip(tmp_path / "src.mp4")
    with pytest.raises(ValueError):
        apply_shot_enhance(src, tmp_path / "x.mp4", [], crf=18, use_nvenc=False)
    with pytest.raises(FileNotFoundError):
        apply_shot_enhance(
            tmp_path / "nope.mp4", tmp_path / "x.mp4",
            [{"start": 0.0, "end": 1.0}], crf=18, use_nvenc=False,
        )


def test_compose_run_with_enhance(tmp_path: Path):
    from recast.config import RecastConfig
    from recast.ffmpeg import probe_video
    from recast.stages import compose

    src = _make_clip(tmp_path / "src.mp4")
    cfg = RecastConfig()
    cfg.compose.use_nvenc = False
    workdir = tmp_path / "work"
    workdir.mkdir()
    plain = compose.run(src, src, workdir, cfg, tmp_path / "plain.mp4")
    assert probe_video(plain).duration == pytest.approx(2.0, abs=0.1)
    enhanced = compose.run(
        src, src, workdir, cfg, tmp_path / "enhanced.mp4",
        enhance_shots=[{"shot_code": "S01", "start": 0.5, "end": 1.5}],
    )
    assert probe_video(enhanced).duration == pytest.approx(2.0, abs=0.1)
    assert enhanced.read_bytes() != plain.read_bytes()
    assert (workdir / "enhanced.mp4").is_file()
    # No enhance flags -> compose does not create the intermediate file.
    workdir2 = tmp_path / "work2"
    workdir2.mkdir()
    compose.run(src, src, workdir2, cfg, tmp_path / "plain2.mp4")
    assert not (workdir2 / "enhanced.mp4").exists()


def test_shot_enhance_flags_roundtrip(task_with_shots):
    svc, task, workdir = task_with_shots
    assert svc.shot_enhance_flags(task["id"]) == {}  # default: all off
    assert svc.shot_enhance_flags("no-such-task") == {}  # unknown task: no raise
    assert svc.shot_enhance_flags(None) == {}
    assert svc.set_shot_enhance(task["id"], "S01", True) == {"S01": True}
    assert svc.set_shot_enhance(task["id"], "S02", True) == {"S01": True, "S02": True}
    assert svc.set_shot_enhance(task["id"], "S01", False) == {"S02": True}
    assert svc.shot_enhance_flags(task["id"]) == {"S02": True}
    # Only switched-on shots are persisted.
    assert (workdir / "shots_enhance.json").read_text(encoding="utf-8").strip() != ""
    import json

    assert json.loads((workdir / "shots_enhance.json").read_text(encoding="utf-8")) == {"S02": True}
    assert svc.shot_enhance_flags(task["id"]) == {"S02": True}


def test_shot_enhance_unknown_shot_errors(task_with_shots):
    svc, task, _workdir = task_with_shots
    with pytest.raises(ValueError, match="S99"):
        svc.set_shot_enhance(task["id"], "S99", True)
    with pytest.raises(ValueError):
        svc.set_shot_enhance(task["id"], "", True)
    with pytest.raises(FileNotFoundError):
        svc.set_shot_enhance("no-such-task", "S01", True)


def test_runner_enhance_shots_for_task(task_with_shots):
    svc, task, workdir = task_with_shots
    from recast.core.runner import enhance_shots_for_task

    assert enhance_shots_for_task(svc.store, task["id"], workdir) == []
    svc.set_shot_enhance(task["id"], "S02", True)
    assert enhance_shots_for_task(svc.store, task["id"], workdir) == [
        {"shot_code": "S02", "start": 1.0, "end": 2.0}
    ]


# ---------------------------------------------------------------------------
# 2. Export quality presets
# ---------------------------------------------------------------------------


def test_export_preset_mapping():
    from recast.config import EXPORT_PRESETS

    # N28: presets never touch concurrency (only CRF / detect interval).
    assert EXPORT_PRESETS["high"] == {"crf": 16, "detect_every": 2}
    assert EXPORT_PRESETS["balanced"] == {"crf": 18, "detect_every": 4}
    assert EXPORT_PRESETS["fast"] == {"crf": 23, "detect_every": 8}
    # "balanced" equals the existing defaults (yaml CRF 18, detect_every 4):
    # no conflict with the old defaults.
    from recast.config import ComposeConfig, ReplaceConfig

    assert ComposeConfig().crf == 18
    assert ReplaceConfig().detect_every == 4


def test_apply_runtime_settings_preset_wins():
    from types import SimpleNamespace

    from recast.config import ComposeConfig, RecastConfig, ReplaceConfig, apply_runtime_settings

    cfg = RecastConfig(compose=ComposeConfig(crf=18), replace=ReplaceConfig(detect_every=4))
    settings = SimpleNamespace(
        export_preset="fast", export_crf=18, detect_every=4, concurrency=1,
        use_nvenc=None, cover_mode=None, scene_threshold=None, min_shot_sec=None,
        pose_quality=None, pose_sample_every=None,
    )
    apply_runtime_settings(cfg, settings)
    # N28: preset sets CRF/detect interval, concurrency stays as it was (1).
    assert (cfg.compose.crf, cfg.replace.detect_every, settings.concurrency) == (23, 8, 1)

    cfg = RecastConfig(compose=ComposeConfig(crf=18), replace=ReplaceConfig(detect_every=4))
    settings = SimpleNamespace(
        export_preset="high", export_crf=23, detect_every=8, concurrency=1,
        use_nvenc=None, cover_mode=None, scene_threshold=None, min_shot_sec=None,
        pose_quality=None, pose_sample_every=None,
    )
    apply_runtime_settings(cfg, settings)
    assert (cfg.compose.crf, cfg.replace.detect_every, settings.concurrency) == (16, 2, 1)


def test_apply_runtime_settings_custom_fields_without_preset():
    from types import SimpleNamespace

    from recast.config import ComposeConfig, RecastConfig, ReplaceConfig, apply_runtime_settings

    cfg = RecastConfig(compose=ComposeConfig(crf=18), replace=ReplaceConfig(detect_every=4))
    settings = SimpleNamespace(
        export_preset=None, export_crf=20, detect_every=6, concurrency=1,
        use_nvenc=None, cover_mode=None, scene_threshold=None, min_shot_sec=None,
        pose_quality=None, pose_sample_every=None,
    )
    apply_runtime_settings(cfg, settings)
    assert (cfg.compose.crf, cfg.replace.detect_every) == (20, 6)
    assert settings.concurrency == 1  # untouched without a preset


def test_manual_field_edit_clears_export_preset(svc):
    svc.update_settings(export_preset="high", export_crf=16, detect_every=2, concurrency=2)
    assert svc.settings.export_preset == "high"
    svc.update_settings(export_crf=19)
    assert svc.settings.export_preset is None  # -> 自定义
    assert svc.settings.export_crf == 19
    svc.update_settings(export_preset="fast", export_crf=23, detect_every=8, concurrency=2)
    assert svc.settings.export_preset == "fast"
    svc.update_settings(detect_every=5)
    assert svc.settings.export_preset is None


def test_settings_dialog_preset_buttons(svc):
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    assert dlg.preset_label.text() == "当前：自定义"
    before = dlg.concurrency.value()
    dlg.preset_buttons["fast"].click()
    # N28: preset buttons set CRF/detect interval only, concurrency untouched.
    assert (dlg.crf.value(), dlg.detect_every.value(), dlg.concurrency.value()) == (23, 8, before)
    assert dlg.preset_label.text() == "当前：快速预览"
    dlg.preset_buttons["high"].click()
    assert (dlg.crf.value(), dlg.detect_every.value(), dlg.concurrency.value()) == (16, 2, before)
    assert dlg.preset_label.text() == "当前：高清"
    # Manual edit of one driven field -> back to 自定义.
    dlg.crf.setValue(19)
    assert dlg.preset_label.text() == "当前：自定义"
    dlg.apply()
    assert svc.settings.export_preset is None
    assert svc.settings.export_crf == 19
    dlg.close()


def test_settings_dialog_preset_persists(svc):
    from recast.core.settings import SettingsManager
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    dlg.preset_buttons["balanced"].click()
    dlg.apply()
    assert svc.settings.export_preset == "balanced"
    assert SettingsManager(svc.store).current.export_preset == "balanced"
    dlg2 = SettingsDialog(svc)
    assert dlg2.preset_label.text() == "当前：均衡"
    dlg.close()
    dlg2.close()


# ---------------------------------------------------------------------------
# 3. Named setting presets
# ---------------------------------------------------------------------------


def test_named_preset_roundtrip(svc):
    mgr = svc.settings_mgr
    svc.update_settings(
        export_crf=20, use_nvenc=False, cover_mode="bar", scene_threshold=30.0,
        min_shot_sec=0.6, pose_quality="speed", pose_sample_every=3,
        detect_every=6, export_preset=None, burn_subs=True, max_cps=3.5,
    )
    assert mgr.save_preset("快剪") is True
    assert mgr.list_presets() == ["快剪"]
    svc.update_settings(
        export_crf=16, use_nvenc=True, cover_mode="blur", scene_threshold=27.0,
        min_shot_sec=0.4, pose_quality="accuracy", pose_sample_every=2,
        detect_every=4, export_preset="high", burn_subs=False, max_cps=4.0,
    )
    loaded = mgr.load_preset("快剪")
    assert loaded.export_crf == 20
    assert loaded.use_nvenc is False
    assert loaded.cover_mode == "bar"
    assert loaded.scene_threshold == 30.0
    assert loaded.min_shot_sec == 0.6
    assert loaded.pose_quality == "speed"
    assert loaded.pose_sample_every == 3
    assert loaded.detect_every == 6
    assert loaded.export_preset is None
    assert loaded.burn_subs is True
    assert loaded.max_cps == 3.5
    # Persisted: a fresh manager reads the same snapshot back.
    from recast.core.settings import SettingsManager

    assert SettingsManager(svc.store).list_presets() == ["快剪"]


def test_named_preset_delete_and_missing(svc):
    mgr = svc.settings_mgr
    mgr.save_preset("A")
    mgr.save_preset("B")
    assert mgr.list_presets() == ["A", "B"]
    mgr.delete_preset("A")
    assert mgr.list_presets() == ["B"]
    with pytest.raises(KeyError):
        mgr.delete_preset("A")
    with pytest.raises(KeyError):
        mgr.load_preset("不存在")
    with pytest.raises(ValueError):
        mgr.save_preset("   ")


def test_named_preset_overwrite_needs_confirmation(svc):
    mgr = svc.settings_mgr
    svc.update_settings(export_crf=16)
    assert mgr.save_preset("P") is True
    svc.update_settings(export_crf=23)
    # Declined: old snapshot survives (and load_preset applied it, so the
    # current value is back to 16 — set 23 again before confirming).
    assert mgr.save_preset("P", confirm_overwrite=lambda name: False) is False
    assert mgr.load_preset("P").export_crf == 16
    svc.update_settings(export_crf=23)
    # Confirmed: replaced.
    assert mgr.save_preset("P", confirm_overwrite=lambda name: True) is True
    assert mgr.load_preset("P").export_crf == 23


def test_settings_dialog_named_preset_flow(svc):
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    svc.update_settings(export_crf=20, detect_every=6, cover_mode="bar", burn_subs=True)
    dlg = SettingsDialog(svc)
    assert dlg.named_preset_combo.count() == 0
    # Save via the dialog's own handler is interactive (QInputDialog); drive
    # the manager + combo refresh the same way the handler does, then check
    # the dialog reflects a load.
    svc.settings_mgr.save_preset("对话框预设")
    dlg._refresh_named_preset_combo(select="对话框预设")
    assert dlg.named_preset_combo.currentData() == "对话框预设"
    svc.update_settings(export_crf=16, detect_every=2, cover_mode="blur", burn_subs=False)
    dlg._load_named_preset()
    assert dlg.crf.value() == 20
    assert dlg.detect_every.value() == 6
    assert dlg.cover_mode.currentData() == "bar"
    assert dlg.burn.isChecked() is True
    dlg.close()


# ---------------------------------------------------------------------------
# 4. Subtitle dialog cover wiring (7a)
# ---------------------------------------------------------------------------


def test_subtitle_dialog_docstring_uses_config_not_hardcoded():
    from pathlib import Path as _Path

    text = (_Path(__file__).resolve().parents[1] / "recast" / "desktop" / "subtitle_dialog.py").read_text(encoding="utf-8")
    assert 'cover_mode="blur"' not in text
    assert "cover_options_for_compose" in text
    assert "cfg.compose.cover_mode" in text
    assert "cover_burned_subs=cover_burned_subs" in text


def test_cover_options_all_three_modes():
    from recast.core.runner import cover_options_for_compose

    assert cover_options_for_compose("blur") == (None, "blur")
    assert cover_options_for_compose("bar") == (None, "bar")
    assert cover_options_for_compose("off") == (False, "blur")
