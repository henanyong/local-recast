"""P2 small items: N2 history [一键复跑] rename + N28 export presets must
not touch GPU concurrency.

N2: the history re-run button must say it creates a NEW task and fully
re-runs it, so it cannot be confused with the main window's
[从失败步续跑] (same task resumes, no new task).

N28: an export quality preset only moves CRF / face-detect interval.
Concurrency is a hardware-risk setting the user changes explicitly in
settings; switching presets must leave it exactly as it was, on every
path that applies a preset (config.apply_runtime_settings, the settings
dialog buttons, loading a named preset that carries an export_preset).
"""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# N2: history re-run button wording
# ---------------------------------------------------------------------------


def test_rerun_button_names_new_task(svc):
    """Button text and tooltip say 新建任务 + 全量重跑 (semantics, not exact match)."""
    from recast.desktop.task_history_dialog import TaskHistoryDialog

    _app()
    dlg = TaskHistoryDialog(svc)
    try:
        assert "新建任务" in dlg.rerun_btn.text()
        tooltip = dlg.rerun_btn.toolTip()
        assert "新建任务" in tooltip
        assert "全量重跑" in tooltip
        # Distinguished from the main-window resume path: same task, no new task.
        assert "从失败步续跑" in tooltip
        assert "不新建任务" in tooltip
    finally:
        dlg.close()


# ---------------------------------------------------------------------------
# N28: export presets never touch concurrency
# ---------------------------------------------------------------------------

EXPECTED_QUALITY = {
    "high": (16, 2),
    "balanced": (18, 4),
    "fast": (23, 8),
}
PRESET_LABELS = {"high": "高清", "balanced": "均衡", "fast": "快速预览"}


def test_export_preset_mapping_has_no_concurrency():
    from recast.config import EXPORT_PRESETS

    assert set(EXPORT_PRESETS) == {"high", "balanced", "fast"}
    for name, (crf, detect_every) in EXPECTED_QUALITY.items():
        preset = EXPORT_PRESETS[name]
        assert "concurrency" not in preset
        assert (preset["crf"], preset["detect_every"]) == (crf, detect_every)


@pytest.mark.parametrize("preset_name", ["high", "balanced", "fast"])
@pytest.mark.parametrize("before", [1, 2])
def test_apply_runtime_settings_preset_keeps_concurrency(preset_name, before):
    """apply_runtime_settings: preset sets CRF/detect, concurrency untouched."""
    from types import SimpleNamespace

    from recast.config import ComposeConfig, RecastConfig, ReplaceConfig, apply_runtime_settings

    cfg = RecastConfig(compose=ComposeConfig(crf=18), replace=ReplaceConfig(detect_every=4))
    settings = SimpleNamespace(
        export_preset=preset_name, export_crf=18, detect_every=4, concurrency=before,
        use_nvenc=None, cover_mode=None, scene_threshold=None, min_shot_sec=None,
        pose_quality=None, pose_sample_every=None,
    )
    apply_runtime_settings(cfg, settings)
    crf, detect_every = EXPECTED_QUALITY[preset_name]
    assert (cfg.compose.crf, cfg.replace.detect_every) == (crf, detect_every)
    assert settings.concurrency == before


def test_settings_dialog_preset_buttons_keep_concurrency(svc):
    """Clicking each preset button in the dialog leaves the concurrency spinbox alone."""
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    try:
        for before in (1, 2):
            dlg.concurrency.setValue(before)
            for name, (crf, detect_every) in EXPECTED_QUALITY.items():
                dlg.preset_buttons[name].click()
                assert dlg.concurrency.value() == before
                assert (dlg.crf.value(), dlg.detect_every.value()) == (crf, detect_every)
                assert dlg.preset_label.text() == f"当前：{PRESET_LABELS[name]}"
            # Tooltip says plainly the preset does NOT change concurrency.
            assert "不改任务并发" in dlg.preset_buttons["high"].toolTip()
            assert "并发 2" not in dlg.preset_buttons["high"].toolTip()
    finally:
        dlg.close()


def test_settings_dialog_concurrency_edit_keeps_preset(svc):
    """Concurrency is no longer preset-driven: changing it keeps the preset active."""
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    try:
        dlg.preset_buttons["high"].click()
        assert dlg.preset_label.text() == "当前：高清"
        dlg.concurrency.setValue(2)
        assert dlg.preset_label.text() == "当前：高清"  # not 自定义
        dlg.apply()
        assert svc.settings.export_preset == "high"
        assert svc.settings.concurrency == 2
        assert svc.settings.export_crf == 16
        assert svc.settings.detect_every == 2
    finally:
        dlg.close()


def test_refresh_from_named_preset_keeps_concurrency(svc):
    """Loading a named preset carrying export_preset must not move concurrency."""
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    svc.update_settings(export_preset="high", export_crf=16, detect_every=2, concurrency=1)
    assert svc.settings_mgr.save_preset("高清档") is True
    dlg = SettingsDialog(svc)
    try:
        dlg.concurrency.setValue(2)  # user-side value, differs from stored 1
        loaded = svc.settings_mgr.load_preset("高清档")
        assert loaded.export_preset == "high"
        dlg._refresh_from_settings(loaded)
        assert dlg.concurrency.value() == 2  # untouched by the preset
        assert (dlg.crf.value(), dlg.detect_every.value()) == (16, 2)
        assert dlg.preset_label.text() == "当前：高清"
    finally:
        dlg.close()
