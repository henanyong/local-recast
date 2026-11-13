"""Tests for the subtitle-cover config (round: service/config).

ComposeConfig.cover_mode ("blur" | "bar" | "off") comes from the yaml
preset, the settings dialog can override it (persisted in AppSettings),
and the pipeline compose call takes its default from it via
``cover_options_for_compose``. No video, no ffmpeg, no compose run.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

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


def test_compose_config_default_is_blur():
    from recast.config import ComposeConfig

    assert ComposeConfig().cover_mode == "blur"


def test_compose_config_loads_from_yaml(tmp_path: Path):
    from recast.config import RecastConfig

    p = tmp_path / "cfg.yaml"
    p.write_text("compose:\n  crf: 20\n  cover_mode: bar\n", encoding="utf-8")
    cfg = RecastConfig.from_yaml(p)
    assert cfg.compose.cover_mode == "bar"
    assert cfg.compose.crf == 20

    p2 = tmp_path / "cfg2.yaml"
    p2.write_text("compose:\n  crf: 20\n", encoding="utf-8")
    assert RecastConfig.from_yaml(p2).compose.cover_mode == "blur"


def test_runtime_settings_override_cover_mode():
    from recast.config import ComposeConfig, RecastConfig, apply_runtime_settings

    cfg = RecastConfig(compose=ComposeConfig(cover_mode="blur"))
    apply_runtime_settings(cfg, SimpleNamespace(cover_mode="bar"))
    assert cfg.compose.cover_mode == "bar"

    cfg = RecastConfig(compose=ComposeConfig(cover_mode="blur"))
    apply_runtime_settings(cfg, SimpleNamespace(cover_mode="off"))
    assert cfg.compose.cover_mode == "off"

    # None / invalid values keep the yaml preset.
    cfg = RecastConfig(compose=ComposeConfig(cover_mode="bar"))
    apply_runtime_settings(cfg, SimpleNamespace(cover_mode=None))
    assert cfg.compose.cover_mode == "bar"
    apply_runtime_settings(cfg, SimpleNamespace(cover_mode="weird"))
    assert cfg.compose.cover_mode == "bar"
    apply_runtime_settings(cfg, None)
    assert cfg.compose.cover_mode == "bar"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("blur", (None, "blur")),
        ("bar", (None, "bar")),
        ("off", (False, "blur")),
        (None, (None, "blur")),
        ("", (None, "blur")),
        ("  bar  ", (None, "bar")),
        # Passed through untouched: compose.run validates unknown modes.
        ("inpaint", (None, "inpaint")),
        ("weird", (None, "weird")),
    ],
)
def test_cover_options_for_compose(value, expected):
    from recast.core.runner import cover_options_for_compose

    assert cover_options_for_compose(value) == expected


def test_settings_dialog_cover_combo_persists(svc, tmp_path: Path):
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    # Default follows the yaml preset (repo preset has no cover_mode).
    assert dlg.cover_mode.currentData() == "blur"
    dlg.cover_mode.setCurrentIndex(dlg.cover_mode.findData("bar"))
    dlg.apply()

    assert svc.settings.cover_mode == "bar"
    # Persisted in the store: a fresh manager reads it back.
    from recast.core.settings import SettingsManager

    assert SettingsManager(svc.store).current.cover_mode == "bar"
    dlg.close()


def test_settings_dialog_cover_off_persists(svc, tmp_path: Path):
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    dlg = SettingsDialog(svc)
    dlg.cover_mode.setCurrentIndex(dlg.cover_mode.findData("off"))
    dlg.apply()
    assert svc.settings.cover_mode == "off"
    dlg.close()


def test_settings_dialog_shows_stored_cover_mode(svc, tmp_path: Path):
    from recast.desktop.settings_dialog import SettingsDialog

    _app()
    svc.update_settings(cover_mode="bar")
    dlg = SettingsDialog(svc)
    assert dlg.cover_mode.currentData() == "bar"
    dlg.close()
