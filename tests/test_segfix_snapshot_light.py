"""Segfault-fix wave 2, item 3: light engine snapshot for the 400ms tick.

The timer tick must reuse a warm ``_heavy_snapshot`` cache past its 5s TTL
instead of re-probing heavy modules, while explicit full refreshes keep the
full recompute. Covers:

1. light engine reuses a warm cache (no recompute) and survives TTL expiry;
2. light engine with an empty cache still computes a complete snapshot;
3. a key change (hardware tier / output root) defeats the light reuse;
4. ``RecastService.snapshot(light_engine=...)`` / ``engine(light=...)``
   pass the flag through to ``engine_snapshot``; defaults stay full;
5. ``MainWindow.refresh()`` (timer tick) passes ``light_engine=True`` and
   ``refresh(full=True)`` passes ``light_engine=False``; a second
   ``refresh()`` while one worker runs still starts only one worker and
   sets ``_snap_again``;
6. ``_snap_busy`` returns to False once ``_apply_snapshot`` has run
   (including after a queued second refresh drains).

Call-site audit for item 4 of the design (done 2026-11-13, see report):
every ``refresh()`` caller in ``recast/desktop/main_window.py`` runs on the
main thread -- constructor (186), filter signal (239), retry/failed-card
callback (826, via ``_paint`` <- ``_apply_snapshot`` signal slot), button /
dialog slots (1459, 1563, 1574, 1592, 1654, 2139, 2160, 2233, 2239, 2249,
2319, 2340), the rerender-result signal slot (2221), and ``_apply_snapshot``
itself (1165). No worker-thread call site exists, so no call site had to be
moved and no extra regression test for a moved call site is needed.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import recast.core.engine as eng
from recast.core.engine import (
    _heavy_snapshot,
    clear_engine_cache,
    engine_snapshot,
)
from recast.core.settings import AppSettings

MODULE_LABELS = {
    "torch",
    "faster_whisper",
    "ultralytics",
    "insightface",
    "onnxruntime",
    "cv2",
    "scenedetect",
    "rapidocr",
    "PySide6",
}
HEAVY_KEYS = {
    "vram_gb",
    "gpu_name",
    "ffmpeg",
    "modules",
    "models",
    "disk_free_gb",
    "comfyui",
    "liveportrait",
    "scene",
    "encode",
}


@pytest.fixture(autouse=True)
def _fresh_engine_cache():
    clear_engine_cache()
    yield
    clear_engine_cache()


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


def _counting_probes(monkeypatch):
    """Count the expensive probes inside _heavy_snapshot's compute body."""
    calls = {"vram": 0, "gpu": 0, "modules": 0}
    orig_vram = eng.detect_vram_gb
    orig_gpu = eng.detect_gpu_name
    orig_modules = eng.module_status

    def counting_vram():
        calls["vram"] += 1
        return orig_vram()

    def counting_gpu():
        calls["gpu"] += 1
        return orig_gpu()

    def counting_modules():
        calls["modules"] += 1
        return orig_modules()

    monkeypatch.setattr(eng, "detect_vram_gb", counting_vram)
    monkeypatch.setattr(eng, "detect_gpu_name", counting_gpu)
    monkeypatch.setattr(eng, "module_status", counting_modules)
    return calls


def _warm(settings: AppSettings) -> dict:
    return engine_snapshot(settings)


def _expire_heavy_cache() -> None:
    """Fake the 5s TTL having passed without waiting for it."""
    assert eng._heavy_cache is not None, "cache must be warm before expiring it"
    ts, key, data = eng._heavy_cache
    eng._heavy_cache = (time.monotonic() - 10.0, key, data)


def _pump(seconds: float = 0.02) -> None:
    _app().processEvents()
    time.sleep(seconds)


def _wait_until(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        _pump()
    raise AssertionError(f"timed out waiting for {what}")


def _settled_window(svc):
    """Build a MainWindow, stop its 400ms timer, wait for the constructor's
    initial full refresh to finish so later assertions see only our calls."""
    from recast.desktop.main_window import MainWindow

    _app()
    window = MainWindow(svc)
    window.timer.stop()
    _wait_until(lambda: not window._snap_busy, what="initial refresh to settle")
    assert window._snap_again is False
    return window


# ---------------------------------------------------------------------------
# 1. light engine reuses a warm cache, even past the TTL
# ---------------------------------------------------------------------------


def test_light_reuses_warm_cache_without_recompute(monkeypatch):
    calls = _counting_probes(monkeypatch)
    settings = AppSettings()
    warm = _warm(settings)
    assert calls["vram"] == 1 and calls["modules"] == 1
    calls["vram"] = calls["gpu"] = calls["modules"] = 0

    first = engine_snapshot(settings, light=True)
    second = engine_snapshot(settings, light=True)
    assert calls == {"vram": 0, "gpu": 0, "modules": 0}, f"light call recomputed: {calls}"
    assert first["modules"] == warm["modules"]
    assert second["modules"] == warm["modules"]
    # Same cached heavy dict is reused, not a fresh probe.
    assert first["modules"] is warm["modules"]


def test_light_returns_stale_cache_after_ttl_but_full_recomputes(monkeypatch):
    calls = _counting_probes(monkeypatch)
    settings = AppSettings()
    warm = _warm(settings)
    _expire_heavy_cache()
    calls["vram"] = calls["gpu"] = calls["modules"] = 0

    stale = engine_snapshot(settings, light=True)
    assert calls == {"vram": 0, "gpu": 0, "modules": 0}, f"light call recomputed past TTL: {calls}"
    assert stale["modules"] is warm["modules"]

    fresh = engine_snapshot(settings)  # default full: TTL has passed
    assert calls["vram"] == 1, f"full call did not recompute past TTL: {calls}"
    assert calls["modules"] == 1
    assert fresh["modules"] == warm["modules"]


# ---------------------------------------------------------------------------
# 2. light engine with an empty cache still computes a complete snapshot
# ---------------------------------------------------------------------------


def test_light_first_call_computes_complete_snapshot():
    snap = engine_snapshot(AppSettings(), light=True)
    assert set(snap["modules"]) == MODULE_LABELS
    for key in ("vram_gb", "gpu_name", "ffmpeg", "models", "disk_free_gb",
                "comfyui", "liveportrait", "scene", "encode", "state",
                "hardware_tier", "capability"):
        assert key in snap, f"light first call missing {key}"
    heavy = _heavy_snapshot(AppSettings(), None)
    assert set(heavy) == HEAVY_KEYS


def test_light_first_call_matches_full_structure():
    clear_engine_cache()
    light = engine_snapshot(AppSettings(), light=True)
    clear_engine_cache()
    full = engine_snapshot(AppSettings())
    assert set(light) == set(full)
    assert light["modules"] == full["modules"]
    assert set(light["modules"]) == MODULE_LABELS
    assert light["scene"] == full["scene"]
    assert light["encode"] == full["encode"]


# ---------------------------------------------------------------------------
# 3. a key change defeats the light reuse
# ---------------------------------------------------------------------------


def test_light_recomputes_when_hardware_tier_changes(monkeypatch):
    calls = _counting_probes(monkeypatch)
    settings = AppSettings()
    _warm(settings)
    calls["vram"] = calls["gpu"] = calls["modules"] = 0

    settings.hardware_tier = "basic"
    snap = engine_snapshot(settings, light=True)
    assert calls["vram"] == 1, f"tier change did not recompute: {calls}"
    assert snap["hardware_tier"] == "basic"
    assert snap["capability"]["tier"] == "basic"


def test_light_recomputes_when_output_root_changes(monkeypatch, tmp_path: Path):
    calls = _counting_probes(monkeypatch)
    settings = AppSettings(output_root=str(tmp_path / "a"))
    _warm(settings)
    calls["vram"] = calls["gpu"] = calls["modules"] = 0

    settings.output_root = str(tmp_path / "b")
    engine_snapshot(settings, light=True)
    assert calls["vram"] == 1, f"output_root change did not recompute: {calls}"
    assert calls["modules"] == 1


# ---------------------------------------------------------------------------
# 4. service.snapshot / service.engine pass the flag through
# ---------------------------------------------------------------------------


def test_service_snapshot_light_engine_passes_through(svc, monkeypatch):
    import recast.core.service as svc_mod

    seen: list[dict] = []
    orig = svc_mod.engine_snapshot

    def recording(settings=None, **kwargs):
        seen.append(dict(kwargs))
        return orig(settings, **kwargs)

    monkeypatch.setattr(svc_mod, "engine_snapshot", recording)
    svc.snapshot(light_engine=True)
    assert seen and seen[-1].get("light") is True
    svc.snapshot()
    assert seen[-1].get("light") is False, "default snapshot() must stay full"


def test_service_engine_light_flag_passes_through(svc, monkeypatch):
    import recast.core.service as svc_mod

    seen: list[dict] = []
    orig = svc_mod.engine_snapshot

    def recording(settings=None, **kwargs):
        seen.append(dict(kwargs))
        return orig(settings, **kwargs)

    monkeypatch.setattr(svc_mod, "engine_snapshot", recording)
    svc.engine(light=True)
    assert seen and seen[-1].get("light") is True
    svc.engine()
    assert seen[-1].get("light") is False


# ---------------------------------------------------------------------------
# 5. MainWindow.refresh(): tick = light, full = full; double refresh = 1 worker
# ---------------------------------------------------------------------------


def _recording_snapshot(svc, calls: list):
    orig = svc.snapshot

    def recording(status_filter=None, **kwargs):
        calls.append(dict(kwargs))
        return orig(status_filter, **kwargs)

    return recording


def test_refresh_tick_uses_light_engine(svc, monkeypatch):
    window = _settled_window(svc)
    calls: list[dict] = []
    monkeypatch.setattr(window.service, "snapshot", _recording_snapshot(svc, calls))
    window.refresh()  # timer tick: full=False
    _wait_until(lambda: calls and not window._snap_busy, what="tick snapshot to finish")
    assert calls[-1].get("light_engine") is True, f"tick passed {calls[-1]}"


def test_refresh_full_uses_full_engine(svc, monkeypatch):
    window = _settled_window(svc)
    calls: list[dict] = []
    monkeypatch.setattr(window.service, "snapshot", _recording_snapshot(svc, calls))
    window.refresh(full=True)
    _wait_until(lambda: calls and not window._snap_busy, what="full snapshot to finish")
    assert calls[-1].get("light_engine") is False, f"full refresh passed {calls[-1]}"


def test_double_refresh_runs_single_worker_and_sets_again(svc, monkeypatch):
    window = _settled_window(svc)
    orig = svc.snapshot
    calls = {"n": 0}
    gate = threading.Event()

    def blocking(status_filter=None, **kwargs):
        calls["n"] += 1
        assert gate.wait(timeout=10), "worker snapshot was never released"
        return orig(status_filter, **kwargs)

    monkeypatch.setattr(window.service, "snapshot", blocking)
    window.refresh()
    _wait_until(lambda: window._snap_busy, what="first worker to start")
    # Second refresh while the first worker is still inside snapshot().
    window.refresh()
    assert window._snap_again is True, "second refresh while busy must set _snap_again"
    assert calls["n"] == 1, f"second refresh started a second worker: {calls}"
    gate.set()
    # The queued refresh fires from _apply_snapshot: a second worker runs.
    _wait_until(lambda: calls["n"] == 2 and not window._snap_busy,
                what="queued refresh to run and settle")
    assert window._snap_again is False


# ---------------------------------------------------------------------------
# 6. _snap_busy returns to False after _apply_snapshot runs
# ---------------------------------------------------------------------------


def test_snap_busy_returns_false_after_apply(svc, monkeypatch):
    window = _settled_window(svc)
    calls: list[dict] = []
    monkeypatch.setattr(window.service, "snapshot", _recording_snapshot(svc, calls))
    assert window._snap_busy is False
    window.refresh()
    assert window._snap_busy is True
    _wait_until(lambda: not window._snap_busy, what="_apply_snapshot to clear _snap_busy")
    assert window._snap_busy is False
    assert window._snap_again is False
    assert window._snap_full is False


def test_snap_busy_false_after_queued_refresh_drains(svc, monkeypatch):
    window = _settled_window(svc)
    orig = svc.snapshot
    calls = {"n": 0}
    gate = threading.Event()

    def blocking(status_filter=None, **kwargs):
        calls["n"] += 1
        assert gate.wait(timeout=10), "worker snapshot was never released"
        return orig(status_filter, **kwargs)

    monkeypatch.setattr(window.service, "snapshot", blocking)
    window.refresh()
    _wait_until(lambda: window._snap_busy, what="first worker to start")
    window.refresh()  # queued while busy
    assert window._snap_again is True
    gate.set()
    _wait_until(lambda: calls["n"] == 2 and not window._snap_busy,
                what="queued refresh to drain")
    assert window._snap_busy is False
    assert window._snap_again is False
    assert window._snap_full is False
