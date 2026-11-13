"""Segfault-fix regression tests for recast/core/engine.py (wave 1).

Covers the two wave-1 items:

1. ``module_status()`` no longer ``__import__``s heavy modules -- presence is
   decided by ``sys.modules`` membership or ``importlib.util.find_spec``.
   (The rapidocr branch still goes through ``ocr_available()`` by design.)
2. ``_heavy_snapshot()`` is serialised by ``_heavy_lock`` (double-checked
   locking, 5s TTL kept); ``clear_engine_cache()`` and the
   ``cached_comfy_available()`` cache accesses take the same lock.

Honest scope note: a segfault is not deterministically reproducible, so these
tests prove the two *mechanisms* -- "no new heavy import is triggered" and
"snapshot computation is serialised" -- not "the segfault can never recur".

Cache-copy choice (pinned here): ``module_status()`` returns a *copy* of the
cached dict on both the cold path and cache hits, so a caller mutating the
returned dict cannot poison the process-wide cache.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
import threading
import types

import pytest

import recast.core.engine as eng
from recast.core.engine import (
    _heavy_snapshot,
    clear_engine_cache,
    engine_snapshot,
    module_status,
)
from recast.core.settings import AppSettings

HEAVY = [
    "torch",
    "faster_whisper",
    "ultralytics",
    "insightface",
    "onnxruntime",
    "cv2",
    "scenedetect",
    "PySide6",
]
ALL_LABELS = HEAVY + ["rapidocr"]
# Fixed label order, matching module_status().
LABEL_ORDER = [
    "torch",
    "faster_whisper",
    "ultralytics",
    "insightface",
    "onnxruntime",
    "cv2",
    "scenedetect",
    "rapidocr",
    "PySide6",
]
PANEL_KEYS = ["torch", "faster_whisper", "insightface", "scenedetect", "rapidocr"]


@pytest.fixture(autouse=True)
def _fresh_engine_cache():
    clear_engine_cache()
    yield
    clear_engine_cache()


def _expected_module_status() -> dict[str, bool]:
    from recast.extract.ocr import ocr_available

    out = {}
    for label in LABEL_ORDER:
        if label == "rapidocr":
            out[label] = ocr_available()[0]
        else:
            out[label] = label in sys.modules or importlib.util.find_spec(label) is not None
    return out


# ---------------------------------------------------------------------------
# 1. module_status() triggers no new heavy import
# ---------------------------------------------------------------------------


def test_module_status_imports_no_heavy_module(monkeypatch):
    real_import = builtins.__import__
    seen: list[str] = []

    def recording_import(name, *args, **kwargs):
        seen.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    before = {m for m in HEAVY if m in sys.modules}
    clear_engine_cache()
    result = module_status()
    after = {m for m in HEAVY if m in sys.modules}
    # None of the 8 heavy modules was passed to __import__ at all. The
    # rapidocr branch's `from recast.extract.ocr import ocr_available` is
    # allowed and shows up as "recast.extract.ocr", not as a heavy name.
    assert not (set(seen) & set(HEAVY)), f"__import__ saw heavy modules: {seen}"
    assert after == before, f"sys.modules gained heavy keys: {after - before}"
    assert set(result) == set(ALL_LABELS)


def test_module_status_find_spec_failure_is_false(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def failing_find_spec(name, *args, **kwargs):
        if name in HEAVY:
            raise RuntimeError(f"boom for {name}")
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", failing_find_spec)
    clear_engine_cache()
    result = module_status()  # must not raise
    # A module already in sys.modules is True via that branch without ever
    # reaching find_spec (other test files may have imported e.g. cv2 first);
    # every other heavy module must come back False from the raised find_spec.
    for label in HEAVY:
        assert result[label] is (label in sys.modules)
    assert result["torch"] is False  # never importable here: proves the path
    assert isinstance(result["rapidocr"], bool)


# ---------------------------------------------------------------------------
# 2. Return-value semantics
# ---------------------------------------------------------------------------


def test_module_status_semantics():
    clear_engine_cache()
    expected = _expected_module_status()
    result = module_status()
    assert list(result) == LABEL_ORDER
    assert set(result) == set(ALL_LABELS)
    assert all(isinstance(v, bool) for v in result.values())
    assert result == expected


def test_module_status_sys_modules_hit_counts_as_installed(monkeypatch):
    # torch is not installed in this environment (find_spec -> None), so a
    # sys.modules hit is the only thing that can make it True here.
    assert importlib.util.find_spec("torch") is None
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    clear_engine_cache()
    assert module_status()["torch"] is True


# ---------------------------------------------------------------------------
# 3. Cache behaviour
# ---------------------------------------------------------------------------


def test_module_status_cache_is_stable_and_copied():
    clear_engine_cache()
    first = module_status()
    second = module_status()
    assert first == second
    assert first is not second  # cache hits return a copy (see module docstring)
    first["torch"] = not first["torch"]  # mutating the result must not poison
    assert module_status() == second


def test_module_status_cache_invalidation_with_fake_torch(monkeypatch):
    clear_engine_cache()
    baseline = module_status()["torch"]
    assert baseline is False  # torch is not installed here
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    # Still cached: the injected module is invisible until the cache is reset.
    assert module_status()["torch"] is False
    clear_engine_cache()
    assert module_status()["torch"] is True
    monkeypatch.delitem(sys.modules, "torch")
    clear_engine_cache()
    assert module_status()["torch"] is False


# ---------------------------------------------------------------------------
# 4. Lock behaviour of _heavy_snapshot()
# ---------------------------------------------------------------------------


def test_heavy_snapshot_computed_once_under_threads(monkeypatch):
    calls = {"n": 0}
    orig_vram = eng.detect_vram_gb

    def counting_vram():
        calls["n"] += 1
        # Widen the race window: without the lock, all 8 threads would be
        # inside the computation at once and each would count.
        threading.Event().wait(0.05)
        return orig_vram()

    monkeypatch.setattr(eng, "detect_vram_gb", counting_vram)
    monkeypatch.setattr(eng, "detect_gpu_name", lambda: None)
    settings = AppSettings()
    clear_engine_cache()

    barrier = threading.Barrier(8)
    results: list[dict] = []
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait(timeout=10)
            results.append(_heavy_snapshot(settings, None))
        except BaseException as exc:  # noqa: BLE001 -- collected and asserted
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not errors, f"threads raised: {errors}"
    assert len(results) == 8
    assert calls["n"] == 1, f"underlying computation ran {calls['n']} times, want 1"
    assert all(r is results[0] for r in results), "threads did not get the same snapshot"
    assert set(results[0]) >= {"vram_gb", "gpu_name", "modules", "ffmpeg"}


def test_clear_cache_racing_snapshot_is_safe(monkeypatch):
    monkeypatch.setattr(eng, "detect_vram_gb", lambda: None)
    monkeypatch.setattr(eng, "detect_gpu_name", lambda: None)
    settings = AppSettings()
    clear_engine_cache()
    stop = threading.Event()
    errors: list[BaseException] = []

    def clearer():
        while not stop.is_set():
            try:
                clear_engine_cache()
            except BaseException as exc:  # noqa: BLE001 -- collected
                errors.append(exc)

    def snapper():
        while not stop.is_set():
            try:
                data = _heavy_snapshot(settings, None)
                assert isinstance(data, dict) and "modules" in data
            except BaseException as exc:  # noqa: BLE001 -- collected
                errors.append(exc)

    threads = [threading.Thread(target=clearer) for _ in range(2)] + [
        threading.Thread(target=snapper) for _ in range(3)
    ]
    for t in threads:
        t.start()
    threading.Event().wait(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=15)
    assert not errors, f"racing clear/snapshot raised: {errors}"
    # Final state is consistent: after the race, a snapshot caches and the
    # next call returns the same cached object.
    clear_engine_cache()
    first = _heavy_snapshot(settings, None)
    second = _heavy_snapshot(settings, None)
    assert first is second
    assert set(first["modules"]) == set(ALL_LABELS)


# ---------------------------------------------------------------------------
# 5. engine_snapshot() level
# ---------------------------------------------------------------------------


def test_engine_snapshot_survives_heavy_import_errors(monkeypatch):
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name in HEAVY:
            raise ImportError(f"no module named {name!r} (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.setattr(eng, "detect_vram_gb", lambda: None)
    monkeypatch.setattr(eng, "detect_gpu_name", lambda: None)
    clear_engine_cache()
    # NOTE on scope: the failing __import__ mock cannot turn an *installed*
    # module False, because the new code never calls __import__ for the heavy
    # names -- installed ones (here: cv2, scenedetect, PySide6) are True via
    # find_spec alone, and modules already in sys.modules are True without
    # any import. What this test proves is the behaviour the mock *can*
    # affect: engine_snapshot() still returns normally, and every value
    # matches the find_spec/sys.modules semantics exactly as if __import__
    # had never been callable.
    expected = _expected_module_status()
    snap = engine_snapshot(AppSettings())  # must not raise
    mods = snap["modules"]
    assert set(mods) == set(ALL_LABELS)
    assert mods == expected
    assert mods["torch"] is False  # not installed here: would previously raise
    assert mods["cv2"] is True  # installed: True via find_spec, no import
    for key in PANEL_KEYS:
        assert key in mods


def test_engine_snapshot_modules_keys_present():
    clear_engine_cache()
    snap = engine_snapshot(AppSettings())
    mods = snap["modules"]
    assert set(mods) == set(ALL_LABELS)
    assert all(isinstance(v, bool) for v in mods.values())
    for key in PANEL_KEYS:
        assert key in mods
    assert snap["modules"] == _expected_module_status()
