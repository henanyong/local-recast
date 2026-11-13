"""Tests for N36: one shared meta.json write lock (P4).

GUI thread and the 3D-generation worker thread must use the *same*
lock object (``recast.core.write_lock.META_WRITE_LOCK``): every GUI-side
meta write goes through ``characters.save_meta`` (which takes the lock),
and the queue worker holds it around its ``generate_fn`` call (the real
registration happens inside ``service.generate_model_3d``).

The fake ``generate_fn`` here is only a stand-in for the minutes-long
TripoSG run — the lock logic itself is exercised with real threads and
the real ``save_meta`` (real file writes, real ``load_meta``), never
with a fake store. A real TripoSG run on Windows is still on Daniel's
verification list; it is not claimed here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _dialog(generate_fn=None):
    from recast.desktop.model3d_queue import Model3dQueueDialog

    _app()
    return Model3dQueueDialog(service=None, generate_fn=generate_fn)


def _wait_until(pred, timeout: float = 10.0) -> bool:
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    return pred()


# -- 锁是同一把 ---------------------------------------------------------------


def test_queue_and_save_meta_use_the_same_lock_object():
    """The queue worker and save_meta must share one lock, not two."""
    from recast.core import write_lock
    from recast.desktop import model3d_queue

    assert model3d_queue.META_WRITE_LOCK is write_lock.META_WRITE_LOCK
    assert model3d_queue.meta_write_lock is write_lock.meta_write_lock


def test_save_meta_blocks_while_another_thread_holds_the_lock():
    """save_meta really takes the shared lock (not a private copy).

    Main thread holds the lock; a save from another thread must wait
    until it is released, then complete with the right content.
    """
    import tempfile

    from recast.core.characters import default_meta, load_meta, save_meta
    from recast.core.write_lock import META_WRITE_LOCK

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "温念"
        save_meta(folder, default_meta("温念"))
        done = threading.Event()

        def _saver():
            meta = load_meta(folder) or {}
            meta["voice"] = {**meta.get("voice", {}), "note": "gui-save"}
            save_meta(folder, meta)
            done.set()

        assert META_WRITE_LOCK.acquire(blocking=False)
        try:
            thread = threading.Thread(target=_saver, daemon=True)
            thread.start()
            assert not done.wait(timeout=0.5), "save_meta ran while the lock was held"
            assert thread.is_alive()
        finally:
            META_WRITE_LOCK.release()
        assert done.wait(timeout=5.0), "save_meta never finished after release"
        assert load_meta(folder)["voice"]["note"] == "gui-save"


def test_lock_is_reentrant_same_thread_save_inside_lock():
    """RLock: save_meta inside an already-held section must not deadlock.

    This is the real shape — the worker holds the lock around generate,
    and the registration path calls save_meta from the same thread.
    """
    import tempfile

    from recast.core.characters import default_meta, load_meta, save_meta
    from recast.core.write_lock import meta_write_lock

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "温念"
        save_meta(folder, default_meta("温念"))
        with meta_write_lock():
            with meta_write_lock():
                meta = load_meta(folder) or {}
                meta["model"] = {"source": "auto", "path": "model/model.glb"}
                save_meta(folder, meta)
        assert load_meta(folder)["model"]["path"] == "model/model.glb"


# -- 多线程并发 save_meta 同一人物 --------------------------------------------


def test_concurrent_save_meta_same_character_keeps_all_sections(tmp_path):
    """8 threads x 20 read-modify-writes, each its own section: nothing lost."""
    from recast.core.characters import default_meta, load_meta, save_meta

    folder = tmp_path / "温念"
    save_meta(folder, default_meta("温念"))

    def _writer(i: int):
        for _ in range(20):
            meta = load_meta(folder) or {}
            meta[f"sec_{i}"] = {"v": i}
            save_meta(folder, meta)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "writer thread hung (deadlock?)"
    meta = load_meta(folder)
    assert meta is not None
    for i in range(8):
        assert meta.get(f"sec_{i}") == {"v": i}, f"section sec_{i} lost: {meta!r}"
    assert meta.get("name") == "温念"
    assert meta.get("updated_at")


def test_concurrent_save_meta_same_key_never_corrupts_file(tmp_path):
    """8 threads hammering the *same* key: file stays valid JSON, no leftovers."""
    from recast.core.characters import default_meta, load_meta, save_meta

    folder = tmp_path / "温念"
    save_meta(folder, default_meta("温念"))

    def _writer(i: int):
        for _ in range(20):
            meta = load_meta(folder) or {}
            meta["voice"] = {**meta.get("voice", {}), "timbre": {"pitch": float(i)}}
            save_meta(folder, meta)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()
    raw = (folder / "meta.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)  # must not raise: never a torn file
    assert parsed["voice"]["timbre"]["pitch"] in {float(i) for i in range(8)}
    leftovers = [p for p in folder.iterdir() if p.name.startswith(".meta-")]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_concurrent_save_meta_different_characters_both_intact(tmp_path):
    """Two characters saved concurrently: both files complete and valid."""
    from recast.core.characters import default_meta, load_meta, save_meta

    folders = [tmp_path / name for name in ("温念", "陆寒峥")]
    for folder in folders:
        save_meta(folder, default_meta(folder.name))

    def _writer(folder: Path):
        for i in range(20):
            meta = load_meta(folder) or {}
            meta["voice"] = {**meta.get("voice", {}), "note": f"{folder.name}-{i}"}
            save_meta(folder, meta)

    threads = [threading.Thread(target=_writer, args=(f,)) for f in folders]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()
    for folder in folders:
        meta = load_meta(folder)
        assert meta["name"] == folder.name
        assert meta["voice"]["note"] == f"{folder.name}-19"


# -- worker 线程生成 vs GUI 线程读写同一人物 ----------------------------------


def test_worker_generate_and_gui_save_same_character_no_loss(tmp_path):
    """Worker registers a model while the GUI thread edits timbre: both survive.

    The fake generate_fn imitates the real registration tail
    (load -> set model -> save_meta, then a second save like
    set_assembly does) with real save_meta. The GUI side does real
    load -> set voice -> save_meta in its own thread at the same time,
    holding the shared lock across its whole read-modify-write span —
    that is the contract in ``recast.core.write_lock``: ``save_meta``
    alone serialises the write, but only a lock held across load -> save
    stops a stale full dict overwriting the other thread's section.
    """
    from recast.core.characters import default_meta, load_meta, save_meta
    from recast.core.write_lock import meta_write_lock

    folder = tmp_path / "温念"
    save_meta(folder, default_meta("温念"))

    def fake_generate(character, image_ref):
        time.sleep(0.3)  # generation in flight while the GUI writes
        meta = load_meta(folder) or {}
        meta["model"] = {"source": "auto", "format": "glb", "path": "model/model.glb"}
        save_meta(folder, meta)
        meta = load_meta(folder) or {}
        assembly = dict(meta.get("assembly") or {})
        assembly["body"] = "model/model.glb"
        meta["assembly"] = assembly
        save_meta(folder, meta)
        return {"path": "model/model.glb"}

    stop = threading.Event()

    def _gui_writer():
        while not stop.is_set():
            with meta_write_lock():
                meta = load_meta(folder) or {}
                meta["voice"] = {**meta.get("voice", {}), "timbre": {"pitch": 2.0}}
                save_meta(folder, meta)
            time.sleep(0.01)

    dialog = _dialog(generate_fn=fake_generate)
    writer = threading.Thread(target=_gui_writer, daemon=True)
    writer.start()
    try:
        dialog.enqueue("温念")
        assert dialog.wait_until_idle(timeout=10.0), dialog.queue_items()
    finally:
        stop.set()
        writer.join(timeout=5.0)
    assert dialog.queue_items()[0]["status"] == "完成"
    meta = load_meta(folder)
    assert meta is not None
    assert meta["model"]["path"] == "model/model.glb", meta
    assert meta["assembly"]["body"] == "model/model.glb", meta
    assert meta["voice"]["timbre"]["pitch"] == 2.0, meta
    dialog.close()


def test_gui_save_waits_while_worker_generates(tmp_path):
    """Worker holds the shared lock during generate: a GUI save waits, then lands.

    This pins the designed trade-off (see _worker_loop docstring): while a
    generation runs, a GUI-thread meta write blocks instead of racing it.
    """
    from recast.core.characters import default_meta, load_meta, save_meta

    folder = tmp_path / "温念"
    save_meta(folder, default_meta("温念"))
    started = threading.Event()
    release = threading.Event()

    def fake_generate(character, image_ref):
        started.set()
        assert release.wait(timeout=10), "generate was never released"
        return {"path": "model/model.glb"}

    dialog = _dialog(generate_fn=fake_generate)
    dialog.enqueue("温念")
    assert started.wait(timeout=10.0), dialog.queue_items()
    done = threading.Event()

    def _saver():
        meta = load_meta(folder) or {}
        meta["voice"] = {**meta.get("voice", {}), "note": "saved-during-generate"}
        save_meta(folder, meta)
        done.set()

    saver = threading.Thread(target=_saver, daemon=True)
    saver.start()
    assert not done.wait(timeout=0.5), "GUI save ran while the worker held the lock"
    release.set()
    assert done.wait(timeout=5.0), "GUI save never finished after generation"
    assert dialog.wait_until_idle(timeout=10.0), dialog.queue_items()
    assert load_meta(folder)["voice"]["note"] == "saved-during-generate"
    dialog.close()


def test_worker_generate_failure_releases_lock(tmp_path):
    """A failing generation must not leave the lock held (no deadlock after)."""
    from recast.core.characters import default_meta, load_meta, save_meta

    folder = tmp_path / "温念"
    save_meta(folder, default_meta("温念"))
    attempts = {"n": 0}

    def fake_generate(character, image_ref):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("TripoSG 生成失败（退出码 1）")
        meta = load_meta(folder) or {}
        meta["model"] = {"source": "auto", "path": "model/model.glb"}
        save_meta(folder, meta)
        return {"path": "model/model.glb"}

    dialog = _dialog(generate_fn=fake_generate)
    dialog.enqueue("温念")
    assert dialog.wait_until_idle(timeout=10.0), dialog.queue_items()
    assert dialog.queue_items()[0]["status"] == "失败"
    # GUI thread can save immediately afterwards — the lock was released.
    meta = load_meta(folder) or {}
    meta["voice"] = {**meta.get("voice", {}), "note": "after-failure"}
    save_meta(folder, meta)
    assert load_meta(folder)["voice"]["note"] == "after-failure"
    # Retry runs cleanly to completion.
    dialog.retry("温念")
    assert dialog.wait_until_idle(timeout=10.0), dialog.queue_items()
    assert dialog.queue_items()[0]["status"] == "完成"
    assert load_meta(folder)["model"]["path"] == "model/model.glb"
    dialog.close()


# -- 异常 / 边界：锁释放、损坏文件 --------------------------------------------


def test_save_meta_serialisation_error_releases_lock_and_keeps_old_file(tmp_path):
    """An unserialisable save raises, releases the lock, old file intact."""
    from recast.core.characters import default_meta, load_meta, save_meta

    folder = tmp_path / "温念"
    save_meta(folder, default_meta("温念"))
    before = (folder / "meta.json").read_text(encoding="utf-8")
    with pytest.raises(TypeError):
        save_meta(folder, {"bad": object()})
    assert (folder / "meta.json").read_text(encoding="utf-8") == before
    meta = load_meta(folder) or {}
    meta["voice"] = {**meta.get("voice", {}), "note": "still-works"}
    save_meta(folder, meta)  # must not deadlock after the failed save
    assert load_meta(folder)["voice"]["note"] == "still-works"
    assert [p for p in folder.iterdir() if p.name.startswith(".meta-")] == []


def test_save_meta_recovers_from_corrupt_on_disk_file(tmp_path):
    """Corrupt meta.json on disk: save falls back to the incoming dict, file valid."""
    from recast.core.characters import default_meta, load_meta, save_meta

    folder = tmp_path / "温念"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text("{not json,,,", encoding="utf-8")
    save_meta(folder, default_meta("温念"))
    meta = load_meta(folder)
    assert meta is not None and meta["name"] == "温念"
    json.loads((folder / "meta.json").read_text(encoding="utf-8"))
