"""Tests for the 3D generation queue (recast/desktop/model3d_queue.py).

No real TripoSG run here: a fake ``generate_fn`` stands in for the
minutes-long generation, so the serial/failed/retry behaviour can be
verified in milliseconds. The real registration path (generate_fn ->
service.generate_model_3d -> generate_model_glb -> model/ + meta.json)
is already covered by tests/test_generate3d.py.
"""

from __future__ import annotations

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


def _row_texts(dialog):
    rows = []
    for r in range(dialog.table.rowCount()):
        rows.append(
            tuple(
                (dialog.table.item(r, c).text() if dialog.table.item(r, c) else "")
                for c in range(4)
            )
        )
    return rows


def test_empty_queue_shows_empty_state():
    dialog = _dialog(generate_fn=lambda c, i: {"path": "model/model.glb"})
    assert dialog.table.rowCount() == 0
    assert dialog.empty_label.isVisibleTo(dialog) or not dialog.empty_label.isHidden()
    assert "队列为空" in dialog.empty_label.text()
    dialog.refresh()  # must not crash on an empty queue
    assert dialog.table.rowCount() == 0
    assert dialog.queue_items() == []
    dialog.close()


def test_enqueue_two_runs_serially():
    """Second person must not start before the first one finishes (8GB)."""
    started: list[str] = []
    finished: list[str] = []
    release_first = threading.Event()

    def fake_generate(character, image_ref):
        started.append(character)
        if character == "温念":
            assert release_first.wait(timeout=10), "first generation was never released"
        finished.append(character)
        return {"path": f"model/{character}.glb"}

    dialog = _dialog(generate_fn=fake_generate)
    dialog.enqueue("温念")
    dialog.enqueue("陆寒峥")
    # First one starts, second one stays queued while the first is running.
    # (Wait for the fake backend to actually start, not just for the status
    # flip — the worker sets 生成中 just before calling generate_fn.)
    assert _wait_until(lambda: started == ["温念"], timeout=10.0), dialog.queue_items()
    assert dialog.queue_items()[0]["status"] == "生成中"
    assert dialog.queue_items()[1]["status"] == "排队"
    assert started == ["温念"], started
    dialog.refresh()
    rows = _row_texts(dialog)
    assert rows[0][1] == "生成中"
    assert rows[1][1] == "排队"
    release_first.set()
    assert dialog.wait_until_idle(timeout=10.0), dialog.queue_items()
    assert started == ["温念", "陆寒峥"], started
    assert finished == ["温念", "陆寒峥"], finished
    assert [it["status"] for it in dialog.queue_items()] == ["完成", "完成"]
    dialog.refresh()
    assert [r[1] for r in _row_texts(dialog)] == ["完成", "完成"]
    dialog.close()


def test_failed_row_shows_error_and_retry_succeeds():
    attempts: dict[str, int] = {}

    def fake_generate(character, image_ref):
        attempts[character] = attempts.get(character, 0) + 1
        if attempts[character] == 1:
            raise RuntimeError("TripoSG 生成失败（退出码 1）：CUDA out of memory")
        return {"path": "model/model.glb"}

    dialog = _dialog(generate_fn=fake_generate)
    dialog.enqueue("温念")
    assert dialog.wait_until_idle(timeout=10.0), dialog.queue_items()
    dialog.refresh()
    item = dialog.queue_items()[0]
    assert item["status"] == "失败"
    assert "CUDA out of memory" in item["error"]
    rows = _row_texts(dialog)
    assert rows[0][1] == "失败"
    assert "CUDA out of memory" in rows[0][3]
    # The failed row has a [重试] button; clicking it re-queues and succeeds.
    from PySide6.QtWidgets import QPushButton

    retry_btn = dialog.table.cellWidget(0, 4)
    assert isinstance(retry_btn, QPushButton)
    assert retry_btn.text() == "重试"
    retry_btn.click()
    assert dialog.wait_until_idle(timeout=10.0), dialog.queue_items()
    dialog.refresh()
    assert dialog.queue_items()[0]["status"] == "完成"
    assert dialog.queue_items()[0]["error"] == ""
    assert attempts["温念"] == 2
    assert dialog.table.cellWidget(0, 4) is None  # no retry button once done
    dialog.close()


def test_refresh_reflects_status_changes():
    release = threading.Event()

    def fake_generate(character, image_ref):
        assert release.wait(timeout=10)
        return {"path": "model/model.glb"}

    dialog = _dialog(generate_fn=fake_generate)
    dialog.enqueue("温念")
    assert _wait_until(lambda: dialog.queue_items()[0]["status"] == "生成中")
    dialog.refresh()
    assert _row_texts(dialog)[0][1] == "生成中"
    assert "生成中" in dialog.summary_label.text()
    release.set()
    assert dialog.wait_until_idle(timeout=10.0)
    dialog.refresh()
    assert _row_texts(dialog)[0][1] == "完成"
    assert "完成 1" in dialog.summary_label.text()
    dialog.close()


def test_enqueue_same_character_twice_does_not_duplicate():
    started: list[str] = []
    release = threading.Event()

    def fake_generate(character, image_ref):
        started.append(character)
        assert release.wait(timeout=10)
        return {"path": "model/model.glb"}

    dialog = _dialog(generate_fn=fake_generate)
    first = dialog.enqueue("温念")
    second = dialog.enqueue("温念")
    assert second is first
    assert len(dialog.queue_items()) == 1
    assert dialog.table.rowCount() == 1
    release.set()
    assert dialog.wait_until_idle(timeout=10.0)
    assert started == ["温念"]
    dialog.close()


def test_enqueue_empty_character_rejected_and_retry_non_failed_is_noop():
    dialog = _dialog(generate_fn=lambda c, i: {"path": "model/model.glb"})
    with pytest.raises(ValueError):
        dialog.enqueue("")
    with pytest.raises(ValueError):
        dialog.enqueue("   ")
    assert dialog.retry("不存在的人物") is None
    dialog.enqueue("温念")
    assert dialog.retry("温念") is None  # queued, not failed: no-op
    assert dialog.wait_until_idle(timeout=10.0)
    assert dialog.retry("温念") is None  # done, not failed: no-op
    dialog.close()
