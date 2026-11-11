from __future__ import annotations

from recast.core.constants import (
    ST_DONE,
    ST_FAILED,
    ST_NEED_ASSETS,
    ST_PENDING,
    ST_REVIEW,
    ST_RUNNING,
)
from recast.core.state_machine import apply, classify_error, format_missing
from recast.runtime import CancelledError


def test_missing_materials_names_files() -> None:
    t = apply(ST_PENDING, "materials_missing", missing=["5.png", "6.png"])
    assert t.state == ST_NEED_ASSETS
    assert t.progress == "缺少 5.png、6.png"


def test_materials_ok_from_need_assets() -> None:
    t = apply(ST_NEED_ASSETS, "materials_ok")
    assert t.state == ST_PENDING
    assert "资料已补齐" in t.progress


def test_stop_returns_pending_not_failed() -> None:
    t = apply(ST_RUNNING, "stop")
    assert t.state == ST_PENDING
    assert t.state != ST_FAILED


def test_step_fail_retries_then_review() -> None:
    t1 = apply(ST_RUNNING, "step_fail", reason="shot 3 ocr failed", retry_count=0, step="切镜3")
    assert t1.state == ST_RUNNING
    assert t1.retry_count == 1
    t2 = apply(ST_RUNNING, "step_fail", reason="again", retry_count=1, step="切镜3")
    assert t2.state == ST_RUNNING
    assert t2.retry_count == 2
    t3 = apply(ST_RUNNING, "step_fail", reason="again", retry_count=2, step="切镜3")
    assert t3.state == ST_REVIEW
    assert t3.state != ST_FAILED


def test_unrecoverable_is_failed() -> None:
    t = apply(ST_RUNNING, "unrecoverable", reason="找不到 ffmpeg")
    assert t.state == ST_FAILED
    assert "ffmpeg" in t.progress


def test_classify_error() -> None:
    assert classify_error(CancelledError("stop")) == "stop"
    err = OSError("No space left on device")
    err.errno = 28
    assert classify_error(err) == "fatal"
    assert classify_error(RuntimeError("找不到模型 inswapper_128.onnx")) == "fatal"
    assert classify_error(RuntimeError("偶发解码失败")) == "retry"


def test_all_done() -> None:
    t = apply(ST_RUNNING, "all_done")
    assert t.state == ST_DONE
