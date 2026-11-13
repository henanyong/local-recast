"""P5/B3 batch, main window: engine panel no truncation at 1280x720,
drama-table "成片" column button fully visible, exactly one primary button.

Only synthetic data + stub service calls: no real video, no TTS engines, no
network. Follows the fixture pattern of tests/test_p3_main_window.py.
"""

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


def _main(svc):
    from recast.desktop.main_window import MainWindow

    _app()
    window = MainWindow(svc)
    window.timer.stop()
    # 切断 __init__ 那次异步初始刷新的回路：工作线程算完快照后经
    # _snap_signal 回主线程重绘，show() 时 processEvents 会把它泵进来、
    # 盖掉测试里同步刷的伪造快照。测试直接调 _paint，不需要这条回路。
    try:
        window._snap_signal.ready.disconnect()
    except (TypeError, RuntimeError):
        pass
    return window


def _paint_main(window, svc):
    """Drive one synchronous paint with a real snapshot (no event loop)."""
    window._paint(svc.snapshot(), full=True)


def _show_1280(window):
    """Show at Daniel's 1280x720 screen size and let layouts settle."""
    from PySide6.QtWidgets import QApplication

    window.resize(1280, 720)
    window.show()
    QApplication.processEvents()


def _add_drama(svc, name: str = "剧", **extra) -> dict:
    fields = {
        "origin_name": name, "root_path": f"/tmp/{name}", "episode_count": 1,
        "entry_type": "import",
    }
    fields.update(extra)
    return svc.store.insert("dramas", **fields)


def _btn(window, text: str):
    """Find a QPushButton child by exact text (scan / template buttons are locals)."""
    from PySide6.QtWidgets import QPushButton

    for btn in window.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    raise AssertionError(f"no button with text {text!r}")


def _assert_label_wrapped_fits(label):
    """A word-wrapped label shows every visual line within its width and
    has enough height for the wrapped text (no horizontal truncation)."""
    from PySide6.QtGui import QTextLayout

    assert label.wordWrap(), "label must wrap long text instead of truncating it"
    width = label.width()
    assert width > 0, "label must have a real width after show()"
    assert label.height() >= label.heightForWidth(width), (
        f"label height {label.height()} < wrapped height "
        f"{label.heightForWidth(width)} for width {width}"
    )
    layout = QTextLayout(label.text(), label.font())
    layout.beginLayout()
    try:
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            assert line.naturalTextWidth() <= width + 1, (
                f"wrapped line {line.text()!r} ({line.naturalTextWidth():.0f}px) "
                f"wider than label ({width}px)"
            )
    finally:
        layout.endLayout()


# ----- 第 1 项：引擎状态面板完整显示 -----


def test_top_bar_two_rows_long_hardware_wraps(svc):
    """Long hardware text wraps onto two lines at 1280x720; full text in tooltip."""
    window = _main(svc)
    # 先 show：排空 __init__ 里那次异步初始刷新，再刷伪造快照，
    # 否则 processEvents 会把真实快照盖回来。
    _show_1280(window)
    snap = svc.snapshot()
    engine = dict(snap.get("engine") or {})
    engine.update({
        "state": "空闲",
        "hardware_label": "高档（独立 GPU）",
        "gpu_name": "NVIDIA GeForce RTX 4060 Ti 8GB 超长显卡名称用来测试换行显示效果如何",
        "vram_gb": 8.0,
    })
    snap["engine"] = engine
    window._paint(snap, full=True)
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    hw_text = window.hw_label.text()
    assert "RTX 4060 Ti" in hw_text
    assert window.hw_label.toolTip() == hw_text, "完整硬件文本必须进 tooltip"
    assert window.engine_label.toolTip() == window.engine_label.text()
    _assert_label_wrapped_fits(window.hw_label)
    _assert_label_wrapped_fits(window.engine_label)
    window.close()


def test_top_bar_short_text_fits_single_line(svc):
    """Short engine/hardware text fits on one line: fontMetrics width <= label width."""
    window = _main(svc)
    _paint_main(window, svc)
    _show_1280(window)
    for label in (window.engine_label, window.hw_label):
        text = label.text()
        assert text, "label must have text after paint"
        fm_width = label.fontMetrics().horizontalAdvance(text)
        assert fm_width <= label.width(), (
            f"{text!r} needs {fm_width}px but label is {label.width()}px"
        )
        assert label.toolTip() == text
    window.close()


def test_detect_label_wraps_and_tooltip_holds_full_text(svc):
    """Right-rail detect label wraps (wordWrap) with full text in tooltip."""
    window = _main(svc)
    _show_1280(window)  # 先排空 __init__ 的异步初始刷新（见上个测试的注释）
    snap = svc.snapshot()
    engine = dict(snap.get("engine") or {})
    engine.update({
        "ffmpeg": {"ok": True},
        "modules": {"torch": True, "faster_whisper": True, "insightface": True, "rapidocr": True},
        "models": [],
        "disk_free_gb": 123.4,
    })
    snap["engine"] = engine
    window._paint(snap, full=True)
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    text = window.detect_label.text()
    assert "ffmpeg" in text and "PyTorch" in text and "磁盘剩余" in text
    assert window.detect_label.toolTip() == text
    assert window.detect_label.width() > 0
    assert window.detect_label.height() >= window.detect_label.heightForWidth(window.detect_label.width()), (
        "detect_label height must fit its wrapped text"
    )
    _assert_label_wrapped_fits(window.detect_label)
    window.close()


def test_detect_label_empty_engine_does_not_crash(svc):
    """Boundary: empty engine snapshot still paints labels with matching tooltips."""
    window = _main(svc)
    _show_1280(window)  # 先排空 __init__ 的异步初始刷新（见上个测试的注释）
    snap = svc.snapshot()
    snap["engine"] = {}
    window._paint(snap, full=True)
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    assert window.engine_label.toolTip() == window.engine_label.text()
    assert window.hw_label.toolTip() == window.hw_label.text()
    assert window.detect_label.toolTip() == window.detect_label.text()
    window.close()


def test_right_rail_has_minimum_width(svc):
    """Right rail keeps a minimum width so the engine panel is not squeezed."""
    window = _main(svc)
    _paint_main(window, svc)
    _show_1280(window)
    assert window._right_splitter.minimumWidth() >= 300
    assert window._right_splitter.width() >= 300
    window.close()


# ----- 第 2 项：剧目表「成片」列 -----


def test_output_column_button_fully_visible(svc):
    """A drama with output has a fully visible [打开] button at 1280x720."""
    from PySide6.QtWidgets import QPushButton

    drama = _add_drama(svc, "一部很长的剧名用来撑宽表格", output_dir="/tmp/out/剧")
    window = _main(svc)
    _paint_main(window, svc)
    _show_1280(window)
    assert window.table.rowCount() == 1
    btn = window.table.cellWidget(0, 6)
    assert isinstance(btn, QPushButton)
    assert btn.text() == "打开"
    assert btn.isEnabled()
    text_w = btn.fontMetrics().horizontalAdvance("打开")
    assert window.table.columnWidth(6) >= text_w + 24, (
        f"成片列 {window.table.columnWidth(6)}px < 按钮文字 {text_w}px + 内边距"
    )
    assert btn.width() >= btn.sizeHint().width(), (
        f"按钮宽 {btn.width()}px < 完整显示所需 {btn.sizeHint().width()}px（会变「…」）"
    )
    assert "…" not in btn.text()
    # 表头「成片」两字也要完整显示。
    header = window.table.horizontalHeader()
    head_w = header.fontMetrics().horizontalAdvance("成片")
    assert window.table.columnWidth(6) >= head_w + 16
    window.close()


def test_output_column_button_disabled_but_not_truncated(svc):
    """Boundary: drama without output -> button disabled, still fully visible."""
    from PySide6.QtWidgets import QPushButton

    _add_drama(svc, "剧")
    window = _main(svc)
    _paint_main(window, svc)
    _show_1280(window)
    btn = window.table.cellWidget(0, 6)
    assert isinstance(btn, QPushButton)
    assert not btn.isEnabled()
    assert btn.width() >= btn.sizeHint().width(), "灰掉的按钮也不得截断成「…」"
    window.close()


def test_other_columns_headers_fit_and_long_cells_have_tooltips(svc):
    """Other columns: header text fits at 1280; long cell text reachable via tooltip."""
    _add_drama(
        svc,
        "一个非常非常长的原始剧名用来测试省略",
        new_name="一个非常非常长的新剧名用来测试省略",
        episode_count=12, status="已完成", progress="12/12 已完成",
        created_at="2026-11-13 09:24:26", output_dir="/tmp/out/剧",
    )
    window = _main(svc)
    _paint_main(window, svc)
    _show_1280(window)
    header = window.table.horizontalHeader()
    fm = header.fontMetrics()
    from recast.desktop.main_window import COLS

    for col, name in enumerate(COLS):
        assert window.table.columnWidth(col) >= fm.horizontalAdvance(name) + 16, (
            f"第 {col} 列「{name}」表头被截断：列宽 {window.table.columnWidth(col)}px"
        )
    # Stretch 列（原名 / 新名称 / 进度）与创建时间列：完整文本在 tooltip 里。
    assert window.table.item(0, 1).toolTip() == "一个非常非常长的原始剧名用来测试省略"
    assert window.table.item(0, 2).toolTip() == "一个非常非常长的新剧名用来测试省略"
    assert window.table.item(0, 5).toolTip() == "12/12 已完成"
    assert window.table.item(0, 7).toolTip() == "2026-11-13 09:24:26"
    window.close()


# ----- 第 3 项：主按钮唯一 -----


def test_only_oneclick_is_primary(svc):
    """Main window has exactly one primary button: [一键成片]."""
    from PySide6.QtWidgets import QPushButton

    window = _main(svc)
    _paint_main(window, svc)  # _paint rewrites 下一步 text; roles must survive it
    primaries = [b for b in window.findChildren(QPushButton) if b.objectName() == "primary"]
    assert [b.text() for b in primaries] == ["一键成片"], (
        f"primary 按钮应只有 [一键成片]，实际是 {[b.text() for b in primaries]}"
    )
    assert window.oneclick_btn.objectName() == "primary"
    assert window.oneclick_btn.minimumHeight() >= 36
    window.close()


def test_demoted_buttons_are_secondary_with_size(svc):
    """[扫描][下一步][人物库] are secondary, min 96x32."""
    window = _main(svc)
    _paint_main(window, svc)
    from PySide6.QtWidgets import QPushButton

    scan = _btn(window, "扫描")
    assert scan.objectName() == "secondary"
    assert window.next_btn.objectName() == "secondary"
    assert window.library_btn.objectName() == "secondary"
    for btn in (scan, window.next_btn, window.library_btn):
        assert btn.minimumWidth() >= 96, f"{btn.text()!r} 最小宽 {btn.minimumWidth()} < 96"
        assert btn.minimumHeight() >= 32, f"{btn.text()!r} 最小高 {btn.minimumHeight()} < 32"
    # _paint 后 [下一步] 文案变长，角色不变。
    assert window.next_btn.text().startswith("下一步")
    assert window.next_btn.objectName() == "secondary"
    window.close()


def test_action_bar_secondary_and_danger_roles(svc):
    """Action-bar secondary buttons + danger [停止当前任务] (same size, never primary)."""
    window = _main(svc)
    _paint_main(window, svc)
    from PySide6.QtWidgets import QPushButton

    for btn in (window.start_btn, window.reface_btn, window.rerender_btn, window.gen_btn):
        assert btn.objectName() == "secondary", f"{btn.text()!r} 应为 secondary"
        assert btn.minimumWidth() >= 96 and btn.minimumHeight() >= 32
    tpl = _btn(window, "模板编辑器")
    assert tpl.objectName() == "secondary"
    assert tpl.minimumWidth() >= 96 and tpl.minimumHeight() >= 32
    stop = _btn(window, "停止当前任务")
    assert stop.objectName() == "danger", "停止按钮保持 danger"
    assert stop.objectName() != "primary", "危险按钮永不做主按钮"
    assert stop.minimumWidth() >= 96 and stop.minimumHeight() >= 32, "危险按钮与次按钮同尺寸"
    window.close()
