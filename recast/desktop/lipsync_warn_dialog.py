"""合成前唇同步警告的对话框（设计 4.3①）。

只做时长拦截，不做真唇同步：配音比镜头长时，在合成开始前把超标的
台词行与超出量列出来，让用户二选一——[继续合成] 或 [取消，回去改台词]。

调用链（与授权关口 ``auth_gate_dialog.py`` 同一模式）：

* ``run_episode`` 合成前调 ``gate_compose_on_lipsync``（``recast/core/
  runner.py``），它收到警告后调注入的 ``lipsync_confirm`` 回调；
* 主窗口在开始前把 ``ask_lipsync_on_gui_thread`` 挂在共用的 settings
  对象上（``MainWindow._bind_lipsync_confirm``），``dispatch_job``
  再把它传进 ``run_episode``。

线程规则：任务跑在调度器的工作线程里，QMessageBox 绝不能在工作
线程里 exec——``ask_lipsync_on_gui_thread`` 在非 GUI 线程时用
``QMetaObject.invokeMethod`` + ``BlockingQueuedConnection`` 把弹框
发到 GUI 线程执行并等回答；已在 GUI 线程时直接弹。绝不在定时器里
exec（队列面板 2 秒刷新那类定时器回调里不调这里）。

``_box_cls`` 只为测试留口：测试传假框，不 exec 真模态框。
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, Slot
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget
from PySide6.QtWidgets import QMessageBox as _RealMessageBox  # noqa: F401 -- see _cap_message_box

from recast.desktop.dialog_utils import cap_dialog_to_screen

log = logging.getLogger("recast")

#: 对话框标题。
WARN_TITLE = "合成前提醒：配音比镜头长"

#: 两个按钮的文案。
CONTINUE_TEXT = "继续合成"
CANCEL_TEXT = "取消，回去改台词"

#: 一次最多列出的超标行数，多了只写总数，不把框撑爆。
MAX_LISTED_LINES = 8


def _cap_message_box(box: Any) -> None:
    """Clamp a real QMessageBox to the screen (P5/B3).

    Same guard as ``auth_gate_dialog._cap_message_box``: test doubles
    (``_box_cls`` fakes) have no ``resize`` and are not QDialogs, so
    they are left untouched. The check uses the real class captured as
    ``_RealMessageBox`` at import time — the module-level ``QMessageBox``
    name is monkeypatched by tests, so an ``isinstance`` against that
    name would wrongly match a fake. The size that really keeps this
    warning on a 1280x720 screen is the 8-line cap
    (``MAX_LISTED_LINES``; verified with 30 warnings: 400x437); the
    cap is defense in depth.
    """
    if not isinstance(box, _RealMessageBox):
        return
    hint = box.sizeHint()
    cap_dialog_to_screen(box, hint.width(), hint.height())


def format_warning_lines(warnings: list[Any]) -> str:
    """超标行写成一行一条的文本：镜头号 + 台词 + 配音/镜头/超出秒数。"""
    rows = []
    for w in warnings[:MAX_LISTED_LINES]:
        if isinstance(w, dict):
            code = w.get("shot_code") or "（未编号镜头）"
            text = str(w.get("text") or "")
            audio = float(w.get("audio_sec") or 0)
            slot = float(w.get("slot_sec") or 0)
            over = float(w.get("over_sec") or 0)
        else:
            code = getattr(w, "shot_code", "") or "（未编号镜头）"
            text = str(getattr(w, "text", "") or "")
            audio = float(getattr(w, "audio_sec", 0) or 0)
            slot = float(getattr(w, "slot_sec", 0) or 0)
            over = float(getattr(w, "over_sec", 0) or 0)
        shown = text if len(text) <= 18 else text[:18] + "…"
        rows.append(
            f"{code}「{shown}」配音 {audio:.2f} 秒，镜头 {slot:.2f} 秒，超出 {over:.2f} 秒"
        )
    if len(warnings) > MAX_LISTED_LINES:
        rows.append(f"…等 {len(warnings)} 行超标")
    return "\n".join(rows)


def lipsync_warn_dialog(
    parent: QWidget | None,
    warnings: list[Any],
    *,
    _box_cls: Any = None,
) -> bool:
    """列出超标台词行与超出量，等用户二选一。

    返回 True = [继续合成]，False = [取消，回去改台词]。
    必须在 GUI 线程调用；工作线程请走 ``ask_lipsync_on_gui_thread``。
    """
    if not warnings:
        return True
    box_cls = _box_cls or QMessageBox
    box = box_cls(parent)
    box.setWindowTitle(WARN_TITLE)
    box.setText(
        f"{len(warnings)} 行台词的配音比镜头长，合成后这几句会盖住下一镜的开头：\n"
        f"{format_warning_lines(warnings)}\n\n"
        "可以先回去改短台词或拆到下一镜，也可以直接继续合成。\n"
        "注意：这只是时长提醒，不检查口型对不对——真唇同步还没做。"
    )
    continue_btn = box.addButton(CONTINUE_TEXT, box_cls.ButtonRole.AcceptRole)
    cancel_btn = box.addButton(CANCEL_TEXT, box_cls.ButtonRole.RejectRole)
    # 默认停在取消上：回车误触也不该直接放行一次超长的合成。
    setter = getattr(box, "setDefaultButton", None)
    if callable(setter):
        setter(cancel_btn)
    _cap_message_box(box)
    box.exec()
    return box.clickedButton() is continue_btn


class _LipSyncAsker(QObject):
    """把警告框发到 GUI 线程执行的搬运工。

    必须是无父对象的 QObject：它在工作线程里创建，带 GUI 线程父对象
    的 QObject 不允许跨线程创建。建好后 ``moveToThread`` 到 GUI 线程，
    再用 BlockingQueuedConnection 调 ``ask``。
    """

    def __init__(self, parent: QWidget | None, warnings: list[Any]) -> None:
        super().__init__()
        self._parent = parent
        self._warnings = warnings
        self.answer = True

    @Slot()
    def ask(self) -> None:
        self.answer = lipsync_warn_dialog(self._parent, self._warnings)


def ask_lipsync_on_gui_thread(
    parent: QWidget | None,
    warnings: list[Any],
) -> bool:
    """给 runner 的 ``lipsync_confirm`` 回调：在 GUI 线程弹警告框并等回答。

    已在 GUI 线程时直接弹；工作线程时发到 GUI 线程并阻塞等结果。
    没有 QApplication（无界面/纯测试环境）时没人可问，记一条日志后
    按“继续”处理——与无回调时“只记警告不拦路”一致。
    """
    app = QApplication.instance()
    if app is None:
        log.warning("没有 QApplication，唇同步警告无法弹框，按继续处理：%s 行超标。", len(warnings))
        return True
    if QThread.currentThread() is app.thread():
        return lipsync_warn_dialog(parent, warnings)
    asker = _LipSyncAsker(parent, warnings)
    asker.moveToThread(app.thread())
    QMetaObject.invokeMethod(asker, "ask", Qt.BlockingQueuedConnection)
    return asker.answer
