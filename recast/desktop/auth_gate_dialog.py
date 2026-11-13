"""替换授权关口的共用对话框（N13/N18/N32）。

三个替换入口共用这一个关口函数，不要各写一份：

* 主窗口一键成片 / [按勾选阶段执行] / [只换脸]（经
  ``MainWindow._auth_gate_ok``，它只是这里的薄壳）；
* 队列面板 [从失败步续跑]（``queue_panel._retry_failed``）。

行为与主窗口原来的 ``_auth_gate_ok`` 一字不差：有参考脸但未授权的
人物先列出来，[去人物库补授权] / [继续，只替换已授权人物] 二选一；
零已授权时 [继续] 灰掉（点了也换不了，只会让任务在替换阶段被
parked，见 N18）；点 [继续] 后被跳过的人物记进持久的 ``auth_skips``
表（见 N11 改表 bundle），runner 重写任务 progress 也冲不掉。

调用方差异只通过两个回调传入，不在这里 import 主窗口或队列面板：

* ``on_go_library``——用户点 [去人物库补授权] 时调用（主窗口传
  ``_open_library``；队列面板传一个先关面板再开人物库的回调）；
* ``on_skipped``——用户点 [继续] 后调用，入参 ``(missing, drama_ids)``。
  主窗口传自己的 ``_record_skipped_characters``（还写任务 progress
  文本 + 日志）；不传时用这里的 ``record_auth_skips``，只写持久记录。

``_box_cls`` 只为测试留口：主窗口的薄壳传进主窗口模块自己的
``QMessageBox``，所以旧测试 ``monkeypatch("recast.desktop.main_window.
QMessageBox", _FakeBox)`` 不用改就能继续生效；队列面板的调用方
patch ``recast.desktop.auth_gate_dialog.QMessageBox``。
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtWidgets import QMessageBox, QWidget
from PySide6.QtWidgets import QMessageBox as _RealMessageBox  # noqa: F401 -- see _cap_message_box

from recast.core.entry_shared import check_replace_auth
from recast.desktop.dialog_utils import cap_dialog_to_screen

#: 关口对话框的标题：与主窗口原来的 _auth_gate_ok 完全一致。
GATE_TITLE = "一键成片停在审核关口"

#: 两个按钮的文案：与主窗口原来的 _auth_gate_ok 完全一致。
GO_LIBRARY_TEXT = "去人物库补授权"
CONTINUE_TEXT = "继续，只替换已授权人物"


def _cap_message_box(box: Any) -> None:
    """Clamp a real QMessageBox to the screen (P5/B3).

    Test doubles (``_box_cls`` fakes) have no ``resize`` and are not
    QDialogs, so they are left untouched. The check uses the real class
    captured as ``_RealMessageBox`` at import time — tests monkeypatch
    the module-level ``QMessageBox`` name with their fake, so an
    ``isinstance`` against that name would wrongly match the fake.
    For a real box the want-size is its content-driven sizeHint; the
    cap only shrinks it when it would outgrow the screen. Note
    QMessageBox re-lays out on show, so the size that really keeps this
    gate on a 1280x720 screen is the wrapping name list (verified with
    40 long names: 400x402); the cap is defense in depth for a
    pathological list.
    """
    if not isinstance(box, _RealMessageBox):
        return
    hint = box.sizeHint()
    cap_dialog_to_screen(box, hint.width(), hint.height())


def _name_in_drama(service: Any, name: str, drama_id: str) -> bool:
    """人名是否属于该剧目（N40 记账过滤）。

    用 ``Store.character_in_scope`` 的名字版映射（``source_task_id``
    或 ``selected_characters`` 快照）；store 没这方法或读失败时
    放行（fail-open），记账绝不因一次读失败丢记录。
    """
    store = getattr(service, "store", None)
    in_scope = getattr(store, "character_in_scope", None)
    if not callable(in_scope):
        return True
    try:
        return bool(in_scope(name, [drama_id]))
    except Exception:
        return True


def record_auth_skips(
    service: Any, missing: list[dict], drama_ids: list[str] | None
) -> None:
    """只写持久跳过记录（``auth_skips`` 表），不碰任务 progress 文本。

    与 ``MainWindow._record_skipped_characters`` 的持久部分同一写法：
    每分集任务记一条，剧目下没分集任务时按剧目级记一条，同一
    (drama, task, 人名) 去重由 ``store.record_auth_skip`` 负责。
    任何一步失败都跳过，不让记账拦住一次续跑。

    N40：只记属于本剧分集任务里实际出现的人物——每个剧目单独
    过滤，``missing`` 里链到别的剧目的人名不写进本剧；多剧同跑
    时（``drama_ids`` 有多个），一个人名只写进它所属的那个剧
    目，不会跨剧串记。归属未知的人名（库里没链路）照样记。
    日志里的提示文案仍按整份 ``missing`` 写一次，那是关口当时
    展示过的名单，不是分剧的记账口径。
    """
    names = "、".join(m["name"] for m in missing)
    note = f"{names}未授权，本次未替换"
    try:
        service.logs.append("WARNING", note + "（只替换已授权人物）。")
    except Exception:
        pass
    for did in drama_ids or []:
        in_scope = [m for m in missing if _name_in_drama(service, m["name"], did)]
        if not in_scope:
            continue
        try:
            tasks = service.store.tasks_for_drama(did)
        except Exception:
            tasks = []
        for task in tasks:
            for m in in_scope:
                try:
                    service.store.record_auth_skip(
                        did, m["name"], task_id=task["id"], reason="未授权"
                    )
                except Exception:
                    pass
        if not tasks:
            for m in in_scope:
                try:
                    service.store.record_auth_skip(
                        did, m["name"], reason="未授权"
                    )
                except Exception:
                    pass


def auth_gate_ok(
    parent: QWidget | None,
    service: Any,
    drama_ids: list[str] | None = None,
    *,
    prompt: bool = True,
    on_go_library: Callable[[], None] | None = None,
    on_skipped: Callable[[list[dict], list[str] | None], None] | None = None,
    _box_cls: Any = None,
) -> bool:
    """进替换之前的授权关口：有未授权人物时弹框确认，否则直接放行。

    返回 False 表示先别进替换；``prompt=False`` 时只检查不弹框（关口
    已提示过，等用户补完授权再来）。service 读不到时不挡路（返回
    True），替换阶段的 ``check_replace_ready`` 还会再拦一次。
    """
    try:
        result = check_replace_auth(service, drama_ids)
    except Exception:
        return True  # 查不到就不挡路，替换阶段的 check_replace_ready 还会拦。
    missing = result["unauthorized"]
    if not missing:
        return True
    if not prompt:
        return False
    names = "、".join(f"「{m['name']}」" for m in missing)
    authorized = result["authorized_count"]
    box_cls = _box_cls or QMessageBox
    box = box_cls(parent)
    box.setWindowTitle(GATE_TITLE)
    if authorized <= 0:
        box.setText(
            f"{len(missing)} 个人物有参考脸但未授权，替换时会跳过：{names}。\n"
            "没有已授权人物，继续也无法替换——请先去人物库补授权。"
        )
    else:
        box.setText(
            f"{len(missing)} 个人物有参考脸但未授权，替换时会跳过：{names}。\n"
            "去人物库补授权，或继续（只替换已授权的人物）。"
        )
    go_library = box.addButton(GO_LIBRARY_TEXT, box_cls.ButtonRole.AcceptRole)
    continue_btn = box.addButton(CONTINUE_TEXT, box_cls.ButtonRole.RejectRole)
    if authorized <= 0:
        # QMessageBox.addButton 返回 QPushButton；测试替身可能没有
        # setEnabled，有才灰掉，没也别炸。
        setter = getattr(continue_btn, "setEnabled", None)
        if callable(setter):
            setter(False)
    _cap_message_box(box)
    box.exec()
    if box.clickedButton() is go_library:
        if on_go_library is not None:
            on_go_library()
        return False
    if authorized <= 0:
        return False
    if on_skipped is not None:
        on_skipped(missing, drama_ids)
    else:
        record_auth_skips(service, missing, drama_ids)
    return True
