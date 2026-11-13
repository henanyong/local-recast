"""Shared dialog helpers (P5/B3): scrollable content with a pinned button bar,
screen-size clamping, and the four button roles from ``styles.py``.

Built for the dialog sweep (group 4 reuses these): every dialog whose content
can outgrow a 1280x720 screen should wrap its content with :func:`scroll_wrap`
and size itself with :func:`cap_dialog_to_screen`.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

#: Pixels left between a clamped dialog and the screen edge (total, not per side:
#: on a 1280x720 screen an 860px-tall dialog becomes 720 - 24 = 696px tall).
SCREEN_MARGIN = 24

BUTTON_ROLES = ("primary", "secondary", "danger", "quiet")


def _available_geometry() -> QRect:
    """Primary screen's available geometry (taskbar / dock excluded).

    Kept as a tiny module-level helper so tests can monkeypatch a 1280x720
    screen: the offscreen platform reports 800x600 by default.
    """
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return QRect(0, 0, 1280, 720)
    return screen.availableGeometry()


def scroll_wrap(content: QWidget, buttons: QWidget | None = None) -> QWidget:
    """Wrap dialog content in a scroll area with an optional pinned button bar.

    Returns a widget whose QVBoxLayout holds, top to bottom: a QScrollArea
    (``widgetResizable=True``, holding ``content``) and, when given, ``buttons``
    fixed at the bottom -- it never scrolls away with the content.

    ``content`` must be a widget, not a layout: pass a container widget that
    already owns the layout.
    """
    if not isinstance(content, QWidget):
        raise TypeError("scroll_wrap content must be a QWidget, not a layout")
    if buttons is not None and not isinstance(buttons, QWidget):
        raise TypeError("scroll_wrap buttons must be a QWidget or None")
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea(wrapper)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(content)
    layout.addWidget(scroll, 1)
    if buttons is not None:
        layout.addWidget(buttons, 0)
    return wrapper


def cap_dialog_to_screen(dialog: QDialog, want_w: int, want_h: int) -> None:
    """Resize ``dialog`` to (``want_w``, ``want_h``) clamped inside the screen.

    The dialog never grows past the primary screen's available geometry minus
    :data:`SCREEN_MARGIN`: on a 1280x720 screen an 860px-tall request becomes
    696px. Smaller requests are kept as-is.
    """
    avail = _available_geometry()
    w = min(int(want_w), avail.width() - SCREEN_MARGIN)
    h = min(int(want_h), avail.height() - SCREEN_MARGIN)
    dialog.resize(max(1, w), max(1, h))


def apply_button_role(btn: QPushButton, role: str) -> None:
    """Assign one of the four stylesheet button roles to ``btn``.

    Sets the objectName so the matching ``QPushButton#<role>`` rule in
    ``styles.py`` applies. A danger button may never be promoted to primary
    (spec: 危险按钮永远不做主按钮); ``secondary`` also gets a 96px minimum
    width so it never collapses below the spec.
    """
    if role not in BUTTON_ROLES:
        raise ValueError(f"unknown button role: {role!r} (want one of {BUTTON_ROLES})")
    if role == "primary" and btn.objectName() == "danger":
        raise ValueError("a danger button must never become primary")
    btn.setObjectName(role)
    if role == "secondary":
        btn.setMinimumWidth(96)
