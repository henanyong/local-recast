"""P5/B3 button roles: the four role selectors in styles.py and the
``apply_button_role`` helper in dialog_utils.py (spec §5.2).

Roles: primary 36px / secondary 32px + min-width 96 / danger 32px /
quiet (transparent + #3c4560 border) 32px. The base QPushButton rule is
deliberately untouched -- no global min-height -- so layouts that never
opted into a role do not shift.
"""

from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _block(selector: str) -> str:
    from recast.desktop.styles import THEME

    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", THEME)
    assert match is not None, f"no stylesheet block for {selector}"
    return match.group(1)


# ----- styles.py: role selectors -------------------------------------------


def test_primary_keeps_color_and_gains_36px_min_height():
    block = _block("QPushButton#primary")
    assert "#3a6df0" in block
    assert "font-weight: 600" in block
    assert "min-height: 36px" in block


def test_secondary_role_defined():
    block = _block("QPushButton#secondary")
    assert "#2a3144" in block
    assert "min-height: 32px" in block
    assert "min-width: 96px" in block


def test_danger_keeps_color_and_gains_32px_min_height():
    block = _block("QPushButton#danger")
    assert "#5a2a2a" in block
    assert "min-height: 32px" in block


def test_quiet_role_is_transparent_with_border():
    block = _block("QPushButton#quiet")
    assert "transparent" in block
    assert "#3c4560" in block
    assert "min-height: 32px" in block


def test_base_pushbutton_rule_untouched():
    # No global min-height: only buttons that opt into a role grow taller,
    # so untouched dialogs keep their current layout.
    assert "min-height" not in _block("QPushButton")
    assert "min-width" not in _block("QPushButton")


def test_roles_take_effect_live():
    from PySide6.QtWidgets import QPushButton, QWidget

    from recast.desktop.styles import THEME

    _app()
    host = QWidget()
    host.setStyleSheet(THEME)
    heights = {}
    for role, want in (("primary", 36), ("secondary", 32), ("danger", 32), ("quiet", 32)):
        btn = QPushButton("x", host)
        btn.setObjectName(role)
        heights[role] = btn.minimumSizeHint().height()
        assert heights[role] >= want, (role, heights[role])
    secondary = QPushButton("x", host)
    secondary.setObjectName("secondary")
    assert secondary.minimumSizeHint().width() >= 96
    host.close()


# ----- dialog_utils.apply_button_role --------------------------------------


@pytest.mark.parametrize("role", ["primary", "secondary", "danger", "quiet"])
def test_apply_button_role_sets_object_name(role):
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.dialog_utils import apply_button_role

    _app()
    btn = QPushButton("x")
    apply_button_role(btn, role)
    assert btn.objectName() == role


def test_apply_button_role_secondary_min_width_96():
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.dialog_utils import apply_button_role

    _app()
    btn = QPushButton("x")
    apply_button_role(btn, "secondary")
    assert btn.minimumWidth() == 96


def test_apply_button_role_danger_cannot_become_primary():
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.dialog_utils import apply_button_role

    _app()
    btn = QPushButton("删除")
    apply_button_role(btn, "danger")
    with pytest.raises(ValueError):
        apply_button_role(btn, "primary")
    assert btn.objectName() == "danger"


def test_apply_button_role_unknown_role_raises():
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.dialog_utils import apply_button_role

    _app()
    with pytest.raises(ValueError):
        apply_button_role(QPushButton("x"), "mega")


def test_apply_button_role_other_reassignments_allowed():
    # Only danger -> primary is forbidden; e.g. a secondary button may be
    # promoted, and a danger button may be demoted to quiet.
    from PySide6.QtWidgets import QPushButton

    from recast.desktop.dialog_utils import apply_button_role

    _app()
    btn = QPushButton("x")
    apply_button_role(btn, "secondary")
    apply_button_role(btn, "primary")
    assert btn.objectName() == "primary"
    apply_button_role(btn, "danger")
    apply_button_role(btn, "quiet")
    assert btn.objectName() == "quiet"
