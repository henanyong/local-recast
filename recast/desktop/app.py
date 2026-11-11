"""Launch the PySide6 desktop workbench."""

from __future__ import annotations

import sys


def launch() -> int:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - import depends on the desktop extra
        print(
            "无法启动桌面工作台：未安装 PySide6。\n"
            "请先激活虚拟环境并执行:  pip install PySide6\n"
            "也可以改用网页工作台:  python -m recast ui-web\n"
            f"原始错误: {exc}",
            file=sys.stderr,
        )
        return 1

    from recast.core.service import RecastService
    from recast.desktop.main_window import MainWindow
    from recast.desktop.styles import THEME

    # High-DPI is default on recent Qt; keep it safe on older bindings.
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("本地再铸")
    app.setOrganizationName("local-recast")
    app.setStyleSheet(THEME)

    service = RecastService()
    window = MainWindow(service)
    window.show()
    return app.exec()
