"""Dark original workbench stylesheet. Not a clone of any third-party product."""

THEME = """
QWidget {
    background-color: #12141a;
    color: #e8eaef;
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Segoe UI", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog, QWidget#root {
    background-color: #12141a;
}
QFrame#topBar {
    background-color: #1a1d27;
    border-bottom: 1px solid #2a2e3a;
}
QFrame#dramaBar, QFrame#stageBar, QFrame#bottomBar {
    background-color: #16181f;
    border: 1px solid #2a2e3a;
    border-radius: 6px;
}
QLabel#productName {
    font-size: 18px;
    font-weight: 700;
    color: #f3f5f8;
    letter-spacing: 1px;
}
QLabel#versionLabel { color: #8b90a0; font-size: 12px; }
QLabel#engineIdle { color: #7ddea0; font-weight: 600; }
QLabel#engineRun { color: #ffd56a; font-weight: 600; }
QLabel#engineMiss { color: #ff7b7b; font-weight: 600; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {
    background-color: #0f1117;
    border: 1px solid #343848;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #3d5a80;
}
QLineEdit:focus, QComboBox:focus { border-color: #6ea8ff; }
QPushButton {
    background-color: #2a3144;
    border: 1px solid #3c4560;
    border-radius: 4px;
    padding: 6px 12px;
    color: #e8eaef;
}
QPushButton:hover { background-color: #35405a; }
QPushButton:pressed { background-color: #243049; }
QPushButton:disabled { color: #6c7080; background-color: #1c1f28; border-color: #2a2e3a; }
QPushButton#primary {
    background-color: #3a6df0;
    border-color: #4c7cff;
    font-weight: 600;
}
QPushButton#primary:hover { background-color: #4b7aff; }
QPushButton#danger { background-color: #5a2a2a; border-color: #7a3a3a; }
QPushButton#danger:hover { background-color: #6e3333; }
QPushButton#danger:disabled { color: #6c7080; background-color: #1c1f28; border-color: #2a2e3a; }
QPushButton#stage {
    min-width: 150px;
    text-align: center;
    background-color: #1e2433;
}
QPushButton#stage:checked {
    background-color: #2c4a7a;
    border-color: #6ea8ff;
    font-weight: 600;
}
QTableWidget {
    background-color: #0f1117;
    alternate-background-color: #161922;
    gridline-color: #2a2e3a;
    border: 1px solid #2a2e3a;
    border-radius: 4px;
}
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background-color: #2c4a7a; }
QHeaderView::section {
    background-color: #1a1d27;
    color: #b8bdc9;
    border: 0;
    border-bottom: 1px solid #2a2e3a;
    border-right: 1px solid #2a2e3a;
    padding: 6px 8px;
    font-weight: 600;
}
QProgressBar {
    background-color: #0f1117;
    border: 1px solid #2a2e3a;
    border-radius: 4px;
    text-align: center;
    color: #e8eaef;
    height: 16px;
}
QProgressBar::chunk { background-color: #3a6df0; border-radius: 3px; }
QListWidget, QTreeWidget {
    background-color: #0f1117;
    border: 1px solid #2a2e3a;
    border-radius: 4px;
}
QListWidget::item { padding: 4px; }
QCheckBox, QRadioButton { spacing: 6px; }
QGroupBox {
    border: 1px solid #2a2e3a;
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #b8bdc9; }
QSplitter::handle { background-color: #2a2e3a; }
QScrollBar:vertical { background: #12141a; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3f50; min-height: 20px; border-radius: 5px; }
QStatusBar { background: #1a1d27; color: #8b90a0; }
QMessageBox { background-color: #1a1d27; }
"""
