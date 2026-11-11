# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir spec for 本地再铸 V1.0.
# Run on the user's Windows PC: scripts\build_windows.bat
# Large model weights are NOT bundled.

from pathlib import Path

block_cipher = None
root = Path(SPECPATH)

# Hidden imports that PyInstaller often misses (optional ones are wrapped in try/except at runtime).
hidden = [
    "recast",
    "recast.desktop",
    "recast.desktop.app",
    "recast.desktop.main_window",
    "recast.core",
    "recast.core.runner",
    "recast.extract",
    "recast.stages",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "yaml",
    "cv2",
    "numpy",
    "PIL",
    "skimage",
    "sklearn",
    "faster_whisper",
    "ultralytics",
    "insightface",
    "onnxruntime",
    "scenedetect",
]

datas = [
    (str(root / "recast" / "workbench_static"), "recast/workbench_static"),
    (str(root / "configs"), "configs"),
    (str(root / "examples"), "examples"),
]

a = Analysis(
    [str(root / "recast" / "__main__.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["models", "outputs"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="本地再铸",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # keep a console so doctor / logs are visible
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="本地再铸",
)
