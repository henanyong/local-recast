"""Assembly view in the character library (offscreen Qt tests).

Click a face / body / costume pick: it applies immediately and is saved
to meta.json. No renderer on this machine, so the 3D preview is the
honest placeholder card, never a fake picture.
"""

import json
import os
import struct
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _image(path: Path, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def _glb(path: Path) -> Path:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


def _service(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window(tmp_path: Path):
    from recast.desktop.library_window import LibraryWindow

    _app()
    svc = _service(tmp_path)
    svc.store.insert(
        "characters",
        name="温念",
        display_name="温念",
        confirmed=1,
        authorized=1,
        cluster_key="cluster:温念",
    )
    folder = Path(svc.ensure_character_folder("温念")["folder"])
    _image(folder / "face" / "a.jpg", (200, 30, 30))
    _image(folder / "face" / "b.jpg", (30, 30, 200))
    _image(folder / "costume" / "dress.png", (30, 200, 30))
    svc.import_model_glb("温念", _glb(tmp_path / "model.glb"))
    _image(tmp_path / "outputs" / "wardrobe" / "red.png", (200, 30, 30))
    _glb(tmp_path / "outputs" / "models" / "fox.glb")
    window = LibraryWindow(svc)
    window._sync_tasks = True
    window.list.setCurrentRow(0)
    window._on_select(window.list.currentItem())
    return svc, window, folder


def _click(row, index: int) -> None:
    item = row.item(index)
    assert item is not None
    row.setCurrentItem(item)
    row.itemClicked.emit(item)


def _click_named(row, name: str) -> None:
    """Click the item whose text starts with ``name`` (rows sort by character)."""
    for index in range(row.count()):
        item = row.item(index)
        if item is not None and item.text().startswith(name):
            row.setCurrentItem(item)
            row.itemClicked.emit(item)
            return
    raise AssertionError(f"装配行里没有 {name}：{[row.item(i).text() for i in range(row.count())]}")


def test_assembly_view_present_with_three_rows(tmp_path: Path):
    _svc, window, _folder = _window(tmp_path)
    from PySide6.QtWidgets import QGroupBox

    titles = [g.title() for g in window.findChildren(QGroupBox)]
    assert any("人物装配" in t for t in titles)
    assert window.assembly_face_list.count() == 2
    assert window.assembly_body_list.count() == 2  # own GLB + shared fox.glb
    assert window.assembly_costume_list.count() == 2  # dress.png + shared red.png
    assert "当前装配" in window.assembly_summary.text()


def test_click_face_applies_immediately(tmp_path: Path):
    _svc, window, folder = _window(tmp_path)
    from recast.core.characters import load_meta

    assert window.assembly_face_list.currentItem().text() == "a.jpg"
    _click(window.assembly_face_list, 1)
    assert load_meta(folder)["assembly"]["face"] == "face/b.jpg"
    assert "脸 b.jpg" in window.assembly_summary.text()


def test_click_shared_body_applies_and_shows_placeholder_preview(tmp_path: Path):
    _svc, window, folder = _window(tmp_path)
    from recast.core.characters import load_meta

    _click_named(window.assembly_body_list, "fox.glb")
    assert load_meta(folder)["assembly"]["body"] == "models/fox.glb"
    assert "身体 fox.glb" in window.assembly_summary.text()
    # No renderer on this machine: placeholder card, never a fake picture.
    assert "Blender" in window.assembly_preview.text()


def test_click_shared_costume_copies_and_binds(tmp_path: Path):
    _svc, window, folder = _window(tmp_path)
    from recast.assets3d.binding import list_bindings
    from recast.core.characters import load_meta

    _click_named(window.assembly_costume_list, "red.png")
    assert load_meta(folder)["assembly"]["costume"] == "wardrobe/red.png"
    assert (folder / "costume" / "red.png").is_file()
    assert len(list_bindings(folder)) == 1
    assert list_bindings(folder)[0]["source"] == "wardrobe/red.png"
    assert "已绑定" in window.assembly_summary.text()


def test_assembly_view_empty_without_selection(tmp_path: Path):
    _svc, window, _folder = _window(tmp_path)
    window.list.setCurrentRow(-1)
    window._on_select(None)
    assert "请从左侧选择人物" in window.assembly_name.text()
    assert window.assembly_face_list.count() == 0
    assert window.assembly_body_list.count() == 0
    assert window.assembly_costume_list.count() == 0
