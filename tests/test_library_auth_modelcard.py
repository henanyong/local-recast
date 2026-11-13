"""Authorization upfront display + 3D model detail card + folder sync wiring.

Covers recast/desktop/library_window.py::auth_flag_text,
replace_block_reason, describe_glb_model, the greyed unauthorized rows,
the replace-entry gating, and the body detail card.
"""

import json
import os
import struct
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_glb(path: Path, payload: dict) -> Path:
    data = json.dumps(payload).encode("utf-8")
    data += b" " * ((-len(data)) % 4)
    total = 12 + 8 + len(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<II", len(data), 0x4E4F534A))
        fh.write(data)
    return path


def _glb_payload(*, triangles: int = 4, skins: int = 0, images: int = 0) -> dict:
    accessors = [{"count": triangles * 3}, {"count": triangles * 3}]
    payload: dict = {
        "asset": {"version": "2.0"},
        "accessors": accessors,
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}
        ],
    }
    if skins:
        payload["skins"] = [{"joints": [0]} for _ in range(skins)]
    if images:
        payload["images"] = [{"uri": f"tex{i}.png"} for i in range(images)]
    return payload


def _face(path: Path) -> Path:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), (180, 140, 120))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 10, 48, 50), fill=(210, 170, 150))
    img.save(path)
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
    window = LibraryWindow(svc)
    window._sync_tasks = True
    return svc, window


def _add_character(svc, name: str, *, authorized: bool, auth_type: str = "") -> dict:
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=1,
        authorized=1 if authorized else 0,
        auth_type=auth_type,
        cluster_key=f"cluster:{name}",
    )


def _select(window, row: int = 0) -> None:
    window.list.setCurrentRow(row)
    window._on_select(window.list.currentItem())


# ----- pure helpers -----


def test_auth_flag_text_variants():
    from recast.desktop.library_window import auth_flag_text

    assert auth_flag_text({"authorized": 1, "auth_type": "自有拍摄"}) == "已授权（自有拍摄）"
    assert auth_flag_text({"authorized": 1, "auth_type": ""}) == "已授权（未选类型）"
    assert auth_flag_text({"authorized": 0}) == "未授权"
    assert auth_flag_text(None) == "未授权"


def test_replace_block_reason():
    from recast.desktop.library_window import replace_block_reason

    assert replace_block_reason({"authorized": 1, "auth_type": "已授权"}, "温念") is None
    assert replace_block_reason(None, "温念") is None
    reason = replace_block_reason({"authorized": 0}, "温念")
    assert reason is not None and "温念" in reason and "未授权，不可替换" in reason
    reason = replace_block_reason({"authorized": 1, "auth_type": ""}, "温念")
    assert reason is not None and "未选授权类型" in reason


def test_describe_glb_model_counts(tmp_path: Path):
    from recast.desktop.library_window import describe_glb_model

    path = _write_glb(tmp_path / "m.glb", _glb_payload(triangles=4, skins=1, images=2))
    info = describe_glb_model(path)
    assert info["ok"] is True
    assert info["triangles"] == 4
    assert info["skeleton_count"] == 1
    assert info["texture_count"] == 2
    assert info["size"] == path.stat().st_size


def test_describe_glb_model_no_skeleton_no_textures(tmp_path: Path):
    from recast.desktop.library_window import describe_glb_model

    path = _write_glb(tmp_path / "m.glb", _glb_payload(triangles=10))
    info = describe_glb_model(path)
    assert info["ok"] is True
    assert info["triangles"] == 10
    assert info["skeleton_count"] == 0
    assert info["texture_count"] == 0


def test_describe_glb_model_invalid_and_missing(tmp_path: Path):
    from recast.desktop.library_window import describe_glb_model

    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb at all")
    assert describe_glb_model(bad)["ok"] is False
    assert describe_glb_model(tmp_path / "gone.glb")["ok"] is False


# ----- library window: authorization display -----


def test_unauthorized_row_greyed_and_flagged(tmp_path: Path):
    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=True, auth_type="自有拍摄")
    _add_character(svc, "陈秋微", authorized=False)
    window.reload()
    assert window.list.count() == 2
    row0 = window.list.item(0)
    row1 = window.list.item(1)
    assert "已授权（自有拍摄）" in row0.text()
    assert "未授权" in row1.text()
    from PySide6.QtGui import QColor

    assert row1.foreground().color() == QColor("#8b90a0")
    assert row0.foreground().color() != QColor("#8b90a0")


def test_audit_line_states_authorization(tmp_path: Path):
    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=False)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    assert "未授权，不可替换" in window.audit_label.text()


def test_swap_preview_blocked_for_unauthorized(tmp_path: Path):
    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=False)
    svc.ensure_character_folder("温念")
    _face(tmp_path / "outputs" / "characters" / "温念" / "face" / "f.jpg")
    window.reload()
    _select(window, 0)
    assert window.swap_preview_btn.isEnabled() is False
    assert "未授权，不可替换" in window.swap_preview_btn.toolTip()


def test_swap_preview_allowed_for_authorized(tmp_path: Path):
    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=True, auth_type="自有拍摄")
    svc.ensure_character_folder("温念")
    _face(tmp_path / "outputs" / "characters" / "温念" / "face" / "f.jpg")
    window.reload()
    _select(window, 0)
    assert window.swap_preview_btn.isEnabled() is True


def test_finish_review_closes_even_for_unauthorized(tmp_path: Path):
    """关库只是导航（回主窗口点「下一步」才跑替换），所以不拦未授权人物。

    真正的替换入口在主窗口，不在本轮文件边界内；人物库里已经把
    未授权写清了（列表行置灰 +「未授权」，审核行写「未授权，不可替换」，
    见 test_audit_line_states_authorization）。
    """
    from PySide6.QtWidgets import QDialog

    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=False)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    assert window.service.list_review_queue() == []
    window._finish_review()
    assert window.result() == QDialog.DialogCode.Accepted


# ----- library window: 3D model detail card -----


def test_body_detail_card_shared_model_no_skeleton(tmp_path: Path):
    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=True, auth_type="自有拍摄")
    svc.ensure_character_folder("温念")
    _write_glb(
        tmp_path / "outputs" / "models" / "shared.glb",
        _glb_payload(triangles=12500, skins=0, images=0),
    )
    window.reload()
    _select(window, 0)
    assert window.assembly_body_list.count() == 1
    window.assembly_body_list.setCurrentRow(0)
    text = window.assembly_body_detail.text()
    assert "模型：shared.glb" in text
    assert "来源：共享模型库" in text
    assert "面数：约 12,500 个三角面" in text
    assert "骨骼：无" in text
    assert "贴图：无" in text
    assert "大小：" in text
    assert "不可直接重定向，需先绑骨" in text


def test_body_detail_card_with_skeleton_no_warning(tmp_path: Path):
    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=True, auth_type="自有拍摄")
    svc.ensure_character_folder("温念")
    _write_glb(
        tmp_path / "outputs" / "models" / "rigged.glb",
        _glb_payload(triangles=100, skins=1, images=1),
    )
    window.reload()
    _select(window, 0)
    window.assembly_body_list.setCurrentRow(0)
    text = window.assembly_body_detail.text()
    assert "骨骼：有（1 套）" in text
    assert "贴图：有（1 张）" in text
    assert "不可直接重定向" not in text


def test_body_detail_card_no_model_placeholder(tmp_path: Path):
    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=True, auth_type="自有拍摄")
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    assert "没有 3D 模型" in window.assembly_body_detail.text()


def test_body_detail_text_own_model_source(tmp_path: Path):
    from recast.assets3d.glb import import_glb

    svc, window = _window(tmp_path)
    _add_character(svc, "温念", authorized=True, auth_type="自有拍摄")
    folder = Path(svc.ensure_character_folder("温念")["folder"])
    src = _write_glb(tmp_path / "src.glb", _glb_payload(triangles=6, skins=1))
    import_glb(src, folder)
    window.reload()
    _select(window, 0)
    text = window._body_detail_text(str(folder / "model" / "model.glb"), window._current_assets())
    assert "来源：手动导入" in text
    assert "面数：约 6 个三角面" in text


# ----- reload() mirrors store embeddings into folders -----


def test_reload_syncs_store_embedding_into_folder(tmp_path: Path):
    import json as _json

    import numpy as np
    from recast.core.characters import load_meta

    svc = _service(tmp_path)
    emb = list(np.array([1.0, 0.0, 0.0]))
    svc.store.insert(
        "characters",
        name="温念",
        display_name="温念",
        confirmed=1,
        authorized=1,
        auth_type="自有拍摄",
        cluster_key="cluster:温念",
        face_embedding=_json.dumps(emb),
    )
    svc.ensure_character_folder("温念")
    assert load_meta(tmp_path / "outputs" / "characters" / "温念").get("face_embedding") is None
    from recast.desktop.library_window import LibraryWindow

    _app()
    window = LibraryWindow(svc)
    window._sync_tasks = True
    meta = load_meta(tmp_path / "outputs" / "characters" / "温念")
    assert meta is not None
    assert np.allclose(meta["face_embedding"], emb)
