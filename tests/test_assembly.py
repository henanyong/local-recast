"""Character assembly: meta.json ``assembly`` {face, body, costume}.

One character's current outfit: which face, which body (GLB), which costume.
Picks must live inside the character folder or the shared wardrobe/models
libraries; anything outside (or a symlink) is refused.
"""

import json
import struct
from pathlib import Path

import pytest
from PIL import Image

from recast.core.characters import (
    default_meta,
    ensure_character_folder,
    get_assembly,
    load_meta,
    resolve_assembly_path,
    set_assembly,
)


def _image(path: Path, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def _glb_bytes() -> bytes:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _glb(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_glb_bytes())
    return path


def _folder(tmp_path: Path, name: str = "温念"):
    folder, _meta = ensure_character_folder(tmp_path / "out", name)
    return folder


def test_default_meta_has_empty_assembly():
    assert default_meta("温念")["assembly"] == {"face": "", "body": "", "costume": ""}


def test_get_assembly_defaults_on_old_meta():
    assert get_assembly({}) == {"face": "", "body": "", "costume": ""}
    assert get_assembly({"assembly": {"face": "face/a.jpg"}}) == {
        "face": "face/a.jpg",
        "body": "",
        "costume": "",
    }


def test_set_assembly_stores_relative_paths_and_keeps_others(tmp_path: Path):
    folder = _folder(tmp_path)
    face = _image(folder / "face" / "a.jpg")
    body = _glb(folder / "model" / "model.glb")
    out = set_assembly(folder, face=face, body=body)
    assert out == {"face": "face/a.jpg", "body": "model/model.glb", "costume": ""}
    # Partial update: only the given field changes.
    costume = _image(folder / "costume" / "dress.png")
    out = set_assembly(folder, costume=costume)
    assert out == {"face": "face/a.jpg", "body": "model/model.glb", "costume": "costume/dress.png"}
    # Clearing one slot leaves the others alone.
    out = set_assembly(folder, face="")
    assert out["face"] == "" and out["body"] == "model/model.glb"
    assert load_meta(folder)["assembly"] == out


def test_set_assembly_accepts_shared_library_paths(tmp_path: Path):
    folder = _folder(tmp_path)
    root = tmp_path / "out"
    dress = _image(root / "wardrobe" / "red.png")
    fox = _glb(root / "models" / "fox.glb")
    out = set_assembly(folder, costume=dress, body=fox)
    assert out["costume"] == "wardrobe/red.png"
    assert out["body"] == "models/fox.glb"
    assert resolve_assembly_path(folder, out["costume"]) == dress
    assert resolve_assembly_path(folder, out["body"]) == fox


def test_set_assembly_refuses_outside_files(tmp_path: Path):
    folder = _folder(tmp_path)
    outside = _image(tmp_path / "outside.jpg")
    with pytest.raises(ValueError, match="文件夹里面"):
        set_assembly(folder, face=outside)
    with pytest.raises(ValueError, match="不是有效文件"):
        set_assembly(folder, face=folder / "face" / "missing.jpg")


def test_set_assembly_refuses_wrong_types(tmp_path: Path):
    folder = _folder(tmp_path)
    text = folder / "face" / "note.txt"
    text.parent.mkdir(parents=True, exist_ok=True)
    text.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="必须是图片"):
        set_assembly(folder, face=text)
    img = _image(folder / "face" / "a.jpg")
    with pytest.raises(ValueError, match="必须是 GLB"):
        set_assembly(folder, body=img)
    bad_glb = folder / "model" / "bad.glb"
    bad_glb.parent.mkdir(parents=True, exist_ok=True)
    bad_glb.write_bytes(b"not a glb at all")
    with pytest.raises(ValueError, match="已损坏"):
        set_assembly(folder, body=bad_glb)


def test_set_assembly_refuses_symlinks(tmp_path: Path):
    folder = _folder(tmp_path)
    target = _image(tmp_path / "real.jpg")
    link = folder / "face" / "link.jpg"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("创建符号链接需要管理员权限或 Windows 开发者模式，跳过。")
    with pytest.raises(ValueError, match="符号链接"):
        set_assembly(folder, face=link)


def test_resolve_assembly_path_variants(tmp_path: Path):
    folder = _folder(tmp_path)
    face = _image(folder / "face" / "a.jpg")
    assert resolve_assembly_path(folder, "") is None
    assert resolve_assembly_path(folder, "face/a.jpg") == face
    assert resolve_assembly_path(folder, str(face)) == face
    assert resolve_assembly_path(folder, "face/gone.jpg") is None
    assert resolve_assembly_path(folder, "wardrobe/gone.png") is None


def test_assembly_choice_takes_effect_downstream(tmp_path: Path):
    """Face/body picks change what the shared resolver returns.

    resolve_face_reference prefers assembly.face (else the first face
    sample); resolve_body_model prefers assembly.body (else the folder's
    own model GLB); resolve_generation_ref keeps returning the face image
    for storyboard refs (image models cannot take a GLB).
    """
    from recast.core.character_assets import (
        resolve_body_model,
        resolve_face_reference,
        resolve_generation_ref,
    )

    root = tmp_path / "out"
    folder = _folder(tmp_path)
    face_a = _image(folder / "face" / "a.jpg", (200, 30, 30))
    face_b = _image(folder / "face" / "b.jpg", (30, 30, 200))
    own_glb = _glb(folder / "model" / "model.glb")
    shared_glb = _glb(root / "models" / "fox.glb")

    # No assembly yet: first face sample, own model GLB.
    assert resolve_face_reference(root, "温念")["face_path"] == str(face_a)
    assert resolve_body_model(root, "温念") == own_glb
    assert resolve_generation_ref(root, "温念") == face_a

    # Assemble another face and a shared body: both take effect.
    set_assembly(folder, face=face_b, body=shared_glb)
    assert resolve_face_reference(root, "温念")["face_path"] == str(face_b)
    assert resolve_body_model(root, "温念") == shared_glb
    assert resolve_generation_ref(root, "温念") == face_b

    # Clearing the face falls back to the first sample again.
    set_assembly(folder, face="")
    assert resolve_face_reference(root, "温念")["face_path"] == str(face_a)
