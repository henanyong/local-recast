"""Character folders: structure, meta.json, idempotent migration."""

import json
from pathlib import Path

from PIL import Image

from recast.core.characters import (
    ensure_character_folder,
    list_character_folders,
    load_meta,
    migrate_to_character_folders,
    safe_folder_name,
)


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (200, 120, 90)).save(path)
    return path


def test_safe_folder_name_strips_windows_forbidden_chars():
    assert safe_folder_name("温念") == "温念"
    assert safe_folder_name('a/b:c"d') == "a_b_c_d"
    assert safe_folder_name("  ") == "未命名"


def test_ensure_creates_tree_and_meta(tmp_path: Path):
    folder, meta = ensure_character_folder(tmp_path / "outputs", "温念")
    assert folder == tmp_path / "outputs" / "characters" / "温念"
    for sub in ("face", "costume", "voice", "cutout", "model"):
        assert (folder / sub).is_dir()
    assert meta["name"] == "温念"
    assert meta["version"] == 1
    assert load_meta(folder)["name"] == "温念"
    # Second call keeps the existing meta (does not reset it).
    meta["costume"]["description"] = "红袄"
    from recast.core.characters import save_meta

    save_meta(folder, meta)
    _folder2, meta2 = ensure_character_folder(tmp_path / "outputs", "温念")
    assert meta2["costume"]["description"] == "红袄"


def test_list_character_folders(tmp_path: Path):
    ensure_character_folder(tmp_path / "outputs", "温念")
    ensure_character_folder(tmp_path / "outputs", "陈秋微")
    names = [c["name"] for c in list_character_folders(tmp_path / "outputs")]
    assert names == ["温念", "陈秋微"]


def test_migrate_face_samples_and_l2_is_idempotent(tmp_path: Path):
    out = tmp_path / "outputs"
    workdir = tmp_path / "work"
    # Fake face_samples layout: index.json + per-person dirs.
    src_face = _image(workdir / "face_samples" / "Person_1" / "00_f10.jpg")
    src_face2 = _image(workdir / "face_samples" / "Person_1" / "01_f20.jpg")
    (workdir / "face_samples" / "index.json").write_text(
        json.dumps({"Person_1": [str(src_face), str(src_face2)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Fake L2 asset dir with asset.json.
    l2_dir = out / "l2" / "char-1"
    l2_dir.mkdir(parents=True)
    (l2_dir / "asset.json").write_text(
        json.dumps(
            {"character_id": "char-1", "display_name": "温念", "kind": "3d"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (l2_dir / "albedo.png").write_bytes(b"fake-png")

    face_samples = json.loads((workdir / "face_samples" / "index.json").read_text(encoding="utf-8"))
    first = migrate_to_character_folders(out, face_samples=face_samples)
    assert first["copied"] == 4, first  # 2 张脸样本 + L2 的 asset.json 和 albedo.png
    assert first["errors"] == [], first
    assert "温念" in first["created"]
    assert "Person_1" in first["created"]
    folder = out / "characters" / "温念"
    assert (folder / "face").is_dir()
    assert (folder / "model" / "asset.json").is_file()
    meta = load_meta(folder)
    assert meta["model"]["source"] == "face_card"
    assert "脸卡占位" in meta["model"]["note"]

    # Sources are untouched, and a second run copies nothing.
    assert src_face.is_file() and (l2_dir / "asset.json").is_file()
    second = migrate_to_character_folders(out, face_samples=face_samples)
    assert second["copied"] == 0, second
    assert second["errors"] == [], second


def test_migrate_roster_costume_and_voice(tmp_path: Path):
    out = tmp_path / "outputs"
    costume = _image(tmp_path / "ref" / "hongao.jpg")
    voice = tmp_path / "ref" / "wennian.wav"
    voice.write_bytes(b"RIFFfake")
    roster = {
        "characters": [
            {"name": "温念", "costume": str(costume), "voice": str(voice)},
            {"name": "陈秋微", "costume": str(tmp_path / "ref" / "missing.jpg")},
        ]
    }
    report = migrate_to_character_folders(out, roster=roster)
    assert report["copied"] == 2, report
    assert (out / "characters" / "温念" / "costume" / "hongao.jpg").is_file()
    assert (out / "characters" / "温念" / "voice" / "wennian.wav").is_file()
    meta = load_meta(out / "characters" / "陈秋微")
    assert meta["costume"]["missing"] == [str(tmp_path / "ref" / "missing.jpg")]
    assert any("不存在" in e for e in report["errors"])
