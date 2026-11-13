"""Shared character-folder resolver: folder first, old logic second."""

from __future__ import annotations

import json
from pathlib import Path

from recast.core.character_assets import (
    canonical_speaker_name,
    resolve_character_assets,
    resolve_face_reference,
    resolve_generation_ref,
    resolve_voice,
    resolve_voices,
)
from recast.core.characters import ensure_character_folder, save_meta


def _make_folder(root: Path, name: str, **meta_patch) -> Path:
    folder, meta = ensure_character_folder(root, name)
    meta.update(meta_patch)
    save_meta(folder, meta)
    return folder


def _write(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_resolve_full_folder(tmp_path: Path):
    root = tmp_path / "outputs"
    folder = _make_folder(root, "温念")
    face = _write(folder / "face" / "a.jpg")
    sample = _write(folder / "voice" / "s1.wav")
    line = _write(folder / "voice" / "lines" / "F1-L1.wav")
    cut = _write(folder / "cutout" / "c1.png")
    shot = _write(folder / "costume" / "dress.jpg")
    glb = _write(folder / "model" / "model.glb")
    _, meta = ensure_character_folder(root, "温念")
    meta["voice"] = {
        "samples": [str(sample)],
        "missing": [],
        "timbre": {"pitch": 2.0, "rate": 1.1, "style": ""},
        "dubbed_lines": {"F1-L1": str(line)},
    }
    meta["costume"] = {"description": "红上衣", "source": "auto", "files": [str(shot)], "missing": []}
    meta["model"] = {"source": "manual", "format": "glb", "path": "model/model.glb", "note": ""}
    save_meta(folder, meta)

    assets = resolve_character_assets(root, "温念")
    assert assets is not None
    assert assets.name == "温念"
    assert assets.face_path == face
    assert assets.voice_samples == [sample]
    assert assets.dubbed_lines == {"F1-L1": line}
    assert assets.cutouts == [cut]
    assert assets.costume_files == [shot]
    assert assets.costume_description == "红上衣"
    assert assets.model_path == glb
    assert assets.model_source == "manual"
    assert assets.voice == str(sample)
    assert assets.timbre == {"pitch": 2.0, "rate": 1.1, "style": ""}


def test_resolve_missing_folder_returns_none(tmp_path: Path):
    assert resolve_character_assets(tmp_path, " nobody ") is None
    assert resolve_face_reference(tmp_path, "温念") is None
    assert resolve_voice(tmp_path, "温念") == (None, {"pitch": 0.0, "rate": 1.0, "style": ""})
    assert canonical_speaker_name(tmp_path, "温念") == "温念"
    assert resolve_generation_ref(tmp_path, "温念") is None


def test_resolve_ignores_missing_files_and_corrupt_meta(tmp_path: Path):
    root = tmp_path / "outputs"
    folder, meta = ensure_character_folder(root, "陈秋微")
    meta["voice"] = {"samples": [str(folder / "voice" / "gone.wav")], "missing": [], "timbre": {}}
    save_meta(folder, meta)
    assets = resolve_character_assets(root, "陈秋微")
    assert assets is not None
    assert assets.voice_samples == []
    assert assets.voice is None

    (folder / "meta.json").write_text("{not json", encoding="utf-8")
    assert resolve_character_assets(root, "陈秋微") is None


def test_resolve_tts_voice_beats_sample(tmp_path: Path):
    root = tmp_path / "outputs"
    folder, meta = ensure_character_folder(root, "陆寒铮")
    sample = _write(folder / "voice" / "s1.wav")
    meta["voice"] = {
        "samples": [str(sample)],
        "missing": [],
        "timbre": {},
        "tts_voice": "zh-CN-YunyangNeural",
    }
    save_meta(folder, meta)
    voice, _ = resolve_voice(root, "陆寒铮")
    assert voice == "zh-CN-YunyangNeural"


def test_resolve_face_reference_shape(tmp_path: Path):
    root = tmp_path / "outputs"
    folder, _ = ensure_character_folder(root, "温雪")
    face = _write(folder / "face" / "f.jpg")
    row = resolve_face_reference(root, "温雪")
    assert row == {"character_id": None, "name": "温雪", "face_path": str(face)}


def test_resolve_by_store_id(tmp_path: Path):
    from recast.store.db import Store

    root = tmp_path / "outputs"
    store = Store(tmp_path / "recast.db")
    store.insert("characters", id="c1", name="温念", display_name="温念")
    folder, _ = ensure_character_folder(root, "温念")
    face = _write(folder / "face" / "f.jpg")
    row = resolve_face_reference(root, "c1", store=store)
    assert row is not None
    assert row["character_id"] == "c1"
    assert row["face_path"] == str(face)


def test_resolve_voices_folder_first_then_fallback(tmp_path: Path):
    root = tmp_path / "outputs"
    folder, meta = ensure_character_folder(root, "温念")
    sample = _write(folder / "voice" / "s1.wav")
    meta["voice"] = {"samples": [str(sample)], "missing": [], "timbre": {"pitch": 3.0, "rate": 1.0, "style": ""}}
    save_meta(folder, meta)
    voices, timbres = resolve_voices(
        root,
        ["温念", "陈秋微"],
        fallback={"温念": "zh-CN-XiaoxiaoNeural", "陈秋微": "zh-CN-YunyangNeural"},
    )
    assert voices["温念"] == str(sample)  # folder wins over fallback
    assert voices["陈秋微"] == "zh-CN-YunyangNeural"  # no folder: fallback kept
    assert timbres == {"温念": {"pitch": 3.0, "rate": 1.0, "style": ""}}


def test_generation_ref_only_for_character_tokens(tmp_path: Path):
    root = tmp_path / "outputs"
    folder, _ = ensure_character_folder(root, "温念")
    face = _write(folder / "face" / "f.jpg")
    assert resolve_generation_ref(root, "温念") == face
    assert resolve_generation_ref(root, "村口") is None
    assert canonical_speaker_name(root, "温念") == "温念"
    assert canonical_speaker_name(root, "路人甲") == "路人甲"
