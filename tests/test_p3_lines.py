"""P3: the three application lines read from character folders first.

Folder first, old logic second: when a character folder exists its assets
win; without a usable folder every line behaves exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recast.core.characters import ensure_character_folder, save_meta
from recast.core.constants import AUTH_LICENSED, AUTH_OWN_SHOOT
from recast.core.runner import resolve_swap_references
from recast.store.db import Store


def _authorized_store(tmp_path: Path, faces: dict[str, Path]) -> Store:
    store = Store(tmp_path / "store.db")
    ids = {}
    for name, auth in (("温念", AUTH_LICENSED), ("陈秋微", AUTH_OWN_SHOOT)):
        row = store.insert(
            "characters", name=name, display_name=name, authorized=1, auth_type=auth, confirmed=1
        )
        ids[name] = row["id"]
        if name in faces:
            store.set_character_asset(row["id"], "face", str(faces[name]), complete=True)
    return store, ids


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


# ----- replace line: folder face wins, old store face is the fallback -----


def test_replace_prefers_folder_face(tmp_path: Path):
    old_face = _write(tmp_path / "old" / "wen.jpg")
    store, ids = _authorized_store(tmp_path, {"温念": old_face})
    root = tmp_path / "outputs"
    folder, _ = ensure_character_folder(root, "温念")
    folder_face = _write(folder / "face" / "best.jpg")

    rows = resolve_swap_references(store=store, character_id=ids["温念"], output_root=root)
    assert [r["face_path"] for r in rows] == [str(folder_face)]
    assert rows[0]["character_id"] == ids["温念"]


def test_replace_falls_back_to_store_face_without_folder(tmp_path: Path):
    old_face = _write(tmp_path / "old" / "wen.jpg")
    store, ids = _authorized_store(tmp_path, {"温念": old_face})
    rows = resolve_swap_references(store=store, character_id=ids["温念"], output_root=tmp_path / "outputs")
    assert [r["face_path"] for r in rows] == [str(old_face)]


def test_replace_all_authorized_mixes_folder_and_store(tmp_path: Path):
    wen_old = _write(tmp_path / "old" / "wen.jpg")
    chen_old = _write(tmp_path / "old" / "chen.jpg")
    store, _ = _authorized_store(tmp_path, {"温念": wen_old, "陈秋微": chen_old})
    root = tmp_path / "outputs"
    folder, _ = ensure_character_folder(root, "温念")
    folder_face = _write(folder / "face" / "best.jpg")

    rows = resolve_swap_references(store=store, output_root=root)
    by_name = {r["name"]: r["face_path"] for r in rows}
    assert by_name == {"温念": str(folder_face), "陈秋微": str(chen_old)}


def test_replace_no_folder_no_face_still_errors(tmp_path: Path):
    store, ids = _authorized_store(tmp_path, {})
    with pytest.raises(RuntimeError, match="缺少参考脸"):
        resolve_swap_references(store=store, character_id=ids["温念"], output_root=tmp_path / "outputs")


def test_replace_without_output_root_unchanged(tmp_path: Path):
    old_face = _write(tmp_path / "old" / "wen.jpg")
    store, ids = _authorized_store(tmp_path, {"温念": old_face})
    rows = resolve_swap_references(store=store, character_id=ids["温念"])
    assert [r["face_path"] for r in rows] == [str(old_face)]


# ----- script line: folder face / voice / timbre first -----


def _minimal_pack(tmp_path: Path) -> Path:
    import json

    pack = tmp_path / "pack"
    (pack / "refs").mkdir(parents=True)
    (pack / "pack.json").write_text(
        json.dumps({"name": "test", "version": "1", "script": "script.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (pack / "script.json").write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
    (pack / "characters.json").write_text(
        json.dumps(
            [{"name": "温念", "id": "wen", "voice": "zh-CN-XiaoxiaoNeural"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write(pack / "refs" / "村口.jpg")
    return pack


def test_script_refs_prefer_folder_face(tmp_path: Path):
    from recast.templates.pack import load_pack

    pack = load_pack(_minimal_pack(tmp_path))
    shot = {"shot_code": "F1-L1", "ref_assets": ["温念", "村口.jpg"]}
    # No folder: only the pack still resolves.
    assert [p.name for p in pack.resolve_shot_refs(shot)] == ["村口.jpg"]
    assert [p.name for p in pack.resolve_shot_refs(shot, output_root=tmp_path / "outputs")] == ["村口.jpg"]

    root = tmp_path / "outputs"
    folder, _ = ensure_character_folder(root, "温念")
    folder_face = _write(folder / "face" / "best.jpg")
    resolved = pack.resolve_shot_refs(shot, output_root=root)
    assert resolved == [folder_face, tmp_path / "pack" / "refs" / "村口.jpg"]


def test_script_dub_uses_folder_voice_and_timbre(tmp_path: Path, monkeypatch):
    import json
    import types

    from recast.core.script_pipeline import run_script_episode

    pack_root = _minimal_pack(tmp_path)
    root = tmp_path / "outputs"
    folder, meta = ensure_character_folder(root, "温念")
    sample = _write(folder / "voice" / "s1.wav")
    meta["voice"] = {"samples": [str(sample)], "missing": [], "timbre": {"pitch": 2.0, "rate": 1.0, "style": ""}}
    save_meta(folder, meta)

    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "shots_plan.json").write_text(
        json.dumps(
            [
                {
                    "shot_code": "F1-L1",
                    "dialogue": "你好",
                    "speaker": "温念",
                    "duration_sec": 5.0,
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seen: list[dict] = []

    def fake_synth(text, dest, **kwargs):
        seen.append({"text": text, "voice": kwargs.get("voice"), "timbre": kwargs.get("timbre")})
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"wav")
        return {"path": str(dest), "source": "stub", "duration": 1.0, "skipped": False}

    monkeypatch.setattr("recast.generate.tts.synthesize_line", fake_synth)
    monkeypatch.setattr(
        "recast.generate.stitch.stitch_and_dub", lambda *a, **k: {"output": str(tmp_path / "out.mp4")}
    )
    settings = types.SimpleNamespace(output_root=str(root))
    summary = run_script_episode(
        workdir=workdir, pack_root=pack_root, stages=["dub"], settings=settings
    )
    assert summary is not None
    assert len(seen) == 1
    assert seen[0]["voice"] == str(sample)  # folder sample beats the pack voice
    assert seen[0]["timbre"] == {"pitch": 2.0, "rate": 1.0, "style": ""}


def test_script_dub_falls_back_to_pack_voice(tmp_path: Path, monkeypatch):
    import json
    import types

    from recast.core.script_pipeline import run_script_episode

    pack_root = _minimal_pack(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "shots_plan.json").write_text(
        json.dumps(
            [
                {
                    "shot_code": "F1-L1",
                    "dialogue": "你好",
                    "speaker": "温念",
                    "duration_sec": 5.0,
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seen: list[dict] = []

    def fake_synth(text, dest, **kwargs):
        seen.append({"voice": kwargs.get("voice"), "timbre": kwargs.get("timbre")})
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"wav")
        return {"path": str(dest), "source": "stub", "duration": 1.0, "skipped": False}

    monkeypatch.setattr("recast.generate.tts.synthesize_line", fake_synth)
    monkeypatch.setattr(
        "recast.generate.stitch.stitch_and_dub", lambda *a, **k: {"output": str(tmp_path / "out.mp4")}
    )
    settings = types.SimpleNamespace(output_root=str(tmp_path / "outputs"))
    run_script_episode(workdir=workdir, pack_root=pack_root, stages=["dub"], settings=settings)
    assert seen[0]["voice"] == "zh-CN-XiaoxiaoNeural"  # no folder: pack voice kept
    assert seen[0]["timbre"] is None  # no folder: no timbre passed


# ----- jianying line: speaker canonicalized from the folder -----


def _draft_shots(tmp_path: Path) -> tuple[list[dict], Path, Path]:
    import json

    shot = tmp_path / "S01.mp4"
    shot.write_bytes(b"s")
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"w")
    shots = [
        {
            "shot_code": "S01",
            "start_time": 0.0,
            "duration_sec": 4.0,
            "output_path": str(shot),
            "dialogue": "你好",
            "speaker": "c1",
            "audio_duration": 1.5,
            "audio_path": str(mix),
        },
        {
            "shot_code": "S02",
            "start_time": 4.0,
            "duration_sec": 4.0,
            "dialogue": "再见",
            "speaker": "路人甲",
            "audio_duration": 2.0,
            "audio_path": str(mix),
        },
    ]
    return shots, shot, mix


def test_jianying_speaker_canonicalized_from_folder(tmp_path: Path):
    import json

    from recast.compose.jianying import export_jianying_draft

    shots, shot, mix = _draft_shots(tmp_path)
    root = tmp_path / "outputs"
    ensure_character_folder(root, "温念")
    store = Store(tmp_path / "store.db")
    store.insert("characters", id="c1", name="温念", display_name="温念")

    dest = export_jianying_draft(
        tmp_path / "draft",
        shots=shots,
        output_video=shot,
        audio=mix,
        title="t",
        output_root=root,
        store=store,
    )
    project = json.loads((dest / "project.json").read_text(encoding="utf-8"))
    speakers = [c["speaker"] for c in project["clips"]]
    assert speakers == ["温念", "路人甲"]  # store id -> folder name; unknown speaker kept


def test_jianying_speaker_unchanged_without_folder(tmp_path: Path):
    import json

    from recast.compose.jianying import export_jianying_draft

    shots, shot, mix = _draft_shots(tmp_path)
    dest = export_jianying_draft(
        tmp_path / "draft",
        shots=shots,
        output_video=shot,
        audio=mix,
        title="t",
        output_root=tmp_path / "outputs",
    )
    project = json.loads((dest / "project.json").read_text(encoding="utf-8"))
    assert [c["speaker"] for c in project["clips"]] == ["c1", "路人甲"]
