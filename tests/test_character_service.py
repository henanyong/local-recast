"""Service wiring for P0: character folders, matting, diarization.

Audio is generated in the test (sine tones + silence); matting uses a
stubbed rembg so no model download is needed.
"""

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recast.core.service import RecastService
from recast.stages import matting

SR = 16000


def _service(tmp_path: Path) -> RecastService:
    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (200, 120, 90)).save(path)
    return path


def _tone(path: Path, parts: list[tuple[float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for freq, secs in parts:
        n = int(secs * SR)
        t = np.arange(n) / SR
        wave_data = np.zeros(n) if freq == 0 else 0.5 * np.sin(2 * math.pi * freq * t)
        frames += (np.clip(wave_data, -1, 1) * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(bytes(frames))
    return path


def test_service_ensure_character_folder(tmp_path: Path):
    svc = _service(tmp_path)
    out = svc.ensure_character_folder("温念")
    assert Path(out["folder"]) == tmp_path / "outputs" / "characters" / "温念"
    assert out["meta"]["name"] == "温念"
    assert (Path(out["folder"]) / "voice").is_dir()


def test_service_migrate_from_workdir(tmp_path: Path):
    svc = _service(tmp_path)
    workdir = tmp_path / "work"
    src = _image(workdir / "face_samples" / "Person_1" / "00_f10.jpg")
    (workdir / "face_samples" / "index.json").write_text(
        json.dumps({"Person_1": [str(src)]}, ensure_ascii=False), encoding="utf-8"
    )
    report = svc.migrate_character_folders(workdir=workdir)
    assert report["copied"] == 1, report
    assert (tmp_path / "outputs" / "characters" / "Person_1" / "face" / "00_f10.jpg").is_file()
    # Sources are not deleted.
    assert src.is_file()


def test_service_run_matting(tmp_path: Path, monkeypatch):
    def _fake_rgba():
        img = Image.new("RGBA", (64, 64), (200, 120, 90, 255))
        for x in range(32):
            for y in range(64):
                img.putpixel((x, y), (200, 120, 90, 0))
        return img

    class FakeModule:
        def new_session(self, model):
            return object()

    monkeypatch.setattr(matting, "_load_rembg", lambda: FakeModule())
    monkeypatch.setattr(matting, "_new_session", lambda r, m, d: object())
    monkeypatch.setattr(matting, "_remove_image", lambda r, img, s: _fake_rgba())

    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    _image(tmp_path / "outputs" / "characters" / "温念" / "face" / "00.jpg")
    report = svc.run_matting("温念")
    assert len(report["outputs"]) == 1, report
    assert (tmp_path / "outputs" / "characters" / "温念" / "cutout" / "00.png").is_file()


def test_service_run_matting_missing_folder(tmp_path: Path):
    svc = _service(tmp_path)
    with pytest.raises(FileNotFoundError, match="找不到人物文件夹"):
        svc.run_matting("温念")


def test_service_run_diarization_wav(tmp_path: Path):
    svc = _service(tmp_path)
    wav = _tone(
        tmp_path / "talk.wav",
        [(220.0, 1.0), (0, 0.6), (440.0, 1.0)],
    )
    report = svc.run_diarization(wav, character_names=["温念", "陈秋微"])
    assert report["segments"] == 2, report
    assert len(report["clips"]) == 2, report
    assert (tmp_path / "outputs" / "characters" / "温念" / "voice").glob("*.wav")
    clips = list((tmp_path / "outputs" / "characters" / "温念" / "voice").glob("*.wav"))
    assert len(clips) == 1
    clips2 = list((tmp_path / "outputs" / "characters" / "陈秋微" / "voice").glob("*.wav"))
    assert len(clips2) == 1


def test_service_run_diarization_missing_file(tmp_path: Path):
    svc = _service(tmp_path)
    with pytest.raises(FileNotFoundError, match="找不到音视频文件"):
        svc.run_diarization(tmp_path / "nope.wav")


# ----- P1: costume tagging, name suggestion, GLB import, 3D generation -----


def _glb_bytes() -> bytes:
    import struct

    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_service_tag_costume_heuristic(tmp_path: Path):
    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 100), (200, 30, 30)).save(photo)
    report = svc.tag_costume("温念", sources=[photo], tagger="heuristic")
    assert len(report["tagged"]) == 1
    assert "红色" in report["description"]
    shot = Path(report["tagged"][0]["screenshot"])
    assert shot.is_file() and shot.parent.name == "costume"


def test_service_tag_costume_missing_folder(tmp_path: Path):
    svc = _service(tmp_path)
    with pytest.raises(FileNotFoundError, match="找不到人物文件夹"):
        svc.tag_costume("nobody", sources=[], tagger="heuristic")


def test_service_suggest_names_from_workdir(tmp_path: Path):
    svc = _service(tmp_path)
    svc.ensure_character_folder("Person_1")
    svc.ensure_character_folder("Person_2")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "identity.json").write_text(
        json.dumps(
            {
                "persons": [
                    {"cluster_id": 0, "auto_name": "Person_1", "name": "Person_1", "track_ids": [1], "first_frame": 0},
                    {"cluster_id": 1, "auto_name": "Person_2", "name": "Person_2", "track_ids": [2], "first_frame": 250},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (workdir / "tracks.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {"track_id": 1, "first_frame": 0, "last_frame": 250},
                    {"track_id": 2, "first_frame": 250, "last_frame": 500},
                ]
            }
        ),
        encoding="utf-8",
    )
    shots = [
        {"shot_code": "F1-L1", "speaker": "温念", "start_time": 1.0, "end_time": 3.0},
        {"shot_code": "F1-L2", "speaker": "温念", "start_time": 4.0, "end_time": 6.0},
        {"shot_code": "F2-L1", "speaker": "陆寒铮", "start_time": 12.0, "end_time": 14.0},
    ]
    out = svc.suggest_names(workdir=workdir, dialogue_shots=shots, fps=25.0)
    by_auto = {item["auto_name"]: item for item in out}
    assert by_auto["Person_1"]["suggested_name"] == "温念"
    assert by_auto["Person_2"]["suggested_name"] == "陆寒铮"
    from recast.core.characters import load_meta

    meta = load_meta(tmp_path / "outputs" / "characters" / "Person_1")
    assert meta is not None
    assert meta["name_suggestion"]["name"] == "温念"
    assert meta["name"] == "Person_1"


def test_service_import_model_glb(tmp_path: Path):
    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    src = tmp_path / "hero.glb"
    src.write_bytes(_glb_bytes())
    report = svc.import_model_glb("温念", src)
    assert report["format"] == "glb"
    assert Path(report["path"]).is_file()
    from recast.core.characters import load_meta

    meta = load_meta(tmp_path / "outputs" / "characters" / "温念")
    assert meta is not None and meta["model"]["source"] == "manual"


def test_service_import_model_glb_bad_file(tmp_path: Path):
    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    src = tmp_path / "bad.glb"
    src.write_bytes(b"not a glb at all, just text........")
    from recast.assets3d.glb import GlbError

    with pytest.raises(GlbError, match="不是 GLB"):
        svc.import_model_glb("温念", src)


def test_service_generate_model_3d_needs_cutout_first(tmp_path: Path):
    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    with pytest.raises(FileNotFoundError, match="没有抠像图片"):
        svc.generate_model_3d("温念")


def test_service_generate_model_3d_no_gpu_here(tmp_path: Path):
    # Real environment check: no torch/GPU on this machine, so the backend
    # must refuse with a clear message instead of pretending to generate.
    svc = _service(tmp_path)
    svc.ensure_character_folder("温念")
    cutout = tmp_path / "outputs" / "characters" / "温念" / "cutout" / "00.png"
    cutout.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(cutout)
    from recast.assets3d.generate3d import GenerationError

    with pytest.raises(GenerationError, match="torch|GPU|TripoSG"):
        svc.generate_model_3d("温念")
