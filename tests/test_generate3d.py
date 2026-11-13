"""Tests for automatic 3D generation (recast/assets3d/generate3d.py).

No real TripoSG run here. Torch/GPU absence is simulated with monkeypatch
so the tests do not depend on whether the machine happens to have torch or
a GPU. The tests cover the honest paths — clear errors when prerequisites
are missing, and the registration flow with a stub backend that writes a
minimal valid GLB.
"""

import json
import struct
import sys
from pathlib import Path

import pytest

from recast.assets3d.generate3d import (
    GenerationError,
    TripoSGBackend,
    generate_model_glb,
    get_backend,
)
from recast.core.characters import ensure_character_folder, load_meta


def _glb_bytes() -> bytes:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


class StubBackend:
    name = "stub"

    def check(self):
        return True, ""

    def generate(self, image, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_glb_bytes())
        return dest


def test_triposg_check_fails_without_torch_or_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Simulate "no torch" regardless of the machine: on a machine where torch
    # IS installed (and a GPU exists) the check would get past the torch/GPU
    # gates and fail on the missing repo checkout instead, so the torch-missing
    # path must not depend on the local environment. The reason must say what
    # is missing instead of pretending generation could run.
    monkeypatch.setitem(sys.modules, "torch", None)
    ok, reason = TripoSGBackend(repo_dir=tmp_path).check()
    assert ok is False
    assert reason
    assert "torch" in reason or "GPU" in reason


def test_triposg_generate_raises_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    img = tmp_path / "cutout.png"
    img.write_bytes(b"fake")
    with pytest.raises(GenerationError, match="torch|GPU"):
        TripoSGBackend(repo_dir=tmp_path).generate(img, tmp_path / "model.glb")


def test_triposg_generate_invokes_real_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The real TripoSG CLI (verified 2026-11-12 against the README) is:
    #   python -m scripts.inference_triposg --image-input <img> --output-path <out.glb>
    # There is no run.py. Fake torch (with a GPU), a fake repo checkout, and
    # a fake subprocess.run: the test proves the exact command local-recast
    # builds, without running any real model.
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    repo = tmp_path / "TripoSG"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "inference_triposg.py").write_text("# stub\n", encoding="utf-8")

    seen: dict = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["cwd"] = kwargs.get("cwd")
        dest = Path(cmd[cmd.index("--output-path") + 1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_glb_bytes())
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)

    img = tmp_path / "cutout.png"
    img.write_bytes(b"fake")
    dest = tmp_path / "model.glb"
    out = TripoSGBackend(repo_dir=repo, python="C:/conda/envs/tripoSG/python.exe").generate(img, dest)
    assert out == dest
    assert dest.is_file()
    assert seen["cmd"] == [
        "C:/conda/envs/tripoSG/python.exe",
        "-m",
        "scripts.inference_triposg",
        "--image-input",
        str(img),
        "--output-path",
        str(dest),
    ]
    assert seen["cwd"] == str(repo)


def test_get_backend_names(tmp_path: Path):
    assert get_backend("triposg").name == "triposg"
    with pytest.raises(GenerationError, match="未知"):
        get_backend("nope")


def test_generate_model_glb_with_stub_backend_registers_auto(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    img = tmp_path / "cutout.png"
    img.write_bytes(b"fake-png")
    report = generate_model_glb(img, folder, backend=StubBackend())
    assert report["source"] == "auto"
    assert report["backend"] == "stub"
    assert Path(report["path"]).is_file()
    meta = load_meta(folder)
    assert meta is not None
    assert meta["model"]["source"] == "auto"
    assert meta["model"]["format"] == "glb"
    assert "不带贴图" in meta["model"]["note"]


def test_generate_model_glb_missing_image_or_folder(tmp_path: Path):
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    with pytest.raises(GenerationError, match="找不到要生成"):
        generate_model_glb(tmp_path / "gone.png", folder, backend=StubBackend())
    img = tmp_path / "cutout.png"
    img.write_bytes(b"fake")
    with pytest.raises(GenerationError, match="找不到人物文件夹"):
        generate_model_glb(img, tmp_path / "characters" / "nobody", backend=StubBackend())


def test_generate_model_glb_registers_time_source_backend_and_assembly(tmp_path: Path):
    # Full registration chain with a stub backend: generate -> validate ->
    # meta.json (path + generated_at + source image + backend) -> assembly.
    folder, _meta = ensure_character_folder(tmp_path, "温念")
    img = folder / "cutout" / "00.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"fake-png")
    report = generate_model_glb(img, folder, backend=StubBackend())
    assert report["backend"] == "stub"
    assert report["generated_at"]
    assert report["source_image"] == "cutout/00.png"
    assert report["assembly_body"] == "model/model.glb"
    meta = load_meta(folder)
    assert meta is not None
    model = meta["model"]
    assert model["path"] == "model/model.glb"
    assert model["backend"] == "stub"
    assert model["generated_at"] == report["generated_at"]
    assert model["source_image"] == "cutout/00.png"
    assert "不带骨骼" in model["note"]
    from recast.core.characters import get_assembly

    assert get_assembly(meta)["body"] == "model/model.glb"


def test_triposg_check_missing_repo_names_next_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Torch + GPU present, repo not cloned: check() must name the missing
    # repo and tell the user the next step (clone + point triposg_repo).
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    ok, reason = TripoSGBackend(repo_dir=None).check()
    assert ok is False
    assert "克隆" in reason
    assert "triposg_repo" in reason
    ok, reason = TripoSGBackend(repo_dir=tmp_path).check()
    assert ok is False
    assert "inference_triposg" in reason


def test_triposg_setup_steps_cover_real_run(tmp_path: Path):
    from recast.assets3d.generate3d import TRIPOSG_SETUP_TEXT

    assert "git clone" in TRIPOSG_SETUP_TEXT
    assert "python=3.10" in TRIPOSG_SETUP_TEXT
    assert "inference_triposg" in TRIPOSG_SETUP_TEXT
    assert "triposg_repo" in TRIPOSG_SETUP_TEXT
    assert "triposg_python" in TRIPOSG_SETUP_TEXT


def test_generate_model_glb_propagates_backend_error(tmp_path: Path):
    class FailingBackend:
        name = "broken"

        def check(self):
            return False, "缺模型"

        def generate(self, image, dest):
            raise GenerationError("缺模型")

    folder, _meta = ensure_character_folder(tmp_path, "温念")
    img = tmp_path / "cutout.png"
    img.write_bytes(b"fake")
    with pytest.raises(GenerationError, match="缺模型"):
        generate_model_glb(img, folder, backend=FailingBackend())
    meta = load_meta(folder)
    assert meta is not None
    assert meta["model"]["source"] == "none", "failed generation must not register a model"
