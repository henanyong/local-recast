"""3D preview rendering (recast/assets3d/preview.py).

Two paths, both covered: the pipeline (load GLB -> render -> save PNG)
with a stub renderer, and the honest fallback (missing libraries / bad
GLB / failed render -> None, no exception, no fake image). The real
pyrender path is attempted on a genuine GLB and is allowed to report
None on machines without a working GL context (this headless VM is one:
EGL finds no devices, Xvfb + PyOpenGL array handlers are broken with
numpy 2.x).
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from recast.assets3d.preview import render_model_preview

REAL_GLB = Path("/home/hatch/workspace/test-3d-characters/CesiumMan.glb")


def _glb_bytes() -> bytes:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _stub_renderer(scene, width, height):
    """Pretend renderer: proves the load -> render -> save pipeline."""
    assert scene is not None
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[100:200, 100:200] = (200, 30, 30)
    return arr


def test_render_with_stub_renderer_writes_real_png(tmp_path: Path):
    pytest.importorskip("trimesh")
    assert REAL_GLB.is_file(), "测试用的真 GLB 不见了：~/workspace/test-3d-characters/CesiumMan.glb"
    dest = tmp_path / "preview.png"
    out = render_model_preview(REAL_GLB, dest, renderer=_stub_renderer)
    assert out == dest and dest.is_file()
    from PIL import Image

    with Image.open(dest) as img:
        assert img.format == "PNG"
        assert img.size == (480, 480)


def test_render_returns_none_without_trimesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "trimesh", None)
    dest = tmp_path / "preview.png"
    assert render_model_preview(REAL_GLB, dest, renderer=_stub_renderer) is None
    assert not dest.exists()


def test_render_returns_none_for_broken_glb(tmp_path: Path):
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb at all")
    dest = tmp_path / "preview.png"
    assert render_model_preview(bad, dest, renderer=_stub_renderer) is None
    assert not dest.exists()


def test_render_returns_none_for_missing_file(tmp_path: Path):
    assert render_model_preview(tmp_path / "gone.glb", tmp_path / "preview.png") is None


def test_render_returns_none_when_renderer_fails(tmp_path: Path):
    pytest.importorskip("trimesh")

    def _boom(scene, width, height):
        raise RuntimeError("没有可用的 GL 上下文")

    assert render_model_preview(REAL_GLB, tmp_path / "preview.png", renderer=_boom) is None


def test_real_renderer_attempt_never_raises_or_fakes(tmp_path: Path):
    """The genuine pyrender path on a real GLB.

    On a machine with a working GL context this produces a PNG; on this
    headless VM it returns None. Either way it must not raise and must
    never write a corrupt file.
    """
    pytest.importorskip("trimesh")
    dest = tmp_path / "preview.png"
    out = render_model_preview(REAL_GLB, dest)
    if out is None:
        assert not dest.exists()
    else:
        from PIL import Image

        with Image.open(dest) as img:
            assert img.format == "PNG" and img.size == (480, 480)
