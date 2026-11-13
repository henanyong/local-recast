"""Matting: transparent PNG output, missing-model errors, per-file failures."""

from pathlib import Path

import pytest
from PIL import Image

from recast.core.characters import ensure_character_folder
from recast.stages import matting
from recast.stages.matting import MattingError, matte_character_folder, matte_image


def _image(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 120, 90)).save(path)
    return path


def _fake_rgba() -> Image.Image:
    """Left half transparent, right half opaque: proves the alpha channel survives."""
    img = Image.new("RGBA", (64, 64), (200, 120, 90, 255))
    for x in range(32):
        for y in range(64):
            img.putpixel((x, y), (200, 120, 90, 0))
    return img


@pytest.fixture
def stub_rembg(monkeypatch):
    """Stand in for rembg so tests never need the real model."""

    class FakeModule:
        def new_session(self, model):
            return object()

    monkeypatch.setattr(matting, "_load_rembg", lambda: FakeModule())
    monkeypatch.setattr(matting, "_new_session", lambda rembg, model, models_dir: object())
    monkeypatch.setattr(matting, "_remove_image", lambda rembg, img, session: _fake_rgba())
    return FakeModule()


def test_matte_image_writes_rgba_png(tmp_path: Path, stub_rembg):
    src = _image(tmp_path / "photo.jpg")
    dest = tmp_path / "cutout" / "photo.png"
    out = matte_image(src, dest)
    assert out == dest
    result = Image.open(dest)
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((63, 0))[3] == 255


def test_matte_image_missing_file(tmp_path: Path, stub_rembg):
    with pytest.raises(MattingError, match="找不到要抠像的图片"):
        matte_image(tmp_path / "nope.jpg", tmp_path / "out.png")


def test_matte_image_missing_rembg_reports_install(tmp_path: Path, monkeypatch):
    def _boom():
        raise MattingError("抠像需要 rembg，未安装。请先安装：pip install rembg onnxruntime")

    monkeypatch.setattr(matting, "_load_rembg", _boom)
    src = _image(tmp_path / "photo.jpg")
    with pytest.raises(MattingError, match="pip install rembg"):
        matte_image(src, tmp_path / "out.png")


def test_matte_image_missing_model_reports_download(tmp_path: Path, monkeypatch):
    class FakeModule:
        def new_session(self, model):
            raise RuntimeError("network down")

    monkeypatch.setattr(matting, "_load_rembg", lambda: FakeModule())
    src = _image(tmp_path / "photo.jpg")
    with pytest.raises(MattingError, match="手动下载模型文件"):
        matte_image(src, tmp_path / "out.png")


def test_matte_character_folder_and_skip(tmp_path: Path, stub_rembg):
    folder, _meta = ensure_character_folder(tmp_path / "outputs", "温念")
    _image(folder / "face" / "00.jpg")
    _image(folder / "face" / "01.jpg")
    report = matte_character_folder(folder)
    assert len(report["outputs"]) == 2, report
    assert report["failed"] == []
    assert (folder / "cutout" / "00.png").is_file()
    assert (folder / "cutout" / "01.png").is_file()
    # Second run skips what is already done.
    again = matte_character_folder(folder)
    assert again["outputs"] == []
    assert len(again["skipped"]) == 2


def test_matte_character_folder_empty_face(tmp_path: Path, stub_rembg):
    folder, _meta = ensure_character_folder(tmp_path / "outputs", "温念")
    report = matte_character_folder(folder)
    assert report["outputs"] == []
    assert "没有可抠像的图片" in report["failed"][0]["reason"]
