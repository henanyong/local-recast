"""Delete and reclaim L2 3D头像资产 without touching anything outside that tree."""

from pathlib import Path

import pytest

from recast.assets3d.avatar import (
    L2PathError,
    bind_path_for,
    build_avatar,
    legacy_bind_path_for,
    locate_avatar_dir,
    require_l2_asset_dir,
)
from recast.core.constants import AUTH_OWN_GEN
from recast.core.service import RecastService, asset_flags


def _face(path: Path) -> Path:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 256), (180, 140, 120))
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 40, 196, 200), fill=(210, 170, 150))
    img.save(path)
    return path


def _service(tmp_path: Path) -> RecastService:
    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _authorize(svc: RecastService, name: str, photo: Path) -> dict:
    ch = svc.add_character_manual(name, str(photo))
    svc.confirm_character(ch["id"], authorized=True, auth_type=AUTH_OWN_GEN)
    return svc.store.get_character(ch["id"])


def test_build_l2_writes_canonical_directory(tmp_path: Path):
    svc = _service(tmp_path)
    photo = _face(tmp_path / "face.jpg")
    ch = _authorize(svc, "温念", photo)
    asset = svc.build_l2(ch["id"], [str(photo)])
    canonical = bind_path_for(ch["id"], svc.settings.output_root).resolve()
    legacy = legacy_bind_path_for(ch["id"], svc.settings.output_root)
    assert Path(asset["dir"]).resolve() == canonical
    assert (canonical / "asset.json").is_file()
    assert not legacy.exists()
    assert locate_avatar_dir(ch["id"], svc.settings.output_root).resolve() == canonical


def test_lookup_accepts_legacy_doubled_path(tmp_path: Path):
    svc = _service(tmp_path)
    photo = _face(tmp_path / "face.jpg")
    ch = _authorize(svc, "温念", photo)
    legacy = legacy_bind_path_for(ch["id"], svc.settings.output_root)
    asset = build_avatar(ch["id"], [photo], legacy, display_name="温念")
    svc.store.set_character_asset(ch["id"], "3d", asset["path"], complete=True)

    found = locate_avatar_dir(ch["id"], svc.settings.output_root)
    assert found is not None and found.resolve() == legacy.resolve()
    flags = asset_flags(svc.store.get_character(ch["id"]))
    assert flags["3d"] in {"已齐", "偏低"}
    assert svc.list_unused_l2() == []

    # A later canonical copy is preferred, and the old folder becomes reclaimable.
    canonical = bind_path_for(ch["id"], svc.settings.output_root)
    fresh = build_avatar(ch["id"], [photo], canonical, display_name="温念")
    svc.store.set_character_asset(ch["id"], "3d", fresh["path"], complete=True)
    assert locate_avatar_dir(ch["id"], svc.settings.output_root).resolve() == canonical.resolve()
    unused = svc.list_unused_l2()
    assert [Path(item["path"]).resolve() for item in unused] == [legacy.resolve()]
    assert "l2/l2" in unused[0]["reason"]


def test_delete_l2_removes_files_and_row_keeps_face(tmp_path: Path):
    svc = _service(tmp_path)
    photo = _face(tmp_path / "face.jpg")
    costume = tmp_path / "costume.png"
    costume.write_bytes(b"png")
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"wav")
    ch = _authorize(svc, "温念", photo)
    svc.store.set_character_asset(ch["id"], "costume", str(costume), complete=True)
    svc.store.set_character_asset(ch["id"], "voice", str(voice), complete=True)
    asset = svc.build_l2(ch["id"], [str(photo)])

    result = svc.delete_l2(ch["id"])
    assert result["status"] == "缺失"
    assert not Path(asset["dir"]).exists()
    saved = svc.store.get_character(ch["id"])
    assert not any(row.get("kind") in {"3d", "3d_avatar"} for row in saved["assets"])
    flags = asset_flags(saved)
    assert flags["3d"] == "缺失"
    assert flags["face"] == "已齐"
    assert flags["costume"] == "已齐"
    assert flags["voice"] == "已齐"
    assert photo.is_file() and costume.is_file() and voice.is_file()

    again = svc.build_l2(ch["id"], [str(photo)])
    assert Path(again["dir"]).resolve() == bind_path_for(ch["id"], svc.settings.output_root).resolve()
    assert asset_flags(svc.store.get_character(ch["id"]))["3d"] in {"已齐", "偏低"}


def test_delete_l2_partial_failure_keeps_row_and_reports(tmp_path: Path, monkeypatch):
    svc = _service(tmp_path)
    photo = _face(tmp_path / "face.jpg")
    ch = _authorize(svc, "温念", photo)
    asset = svc.build_l2(ch["id"], [str(photo)])
    asset_dir = Path(asset["dir"])
    files = sorted(p for p in asset_dir.rglob("*") if p.is_file())
    assert files
    locked = files[0]

    real_unlink = Path.unlink

    def _flaky(self, missing_ok=False):
        if self.resolve() == locked.resolve():
            raise PermissionError("文件被占用")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _flaky)
    result = svc.delete_l2(ch["id"])
    assert result["partial"] is True
    assert result["status"] == "部分删除"
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"] == str(asset_dir)
    assert result["failed"][0]["reason"] == "文件被占用"
    assert locked.is_file()
    saved = svc.store.get_character(ch["id"])
    assert any(row.get("kind") in {"3d", "3d_avatar"} for row in saved["assets"])
    assert photo.is_file()

    monkeypatch.undo()
    result = svc.delete_l2(ch["id"])
    assert result["partial"] is False
    assert result["status"] == "缺失"
    assert not asset_dir.exists()


def test_delete_l2_refuses_paths_outside_l2_tree(tmp_path: Path):
    svc = _service(tmp_path)
    photo = _face(tmp_path / "face.jpg")
    ch = _authorize(svc, "温念", photo)
    output = Path(svc.settings.output_root)
    models = tmp_path / "models" / "flame"
    models.mkdir(parents=True)
    weight = models / "generic_model.pkl"
    weight.write_bytes(b"weight")
    venv_file = tmp_path / ".venv" / "lib" / "stay.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("x", encoding="utf-8")
    other = output / "exports" / "ep1" / "final.mp4"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"video")

    svc.store.set_character_asset(ch["id"], "3d", str(weight), complete=True)
    with pytest.raises(PermissionError, match="拒绝删除"):
        svc.delete_l2(ch["id"])
    assert weight.is_file()
    saved = svc.store.get_character(ch["id"])
    assert any(row.get("kind") == "3d" and row.get("path") == str(weight) for row in saved["assets"])
    assert photo.is_file()

    for outside in (venv_file, other, models, output / "l3" / "prop"):
        with pytest.raises(L2PathError, match="拒绝删除"):
            require_l2_asset_dir(outside, output)

    link_target = tmp_path / "models" / "linked"
    link_target.mkdir()
    (link_target / "secret.bin").write_text("keep", encoding="utf-8")
    link = output / "l2" / "escape"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(link_target, target_is_directory=True)
    except OSError:
        # Windows without admin rights or Developer Mode cannot create
        # symlinks (WinError 1314): skip instead of failing the suite.
        pytest.skip("创建符号链接需要管理员权限或 Windows 开发者模式，跳过。")
    svc.store.set_character_asset(ch["id"], "3d", str(link / "asset.json"), complete=True)
    with pytest.raises(PermissionError, match="拒绝删除"):
        svc.delete_l2(ch["id"])
    assert (link_target / "secret.bin").is_file()
    assert link.is_symlink()


def test_delete_unused_l2_removes_orphans_and_keeps_referenced(tmp_path: Path):
    svc = _service(tmp_path)
    photo = _face(tmp_path / "face.jpg")
    output = Path(svc.settings.output_root)
    kept = _authorize(svc, "温念", photo)
    costume = tmp_path / "costume.png"
    costume.write_bytes(b"png")
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"wav")
    svc.store.set_character_asset(kept["id"], "costume", str(costume), complete=True)
    svc.store.set_character_asset(kept["id"], "voice", str(voice), complete=True)
    kept_asset = svc.build_l2(kept["id"], [str(photo)])

    legacy = legacy_bind_path_for(kept["id"], output)
    build_avatar(kept["id"], [photo], legacy, display_name="温念旧档")

    gone = _authorize(svc, "陈秋微", photo)
    gone_asset = svc.build_l2(gone["id"], [str(photo)])
    svc.store.delete("characters", gone["id"])
    assert Path(gone_asset["dir"]).is_dir()

    stray = svc.add_character_manual("路人", str(photo))
    stray_dir = bind_path_for(stray["id"], output)
    build_avatar(stray["id"], [photo], stray_dir, display_name="路人")

    models = tmp_path / "models" / "flame" / "generic_model.pkl"
    models.parent.mkdir(parents=True)
    models.write_bytes(b"weight")
    venv_file = tmp_path / ".venv" / "lib" / "stay.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("x", encoding="utf-8")
    other = output / "exports" / "ep1" / "final.mp4"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"video")
    l3 = output / "l3" / "prop" / "mesh.obj"
    l3.parent.mkdir(parents=True)
    l3.write_text("o Prop\n", encoding="utf-8")

    unused = svc.list_unused_l2()
    by_path = {Path(item["path"]).resolve(): item for item in unused}
    assert legacy.resolve() in by_path
    assert Path(gone_asset["dir"]).resolve() in by_path
    assert stray_dir.resolve() in by_path
    assert Path(kept_asset["dir"]).resolve() not in by_path
    assert by_path[legacy.resolve()]["display_name"] == "温念旧档"
    assert "l2/l2" in by_path[legacy.resolve()]["reason"]
    assert by_path[Path(gone_asset["dir"]).resolve()]["name"] == "陈秋微"
    assert "不存在" in by_path[Path(gone_asset["dir"]).resolve()]["reason"]
    assert by_path[stray_dir.resolve()]["name"] == "路人"
    assert by_path[stray_dir.resolve()]["reason"] == "未被任何人物引用"

    # A path outside the tree is not a cleanup target, even if the caller names it.
    skipped = svc.delete_unused_l2(only=[str(models)])
    assert skipped["count"] == 0
    assert legacy.is_dir() and models.is_file()

    result = svc.delete_unused_l2()
    assert result["count"] == 3
    assert {item["character_id"] for item in result["removed"]} == {kept["id"], gone["id"], stray["id"]}
    assert not legacy.exists()
    assert not Path(gone_asset["dir"]).exists()
    assert not stray_dir.exists()
    assert (Path(kept_asset["dir"]) / "asset.json").is_file()
    assert photo.is_file() and costume.is_file() and voice.is_file()
    assert models.is_file() and venv_file.is_file() and other.is_file() and l3.is_file()
    flags = asset_flags(svc.store.get_character(kept["id"]))
    assert flags["3d"] in {"已齐", "偏低"}
    assert flags["face"] == "已齐"
    assert flags["costume"] == "已齐"
    assert flags["voice"] == "已齐"
    assert svc.list_unused_l2() == []


def test_cleanup_keeps_face_file_stored_inside_orphan_dir(tmp_path: Path):
    svc = _service(tmp_path)
    ch = svc.add_character_manual("温念", None)
    folder = bind_path_for("orphan_face", svc.settings.output_root)
    folder.mkdir(parents=True)
    face = _face(folder / "kept-face.jpg")
    (folder / "asset.json").write_text('{"label": "3D头像资产"}', encoding="utf-8")
    (folder / "head_card.obj").write_text("o Head\n", encoding="utf-8")
    svc.store.set_character_asset(ch["id"], "face", str(face), complete=True)

    svc.delete_unused_l2()
    assert face.is_file()
    assert not (folder / "asset.json").exists()
    assert not (folder / "head_card.obj").exists()
    saved = svc.store.get_character(ch["id"])
    assert any(row.get("kind") == "face" and Path(row["path"]).is_file() for row in saved["assets"])


def test_bind_path_rejects_path_segments():
    with pytest.raises(L2PathError):
        bind_path_for("../models", "/tmp/outputs")
    with pytest.raises(L2PathError):
        bind_path_for("a/b", "/tmp/outputs")


# ----- 资料完整度 must agree with the folder, not only with the index -----


def test_asset_flags_reads_the_folder_when_the_db_has_no_rows(tmp_path):
    """A character imported by copying its folder has files but no asset rows.

    The panel used to print "脸：缺失" while showing the face photo one column
    over. Every pipeline stage reads the folder, so the folder decides.
    """
    from recast.core.characters import character_folder

    svc = _service(tmp_path)
    ch = svc.store.insert(
        "characters", name="林知夏", display_name="林知夏", confirmed=1, cluster_key="c:1"
    )
    folder = character_folder(svc.settings.output_root, "林知夏")
    for sub in ("face", "costume", "voice"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    _face(folder / "face" / "01.png")
    _face(folder / "costume" / "01.png")

    row = svc.store.get_character(ch["id"])
    on_disk = asset_flags(row, output_root=svc.settings.output_root)
    assert on_disk["face"] == "已齐"
    assert on_disk["costume"] == "已齐"
    assert on_disk["voice"] == "缺失", "没有样本就应当仍报缺失"

    db_only = asset_flags(row)
    assert db_only["face"] == "缺失", "不传 output_root 时保持原来的索引语义"


def test_asset_flags_prefers_a_real_row_path(tmp_path):
    svc = _service(tmp_path)
    photo = _face(tmp_path / "p.png")
    ch = svc.add_character_manual("温念", str(photo))
    row = svc.store.get_character(ch["id"])
    # The row points at a real file, so the answer is 已齐 either way.
    assert asset_flags(row, output_root=svc.settings.output_root)["face"] == "已齐"
