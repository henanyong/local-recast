from pathlib import Path

from recast.assets3d.avatar import bind_path_for, build_avatar, load_avatar
from recast.core.service import RecastService, asset_flags
from recast.core.constants import AUTH_OWN_GEN


def _face(path: Path) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), (180, 140, 120))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 40, 196, 200), fill=(210, 170, 150))
    img.save(path)
    return path


def test_l2_bound_by_character_id_and_reused(tmp_path: Path):
    photo = _face(tmp_path / "face.jpg")
    cid = "wen_nian"
    dest1 = bind_path_for(cid, tmp_path / "ep1")
    dest2 = bind_path_for(cid, tmp_path / "ep1")  # same character id → same bind path under a shared root
    shared = bind_path_for(cid, tmp_path / "library")
    asset = build_avatar(cid, [photo], shared, display_name="温念")
    assert asset["character_id"] == cid
    assert asset["label"] == "3D头像资产"
    assert asset["production_ready_mesh"] is False
    assert "不是生产级" in asset["honesty"] or "3D头像资产" in asset["honesty"]
    assert Path(asset["dir"], "albedo.png").is_file()
    assert Path(asset["dir"], "landmarks.json").is_file()
    assert Path(asset["dir"], "drive_config.json").is_file()
    assert Path(asset["dir"], "head_card.obj").is_file()
    # reuse across two "episodes": load the same character id asset
    loaded = load_avatar(shared)
    assert loaded and loaded["character_id"] == cid
    again = load_avatar(Path(asset["path"]))
    assert again["character_id"] == cid
    assert again["reusable"] is True


def test_library_flags_l2_not_unbuilt(tmp_path: Path):
    svc = RecastService(tmp_path / "db.db")
    photo = _face(tmp_path / "face.jpg")
    ch = svc.add_character_manual("温念", str(photo))
    svc.confirm_character(ch["id"], authorized=True, auth_type=AUTH_OWN_GEN)
    asset = svc.build_l2(ch["id"], [str(photo)])
    ch2 = svc.store.get_character(ch["id"])
    flags = asset_flags(ch2)
    assert flags["3d"] != "未建设"
    assert flags["3d"] in {"已齐", "偏低"}
