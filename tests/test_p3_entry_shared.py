"""P3 batch, item 4: recast/core/entry_shared.py — ENTRY_STAGES single
source + check_replace_auth pure logic.

No Qt, no video, no network: real RecastService on a tmp sqlite db.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def svc(tmp_path: Path):
    from recast.core.service import RecastService

    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


def _add_character(svc, name: str, *, confirmed: int = 1, authorized: int = 0, auth_type: str | None = None):
    fields: dict = {
        "name": name, "display_name": name, "confirmed": confirmed,
        "authorized": authorized, "cluster_key": f"test:{name}",
    }
    if auth_type:
        fields["auth_type"] = auth_type
    return svc.store.insert("characters", **fields)


def _real_face(tmp_path: Path, name: str = "face.jpg") -> Path:
    """A real readable JPEG (200x200 checkerboard): scores ~1.0."""
    from PIL import Image

    img = Image.new("L", (200, 200))
    px = img.load()
    for y in range(200):
        for x in range(200):
            px[x, y] = 255 if (x // 10 + y // 10) % 2 == 0 else 0
    path = tmp_path / name
    img.convert("RGB").save(path, "JPEG")
    return path


def _add_face_asset(svc, character_id: str, path: str):
    return svc.store.insert(
        "character_assets", character_id=character_id, kind="face", path=path,
    )


# ----- ENTRY_STAGES：唯一来源 -----


def test_entry_stages_content():
    from recast.core.entry_shared import ENTRY_STAGES

    assert ENTRY_STAGES == {
        "import": frozenset({"scan", "extract", "library", "replace", "compose", "export"}),
        "script": frozenset({"template", "generate", "dub", "compose", "export"}),
    }
    assert "replace" in ENTRY_STAGES["import"]
    assert "replace" not in ENTRY_STAGES["script"]
    assert "generate" in ENTRY_STAGES["script"]


def test_main_window_imports_shared_entry_stages():
    """主窗口不再留自己的拷贝：import 进来的必须是同一个对象。"""
    import recast.core.entry_shared as shared
    import recast.desktop.main_window as mw

    assert mw.ENTRY_STAGES is shared.ENTRY_STAGES
    assert not hasattr(mw, "_ENTRY_STAGES")


# ----- check_replace_auth -----


def test_check_replace_auth_counts_authorized_and_lists_unauthorized(svc, tmp_path: Path):
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    ok = _add_character(svc, "温雪", authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    bad = _add_character(svc, "温念", authorized=0)
    _add_face_asset(svc, bad["id"], str(face_ok))
    # 有授权但没参考脸：两边都不算。
    _add_character(svc, "路人", authorized=1, auth_type="自有拍摄")
    # 没授权也没参考脸：两边都不算。
    _add_character(svc, "阿丙", authorized=0)

    result = check_replace_auth(svc, ["drama-1"])
    assert result["authorized_count"] == 1
    assert [m["name"] for m in result["unauthorized"]] == ["温念"]
    assert all("name" in m for m in result["unauthorized"])


def test_check_replace_auth_empty_library(svc):
    from recast.core.entry_shared import check_replace_auth

    result = check_replace_auth(svc)
    assert result == {"authorized_count": 0, "unauthorized": []}


def test_check_replace_auth_face_file_missing_not_counted(svc, tmp_path: Path):
    """边界：授权了、登记了参考脸路径，但文件已不在磁盘上——不算已授权。"""
    from recast.core.entry_shared import check_replace_auth

    ch = _add_character(svc, "温雪", authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ch["id"], str(tmp_path / "gone.jpg"))
    result = check_replace_auth(svc)
    assert result["authorized_count"] == 0
    assert result["unauthorized"] == []


def test_check_replace_auth_accepts_drama_ids(svc, tmp_path: Path):
    """drama_ids 目前不参与计算（授权是人物库全局的），但签名接受它，
    将来按剧目收窄时调用方不用改。"""
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    ok = _add_character(svc, "温雪", authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    assert check_replace_auth(svc, ["d1"]) == check_replace_auth(svc, None)
    assert check_replace_auth(svc, [])["authorized_count"] == 1


def test_check_replace_auth_unauthorized_items_carry_identity(svc, tmp_path: Path):
    """未授权项至少含 name，且 id / face_paths 透传 service 的结果。"""
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    ch = _add_character(svc, "温念", authorized=0)
    _add_face_asset(svc, ch["id"], str(face_ok))
    (item,) = check_replace_auth(svc)["unauthorized"]
    assert item["name"] == "温念"
    assert item["id"] == ch["id"]
    assert item["face_paths"] == [str(face_ok)]
