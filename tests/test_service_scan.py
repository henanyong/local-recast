from __future__ import annotations

from pathlib import Path

from recast.core.constants import ST_NEED_ASSETS, ST_PENDING
from recast.core.service import RecastService


def test_service_scan_marks_named_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "works"
    drama = root / "示例剧"
    drama.mkdir(parents=True)
    (drama / "第01集.mp4").write_bytes(b"x")
    (drama / "参考图").mkdir()
    (drama / "名单.yaml").write_text("5: 沈黛\n6: 男主\n", encoding="utf-8")
    svc = RecastService(tmp_path / "store.db")
    snap = svc.scan(root)
    assert len(snap["dramas"]) == 1
    row = snap["dramas"][0]
    assert row["status"] == ST_NEED_ASSETS
    assert row["progress"] == "缺少 5.png、6.png"
    # unchecking must not delete files
    svc.set_selected(row["id"], False)
    assert (drama / "第01集.mp4").is_file()
    # after adding faces and recheck, status becomes 待执行
    (drama / "参考图" / "5.png").write_bytes(b"img")
    (drama / "参考图" / "6.png").write_bytes(b"img")
    snap = svc.recheck_materials([row["id"]])
    row = snap["dramas"][0]
    assert row["status"] == ST_PENDING
    assert "缺少" not in (row.get("progress") or "")


def test_clear_completed_only_removes_queue_rows(tmp_path: Path) -> None:
    root = tmp_path / "works" / "剧"
    root.mkdir(parents=True)
    video = root / "a.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "outputs" / "a" / "output.mp4"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"done")
    svc = RecastService(tmp_path / "store.db")
    svc.scan(root.parent)
    drama = svc.store.list_dramas()[0]
    svc.store.update("dramas", drama["id"], status="已完成", output_dir=str(out.parent))
    svc.clear_completed()
    assert svc.store.list_dramas() == []
    assert out.is_file()
