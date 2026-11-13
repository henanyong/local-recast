"""B0-2: rejected (held-back) samples are hidden from the review queue by
default and reappear with include_rejected=True.

Synthetic images only: a sharp checkerboard scores high, a solid colour
scores low. No real video, no detection model.
"""

from pathlib import Path

import pytest
from PIL import Image

from recast.core.review import REVIEW_TRASH_DIR, build_review_queue
from recast.core.service import RecastService


def _checker(path: Path, size: int = 140) -> Path:
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if (x // 7 + y // 7) % 2 else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)
    return path


def _solid(path: Path, size: int = 140) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (200, 30, 30)).save(path)
    return path


class _Store:
    """Minimal stand-in for the real Store: only list_characters."""

    def __init__(self, characters):
        self._characters = characters

    def list_characters(self):
        return self._characters


def _character(name: str, *, confirmed: int = 1, assets: list[dict] | None = None) -> dict:
    return {
        "id": f"c-{name}",
        "name": name,
        "display_name": name,
        "confirmed": confirmed,
        "assets": assets or [],
    }


def _rejected_character(tmp_path: Path, bad: Path) -> dict:
    """Confirmed character whose only face asset is a held-back sample
    (notes carry the extraction reason, as _mark_rejected_assets writes)."""
    return _character(
        "温念",
        assets=[{"kind": "face", "path": str(bad), "notes": "待审核：模糊"}],
    )


# ----- 默认排除：store 资产（notes 带“待审核：”） -----

def test_default_excludes_rejected_store_asset(tmp_path: Path):
    out = tmp_path / "out"
    bad = _solid(tmp_path / "bad.png")
    items = build_review_queue(_Store([_rejected_character(tmp_path, bad)]), out)
    assert items == []
    shown = build_review_queue(
        _Store([_rejected_character(tmp_path, bad)]), out, include_rejected=True
    )
    assert len(shown) == 1
    assert shown[0].kind == "sample"
    assert shown[0].thumb == str(bad)


def test_rejected_store_asset_with_good_score_still_excluded_without_flag(tmp_path: Path):
    # A rejected crop that happens to score >= 0.5 never queues either way:
    # the flag only controls whether rejected rows are *considered*, the
    # score bar still applies when they are.
    out = tmp_path / "out"
    good = _checker(tmp_path / "good.png")
    items = build_review_queue(
        _Store([_rejected_character(tmp_path, good)]), out, include_rejected=True
    )
    assert items == []


# ----- 默认排除：文件夹扫描（文件名 rejected_ 开头） -----

def test_default_excludes_rejected_folder_file(tmp_path: Path):
    from recast.core.characters import character_folder

    out = tmp_path / "out"
    face_dir = character_folder(out, "温念") / "face"
    bad = _solid(face_dir / "rejected_模糊_001.jpg")
    # No store row at all: the file is found only by the folder scan.
    items = build_review_queue(_Store([_character("温念")]), out)
    assert items == []
    shown = build_review_queue(
        _Store([_character("温念")]), out, include_rejected=True
    )
    assert len(shown) == 1
    assert shown[0].kind == "sample"
    assert shown[0].thumb == str(bad)


def test_folder_file_named_rejected_but_high_score_stays_out(tmp_path: Path):
    # Same score-bar rule as the store case: a sharp rejected_ file does
    # not queue even with the flag on.
    from recast.core.characters import character_folder

    out = tmp_path / "out"
    face_dir = character_folder(out, "温念") / "face"
    _checker(face_dir / "rejected_侧脸_001.jpg")
    assert build_review_queue(
        _Store([_character("温念")]), out, include_rejected=True
    ) == []


# ----- 不受开关影响的两类：普通低分样本、纠错移出样本 -----

def test_ordinary_low_score_sample_still_queued_by_default(tmp_path: Path):
    out = tmp_path / "out"
    bad = _solid(tmp_path / "bad.png")
    ch = _character(
        "温念",
        assets=[{"kind": "face", "path": str(bad), "notes": "聚类样本"}],
    )
    items = build_review_queue(_Store([ch]), out)
    assert len(items) == 1
    assert items[0].kind == "sample"
    assert items[0].thumb == str(bad)


def test_correction_trash_sample_still_queued_by_default(tmp_path: Path):
    out = tmp_path / "out"
    trash = out / REVIEW_TRASH_DIR
    bad = _solid(trash / "moved.png")
    # Even a rejected-looking file in review_trash/ is a correction-moved
    # sample, not a held-back one: it must always queue.
    _solid(trash / "rejected_模糊_002.jpg")
    items = build_review_queue(_Store([_character("温念")]), out)
    assert len(items) == 2
    assert {i.character_name for i in items} == {"待审核（已移出）"}
    assert {i.thumb for i in items} == {str(bad), str(trash / "rejected_模糊_002.jpg")}


def test_mixed_queue_default_vs_include_rejected(tmp_path: Path):
    # Daniel's screenshot in miniature: 1 unconfirmed cluster + 1
    # ordinary low sample + 1 rejected sample. Default hides only the
    # rejected one (3 -> 2); the flag restores it (2 -> 3).
    out = tmp_path / "out"
    bad = _solid(tmp_path / "bad.png")
    rejected = _solid(tmp_path / "rejected.png")
    chars = [
        _character("人物1", confirmed=0),
        _character(
            "温念",
            assets=[
                {"kind": "face", "path": str(bad), "notes": "聚类样本"},
                {"kind": "face", "path": str(rejected), "notes": "待审核：太小"},
            ],
        ),
    ]
    default = build_review_queue(_Store(chars), out)
    assert [i.kind for i in default] == ["cluster", "sample"]
    assert default[1].thumb == str(bad)
    shown = build_review_queue(_Store(chars), out, include_rejected=True)
    assert len(shown) == 3
    assert shown[2].thumb == str(rejected)


# ----- service 直通 -----

@pytest.fixture()
def svc(tmp_path: Path):
    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


def _add_character(svc, name: str, *, confirmed: int = 1, samples: list[tuple[Path, str]] | None = None):
    ch = svc.store.insert(
        "characters", name=name, display_name=name, confirmed=confirmed,
        authorized=0, cluster_key=f"test:{name}",
    )
    for p, notes in samples or []:
        svc.store.insert(
            "character_assets", character_id=ch["id"], kind="face",
            path=str(p), complete=1, notes=notes,
        )
    return svc.store.get_character(ch["id"])


def test_service_list_review_queue_passthrough(tmp_path: Path, svc):
    rejected = _solid(tmp_path / "works" / "rejected.png")
    ordinary = _solid(tmp_path / "works" / "ordinary.png")
    _add_character(
        svc, "温念",
        samples=[(rejected, "待审核：模糊"), (ordinary, "聚类样本")],
    )
    default = svc.list_review_queue()
    assert len(default) == 1
    assert default[0]["thumb"] == str(ordinary)
    shown = svc.list_review_queue(include_rejected=True)
    assert len(shown) == 2
    assert {row["thumb"] for row in shown} == {str(rejected), str(ordinary)}
    # Explicit False behaves like the default.
    assert svc.list_review_queue(include_rejected=False) == default


def test_service_default_empty_when_only_rejected(tmp_path: Path, svc):
    rejected = _solid(tmp_path / "works" / "rejected.png")
    _add_character(svc, "温念", samples=[(rejected, "待审核：太小")])
    assert svc.list_review_queue() == []
    assert len(svc.list_review_queue(include_rejected=True)) == 1
