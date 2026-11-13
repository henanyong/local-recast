"""B0-2 UI: the review queue's "显示低质量样本" checkbox.

Synthetic images + stub service data only. No real video, no detection model.
"""

import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import Qt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _solid(path: Path, size: int = 140) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (200, 30, 30)).save(path)
    return path


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


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


def _library(svc):
    from recast.desktop.library_window import LibraryWindow

    _app()
    return LibraryWindow(svc)


def _data_rows(window):
    """Review rows that are real queue items (group headers excluded)."""
    out = []
    for row in range(window.review_list.count()):
        item = window.review_list.item(row)
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and not data.get("review_header"):
            out.append(item)
    return out


def _setup_mixed_queue(tmp_path: Path, svc):
    rejected = _solid(tmp_path / "works" / "rejected.png")
    ordinary = _solid(tmp_path / "works" / "ordinary.png")
    _add_character(
        svc, "温念",
        samples=[(rejected, "待审核：模糊"), (ordinary, "聚类样本")],
    )
    return rejected, ordinary


def test_checkbox_exists_and_unchecked_by_default(tmp_path: Path, svc):
    _setup_mixed_queue(tmp_path, svc)
    window = _library(svc)
    assert window.review_show_rejected is not None
    assert window.review_show_rejected.text() == "显示低质量样本"
    assert not window.review_show_rejected.isChecked()
    window.close()


def test_checkbox_toggles_rejected_rows(tmp_path: Path, svc):
    rejected, ordinary = _setup_mixed_queue(tmp_path, svc)
    window = _library(svc)
    # Default: only the ordinary low-score sample is shown.
    rows = _data_rows(window)
    assert len(rows) == 1
    assert rows[0].data(Qt.ItemDataRole.UserRole)["thumb"] == str(ordinary)
    # Check: the rejected sample reappears.
    window.review_show_rejected.setChecked(True)
    rows = _data_rows(window)
    assert len(rows) == 2
    assert {r.data(Qt.ItemDataRole.UserRole)["thumb"] for r in rows} == {
        str(rejected), str(ordinary),
    }
    # Uncheck: back to one row.
    window.review_show_rejected.setChecked(False)
    rows = _data_rows(window)
    assert len(rows) == 1
    assert rows[0].data(Qt.ItemDataRole.UserRole)["thumb"] == str(ordinary)
    window.close()


def test_checkbox_refresh_passes_include_rejected(tmp_path: Path, svc, monkeypatch):
    _setup_mixed_queue(tmp_path, svc)
    window = _library(svc)
    calls: list[dict] = []
    orig = svc.list_review_queue

    def _spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return orig(*args, **kwargs)

    monkeypatch.setattr(svc, "list_review_queue", _spy)
    calls.clear()
    window.review_show_rejected.setChecked(True)
    assert calls and calls[-1].get("include_rejected") is True
    calls.clear()
    window.review_show_rejected.setChecked(False)
    assert calls and calls[-1].get("include_rejected") is False
    window.close()
