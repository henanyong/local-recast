"""Folder-level merge completion: embedding mirror + migrate merge.

Covers recast/core/merge.py::resolve_output_root / sync_folder_embeddings
and recast/core/characters.py::migrate_to_character_folders (embedding
comparison + store-driven folder resolution). Pure logic + tmp folders;
no real video, no InsightFace.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from recast.core.merge import resolve_output_root, sync_folder_embeddings


def _vec(*values: float) -> list[float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr / np.linalg.norm(arr)
    return [float(x) for x in arr]


def _store(path: Path):
    from recast.store.db import Store

    return Store(path)


def _character_row(store, name: str, embedding: list[float] | None, *, cluster_key: str = ""):
    return store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=0,
        authorized=0,
        cluster_key=cluster_key or f"cluster:{name}",
        face_embedding=json.dumps(embedding) if embedding is not None else None,
    )


# ----- resolve_output_root -----


def test_resolve_output_root_explicit_wins(tmp_path: Path):
    assert resolve_output_root(None, tmp_path / "out") == tmp_path / "out"


def test_resolve_output_root_from_default_store_layout(tmp_path: Path):
    store = _store(tmp_path / "outputs" / "store" / "recast.db")
    assert resolve_output_root(store) == tmp_path / "outputs"


def test_resolve_output_root_unknown_layout_returns_none(tmp_path: Path):
    store = _store(tmp_path / "weird-place" / "db.db")
    assert resolve_output_root(store) is None
    assert resolve_output_root(None) is None


# ----- sync_folder_embeddings -----


def test_sync_writes_mirror_into_existing_folder(tmp_path: Path):
    from recast.core.characters import ensure_character_folder, load_meta

    out = tmp_path / "outputs"
    store = _store(out / "store" / "recast.db")
    emb = _vec(1.0, 0.0, 0.0)
    _character_row(store, "温念", emb)
    ensure_character_folder(out, "温念")

    report = sync_folder_embeddings(store)
    assert report["error"] is None
    assert report["written"] == ["温念"]
    meta = load_meta(out / "characters" / "温念")
    assert meta is not None
    assert np.allclose(meta["face_embedding"], emb)


def test_sync_skips_when_already_in_sync(tmp_path: Path):
    from recast.core.characters import ensure_character_folder, load_meta

    out = tmp_path / "outputs"
    store = _store(out / "store" / "recast.db")
    emb = _vec(0.0, 1.0, 0.0)
    _character_row(store, "温念", emb)
    folder, _ = ensure_character_folder(out, "温念")
    meta = load_meta(folder)
    meta["face_embedding"] = list(emb)
    from recast.core.characters import save_meta

    save_meta(folder, meta)

    report = sync_folder_embeddings(store)
    assert report["written"] == []
    assert report["skipped"] == [{"name": "温念", "reason": "已一致"}]


def test_sync_skips_missing_folder_and_missing_embedding(tmp_path: Path):
    out = tmp_path / "outputs"
    store = _store(out / "store" / "recast.db")
    _character_row(store, "没文件夹的人", _vec(1.0, 1.0, 0.0))
    _character_row(store, "没 embedding 的人", None)

    report = sync_folder_embeddings(store)
    assert report["error"] is None
    assert report["written"] == []
    reasons = {s["name"]: s["reason"] for s in report["skipped"]}
    assert reasons["没文件夹的人"] == "没有人物文件夹"
    assert reasons["没 embedding 的人"] == "库里没有可用的 face_embedding"


def test_sync_reports_error_when_root_cannot_be_derived(tmp_path: Path):
    store = _store(tmp_path / "weird-place" / "db.db")
    _character_row(store, "温念", _vec(1.0, 0.0, 0.0))
    report = sync_folder_embeddings(store)
    assert report["written"] == []
    assert report["error"] is not None
    assert "推导不出输出根目录" in report["error"]


# ----- migrate_to_character_folders: folder-level merge -----


def _sample_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-jpg-bytes-0123456789")
    return path


def test_migrate_merges_by_embedding_into_existing_folder(tmp_path: Path):
    from recast.core.characters import (
        ensure_character_folder,
        load_meta,
        migrate_to_character_folders,
    )

    out = tmp_path / "outputs"
    folder, _ = ensure_character_folder(out, "温念")
    meta = load_meta(folder)
    meta["face_embedding"] = _vec(1.0, 0.0, 0.0)
    from recast.core.characters import save_meta

    save_meta(folder, meta)
    sample = _sample_file(tmp_path / "src" / "face_a.jpg")

    report = migrate_to_character_folders(
        out,
        face_samples={"Person_2": [str(sample)]},
        embeddings={"Person_2": _vec(0.99, 0.14, 0.0)},
    )
    assert not (out / "characters" / "Person_2").exists()
    assert (out / "characters" / "温念" / "face" / "face_a.jpg").is_file()
    assert report["merged"] == [
        {"from": "Person_2", "into": "温念", "via": "embedding比对", "detail": "相似度 0.99"}
    ]
    assert report["created"] == []
    meta = load_meta(folder)
    assert meta["face"] == {"files": [str(out / "characters" / "温念" / "face" / "face_a.jpg")]}
    # embedding mirror updated toward the new cluster (still ~ the old one)
    assert np.allclose(meta["face_embedding"], _vec(1.0, 0.0, 0.0), atol=0.1)


def test_migrate_embedding_below_threshold_creates_new_folder(tmp_path: Path):
    from recast.core.characters import ensure_character_folder, migrate_to_character_folders

    out = tmp_path / "outputs"
    folder, _ = ensure_character_folder(out, "温念")
    from recast.core.characters import load_meta, save_meta

    meta = load_meta(folder)
    meta["face_embedding"] = _vec(1.0, 0.0, 0.0)
    save_meta(folder, meta)
    sample = _sample_file(tmp_path / "src" / "face_b.jpg")

    report = migrate_to_character_folders(
        out,
        face_samples={"Person_9": [str(sample)]},
        embeddings={"Person_9": _vec(0.0, 1.0, 0.0)},
    )
    assert (out / "characters" / "Person_9").is_dir()
    assert report["merged"] == []
    assert report["created"] == ["Person_9"]


def test_migrate_resolves_folder_via_store_asset_ownership(tmp_path: Path):
    from recast.core.characters import migrate_to_character_folders

    out = tmp_path / "outputs"
    store = _store(out / "store" / "recast.db")
    sample = _sample_file(tmp_path / "src" / "face_c.jpg")
    row = _character_row(store, "温念", _vec(1.0, 0.0, 0.0), cluster_key="ep1:3")
    store.insert(
        "character_assets",
        character_id=row["id"],
        kind="face",
        path=str(sample),
    )

    # No store passed: migrate auto-opens <output_root>/store/recast.db.
    report = migrate_to_character_folders(out, face_samples={"Person_2": [str(sample)]})
    assert not (out / "characters" / "Person_2").exists()
    assert (out / "characters" / "温念" / "face" / "face_c.jpg").is_file()
    assert report["resolved"] == [
        {"from": "Person_2", "into": "温念", "via": "store资产归属", "detail": "样本已属于「温念」"}
    ]


def test_migrate_idempotent_with_embeddings(tmp_path: Path):
    from recast.core.characters import ensure_character_folder, migrate_to_character_folders

    out = tmp_path / "outputs"
    folder, _ = ensure_character_folder(out, "温念")
    from recast.core.characters import load_meta, save_meta

    meta = load_meta(folder)
    meta["face_embedding"] = _vec(1.0, 0.0, 0.0)
    save_meta(folder, meta)
    sample = _sample_file(tmp_path / "src" / "face_d.jpg")
    kwargs = dict(
        face_samples={"Person_2": [str(sample)]},
        embeddings={"Person_2": _vec(0.99, 0.14, 0.0)},
    )
    first = migrate_to_character_folders(out, **kwargs)
    second = migrate_to_character_folders(out, **kwargs)
    assert first["copied"] == 1 and first["merged"]
    assert second["copied"] == 0 and second["merged"] == []
    assert second["resolved"]  # mapping decision is still recorded


def test_migrate_notes_when_no_merge_input(tmp_path: Path):
    from recast.core.characters import migrate_to_character_folders

    out = tmp_path / "outputs"
    sample = _sample_file(tmp_path / "src" / "face_e.jpg")
    report = migrate_to_character_folders(out, face_samples={"Person_2": [str(sample)]})
    assert (out / "characters" / "Person_2").is_dir()
    assert "未做归并比对" in report["merge_note"]
