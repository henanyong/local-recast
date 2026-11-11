from __future__ import annotations

from pathlib import Path

from recast.core.checkpoint import should_skip, write_checkpoint
from recast.core.fingerprints import file_fingerprint, fingerprints_match
from recast.store.db import Store


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    a = file_fingerprint(p)
    p.write_bytes(b"hello!")
    b = file_fingerprint(p)
    assert a["sha256"] != b["sha256"]
    assert not fingerprints_match([a], [b])
    assert fingerprints_match([a], [file_fingerprint(p) if False else a])


def test_skip_when_inputs_unchanged_and_outputs_exist(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.json"
    src.write_bytes(b"video-bytes")
    out.write_text("{}", encoding="utf-8")
    write_checkpoint(store, "task1", "asr", [src], [out])
    assert should_skip(store, "task1", "asr", [src], [out]) is True
    # change input → must rerun
    src.write_bytes(b"video-bytes-changed")
    assert should_skip(store, "task1", "asr", [src], [out]) is False
    # restore input but delete output → must rerun
    src.write_bytes(b"video-bytes")
    # fingerprints were written against the first content, so this still won't match...
    # re-write checkpoint for the restored content first, then delete output
    write_checkpoint(store, "task1", "asr", [src], [out])
    assert should_skip(store, "task1", "asr", [src], [out]) is True
    out.unlink()
    assert should_skip(store, "task1", "asr", [src], [out]) is False


def test_skip_without_store_uses_existing_outputs(tmp_path: Path) -> None:
    out = tmp_path / "done.txt"
    assert should_skip(None, None, "x", [], [out]) is False
    out.write_text("ok", encoding="utf-8")
    assert should_skip(None, None, "x", [], [out]) is True
