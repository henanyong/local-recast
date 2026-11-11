from recast.cloud.channels import CloudDisabled, invoke_chat
from recast.cloud.secrets import SecretStore
from recast.core.settings import AppSettings
from recast.upload import run_upload, upload_default_off
from pathlib import Path


def test_cloud_not_called_when_disabled(monkeypatch):
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("should not perform HTTP when disabled")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    rec = invoke_chat(
        "deepseek",
        [{"role": "user", "content": "hi"}],
        enabled=False,
        credentials={"api_key": "sk-test-should-not-be-used"},
        auto_confirm=True,
    )
    assert rec.skipped is True
    assert called["n"] == 0
    assert rec.ok is False


def test_cloud_not_called_without_confirm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("HTTP should not run without confirm")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    try:
        invoke_chat(
            "deepseek",
            [{"role": "user", "content": "hi"}],
            enabled=True,
            credentials={"api_key": "sk-test"},
            confirm=None,
            auto_confirm=False,
        )
        raised = False
    except Exception:
        raised = True
    assert raised


def test_secrets_roundtrip_not_plaintext(tmp_path: Path):
    store = SecretStore(tmp_path / "secrets.bin", tmp_path / "master.key")
    store.set("deepseek", {"api_key": "sk-secret-value-123456"})
    blob = (tmp_path / "secrets.bin").read_bytes()
    assert b"sk-secret-value-123456" not in blob
    assert store.get("deepseek")["api_key"] == "sk-secret-value-123456"


def test_upload_default_off():
    s = AppSettings()
    assert s.auto_upload is False
    assert upload_default_off(s) is True
    rec = run_upload("/tmp/nope", enabled=s.auto_upload, mode="folder", folder="/tmp")
    assert rec.get("skipped") is True


def test_upload_copy_to_folder(tmp_path: Path):
    src = tmp_path / "out.mp4"
    src.write_bytes(b"fake")
    dest = tmp_path / "dest"
    rec = run_upload(src, enabled=True, mode="folder", folder=str(dest))
    assert rec["ok"] is True
    assert (dest / "out.mp4").is_file()
