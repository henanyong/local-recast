"""Encrypt cloud secrets. Windows DPAPI when available, otherwise a local Fernet/key file.

Secrets never go into logs or export files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from recast.config import PROJECT_ROOT


def default_secrets_path() -> Path:
    dest = PROJECT_ROOT / "outputs" / "store"
    dest.mkdir(parents=True, exist_ok=True)
    return dest / "secrets.bin"


def default_key_path() -> Path:
    dest = PROJECT_ROOT / "outputs" / "store"
    dest.mkdir(parents=True, exist_ok=True)
    return dest / "master.key"


def redact(text: str | None) -> str:
    if not text:
        return ""
    # hide tokens that look like keys
    import re

    out = text
    out = re.sub(r"(sk-[A-Za-z0-9]{8})[A-Za-z0-9]+", r"\1…", out)
    out = re.sub(r"(?i)(api[_-]?key|secret|ak|sk)\s*[:=]\s*\S+", r"\1=***", out)
    return out


def _dpapi_available() -> bool:
    return os.name == "nt"


def _dpapi_protect(data: bytes) -> bytes:
    import ctypes
    import ctypes.wintypes as wt

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    in_blob = DATA_BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
    out_blob = DATA_BLOB()
    if not crypt32.CryptProtectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
        raise OSError("CryptProtectData failed")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    import ctypes
    import ctypes.wintypes as wt

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    in_blob = DATA_BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
    out_blob = DATA_BLOB()
    if not crypt32.CryptUnprotectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
        raise OSError("CryptUnprotectData failed")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _fernet():
    try:
        from cryptography.fernet import Fernet  # type: ignore
    except Exception:
        return None
    return Fernet


def _load_or_create_key(path: Path) -> bytes:
    Fernet = _fernet()
    if Fernet is None:
        # fallback: random 32 bytes, used with a simple xor+hmac scheme
        if path.is_file():
            return path.read_bytes()
        key = os.urandom(32)
        path.write_bytes(key)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return key
    if path.is_file():
        return path.read_bytes()
    key = Fernet.generate_key()
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def _xor_crypt(data: bytes, key: bytes) -> bytes:
    import hashlib
    import hmac

    # prefix 16 bytes nonce, then xor keystream, then 32 byte hmac
    if len(data) >= 48 and data[:3] != b"RC1":
        # decrypt path: nonce|cipher|mac
        pass
    nonce = os.urandom(16)
    stream = hashlib.sha256(key + nonce).digest()
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ stream[i % len(stream)])
        if (i + 1) % len(stream) == 0:
            stream = hashlib.sha256(stream + key).digest()
    mac = hmac.new(key, nonce + bytes(out), hashlib.sha256).digest()
    return b"RC1" + nonce + bytes(out) + mac


def _xor_decrypt(blob: bytes, key: bytes) -> bytes:
    import hashlib
    import hmac

    if not blob.startswith(b"RC1") or len(blob) < 3 + 16 + 32:
        raise ValueError("bad secret blob")
    nonce = blob[3:19]
    mac = blob[-32:]
    cipher = blob[19:-32]
    expect = hmac.new(key, nonce + cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expect):
        raise ValueError("secret mac mismatch")
    stream = hashlib.sha256(key + nonce).digest()
    out = bytearray()
    for i, b in enumerate(cipher):
        out.append(b ^ stream[i % len(stream)])
        if (i + 1) % len(stream) == 0:
            stream = hashlib.sha256(stream + key).digest()
    return bytes(out)


class SecretStore:
    def __init__(self, path: str | Path | None = None, key_path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_secrets_path()
        self.key_path = Path(key_path) if key_path else default_key_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _encrypt(self, data: bytes) -> bytes:
        if _dpapi_available():
            return b"DP1" + _dpapi_protect(data)
        Fernet = _fernet()
        if Fernet is not None:
            key = _load_or_create_key(self.key_path)
            return b"FT1" + Fernet(key).encrypt(data)
        key = _load_or_create_key(self.key_path)
        return _xor_crypt(data, key)

    def _decrypt(self, blob: bytes) -> bytes:
        if blob.startswith(b"DP1"):
            return _dpapi_unprotect(blob[3:])
        if blob.startswith(b"FT1"):
            Fernet = _fernet()
            if Fernet is None:
                raise RuntimeError("需要 cryptography 才能读取该密钥文件")
            key = _load_or_create_key(self.key_path)
            return Fernet(key).decrypt(blob[3:])
        key = _load_or_create_key(self.key_path)
        return _xor_decrypt(blob, key)

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            raw = self._decrypt(self.path.read_bytes())
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        blob = self._encrypt(payload)
        self.path.write_bytes(blob)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def get(self, name: str) -> dict[str, Any]:
        return dict((self.load().get(name) or {}))

    def set(self, name: str, values: dict[str, Any]) -> None:
        data = self.load()
        data[name] = {k: v for k, v in values.items() if v is not None}
        self.save(data)

    def delete(self, name: str) -> None:
        data = self.load()
        data.pop(name, None)
        self.save(data)

    def configured(self, name: str) -> bool:
        rec = self.get(name)
        return bool(rec.get("api_key") or rec.get("ak") or rec.get("access_key"))
