"""One parser for character folders, shared by the three application lines.

The realignment plan makes ``characters/<名字>/`` the unit of truth. The
face-swap line, the script-generation line and the Jianying export line must
all read character data from here instead of each keeping its own copy of
the lookup. Folder first, old logic second: every function here returns
``None`` (or an empty value) when there is no usable folder, so callers can
fall back to the store / pack / roster path without breaking old behaviour.

A folder is usable when the directory exists and meta.json parses. A
corrupt meta.json is logged and treated as unusable, never as an excuse to
guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recast.core.characters import (
    character_folder,
    get_assembly,
    get_timbre,
    list_character_folders,
    load_meta,
    resolve_assembly_path,
)
from recast.core.constants import IMAGE_EXTS

log = logging.getLogger("recast")

_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass
class CharacterAssets:
    """Everything one application line needs about one person."""

    name: str
    folder: Path
    meta: dict[str, Any] = field(default_factory=dict)
    character_id: str | None = None
    face_files: list[Path] = field(default_factory=list)
    voice_samples: list[Path] = field(default_factory=list)
    dubbed_lines: dict[str, Path] = field(default_factory=dict)
    cutouts: list[Path] = field(default_factory=list)
    costume_files: list[Path] = field(default_factory=list)
    model_source: str = "none"
    model_format: str = ""
    model_note: str = ""
    model_path: Path | None = None
    # Current assembly (meta.json ``assembly``), resolved to absolute paths.
    # Each value is None when nothing is chosen or the file is gone.
    assembly: dict[str, Path | None] = field(default_factory=dict)

    @property
    def face_path(self) -> Path | None:
        """Reference face: the assembled face first, else the first sample."""
        chosen = (self.assembly or {}).get("face")
        if chosen is not None:
            return chosen
        return self.face_files[0] if self.face_files else None

    @property
    def body_path(self) -> Path | None:
        """Assembled body (GLB) first, else the folder's own model GLB."""
        chosen = (self.assembly or {}).get("body")
        if chosen is not None:
            return chosen
        if self.model_path is not None and self.model_path.suffix.lower() == ".glb":
            return self.model_path
        return None

    @property
    def costume_path(self) -> Path | None:
        """Assembled costume first, else the first costume file."""
        chosen = (self.assembly or {}).get("costume")
        if chosen is not None:
            return chosen
        return self.costume_files[0] if self.costume_files else None

    @property
    def voice_reference(self) -> Path | None:
        """First usable voice sample: the reference wav for voice cloning."""
        return self.voice_samples[0] if self.voice_samples else None

    @property
    def tts_voice(self) -> str | None:
        """Explicit TTS voice id from meta, when one was set."""
        voice = self.meta.get("voice") if isinstance(self.meta.get("voice"), dict) else {}
        raw = str(voice.get("tts_voice") or "").strip()
        return raw or None

    @property
    def voice(self) -> str | None:
        """The voice string to hand to TTS: explicit id, else a sample path."""
        if self.tts_voice:
            return self.tts_voice
        if self.voice_reference is not None:
            return str(self.voice_reference)
        return None

    @property
    def timbre(self) -> dict[str, Any]:
        return get_timbre(self.meta)

    @property
    def costume_description(self) -> str:
        costume = self.meta.get("costume") if isinstance(self.meta.get("costume"), dict) else {}
        return str(costume.get("description") or "").strip()


def _existing(paths: Any) -> list[Path]:
    """Keep only the listed paths that are files on disk, in order."""
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths or []:
        try:
            path = Path(str(raw)).expanduser()
        except (TypeError, ValueError):
            continue
        if not path.is_file():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _dir_files(folder: Path, sub: str, suffixes: set[str]) -> list[Path]:
    """Existing files directly under ``folder/sub`` with a wanted suffix."""
    target = folder / sub
    if not target.is_dir():
        return []
    try:
        items = sorted(target.iterdir(), key=lambda p: p.name)
    except OSError:
        return []
    return [p for p in items if p.is_file() and p.suffix.lower() in suffixes]


def _display_names(name: str, store: Any | None) -> list[str]:
    """Folder-name candidates for a lookup key: display name first, then raw."""
    candidates: list[str] = []
    if store is not None:
        try:
            ch = store.get_character(str(name))
        except Exception:
            ch = None
        if ch:
            for key in (ch.get("display_name"), ch.get("name")):
                if key and str(key) not in candidates:
                    candidates.append(str(key))
    if str(name) not in candidates:
        candidates.append(str(name))
    return candidates


def resolve_character_assets(
    output_root: str | Path | None,
    name: str,
    *,
    store: Any | None = None,
) -> CharacterAssets | None:
    """Read one person's folder, or None when there is no usable folder.

    ``name`` may be a display name, a store character id, or an auto name.
    When a store is given, an id is mapped to its display name first. A
    corrupt meta.json is logged and treated as unusable.
    """
    if not output_root or not str(name or "").strip():
        return None
    key = str(name).strip()
    character_id: str | None = None
    if store is not None:
        try:
            ch = store.get_character(key)
        except Exception:
            ch = None
        if ch:
            character_id = str(ch.get("id") or key)
    folder: Path | None = None
    for candidate in _display_names(key, store):
        trial = character_folder(output_root, candidate)
        if trial.is_dir():
            folder = trial
            break
    if folder is None and store is not None:
        # Folder may be keyed by an auto name while meta carries the display
        # name (or the other way round): scan once instead of guessing.
        try:
            for entry in list_character_folders(output_root):
                meta = entry.get("meta") or {}
                if str(meta.get("name") or "") in (key, * _display_names(key, store)):
                    folder = Path(str(entry["folder"]))
                    break
        except (OSError, ValueError):
            folder = None
    if folder is None:
        return None
    try:
        meta = load_meta(folder) or {}
    except ValueError as exc:
        log.warning("人物文件夹 %s 的 meta.json 读不懂，已回落旧逻辑：%s", folder, exc)
        return None
    voice = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
    costume = meta.get("costume") if isinstance(meta.get("costume"), dict) else {}
    face_meta = meta.get("face") if isinstance(meta.get("face"), dict) else {}
    model_meta = meta.get("model") if isinstance(meta.get("model"), dict) else {}

    face_files = _existing(face_meta.get("files"))
    for extra in _dir_files(folder, "face", set(IMAGE_EXTS)):
        if str(extra) not in {str(p) for p in face_files}:
            face_files.append(extra)
    voice_samples = _existing(voice.get("samples"))
    for extra in _dir_files(folder, "voice", _AUDIO_SUFFIXES):
        if str(extra) not in {str(p) for p in voice_samples}:
            voice_samples.append(extra)
    dubbed: dict[str, Path] = {}
    for code, raw in (voice.get("dubbed_lines") or {}).items():
        try:
            path = Path(str(raw)).expanduser()
        except (TypeError, ValueError):
            continue
        if path.is_file():
            dubbed[str(code)] = path
    for extra in _dir_files(folder / "voice", "lines", _AUDIO_SUFFIXES):
        if extra.stem not in dubbed:
            dubbed[extra.stem] = extra

    model_path: Path | None = None
    raw_model = str(model_meta.get("path") or "").strip()
    if raw_model:
        candidate = Path(raw_model).expanduser()
        if not candidate.is_absolute():
            candidate = folder / candidate
        if candidate.is_file() or candidate.is_dir():
            model_path = candidate
    if model_path is None:
        # A GLB copied into model/ by hand (e.g. model/model.glb) counts
        # even when meta.json never registered it.
        found_glbs = _dir_files(folder, "model", {".glb"})
        if found_glbs:
            model_path = found_glbs[0]

    raw_assembly = get_assembly(meta)
    assembly = {
        key: resolve_assembly_path(folder, raw_assembly[key], output_root=output_root)
        for key in ("face", "body", "costume")
    }

    return CharacterAssets(
        name=str(meta.get("name") or folder.name),
        folder=folder,
        meta=meta,
        character_id=character_id,
        face_files=face_files,
        voice_samples=voice_samples,
        dubbed_lines=dubbed,
        cutouts=_dir_files(folder, "cutout", {".png"}),
        costume_files=_existing(costume.get("files")) + [
            p
            for p in _dir_files(folder, "costume", set(IMAGE_EXTS))
            if str(p) not in {str(q) for q in _existing(costume.get("files"))}
        ],
        model_source=str(model_meta.get("source") or "none"),
        model_format=str(model_meta.get("format") or ""),
        model_note=str(model_meta.get("note") or ""),
        model_path=model_path,
        assembly=assembly,
    )


def resolve_face_reference(
    output_root: str | Path | None,
    name: str,
    *,
    store: Any | None = None,
) -> dict[str, Any] | None:
    """Reference face for one person, in the replace line's row shape."""
    assets = resolve_character_assets(output_root, name, store=store)
    if assets is None or assets.face_path is None:
        return None
    return {
        "character_id": assets.character_id,
        "name": assets.name,
        "face_path": str(assets.face_path),
    }


def resolve_voice(
    output_root: str | Path | None,
    speaker: str,
    *,
    store: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """(voice, timbre) for a speaker, folder first.

    Voice is the explicit TTS id when set, else the first voice sample path
    (a clone reference). Timbre comes from the folder meta. No folder means
    (None, default timbre): the caller fills in from the old source.
    """
    from recast.generate.timbre import normalize_timbre

    assets = resolve_character_assets(output_root, speaker, store=store)
    if assets is None:
        return None, normalize_timbre(None)
    return assets.voice, assets.timbre


def canonical_speaker_name(
    output_root: str | Path | None,
    speaker: str,
    *,
    store: Any | None = None,
) -> str:
    """Display name from the character folder, or the speaker key unchanged."""
    assets = resolve_character_assets(output_root, speaker, store=store)
    if assets is None:
        return str(speaker or "")
    return assets.name


def resolve_body_model(
    output_root: str | Path | None,
    name: str,
    *,
    store: Any | None = None,
) -> Path | None:
    """The body GLB for one person: the assembled body first.

    Falls back to the folder's own model/ GLB when nothing is assembled.
    This is the single place downstream code asks "which body does this
    person use" — the assembly view, the 3D preview and the costume
    binding all read from here.
    """
    assets = resolve_character_assets(output_root, name, store=store)
    if assets is None:
        return None
    return assets.body_path


def resolve_generation_ref(
    output_root: str | Path | None,
    token: str,
    *,
    store: Any | None = None,
) -> Path | None:
    """Face file for a ref token that names a character, or None.

    Used when a storyboard ref token is a person (e.g. “温念”): the folder's
    face wins over the pack's refs directory. Scene stills never match a
    character folder, so they fall through untouched.

    This deliberately returns the *face image*, not the body GLB: its
    callers (``pack.resolve_shot_refs`` → script pipeline / animatic) feed
    the path to an image model as a reference still, which cannot take a
    GLB. The assembled face already applies here (``assets.face_path``
    prefers ``assembly.face``). For the body model, use
    :func:`resolve_body_model`.
    """
    assets = resolve_character_assets(output_root, token, store=store)
    if assets is None:
        return None
    return assets.face_path


def resolve_voices(
    output_root: str | Path | None,
    speakers: list[str],
    *,
    store: Any | None = None,
    fallback: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """(voices, timbres) for several speakers, folder first, fallback fills.

    ``fallback`` is the old mapping (e.g. ``voices_by_speaker(pack.characters)``).
    A folder voice wins over a fallback voice for the same speaker; a speaker
    with no folder keeps the fallback voice. Timbres only come from folders.
    """
    from recast.generate.timbre import is_default

    voices: dict[str, str] = {}
    timbres: dict[str, dict[str, Any]] = {}
    for speaker in speakers:
        voice, timbre = resolve_voice(output_root, speaker, store=store)
        if voice:
            voices[str(speaker)] = voice
        # A non-default timbre is worth passing even without a folder voice.
        if not is_default(timbre):
            timbres[str(speaker)] = timbre
    for key, voice in (fallback or {}).items():
        if key and key not in voices and str(voice or "").strip():
            voices[str(key)] = str(voice)
    return voices, timbres
