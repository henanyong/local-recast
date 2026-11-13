"""第 ⑤ 步声音与台词面板的纯逻辑（R6 拆分，P4）。

主窗口 ``_diar_segments`` / ``_character_voice_samples`` /
``_update_voice_samples`` / ``_on_reassign_segment`` 的文件与
meta 操作原来都在 ``main_window.py`` 里：这里是唯一来源，主窗口
只留读下拉框、写状态文案与刷 UI 的薄委托，建表建按钮在
``recast.desktop.audio_panel``。

行为与抽取前一字不差：分轨记录就是各人物文件夹 voice/ 下
``SPK_<起>_<止>.wav`` 的文件名（不靠别的索引）；改派时把文件挪
到目标人物 voice/（或未分配区），重名加 ``_moved``，并同步两边
meta.json 的 ``voice.samples``；``_update_voice_samples`` 只在文件
已挪完后调。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from recast.core.characters import character_folder, load_meta, save_meta

#: 分轨片段文件名：SPK_<说话人>_<起秒>_<止秒>.wav（run_diarization 的写法）。
_SEGMENT_PATTERN = re.compile(r"^(SPK_\d+)_(\d+\.\d+)_(\d+\.\d+)\.wav$")


def collect_diar_segments(service: Any) -> list[dict]:
    """分轨结果：各人物文件夹 voice/ 下的 SPK 片段，按时间排。

    只读 service（characters + settings.output_root）与人物文件夹，
    不碰 UI；没跑过分轨就返回空列表。读不到人物列表时同样返回空列表。
    """
    segments: list[dict] = []
    try:
        characters = service.characters()
    except Exception:
        return segments
    for ch in characters:
        name = ch.get("display_name") or ch.get("name") or ""
        voice_dir = character_folder(service.settings.output_root, str(name)) / "voice"
        if not voice_dir.is_dir():
            continue
        for path in sorted(voice_dir.glob("SPK_*.wav")):
            m = _SEGMENT_PATTERN.match(path.name)
            if not m:
                continue
            segments.append({
                "start": float(m.group(2)), "end": float(m.group(3)),
                "speaker": m.group(1), "owner_id": ch["id"],
                "owner_name": str(name), "path": str(path),
            })
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments


def character_voice_samples(service: Any, char_id: str) -> list[Path]:
    """一个角色已有的声音样本：meta 登记的 + voice/ 目录里直接扫到的。

    只读 service 与人物文件夹，不碰 UI。
    """
    ch = service.store.get_character(char_id) or {}
    name = ch.get("display_name") or ch.get("name") or char_id
    folder = character_folder(service.settings.output_root, str(name))
    out: list[Path] = []
    meta = load_meta(folder) or {}
    voice = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
    for s in (voice.get("samples") or []):
        p = Path(str(s))
        if p.is_file() and p.suffix.lower() == ".wav" and p not in out:
            out.append(p)
    voice_dir = folder / "voice"
    if voice_dir.is_dir():
        for p in sorted(voice_dir.glob("*.wav")):
            if p not in out:
                out.append(p)
    return out


def update_voice_samples(
    service: Any, char_id: str | None, *, remove_path: Any, add_path: Any
) -> None:
    """同步一个人物 meta.json 的 voice.samples：移走一个、加一个（文件已挪完才调）。

    ``char_id`` 为空时什么都不做；只读写该人物的 meta.json，不碰 UI。
    """
    if not char_id:
        return
    ch = service.store.get_character(char_id) or {}
    name = ch.get("display_name") or ch.get("name") or char_id
    folder = character_folder(service.settings.output_root, str(name))
    meta = load_meta(folder) or {}
    voice = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
    samples = [str(s) for s in (voice.get("samples") or [])]
    if remove_path is not None:
        samples = [s for s in samples if Path(s) != Path(remove_path)]
    if add_path is not None and str(add_path) not in samples:
        samples.append(str(add_path))
    meta["voice"] = {**voice, "samples": samples}
    save_meta(folder, meta)


def reassign_segment(service: Any, seg: dict, target_id: str | None) -> str:
    """改派：把片段文件挪到目标人物的 voice/（或未分配区），并同步两边 meta.json。

    原地改 ``seg`` 的 ``owner_id`` / ``owner_name`` / ``path``（调用方
    的分轨行列表直接看到变化），返回要写进状态栏的文案。片段文件找
    不到时抛 ``FileNotFoundError``，什么都不动（调用方据此写
    “片段文件找不到了”）。``target_id`` 为 None 表示改派到未分配区。
    """
    src = Path(seg["path"])
    if not src.is_file():
        raise FileNotFoundError(src)
    if target_id:
        target = service.store.get_character(target_id) or {}
        target_name = target.get("display_name") or target.get("name") or target_id
        dest_dir = character_folder(service.settings.output_root, str(target_name)) / "voice"
    else:
        dest_dir = Path(service.settings.output_root) / "_diarization" / "unassigned"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() != src.resolve():
        if dest.exists():
            dest = dest_dir / f"{src.stem}_moved{src.suffix}"
        src.rename(dest)
    update_voice_samples(service, seg["owner_id"], remove_path=src, add_path=None)
    if target_id:
        update_voice_samples(service, target_id, remove_path=None, add_path=dest)
    seg["owner_id"] = target_id
    seg["owner_name"] = target_name if target_id else "未分配"
    seg["path"] = str(dest)
    return (
        f"已把 {seg['start']:.2f}s–{seg['end']:.2f}s 的片段"
        f"改派给{seg['owner_name']}。"
    )
