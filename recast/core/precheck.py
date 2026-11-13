"""导入预检卡的纯计算与后台探测（R6 拆分，P4）。

主窗口 ``_apply_precheck_result`` / ``_precheck_dot`` 与两个后台
探测 worker（选片预检、行状态点轻量预检）原来都在
``main_window.py`` 里：这里是唯一来源，主窗口只留线程启动、
信号回主线程、写缓存与刷 UI 的薄委托，判断与文案都在这里。

行为与抽取前一字不差：只有真正挡路的（没有视频文件 / 没有视频
流 / 时长读不到）才阻止导入；没有音轨、烧录字幕、OCR 不可用都
只是警告，不阻止。``probe_precheck_payload`` 里的 lazy import
（``recast.ffmpeg.probe_video``、``recast.stages.subtitles.
detect_burned_subtitles``）与原来 worker 里的写法一致，探测失败
时 ``error`` / ``burned=None`` 的落点也不变。
"""

from __future__ import annotations

from typing import Any

#: 时长读不到时的挡路原因：与主窗口原来的文案一字不差。
ZERO_DURATION_REASON = "视频时长读不到（0 秒）。"


def precheck_dot(rec: dict | None) -> tuple[str, str]:
    """剧目行的预检状态点（N8）：✓ 通过 / ⚠ 警告 / ✗ 阻止；没查过返回空。

    纯函数：只读传入的一条预检记录（``MainWindow._precheck`` 里的值）。
    """
    if not rec:
        return "", ""
    if rec.get("blocked"):
        return "✗ ", f"预检阻止：{rec['blocked']}"
    warns = []
    if rec.get("has_audio") is False:
        warns.append("无音轨")
    if rec.get("burned") is True:
        warns.append("有烧录字幕")
    elif "burned" in rec and rec.get("burned") is None:
        warns.append("烧录字幕未知")
    if warns:
        return "⚠ ", "预检警告：" + "、".join(warns)
    return "✓ ", "预检通过"


def format_precheck_text(duration: float, has_audio: Any, burned: Any) -> str:
    """预检通过时的卡片文案：时长 / 音轨 / 烧录字幕 + 警告行。

    纯函数：与主窗口原来的 ``_apply_precheck_result`` 通过分支一字不差。
    """
    lines = [
        f"时长：{duration:.1f} 秒",
        f"音轨：{'有' if has_audio else '无'}",
        "烧录字幕：" + ("有" if burned is True else "无" if burned is False else "未知"),
    ]
    warns: list[str] = []
    if not has_audio:
        warns.append("没有音轨：分轨 / 配音不可用（只能做画面替换）。")
    if burned is True:
        warns.append("检测到烧录字幕：合成时建议开「遮盖原字幕」（在合成设置里）。")
    elif burned is None:
        warns.append("烧录字幕无法判断（OCR 不可用）：合成后请人工检查画面底部。")
    text = "预检通过。\n" + "\n".join(lines)
    if warns:
        text += "\n" + "\n".join("⚠ " + w for w in warns)
    return text


def summarize_precheck(payload: dict) -> dict:
    """把一次预检结果归纳成缓存记录 + 挡路原因 + 卡片文案。

    返回 ``{"record", "blocked", "label_text"}``：``record`` 是要写进
    ``MainWindow._precheck[drama_id]`` 的值（形状与原来一致：挡路时
    只有 ``video`` + ``blocked``，通过时多 ``duration`` / ``has_audio`` /
    ``burned``）；``blocked`` 为 None 表示通过，同时调用方应把该剧从
    ``_precheck_blocked`` 里删掉；``label_text`` 是要写进预检卡的文案
    （行状态点专用的 ``row_only`` 结果同样算出文案，调用方按
    ``show_label`` 决定写不写卡）。
    """
    video = payload.get("video")
    if payload.get("error"):
        reason = str(payload["error"])
        return {
            "record": {"video": video, "blocked": reason},
            "blocked": reason,
            "label_text": f"预检没通过：{reason}\n导入已被阻止。",
        }
    duration = payload.get("duration")
    if not duration or duration <= 0:
        reason = ZERO_DURATION_REASON
        return {
            "record": {"video": video, "blocked": reason},
            "blocked": reason,
            "label_text": f"预检没通过：{reason}\n导入已被阻止。",
        }
    has_audio = payload.get("has_audio")
    burned = payload.get("burned")
    return {
        "record": {
            "video": video, "blocked": None,
            "duration": duration, "has_audio": has_audio, "burned": burned,
        },
        "blocked": None,
        "label_text": format_precheck_text(duration, has_audio, burned),
    }


def probe_precheck_payload(
    video: str, drama_id: str, *, row_only: bool = False, with_burned: bool = True
) -> dict:
    """后台探测一次视频，返回要发回主线程的 payload（不发信号、不碰 UI）。

    ``video`` 必须非空（没视频的分支由调用方处理：选片预检直接写卡，
    行预检发 ``error``）。``with_burned=False`` 时不跑 OCR（行状态点
    轻量预检）；探测失败只填 ``error``，烧录检测失败只填 ``burned=None``。
    """
    payload: dict = {"drama_id": drama_id, "video": video}
    if row_only:
        payload["row_only"] = True
    try:
        from recast.ffmpeg import probe_video

        info = probe_video(video)
        payload["duration"] = info.duration
        payload["has_audio"] = info.has_audio
    except Exception as exc:
        payload["error"] = str(exc)
        return payload
    if with_burned:
        try:
            from recast.stages.subtitles import detect_burned_subtitles

            payload["burned"] = detect_burned_subtitles(video)
        except Exception:
            payload["burned"] = None
    return payload
