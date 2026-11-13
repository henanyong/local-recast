"""第 ⑤ 步声音与台词面板的建 UI（R6 拆分，P4）。

主窗口中栏「⑤ 声音与台词」组的建组、分轨表的重建与每人物一行
试听 / 调音色 / 改台词按钮原来都在 ``main_window.py`` 里：这里
是唯一来源，主窗口只留取数（分轨结果与人物列表，见
``recast.core.audio_segments``）、动作（分轨 / 试听 / 调音色 /
改台词 / 重配音 / 编辑字幕，仍在主窗口）与刷新触发，建表建按钮
的循环都在这里。

文案、对象名、提示与下拉框行为与抽取前一字不差：分轨表 4 列
（开始 / 结束 / 说话人 / 改派给），「改派给」默认“未分配”；
每人物一行三个按钮（试听 3 秒 / 调节音色 / 改台词），对象名
``audition:<id>`` / ``timbre:<id>`` / ``dialogue:<id>``；状态栏
两句文案按有无分轨结果区分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AudioPanelWidgets(NamedTuple):
    """``build_audio_panel`` 建出的整组 widgets，主窗口按名挂到 self 上。"""

    group: QWidget
    status: QLabel
    diar_table: QTableWidget
    char_layout: QVBoxLayout
    rediar_btn: QPushButton
    redub_btn: QPushButton
    edit_subs_btn: QPushButton


def build_audio_panel(
    *,
    on_rediarize: Callable[[], None],
    on_redub: Callable[[], None],
    on_edit_subtitles: Callable[[], None],
) -> AudioPanelWidgets:
    """建「⑤ 声音与台词」组（默认隐藏，点左侧第 ⑤ 步才打开）。"""
    audio_group = QGroupBox("⑤ 声音与台词")
    acol = QVBoxLayout(audio_group)
    status = QLabel("先选一部剧，再点第 ⑤ 步。")
    status.setWordWrap(True)
    acol.addWidget(status)
    diar_table = QTableWidget(0, 4)
    diar_table.setHorizontalHeaderLabels(["开始", "结束", "说话人", "改派给"])
    diar_table.setColumnWidth(3, 180)
    diar_table.horizontalHeader().setStretchLastSection(True)
    acol.addWidget(diar_table)
    char_layout = QVBoxLayout()
    acol.addLayout(char_layout)
    abtn_row = QHBoxLayout()
    rediar_btn = QPushButton("重新分轨")
    rediar_btn.setToolTip("按说话人重新切分音频（启发式分轨，会错配；错了在上表改派）。")
    rediar_btn.clicked.connect(on_rediarize)
    redub_btn = QPushButton("重配音")
    redub_btn.setToolTip("只重配改过的台词行，没改的行不动；配音从不加速，改行后后面的台词顺延。")
    redub_btn.clicked.connect(on_redub)
    edit_subs_btn = QPushButton("编辑字幕")
    edit_subs_btn.setToolTip("打开字幕编辑器（改字、拆分、合并、检查时间轴）。")
    edit_subs_btn.clicked.connect(on_edit_subtitles)
    abtn_row.addWidget(rediar_btn)
    abtn_row.addWidget(redub_btn)
    abtn_row.addWidget(edit_subs_btn)
    acol.addLayout(abtn_row)
    audio_group.setVisible(False)
    return AudioPanelWidgets(
        group=audio_group,
        status=status,
        diar_table=diar_table,
        char_layout=char_layout,
        rediar_btn=rediar_btn,
        redub_btn=redub_btn,
        edit_subs_btn=edit_subs_btn,
    )


def refresh_diar_table(
    diar_table: QTableWidget,
    segments: list[dict],
    characters: list[dict],
    on_reassign: Callable[[int], None],
) -> None:
    """重建分轨表：每行一个「改派给」下拉框，改派时调 ``on_reassign(row)``。"""
    diar_table.setRowCount(0)
    for row, seg in enumerate(segments):
        diar_table.insertRow(row)
        for col, text in ((0, f"{seg['start']:.2f}s"), (1, f"{seg['end']:.2f}s"), (2, seg["speaker"])):
            cell = QTableWidgetItem(text)
            cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            diar_table.setItem(row, col, cell)
        combo = QComboBox()
        combo.addItem("未分配", None)
        for ch in characters:
            combo.addItem(ch.get("display_name") or ch.get("name") or ch["id"], ch["id"])
        combo.setCurrentIndex(max(0, combo.findData(seg["owner_id"])))
        combo.currentIndexChanged.connect(
            lambda _idx, r=row: on_reassign(r)
        )
        diar_table.setCellWidget(row, 3, combo)


def rebuild_character_rows(
    char_layout: QVBoxLayout,
    characters: list[dict],
    *,
    on_audition: Callable[[str], None],
    on_timbre: Callable[[str], None],
    on_dialogue: Callable[[str], None],
) -> dict[str, dict[str, QPushButton]]:
    """重建每人物一行试听 / 调音色 / 改台词按钮，返回 ``{char_id: {audition, timbre, dialogue}}``。

    先清掉旧行（``deleteLater``），再按当前人物列表重建。
    """
    while char_layout.count():
        item = char_layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
    char_btns: dict[str, dict[str, QPushButton]] = {}
    for ch in characters:
        name = ch.get("display_name") or ch.get("name") or ch["id"]
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.addWidget(QLabel(str(name)))
        audition_btn = QPushButton("试听 3 秒")
        audition_btn.setObjectName(f"audition:{ch['id']}")
        audition_btn.setToolTip("用该角色的音色合成一句约 3 秒的试听；合不出来就放已有的声音样本。")
        audition_btn.clicked.connect(
            lambda _c=False, cid=ch["id"]: on_audition(cid)
        )
        timbre_btn = QPushButton("调节音色")
        timbre_btn.setObjectName(f"timbre:{ch['id']}")
        timbre_btn.clicked.connect(
            lambda _c=False, cid=ch["id"]: on_timbre(cid)
        )
        dialogue_btn = QPushButton("改台词")
        dialogue_btn.setObjectName(f"dialogue:{ch['id']}")
        dialogue_btn.clicked.connect(
            lambda _c=False, cid=ch["id"]: on_dialogue(cid)
        )
        for btn in (audition_btn, timbre_btn, dialogue_btn):
            row_l.addWidget(btn)
        char_btns[ch["id"]] = {
            "audition": audition_btn, "timbre": timbre_btn, "dialogue": dialogue_btn,
        }
        char_layout.addWidget(row_w)
    return char_btns


def audio_status_text(segments: list[dict]) -> str:
    """第 ⑤ 步面板的状态栏文案：有无分轨结果各一句。"""
    if not segments:
        return (
            "这部剧还没有分轨结果：点[重新分轨]，按说话人切出每段；"
            "分轨是启发式，会错配，错了在上表里改派。"
        )
    return (
        f"分轨结果共 {len(segments)} 段：点每行的「改派给」可以改分给谁"
        "（分轨是启发式，会错配）；下面每行可以试听、调音色、改台词。"
    )


def find_subtitle_target(
    service: Any, drama_id: str | None
) -> tuple[Path | None, str | None, str | None]:
    """「编辑字幕」要开的字幕文件：返回 ``(srt_path, task_id, error_text)``。

    没分集工作目录或没字幕文件时返回 ``(None, None, 原因)``（调用方
    弹提示）；有文件时 ``error_text`` 为 None。只读 service 与磁盘，
    不开对话框。
    """
    tasks = service.store.tasks_for_drama(drama_id) if drama_id else []
    workdir = tasks[0].get("workdir") if tasks else None
    if not workdir:
        return None, None, "这部剧还没有分集工作目录（先跑一次提取）。"
    srt = Path(workdir) / "subtitles.srt"
    if not srt.is_file():
        return None, None, f"还没有字幕文件：{srt}\n先跑「解析提取」。"
    return srt, tasks[0]["id"], None


def apply_audio_result(host: Any, payload: dict) -> None:
    """第 ⑤ 步后台任务的结果回主线程：分轨 / 试听 / 重配音三路。

    ``host`` 是主窗口（只用它的 ``audio_status`` / ``rediar_btn`` /
    ``_refresh_audio_panel`` / ``_character_voice_samples``）。文案与
    回退逻辑与抽取前一字不差：试听先放合成音频，合不出来放已有声音
    样本，都没有才明说缺什么——绝不伪造音频。
    """
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    kind = payload.get("kind")
    if kind == "rediarize":
        host.rediar_btn.setEnabled(True)
        if payload.get("error"):
            host.audio_status.setText(f"分轨失败：{payload['error']}")
            return
        report = payload.get("report") or {}
        segments = report.get("segments") or []
        host.audio_status.setText(f"分轨完成：{len(segments)} 段。")
        host._refresh_audio_panel()
        return
    if kind == "audition":
        char_id = payload.get("character")
        rec = payload.get("rec") or {}
        path = rec.get("path") if rec.get("ok") else None
        if path and Path(path).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            host.audio_status.setText(
                f"试听已生成：{Path(path).name}（{rec.get('duration') or '约 3'} 秒），已用默认播放器打开。"
            )
            return
        err = payload.get("error") or rec.get("error") or rec.get("skipped") or "合成未成功"
        samples = host._character_voice_samples(char_id) if char_id else []
        if samples:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(samples[0])))
            host.audio_status.setText(
                f"在线合成失败（{err}），改放该角色已有的声音样本：{samples[0].name}。"
            )
            return
        host.audio_status.setText(
            f"试听失败：{err}；该角色也没有声音样本——先点[重新分轨]，"
            "或在人物库里导入声音样本。"
        )
        return
    if kind == "redub":
        if payload.get("error"):
            host.audio_status.setText(f"重配音失败：{payload['error']}")
            return
        report = payload.get("report") or {}
        redubbed = report.get("redubbed")
        if isinstance(redubbed, int):
            host.audio_status.setText(f"重配音完成：重配了 {redubbed} 行（没改的行不动）。")
        else:
            host.audio_status.setText("重配音完成。")
        return
