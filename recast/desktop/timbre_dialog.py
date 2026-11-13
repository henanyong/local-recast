"""Adjust one character's voice timbre (pitch / rate / style) + audition."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from recast.core.characters import character_folder, get_timbre, load_meta
from recast.core.service import RecastService
from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap


def stale_timbre_message(affected: list[dict]) -> str:
    """Prompt text after a timbre change invalidated old dubs.

    Pure function (no Qt) so tests can check the wording directly.
    """
    total = sum(int(a.get("lines") or 0) for a in affected)
    return (
        f"音色已改：{len(affected)} 集的旧配音已失效"
        f"（共 {total} 句待重配），第 ⑤ 步会显示未完成。"
    )


class _AuditionSignal(QObject):
    finished = Signal(object)


class _RedubSignal(QObject):
    finished = Signal(object)


class TimbreDialog(QDialog):
    """[调节音色]：音调（半音）、语速、风格参数，存进人物文件夹 meta.json。

    [试听] 是只读预览：把当前滑块值当参数传给 ``audition_voice`` 合成
    一句约 3 秒的试听（voice/audition.wav），并用系统默认播放器打开。
    试听不写 meta.json、不标 dirty、不删旧混音；只有点 [保存] 才会
    调 ``set_voice_timbre`` 生效。点 [取消] 不产生任何写操作。
    """

    def __init__(self, service: RecastService, character: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.character = character
        self._sync_tasks = False
        self._audition_running = False
        self._redub_running = False
        folder = character_folder(service.settings.output_root, character)
        timbre = get_timbre(load_meta(folder) or {})
        self.setWindowTitle(f"调节音色 — {character}")
        # P5/B3: content scrolls, the Save/Cancel bar is pinned at the
        # bottom, dialog clamped to the screen (was resize(480, 300)).
        cap_dialog_to_screen(self, 480, 300)

        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-24, 24)
        self.pitch_slider.setValue(int(round(float(timbre.get("pitch") or 0) * 2)))
        self.pitch_label = QLabel()
        self.pitch_slider.valueChanged.connect(self._sync_labels)
        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setRange(10, 40)
        self.rate_slider.setValue(int(round(float(timbre.get("rate") or 1.0) * 20)))
        self.rate_label = QLabel()
        self.rate_slider.valueChanged.connect(self._sync_labels)
        self.style_edit = QLineEdit(str(timbre.get("style") or ""))
        self.style_edit.setPlaceholderText("风格，例如：温柔、冷淡、急促（可空）")

        content = QWidget()
        layout = QVBoxLayout(content)
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("音调（半音，-12 ~ +12）"))
        pitch_row.addWidget(self.pitch_slider, 1)
        pitch_row.addWidget(self.pitch_label)
        layout.addLayout(pitch_row)
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("语速（0.5 ~ 2.0 倍）"))
        rate_row.addWidget(self.rate_slider, 1)
        rate_row.addWidget(self.rate_label)
        layout.addLayout(rate_row)
        layout.addWidget(QLabel("风格"))
        layout.addWidget(self.style_edit)
        hint = QLabel("试听只预览当前滑块值，不保存；点 [保存] 音色才写入人物文件夹的 meta.json，下次配音 / 试听都会用它。参考 wav 克隆音色不支持调参（会保持原音色）。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b90a0;")
        layout.addWidget(hint)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.audition_btn = QPushButton("试听")
        self.audition_btn.clicked.connect(self._audition)
        layout.addWidget(self.audition_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))
        self._audition_signal = _AuditionSignal()
        self._audition_signal.finished.connect(self._on_audition_finished)
        self._redub_signal = _RedubSignal()
        self._redub_signal.finished.connect(self._on_redub_finished)
        self._sync_labels()

    def _sync_labels(self) -> None:
        self.pitch_label.setText(f"{self.pitch():+.1f} 半音")
        self.rate_label.setText(f"{self.rate():.2f} 倍")

    def pitch(self) -> float:
        return self.pitch_slider.value() / 2.0

    def rate(self) -> float:
        return self.rate_slider.value() / 20.0

    def style(self) -> str:
        return self.style_edit.text().strip()

    def _save(self) -> None:
        try:
            result = self.service.set_voice_timbre(
                self.character, pitch=self.pitch(), rate=self.rate(), style=self.style()
            )
        except Exception as exc:
            self.status.setText(f"保存失败：{exc}")
            return
        affected = (result or {}).get("affected") or []
        if not affected:
            self.status.setText("音色已保存，下次配音生效。")
            self.accept()
            return
        self.status.setText(stale_timbre_message(affected))
        if not self._ask_redub_now(affected):
            self.accept()
            return
        self._run_redub(affected)

    def _ask_redub_now(self, affected: list[dict]) -> bool:
        """Ask whether to re-dub the invalidated episodes now.

        [重配这 N 集] runs service.redub_dialogue per affected workdir
        (off the UI thread); [稍后] just closes the dialog — the lines
        stay dirty and step 5 shows unfinished until they are re-dubbed.
        """
        box = QMessageBox(self)
        box.setWindowTitle("音色已改")
        box.setText(stale_timbre_message(affected))
        redub_btn = box.addButton(
            f"重配这 {len(affected)} 集", QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton("稍后", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is redub_btn

    def _run_redub(self, affected: list[dict]) -> None:
        if self._redub_running:
            return
        self._redub_running = True
        self.audition_btn.setEnabled(False)
        self._buttons.setEnabled(False)
        self.status.setText(f"正在重配 {len(affected)} 集（只重配失效的行）…")
        if self._sync_tasks:
            self._on_redub_finished(self._do_redub(affected))
            return

        def work() -> None:
            self._redub_signal.finished.emit(self._do_redub(affected))

        threading.Thread(target=work, name="recast-timbre-redub", daemon=True).start()

    def _do_redub(self, affected: list[dict]) -> dict:
        results: list[dict] = []
        for item in affected:
            workdir = item.get("workdir")
            try:
                report = self.service.redub_dialogue(workdir)
                results.append(
                    {
                        "workdir": workdir,
                        "ok": True,
                        "redubbed": len(report.get("redubbed") or []),
                        "skipped": len(report.get("skipped") or []),
                    }
                )
            except Exception as exc:
                results.append({"workdir": workdir, "ok": False, "error": str(exc)})
        return {"results": results}

    def _on_redub_finished(self, payload: dict) -> None:
        self._redub_running = False
        self.audition_btn.setEnabled(True)
        self._buttons.setEnabled(True)
        results = payload.get("results") or []
        ok = [r for r in results if r.get("ok")]
        done_lines = sum(int(r.get("redubbed") or 0) for r in ok)
        failed = [r for r in results if not r.get("ok")]
        if failed:
            first = failed[0]
            self.status.setText(
                f"重配完成 {len(ok)}/{len(results)} 集，共 {done_lines} 句已用新音色重配；"
                f"{len(failed)} 集失败：{first.get('error')}"
            )
            return
        self.status.setText(
            f"重配完成：{len(results)} 集，共 {done_lines} 句已用新音色重配，第 ⑤ 步已恢复完成。"
        )

    def _audition(self) -> None:
        """Preview the slider values without saving (N19).

        The current pitch/rate/style are passed as a ``timbre`` override
        to ``audition_voice``; nothing is written to meta.json, no lines
        are marked dirty and no stale mix is removed. Saving happens only
        via [_保存] -> ``set_voice_timbre``.
        """
        if self._audition_running:
            return
        preview = {"pitch": self.pitch(), "rate": self.rate(), "style": self.style()}
        self._audition_running = True
        self.audition_btn.setEnabled(False)
        self.status.setText("正在试听…（合成一句约 3 秒，试听不保存音色）")
        if self._sync_tasks:
            try:
                result = self.service.audition_voice(self.character, timbre=preview)
                outcome: dict = {"ok": True, "result": result}
            except Exception as exc:
                outcome = {"ok": False, "message": str(exc)}
            self._on_audition_finished(outcome)
            return

        def work() -> None:
            try:
                result = self.service.audition_voice(self.character, timbre=preview)
                outcome = {"ok": True, "result": result}
            except Exception as exc:
                outcome = {"ok": False, "message": str(exc)}
            self._audition_signal.finished.emit(outcome)

        threading.Thread(target=work, name="recast-timbre-audition", daemon=True).start()

    def _on_audition_finished(self, outcome: dict) -> None:
        self._audition_running = False
        self.audition_btn.setEnabled(True)
        if not outcome.get("ok"):
            self.status.setText(f"试听失败：{outcome.get('message')}")
            return
        result = outcome.get("result") or {}
        path = result.get("path") or result.get("dest") or ""
        duration = result.get("duration")
        dur = f"，{float(duration):.1f} 秒" if duration else ""
        self.status.setText(f"试听已生成：{path}{dur}（已用默认播放器打开）")
        if path:
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            except Exception:
                pass
