"""Write cost.xlsx from the runs table. Numbers must match the logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

HEADERS = ("步骤", "时长(秒)", "云费用", "模型", "状态", "开始时间", "结束时间", "说明")


def _as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def cost_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in runs:
        rows.append(
            {
                "step": r.get("step") or "",
                "duration_sec": _as_float(r.get("duration_sec")),
                "cost": _as_float(r.get("cost")),
                "model": r.get("model") or "",
                "status": r.get("status") or "",
                "started_at": r.get("started_at") or "",
                "finished_at": r.get("finished_at") or "",
                "message": r.get("message") or r.get("error") or "",
            }
        )
    return rows


def totals_from_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    rows = cost_rows(runs)
    return {
        "duration_sec": round(sum(r["duration_sec"] for r in rows), 3),
        "cost": round(sum(r["cost"] for r in rows), 6),
        "count": len(rows),
    }


def write_cost_xlsx(runs: list[dict[str, Any]], dest: str | Path, *, title: str = "成本表") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError as exc:
        # still write a CSV so the path is not empty
        csv_path = dest.with_suffix(".csv")
        _write_csv(runs, csv_path)
        raise RuntimeError(f"未安装 openpyxl，已先写出 {csv_path}。请 pip install openpyxl 后再导出 xlsx。") from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "成本"
    ws["A1"] = title
    ws["A1"].font = Font(bold=True, size=14)
    for col, head in enumerate(HEADERS, start=1):
        cell = ws.cell(2, col, head)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="3D4A6B")
        cell.alignment = Alignment(horizontal="center")
    rows = cost_rows(runs)
    for i, r in enumerate(rows, start=3):
        ws.cell(i, 1, r["step"])
        ws.cell(i, 2, r["duration_sec"])
        ws.cell(i, 3, r["cost"])
        ws.cell(i, 4, r["model"])
        ws.cell(i, 5, r["status"])
        ws.cell(i, 6, r["started_at"])
        ws.cell(i, 7, r["finished_at"])
        ws.cell(i, 8, r["message"])
    total = totals_from_runs(runs)
    trow = 3 + len(rows)
    ws.cell(trow, 1, "合计").font = Font(bold=True)
    ws.cell(trow, 2, total["duration_sec"]).font = Font(bold=True)
    ws.cell(trow, 3, total["cost"]).font = Font(bold=True)
    ws.cell(trow, 8, f"共 {total['count']} 步，时长与云费用均来自 runs 日志。")
    widths = (28, 12, 12, 24, 12, 22, 22, 48)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + i)].width = w
    wb.save(dest)
    return dest


def _write_csv(runs: list[dict[str, Any]], dest: Path) -> Path:
    import csv

    rows = cost_rows(runs)
    total = totals_from_runs(runs)
    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for r in rows:
            w.writerow([r["step"], r["duration_sec"], r["cost"], r["model"], r["status"], r["started_at"], r["finished_at"], r["message"]])
        w.writerow(["合计", total["duration_sec"], total["cost"], "", "", "", "", f"共 {total['count']} 步"])
    return dest
