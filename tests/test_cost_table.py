from pathlib import Path

from recast.compose.cost import cost_rows, totals_from_runs, write_cost_xlsx
from recast.store.db import Store


def test_cost_totals_match_runs(tmp_path: Path):
    store = Store(tmp_path / "t.db")
    store.record_run(task_id="t1", drama_id="d1", step="扫描", status="ok", duration_sec=1.25, cost=0, model="cpu")
    store.record_run(task_id="t1", drama_id="d1", step="人物替换", status="ok", duration_sec=8.5, cost=0, model="inswapper")
    store.record_run(task_id="t1", drama_id="d1", step="云调用", status="ok", duration_sec=2.0, cost=0.0123, model="deepseek")
    runs = store.list_runs(task_id="t1")
    tot = totals_from_runs(runs)
    assert tot["count"] == 3
    assert tot["duration_sec"] == 11.75
    assert abs(tot["cost"] - 0.0123) < 1e-9
    rows = cost_rows(runs)
    assert sum(r["duration_sec"] for r in rows) == tot["duration_sec"]
    dest = tmp_path / "cost.xlsx"
    write_cost_xlsx(runs, dest, title="测试")
    assert dest.is_file()
    from openpyxl import load_workbook

    wb = load_workbook(dest)
    ws = wb.active
    # header on row 2, data starts 3, total after 3 rows → row 6
    assert ws.cell(6, 1).value == "合计"
    assert float(ws.cell(6, 2).value) == 11.75
    assert float(ws.cell(6, 3).value) == 0.0123
