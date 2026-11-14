"""
One-click report export for a finished analysis session.

    build_excel_report() -> .xlsx bytes : Summary, Data health, one sheet per question
                                           (with a native, editable Excel chart where useful)
    build_html_report()  -> .html str   : self-contained, interactive charts, prints cleanly
                                           to PDF from any browser (Ctrl+P -> Save as PDF)

Nothing is sent anywhere; both are built locally from the session's results.
"""

import html
import io
from datetime import datetime

import pandas as pd
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from charts import make_figure, suggest_chart

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def _stamp() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


def _successful(turns: list[dict]) -> list[dict]:
    return [t for t in turns if "error" not in t]


def _style_header(ws, row: int, ncols: int):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(vertical="center")


def _autowidth(ws, max_width: int = 60):
    for col in ws.columns:
        longest = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(longest + 2, max_width)


# ---------------------------------------------------------------- Excel
def _add_native_chart(ws, df: pd.DataFrame, spec: dict):
    n = len(df)
    anchor_col = len(df.columns) + 2
    cols = list(df.columns)
    kind, x, ys, color = spec["kind"], spec["x"], spec["y"], spec["color"]

    # two label columns: pivot so each series is its own column, written beside the raw data
    source, src_col0, header_row = df, 1, 1
    if color:
        pivot = df.pivot_table(index=x, columns=color, values=ys[0], aggfunc="sum").reset_index()
        pivot.columns = [str(c) for c in pivot.columns]
        for j, name in enumerate(pivot.columns):
            ws.cell(row=1, column=anchor_col + j, value=name).font = Font(bold=True)
            for i, v in enumerate(pivot[name], start=2):
                ws.cell(row=i, column=anchor_col + j, value=None if pd.isna(v) else v)
        source, src_col0 = pivot, anchor_col
        x_idx = src_col0
        y_idxs = list(range(src_col0 + 1, src_col0 + len(pivot.columns)))
        n = len(pivot)
        chart_row = n + 4
    else:
        x_idx = cols.index(x) + 1
        y_idxs = [cols.index(y) + 1 for y in ys]
        chart_row = 2

    if kind == "scatter":
        chart = ScatterChart()
        series = Series(Reference(ws, min_col=y_idxs[0], min_row=2, max_row=n + 1),
                        Reference(ws, min_col=x_idx, min_row=2, max_row=n + 1), title=ys[0])
        series.marker.symbol = "circle"
        series.graphicalProperties.line.noFill = True
        chart.series.append(series)
    else:
        chart = LineChart() if kind == "line" else BarChart()
        for yi in y_idxs:
            chart.add_data(Reference(ws, min_col=yi, min_row=header_row, max_row=n + 1), titles_from_data=True)
        chart.set_categories(Reference(ws, min_col=x_idx, min_row=2, max_row=n + 1))

    chart.title = spec["title"]
    chart.width, chart.height = 18, 9
    ws.add_chart(chart, f"{get_column_letter(anchor_col)}{chart_row}")


def build_excel_report(turns: list[dict], dataset, health: dict | None = None) -> bytes:
    ok = _successful(turns)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        wb = writer.book
        ws = wb.create_sheet("Summary", 0)
        ws["A1"] = "NL2Query analysis report"
        ws["A1"].font = Font(bold=True, size=16, color="1F3864")
        ws["A2"] = f"Generated {_stamp()} · built locally, no data left this machine"

        ws["A4"] = "Data analysed"
        ws["A4"].font = Font(bold=True)
        for i, (name, df) in enumerate(dataset.tables.items(), start=5):
            ws.cell(row=i, column=1, value=name)
            ws.cell(row=i, column=2, value=f"{len(df):,} rows × {len(df.columns)} columns")
        row = 5 + len(dataset.tables) + 1

        headers = ["#", "Question", "Answer", "SQL", "Sheet"]
        for j, h in enumerate(headers, start=1):
            ws.cell(row=row, column=j, value=h)
        _style_header(ws, row, len(headers))
        for i, t in enumerate(ok, start=1):
            r = row + i
            ws.cell(row=r, column=1, value=i)
            ws.cell(row=r, column=2, value=t["question"])
            ws.cell(row=r, column=3, value=t.get("explanation", ""))
            ws.cell(row=r, column=4, value=t["sql"])
            ws.cell(row=r, column=5, value=f"Q{i}")
            for c in (2, 3, 4):
                ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        for col, w in zip("ABCDE", (24, 40, 70, 70, 8)):
            ws.column_dimensions[col].width = w

        if health and health["issues"]:
            hs = wb.create_sheet("Data health")
            for j, h in enumerate(["Severity", "Table", "Column", "Finding"], start=1):
                hs.cell(row=1, column=j, value=h)
            _style_header(hs, 1, 4)
            for i, issue in enumerate(health["issues"], start=2):
                hs.cell(row=i, column=1, value=issue["severity"].upper())
                hs.cell(row=i, column=2, value=issue["table"])
                hs.cell(row=i, column=3, value=issue["column"] or "(whole table)")
                hs.cell(row=i, column=4, value=issue["message"])
            _autowidth(hs)

        for i, t in enumerate(ok, start=1):
            df = t["result"]
            df.to_excel(writer, sheet_name=f"Q{i}", index=False)
            qs = writer.sheets[f"Q{i}"]
            _style_header(qs, 1, len(df.columns))
            _autowidth(qs, 40)
            spec = suggest_chart(df)
            if spec:
                _add_native_chart(qs, df, spec)

        if "Sheet" in wb.sheetnames and len(wb.sheetnames) > 1:
            del wb["Sheet"]
    return buf.getvalue()


# ---------------------------------------------------------------- HTML
_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 20px;color:#222;line-height:1.5}
h1{color:#1F3864;margin-bottom:0}.sub{color:#666;margin-top:4px}
h2{color:#1F3864;border-bottom:2px solid #1F3864;padding-bottom:4px;margin-top:36px}
.q{background:#f4f6fb;border-left:4px solid #1F3864;padding:8px 14px;margin-top:28px;font-weight:600}
table{border-collapse:collapse;font-size:13px;margin:10px 0}th{background:#1F3864;color:#fff;text-align:left}
td,th{padding:5px 10px;border:1px solid #d5d9e4}pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:6px;white-space:pre-wrap;font-size:12.5px}
.warn{color:#b45309;font-weight:600}.info{color:#475569}.turn{page-break-inside:avoid}
@media print{body{margin:0;max-width:none}pre{background:#f1f5f9;color:#111}}
"""


def build_html_report(turns: list[dict], dataset, health: dict | None = None) -> str:
    ok = _successful(turns)
    e = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>NL2Query report</title>"
             f"<style>{_CSS}</style></head><body>",
             "<h1>NL2Query analysis report</h1>",
             f"<p class='sub'>Generated {_stamp()} · built locally · "
             "use your browser's Print → Save as PDF for a PDF copy</p>",
             "<h2>Data analysed</h2><table><tr><th>Table</th><th>Rows</th><th>Columns</th></tr>"]
    for name, df in dataset.tables.items():
        parts.append(f"<tr><td>{e(name)}</td><td>{len(df):,}</td><td>{len(df.columns)}</td></tr>")
    parts.append("</table>")

    if health and health["issues"]:
        parts.append("<h2>Data health</h2><table><tr><th>Severity</th><th>Where</th><th>Finding</th></tr>")
        for i in health["issues"]:
            where = f"{i['table']}.{i['column']}" if i["column"] else i["table"]
            parts.append(f"<tr><td class='{i['severity'].replace('warning', 'warn')}'>"
                         f"{i['severity'].upper()}</td><td>{e(where)}</td><td>{e(i['message'])}</td></tr>")
        parts.append("</table>")

    parts.append("<h2>Questions and answers</h2>")
    js_loaded = False
    for n, t in enumerate(ok, start=1):
        parts.append(f"<div class='turn'><div class='q'>Q{n}. {e(t['question'])}</div>")
        parts.append(f"<p>{e(t.get('explanation', ''))}</p>")
        spec = suggest_chart(t["result"])
        if spec:
            fig = make_figure(t["result"], spec)
            parts.append(fig.to_html(full_html=False, include_plotlyjs=not js_loaded))
            js_loaded = True
        df = t["result"]
        shown = df.head(25)
        parts.append(shown.to_html(index=False, border=0))
        if len(df) > len(shown):
            parts.append(f"<p class='info'>Showing first {len(shown)} of {len(df):,} rows.</p>")
        parts.append(f"<details open><summary>SQL used</summary><pre>{e(t['sql'])}</pre></details></div>")

    parts.append("</body></html>")
    return "".join(parts)
