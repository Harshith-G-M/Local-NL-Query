"""
Automatic chart selection for query results.

The chart type is chosen by rules on the *shape* of the result (column types,
row count, whether a column looks like a time axis). That keeps it instant,
deterministic and testable — no extra LLM call that could pick a wrong chart.
"""

import re

import pandas as pd

MAX_CHART_ROWS = 60
_TIME_NAME = re.compile(r"date|month|year|week|day|quarter|time", re.IGNORECASE)
_TIME_VALUE = re.compile(r"^\d{4}(-\d{2}){0,2}$")


def _is_time_axis(series: pd.Series) -> bool:
    if _TIME_NAME.search(str(series.name)):
        return True
    sample = series.dropna().astype(str).head(20)
    return len(sample) > 0 and all(_TIME_VALUE.match(v) for v in sample)


def suggest_chart(df: pd.DataFrame) -> dict | None:
    """
    Returns a spec {kind, x, y, color, title} or None when a chart wouldn't help.
    kind is one of: bar, line, scatter.
    """
    if df is None or df.empty or len(df) < 2 or len(df) > MAX_CHART_ROWS:
        return None

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]

    # one label column + one or more measures
    if len(categorical) == 1 and numeric:
        x = categorical[0]
        kind = "line" if _is_time_axis(df[x]) else "bar"
        return {"kind": kind, "x": x, "y": numeric, "color": None,
                "title": f"{', '.join(numeric)} by {x}"}

    # two label columns + one measure -> second label becomes the series
    if len(categorical) == 2 and len(numeric) == 1:
        a, b = categorical
        x, color = (b, a) if _is_time_axis(df[b]) else (a, b)
        kind = "line" if _is_time_axis(df[x]) else "bar"
        return {"kind": kind, "x": x, "y": numeric, "color": color,
                "title": f"{numeric[0]} by {x} and {color}"}

    # pure numbers: relationship between two measures
    if not categorical and len(numeric) == 2 and len(df) >= 3:
        return {"kind": "scatter", "x": numeric[0], "y": [numeric[1]], "color": None,
                "title": f"{numeric[1]} vs {numeric[0]}"}

    return None


def make_figure(df: pd.DataFrame, spec: dict):
    import plotly.express as px

    kind, x, y, color = spec["kind"], spec["x"], spec["y"], spec["color"]
    y_arg = y if len(y) > 1 else y[0]

    if kind == "bar":
        fig = px.bar(df, x=x, y=y_arg, color=color, barmode="group", title=spec["title"])
    elif kind == "line":
        fig = px.line(df, x=x, y=y_arg, color=color, markers=True, title=spec["title"])
    else:
        fig = px.scatter(df, x=x, y=y_arg, title=spec["title"])

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig
