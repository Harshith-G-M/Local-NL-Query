
"""
Automatic and question-aware chart selection for query results.

Chart selection is deterministic and does not require an additional LLM call.
"""

import re

import pandas as pd

MAX_CHART_ROWS = 60

_TIME_NAME = re.compile(
    r"date|month|year|week|day|quarter|time", re.IGNORECASE
)
_TIME_VALUE = re.compile(r"^\d{4}(-\d{2}){0,2}$")


def _is_time_axis(series: pd.Series) -> bool:
    if _TIME_NAME.search(str(series.name)):
        return True

    sample = series.dropna().astype(str).head(20)
    return len(sample) > 0 and all(
        _TIME_VALUE.match(v) for v in sample
    )


def _numeric_columns(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def suggest_chart(
    df: pd.DataFrame,
    question: str = "",
) -> dict | None:
    """
    Returns a chart specification or None.

    Supported kinds:
    bar, line, pie, donut, scatter, histogram, box, area.
    """
    if (
        df is None
        or df.empty
        or len(df) < 2
        or len(df) > MAX_CHART_ROWS
    ):
        return None

    numeric = _numeric_columns(df)
    categorical = [c for c in df.columns if c not in numeric]
    q = question.lower()

    # ------------------------------------------------ explicit requests

    # Pie / donut: category + one numeric measure
    if re.search(
        r"\b(pie|donut|doughnut|proportion|percentage|percent|share)\b", q
    ):
        if len(categorical) == 1 and len(numeric) == 1:
            return {
                "kind": "donut" if re.search(r"\b(donut|doughnut)\b", q) else "pie",
                "x": categorical[0],
                "y": numeric,
                "color": None,
                "title": f"{numeric[0]} by {categorical[0]}",
            }

    # Histogram: numeric distribution
    if re.search(r"\b(histogram|distribution)\b", q):
        if numeric:
            return {
                "kind": "histogram",
                "x": numeric[0],
                "y": [],
                "color": None,
                "title": f"Distribution of {numeric[0]}",
            }

    # Scatter: relationship between two numeric measures
    if re.search(r"\b(scatter|relationship|correlation)\b", q):
        if len(numeric) >= 2:
            return {
                "kind": "scatter",
                "x": numeric[0],
                "y": [numeric[1]],
                "color": None,
                "title": f"{numeric[1]} vs {numeric[0]}",
            }

    # Explicit chart type requests
    if re.search(r"\b(line|trend)\b", q):
        if categorical and numeric:
            x = next(
                (c for c in categorical if _is_time_axis(df[c])),
                categorical[0],
            )
            return {
                "kind": "line",
                "x": x,
                "y": numeric,
                "color": None,
                "title": f"{', '.join(numeric)} by {x}",
            }

    if re.search(r"\b(area)\b", q):
        if categorical and numeric:
            x = next(
                (c for c in categorical if _is_time_axis(df[c])),
                categorical[0],
            )
            return {
                "kind": "area",
                "x": x,
                "y": numeric,
                "color": None,
                "title": f"{', '.join(numeric)} by {x}",
            }

    if re.search(r"\b(box|boxplot|box plot)\b", q):
        if numeric:
            return {
                "kind": "box",
                "x": categorical[0] if categorical else None,
                "y": numeric,
                "color": None,
                "title": f"Distribution of {', '.join(numeric)}",
            }

    if re.search(r"\b(bar|column chart)\b", q):
        if categorical and numeric:
            return {
                "kind": "bar",
                "x": categorical[0],
                "y": numeric,
                "color": None,
                "title": f"{', '.join(numeric)} by {categorical[0]}",
            }

    # ------------------------------------------------ automatic selection

    # One label column + one or more measures
    if len(categorical) == 1 and numeric:
        x = categorical[0]
        kind = "line" if _is_time_axis(df[x]) else "bar"

        return {
            "kind": kind,
            "x": x,
            "y": numeric,
            "color": None,
            "title": f"{', '.join(numeric)} by {x}",
        }

    # Two label columns + one measure
    if len(categorical) == 2 and len(numeric) == 1:
        a, b = categorical
        x, color = (b, a) if _is_time_axis(df[b]) else (a, b)
        kind = "line" if _is_time_axis(df[x]) else "bar"

        return {
            "kind": kind,
            "x": x,
            "y": numeric,
            "color": color,
            "title": f"{numeric[0]} by {x} and {color}",
        }

    # Pure numbers: relationship between two measures
    if not categorical and len(numeric) == 2 and len(df) >= 3:
        return {
            "kind": "scatter",
            "x": numeric[0],
            "y": [numeric[1]],
            "color": None,
            "title": f"{numeric[1]} vs {numeric[0]}",
        }

    return None


def suggest_charts(
    df: pd.DataFrame,
    question: str = "",
) -> list[dict]:
    """
    Return up to three useful chart specifications.

    Uses the existing question-aware selector as the primary chart,
    then adds complementary charts for broad analysis questions.
    """
    primary = suggest_chart(df, question)

    if primary is None:
        return []

    charts = [primary]

    if (
        df is None
        or df.empty
        or len(df) < 2
        or len(df) > MAX_CHART_ROWS
    ):
        return charts

    numeric = _numeric_columns(df)
    categorical = [c for c in df.columns if c not in numeric]

    q = question.lower()

    overview_patterns = (
        "overview",
        "performance",
        "analyze",
        "analysis",
        "insights",
        "comprehensive",
        "overall",
    )

    if not any(word in q for word in overview_patterns):
        return charts

    # Add a category comparison when it is different
    # from the primary chart.
    if categorical and numeric:
        x = categorical[0]

        if not (
            primary["kind"] == "bar"
            and primary["x"] == x
        ):
            charts.append({
                "kind": "bar",
                "x": x,
                "y": numeric[:1],
                "color": None,
                "title": f"{numeric[0]} by {x}",
            })

    # A donut chart is useful for non-time categories.
    if (
        len(categorical) == 1
        and len(numeric) == 1
        and not _is_time_axis(df[categorical[0]])
        and primary["kind"] not in ("pie", "donut")
    ):
        charts.append({
            "kind": "donut",
            "x": categorical[0],
            "y": numeric[:1],
            "color": None,
            "title": (
                f"{numeric[0]} contribution "
                f"by {categorical[0]}"
            ),
        })

    # Remove duplicate chart specifications.
    unique = []
    seen = set()

    for spec in charts:
        key = (
            spec["kind"],
            spec["x"],
            tuple(spec["y"]),
            spec.get("color"),
        )

        if key not in seen:
            seen.add(key)
            unique.append(spec)

    return unique[:3]

def make_figure(df: pd.DataFrame, spec: dict):
    import plotly.express as px

    CHART_COLORS = [
    "#4E79A7",  # Blue
    "#59A14F",  # Green
    "#F28E2B",  # Orange
    "#B07AA1",  # Purple
    "#76B7B2",  # Teal
    "#EDC948",  # Yellow
    "#E15759",  # Red
    "#9C755F",  # Brown
]

    kind = spec["kind"]
    x = spec["x"]
    y = spec["y"]
    color = spec.get("color")
    title = spec["title"]

    y_arg = y if len(y) > 1 else (y[0] if y else None)

    if kind == "bar":
        fig = px.bar(
            df, x=x, y=y_arg, color=color,
            barmode="group", title=title,color_discrete_sequence=CHART_COLORS,
        )

    elif kind == "line":
        fig = px.line(
            df, x=x, y=y_arg, color=color,
            markers=True, title=title,color_discrete_sequence=CHART_COLORS,
        )

    elif kind == "pie":
        fig = px.pie(
            df, names=x, values=y[0], title=title,color_discrete_sequence=CHART_COLORS
        )

    elif kind == "donut":
        fig = px.pie(
            df, names=x, values=y[0],
            hole=0.45, title=title,color_discrete_sequence=CHART_COLORS
        )

    elif kind == "scatter":
        fig = px.scatter(
            df, x=x, y=y_arg, title=title,color_discrete_sequence=CHART_COLORS,
        )

    elif kind == "histogram":
        fig = px.histogram(
            df, x=x, title=title,color_discrete_sequence=CHART_COLORS,
        )

    elif kind == "box":
        fig = px.box(
            df, x=x, y=y_arg, color=color,color_discrete_sequence=CHART_COLORS,
            title=title,
        )

    elif kind == "area":
        fig = px.area(
            df, x=x, y=y_arg, color=color,color_discrete_sequence=CHART_COLORS,
            title=title,
        )

    else:
        return None

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
        color="#222222",
        family="Arial, sans-serif",
        size=12,
    ),
    title_font=dict(color="#111111"),
    legend_title_text="",
    margin=dict(l=10, r=10, t=50, b=10),
)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zerolinecolor="#CBD5E1",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zerolinecolor="#CBD5E1",
    )

    return fig
