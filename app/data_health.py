"""
Automatic data health check and question suggestions.

Everything here is deterministic (no LLM): it runs instantly when data is loaded
and its findings are unit-tested. Issues are reported with a severity:
    warning — likely to distort answers (duplicates, many missing values, unparsed dates)
    info    — worth knowing (some missing values, outliers, mixed date formats fixed)
"""

import re

import pandas as pd

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MISSING_WARN_PCT = 5.0


def _human(name: str) -> str:
    return name.replace("_", " ").lower()


def _is_date_col(df: pd.DataFrame, col: str, noted: set[str]) -> bool:
    if col in noted:
        return True
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or "date" not in col.lower():
        return False
    values = s.dropna().astype(str)
    return len(values) > 0 and values.map(lambda v: bool(_ISO.match(v))).all()


def _is_id_like(s: pd.Series, name: str) -> bool:
    n = s.notna().sum()
    return bool(n) and (s.nunique() == n or name.lower().endswith("id") or name.lower().endswith("_id"))


def profile_dataset(dataset) -> dict:
    """Returns {'tables': [...], 'issues': [...], 'warnings': int, 'notices': int}."""
    tables, issues = [], []

    def add(severity, table, column, message):
        issues.append({"severity": severity, "table": table, "column": column, "message": message})

    for name, df in dataset.tables.items():
        rows = len(df)
        noted = {n["column"] for n in dataset.date_notes.get(name, [])}
        cols = []

        dup = int(df.duplicated().sum())
        if dup:
            add("warning", name, None, f"{dup} fully duplicated row{'s' if dup > 1 else ''}")

        for col in df.columns:
            s = df[col]
            missing = int(s.isna().sum())
            pct = round(100 * missing / rows, 1) if rows else 0.0
            is_num = pd.api.types.is_numeric_dtype(s)
            is_date = _is_date_col(df, col, noted)
            info = {
                "column": col,
                "type": "number" if is_num else "date" if is_date else "text",
                "missing": missing,
                "missing_pct": pct,
                "unique": int(s.nunique(dropna=True)),
                "outliers": 0,
            }

            if missing:
                add("warning" if pct >= MISSING_WARN_PCT else "info", name, col,
                    f"{missing} missing value{'s' if missing > 1 else ''} ({pct}%)")

            if rows > 1 and info["unique"] == 1:
                add("info", name, col, "contains only one distinct value")

            if is_num and s.notna().sum() >= 8 and info["unique"] > 2 and not _is_id_like(s, col):
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    n_out = int(((s < lo) | (s > hi)).sum())
                    info["outliers"] = n_out
                    if n_out:
                        add("info", name, col,
                            f"{n_out} unusual value{'s' if n_out > 1 else ''} outside {lo:,.4g} to {hi:,.4g} "
                            f"(IQR rule)")

            cols.append(info)

        for note in dataset.date_notes.get(name, []):
            if note["formats"] > 1:
                add("info", name, note["column"],
                    f"mixed {note['formats']} date formats — standardised to YYYY-MM-DD")
            if note["unparsed"]:
                add("warning", name, note["column"],
                    f"{note['unparsed']} value(s) could not be read as dates and were left empty")

        tables.append({"table": name, "rows": rows, "columns": cols, "duplicates": dup})

    order = {"warning": 0, "info": 1}
    issues.sort(key=lambda i: (order[i["severity"]], i["table"]))
    return {
        "tables": tables,
        "issues": issues,
        "warnings": sum(i["severity"] == "warning" for i in issues),
        "notices": sum(i["severity"] == "info" for i in issues),
    }


def suggest_questions(dataset, n: int = 5) -> list[str]:
    """Builds starter questions from the schema: counts, averages, trends, joins, data gaps."""
    profile = {t["table"]: t for t in profile_dataset(dataset)["tables"]}
    parts = {}
    for name, df in dataset.tables.items():
        noted = {x["column"] for x in dataset.date_notes.get(name, [])}
        cols = profile[name]["columns"]
        parts[name] = {
            "cats": [c["column"] for c in cols if c["type"] == "text" and 2 <= c["unique"] <= 20],
            "nums": [c["column"] for c in cols
                     if c["type"] == "number" and not _is_id_like(df[c["column"]], c["column"])],
            "dates": [c["column"] for c in cols if c["type"] == "date"],
            "missing": [c["column"] for c in cols if c["missing"]],
        }

    by_size = sorted(dataset.tables, key=lambda t: -len(dataset.tables[t]))
    pool: dict[str, list[str]] = {k: [] for k in ("count", "join", "trend", "avg", "top", "missing")}

    for t in by_size:
        p, h = parts[t], _human(t)
        if p["cats"]:
            pool["count"].append(f"How many {h} records are there for each {_human(p['cats'][0])}?")
        if p["cats"] and p["nums"]:
            pool["avg"].append(f"What is the average {_human(p['nums'][0])} by {_human(p['cats'][0])}?")
            cat = p["cats"][-1]
            pool["top"].append(f"Which {_human(cat)} has the highest total {_human(p['nums'][-1])}?")
        if p["dates"]:
            what = f"total {_human(p['nums'][0])}" if p["nums"] else f"number of {h} records"
            pool["trend"].append(f"Show the monthly {what} over time")
        if p["missing"]:
            pool["missing"].append(f"How many {h} records have a missing {_human(p['missing'][0])}?")

    for ta, ca, tb, cb in dataset.join_keys:
        cats_a = [c for c in parts[ta]["cats"] if c != ca]
        nums_b = parts[tb]["nums"]
        if cats_a and nums_b:
            pool["join"].append(
                f"What is the average {_human(nums_b[0])} for each {_human(cats_a[0])}?")
            break

    out: list[str] = []
    for key in ("count", "join", "trend", "avg", "top", "missing"):
        for q in pool[key][:1]:
            if q not in out:
                out.append(q)
    for key in ("avg", "top", "trend", "count"):
        for q in pool[key][1:]:
            if q not in out:
                out.append(q)
    return out[:n]
