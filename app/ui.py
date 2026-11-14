"""
Streamlit front-end for NL2Query.

Run with:
    streamlit run app/ui.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))
from charts import make_figure, suggest_chart  # noqa: E402
from data_health import profile_dataset, suggest_questions  # noqa: E402
from engine import (  # noqa: E402
    answer_question, build_dataset, llm_suggest_questions, read_tables,
)
from report import build_excel_report, build_html_report  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_FILES = {
    "admissions": "raw_admissions.csv",
    "billing": "raw_billing.csv",
    "bed_occupancy": "raw_bed_occupancy.csv",
    "staffing": "raw_staffing.csv",
}

st.set_page_config(page_title="NL2Query — Local AI Data Analyst", layout="wide")
st.title("🔒 NL2Query — a local AI data analyst")
st.caption(
    "Upload CSV or Excel files, ask questions in plain English (follow-ups work too), and get answers, "
    "charts and the SQL behind them. Everything runs on your machine through Ollama — "
    "your data never leaves it."
)

# ------------------------------------------------------------------ data source
with st.sidebar:
    st.header("1. Your data")
    uploads = st.file_uploader(
        "Upload CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True
    )
    st.caption("No file? The built-in hospital operations dataset is used.")


def load_dataset(uploaded_files):
    tables = {}
    if uploaded_files:
        for f in uploaded_files:
            tables.update(read_tables(f, f.name))
    else:
        for name, fname in SAMPLE_FILES.items():
            tables[name] = pd.read_csv(os.path.join(DATA_DIR, fname))
    return build_dataset(tables)


def reset_session():
    for k in ("turns", "reports", "ai_suggestions", "pending_q"):
        st.session_state.pop(k, None)


key = tuple((f.name, f.size) for f in uploads) if uploads else ("sample",)
if st.session_state.get("dataset_key") != key:
    try:
        ds = load_dataset(uploads)
    except Exception as exc:
        st.sidebar.error(f"Could not read file: {exc}")
        st.stop()
    reset_session()
    st.session_state.update(
        dataset=ds, dataset_key=key, health=profile_dataset(ds), suggestions=suggest_questions(ds)
    )

dataset = st.session_state["dataset"]
health = st.session_state["health"]
turns: list[dict] = st.session_state.setdefault("turns", [])
good_turns = [t for t in turns if "error" not in t]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Tables loaded")
    for name, df in dataset.tables.items():
        st.write(f"**{name}** — {len(df):,} rows, {len(df.columns)} columns")
    with st.expander("What the model sees (schema)"):
        st.code(dataset.schema, language="text")

    st.header("2. Export")
    if good_turns:
        cached = st.session_state.get("reports")
        if not cached or cached["n"] != len(good_turns):
            with st.spinner("Building report..."):
                cached = {
                    "n": len(good_turns),
                    "xlsx": build_excel_report(good_turns, dataset, health),
                    "html": build_html_report(good_turns, dataset, health),
                }
            st.session_state["reports"] = cached
        st.download_button(
            "📊 Excel report (with charts)", cached["xlsx"], file_name="nl2query_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "📄 HTML report (print to PDF)", cached["html"], file_name="nl2query_report.html",
            mime="text/html",
        )
        st.caption(f"Includes all {len(good_turns)} answered question(s) and the data health findings.")
    else:
        st.caption("Ask a question first — then export the whole session as a report.")

    if turns and st.button("Clear conversation"):
        reset_session()
        st.rerun()

# ------------------------------------------------------------------ data health
w, n = health["warnings"], health["notices"]
label = "🩺 Data health check — " + (
    "no problems found" if not (w or n)
    else f"{w} warning{'s' if w != 1 else ''}, {n} notice{'s' if n != 1 else ''}"
)
with st.expander(label, expanded=not turns and bool(w or n)):
    if not health["issues"]:
        st.success("No missing values, duplicates, unusual values or messy dates detected.")
    for issue in health["issues"]:
        where = f"`{issue['table']}.{issue['column']}`" if issue["column"] else f"`{issue['table']}`"
        icon = "⚠️" if issue["severity"] == "warning" else "ℹ️"
        st.markdown(f"{icon} {where} — {issue['message']}")
    tabs = st.tabs([t["table"] for t in health["tables"]])
    for tab, t in zip(tabs, health["tables"]):
        with tab:
            st.dataframe(pd.DataFrame(t["columns"]).rename(columns={
                "column": "Column", "type": "Type", "missing": "Missing", "missing_pct": "Missing %",
                "unique": "Distinct", "outliers": "Unusual values"}), hide_index=True)


# ------------------------------------------------------------------ suggestions
def _ask(q: str):
    st.session_state["pending_q"] = q


with st.expander("💡 Suggested questions", expanded=not turns):
    for i, q in enumerate(st.session_state["suggestions"]):
        st.button(q, key=f"sugg_{i}", on_click=_ask, args=(q,))
    if st.button("✨ More ideas from the AI"):
        with st.spinner("Thinking of questions..."):
            st.session_state["ai_suggestions"] = llm_suggest_questions(dataset.schema)
        if not st.session_state["ai_suggestions"]:
            st.warning("Couldn't get suggestions — is Ollama running?")
    for i, q in enumerate(st.session_state.get("ai_suggestions", [])):
        st.button("✨ " + q, key=f"ai_sugg_{i}", on_click=_ask, args=(q,))


# ------------------------------------------------------------------ conversation
def render_turn(i: int, t: dict):
    with st.chat_message("user"):
        st.write(t["question"])
    with st.chat_message("assistant"):
        if "error" in t:
            st.error(t["error"])
        else:
            st.write(t["explanation"])
            spec = suggest_chart(t["result"])
            if spec:
                st.plotly_chart(make_figure(t["result"], spec), key=f"chart_{i}")
            st.dataframe(t["result"])
            st.download_button("Download as CSV", t["result"].to_csv(index=False).encode("utf-8"),
                               file_name=f"result_{i + 1}.csv", mime="text/csv", key=f"csv_{i}")
        with st.expander("How this answer was produced (SQL + self-correction log)",
                         expanded="error" in t):
            for a in t["attempts"]:
                status = "✅ worked" if not a["error"] else (
                    "⛔ blocked by safety validator" if a["blocked"]
                    else "❌ failed — sent back to the model")
                st.markdown(f"**Attempt {a['attempt']}** — {status}")
                st.code(a["sql"], language="sql")
                if a["error"]:
                    st.caption(f"Reason: {a['error']}")


for i, t in enumerate(turns):
    render_turn(i, t)

typed = st.chat_input("Ask about your data — follow-ups work, e.g. “now split that by payer type”")
prompt = st.session_state.pop("pending_q", None) or typed

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Writing SQL, running it locally and checking the result..."):
            result = answer_question(dataset, prompt.strip(), history=good_turns)
    turns.append(result)
    st.rerun()

st.divider()
st.caption(
    "SQL: qwen2.5-coder:7b · Explanations: gemma3:4b · Safety: every query is validated "
    "(single SELECT only) and the database is opened read-only."
)
