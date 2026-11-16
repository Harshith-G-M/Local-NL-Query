"""Tests for follow-ups, join detection, data health, suggestions and report export."""

import io
import os
import sys

import openpyxl
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
import engine  # noqa: E402
from data_health import profile_dataset, suggest_questions  # noqa: E402
from engine import answer_question, build_dataset, llm_suggest_questions, run_query  # noqa: E402
from report import build_excel_report, build_html_report  # noqa: E402


def _hospital():
    admissions = pd.DataFrame({
        "Patient ID": [f"P{i}" for i in range(12)] + ["P0"],           # P0 duplicated below
        "Admit Date": ["01-Apr-2026", "31/03/2026", "2026-04-10"] * 4 + ["01-Apr-2026"],
        "department": ["Cardiology", "Neurology", "Oncology"] * 4 + ["Cardiology"],
        "Age": [30, 40, 50, 60, 35, 45, 55, 65, 38, 48, 58, 900, 30],   # 900 = outlier
        "Note": ["x"] * 13,                                              # constant column
    })
    admissions.loc[12] = admissions.loc[0]                               # exact duplicate row
    billing = pd.DataFrame({
        "PatientID": [f"P{i}" for i in range(12)],
        "Dept": ["Cardiology", "Neurology", "Oncology"] * 4,
        "Total Cost (INR)": [float(1000 * (i + 1)) for i in range(12)],
    })
    return build_dataset({"admissions": admissions, "billing": billing})


# ------------------------------------------------------------ join detection
def test_join_keys_found_by_name_and_by_values():
    ds = _hospital()
    pairs = {(a, ca, b, cb) for a, ca, b, cb in ds.join_keys}
    assert ("admissions", "Patient_ID", "billing", "PatientID") in pairs
    assert ("admissions", "department", "billing", "Dept") in pairs
    assert "JOIN on these" in ds.schema


def test_unrelated_columns_are_not_join_keys():
    ds = _hospital()
    assert not any(ca == "Note" or cb == "Note" for _, ca, _, cb in ds.join_keys)


def test_cross_table_query_runs():
    ds = _hospital()
    df = run_query(ds.conn, "SELECT a.department, SUM(b.Total_Cost_INR) AS t FROM admissions a "
                            "JOIN billing b ON a.Patient_ID = b.PatientID GROUP BY a.department")
    assert len(df) == 3


# ------------------------------------------------------------ data health
def test_health_flags_expected_problems():
    health = profile_dataset(_hospital())
    msgs = " | ".join(f"{i['table']}.{i['column']}: {i['message']}" for i in health["issues"])
    assert "duplicated row" in msgs                                   # exact duplicate
    assert "Age: " in msgs and "unusual value" in msgs                # 900 outlier
    assert "Note: contains only one distinct value" in msgs
    assert "mixed 3 date formats" in msgs
    assert health["warnings"] >= 1


def test_missing_values_severity_depends_on_share():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3, "b": ["x", "y"] * 15})
    df.loc[10, "a"] = None                                             # 1/30 = 3.3%  -> info
    df.loc[:4, "b"] = None                                             # 5/30 = 16.7% -> warning
    health = profile_dataset(build_dataset({"t": df}))
    sev = {i["column"]: i["severity"] for i in health["issues"] if "missing" in i["message"]}
    assert sev == {"a": "info", "b": "warning"}


def test_clean_data_reports_no_issues():
    df = pd.DataFrame({"city": ["A", "B", "C", "D"] * 5, "sales": list(range(100, 120))})
    assert profile_dataset(build_dataset({"t": df}))["issues"] == []


def test_suggestions_are_relevant_and_capped():
    qs = suggest_questions(_hospital(), n=5)
    assert 1 <= len(qs) <= 5 and len(set(qs)) == len(qs)
    assert all(q.endswith("?") or q.startswith("Show") for q in qs)


# ------------------------------------------------------------ follow-ups
def test_followup_prompt_contains_previous_sql(monkeypatch):
    seen = []

    def fake(model, prompt, system=""):
        if model == engine.SQL_MODEL:
            seen.append(prompt)
            return "SELECT department, COUNT(*) AS n FROM admissions GROUP BY department"
        return "ok"

    monkeypatch.setattr(engine, "call_ollama", fake)
    ds = _hospital()
    first = answer_question(ds, "admissions per department")
    answer_question(ds, "now only for Cardiology", history=[first])
    assert "Conversation so far" not in seen[0]
    assert "admissions per department" in seen[1]
    assert first["sql"] in seen[1]
    assert "build on the most recent SQL" in seen[1]


def test_history_is_limited_to_last_three_turns(monkeypatch):
    seen = []
    monkeypatch.setattr(engine, "call_ollama",
                        lambda m, p, system="": seen.append(p) or "SELECT 1 AS x")
    hist = [{"question": f"question number {i}", "sql": f"SELECT {i}"} for i in range(6)]
    engine.generate_sql("next?", "schema", history=hist)
    assert "question number 5" in seen[0] and "question number 3" in seen[0]
    assert "question number 2" not in seen[0]


# ------------------------------------------------------------ AI suggestions
def test_llm_suggestions_are_parsed_and_filtered(monkeypatch):
    monkeypatch.setattr(engine, "call_ollama", lambda m, p, system="": (
        "1. Which department has the highest total billing?\n"
        "- What is the average age of patients per department?\n"
        "Here are some ideas\n"                                    # chatter: not a question
        "Short?\n"))                                                # too short
    qs = llm_suggest_questions("schema", n=5)
    assert qs == ["Which department has the highest total billing?",
                  "What is the average age of patients per department?"]


def test_llm_suggestions_fail_safe_when_ollama_down(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no ollama")
    monkeypatch.setattr(engine, "call_ollama", boom)
    assert llm_suggest_questions("schema") == []


# ------------------------------------------------------------ report export
def _turns(ds):
    sqls = {
        "Admissions by department?": "SELECT department, COUNT(*) AS n FROM admissions GROUP BY department",
        "Cost by dept and payer?": "SELECT Dept, SUM(Total_Cost_INR) AS cost FROM billing GROUP BY Dept",
    }
    return [{"question": q, "sql": s, "explanation": f"Answer <b>{q}</b>", "result": run_query(ds.conn, s),
             "attempts": []} for q, s in sqls.items()]


def test_excel_report_structure_and_native_charts():
    ds = _hospital()
    wb = openpyxl.load_workbook(io.BytesIO(build_excel_report(_turns(ds), ds, profile_dataset(ds))))
    assert wb.sheetnames == ["Summary", "Data health", "Q1", "Q2"]
    assert len(wb["Q1"]._charts) == 1 and len(wb["Q2"]._charts) == 1
    assert wb["Summary"]["A1"].value == "NL2Query analysis report"
    questions = [c.value for c in wb["Summary"]["B"] if c.value]
    assert "Admissions by department?" in questions


def test_report_skips_failed_turns():
    ds = _hospital()
    turns = _turns(ds) + [{"question": "FAILED-QUESTION-XYZ", "sql": "x", "attempts": [], "error": "boom"}]
    wb = openpyxl.load_workbook(io.BytesIO(build_excel_report(turns, ds)))
    assert "Q3" not in wb.sheetnames
    assert "FAILED-QUESTION-XYZ" not in build_html_report(turns, ds)


def test_html_report_escapes_content_and_embeds_charts():
    ds = _hospital()
    html_doc = build_html_report(_turns(ds), ds, profile_dataset(ds))

    parts = html_doc.split("plotly-graph-div")
    for i in range(len(parts) - 1):
        print(f"\nMATCH {i + 1}:")
        print(parts[i][-100:] + "[MATCH]" + parts[i + 1][:100])

    assert "&lt;b&gt;" in html_doc and "<b>Admissions" not in html_doc
    assert html_doc.count('class="plotly-graph-div"') == 2
    assert "Data health" in html_doc

    print("TOTAL MATCHES:", html_doc.count("plotly-graph-div"))
    print("MATCH POSITIONS:", [
        i for i in range(len(html_doc))
        if html_doc.startswith("plotly-graph-div", i)
    ])
