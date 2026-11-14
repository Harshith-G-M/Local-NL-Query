"""
Core engine for NL2Query — a local, privacy-first AI data analyst.

Pipeline:
    uploaded files --> cleaned tables in SQLite (read-only) --> schema profile
    question --> LLM writes SQL --> safety validator --> execute
         \\__ on execution error: error is fed back to the LLM and it retries
    result --> LLM explains in plain English --> (charts.py picks a chart)
"""

import re
import psutil
import os
import time
import sqlite3
from dataclasses import dataclass, field

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
SQL_MODEL = "qwen2.5-coder:7b"   # code-tuned: best SQL accuracy of the local set
EXPLAIN_MODEL = "gemma3:4b"      # fast, good enough for a short summary
MAX_ATTEMPTS = 3                 # 1 first try + up to 2 self-corrections
MAX_RESULT_ROWS = 1000


# ---------------------------------------------------------------------------
# 1. Loading + cleaning data
# ---------------------------------------------------------------------------

def clean_name(name: str) -> str:
    """'Total Cost (INR)' -> 'Total_Cost_INR'. The LLM never has to quote or escape anything."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_")
    if not cleaned:
        cleaned = "col"
    if cleaned[0].isdigit():
        cleaned = "t_" + cleaned
    return cleaned


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


_NUMERIC_DATE = re.compile(r"^(\d{1,2})([/\-.])(\d{1,2})\2(\d{2,4})$")


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Parses a column that mixes several date formats (common in messy exports).

    - ISO dates (2026-04-10) are parsed as-is.
    - Numeric dates like 31/03/2026 or 04-16-2026 are ambiguous (day-first or month-first?),
      so the order is *inferred per separator style* from the unambiguous rows: if a first
      number is ever > 12 the style is day-first; if a second number is ever > 12 it is
      month-first. With no evidence, day-first is assumed (Indian/UK convention).
    - Anything else (01-Apr-2026, 5 April 2026 ...) is parsed generically.
    """
    text = series.astype("string").str.strip()
    parts = []

    is_iso = text.str.match(r"^\d{4}-\d{2}-\d{2}").fillna(False).astype(bool)
    if is_iso.any():
        parts.append(pd.to_datetime(text[is_iso], format="ISO8601", errors="coerce"))

    numeric = text.str.extract(_NUMERIC_DATE)
    is_numeric = numeric[0].notna() & ~is_iso
    for sep in numeric.loc[is_numeric, 1].unique():
        grp = is_numeric & (numeric[1] == sep)
        first = pd.to_numeric(numeric.loc[grp, 0])
        second = pd.to_numeric(numeric.loc[grp, 2])
        month_first = bool((second > 12).any() and not (first > 12).any())
        parts.append(pd.to_datetime(text[grp], format="mixed", dayfirst=not month_first, errors="coerce"))

    is_other = text.notna() & ~is_iso & ~is_numeric
    if is_other.any():
        parts.append(pd.to_datetime(text[is_other], format="mixed", dayfirst=True, errors="coerce"))

    if not parts:
        return pd.Series(pd.NaT, index=series.index)
    return pd.concat(parts).reindex(series.index)


def _date_pattern(value: str) -> str:
    """'01-Apr-2026' -> '99-a-9999'. Used to count how many formats a column mixes."""
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d", "9", str(value)))


def _normalise_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Standardises date columns to ISO text (YYYY-MM-DD) so SQLite date functions work.
    Returns the frame plus notes about what was changed (for the data health check).
    """
    notes = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            df[col] = series.dt.strftime("%Y-%m-%d")
            continue
        if "date" not in col.lower() or pd.api.types.is_numeric_dtype(series):
            continue
        parsed = _parse_dates(series)
        non_null = int(series.notna().sum())
        if non_null and parsed.notna().sum() / non_null >= 0.8:
            formats = series.dropna().astype(str).str.strip().map(_date_pattern).nunique()
            unparsed = int((series.notna() & parsed.isna()).sum())
            notes.append({"column": col, "formats": int(formats), "unparsed": unparsed})
            df[col] = parsed.dt.strftime("%Y-%m-%d")
    return df, notes


def prepare_with_notes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy()
    df.columns = _dedupe([clean_name(c) for c in df.columns])
    df = df.dropna(how="all").reset_index(drop=True)
    return _normalise_dates(df)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_with_notes(df)[0]


def read_tables(source, filename: str) -> dict[str, pd.DataFrame]:
    """
    Reads a CSV or Excel file (path or file-like object) into {table_name: DataFrame}.
    Each Excel sheet becomes its own table.
    """
    stem, _, ext = filename.rpartition(".")
    ext = ext.lower()
    base = clean_name(stem).lower()

    if ext == "csv":
        return {base: pd.read_csv(source)}
    if ext in ("xlsx", "xls"):
        sheets = pd.read_excel(source, sheet_name=None)
        if len(sheets) == 1:
            return {base: next(iter(sheets.values()))}
        return {f"{base}_{clean_name(s).lower()}": df for s, df in sheets.items()}
    raise ValueError(f"Unsupported file type: .{ext} (use CSV or Excel)")


# ---------------------------------------------------------------------------
# 2. Dataset: SQLite connection + schema profile for the prompt
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    conn: sqlite3.Connection
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    schema: str = ""
    date_notes: dict[str, list[dict]] = field(default_factory=dict)
    join_keys: list[tuple[str, str, str, str]] = field(default_factory=list)


def _describe_table(name: str, df: pd.DataFrame) -> str:
    lines = [f"Table {name} ({len(df)} rows):"]
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                lines.append(f"  - {col} (number) range {s.min():g} to {s.max():g}")
            else:
                lines.append(f"  - {col} (number)")
        else:
            uniques = s.dropna().unique()
            if "date" in col.lower() and len(uniques):
                lines.append(f"  - {col} (date, YYYY-MM-DD) from {min(uniques)} to {max(uniques)}")
            elif 0 < len(uniques) <= 12:
                vals = ", ".join(f"'{u}'" for u in uniques)
                lines.append(f"  - {col} (text) values: {vals}")
            else:
                vals = ", ".join(f"'{u}'" for u in uniques[:3])
                lines.append(f"  - {col} (text) e.g. {vals}")
    return "\n".join(lines)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_join_keys(tables: dict[str, pd.DataFrame], limit: int = 12) -> list[tuple[str, str, str, str]]:
    """
    Finds columns in different tables that hold the same values, so the model knows what to JOIN on.
    Text columns qualify when their values overlap >= 90% (and there are >= 3 distinct values);
    columns with the same normalised name (PatientID / Patient_ID) qualify at >= 50% overlap.
    """
    values = {}
    for t, df in tables.items():
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                values[(t, c)] = set(df[c].dropna().astype(str))

    found = []
    names = list(tables)
    for i, ta in enumerate(names):
        for tb in names[i + 1:]:
            for (t1, c1), v1 in values.items():
                if t1 != ta:
                    continue
                for (t2, c2), v2 in values.items():
                    if t2 != tb or not v1 or not v2:
                        continue
                    smaller = min(len(v1), len(v2))
                    overlap = len(v1 & v2) / smaller
                    same_name = _norm(c1) == _norm(c2)
                    if (same_name and overlap >= 0.5) or (smaller >= 3 and overlap >= 0.9):
                        found.append((ta, c1, tb, c2))
    return found[:limit]


def build_dataset(tables: dict[str, pd.DataFrame]) -> Dataset:
    """Loads cleaned tables into an in-memory SQLite DB and locks it read-only."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    prepared, date_notes = {}, {}
    for name, df in tables.items():
        clean_df, notes = prepare_with_notes(df)
        clean_df.to_sql(name, conn, index=False)
        prepared[name] = clean_df
        date_notes[name] = notes

    # Layer 2 of safety: even if a write slipped past the validator, SQLite itself refuses it.
    conn.execute("PRAGMA query_only = ON")

    join_keys = find_join_keys(prepared) if len(prepared) > 1 else []
    schema = "\n\n".join(_describe_table(n, d) for n, d in prepared.items())
    if join_keys:
        hints = "\n".join(f"  - {a}.{ca} = {b}.{cb}" for a, ca, b, cb in join_keys)
        schema += "\n\nColumns holding the same values in different tables (JOIN on these):\n" + hints
    return Dataset(conn=conn, tables=prepared, schema=schema, date_notes=date_notes, join_keys=join_keys)


# ---------------------------------------------------------------------------
# 3. LLM calls
# ---------------------------------------------------------------------------

SQL_SYSTEM_PROMPT = """You are an expert SQLite query writer.
Rules:
- Output ONLY one SQL query. No explanation, no markdown.
- Only SELECT queries (a WITH ... SELECT is allowed). Never modify data.
- Use only the tables and columns in the schema, spelled exactly as given.
- Dates are text in YYYY-MM-DD format: use strftime(), date() and julianday() for date logic
  (e.g. length of stay = julianday(end) - julianday(start); month = strftime('%Y-%m', col)).
- To combine tables, JOIN on the columns that hold the same identifier.
- Give computed columns short readable aliases.
- If the question cannot be answered from the schema, output exactly: NOT_POSSIBLE
"""

def call_ollama(model: str, prompt: str, system: str = "") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0},
    }

    start_time = time.perf_counter()

    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()

    elapsed_time = time.perf_counter() - start_time
    data = resp.json()

    response_text = data["response"].strip()

    # Ollama reports durations in nanoseconds
    load_time = data.get("load_duration", 0) / 1e9
    prompt_time = data.get("prompt_eval_duration", 0) / 1e9
    generation_time = data.get("eval_duration", 0) / 1e9
    total_time = data.get("total_duration", 0) / 1e9

    input_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)

    tokens_per_second = (
        output_tokens / generation_time if generation_time > 0 else 0
    )

    print("\n========== OLLAMA PERFORMANCE ==========")
    print(f"Model: {model}")
    print(f"HTTP response time: {elapsed_time:.3f} seconds")
    print(f"Total Ollama time: {total_time:.3f} seconds")
    print(f"Model loading time: {load_time:.3f} seconds")
    print(f"Prompt processing: {prompt_time:.3f} seconds")
    print(f"Generation time: {generation_time:.3f} seconds")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")
    print("========================================\n")

    return response_text

def extract_sql(raw: str) -> str:
    """Pulls the SQL out of a model reply: removes fences, comments and chatter before SELECT/WITH."""
    text = raw.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", " ", text)
    start = re.search(r"\b(SELECT|WITH)\b", text, flags=re.IGNORECASE)
    if start and text.strip() != "NOT_POSSIBLE":
        text = text[start.start():]
    return text.strip()


def _history_block(history: list[dict] | None) -> str:
    if not history:
        return ""
    lines = ["Conversation so far (oldest first):"]
    for h in history[-3:]:
        lines.append(f"Q: {h['question']}\nSQL: {h['sql']}")
    lines.append(
        "If the new question refers back to these (e.g. 'break that down by payer type', "
        "'only for Cardiology', 'and the same for staffing'), build on the most recent SQL. "
        "Otherwise write a fresh query."
    )
    return "\n".join(lines) + "\n\n"


def generate_sql(question: str, schema: str, previous_sql: str = "", error: str = "",
                 history: list[dict] | None = None) -> str:
    prompt = f"Schema:\n{schema}\n\n{_history_block(history)}Question: {question}\n"
    if previous_sql and error:
        prompt += (
            f"\nYour previous query failed.\nPrevious query: {previous_sql}\n"
            f"Database error: {error}\nWrite a corrected query.\n"
        )
    prompt += "\nSQL:"
    return extract_sql(call_ollama(SQL_MODEL, prompt, system=SQL_SYSTEM_PROMPT))


# ---------------------------------------------------------------------------
# 4. Safety validation (layer 1 of 2)
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "pragma", "replace", "vacuum", "reindex", "truncate",
]


class UnsafeQueryError(Exception):
    """Raised for queries that are not a single read-only SELECT."""


def validate_sql(sql: str) -> str:
    """Returns the cleaned query, or raises UnsafeQueryError."""
    if sql.strip() == "NOT_POSSIBLE":
        raise UnsafeQueryError("The model could not map this question to the available data.")

    stripped = sql.strip().rstrip(";").strip()
    # ignore the contents of string literals so a value like 'Update' isn't a false alarm
    skeleton = re.sub(r"'(?:[^']|'')*'", "''", stripped)

    if ";" in skeleton:
        raise UnsafeQueryError("Multiple statements are not allowed.")
    if not re.match(r"^\s*(SELECT|WITH)\b", skeleton, flags=re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT statements are allowed.")

    lowered = skeleton.lower()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            raise UnsafeQueryError(f"Forbidden keyword detected: {kw}")
    return stripped


# ---------------------------------------------------------------------------
# 5. Execute, explain, and the self-correcting loop
# ---------------------------------------------------------------------------

def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    process = psutil.Process(os.getpid())

    memory_before = process.memory_info().rss / (1024 ** 2)
    start_time = time.perf_counter()

    result = pd.read_sql_query(sql, conn).head(MAX_RESULT_ROWS)

    execution_time = time.perf_counter() - start_time
    memory_after = process.memory_info().rss / (1024 ** 2)
    memory_change = memory_after - memory_before

    print("\n========== SQL PERFORMANCE ==========")
    print(f"SQL execution time: {execution_time:.4f} seconds")
    print(f"Rows returned: {len(result)}")
    print(f"RAM before query: {memory_before:.2f} MB")
    print(f"RAM after query: {memory_after:.2f} MB")
    print(f"RAM change: {memory_change:+.2f} MB")
    print("=====================================\n")

    return result

def explain_result(question: str, sql: str, result_df: pd.DataFrame) -> str:
    if result_df.empty:
        return "The query ran successfully but returned no rows."
    preview = result_df.head(10).to_string(index=False)
    prompt = (
        f"Question: {question}\nSQL used: {sql}\nResult (first rows):\n{preview}\n\n"
        "In 2-3 plain-English sentences, answer the question using this result. "
        "Quote concrete numbers. Do not mention SQL or tables."
    )
    return call_ollama(EXPLAIN_MODEL, prompt)


def answer_question(dataset: Dataset, question: str, history: list[dict] | None = None,
                    max_attempts: int = MAX_ATTEMPTS) -> dict:
    """
    Returns a dict with: question, sql, attempts (log of each try), and either
    error  OR  result (DataFrame) + explanation.
    Execution errors are fed back to the model for correction. Safety rejections are final.
    `history` = earlier successful turns (dicts with question + sql) so follow-ups work.
    """
    attempts: list[dict] = []
    sql, error = "", ""
    request_start = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        raw_sql = generate_sql(question, dataset.schema, previous_sql=sql, error=error, history=history)
        sql = raw_sql

        try:
            safe_sql = validate_sql(raw_sql)
        except UnsafeQueryError as exc:
            attempts.append({"attempt": attempt, "sql": raw_sql, "error": str(exc), "blocked": True})
            return {"question": question, "sql": raw_sql, "attempts": attempts, "error": str(exc)}

        try:
            result_df = run_query(dataset.conn, safe_sql)
        except Exception as exc:  # execution failure -> feed back and retry
            error = str(exc)
            sql = safe_sql
            attempts.append({"attempt": attempt, "sql": safe_sql, "error": error, "blocked": False})
            continue

        attempts.append({"attempt": attempt, "sql": safe_sql, "error": None, "blocked": False})
        explanation = explain_result(question, safe_sql, result_df)
        total_time = time.perf_counter() - request_start

        print("\n========== REQUEST PERFORMANCE ==========")
        print(f"Total response time: {total_time:.3f} seconds")
        print("=========================================\n")

        return {
            "question": question,
            "sql": safe_sql,
            "attempts": attempts,
            "result": result_df,
            "explanation": explanation,
            "total_execution_time": round(total_time, 3),
        }

    return {
        "question": question,
        "sql": sql,
        "attempts": attempts,
        "error": f"Could not produce a working query after {max_attempts} attempts. Last error: {error}",
    }


def llm_suggest_questions(schema: str, n: int = 5) -> list[str]:
    """Asks the explainer model for analyst-style questions. Returns [] on any failure."""
    prompt = (
        f"Schema:\n{schema}\n\n"
        f"Write {n} different, specific questions a business analyst would ask about this data. "
        "Each must be answerable from the tables above. One question per line, no numbering, no extra text."
    )
    try:
        raw = call_ollama(EXPLAIN_MODEL, prompt)
    except Exception:
        return []
    out = []
    for line in raw.splitlines():
        q = re.sub(r"^[\s\-*\d.)]+", "", line).strip()
        if 15 <= len(q) <= 160 and q.endswith("?"):
            out.append(q)
    return out[:n]
