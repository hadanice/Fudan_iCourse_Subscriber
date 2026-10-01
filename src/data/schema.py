"""Single source of truth for the SQLite schema.

This module is imported by every Python component that creates or migrates
the database (Database, sharder, merge_db) so the column list lives in
exactly one place.

frontend/js/schema.js is a **manual mirror** of these constants.  When you
change SCHEMA_SQL, LECTURES_MIGRATION_COLUMNS, or PPT_PAGES_MIGRATION_COLUMNS
here, update that file too — there is no automated sync.  Both run in
different processes (Python on the CI runner, JS in the browser) and have
to agree on what tables and columns exist.
"""

from __future__ import annotations


def recover_no_content(conn, schema: str = "main") -> int:
    """Requeue legacy 'processed without summary' rows without deleting data.

    Idempotent: errors and transcripts survive, successful summaries are
    untouched. Used both on startup and before merging old data snapshots
    so COALESCE cannot resurrect a terminal No Content marker.
    The caller owns the transaction.
    """
    if schema not in ("main", "local"):
        raise ValueError("Unsupported database schema")
    return conn.execute(f"""
        UPDATE {schema}.lectures
        SET processed_at = NULL, emailed_at = NULL
        WHERE processed_at IS NOT NULL
          AND NULLIF(TRIM(summary, char(9) || char(10) || char(13) || ' '), '') IS NULL
    """).rowcount


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,
    title TEXT,
    teacher TEXT
);
CREATE TABLE IF NOT EXISTS lectures (
    sub_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    sub_title TEXT, date TEXT,
    transcript TEXT, summary TEXT,
    processed_at TEXT, emailed_at TEXT,
    error_msg TEXT, error_count INTEGER DEFAULT 0,
    error_stage TEXT, summary_model TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
CREATE TABLE IF NOT EXISTS ppt_pages (
    sub_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    created_sec INTEGER NOT NULL,
    pptimgurl TEXT,
    text TEXT,
    ocr_status TEXT NOT NULL DEFAULT 'pending',
    ocr_at TEXT,
    dhash TEXT,
    PRIMARY KEY (sub_id, page_num),
    FOREIGN KEY (sub_id) REFERENCES lectures(sub_id)
);
CREATE INDEX IF NOT EXISTS idx_ppt_pages_sub_status
    ON ppt_pages(sub_id, ocr_status);
-- ``all_courses`` is the catalog of every course offered by the school in
-- a given term, regardless of whether the user has subscribed to it.  Used
-- by the frontend's subscription editor to render a searchable picker;
-- separate from ``courses`` (which only holds subscribed courses with
-- locally-cached lectures).
CREATE TABLE IF NOT EXISTS all_courses (
    course_id TEXT NOT NULL,
    term TEXT NOT NULL,
    title TEXT,
    teacher TEXT,
    dept TEXT,
    last_seen_at TEXT,
    PRIMARY KEY (course_id, term)
);
CREATE INDEX IF NOT EXISTS idx_all_courses_term
    ON all_courses(term);
-- ``meta`` holds key-value configuration that the frontend needs without
-- loading the full course-data shards (e.g. currently-subscribed course IDs).
-- Populated by the CI runner from secrets / runtime state.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# Columns added to ``lectures`` after the v1 schema shipped.  Existing DBs
# get them via ALTER TABLE in Database._init_tables / merge_db._ensure_schema.
LECTURES_MIGRATION_COLUMNS: list[tuple[str, str]] = [
    ("error_msg", "TEXT"),
    ("error_count", "INTEGER DEFAULT 0"),
    ("error_stage", "TEXT"),
    ("summary_model", "TEXT"),
]

# Columns added to ``ppt_pages`` after its initial shape shipped.
PPT_PAGES_MIGRATION_COLUMNS: list[tuple[str, str]] = [
    ("dhash", "TEXT"),
]
