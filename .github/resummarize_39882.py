"""One-time regeneration of course 39882 from cached study materials."""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai import bucketer
from src.data.database import Database
from src.runtime import config

COURSE_ID = "39882"
REPORT = Path("data/resummary-metrics.json")


def summary_hash(value):
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def generate():
    from src.ai.summarizer import Summarizer

    database = Database()
    changes = []
    failures = []
    try:
        rows = database.conn.execute(
            "SELECT l.*, c.title AS course_title FROM lectures l "
            "JOIN courses c ON c.course_id = l.course_id "
            "WHERE l.course_id = ? AND COALESCE(l.summary, '') != '' "
            "ORDER BY l.date, l.sub_id",
            (COURSE_ID,),
        ).fetchall()
        if not rows:
            raise RuntimeError("No existing summaries were found for course 39882")
        print(f"[Resummarize] Course {COURSE_ID}: {len(rows)} existing summaries", flush=True)
        summarizer = Summarizer()
        for row in rows:
            sub_id = row["sub_id"]
            old_summary = row["summary"]
            transcript = row["transcript"] or ""
            try:
                if not transcript.strip():
                    raise ValueError("Cached transcript is missing")
                pages = database.get_done_ppt_pages(sub_id)
                content, mode = bucketer.assemble(transcript, None, pages)
                print(
                    f"[Resummarize] Start {sub_id}: {len(content)} input chars, "
                    f"{len(old_summary)} old summary chars, target {len(content) // 4}",
                    flush=True,
                )
                summary, model = summarizer.summarize(row["course_title"], content)
                if len(summary) <= len(old_summary):
                    raise ValueError("Replacement did not increase summary length")
                database.update_summary(sub_id, summary, model)
                change = {
                    "sub_id": sub_id,
                    "date": row["date"],
                    "before_hash": summary_hash(old_summary),
                    "after_hash": summary_hash(summary),
                    "before_chars": len(old_summary),
                    "after_chars": len(summary),
                    "input_chars": len(content),
                    "mode": mode,
                }
                changes.append(change)
                REPORT.write_text(json.dumps(changes, ensure_ascii=False), encoding="utf-8")
                print("[Resummarize] Saved " + json.dumps(change, ensure_ascii=False), flush=True)
            except Exception as error:
                failures.append(sub_id)
                print(f"[Resummarize] Failed {sub_id}: {type(error).__name__}: {error}", flush=True)
        print(
            "[Resummarize] Result " + json.dumps({
                "course_id": COURSE_ID,
                "total": len(rows),
                "updated": len(changes),
                "failures": failures,
                "before_chars": sum(change["before_chars"] for change in changes),
                "after_chars": sum(change["after_chars"] for change in changes),
            }),
            flush=True,
        )
        if failures:
            raise RuntimeError(f"Summary regeneration needs attention for {failures}")
    finally:
        database.conn.close()


def merge():
    """Replace successful summaries in the latest remote snapshot only."""
    changes = json.loads(REPORT.read_text(encoding="utf-8"))
    source = sqlite3.connect(config.DB_PATH)
    destination = sqlite3.connect("data/remote.db")
    try:
        with destination:
            for change in changes:
                row = source.execute(
                    "SELECT summary, summary_model FROM lectures "
                    "WHERE sub_id = ? AND course_id = ?",
                    (change["sub_id"], COURSE_ID),
                ).fetchone()
                current = destination.execute(
                    "SELECT summary FROM lectures WHERE sub_id = ? AND course_id = ?",
                    (change["sub_id"], COURSE_ID),
                ).fetchone()
                if not row or not current or summary_hash(row[0]) != change["after_hash"]:
                    raise RuntimeError(f"Summary snapshot mismatch for {change['sub_id']}")
                current_hash = summary_hash(current[0])
                if current_hash not in (change["before_hash"], change["after_hash"]):
                    raise RuntimeError(f"Concurrent summary update for {change['sub_id']}")
                destination.execute(
                    "UPDATE lectures SET summary = ?, summary_model = ? "
                    "WHERE sub_id = ? AND course_id = ?",
                    (row[0], row[1], change["sub_id"], COURSE_ID),
                )
        print(f"[Resummarize] Verified {len(changes)} replacements in the latest database", flush=True)
    finally:
        source.close()
        destination.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["generate"]:
        generate()
    elif sys.argv[1:] == ["merge"]:
        merge()
    else:
        raise SystemExit("Expected generate or merge")
