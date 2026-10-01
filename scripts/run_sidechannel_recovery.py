"""Targeted recovery using the reference fork's (L-R)/2 ffmpeg wrapper.

The ordinary pipeline stays unchanged. Only explicitly selected lectures
are reset and processed. Reset transcript AND summary: flag-only resets
would reuse the unusable official transcript/summary from an earlier run.
No decrypted backup or course content is uploaded as a workflow artifact.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_ids(value: str) -> list[str]:
    ids = list(dict.fromkeys(part.strip() for part in value.split(",")))
    if not ids or any(not re.fullmatch(r"[0-9]+", item) for item in ids):
        raise ValueError("Course and lecture IDs must be nonempty comma-separated numbers")
    return ids


def select_targets(conn, course_ids: list[str], sub_ids: list[str]):
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in sub_ids)
    rows = conn.execute(f"""
        SELECT l.*, c.title AS course_title
        FROM lectures l JOIN courses c ON c.course_id = l.course_id
        WHERE l.sub_id IN ({placeholders})
        ORDER BY l.sub_title, l.sub_id
    """, sub_ids).fetchall()
    if {str(row["sub_id"]) for row in rows} != set(sub_ids):
        raise ValueError("Some selected lectures are absent from the database")
    if any(str(row["course_id"]) not in course_ids for row in rows):
        raise ValueError("Selected lecture does not belong to the selected course")
    return [dict(row) for row in rows]


def reset_targets(conn, course_ids: list[str], sub_ids: list[str]):
    rows = select_targets(conn, course_ids, sub_ids)
    with conn:
        conn.executemany("""
            UPDATE lectures SET transcript = NULL, summary = NULL,
                summary_model = NULL, processed_at = NULL, emailed_at = NULL,
                error_count = 0, error_stage = NULL, error_msg = NULL
            WHERE sub_id = ? AND course_id = ?
        """, [(row["sub_id"], row["course_id"]) for row in rows])
    return rows


def usable_transcript(text: str | None) -> bool:
    # Recovery is for full-length lectures. Reject a few filler words or
    # repetitions before spending an LLM call or publishing another empty note.
    chars = [char for char in (text or "") if char.isalnum()]
    return len(chars) >= 200 and len(set(chars)) >= 20


def recovered(row) -> bool:
    return bool(usable_transcript(row["transcript"])
                and (row["summary"] or "").strip()
                and row["processed_at"] and not row["error_stage"])


def verify_targets(conn, course_ids: list[str], sub_ids: list[str]):
    rows = select_targets(conn, course_ids, sub_ids)
    failed = []
    for row in rows:
        valid = recovered(row)
        print(f"[Recovery] sub_id={row['sub_id']} "
              f"transcript_chars={len(row['transcript'] or '')} "
              f"summary_chars={len(row['summary'] or '')} "
              f"result={'OK' if valid else 'FAILED'}", flush=True)
        if not valid:
            failed.append(str(row["sub_id"]))
    if failed:
        raise RuntimeError("Recovery incomplete; refusing publication: " + ",".join(failed))
    return rows


def finalize_publication(conn, course_ids: list[str], sub_ids: list[str]):
    # The normal additive merge preserves remote emailed_at when local is NULL.
    # Clear that stale marker only after checking every recovered target again.
    rows = verify_targets(conn, course_ids, sub_ids)
    with conn:
        conn.executemany("UPDATE lectures SET emailed_at=NULL WHERE sub_id=? AND course_id=?",
                         [(row["sub_id"], row["course_id"]) for row in rows])


def run(db_path: str):
    course_ids = parse_ids(os.environ.get("COURSE_IDS", ""))
    sub_ids = parse_ids(os.environ.get("RECOVERY_SUB_IDS", ""))
    if not Path(db_path).is_file():
        raise FileNotFoundError("Recovery requires an existing decrypted database")

    # Heavy dependencies/models are imported only for the actual recovery;
    # selection/reset/quality checks can be tested with the standard library.
    from main import login_with_retry, _drive_lectures
    from src.api.icourse import ICourseClient
    from src.ai.transcriber import Transcriber
    from src.ai.summarizer import Summarizer
    from src.data.database import Database
    from src.runtime import config
    from src.runtime.reporter import Reporter
    from src.runtime.scheduler import Scheduler

    if config.USE_OFFICIAL_TRANSCRIPT:
        raise ValueError("Side-channel recovery requires USE_OFFICIAL_TRANSCRIPT=0")

    class RecoveryTranscriber(Transcriber):
        def transcribe_tail(self, *args, **kwargs):
            text, segments = super().transcribe_tail(*args, **kwargs)
            if not usable_transcript(text):
                raise RuntimeError("Side-channel ASR did not yield enough meaningful lecture text")
            return text, segments

    db = Database(db_path)
    try:
        targets = select_targets(db.conn, course_ids, sub_ids)
        # Keep a local backup in the ephemeral runner. The workflow also
        # preserves the encrypted data-branch commit before publication.
        with sqlite3.connect(db_path + ".before-recovery") as backup:
            db.conn.backup(backup)
        reporter = Reporter()
        transcriber = RecoveryTranscriber()
        summarizer = Summarizer()
        client = ICourseClient(login_with_retry())
        reset_targets(db.conn, course_ids, sub_ids)
        lectures = [(str(row["course_id"]), row["course_title"], row) for row in targets]
        scheduler = Scheduler(reporter=reporter)
        try:
            for attempt in range(1, 4):
                print(f"[Recovery] attempt {attempt}/3: {len(lectures)} lecture(s)", flush=True)
                _drive_lectures(client, db, scheduler, transcriber, summarizer,
                                reporter, lectures, [])
                pending = [row for row in select_targets(db.conn, course_ids, sub_ids)
                           if not recovered(row)]
                if not pending:
                    break
                lectures = [(str(row["course_id"]), row["course_title"], row) for row in pending]
        finally:
            scheduler.shutdown()
        verify_targets(db.conn, course_ids, sub_ids)
    finally:
        db.conn.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--finalize":
        if not Path(sys.argv[2]).is_file():
            raise SystemExit("Publication requires an existing database")
        with sqlite3.connect(sys.argv[2]) as conn:
            finalize_publication(conn, parse_ids(os.environ.get("RECOVERY_COURSE_IDS", "")),
                                 parse_ids(os.environ.get("RECOVERY_SUB_IDS", "")))
    elif len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        raise SystemExit("Usage: run_sidechannel_recovery.py [--finalize] <decrypted_database>")
