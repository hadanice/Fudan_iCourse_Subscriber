"""One-time normal-audio recovery of course 39624, lecture 749935."""
import os
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sidechannel_recovery import run

if __name__ == '__main__':
    if os.environ.get('COURSE_IDS') != '39624' or os.environ.get('RECOVERY_SUB_IDS') != '749935':
        raise SystemExit('This one-time operation only supports 39624 / 749935')
    db_path = sys.argv[1]
    if not Path(db_path).is_file():
        raise SystemExit('Existing database required')
    from main import login_with_retry
    from src.api.icourse import ICourseClient
    from src.data.database import Database
    client = ICourseClient(login_with_retry())
    detail = client.get_course_detail('39624')
    lecture = next(l for l in detail['lectures'] if str(l['sub_id']) == '749935')
    assert lecture.get('has_playback'), 'Target has no playback'
    assert '2026-09-17' in (lecture.get('sub_title', '') + lecture.get('date', '')), 'Unexpected lecture date'
    db = Database(db_path)
    try:
        with sqlite3.connect(db_path + '.before-insertion') as backup:
            db.conn.backup(backup)
        db.upsert_course('39624', detail['title'], detail['teacher'])
        db.insert_lecture('749935', '39624', lecture['sub_title'], lecture.get('date', ''))
    finally:
        db.conn.close()
    # Explicit target selection bypasses title dedup. Normal ffmpeg remains active.
    # Reuse quality verification, retries and no-email recovery orchestration.
    run(db_path)
