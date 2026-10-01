"""No models, credentials, network calls or real course data are needed.

Run: python -m unittest discover -s tests -p test_no_content.py -v
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from main import _enumerate_lectures
from scripts.merge_db import merge
from src.api.icourse import ICourseClient
from src.ai.transcriber import Transcriber, NoAudioStreamError, IncompleteAudioError
from src.data.database import Database
from src.pipeline.lecture_runner import LectureRunner
from src.runtime.scheduler import AudioDownloader, AudioHandle


class VideoCandidatesTests(unittest.TestCase):
    def client(self, info):
        client = ICourseClient.__new__(ICourseClient)
        client.get_sub_info = Mock(return_value=info)
        client.get_sub_detail = Mock(return_value={})
        client.sign_video_url = Mock(side_effect=lambda url, now=None: url + "&signed")
        return client

    def test_retains_distinct_sources_and_query_strings(self):
        client = self.client({
            "now": "123",
            "video_list": {
                "screen": {"preview_url": "https://cdn/screen.mp4"},
                "teacher": {"preview_url": "https://cdn/teacher.mp4?quality=hd"},
            },
            "playurl": {"now": 123, "duplicate": "https://cdn/screen.mp4?t=old",
                        "alternate": "https://cdn/other.mp4"},
            "content": {"playback": {"url": "https://cdn/nested.mp4"}},
        })
        self.assertEqual(client.get_video_urls("1", "2"), [
            "https://cdn/screen.mp4&signed", "https://cdn/teacher.mp4?quality=hd&signed",
            "https://cdn/other.mp4&signed", "https://cdn/nested.mp4&signed",
        ])
        self.assertEqual(client.sign_video_url.call_count, 4)
        self.assertEqual(client.sign_video_url.call_args.kwargs["now"], 123)

    def test_array_video_list_and_empty_sources(self):
        client = self.client({"video_list": [None, {"preview_url": "https://cdn/a.MP4?q=1"}]})
        self.assertEqual(client.get_video_url("1", "2"), "https://cdn/a.MP4?q=1&signed")
        client = self.client({"video_list": None, "playurl": None})
        self.assertEqual(client.get_video_urls("1", "2"), [])

    def test_review_gate_nested_and_detail_fallback(self):
        client = self.client({"content": {"now": "456", "playback": {"url": "https://cdn/a.mp4"}}})
        self.assertEqual(len(client.get_video_urls("1", "2")), 1)
        client.sign_video_url.assert_called_once_with("https://cdn/a.mp4", now=456)
        client.get_sub_info.side_effect = RuntimeError("unavailable")
        client.get_sub_detail.return_value = {"content": {"playback": {"url": "https://cdn/b.mp4"}}}
        self.assertEqual(client.get_video_urls("1", "2"), ["https://cdn/b.mp4&signed"])


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def database(self, name="db"):
        db = Database(os.path.join(self.tmp.name, name + ".db"))
        self.addCleanup(db.conn.close)
        return db

    def test_startup_recovers_legacy_rows_without_losing_notes_or_ppt(self):
        db = self.database()
        for sid in ("good", "empty", "whitespace", "video_only"):
            db.insert_lecture(sid, "1", sid, "2026-09-16")
            db.mark_processed(sid)
        db.update_summary("good", "Existing notes", "model")
        db.mark_emailed("good")
        db.update_transcript("empty", "")
        db.update_summary("whitespace", " \n\t\r ", "model")
        db.update_error("video_only", "transcribe", "No audio")
        with db.conn:
            db.conn.execute("INSERT INTO ppt_pages(sub_id,page_num,created_sec,text) VALUES ('empty',1,0,'slide')")
        before = db.get_lecture("good")
        db._init_tables()
        self.assertEqual(db.get_lecture("good"), before)
        self.assertIsNone(db.get_lecture("empty")["processed_at"])
        self.assertIsNone(db.get_lecture("whitespace")["processed_at"])
        self.assertEqual(db.get_lecture("video_only")["error_count"], 1)
        self.assertEqual(db.conn.execute("SELECT text FROM ppt_pages").fetchone()[0], "slide")
        self.assertEqual(db.get_processed_sub_ids("1"), {"good"})
        db._init_tables()
        self.assertEqual(len(db.get_unprocessed_lectures("1")), 3)

    def test_merge_does_not_resurrect_no_content_or_erase_retry_error(self):
        local, remote = self.database("local"), self.database("remote")
        for db in (local, remote):
            db.insert_lecture("silent", "1", "Silent", "")
            db.insert_lecture("ready", "1", "Ready", "")
            db.update_summary("ready", "Saved notes", "model")
            db.mark_processed("ready")
            db.mark_emailed("ready")
        remote.mark_processed("silent")
        remote.mark_emailed("silent")
        local.update_error("silent", "empty_transcript", "all sources silent")
        merge(local.db_path, remote.db_path)
        row = remote.get_lecture("silent")
        self.assertIsNone(row["processed_at"])
        self.assertIsNone(row["emailed_at"])
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["error_stage"], "empty_transcript")
        self.assertTrue(remote.get_lecture("ready")["emailed_at"])
        # Also works when the stale/terminal snapshot is the local one.
        local.mark_processed("silent")
        local.update_summary("ready", " \n\t ", "wrong model")
        merge(local.db_path, remote.db_path)
        self.assertIsNone(remote.get_lecture("silent")["processed_at"])
        self.assertEqual(remote.get_lecture("silent")["error_count"], 1)
        self.assertEqual(remote.get_lecture("ready")["summary"], "Saved notes")
        self.assertEqual(remote.get_lecture("ready")["summary_model"], "model")

    def test_exhausted_remote_lectures_are_not_readded_as_new(self):
        db = self.database()
        db.insert_lecture("2", "1", "Silent", "")
        for _ in range(3):
            db.update_error("2", "empty_transcript", "silent")
        client = Mock()
        client.get_course_detail.return_value = {
            "title": "Course", "teacher": "Teacher",
            "lectures": [{"sub_id": "2", "sub_title": "Silent", "has_playback": True}],
        }
        with patch("src.runtime.config.COURSE_IDS", ["1"]):
            self.assertEqual(_enumerate_lectures(client, db, Mock()), [])
        self.assertEqual(db.get_exhausted_sub_ids("1"), {"2"})
        self.assertEqual(db.get_unprocessed_lectures("1"), [])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Database(os.path.join(self.tmp.name, "db.sqlite"))
        self.addCleanup(self.db.conn.close)
        self.db.insert_lecture("2", "1", "Lecture", "")
        self.client, self.scheduler, self.transcriber = Mock(), Mock(), Mock()
        self.client.get_transcript_segments.return_value = []
        self.reporter = Mock()
        self.runner = LectureRunner(self.client, self.db, self.scheduler,
                                    self.transcriber, Mock(), self.reporter)
        self.runner._ppt = Mock()
        self.patch = patch("src.runtime.config.USE_OFFICIAL_TRANSCRIPT", False)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def audio(self, results):
        urls = tuple(f"https://cdn/{i}.mp4?t=SECRET" for i in range(len(results)))
        self.scheduler.audio_downloader.get.side_effect = [
            AudioHandle("2", f"{i}.raw", Mock(), [], urls) for i in range(len(results))
        ]
        self.transcriber.transcribe_tail.side_effect = results
        return urls

    def test_silent_first_source_uses_second_and_saves_summary(self):
        urls = self.audio([("", []), ("actual speech", [{"text": "actual speech"}])])
        self.runner._summarize = Mock(side_effect=lambda *a: self.db.update_summary("2", "notes", "model") or "notes")
        self.assertEqual(self.runner.run("1", "Course", {"sub_id": "2"}), "notes")
        row = self.db.get_lecture("2")
        self.assertEqual(row["transcript"], "actual speech")
        self.assertTrue(row["processed_at"])
        self.assertEqual(row["error_count"], 0)
        self.scheduler.audio_downloader.schedule.assert_any_call(self.client, "1", "2", video_url=urls[1])
        self.assertNotIn("SECRET", str(self.reporter.mock_calls))

    def test_video_only_first_source_uses_second(self):
        self.audio([NoAudioStreamError("video only"), ("speech", [])])
        text, _ = self.runner._get_transcript(None, "1", "2")
        self.assertEqual(text, "speech")
        self.assertEqual(self.db.get_lecture("2")["error_count"], 0)

    def test_all_sources_empty_records_one_failure_and_drains_ppt(self):
        self.audio([("", []), NoAudioStreamError("video only")])
        self.assertIsNone(self.runner.run("1", "Course", {"sub_id": "2"}))
        row = self.db.get_lecture("2")
        self.assertIsNone(row["processed_at"])
        self.assertIsNone(row["transcript"])
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["error_stage"], "empty_transcript")
        self.runner._ppt.submit.return_value.drain.assert_called_once()
        self.assertEqual(self.scheduler.audio_downloader.release.call_count, 2)

    def test_complete_official_transcript_rescues_silent_asr(self):
        self.audio([("", [])])
        self.client.get_transcript_segments.return_value = [{"start_ms": 0, "end_ms": 60000, "text": "official speech"}]
        text, _ = self.runner._get_transcript(None, "1", "2")
        self.assertEqual(text, "official speech")
        self.assertEqual(self.db.get_lecture("2")["transcript"], text)

    def test_truncated_official_transcript_is_not_accepted(self):
        self.audio([("", [])])
        self.client.get_transcript_segments.return_value = [{"start_ms": 0, "end_ms": 60000, "text": "partial"}]
        with self.db.conn:
            self.db.conn.execute("INSERT INTO ppt_pages(sub_id,page_num,created_sec) VALUES ('2',1,3600)")
        self.assertEqual(self.runner._get_transcript(None, "1", "2"), (None, None))
        self.assertIsNone(self.db.get_lecture("2")["transcript"])

    def test_incomplete_audio_does_not_cache_partial_transcript(self):
        self.audio([IncompleteAudioError("truncated", 30, 3600, "partial")])
        self.assertEqual(self.runner._get_transcript(None, "1", "2"), (None, None))
        self.assertIsNone(self.db.get_lecture("2")["transcript"])
        self.assertEqual(self.db.get_lecture("2")["error_stage"], "transcribe")

    def test_existing_summary_is_preserved_without_audio(self):
        self.db.update_summary("2", "existing", "model")
        self.db.mark_processed("2")
        self.db.mark_emailed("2")
        self.assertIsNone(self.runner.run("1", "Course", {"sub_id": "2"}))
        self.transcriber.transcribe_tail.assert_not_called()

    def test_empty_summary_is_an_error_not_success(self):
        self.audio([("speech", [])])
        self.runner._summarizer.summarize.return_value = ("  ", "model")
        with patch("src.ai.bucketer.assemble", return_value=("prompt", "flat")):
            with self.assertRaisesRegex(RuntimeError, "empty summary"):
                self.runner.run("1", "Course", {"sub_id": "2"})
        self.assertIsNone(self.db.get_lecture("2")["processed_at"])
        self.assertEqual(self.db.get_lecture("2")["error_stage"], "summarize")

    def test_whitespace_cache_does_not_skip_audio(self):
        self.db.update_transcript("2", "  \n")
        self.db.update_summary("2", "  ", "model")
        self.audio([("speech", [])])
        self.assertFalse(self.runner._has_summary(self.db.get_lecture("2")))
        self.assertTrue(self.runner._needs_audio("2"))
        self.assertEqual(self.runner._get_transcript(self.db.get_lecture("2"), "1", "2")[0], "speech")


class DownloaderTests(unittest.TestCase):
    def test_alternate_download_uses_its_url_and_cleans_each_process_file(self):
        # Real child-process lifecycle and file cleanup, but no ffmpeg,
        # external URL or model is needed for this integration boundary.
        popen = subprocess.Popen
        commands = []

        def fake_decoder(cmd, **kwargs):
            commands.append(cmd)
            return popen([sys.executable, "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'PCM')",
                cmd[-1]], **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            downloader = AudioDownloader(directory, max_concurrent=1)
            client = Mock()
            client.get_video_urls.return_value = ["screen.mp4", "teacher.mp4"]
            client.get_stream_params.side_effect = lambda url: (url, "Cookie: test\r\n")
            try:
                with patch("src.runtime.scheduler.subprocess.Popen", side_effect=fake_decoder):
                    downloader.schedule(client, "1", "2")
                    first = downloader.get("2", timeout=5)
                    first.process.wait(timeout=5)
                    self.assertEqual(first.video_urls, ("screen.mp4", "teacher.mp4"))
                    self.assertTrue(os.path.exists(first.path))
                    downloader.release("2")
                    downloader.schedule(client, "1", "2", video_url=first.video_urls[1])
                    second = downloader.get("2", timeout=5)
                    second.process.wait(timeout=5)
                    downloader.release("2")
                    self.assertNotEqual(first.path, second.path)
                    self.assertFalse(os.path.exists(first.path))
                    self.assertFalse(os.path.exists(second.path))
                    self.assertEqual([cmd[cmd.index("-i") + 1] for cmd in commands],
                                     ["screen.mp4", "teacher.mp4"])
            finally:
                downloader.shutdown()

    def test_spawn_failure_is_not_misreported_as_no_video(self):
        with tempfile.TemporaryDirectory() as directory:
            downloader = AudioDownloader(directory, max_concurrent=1)
            client = Mock()
            client.get_video_urls.side_effect = RuntimeError("API unavailable")
            downloader.schedule(client, "1", "2")
            with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                downloader.get("2", timeout=2)
            downloader.release("2")
            self.assertTrue(downloader._sem.acquire(timeout=1))
            downloader._sem.release()

    def test_early_ffmpeg_exit_without_output_is_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            transcriber = Transcriber.__new__(Transcriber)
            proc = Mock()
            proc.poll.return_value = 234
            proc.returncode = 234
            with self.assertRaises(NoAudioStreamError):
                transcriber.transcribe_tail(os.path.join(directory, "missing.raw"), proc,
                    [b"Output file does not contain any stream\n"])


if __name__ == "__main__":
    unittest.main()
