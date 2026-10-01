"""Offline tests, including actual ffmpeg decoding of synthetic stereo audio."""
import array
import contextlib
import io
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sidechannel_recovery import (
    parse_ids, select_targets, reset_targets, usable_transcript, verify_targets,
    finalize_publication,
)
from src.data.schema import SCHEMA_SQL


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript(SCHEMA_SQL)
        self.db.execute("INSERT INTO courses VALUES ('39173', 'Course', 'Teacher')")
        for sub_id in ("good", "660467", "666413", "671966"):
            self.db.execute("""INSERT INTO lectures
                (sub_id, course_id, sub_title, transcript, summary, summary_model,
                 processed_at, emailed_at, error_count)
                VALUES (?, '39173', ?, 'old text', 'old summary', 'model', 'done', 'sent', 2)
            """, (sub_id, sub_id))
        self.db.execute("""INSERT INTO ppt_pages(sub_id,page_num,created_sec,text)
            VALUES ('660467',1,0,'existing slide')""")
        self.db.commit()

    def test_reset_clears_cached_content_but_preserves_other_lecture_and_ppt(self):
        untouched = self.db.execute("SELECT * FROM lectures WHERE sub_id='good'").fetchone()
        slides = self.db.execute("SELECT * FROM ppt_pages").fetchall()
        reset_targets(self.db, ["39173"], ["660467", "666413", "671966"])
        self.assertEqual(tuple(self.db.execute("SELECT * FROM lectures WHERE sub_id='good'").fetchone()), untouched)
        self.assertEqual([tuple(r) for r in self.db.execute("SELECT * FROM ppt_pages")], slides)
        for row in select_targets(self.db, ["39173"], ["660467", "666413", "671966"]):
            for field in ("transcript", "summary", "summary_model", "processed_at", "emailed_at", "error_stage", "error_msg"):
                self.assertIsNone(row[field], field)
            self.assertEqual(row["error_count"], 0)

    def test_missing_or_wrong_course_aborts_before_any_reset(self):
        for courses, subs in [(["999"], ["660467"]), (["39173"], ["660467", "999"])]:
            with self.assertRaises(ValueError):
                reset_targets(self.db, courses, subs)
        self.assertEqual(self.db.execute("SELECT summary FROM lectures WHERE sub_id='660467'").fetchone()[0], "old summary")

    def test_id_validation(self):
        self.assertEqual(parse_ids("660467, 666413,660467"), ["660467", "666413"])
        for raw in ("", "1,", "a", "1; DROP TABLE lectures", "1,2\n3", "１２３"):
            with self.assertRaises(ValueError):
                parse_ids(raw)

    def test_filler_or_empty_transcript_does_not_qualify(self):
        for text in (None, "", "嗯 " * 1000, "很短的文字", "  " * 1000):
            self.assertFalse(usable_transcript(text))
        self.assertTrue(usable_transcript("abcdefghijklmnopqrstuvwxyz0123456789" * 10))

    def test_incomplete_recovery_refuses_publication(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "refusing publication"):
                verify_targets(self.db, ["39173"], ["660467", "666413"])

    def test_verified_output_requires_summary_and_no_error(self):
        self.db.execute("UPDATE lectures SET transcript=?, error_stage=NULL WHERE sub_id='660467'",
                        ("abcdefghijklmnopqrstuvwxyz0123456789" * 10,))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(verify_targets(self.db, ["39173"], ["660467"])), 1)
            self.db.execute("UPDATE lectures SET error_stage='transcribe' WHERE sub_id='660467'")
            with self.assertRaises(RuntimeError):
                verify_targets(self.db, ["39173"], ["660467"])

    def test_publication_only_clears_email_marker_for_verified_targets(self):
        self.db.execute("UPDATE lectures SET transcript=? WHERE sub_id='660467'",
                        ("abcdefghijklmnopqrstuvwxyz0123456789" * 10,))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                finalize_publication(self.db, ["39173"], ["660467", "666413"])
            self.assertEqual(self.db.execute("SELECT emailed_at FROM lectures WHERE sub_id='660467'").fetchone()[0], "sent")
            finalize_publication(self.db, ["39173"], ["660467"])
        self.assertIsNone(self.db.execute("SELECT emailed_at FROM lectures WHERE sub_id='660467'").fetchone()[0])
        self.assertEqual(self.db.execute("SELECT emailed_at FROM lectures WHERE sub_id='good'").fetchone()[0], "sent")


class AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = shutil.which("ffmpeg")
        if not cls.ffmpeg:
            try:
                import imageio_ffmpeg
                cls.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                raise unittest.SkipTest("Install ffmpeg to run real audio tests")

    def render(self, inverted, side):
        # Deliberately anti-phase L/R: mono downmix cancels it exactly.
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "stereo.wav")
            with wave.open(path, "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"".join(
                    struct.pack("<hh", value, -value if inverted else value)
                    for value in (int(10000 * math.sin(2 * math.pi * 440 * i / 16000))
                                  for i in range(32000))
                ))
            args = [self.ffmpeg, "-v", "error", "-i", path, "-ar", "16000"]
            args += ["-af", "pan=mono|c0=0.5*FL-0.5*FR"] if side else ["-ac", "1"]
            result = subprocess.run(args + ["-f", "f32le", "pipe:1"],
                                    check=True, capture_output=True)
            pcm = array.array("f")
            pcm.frombytes(result.stdout)
            return math.sqrt(sum(v * v for v in pcm) / len(pcm))

    def test_phase_inverted_audio_cancelled_by_mono_is_recovered_by_side(self):
        mono = self.render(inverted=True, side=False)
        side = self.render(inverted=True, side=True)
        self.assertLess(mono, 0.00001)
        self.assertGreater(side, 0.2)
        print(f"[Audio test] inverted stereo RMS: mono={mono:.6f}, side={side:.6f}")

    def test_side_is_not_a_global_replacement_for_normal_audio(self):
        self.assertGreater(self.render(inverted=False, side=False), 0.2)
        self.assertLess(self.render(inverted=False, side=True), 0.00001)

    @unittest.skipIf(os.name == "nt", "The wrapper runs on the Linux Actions runner")
    def test_wrapper_only_rewrites_f32le_and_passes_other_commands_through(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            wbin, realbin = directory / "wrapper", directory / "real"
            wbin.mkdir()
            realbin.mkdir()
            wrapper = wbin / "ffmpeg"
            wrapper.write_bytes((Path(__file__).parent / "ffmpeg_side_channel.sh").read_bytes())
            wrapper.chmod(0o755)
            real = realbin / "ffmpeg"
            real.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            real.chmod(0o755)
            env = dict(os.environ, PATH=f"{wbin}:{realbin}:{os.environ['PATH']}")
            args = ["-i", "file with spaces.mp4", "-ac", "1", "-f", "f32le", "out.raw"]
            output = subprocess.run([str(wrapper), *args], env=env, check=True,
                                    capture_output=True, text=True).stdout.splitlines()
            self.assertNotIn("-ac", output)
            self.assertIn("pan=mono|c0=0.5*FL-0.5*FR", output)
            self.assertIn("file with spaces.mp4", output)
            normal = ["-i", "input.mp4", "-ac", "1", "output.wav"]
            output = subprocess.run([str(wrapper), *normal], env=env, check=True,
                                    capture_output=True, text=True).stdout.splitlines()
            self.assertEqual(output, normal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
