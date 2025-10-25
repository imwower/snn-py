from __future__ import annotations

import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

from snn_py.cli.replay_plus import main


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl_gz(path: Path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reset_logging() -> None:
    import logging

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


class ReplayPlusTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()

    def tearDown(self) -> None:
        _reset_logging()

    def test_replay_expect_and_timetravel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            segs = [{"event": "segment", "meta": {"q": 0.3 + 0.2 * (i % 5)}} for i in range(100)]
            proposals = [
                {"event": "proposal", "meta": {"from": "segment", "i": i}}
                for i, seg in enumerate(segs)
                if seg["meta"]["q"] > 0.6
            ]
            audits = [{"event": "audit_decision", "meta": {"status": "APPROVED"}} for _ in proposals]
            rows_a = segs[:60] + proposals[:36] + audits[:36]
            rows_b = segs[60:] + proposals[36:] + audits[36:]

            f1 = tmp_path / "part1.jsonl"
            f2 = tmp_path / "part2.jsonl.gz"
            _write_jsonl(f1, rows_a)
            _write_jsonl_gz(f2, rows_b)

            expect = tmp_path / "expect.json"
            expect.write_text(
                json.dumps({"counts": {"segment": 100, "proposal": 60, "audit_decision": 60}}),
                encoding="utf-8",
            )

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = main(["--jsonl-dir", str(tmp_path), "--expect", str(expect)])
                out = buf.getvalue()
                buf.truncate(0)
                buf.seek(0)
                rc2 = main(["--jsonl-dir", str(tmp_path), "--time-travel", "--theta", "0.6"])
                out2 = buf.getvalue()
            self.assertEqual(rc, 0, out)
            self.assertIn('"replay_summary"', out)
            self.assertIn('"replay_diff"', out)

            self.assertEqual(rc2, 0, out2)
            self.assertIn('"replay_recomputed"', out2)
            self.assertIn('"replay_match_rate"', out2)
