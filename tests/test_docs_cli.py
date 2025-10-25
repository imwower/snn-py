import os
import tempfile
import unittest
from pathlib import Path

from snn_py import logging_config
from snn_py.cli import docs
from tests.utils import record_payload


class DocsCLITests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_generates_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "docs.md"
            with self.assertLogs("snn_py.cli.docs", level="INFO") as captured:
                exit_code = docs.main(["--out", str(out_path)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_path.is_file())
            content = out_path.read_text(encoding="utf-8")
        self.assertTrue(content.strip())
        self.assertIn("IntentGate", content)
        events = [record_payload(record) for record in captured.records]
        self.assertTrue(any(entry.get("event") == "docs_generated" for entry in events))


if __name__ == "__main__":
    unittest.main()
