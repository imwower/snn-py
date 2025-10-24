import json
import tempfile
import unittest
from pathlib import Path

from snn_py import logging_config
from snn_py.cli import sweep
from tests.utils import record_payload


class SweepCLITests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_sweep_writes_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "results.json"
            args = sweep._parse_args(
                [
                    "--seeds",
                    "1,2",
                    "--theta",
                    "0.6:0.7:0.1",
                    "--decay",
                    "0.5,0.8",
                    "--T",
                    "1.0",
                    "--dt",
                    "0.05",
                    "--output",
                    str(output),
                    "--loglevel",
                    "INFO",
                ]
            )

            with self.assertLogs("snn_py.cli.sweep", level="INFO") as captured:
                result = sweep._run_sweep(args)

            self.assertTrue(output.exists())
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("run_id", data)
            cases = data["cases"]
            expected = {(1, 0.6, 0.5), (1, 0.6, 0.8), (1, 0.7, 0.5), (1, 0.7, 0.8), (2, 0.6, 0.5), (2, 0.6, 0.8), (2, 0.7, 0.5), (2, 0.7, 0.8)}
            found = {(item["seed"], item["theta"], item["decay"]) for item in cases}
            self.assertEqual(expected, found)

            events = [record_payload(record) for record in captured.records]
            self.assertEqual(result["run_id"], data["run_id"])
            self.assertEqual(len(result["cases"]), len(cases))
            self.assertTrue(any(entry.get("event") == "sweep_complete" for entry in events))


if __name__ == "__main__":
    unittest.main()
