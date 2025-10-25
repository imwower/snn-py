import json
import tempfile
import unittest
from pathlib import Path

from snn_py.cli.mock_proposals import main as mock_proposals_main


class TestMockProposals(unittest.TestCase):
    def test_emits_requested_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "proposals.jsonl"
            rc = mock_proposals_main(
                ["--out", str(out), "--count", "5", "--interval-ms", "0", "--seed", "42"]
            )
            self.assertEqual(rc, 0)
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 5)
            events = [json.loads(line) for line in lines]
            self.assertTrue(all(event.get("event") == "proposal" for event in events))
            qs = [event.get("meta", {}).get("q") for event in events]
            self.assertTrue(all(0.1 <= float(q) <= 0.95 for q in qs if q is not None))


if __name__ == "__main__":
    unittest.main()

