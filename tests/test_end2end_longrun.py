import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from snn_py.cli.scribe_loop import main as scribe_main
from snn_py.text.corpus import load_corpus
from snn_py.text.ngram import NGramModel


class TestEnd2EndLongRun(unittest.TestCase):
    def test_end2end_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "c.txt").write_text("the agent explores. the world is big.", encoding="utf-8")
            sentences = load_corpus(root)
            model = NGramModel()
            model.fit(sentences)
            model_path = root / "model.json"
            model.save(model_path)

            proposals_file = root / "proposals.jsonl"
            output_file = root / "narrations.jsonl"
            state_file = root / "state.json"

            def writer() -> None:
                with proposals_file.open("a", encoding="utf-8") as handle:
                    for q in [0.4, 0.7, 0.9, 0.2, 0.8]:
                        payload = {"event": "proposal", "ts": time.time(), "meta": {"q": q}}
                        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        handle.flush()
                        time.sleep(0.01)

            writer_thread = threading.Thread(target=writer, daemon=True)
            writer_thread.start()

            rc = scribe_main(
                [
                    "--in",
                    str(proposals_file),
                    "--out",
                    str(output_file),
                    "--state",
                    str(state_file),
                    "--model",
                    str(model_path),
                    "--poll-ms",
                    "5",
                    "--max-events",
                    "5",
                ]
            )
            self.assertEqual(rc, 0)
            lines = output_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 5)


if __name__ == "__main__":
    unittest.main()

