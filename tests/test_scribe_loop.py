import json
import tempfile
import unittest
from pathlib import Path

from snn_py.cli.scribe_loop import main as scribe_main
from snn_py.text.corpus import load_corpus
from snn_py.text.ngram import NGramConfig, NGramModel


class TestScribeLoop(unittest.TestCase):
    def test_scribe_processes_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus.txt").write_text("the agent explores. the world is big.", encoding="utf-8")
            sentences = load_corpus(root)
            model = NGramModel(NGramConfig(order=3, k=0.5))
            model.fit(sentences)
            model_path = root / "model.json"
            model.save(model_path)

            proposals = [
                {"event": "proposal", "ts": 0, "meta": {"q": 0.3}},
                {"event": "proposal", "ts": 1, "meta": {"q": 0.6}},
                {"event": "proposal", "ts": 2, "meta": {"q": 0.9}},
            ]
            input_file = root / "in.jsonl"
            with input_file.open("w", encoding="utf-8") as handle:
                for record in proposals:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            output_file = root / "out.jsonl"
            state_file = root / "state.json"
            rc = scribe_main(
                [
                    "--in",
                    str(input_file),
                    "--out",
                    str(output_file),
                    "--state",
                    str(state_file),
                    "--model",
                    str(model_path),
                    "--max-events",
                    "3",
                    "--poll-ms",
                    "10",
                ]
            )
            self.assertEqual(rc, 0)
            payload = output_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(payload), 3)
            self.assertTrue(all('"narration"' in line for line in payload))


if __name__ == "__main__":
    unittest.main()

