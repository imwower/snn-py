import tempfile
import unittest
from pathlib import Path

from snn_py.cli.learn_watch import main as learn_watch_main
from snn_py.text.ngram import NGramModel


class TestCorpusWatch(unittest.TestCase):
    def test_watch_and_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_dir = root / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "a.txt").write_text("the agent explores.\n", encoding="utf-8")
            state_file = root / "fp.json"
            model_path = root / "model.json"

            rc = learn_watch_main(
                ["--corpus", str(corpus_dir), "--state", str(state_file), "--out", str(model_path)]
            )
            self.assertEqual(rc, 0)
            vocab_first = len(NGramModel.load(model_path).vocab)

            (corpus_dir / "b.txt").write_text("the world is big.\n", encoding="utf-8")
            rc = learn_watch_main(
                ["--corpus", str(corpus_dir), "--state", str(state_file), "--out", str(model_path)]
            )
            self.assertEqual(rc, 0)
            vocab_second = len(NGramModel.load(model_path).vocab)
            self.assertGreaterEqual(vocab_second, vocab_first)


if __name__ == "__main__":
    unittest.main()

