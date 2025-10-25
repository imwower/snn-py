import tempfile
import unittest
from pathlib import Path

from snn_py.cli.text_eval import main as text_eval_main
from snn_py.text.corpus import load_corpus
from snn_py.text.ngram import NGramModel


class TestTextEval(unittest.TestCase):
    def test_eval_pp_oov(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()
            (train_dir / "a.txt").write_text("the agent explores. the world is big.", encoding="utf-8")
            (val_dir / "b.txt").write_text("the agent acts. the agent learns.", encoding="utf-8")

            sentences = load_corpus(train_dir)
            model = NGramModel()
            model.fit(sentences)
            model_path = root / "model.json"
            model.save(model_path)

            rc = text_eval_main(["--model", str(model_path), "--corpus", str(val_dir)])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

