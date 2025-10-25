import unittest

from snn_py import logging_config
from snn_py.intent.scoring import NoveltyScorer
from snn_py.plugins.registry import load_symbol
from tests.utils import record_payload


class PluginRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_load_builtin_symbol(self) -> None:
        with self.assertLogs("snn_py.plugins.registry", level="INFO") as captured:
            symbol_obj = load_symbol("snn_py.intent.scoring:NoveltyScorer")
        events = [record_payload(record) for record in captured.records]
        self.assertTrue(any(entry.get("event") == "plugin_loaded" for entry in events))
        self.assertIs(symbol_obj, NoveltyScorer)

    def test_missing_symbol_logs_error(self) -> None:
        with self.assertLogs("snn_py.plugins.registry", level="ERROR") as captured:
            with self.assertRaises(AttributeError):
                load_symbol("snn_py.intent.scoring:MissingClass")
        events = [record_payload(record) for record in captured.records]
        error = next((entry for entry in events if entry.get("event") == "plugin_error"), None)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertEqual(error["meta"]["symbol"], "snn_py.intent.scoring:MissingClass")


if __name__ == "__main__":
    unittest.main()
