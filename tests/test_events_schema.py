from __future__ import annotations

import unittest

from snn_py.schema.events_schema import validate_event, validate_stream


class EventsSchemaTests(unittest.TestCase):
    def test_validate_good_segment_event(self) -> None:
        ok, reason = validate_event({"event": "segment", "ts": 1.23, "meta": {"q": 0.7}})
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_validate_bad_segment_missing_q(self) -> None:
        ok, reason = validate_event({"event": "segment", "ts": 1.23, "meta": {}})
        self.assertFalse(ok)
        self.assertEqual(reason, "segment_missing_q")

    def test_validate_stream_mixed(self) -> None:
        events = [
            {"event": "segment", "ts": 1.0, "meta": {"q": 0.1}},
            {"event": "proposal", "ts": 1.1},
            {"event": "audit_decision", "ts": 1.2, "meta": {"status": "APPROVED"}},
            {"event": "segment", "ts": 1.3, "meta": {}},
        ]
        stats = validate_stream(events)
        self.assertEqual(stats.total, 4)
        self.assertEqual(stats.ok, 3)
        self.assertEqual(stats.bad, 1)
        self.assertEqual(len(stats.samples_bad), 1)
