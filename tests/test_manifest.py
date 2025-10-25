import json
import os
import tempfile
import unittest
from pathlib import Path

from snn_py.manifest import build, save_manifest, _sha256_of_file


class ManifestTests(unittest.TestCase):
    def test_build_and_save_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pol = tmp_path / "policy.json"
            pol.write_text('{"tools":{"allowlist":[]}}', encoding="utf-8")
            sha = _sha256_of_file(pol)

            manifest = build(pol)
            out_dir = tmp_path / "run_dir"
            p = save_manifest(manifest, out_dir)
            data = json.loads(p.read_text(encoding="utf-8"))

            self.assertTrue(data.get("run_id"))
            self.assertEqual(data["policy_sha256"], sha)
            self.assertEqual(data["artifacts"], {})

    def test_sha256_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.assertIsNone(_sha256_of_file(tmp_path / "nope.json"))
