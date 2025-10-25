import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from snn_py.manifest import Manifest, save_manifest


class ManifestTests(unittest.TestCase):
    def test_build_and_save_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.json"
            policy_path.write_text(json.dumps({"policy": "demo"}), encoding="utf-8")
            argv = ["--policy", str(policy_path), "--foo", "bar"]
            env = {"SNN_PY_LOGLEVEL": "DEBUG", "SHOULD_IGNORE": "1", "SNN_PY_SEED": "2024"}
            original_argv = sys.argv
            try:
                sys.argv = ["demo"] + argv
                manifest = Manifest.build(policy_path, env)
            finally:
                sys.argv = original_argv

            self.assertTrue(manifest.run_id)
            self.assertGreater(manifest.ts_start, 0.0)
            expected_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            self.assertEqual(manifest.policy_sha256, expected_hash)
            self.assertEqual(manifest.policy_path, str(policy_path))
            self.assertEqual(manifest.argv, argv)
            self.assertEqual(
                manifest.env_whitelist,
                {"SNN_PY_LOGLEVEL": "DEBUG", "SNN_PY_SEED": "2024"},
            )
            self.assertEqual(manifest.artifacts, {})

            out_dir = Path(tmpdir) / "run"
            manifest_path = save_manifest(manifest, out_dir)
            self.assertTrue(manifest_path.is_file())
            content = manifest_path.read_text(encoding="utf-8")
            data = json.loads(content)
            self.assertEqual(data["run_id"], manifest.run_id)
            self.assertEqual(data["policy_sha256"], expected_hash)
            self.assertEqual(data["argv"], argv)
            self.assertEqual(data["env_whitelist"], manifest.env_whitelist)
            self.assertIn("seeds", data)
            self.assertEqual(data["seeds"], {})
