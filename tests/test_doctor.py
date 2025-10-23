import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from snn_py.cli import doctor

POLICY_GOOD = {
    "tools": {
        "allowlist": ["ProposeAction"],
        "guarded": [],
        "rules": {},
    }
}


class DoctorTests(unittest.TestCase):
    def _write_policy(self, tmpdir: str, doc: dict) -> str:
        path = os.path.join(tmpdir, "policy.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        return path

    def test_doctor_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self._write_policy(tmpdir, POLICY_GOOD)
            exit_code = doctor.main([
                "--policy",
                policy_path,
                "--check-dir",
                tmpdir,
                "--loglevel",
                "INFO",
            ])
            self.assertEqual(exit_code, 0)

    def test_doctor_fail_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = self._write_policy(tmpdir, {})
            exit_code = doctor.main([
                "--policy",
                bad_path,
                "--check-dir",
                tmpdir,
                "--loglevel",
                "INFO",
            ])
            self.assertNotEqual(exit_code, 0)

    def test_doctor_fail_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self._write_policy(tmpdir, POLICY_GOOD)
            ro_dir = Path(tmpdir) / "ro"
            ro_dir.mkdir()
            ro_dir.chmod(stat.S_IREAD | stat.S_IEXEC)
            try:
                exit_code = doctor.main([
                    "--policy",
                    policy_path,
                    "--check-dir",
                    str(ro_dir),
                    "--loglevel",
                    "INFO",
                ])
                self.assertNotEqual(exit_code, 0)
            finally:
                ro_dir.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

