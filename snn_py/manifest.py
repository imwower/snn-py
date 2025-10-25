"""Run manifest collection and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional
from uuid import uuid4


ENV_WHITELIST = ("SNN_PY_LOGLEVEL", "SNN_PY_SEED")


def _read_git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _hash_policy(policy_path: Optional[Path]) -> Optional[str]:
    if policy_path is None:
        return None
    try:
        data = policy_path.read_bytes()
    except OSError:
        return None
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


@dataclass
class Manifest:
    run_id: str
    ts_start: float
    git_sha: Optional[str]
    policy_sha256: Optional[str]
    policy_path: Optional[str]
    argv: list[str] = field(default_factory=list)
    env_whitelist: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, object] = field(default_factory=dict)
    seeds: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, policy_path: Optional[Path], env: Mapping[str, str]) -> "Manifest":
        run_id = uuid4().hex
        ts_start = time.time()
        git_sha = _read_git_sha()
        policy_sha256 = _hash_policy(policy_path)
        collected_env: Dict[str, str] = {}
        for key in ENV_WHITELIST:
            if key in env:
                collected_env[key] = env[key]
        argv = list(sys.argv[1:])
        policy_path_str = str(policy_path) if policy_path else None
        return cls(
            run_id=run_id,
            ts_start=ts_start,
            git_sha=git_sha,
            policy_sha256=policy_sha256,
            policy_path=policy_path_str,
            argv=argv,
            env_whitelist=collected_env,
        )


def save_manifest(manifest: Manifest, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.json"
    payload = asdict(manifest)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    out_path.write_text(text, encoding="utf-8")
    return out_path


__all__ = ["Manifest", "save_manifest", "ENV_WHITELIST"]
