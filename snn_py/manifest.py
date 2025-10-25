"""Run manifest collection and persistence."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Dict, Optional, List
import json
import time
import uuid
import hashlib
import os
import subprocess

ENV_WHITELIST = ("SNN_PY_LOGLEVEL", "SNN_PY_SEED")


def _git_sha_short() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256_of_file(p: Optional[Path]) -> Optional[str]:
    if not p or not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    run_id: str
    ts_start: float
    git_sha: Optional[str]
    policy_path: Optional[str]
    policy_sha256: Optional[str]
    argv: List[str]
    env_whitelist: Dict[str, str]
    artifacts: Dict[str, str]
    seeds: Dict[str, int]

    @classmethod
    def build(cls, policy_path: Optional[Path], env: Mapping[str, str] = os.environ) -> "Manifest":
        """Backward-compatible constructor for legacy callers."""
        return build(policy_path, env)


def build(policy_path: Optional[Path], env: Mapping[str, str] = os.environ) -> Manifest:
    run_id = uuid.uuid4().hex
    ts = time.time()
    git = _git_sha_short()
    pol_sha = _sha256_of_file(policy_path)
    env_wl = {k: env[k] for k in ENV_WHITELIST if k in env}
    return Manifest(
        run_id=run_id,
        ts_start=ts,
        git_sha=git,
        policy_path=str(policy_path) if policy_path else None,
        policy_sha256=pol_sha,
        argv=list(os.sys.argv),
        env_whitelist=env_wl,
        artifacts={},
        seeds={},
    )


def save_manifest(m: Manifest, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "manifest.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(m), f, ensure_ascii=False, sort_keys=True)
    return p


__all__ = [
    "Manifest",
    "ENV_WHITELIST",
    "build",
    "save_manifest",
    "_sha256_of_file",
]
