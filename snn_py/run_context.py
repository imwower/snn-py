"""运行上下文构建工具。"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from typing import Dict, Mapping, MutableMapping, Optional


def _current_git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    sha = result.stdout.strip()
    return sha or None


def build_run_context(env: Mapping[str, str] | MutableMapping[str, str] = os.environ) -> Dict[str, object]:
    """根据环境变量生成运行上下文。"""

    seed_str = env.get("SNN_PY_SEED")
    seed: Optional[int]
    if seed_str is None:
        seed = None
    else:
        try:
            seed = int(seed_str)
        except ValueError:
            seed = None

    context: Dict[str, object] = {
        "run_id": str(uuid.uuid4()),
        "ts_start": time.time(),
        "git_sha": _current_git_sha(),
        "seed": seed,
    }
    return context

