#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EMIT_MS="${EMIT_MS:-1500}"
POLL_MS="${POLL_MS:-500}"
Q_MIN="${Q_MIN:-0.2}"
Q_MAX="${Q_MAX:-0.8}"
CONTEXT_TOKENS=("${AUTO_CONTEXT:-叙述者}")

echo "[auto-cn] starting continuous train + narrate loop..."
"${PYTHON_BIN}" -m snn_py.cli.auto_narrate \
  --corpus corpus \
  --state runs/corpus_cn.fp.json \
  --model models/ngram_cn.json \
  --out runs/narrations_cn.auto.jsonl \
  --emit-ms "${EMIT_MS}" \
  --poll-ms "${POLL_MS}" \
  --q-min "${Q_MIN}" \
  --q-max "${Q_MAX}" \
  --context "${CONTEXT_TOKENS[@]}" \
  "$@"
