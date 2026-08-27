#!/usr/bin/env bash
# Reproduce CI locally, in CI's own environment, BEFORE pushing.
#
# This exists because "it passes locally" was true and useless: the dev venv has torch and
# datasets, CI's fast job does not, and `src/chartqa_dt/data/` was excluded by .gitignore
# for nine commits while every local run stayed green (DECISIONS.md 0050).
#
# Usage:  bash scripts/preflight.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CIENV="${CIENV:-/tmp/chartqa-cienv}"
PY311="${PY311:-python3.11}"

if [ ! -x "$CIENV/bin/pytest" ]; then
  echo "== building a CI-equivalent venv (core + dev only, no torch/datasets) =="
  "$PY311" -m venv "$CIENV"
  "$CIENV/bin/pip" -q install --upgrade pip
  "$CIENV/bin/pip" -q install -e ".[dev]"
fi

echo "== 1/4 nothing a source file needs is gitignored =="
"$CIENV/bin/pytest" -q tests/test_repo_completeness.py

echo "== 2/4 lint (CI: ruff check src tests scripts) =="
"$CIENV/bin/ruff" check src tests scripts

echo "== 3/4 fast CPU tests (CI's exact selection) =="
"$CIENV/bin/pytest" -q -m "not slow and not gpu and not network and not official"

echo "== 4/4 documentation consistency =="
"$CIENV/bin/pytest" -q tests/test_docs_consistency.py

echo
echo "preflight passed — safe to push."
