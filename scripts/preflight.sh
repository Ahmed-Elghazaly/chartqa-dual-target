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

echo "== 1/6 nothing a source file needs is gitignored =="
"$CIENV/bin/pytest" -q tests/test_repo_completeness.py

echo "== 2/6 lint (CI: ruff check src tests scripts) =="
"$CIENV/bin/ruff" check src tests scripts

echo "== 3/6 fast CPU tests (CI's exact selection) =="
"$CIENV/bin/pytest" -q -m "not slow and not gpu and not network and not official"

echo "== 4/6 documentation consistency =="
"$CIENV/bin/pytest" -q tests/test_docs_consistency.py

# The count STATUS.md quotes. Printed rather than remembered: I have written the wrong
# number into it twice, and a figure a human retypes is a figure that drifts.
# `|| true` matters: this script runs under `set -e`, and a grep that matches nothing
# exits 1 — so a cosmetic count would abort a preflight that had already passed.
# Summed from the per-file counts rather than parsed from a summary line: the two pytest
# versions in play here print different summaries, and `|| true` matters because this runs
# under `set -e` and a grep matching nothing exits 1 — a cosmetic count must never abort a
# preflight that has already passed.
N_TESTS=$("$CIENV/bin/pytest" --collect-only -q 2>/dev/null \
    | awk -F': ' '/^tests\/.*: [0-9]+$/ {n += $2} END {print n}' || true)
[ -n "$N_TESTS" ] && [ "$N_TESTS" != "0" ] || N_TESTS="?"

echo
echo "preflight passed — safe to push.   ${N_TESTS} tests collected."
echo
echo "NOTE: run this WITHOUT a pipe. \`preflight.sh | tail -3\` reports tail's exit"
echo "status, not preflight's, so a failure chained with && is silently skipped."
echo "That is how a red commit reached main once already."

# CI runs a `guardrails` job that preflight did not, and the gap was not academic: the
# rule-7 history scan failed on every push for a week while preflight reported green,
# because `presentation/figures/*.png` was never added to its allow-list. A preflight that
# claims to be CI-equivalent has to run the same checks (`DECISIONS.md` 0120).
echo "== 5/6 guardrails (CI: rule 7 and credentials, over ALL git history) =="
bad=$(git log --all --pretty=format: --name-only --diff-filter=A \
      | sort -u \
      | grep -Ei '\.(png|jpg|jpeg|gif|bmp|webp|tiff|zip|parquet|arrow)$' \
      | grep -Ev '^(report/figures/|demo/examples/|presentation/figures/)' || true)
if [ -n "$bad" ]; then
  echo "dataset content found in git history (non-negotiable rule 7):"
  echo "$bad"
  exit 1
fi
bad=$(git log --all --pretty=format: --name-only --diff-filter=A \
      | sort -u | grep -Ei '(^|/)(\.env|kaggle\.json|.*\.pem|.*\.key)$' || true)
if [ -n "$bad" ]; then
  echo "credential file found in git history:"; echo "$bad"; exit 1
fi
echo "clean: no dataset content or credentials in history"

# The check that counting cannot do. `scripts/e2e.py` builds real targets, prints several
# in full, and fails if a per-source usable count has drifted beyond tolerance against
# `data/composition_snapshot.json`. It exists because a change once took ChartQA from 5
# records to 4,944 and every aggregate said it was working, while one printed target read
# "Which year has the most crime?" -> evidence: all six years (`DECISIONS.md` 0116).
#
# Skipped when the datasets are not cached locally, which is the case in CI.
echo "== 6/6 end-to-end smoke over real data =="
if [ -f "$HOME/.cache/chartqa_dt/data/refchartqa_train.jsonl" ]; then
  ./.venv/bin/python scripts/e2e.py --show 1
else
  echo "  datasets not cached locally; skipped"
fi
