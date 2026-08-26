#!/usr/bin/env bash
# Run the full pipeline in --dev mode (a ~200-example subset, no full download).
# This is the "does the whole thing still work end to end" check, not the real run.
set -euo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-python3.11}"

echo "== 1. environment =="
$PY -c "from chartqa_dt.env import get_env; print(get_env().describe())"

echo "== 2. lint and fast tests =="
ruff check src tests scripts
pytest -q -m "not slow and not gpu and not network"

echo "== 3. data (dev subset) =="
cdt-data download --dev --datasets chartqa,refchartqa

echo "== 4. synthetic generation + the mandatory box-correctness self-test =="
cdt-gen --dev -n 50 --verify

echo "== 5. plan mining (yield reported separately by chart source) =="
cdt-mine --dev --sample 200

echo "== 6. mixtures (asserts zero val/test records) =="
cdt-data mixture --dev

echo "== 7. evaluation on the dev subset =="
cdt-eval --dev --dataset chartqa --split val
cdt-eval --dev --dataset refchartqa --split val

echo "== 8. report tables =="
cdt-report --what all

echo "done."
