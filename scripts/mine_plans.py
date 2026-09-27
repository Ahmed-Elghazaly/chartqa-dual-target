"""
    python scripts/mine_plans.py prepare --run runs/mining-1
    NVIDIA_API_KEY=<key> python scripts/mine_plans.py ask --run runs/mining-1 \\
        --model deepseek-ai/deepseek-v4.1-flash
    python scripts/mine_plans.py score --run runs/mining-1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from chartqa_dt.config import build_config
from chartqa_dt.data.pool import load_real_pool
from chartqa_dt.data.records import ChartRecord
from chartqa_dt.data.sources import chartqa_archive_path
from chartqa_dt.paths import data_root, load_dotenv
from chartqa_dt.plans import mining
from chartqa_dt.plans.cache import attach_plans, read_plans, write_plans
from chartqa_dt.plans.facts import mining_refusal_reason
from chartqa_dt.train.targets import TargetError, build_target

BATCH_SIZE = 10

WRAPPER = """The {n} records below are independent requests under the instructions above. Answer \
each record on its own: never use a fact, question, or answer from one record for another.

Reply with ONE ```json block containing an object with exactly two keys:
  "outcomes": {{"<record id>": <exactly one outcome object as defined above>, ...}}
      Every record id listed below must appear exactly once.
  "feedback": {{"<record id>": "<one or two sentences>", ...}}
      Optional; may be empty. Add an entry only when something about a record is worth
      reporting to the people who design the operation list: an operation you wished existed,
      an ambiguity in the question or data, missing context, or why you refused. Feedback is
      collected for later review and never changes how an outcome is scored.
"""
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_OUTCOME_KEYS = ("op", "refused", "needs_operator", "needs_context")


def real_pool() -> list[ChartRecord]:
    cfg = build_config("configs/base.yaml").data
    root = data_root()
    records = load_real_pool(chartqa_archive_path(root), chartqa_limit=cfg.chartqa_limit_per_kind,
                             seed=cfg.mixture_seed, refchartqa_cache=root / "refchartqa_train.jsonl",
                             aligned_cache=root / "refchartqa_aligned.jsonl")
    return attach_plans(records, root / "real_plans.jsonl")


def prepare(run: Path) -> None:
    records = [r for r in real_pool() if mining_refusal_reason(r) is None and r.plan is None]
    run.mkdir(parents=True, exist_ok=True)
    system = mining.build_system()
    for number, start in enumerate(range(0, len(records), BATCH_SIZE)):
        chunk = records[start:start + BATCH_SIZE]
        (run / f"batch_{number:06d}.json").write_text(json.dumps({
            "system": system,
            "requests": [mining.request_for(r) for r in chunk],
            "records": [r.to_dict() for r in chunk],
        }, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(records):,} records without a plan -> {run}")


def _chat(api_base: str, key: str, body: dict[str, Any], attempts: int = 8) -> str:
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=600) as resp:
                message = json.loads(resp.read().decode())["choices"][0]["message"]
            content = message.get("content") or ""
            content = _THINK.sub("", content)
            return content.partition("</think>")[2].strip() if "</think>" in content else content.strip()
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
                raise
            time.sleep(min(300.0, 10.0 * 2 ** attempt))
    raise RuntimeError("teacher request failed repeatedly")


def _batched_outcomes(content: str, ids: list[str]) -> dict[str, dict[str, Any]]:
    start, end = content.find("{"), content.rfind("}")
    candidates = list(reversed(_JSON_BLOCK.findall(content))) + [content]
    if start != -1 and end > start:
        candidates.append(content[start:end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate.strip())
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("outcomes"), dict):
            return {rid: v for rid, v in obj["outcomes"].items()
                    if rid in ids and isinstance(v, dict) and sum(k in v for k in _OUTCOME_KEYS) == 1}
    return {}


def ask_batch(path: Path, *, api_base: str, key: str, params: dict[str, Any]) -> None:
    replies_path = path.with_name(path.name.replace("batch_", "replies_"))
    if replies_path.exists():
        return
    batch = json.loads(path.read_text(encoding="utf-8"))
    ids = [r["custom_id"] for r in batch["requests"]]
    user = WRAPPER.format(n=len(ids)) + "".join(
        f"\n### RECORD {r['custom_id']}\n{r['user']}\n" for r in batch["requests"])
    body = {**params, "messages": [{"role": "system", "content": batch["system"]},
                                   {"role": "user", "content": user}]}
    outcomes = _batched_outcomes(_chat(api_base, key, body), ids)
    replies = {rid: "```json\n" + json.dumps(o, ensure_ascii=False) + "\n```" for rid, o in outcomes.items()}
    for request in batch["requests"]:
        if request["custom_id"] not in replies:
            single = {**params, "messages": [{"role": "system", "content": batch["system"]},
                                             {"role": "user", "content": request["user"]}]}
            replies[request["custom_id"]] = _chat(api_base, key, single)
    replies_path.write_text(json.dumps(replies, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def ask(run: Path, *, model: str, api_base: str, workers: int, extra: dict[str, Any]) -> None:
    load_dotenv()
    key = os.environ["NVIDIA_API_KEY"]
    params = {"model": model, "max_tokens": 32768, **extra}
    batches = sorted(run.glob("batch_*.json"))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(lambda p: ask_batch(p, api_base=api_base, key=key, params=params),
                                       batches), start=1):
            if n % 25 == 0:
                print(f"  {n}/{len(batches)} batches", flush=True)


def score(run: Path, *, teacher: str) -> None:
    outcomes: Counter[str] = Counter()
    accepted: dict[str, tuple[ChartRecord, dict[str, Any], str]] = {}
    for path in sorted(run.glob("batch_*.json")):
        replies_path = path.with_name(path.name.replace("batch_", "replies_"))
        if not replies_path.exists():
            continue
        batch = json.loads(path.read_text(encoding="utf-8"))
        replies = json.loads(replies_path.read_text(encoding="utf-8"))
        questions = {r["custom_id"]: mining.question_of(r["user"]) for r in batch["requests"]}
        for raw_record in batch["records"]:
            record = ChartRecord.from_dict(raw_record)
            status, plan = mining.score_reply(record, replies.get(record.record_id, ""))
            outcomes[status] += 1
            if plan is not None:
                accepted[record.record_id] = (record, plan, questions[record.record_id])

    pool = {r.record_id: r for r in real_pool()}
    kept, set_aside = [], Counter()
    for record_id, (_record, plan, question) in accepted.items():
        try:
            build_target(replace(pool[record_id], plan=plan))
            reason = mining.quarantine_reason(plan, question)
        except TargetError:
            reason = "untrainable"
        if reason:
            set_aside[reason] += 1
        else:
            kept.append({"record_id": record_id, "plan": plan, "teacher": teacher})

    path = data_root() / "real_plans.jsonl"
    existing = list(read_plans(path).values()) if path.exists() else []
    write_plans(path, existing + kept)
    print(f"outcomes: {dict(outcomes.most_common())}")
    print(f"accepted {len(accepted):,}; set aside {dict(set_aside)}; kept {len(kept):,}")
    print(f"{len(existing) + len(kept):,} plans in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "ask", "score"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-ai/deepseek-v4.1-flash")
    ap.add_argument("--api-base", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra-body", default='{"temperature": 0.5, "top_p": 1}',
                    help="JSON merged into every chat request")
    args = ap.parse_args()
    if args.step == "prepare":
        prepare(args.run)
    elif args.step == "ask":
        ask(args.run, model=args.model, api_base=args.api_base, workers=args.workers,
            extra=json.loads(args.extra_body))
    else:
        score(args.run, teacher=args.model)


if __name__ == "__main__":
    main()
