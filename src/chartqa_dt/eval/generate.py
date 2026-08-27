"""Zero-shot generation over a frozen slice — `PLAN.md` 5.2, 5.3, 5.4.

Produces predictions; scoring is `chartqa_dt.eval.runner`'s job and the number of record
is the official evaluator's. Keeping generation separate means a prediction file can be
re-scored later without re-running the model, which matters when a metric turns out to
need correcting (`DECISIONS.md` 0053 corrected three).

**Decoding is greedy and fixed.** `PLAN.md` 5.5 seals decoding parameters into the
pre-registration, and sampling would make the baseline a distribution rather than a
number — the "before" figure has to be exactly reproducible from the committed config, or
the before/after comparison inherits sampling noise it cannot separate from a real effect.

**Invalid output is a failure, not a retry.** Non-negotiable rule 3. A second attempt with
a nudged prompt would measure the nudge. Failures are counted, reported, and scored as
wrong.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.prompting.parsing import (
    ParseStats,
    answer_of,
    coerce_boxes,
    parse_record,
)
from chartqa_dt.prompting.prompts import build_plain_prompt, build_structured_prompt

#: Greedy. Sealed by the pre-registration; see the module docstring.
DECODING = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
            "num_beams": 1}
MAX_NEW_TOKENS_STRUCTURED = 512
MAX_NEW_TOKENS_PLAIN = 32


@dataclass
class Generation:
    """One model output and everything measured about producing it."""

    record_id: str
    raw: str
    seconds: float
    prompt_mode: str
    parsed_ok: bool = False
    repairs: list[str] = field(default_factory=list)
    reason: str = ""
    answer: str = ""
    boxes: list[list[float]] = field(default_factory=list)
    plan: dict | None = None
    image_size: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "raw": self.raw, "seconds": self.seconds,
                "prompt_mode": self.prompt_mode, "parsed_ok": self.parsed_ok,
                "repairs": self.repairs, "reason": self.reason, "answer": self.answer,
                "boxes": self.boxes, "plan": self.plan,
                "image_size": list(self.image_size) if self.image_size else None}


@dataclass
class GenerationReport:
    n: int = 0
    seconds_total: float = 0.0
    latencies: list[float] = field(default_factory=list)
    parse: ParseStats = field(default_factory=ParseStats)

    @property
    def median_latency(self) -> float:
        import statistics

        return statistics.median(self.latencies) if self.latencies else 0.0

    def describe(self) -> str:
        return (f"  generated       : {self.n} in {self.seconds_total:.0f}s "
                f"(median {self.median_latency:.2f}s/item)\n"
                f"{self.parse.describe()}")


def build_messages(question: str, image: Any, mode: str) -> list[dict[str, Any]]:
    """The chat structure the processor expects, with the sealed prompt text."""
    text = build_structured_prompt(question) if mode == "structured" \
        else build_plain_prompt(question)
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": text}]}]


def generate_one(loaded: Any, question: str, image: Any, *, mode: str = "structured",
                 max_new_tokens: int | None = None) -> tuple[str, float]:
    """Greedy generation for one item. Returns the decoded continuation and its latency."""
    import torch

    processor, model = loaded.processor, loaded.model
    messages = build_messages(question, image, mode)
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    budget = max_new_tokens or (MAX_NEW_TOKENS_STRUCTURED if mode == "structured"
                                else MAX_NEW_TOKENS_PLAIN)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             num_beams=1,
                             pad_token_id=processor.tokenizer.pad_token_id
                             or processor.tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return decoded, elapsed


def generate_over(loaded: Any, items: Iterable[dict[str, Any]], *,
                  mode: str = "structured", limit: int | None = None,
                  progress_every: int = 25,
                  max_new_tokens: int | None = None
                  ) -> tuple[list[Generation], GenerationReport]:
    """Run a slice. Each item supplies `record_id`, `question` and a PIL `image`."""
    out: list[Generation] = []
    report = GenerationReport()
    for i, item in enumerate(items):
        if limit and i >= limit:
            break
        image = item["image"]
        raw, seconds = generate_one(loaded, item["question"], image, mode=mode,
                                    max_new_tokens=max_new_tokens)
        gen = Generation(record_id=item["record_id"], raw=raw, seconds=seconds,
                         prompt_mode=mode,
                         image_size=tuple(image.size) if hasattr(image, "size") else None)

        if mode == "plain":
            # No record to parse: the published-prompt baseline emits a bare answer.
            gen.parsed_ok, gen.answer = True, raw.strip()
        else:
            result = parse_record(raw)
            report.parse.add(result)
            gen.parsed_ok, gen.repairs, gen.reason = result.ok, result.repairs, result.reason
            if result.ok and result.record is not None:
                gen.answer = answer_of(result.record)
                gen.boxes = coerce_boxes(result.record)
                gen.plan = result.record.get("plan")

        report.n += 1
        report.seconds_total += seconds
        report.latencies.append(seconds)
        out.append(gen)
        if progress_every and report.n % progress_every == 0:
            print(f"    {report.n} items, {report.seconds_total:.0f}s, "
                  f"valid {report.parse.valid}/{report.parse.total}", flush=True)
    return out, report


def write_generations(generations: Sequence[Generation], path: str | Path) -> Path:
    """Raw generations, kept so a prediction set can be re-scored without a GPU."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(g.to_dict()) for g in generations) + "\n",
                 encoding="utf-8")
    return p


def read_generations(path: str | Path) -> list[Generation]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        size = d.get("image_size")
        out.append(Generation(record_id=d["record_id"], raw=d["raw"],
                              seconds=d["seconds"], prompt_mode=d["prompt_mode"],
                              parsed_ok=d["parsed_ok"], repairs=d.get("repairs", []),
                              reason=d.get("reason", ""), answer=d.get("answer", ""),
                              boxes=d.get("boxes", []), plan=d.get("plan"),
                              image_size=tuple(size) if size else None))
    return out


__all__ = ["DECODING", "MAX_NEW_TOKENS_PLAIN", "MAX_NEW_TOKENS_STRUCTURED", "Generation",
           "GenerationReport", "build_messages", "generate_one", "generate_over",
           "read_generations", "write_generations"]
