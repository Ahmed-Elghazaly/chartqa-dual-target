"""Assert that LoRA actually reached both the vision tower and the language model.

Non-negotiable rule 3. This is the most important assertion in the project.

The failure it catches
----------------------
Qwen3-VL's own fine-tuning tooling advertises component flags (``tune_mm_vision``,
``tune_mm_mlp``, ``tune_mm_llm``), but two still-open official issues — QwenLM/Qwen3-VL
`#2016 <https://github.com/QwenLM/Qwen3-VL/issues/2016>`_ and
`#2079 <https://github.com/QwenLM/Qwen3-VL/issues/2079>`_ — show the LoRA branch can
silently ignore them, because the model is frozen *before* PEFT is applied.

If that happens, training runs normally. Loss goes down. Checkpoints save. Evaluation
produces plausible numbers. The only thing that is different is that the vision half of
the model never learned anything — which would quietly delete the entire computer-vision
contribution of this project, and nobody would find out.

Nothing warns you. So we assert, and we fail the run.

Why the counts are printed and not just checked
-----------------------------------------------
A non-zero count is necessary but not sufficient. A run that adapts three tensors in the
vision tower and four hundred in the language model passes a naive `> 0` check while
being, in substance, a language-only fine-tune. The report needs the actual numbers, so
they are printed, returned, and written into the run log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Substrings that identify a parameter as belonging to the visual side or the
# language side. Deliberately broad: different backbones and different wrapper
# libraries name these differently, and a missed pattern would produce a false
# failure rather than a false pass.
VISION_PATTERNS: tuple[str, ...] = (
    "visual", "vision_tower", "vision_model", "vision_encoder", "image_encoder", "vit",
)
LANGUAGE_PATTERNS: tuple[str, ...] = (
    "language_model", "model.layers", "llm", "text_model", "text_decoder",
)


class LoRACoverageError(RuntimeError):
    """Raised when LoRA did not reach one of the two sides. Never caught and logged away."""


@dataclass
class LoRACoverage:
    vision_params: int = 0
    language_params: int = 0
    other_params: int = 0
    vision_tensors: int = 0
    language_tensors: int = 0
    other_tensors: int = 0
    total_trainable: int = 0
    total_params: int = 0
    rows: list[tuple[str, int]] = field(default_factory=list)
    unclassified_names: list[str] = field(default_factory=list)

    @property
    def trainable_fraction(self) -> float:
        return self.total_trainable / self.total_params if self.total_params else 0.0

    @property
    def vision_share(self) -> float:
        return self.vision_params / self.total_trainable if self.total_trainable else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "vision_params": self.vision_params,
            "language_params": self.language_params,
            "other_params": self.other_params,
            "vision_tensors": self.vision_tensors,
            "language_tensors": self.language_tensors,
            "other_tensors": self.other_tensors,
            "total_trainable": self.total_trainable,
            "total_params": self.total_params,
            "trainable_fraction": round(self.trainable_fraction, 8),
            "vision_share_of_trainable": round(self.vision_share, 6),
            "unclassified_names": self.unclassified_names[:20],
        }

    def describe(self, max_rows: int = 40) -> str:
        lines = [
            f"trainable tensors: {len(self.rows)}   "
            f"vision params: {self.vision_params:,}   language params: {self.language_params:,}   "
            f"other: {self.other_params:,}",
            f"trainable {self.total_trainable:,} / {self.total_params:,} "
            f"({100 * self.trainable_fraction:.4f}%)   "
            f"vision share of trainable: {100 * self.vision_share:.1f}%",
        ]
        for name, n in self.rows[:max_rows]:
            lines.append(f"  {n:>12,}  {name}")
        if len(self.rows) > max_rows:
            lines.append(f"  ... and {len(self.rows) - max_rows} more trainable tensors")
        if self.unclassified_names:
            lines.append(
                f"  NOTE: {len(self.unclassified_names)} trainable tensor(s) matched neither side, e.g. "
                + ", ".join(self.unclassified_names[:3])
            )
        return "\n".join(lines)


def classify_parameter(
    name: str,
    vision_patterns: tuple[str, ...] = VISION_PATTERNS,
    language_patterns: tuple[str, ...] = LANGUAGE_PATTERNS,
) -> str:
    """Return ``"vision"``, ``"language"`` or ``"other"`` for a parameter name.

    Vision is tested first on purpose. In several architectures the vision tower's
    own blocks are also called ``...layers...``, so a language-first test would
    misfile them and could make a vision-only failure look like success.
    """
    low = name.lower()
    if any(k in low for k in vision_patterns):
        return "vision"
    if any(k in low for k in language_patterns):
        return "language"
    return "other"


def summarise_lora(
    model: Any,
    *,
    vision_patterns: tuple[str, ...] = VISION_PATTERNS,
    language_patterns: tuple[str, ...] = LANGUAGE_PATTERNS,
) -> LoRACoverage:
    """Walk ``named_parameters()`` and tally trainable parameters by side."""
    cov = LoRACoverage()
    for name, p in model.named_parameters():
        n = int(getattr(p, "numel", lambda: 0)())
        cov.total_params += n
        if not getattr(p, "requires_grad", False):
            continue
        cov.total_trainable += n
        cov.rows.append((name, n))
        side = classify_parameter(name, vision_patterns, language_patterns)
        if side == "vision":
            cov.vision_params += n
            cov.vision_tensors += 1
        elif side == "language":
            cov.language_params += n
            cov.language_tensors += 1
        else:
            cov.other_params += n
            cov.other_tensors += 1
            cov.unclassified_names.append(name)
    cov.rows.sort(key=lambda r: -r[1])
    return cov


def assert_lora_on_both_sides(
    model: Any,
    *,
    vision_patterns: tuple[str, ...] = VISION_PATTERNS,
    language_patterns: tuple[str, ...] = LANGUAGE_PATTERNS,
    min_vision_share: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fail the run unless LoRA reached both sides.

    ``min_vision_share`` optionally requires the vision side to hold at least that
    fraction of all trainable parameters, catching the subtler failure where LoRA
    technically attaches to the vision tower but only to a token few tensors.

    Returns the coverage dictionary so it can be written into the run log.
    """
    cov = summarise_lora(model, vision_patterns=vision_patterns, language_patterns=language_patterns)
    if verbose:
        print(cov.describe())

    if cov.total_trainable == 0:
        raise LoRACoverageError(
            "No trainable parameters at all. LoRA was not applied, or every module was frozen "
            "after PEFT wrapped the model."
        )
    if cov.vision_params == 0:
        raise LoRACoverageError(
            "No trainable VISION parameters. LoRA did not reach the vision tower.\n"
            "This is the documented Qwen3-VL trainer bug (QwenLM/Qwen3-VL issues #2016, #2079): "
            "component-tuning flags can be silently ignored because the model is frozen before "
            "PEFT is applied. Do not trust the flags.\n"
            f"Trainable tensor names seen: {[n for n, _ in cov.rows[:15]]}\n"
            "If the vision tower in this backbone is named something not in "
            f"{vision_patterns}, pass the right patterns — but verify against the printed names "
            "first, and never widen the patterns just to make this pass."
        )
    if cov.language_params == 0:
        raise LoRACoverageError(
            "No trainable LANGUAGE parameters. LoRA did not reach the language model.\n"
            f"Trainable tensor names seen: {[n for n, _ in cov.rows[:15]]}"
        )
    if cov.vision_share < min_vision_share:
        raise LoRACoverageError(
            f"Vision side holds only {100 * cov.vision_share:.2f}% of trainable parameters, "
            f"below the required {100 * min_vision_share:.2f}%. LoRA is attached to the vision "
            "tower but too thinly for this to be a genuine joint fine-tune."
        )
    return cov.as_dict()


def print_parameter_names(model: Any, pattern: str | None = None, limit: int = 60) -> list[str]:
    """List parameter names, optionally filtered by regex.

    `PLAN.md` 2.3 says to verify the pattern lists against the model's actual
    parameter names — print them first, then adjust, and do not assume. This is
    the tool for doing that.
    """
    rx = re.compile(pattern) if pattern else None
    names = [n for n, _ in model.named_parameters() if rx is None or rx.search(n)]
    for n in names[:limit]:
        print(f"  {n}")
    if len(names) > limit:
        print(f"  ... and {len(names) - limit} more")
    return names
