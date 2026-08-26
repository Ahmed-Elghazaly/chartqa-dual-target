"""The Phase 2 smoke test: does this backbone actually train on a free GPU?

`PLAN.md` Phase 2 exists because of a verified risk (`IDEA.md` 7, re-confirmed in
Phase 0): Unsloth publishes vision fine-tuning notebooks for Qwen3-VL **8B**,
Qwen2.5-VL **7B** and Qwen3.5 **2B/4B**, but **none for Qwen3-VL-2B**. Nobody has
published that this model trains at this size on this hardware. So we measure it
before building anything on top.

What is recorded, per configuration
-----------------------------------
* peak **reserved** GPU memory (the Phase 2 gate is 13.5 GiB)
* seconds per optimizer step, and the projected full-run wall time (gate: 10 h)
* trainable parameter names and counts, split by vision and language side
* loss trajectory, and whether it decreased and stayed finite
* checkpoint save -> kill -> resume, verified by comparing the loss *after*
  resume against the same steps run without interruption

Decision 0010 adds a second axis: every configuration is measured at both the
plan's 512-pixel budget and at native resolution, so the base input size is
chosen on evidence rather than inherited from an analysis that used the wrong
visual-token factor.

The data here is deliberately trivial
-------------------------------------
Synthetic bar charts drawn with PIL, with the answer written into the target
string. This phase asks "does the machinery run, fit and converge at all", not
"does it learn chart QA". Real data arrives in Phase 3, and using it here would
confound a memory measurement with a data question.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.config import Config
from chartqa_dt.modeling.backends.base import (
    Backend,
    LoadedModel,
    get_backend,
    peak_reserved_gb,
    reset_peak_memory,
)
from chartqa_dt.modeling.lora_assert import (
    assert_lora_on_both_sides,
    describe_quantisation,
    summarise_quantisation,
)

# Phase 2 hard gates (IDEA.md 14, PLAN.md Appendix F).
MEMORY_GATE_GB = 13.5
FULL_RUN_GATE_HOURS = 10.0
# Pre-registered budget: 24,000 example presentations at effective batch 8.
PLANNED_OPTIMIZER_STEPS = 3000


# --------------------------------------------------------------------------- #
# Trivial synthetic data
# --------------------------------------------------------------------------- #


def make_bar_chart(width: int, height: int, rng: random.Random) -> tuple[Any, str, str]:
    """A crude bar chart plus a question and its exact answer.

    Not the Phase 3 generator. No matplotlib, no exact artist geometry, no typed
    plan — just enough pixels and text to drive a forward and backward pass.
    """
    from PIL import Image, ImageDraw

    width, height = max(64, int(width)), max(64, int(height))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    n = rng.randint(3, 5)
    values = [rng.randint(10, 99) for _ in range(n)]
    labels = [str(2018 + i) for i in range(n)]

    # Margins scale with the image. Hard-coded ones produced negative bar heights
    # on small images, which PIL rejects outright — caught by running this on CPU
    # before spending GPU time on it.
    margin_x = max(6, width // 12)
    margin_b = max(10, height // 6)
    margin_t = max(6, height // 8)
    base = height - margin_b
    plot_h = max(4, base - margin_t)

    slot = max(4, (width - 2 * margin_x) // n)
    bar_w = max(3, int(slot * 0.6))
    for i, v in enumerate(values):
        x = margin_x + i * slot
        h = max(2, int((v / 100.0) * plot_h))
        draw.rectangle([x, base - h, x + bar_w, base], fill=(60, 110, 200))
        draw.text((x, min(base + 2, height - 10)), labels[i], fill="black")
    draw.line([margin_x - 2, base, width - margin_x, base], fill="black", width=1)

    i = rng.randrange(n)
    question = f"What is the value for {labels[i]}?"
    return img, question, str(values[i])


def build_batch(processor: Any, images_and_text: list[tuple[Any, str, str]], max_len: int) -> dict:
    """Tokenise a batch and mask everything except the answer out of the loss.

    Only the answer tokens carry loss. Training on the prompt as well still
    "works" and still shows a falling loss, which is exactly why it is worth
    being explicit: a smoke test that optimises the wrong objective still passes
    every gate. The first draft here masked only padding and image tokens, and
    23 of 45 label positions were supervised instead of the 8 that should be.

    The prompt length is measured by running the processor on the prompt alone
    with the same image, rather than by counting text tokens: the image expands
    into a variable number of visual tokens, so a text-only count would put the
    boundary in the wrong place.
    """
    import torch

    tok = processor.tokenizer
    prev_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"  # masking leading positions requires right padding
    try:
        texts, images, prompt_lens = [], [], []
        for img, question, answer in images_and_text:
            user_turn = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
            ]
            prompt_text = processor.apply_chat_template(
                user_turn, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + answer
            texts.append(full_text)
            images.append(img)
            prompt_only = processor(
                text=[prompt_text], images=[img], return_tensors="pt", truncation=True, max_length=max_len
            )
            prompt_lens.append(int(prompt_only["input_ids"].shape[1]))

        batch = processor(
            text=texts, images=images, return_tensors="pt",
            padding=True, truncation=True, max_length=max_len,
        )
    finally:
        tok.padding_side = prev_side

    labels = batch["input_ids"].clone()
    for row, n_prompt in enumerate(prompt_lens):
        labels[row, :n_prompt] = -100
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            labels[labels == tid] = -100

    batch["labels"] = labels
    return {k: (v.to(torch.long) if k in ("input_ids", "labels") else v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class SmokeResult:
    backend: str
    model_id: str
    image_max_pixels: int
    label: str
    ok: bool = False
    error: str = ""

    steps: int = 0
    peak_reserved_gb: float = 0.0
    seconds_per_step: float = 0.0
    projected_full_run_hours: float = 0.0
    load_seconds: float = 0.0

    median_visual_tokens: int = 0
    lora: dict[str, Any] = field(default_factory=dict)
    losses: list[float] = field(default_factory=list)
    loss_first_10: float = 0.0
    loss_last_10: float = 0.0
    loss_decreased: bool = False
    any_nan: bool = False

    quantisation: dict[str, Any] = field(default_factory=dict)
    vision_kept_full_precision: bool | None = None

    resume_verified: bool | None = None
    resume_loss_delta: float | None = None

    gpu_name: str = "cpu"
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def passes_memory_gate(self) -> bool:
        return self.peak_reserved_gb <= MEMORY_GATE_GB

    @property
    def passes_time_gate(self) -> bool:
        return self.projected_full_run_hours <= FULL_RUN_GATE_HOURS

    @property
    def passes_all_gates(self) -> bool:
        return bool(
            self.ok
            and self.passes_memory_gate
            and self.passes_time_gate
            and not self.any_nan
            and self.loss_decreased
            and self.lora.get("vision_params", 0) > 0
            and self.lora.get("language_params", 0) > 0
        )

    def row(self) -> str:
        def tick(b: object) -> str:
            return "PASS" if b else ("—" if b is None else "FAIL")
        return (
            f"{self.label:<28} {self.peak_reserved_gb:>7.2f} {self.seconds_per_step:>8.2f} "
            f"{self.projected_full_run_hours:>7.2f} {self.median_visual_tokens:>7} "
            f"{self.lora.get('vision_params', 0):>10,} {self.lora.get('language_params', 0):>11,} "
            f"{tick(self.loss_decreased):>6} {tick(self.resume_verified):>7} {tick(self.passes_all_gates):>6}"
        )


HEADER = (
    f"{'configuration':<28} {'peakGB':>7} {'s/step':>8} {'proj_h':>7} {'vistok':>7} "
    f"{'vis_params':>10} {'lang_params':>11} {'loss':>6} {'resume':>7} {'GATES':>6}"
)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def _train_steps(
    loaded: LoadedModel,
    *,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_len: int,
    image_px: int,
    seed: int,
    optimizer: Any | None = None,
    on_step: Any = None,
) -> tuple[list[float], Any]:
    """Run ``steps`` optimizer steps and return (losses, optimizer)."""
    import torch

    model = loaded.model
    model.train()
    device = next(model.parameters()).device

    if optimizer is None:
        params = [p for p in model.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=lr)
        except (ImportError, RuntimeError):
            # No CUDA / no bitsandbytes: the memory numbers are then not
            # comparable, and the caller records that in `notes`.
            optimizer = torch.optim.AdamW(params, lr=lr)

    rng = random.Random(seed)
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(grad_accum):
            samples = [make_bar_chart(image_px, int(image_px * 0.7), rng) for _ in range(batch_size)]
            batch = build_batch(loaded.processor, samples, max_len)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(total)
        if on_step is not None:
            on_step(step, total)
    return losses, optimizer


def run_smoke(
    cfg: Config,
    *,
    backend_name: str,
    image_max_pixels: int,
    label: str,
    steps: int = 100,
    out_dir: Path | None = None,
    logger: Any = None,
    test_resume: bool = True,
) -> SmokeResult:
    """One configuration: load, assert LoRA, train ``steps``, measure, test resume."""
    import copy

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backend = backend_name
    model_cfg.image_max_pixels = image_max_pixels

    result = SmokeResult(
        backend=backend_name,
        model_id=model_cfg.hf_id,
        image_max_pixels=image_max_pixels,
        label=label,
    )

    try:
        from chartqa_dt.env import gpu_name

        result.gpu_name = gpu_name()
        reset_peak_memory()

        backend: Backend = get_backend(backend_name)
        loaded = backend.load(model_cfg)
        result.load_seconds = loaded.load_seconds
        result.median_visual_tokens = loaded.geometry.n_visual_tokens(557, 800)

        # Measured on the loaded model, before adapters change the module tree.
        result.quantisation = summarise_quantisation(loaded.model)
        result.vision_kept_full_precision = result.quantisation.get("vision_kept_full_precision")
        print(describe_quantisation(result.quantisation))
        if logger:
            logger.event("quantisation", label=label,
                         **{k: v for k, v in result.quantisation.items() if k != "examples"})

        loaded = backend.apply_lora(loaded, model_cfg)
        loaded = backend.prepare_for_training(loaded, model_cfg)

        # Rule 3. Fails the run rather than warning.
        result.lora = assert_lora_on_both_sides(loaded.model, verbose=True)
        if logger:
            logger.event("lora_coverage", label=label, **result.lora)

        image_px = int(math.sqrt(image_max_pixels))
        t0 = time.time()
        losses, optimizer = _train_steps(
            loaded,
            steps=steps,
            batch_size=cfg.train.per_device_batch,
            grad_accum=cfg.train.grad_accum,
            lr=cfg.train.lr,
            max_len=model_cfg.max_seq_len,
            image_px=image_px,
            seed=cfg.seed,
            on_step=(lambda s, v: logger.log({"smoke_loss": v, "label": label}, step=s)) if logger else None,
        )
        elapsed = time.time() - t0

        result.steps = steps
        result.losses = [round(v, 5) for v in losses]
        result.seconds_per_step = elapsed / max(1, steps)
        result.projected_full_run_hours = result.seconds_per_step * PLANNED_OPTIMIZER_STEPS / 3600.0
        result.peak_reserved_gb = peak_reserved_gb()
        result.any_nan = any(not math.isfinite(v) for v in losses)
        k = max(1, min(10, steps // 5))
        result.loss_first_10 = sum(losses[:k]) / k
        result.loss_last_10 = sum(losses[-k:]) / k
        result.loss_decreased = result.loss_last_10 < result.loss_first_10

        # Peak memory is already recorded above; the resume test resets the
        # counter so that its own second model load cannot inflate the figure
        # the 13.5 GiB gate is judged on.
        if test_resume and out_dir is not None:
            result.resume_verified, result.resume_loss_delta = _verify_resume(
                backend, loaded, model_cfg, cfg, out_dir / f"ckpt_{label}", optimizer, image_px
            )

        if result.peak_reserved_gb == 0.0:
            result.notes["warning"] = (
                "no CUDA device: memory and timing figures are not comparable to the gates"
            )
        result.ok = True

    except Exception as exc:  # noqa: BLE001 - a failing candidate is a RESULT, not a crash
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        if logger:
            logger.event("smoke_failed", label=label, error=result.error)

    return result


def _verify_resume(
    backend: Backend,
    loaded: LoadedModel,
    model_cfg: Any,
    cfg: Config,
    ckpt_dir: Path,
    optimizer: Any,
    image_px: int,
) -> tuple[bool, float]:
    """Save, reload into a fresh model, and check the next steps agree.

    `PLAN.md` 6.3: *"a resume that has never been tested does not work."* Saving
    and reloading without comparing subsequent loss proves only that files were
    written, so this runs the same steps from the same seed on both the live
    model and the restored one and compares.
    """
    import torch

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loaded.model.save_pretrained(str(ckpt_dir))
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    seed = cfg.seed + 12345
    live_losses, _ = _train_steps(
        loaded, steps=3, batch_size=cfg.train.per_device_batch, grad_accum=1,
        lr=cfg.train.lr, max_len=model_cfg.max_seq_len, image_px=image_px,
        seed=seed, optimizer=optimizer,
    )

    # Free the live model BEFORE loading the fresh one. Holding two copies of a
    # 2B model plus two optimizer states on a 15 GiB card is a needless way to
    # OOM during a test whose whole purpose is to prove the run survives.
    del optimizer
    loaded.model = None
    import gc

    gc.collect()
    reset_peak_memory()

    from peft import PeftModel

    fresh = backend.load(model_cfg)
    fresh.model = PeftModel.from_pretrained(fresh.model, str(ckpt_dir), is_trainable=True)
    fresh = backend.prepare_for_training(fresh, model_cfg)
    params = [p for p in fresh.model.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(params, lr=cfg.train.lr)
    opt2.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location="cpu"))

    resumed_losses, _ = _train_steps(
        fresh, steps=3, batch_size=cfg.train.per_device_batch, grad_accum=1,
        lr=cfg.train.lr, max_len=model_cfg.max_seq_len, image_px=image_px,
        seed=seed, optimizer=opt2,
    )

    delta = max(abs(a - b) for a, b in zip(live_losses, resumed_losses))
    # Tolerance, not equality: 4-bit matmuls and some CUDA kernels are not
    # bit-deterministic, so demanding exactness would fail for the wrong reason.
    return bool(delta < 1e-2), float(delta)


def write_report(results: list[SmokeResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "smoke_results.json"
    path.write_text(
        json.dumps(
            {
                "gates": {
                    "memory_gb": MEMORY_GATE_GB,
                    "full_run_hours": FULL_RUN_GATE_HOURS,
                    "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
                },
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
