"""Typed configuration: YAML files, CLI overrides, and a resolved dump.

Design rules, each of which exists because of a specific way experiments go wrong:

1. **Typed dataclasses, not dicts.** A dict lets ``cfg["lr"]`` silently be a
   string ``"5e-5"`` that trains at a nonsense rate. Field types coerce.
2. **Unknown keys are a hard error.** A typo like ``--train.leraning_rate`` that
   is quietly ignored produces a run you believe used your setting and did not.
   This is the single most common way a "reproducible" experiment isn't.
3. **The fully resolved config is dumped at the start of every run**, together
   with the git SHA and a dirty-tree flag. Any number this project reports must
   be traceable to the exact settings and exact code that produced it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    hf_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    hf_id_4bit: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    backend: str = "hf_peft"  # "hf_peft" | "unsloth"
    load_in_4bit: bool = True
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"

    # LoRA. Rank 16 on BOTH vision and language sides (IDEA.md 8, rule 3).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_on_vision: bool = True
    lora_on_language: bool = True

    max_seq_len: int = 1024
    # Image budget. `image_max_pixels` maps to the processor's
    # size.longest_edge; None means "leave the model's own default alone".
    # NOTE (decision 0008): the visual-token factor is derived from the loaded
    # processor (patch_size * merge_size), never hard-coded.
    image_max_pixels: int | None = 512 * 512
    image_min_pixels: int | None = None
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    dev: bool = False
    dev_size: int = 200
    mixture_path: str | None = None
    max_examples: int | None = None
    # Caps from IDEA.md 6.3 / PLAN.md 3.7.
    refchartqa_train_cap: int = 4000
    stage1_cap: int = 12000
    stage2_cap: int = 12000
    synthetic_replay: int = 2000
    num_workers: int = 2
    box_field: str = "bbox"  # decision 0006: measured against "bbox_2d" in Phase 5.1


@dataclass
class TrainConfig:
    stage: str = "stage1"
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    epochs: float = 1.0
    max_steps: int | None = None
    per_device_batch: int = 2
    grad_accum: int = 4  # effective batch 8
    optim: str = "adamw_8bit"
    max_grad_norm: float = 1.0
    save_every_steps: int = 100
    eval_every_steps: int = 250
    early_stop_patience: int | None = None
    resume_from: str | None = None
    push_to_hub_every_save: bool = True


@dataclass
class EvalConfig:
    dataset: str = "chartqa"  # "chartqa" | "refchartqa" | "chartqapro"
    split: str = "val"  # test splits stay sealed until Phase 7 (rule 1)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    bootstrap_resamples: int = 10000
    bootstrap_alpha: float = 0.05
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    structured_output: bool = True
    batch_size: int = 1
    limit: int | None = None


@dataclass
class HubConfig:
    repo_id: str | None = None  # e.g. "user/chartqa-dt-artifacts"
    private: bool = True
    enabled: bool = True


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "chartqa-dual-target"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    run_name: str = "unnamed"
    seed: int = 0
    output_dir: str | None = None  # resolved from env when None
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# --------------------------------------------------------------------------- #
# Construction from nested dicts, with strict key checking
# --------------------------------------------------------------------------- #


def _is_optional(tp: Any) -> tuple[bool, Any]:
    """Return (is_optional, inner_type) for ``X | None`` / ``Optional[X]``."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(typing.get_args(tp)) != len(args):
            return True, args[0] if len(args) == 1 else tp
    return False, tp


def _coerce(value: Any, tp: Any, path: str) -> Any:
    optional, inner = _is_optional(tp)
    if value is None:
        if optional:
            return None
        raise TypeError(f"{path}: None is not allowed for type {tp}")
    tp = inner

    origin = typing.get_origin(tp)
    if origin in (list, typing.List):  # noqa: UP006
        (arg,) = typing.get_args(tp) or (Any,)
        if isinstance(value, str):
            value = [v for v in (s.strip() for s in value.split(",")) if v]
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(v, arg, f"{path}[{i}]") for i, v in enumerate(value)]

    if is_dataclass(tp):
        return from_dict(tp, value, path=path)

    if tp is bool:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise TypeError(f"{path}: cannot read {value!r} as a boolean")

    if tp is int:
        # Accept 1e3 / "1000" but refuse a silent truncation of 0.5 -> 0.
        f = float(value)
        if f != int(f):
            raise TypeError(f"{path}: {value!r} is not an integer")
        return int(f)

    if tp is float:
        return float(value)
    if tp is str:
        return str(value)
    if tp is Any:
        return value
    return value


def from_dict(cls: Any, data: Any, path: str = "") -> Any:
    """Build a dataclass from a nested dict. Unknown keys raise."""
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = path or cls.__name__
        raise KeyError(
            f"unknown config key(s) {sorted(unknown)} at {where}. "
            f"Valid keys here: {sorted(known)}. "
            "(Unknown keys are fatal on purpose: a typo that is silently ignored "
            "produces a run you believe used your setting and did not.)"
        )
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name in data:
            kwargs[name] = _coerce(data[name], hints[name], f"{path}.{name}" if path else name)
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# YAML loading with `_base_` inheritance
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_tree(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML config, resolving a ``_base_`` chain relative to each file."""
    p = Path(path).resolve()
    _seen = _seen or set()
    if p in _seen:
        raise ValueError(f"circular _base_ chain at {p}")
    _seen.add(p)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{p}: top level of a config must be a mapping")
    bases = raw.pop("_base_", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = _deep_merge(merged, load_yaml_tree((p.parent / b), set(_seen)))
    return _deep_merge(merged, raw)


# --------------------------------------------------------------------------- #
# CLI overrides:  --train.lr 5e-5  /  --data.max_examples=2000  /  --model.lora_on_vision false
# --------------------------------------------------------------------------- #


def apply_overrides(tree: dict, overrides: list[str]) -> dict:
    """Apply ``--a.b.c value`` pairs onto a nested dict, in place-ish."""
    out = json.loads(json.dumps(tree))  # cheap deep copy of plain data
    i = 0
    while i < len(overrides):
        tok = overrides[i]
        if not tok.startswith("--"):
            raise ValueError(f"expected an override starting with '--', got {tok!r}")
        key = tok[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(overrides):
                raise ValueError(f"override --{key} has no value")
            value = overrides[i + 1]
            i += 2
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"--{key}: '{part}' is not a section")
        node[parts[-1]] = yaml.safe_load(value)  # gives int/float/bool/str naturally
    return out


def build_config(config_path: str | Path | None, overrides: list[str] | None = None) -> Config:
    tree = load_yaml_tree(config_path) if config_path else {}
    if overrides:
        tree = apply_overrides(tree, overrides)
    return from_dict(Config, tree)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Commit SHA, branch, and whether the tree is dirty.

    A dirty tree means the code that ran is not the code at that SHA, so the
    result is not reproducible from the SHA alone. We record it rather than
    forbid it — refusing to run would just make people commit noise.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[2])

    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status.strip()),
        "git_dirty_files": [] if not status else [ln[3:] for ln in status.splitlines()][:50],
    }


def dump_resolved(cfg: Config, output_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Write ``resolved_config.yaml`` at the start of a run. Mandatory (PLAN 1.3)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": to_dict(cfg),
        "provenance": git_provenance(),
        "environment": {
            k: v for k, v in os.environ.items()
            if k.startswith(("CDT_", "KAGGLE_KERNEL", "COLAB_")) and "TOKEN" not in k and "KEY" not in k
        },
    }
    if extra:
        payload["extra"] = extra
    path = out / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False), encoding="utf-8")
    return path


def dataclass_field_names(cls: Any = Config) -> list[str]:
    """Flat dotted names of every config field, for CLI help and tests."""
    names: list[str] = []

    def walk(c: Any, prefix: str) -> None:
        for f in dataclasses.fields(c):
            tp = typing.get_type_hints(c)[f.name]
            _, inner = _is_optional(tp)
            if is_dataclass(inner):
                walk(inner, f"{prefix}{f.name}.")
            else:
                names.append(f"{prefix}{f.name}")

    walk(cls, "")
    return names
