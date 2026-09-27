from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    hf_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    revision: str = "89644892e4d85e24eaac8bacfd4f463576704203"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_on_vision: bool = True
    lora_on_language: bool = True
    max_seq_len: int = 1024
    image_max_pixels: int | None = None
    image_min_pixels: int | None = None
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    mixture_seed: int = 20250824
    chartqa_limit_per_kind: int = 30_000
    synthetic_stage1: int = 6_000
    stage1_cap: int = 1_000_000
    stage2_cap: int = 1_000_000
    synthetic_replay: int = 6_000


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
    grad_accum: int = 4
    optim: str = "adamw_fused"
    max_grad_norm: float = 1.0
    save_every_steps: int = 100
    eval_every_steps: int = 250
    metric_every_steps: int = 1000
    metric_batch_size: int = 8


@dataclass
class Config:
    run_name: str = "run"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _coerce(value: Any, tp: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = [a for a in typing.get_args(tp) if a is not type(None)]
    if value is None:
        if len(args) != len(typing.get_args(tp)):
            return None
        raise TypeError(f"{path}: None is not allowed")
    if origin is not None and origin is not list and len(args) == 1:
        tp = args[0]
        origin = typing.get_origin(tp)
    if origin is list:
        if isinstance(value, str):
            value = [v for v in (s.strip() for s in value.split(",")) if v]
        return [str(v) for v in value]
    if is_dataclass(tp):
        return from_dict(tp, value, path)
    if tp is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes", "on")
    if tp is int:
        return int(float(value))
    if tp is float:
        return float(value)
    return value


def from_dict(cls: Any, data: dict[str, Any], path: str = "") -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} at {path or cls.__name__}")
    hints = typing.get_type_hints(cls)
    return cls(**{name: _coerce(value, hints[name], f"{path}.{name}" if path else name)
                  for name, value in data.items()})


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml_tree(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    base = raw.pop("_base_", None)
    if base is None:
        return raw
    return _deep_merge(load_yaml_tree(p.parent / base), raw)


def apply_overrides(tree: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(tree))
    i = 0
    while i < len(overrides):
        key = overrides[i]
        if not key.startswith("--"):
            raise ValueError(f"expected an override starting with '--', got {key!r}")
        key = key[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        else:
            value = overrides[i + 1]
            i += 2
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def build_config(config_path: str | Path | None, overrides: list[str] | None = None) -> Config:
    tree = load_yaml_tree(config_path) if config_path else {}
    if overrides:
        tree = apply_overrides(tree, overrides)
    config: Config = from_dict(Config, tree)
    return config


def dump_config(cfg: Config, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False), encoding="utf-8")
    return path
