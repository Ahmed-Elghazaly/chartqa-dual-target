"""Shared CLI plumbing: one setup path every command goes through.

Every command resolves its config the same way, dumps the same provenance
record, and opens the same logger. Doing this once means a result produced by
``cdt-train`` and one produced by ``cdt-eval`` are traceable in exactly the same
way, and it is the mechanism behind PLAN 1.3's "any result must be traceable to
the exact settings that produced it".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from chartqa_dt.config import Config, build_config, dataclass_field_names, dump_resolved
from chartqa_dt.env import Environment, get_env, load_dotenv
from chartqa_dt.logging_utils import RunLogger
from chartqa_dt.seeding import SeedReport, set_seed


class NotYetBuilt(SystemExit):
    """Raised by a command whose phase has not been reached yet.

    Deliberately loud. A command that silently does nothing is worse than one
    that refuses, because you find out much later and cannot tell which runs
    were real.
    """

    def __init__(self, what: str, phase: str) -> None:
        super().__init__(
            f"\n'{what}' is not built yet — it lands in {phase}.\n"
            f"PLAN.md gates are hard: phases are built in order.\n"
        )


def base_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Any config field can be overridden on the command line using its dotted path, e.g.\n"
            "    --train.lr 5e-5  --data.max_examples 2000  --model.lora_on_vision false\n\n"
            "The fully resolved config, the git SHA and a dirty-tree flag are written to\n"
            "<output_dir>/resolved_config.yaml at the start of every run.\n"
        ),
    )
    p.add_argument("--config", type=str, default=None, help="path to a YAML config in configs/")
    p.add_argument("--run-name", type=str, default=None, help="name for this run's output directory")
    p.add_argument("--output-dir", type=str, default=None, help="override the resolved output directory")
    p.add_argument("--seed", type=int, default=None, help="master seed")
    p.add_argument("--dev", action="store_true", help="use the small development subset, no full download")
    p.add_argument("--no-wandb", action="store_true", help="disable the W&B mirror (JSONL mirror is always on)")
    p.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    p.add_argument("--list-fields", action="store_true", help="list every overridable config field and exit")
    return p


@dataclass
class Ctx:
    """Everything a command needs, resolved once."""

    cfg: Config
    env: Environment
    args: argparse.Namespace
    out_dir: Path
    seed_report: SeedReport
    dotenv_keys: list[str]

    def logger(self, **kw) -> RunLogger:
        from chartqa_dt.config import to_dict

        return RunLogger(
            self.out_dir,
            run_name=self.cfg.run_name,
            config=to_dict(self.cfg),
            wandb_enabled=self.cfg.wandb.enabled and not self.args.no_wandb,
            wandb_project=self.cfg.wandb.project,
            wandb_entity=self.cfg.wandb.entity,
            wandb_tags=self.cfg.wandb.tags,
            **kw,
        )


def setup(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> Ctx:
    args, overrides = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    if args.list_fields:
        for name in dataclass_field_names():
            print(name)
        raise SystemExit(0)

    dotenv_keys = load_dotenv()

    if args.run_name:
        overrides = [*overrides, "--run_name", args.run_name]
    if args.seed is not None:
        overrides = [*overrides, "--seed", str(args.seed)]
    if args.dev:
        overrides = [*overrides, "--data.dev", "true"]

    cfg = build_config(args.config, overrides)

    env = get_env()
    out_dir = Path(args.output_dir or cfg.output_dir or (env.output_root / cfg.run_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_report = set_seed(cfg.seed)

    if args.print_config:
        import yaml

        from chartqa_dt.config import to_dict

        print(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        raise SystemExit(0)

    dump_resolved(cfg, out_dir, extra={
        "platform": env.platform,
        "gpu": env.gpu_name,
        "vram_gb": round(env.vram_gb, 3),
        "seed_report": seed_report.describe(),
        "credentials_present": sorted(dotenv_keys),
    })

    print(env.describe())
    print(f"run_name   : {cfg.run_name}")
    print(f"output_dir : {out_dir}")
    print(f"seeding    : {seed_report.describe()}")
    return Ctx(cfg=cfg, env=env, args=args, out_dir=out_dir, seed_report=seed_report, dotenv_keys=dotenv_keys)
