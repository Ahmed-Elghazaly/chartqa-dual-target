"""Run logging: Weights & Biases when available, always a local JSONL mirror.

The mirror is not a nicety. Free GPU sessions die without warning, W&B needs a
network that Kaggle sometimes does not have, and a training run whose only
record was in a browser tab is a run you cannot report. Every metric is written
to ``<output_dir>/metrics.jsonl`` first; W&B is strictly a bonus and can never
crash the run (PLAN 1.6).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "item"):  # numpy / torch scalars
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            pass
    return str(v)


class RunLogger:
    """Append-only metric log with an optional W&B mirror.

    Usage::

        with RunLogger(out_dir, run_name="stage1", config=cfg_dict) as log:
            log.log({"loss": 1.23}, step=10)
            log.event("checkpoint", path=str(p))
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_name: str = "run",
        config: dict[str, Any] | None = None,
        wandb_enabled: bool = True,
        wandb_project: str = "chartqa-dual-target",
        wandb_entity: str | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "metrics.jsonl"
        self.run_name = run_name
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.time()
        self._wandb = None

        self.event("run_start", run_name=run_name, argv=" ".join(sys.argv), config=config or {})

        if wandb_enabled and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self._wandb = wandb.init(
                    project=wandb_project, entity=wandb_entity, name=run_name,
                    config=config or {}, tags=wandb_tags or [], reinit=True,
                )
            except Exception as exc:  # noqa: BLE001 - never let logging kill a run
                self.event("wandb_init_failed", error=f"{type(exc).__name__}: {exc}")
                self._wandb = None
        elif wandb_enabled:
            self.event("wandb_skipped", reason="WANDB_API_KEY not set")

    # ------------------------------------------------------------------ #

    def _write(self, record: dict[str, Any]) -> None:
        record = {"t": round(time.time() - self._t0, 3), **record}
        self._fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        self._fh.flush()  # a killed session must not lose the last lines
        os.fsync(self._fh.fileno())

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        self._write({"kind": "metrics", "step": step, **metrics})
        if self._wandb is not None:
            try:
                self._wandb.log(_jsonable(metrics), step=step)
            except Exception as exc:  # noqa: BLE001
                self._write({"kind": "event", "event": "wandb_log_failed", "error": str(exc)})
                self._wandb = None

    def event(self, event: str, **fields: Any) -> None:
        self._write({"kind": "event", "event": event, **fields})

    def summary(self, **fields: Any) -> None:
        self._write({"kind": "summary", **fields})
        if self._wandb is not None:
            try:
                for k, v in fields.items():
                    self._wandb.summary[k] = _jsonable(v)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self.event("run_end", elapsed_s=round(time.time() - self._t0, 3))
        try:
            self._fh.close()
        finally:
            if self._wandb is not None:
                with contextlib.suppress(Exception):
                    self._wandb.finish()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read a metrics.jsonl back, tolerating a truncated final line."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a session killed mid-write leaves one partial line
    return out
