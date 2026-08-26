"""Seeding, and an honest account of what it does and does not buy you.

``set_seed`` makes a run *repeatable on the same machine with the same library
versions* for everything that draws from Python's ``random``, NumPy or PyTorch.

It does **not** make results bit-identical across machines, GPU models, or
library versions. Several CUDA kernels — notably some backward passes used by
attention and by convolutions — accumulate in a non-deterministic order, and
floating-point addition is not associative, so the same seed can produce
slightly different weights on two different cards. cuBLAS additionally needs an
environment variable set *before* the CUDA context is created.

The project's response to this is not to pretend: seeds are recorded, evaluation
is run across three of them, and headline numbers carry bootstrap confidence
intervals so that run-to-run variance is visible instead of hidden.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

# Must be set before the CUDA context exists for deterministic cuBLAS GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class SeedReport:
    seed: int
    python: bool = True
    numpy: bool = False
    torch: bool = False
    torch_cuda: bool = False
    deterministic_algorithms: bool = False
    note: str = ""

    def describe(self) -> str:
        bits = [f"seed={self.seed}", f"python={self.python}", f"numpy={self.numpy}", f"torch={self.torch}"]
        if self.torch:
            bits += [f"cuda={self.torch_cuda}", f"deterministic={self.deterministic_algorithms}"]
        if self.note:
            bits.append(f"note={self.note}")
        return "  ".join(bits)


def set_seed(seed: int, *, deterministic: bool = True) -> SeedReport:
    """Seed Python, NumPy and PyTorch. Safe to call without NumPy or PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_ok = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_ok = True
    except ImportError:
        pass

    torch_ok = cuda_ok = det_ok = False
    note = ""
    try:
        import torch

        torch.manual_seed(seed)
        torch_ok = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_ok = True
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                det_ok = True
            except Exception as exc:  # noqa: BLE001
                note = f"deterministic algorithms unavailable: {type(exc).__name__}"
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        pass

    return SeedReport(
        seed=seed, numpy=numpy_ok, torch=torch_ok, torch_cuda=cuda_ok,
        deterministic_algorithms=det_ok, note=note,
    )


def rng_state() -> dict[str, Any]:
    """Capture every RNG state, for a checkpoint that resumes exactly (PLAN 6.3)."""
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    """Restore states captured by :func:`rng_state`. Missing keys are skipped."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass
