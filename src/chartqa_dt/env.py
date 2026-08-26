"""Runtime environment detection and path resolution.

Every path in this project is resolved through this module. Nothing anywhere
else may hard-code ``/kaggle/working`` or ``/content`` or ``~/data``.

Why this exists
---------------
The same code has to run unchanged on a Kaggle kernel, a Colab VM, a rented
cloud box and a laptop. Those four disagree about where it is safe to write, how
much of what you write survives the session, and how much disk you get. Encoding
that knowledge once, here, is what lets the rest of the codebase be
platform-blind.

Overrides
---------
Every resolved root can be overridden with an environment variable, which is how
the tests simulate platforms they are not running on:

``CDT_PLATFORM``, ``CDT_DATA_ROOT``, ``CDT_CACHE_ROOT``, ``CDT_OUTPUT_ROOT``.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Platform = Literal["kaggle", "colab", "local"]

_PLATFORMS: tuple[str, ...] = ("kaggle", "colab", "local")


def detect_platform() -> Platform:
    """Return the platform we are running on.

    ``CDT_PLATFORM`` wins if set, so tests (and a user on a rented box that
    happens to look like Colab) can force the answer.
    """
    forced = os.environ.get("CDT_PLATFORM", "").strip().lower()
    if forced:
        if forced not in _PLATFORMS:
            raise ValueError(f"CDT_PLATFORM={forced!r} is not one of {_PLATFORMS}")
        return forced  # type: ignore[return-value]

    # Kaggle sets KAGGLE_KERNEL_RUN_TYPE for both interactive and batch kernels.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"

    # Colab. The documented test is whether `google.colab` imports; checking
    # sys.modules alone is not enough, because in a fresh process nothing has
    # imported it yet. Environment variables are kept as secondary signals since
    # they are undocumented and have changed name across Colab generations.
    if _colab_module_importable():
        return "colab"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return "colab"
    if Path("/content").is_dir() and Path("/opt/deeplearning").exists():
        return "colab"

    return "local"


def _colab_module_importable() -> bool:
    """True inside a Colab runtime. Cheap and side-effect free.

    ``google.colab`` exists only in Colab, so a successful find is conclusive.
    ``find_spec`` is used rather than a real import so that nothing is executed
    just to answer a question about the environment.
    """
    if "google.colab" in sys.modules:
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _first_writable(*candidates: Path) -> Path:
    """Return the first candidate whose parent we can actually write to."""
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".cdt_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return c
        except OSError:
            continue
    raise RuntimeError(f"none of these paths is writable: {candidates}")


def _free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1024**3
    except OSError:
        return float("nan")


def vram_gb() -> float:
    """Total VRAM of device 0 in GiB, or 0.0 when there is no CUDA device.

    Imported lazily: the core package must remain installable without torch.
    """
    try:
        import torch
    except ImportError:
        return 0.0
    try:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:  # noqa: BLE001 - a broken CUDA install must not be fatal
        return 0.0


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


@dataclass(frozen=True)
class Environment:
    """Resolved roots for the current machine.

    ``data_root``   large, re-downloadable inputs (archives, extracted images).
    ``cache_root``  Hugging Face model/dataset caches.
    ``output_root`` run outputs: adapters, metrics, resolved configs, logs.
    """

    platform: Platform
    data_root: Path
    cache_root: Path
    output_root: Path
    vram_gb: float = 0.0
    gpu_name: str = "cpu"
    extras: dict[str, str] = field(default_factory=dict)

    def run_dir(self, run_name: str) -> Path:
        d = self.output_root / run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def describe(self) -> str:
        return (
            f"platform   : {self.platform}\n"
            f"gpu        : {self.gpu_name} ({self.vram_gb:.3f} GiB)\n"
            f"data_root  : {self.data_root}  (free {_free_gb(self.data_root):.1f} GiB)\n"
            f"cache_root : {self.cache_root}\n"
            f"output_root: {self.output_root}  (free {_free_gb(self.output_root):.1f} GiB)"
        )


def _local_roots() -> tuple[Path, Path, Path]:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "chartqa_dt"
    return base / "data", base / "hf", Path.cwd() / "outputs"


def _root_candidates(platform: Platform) -> tuple[list[Path], list[Path], list[Path]]:
    """Ordered fallback chains of (data, cache, output) roots for a platform.

    Pure: this function never touches the filesystem, so it states policy and can
    be asserted directly in tests that are not running on the platform. Choosing
    among the candidates is :func:`_first_writable`'s job.
    """
    ldata, lcache, lout = _local_roots()
    if platform == "kaggle":
        # /kaggle/working persists into the kernel's output and is size-capped,
        # so bulky re-downloadable data goes to /kaggle/temp first. Anything that
        # must survive the session is pushed to the Hub regardless.
        return (
            [Path("/kaggle/temp/cdt-data"), Path("/kaggle/working/cdt-data"), ldata],
            [Path("/kaggle/temp/cdt-cache"), Path("/kaggle/working/cdt-cache"), lcache],
            [Path("/kaggle/working/cdt-outputs"), lout],
        )
    if platform == "colab":
        return (
            [Path("/content/cdt-data"), ldata],
            [Path("/content/cdt-cache"), lcache],
            [Path("/content/cdt-outputs"), lout],
        )
    return [ldata], [lcache], [lout]


def _roots_for(platform: Platform) -> tuple[Path, Path, Path]:
    """The preferred (data_root, cache_root, output_root) for a platform."""
    d, c, o = _root_candidates(platform)
    return d[0], c[0], o[0]


def get_env(*, create: bool = True) -> Environment:
    """Detect the platform and resolve all roots.

    Set ``create=False`` in tests that only want to inspect the paths.
    """
    platform = detect_platform()
    data_c, cache_c, out_c = _root_candidates(platform)

    # An explicit override replaces the whole chain except the local last resort.
    if env_data := os.environ.get("CDT_DATA_ROOT"):
        data_c = [Path(env_data)]
    if env_cache := os.environ.get("CDT_CACHE_ROOT"):
        cache_c = [Path(env_cache)]
    if env_out := os.environ.get("CDT_OUTPUT_ROOT"):
        out_c = [Path(env_out)]

    if create:
        # Walk the chain and take the first root we can actually write to;
        # better a working run in the second-choice place than a crash.
        data, cache, out = _first_writable(*data_c), _first_writable(*cache_c), _first_writable(*out_c)
    else:
        data, cache, out = data_c[0], cache_c[0], out_c[0]

    # Point the HF libraries at our cache root so model downloads land on the
    # disk we chose rather than in the home directory of an ephemeral VM.
    os.environ.setdefault("HF_HOME", str(cache))

    return Environment(
        platform=platform,
        data_root=data,
        cache_root=cache,
        output_root=out,
        vram_gb=vram_gb(),
        gpu_name=gpu_name(),
        extras={"python": sys.version.split()[0]},
    )


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> list[str]:
    """Load ``KEY=value`` lines from a ``.env`` file into ``os.environ``.

    Returns the names (never the values) of the keys that were set, so a run log
    can record which credentials were present without ever recording a secret.
    """
    if path is None:
        here = Path(__file__).resolve()
        for parent in [Path.cwd(), *here.parents[:4]]:
            cand = parent / ".env"
            if cand.is_file():
                path = cand
                break
        else:
            return []
    p = Path(path)
    if not p.is_file():
        return []
    loaded: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
