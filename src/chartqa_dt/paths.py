from __future__ import annotations

import os
from pathlib import Path


def _root(variable: str, default: Path) -> Path:
    path = Path(os.environ.get(variable) or default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_root() -> Path:
    return _root("CDT_DATA_ROOT", Path.home() / ".cache" / "chartqa_dt" / "data")


def cache_root() -> Path:
    root = _root("CDT_CACHE_ROOT", Path.home() / ".cache" / "chartqa_dt" / "hf")
    if os.environ.get("CDT_CACHE_ROOT"):
        os.environ["HF_HOME"] = str(root)
    else:
        os.environ.setdefault("HF_HOME", str(root))
    return root


def output_root() -> Path:
    return _root("CDT_OUTPUT_ROOT", Path.cwd() / "outputs")


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'").strip('"')
