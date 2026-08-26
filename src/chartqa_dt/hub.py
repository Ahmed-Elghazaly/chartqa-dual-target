"""Push and pull artifacts to a **private** Hugging Face Hub repository.

Why this exists
---------------
Free Kaggle and Colab sessions are killed without warning, and everything on
local disk goes with them. A checkpoint that was never pushed is a checkpoint
that does not exist. So: every save writes locally *and* pushes, and long jobs
resume by pulling the last pushed state (PLAN 1.7, 6.3).

Two hard rules are enforced here rather than left to discipline:

* repositories are created **private** (non-negotiable rule 8);
* :func:`assert_no_dataset_content` refuses to upload chart images or dataset
  archives (non-negotiable rule 7 — ChartQA is GPL-3.0, RefChartQA AGPL-3.0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Extensions that may never be pushed: they are chart images or dataset dumps.
_FORBIDDEN_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".zip", ".parquet", ".arrow"}
# ...unless they live under a directory we have explicitly cleared (our own
# generated charts for the demo, and report figures we drew ourselves).
_ALLOWED_IMAGE_DIRS = ("report/figures", "demo/examples", "figures", "qualitative")


class HubError(RuntimeError):
    pass


def get_token() -> str | None:
    """HF token from the environment, then from the standard CLI login file."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok.strip()
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if p.is_file():
            tok = p.read_text(encoding="utf-8").strip()
            if tok:
                return tok
    return None


def assert_no_dataset_content(root: str | Path) -> None:
    """Raise if a directory about to be uploaded contains dataset content.

    Rule 7 is easy to break by accident — one stray qualitative-example PNG in a
    results folder and a GPL-3.0 chart image is on the Hub. Checking is cheap.
    """
    root = Path(root)
    offenders: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _FORBIDDEN_SUFFIXES:
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in rel for part in _ALLOWED_IMAGE_DIRS):
            continue
        offenders.append(rel)
    if offenders:
        raise HubError(
            "refusing to upload dataset content (non-negotiable rule 7). "
            f"Offending files under {root}: {offenders[:10]}"
            + (f" ... and {len(offenders) - 10} more" if len(offenders) > 10 else "")
        )


@dataclass
class HubStore:
    """Thin wrapper over ``huggingface_hub`` scoped to one private repo."""

    repo_id: str
    repo_type: str = "model"
    private: bool = True
    token: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        self.token = self.token or get_token()
        if self.enabled and not self.token:
            self.enabled = False
            self._disabled_reason = "no HF token found (set HF_TOKEN)"
        else:
            self._disabled_reason = ""

    # ------------------------------------------------------------------ #

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover
            raise HubError("huggingface_hub is not installed") from exc
        return HfApi(token=self.token)

    def ensure_repo(self) -> str:
        """Create the repo if it does not exist. Always private."""
        if not self.enabled:
            raise HubError(f"hub disabled: {self._disabled_reason}")
        api = self._api()
        api.create_repo(
            repo_id=self.repo_id, repo_type=self.repo_type,
            private=self.private, exist_ok=True,
        )
        return self.repo_id

    def push_dir(
        self,
        local_dir: str | Path,
        path_in_repo: str,
        *,
        commit_message: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        strict: bool = True,
    ) -> bool:
        """Upload a folder. Returns True on success.

        ``strict=False`` downgrades a failure to a return value, for the case
        where losing a periodic push is better than losing the whole run.
        """
        local_dir = Path(local_dir)
        if not self.enabled:
            if strict:
                raise HubError(f"hub disabled: {self._disabled_reason}")
            return False
        assert_no_dataset_content(local_dir)
        try:
            self.ensure_repo()
            self._api().upload_folder(
                folder_path=str(local_dir),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=commit_message or f"push {path_in_repo}",
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns or ["*.png", "*.jpg", "*.zip", "*.parquet"],
            )
            return True
        except Exception as exc:
            if strict:
                raise HubError(f"push_dir failed: {type(exc).__name__}: {exc}") from exc
            return False

    def pull_dir(self, path_in_repo: str, local_dir: str | Path, *, strict: bool = True) -> Path | None:
        """Download a subfolder of the repo into ``local_dir``."""
        local_dir = Path(local_dir)
        if not self.enabled:
            if strict:
                raise HubError(f"hub disabled: {self._disabled_reason}")
            return None
        try:
            from huggingface_hub import snapshot_download

            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=self.repo_id, repo_type=self.repo_type, token=self.token,
                allow_patterns=[f"{path_in_repo.rstrip('/')}/*"], local_dir=str(local_dir),
            )
            return local_dir / path_in_repo
        except Exception as exc:
            if strict:
                raise HubError(f"pull_dir failed: {type(exc).__name__}: {exc}") from exc
            return None

    def exists(self, path_in_repo: str) -> bool:
        if not self.enabled:
            return False
        try:
            files = self._api().list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
            prefix = path_in_repo.rstrip("/") + "/"
            return any(f == path_in_repo or f.startswith(prefix) for f in files)
        except Exception:  # noqa: BLE001
            return False

    def status(self) -> str:
        if not self.enabled:
            return f"hub DISABLED ({self._disabled_reason})"
        return f"hub enabled -> {self.repo_type}:{self.repo_id} (private={self.private})"
