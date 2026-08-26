"""Drive a Kaggle GPU kernel from this machine, non-interactively.

Every GPU phase of this project runs on a free Kaggle T4. This script packages
the repository, pushes it, starts a kernel, waits, and pulls the results back —
so a phase can be run and re-run without a browser.

Why a private Kaggle **dataset** rather than `git clone`
-------------------------------------------------------
The obvious design is to have the kernel clone the private GitHub repo. That
requires a GitHub token *inside* the kernel, which means writing a credential
into a remote service — for a repository that also holds the project's history.
Shipping the code as a private Kaggle dataset instead needs only the Kaggle
credential that already exists locally, and no token ever leaves this machine.

It also means the Phase 2 smoke test needs **no secrets at all**: it produces a
few kilobytes of JSON, returned as kernel output. Hugging Face push (Phase 6)
does need `HF_TOKEN` in Kaggle *Add-ons -> Secrets*, which is a one-time manual
step and is documented in SETUP.md rather than automated around.

Usage
-----
    python scripts/kaggle_run.py smoke                    # Phase 2 backbone test
    python scripts/kaggle_run.py smoke --steps 20 --dry-run
    python scripts/kaggle_run.py --status                 # poll the last kernel
    python scripts/kaggle_run.py --logs                   # fetch output + log
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "chartqa-dt-src"
KERNEL_SLUG = "chartqa-dt-run"

# Files the kernel needs. Deliberately explicit: a glob would eventually sweep up
# a `.env`, a dataset archive, or a chart image, and non-negotiable rule 7 says
# none of those may be redistributed.
INCLUDE = ["src", "configs", "pyproject.toml", "README.md"]
EXCLUDE_SUFFIXES = {".png", ".jpg", ".jpeg", ".zip", ".parquet", ".arrow", ".safetensors", ".bin", ".pt"}
EXCLUDE_NAMES = {".env", "kaggle.json", "access_token", "__pycache__", ".git", ".venv"}


def _api():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _username() -> str:
    """Kaggle slugs are lowercase; a mixed-case ref in `dataset_sources` silently
    fails to attach and the kernel starts with no code in /kaggle/input."""
    for path, key in (
        (Path.home() / ".kaggle" / "kaggle.json", "username"),
    ):
        if path.is_file():
            try:
                user = json.loads(path.read_text()).get(key)
                if user:
                    return user.lower()
            except json.JSONDecodeError:
                pass
    user = (os.environ.get("KAGGLE_USERNAME") or "").lower()
    if not user:
        raise SystemExit(
            "Cannot determine your Kaggle username.\n"
            "Set KAGGLE_USERNAME, or keep a kaggle.json with a 'username' field "
            "alongside ~/.kaggle/access_token."
        )
    return user


def _copy_sources(dest: Path) -> int:
    """Copy the package into a staging folder, skipping anything unshippable."""
    n = 0
    for item in INCLUDE:
        src = REPO_ROOT / item
        if not src.exists():
            continue
        if src.is_file():
            shutil.copy2(src, dest / src.name)
            n += 1
            continue
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            if any(part in EXCLUDE_NAMES for part in path.parts):
                continue
            rel = path.relative_to(REPO_ROOT)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            n += 1
    return n


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def push_dataset(api, staging: Path, *, notes: str) -> str:
    """Create or update the private code dataset. Returns its full slug.

    The staged tree is zipped into a single ``code.zip`` here rather than left to
    Kaggle's ``dir_mode``, whose behaviour proved inconsistent between an initial
    create and a later version — one produced individual files with their paths
    preserved, the next produced a ``src.zip``. Uploading one archive we control
    means the kernel never has to guess what it will find.
    """
    import zipfile

    user = _username()
    full = f"{user}/{DATASET_SLUG}"

    upload = staging.parent / f"{staging.name}_upload"
    if upload.exists():
        shutil.rmtree(upload)
    upload.mkdir(parents=True)

    with zipfile.ZipFile(upload / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(staging).as_posix())

    (upload / "dataset-metadata.json").write_text(
        json.dumps({"title": "chartqa-dt source", "id": full,
                    "licenses": [{"name": "CC0-1.0"}]}, indent=2),
        encoding="utf-8",
    )

    try:
        api.dataset_status(full)
        exists = True
    except Exception:  # noqa: BLE001 - "not found" is not exceptional here
        exists = False

    if exists:
        api.dataset_create_version(str(upload), version_notes=notes, dir_mode="skip", quiet=False)
    else:
        api.dataset_create_new(str(upload), public=False, dir_mode="skip", quiet=False)
    return full


# Raw string: this is Python source being written to a file, so escapes inside it
# must survive verbatim. A plain triple-quoted string turned a "\n" into a real
# newline and shipped a kernel that would not parse.
KERNEL_SCRIPT = r"""# Generated by scripts/kaggle_run.py. Do not edit here.
import glob
import os
import shutil
import subprocess
import sys
import zipfile

SRC = "/kaggle/input/{dataset}"
WORK = "/kaggle/working/repo"

# ---------------------------------------------------------------- fail fast
# Everything expensive comes after this. torch is preinstalled on Kaggle, so the
# accelerator can be checked in seconds -- before a 4.2 GB download and two model
# loads. A kernel that silently fell back to CPU would look identical to a slow
# one for the better part of an hour, and Kaggle exposes no logs while running.
import torch as _torch

_gpu = _torch.cuda.is_available()
print("accelerator:", _torch.cuda.get_device_name(0) if _gpu else "NONE (CPU)", flush=True)
if not _gpu and not {allow_cpu}:
    raise SystemExit(
        "no CUDA device. kernel-metadata.json set enable_gpu=true, so either the "
        "accelerator was not granted or the account is not phone-verified. "
        "Refusing to run a memory benchmark on CPU: the numbers would be "
        "meaningless and the run would take hours."
    )

listing = os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "(missing)"
print("contents of /kaggle/input:", listing, flush=True)
if not os.path.isdir(SRC):
    raise SystemExit(
        "code dataset not attached at " + SRC + " -- Kaggle normalises refs to "
        "lowercase, so check dataset_sources in kernel-metadata.json"
    )

# Kaggle extracts uploaded archives itself, but not always: the upload is one
# code.zip, and it may arrive as that archive or already expanded beside it.
# Handle both rather than betting on which.
os.makedirs(WORK, exist_ok=True)
archive = os.path.join(SRC, "code.zip")
if os.path.isfile(archive):
    print("found code.zip; extracting", flush=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(WORK)
elif os.path.isfile(os.path.join(SRC, "pyproject.toml")):
    print("dataset arrived already expanded; copying", flush=True)
    shutil.copytree(SRC, WORK, dirs_exist_ok=True)
else:
    raise SystemExit(
        "no code.zip and no pyproject.toml in " + SRC + ", found: " + str(os.listdir(SRC))
    )

root = WORK
if not os.path.exists(os.path.join(root, "pyproject.toml")):
    found = glob.glob(os.path.join(WORK, "*", "pyproject.toml"))
    if not found:
        raise SystemExit("no pyproject.toml under " + WORK + ": " + str(os.listdir(WORK)))
    root = os.path.dirname(found[0])
print("package root:", root, flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", root], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "peft>=0.14.0", "bitsandbytes>=0.44.0", "accelerate>=1.0.0"], check=True)

print("torch", _torch.__version__, flush=True)
subprocess.run([sys.executable, "-c",
                "from chartqa_dt.env import get_env; print(get_env().describe())"],
               cwd=root, check=True)

print("=" * 70, flush=True)
rc = subprocess.run({command}, cwd=root, check=False).returncode
print("command exit code:", rc, flush=True)

# Surface results in the kernel output so they come back with kernels_output().
for pattern in ("/kaggle/working/cdt-outputs/**/smoke_results.json",
                "/kaggle/working/cdt-outputs/**/metrics.jsonl",
                "/kaggle/working/cdt-outputs/**/resolved_config.yaml"):
    for f in glob.glob(pattern, recursive=True):
        dst = os.path.join(
            "/kaggle/working",
            os.path.basename(os.path.dirname(f)) + "__" + os.path.basename(f),
        )
        with open(f, "rb") as src_f, open(dst, "wb") as dst_f:
            dst_f.write(src_f.read())
        print("copied to output:", dst, flush=True)

sys.exit(rc)
"""


def render_kernel_script(dataset: str, command: list[str], *, allow_cpu: bool = False) -> str:
    """Render the kernel script, and refuse to ship one that does not parse."""
    script = KERNEL_SCRIPT.format(
        dataset=dataset, command=json.dumps(command), allow_cpu=repr(bool(allow_cpu))
    )
    compile(script, "main.py", "exec")  # generated code is code; check it
    return script


def push_kernel(api, dataset_slug: str, command: list[str], *, title: str, gpu: bool = True) -> str:
    user = _username()
    full = f"{user}/{KERNEL_SLUG}"
    staging = REPO_ROOT / ".kaggle_kernel"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (staging / "main.py").write_text(
        render_kernel_script(DATASET_SLUG, command, allow_cpu=not gpu), encoding="utf-8"
    )
    (staging / "kernel-metadata.json").write_text(
        json.dumps({
            "id": full,
            "title": title,
            "code_file": "main.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": gpu,
            "enable_internet": True,
            "dataset_sources": [dataset_slug],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=2),
        encoding="utf-8",
    )
    api.kernels_push(str(staging))
    return full


def wait_for(api, kernel: str, *, poll: int = 30, timeout_min: int = 240) -> str:
    deadline = time.time() + timeout_min * 60
    last = ""
    while time.time() < deadline:
        try:
            status = str(api.kernels_status(kernel))
        except Exception as exc:  # noqa: BLE001
            status = f"(status error: {type(exc).__name__})"
        if status != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {status}", flush=True)
            last = status
        low = status.lower()
        if any(k in low for k in ("complete", "error", "cancel")):
            return status
        time.sleep(poll)
    return "TIMEOUT"


def fetch_output(api, kernel: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_output(kernel, str(dest))
        print(f"  output -> {dest}")
        for f in sorted(dest.rglob("*")):
            if f.is_file():
                print(f"    {f.stat().st_size:>10,}  {f.relative_to(dest)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not fetch output: {type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", nargs="?", default="smoke", choices=["smoke"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--resolutions", type=str, default="512,native")
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-resume-test", action="store_true",
                   help="skip the kill-and-resume check (halves the number of model loads)")
    p.add_argument("--dry-run", action="store_true", help="stage everything, push nothing")
    p.add_argument("--status", action="store_true", help="poll the existing kernel and exit")
    p.add_argument("--logs", action="store_true", help="fetch output of the existing kernel and exit")
    p.add_argument("--timeout-min", type=int, default=240)
    args = p.parse_args()

    api = _api()
    user = _username()
    kernel = f"{user}/{KERNEL_SLUG}"

    if args.status:
        print(api.kernels_status(kernel))
        return 0
    if args.logs:
        fetch_output(api, kernel, REPO_ROOT / "outputs" / "kaggle")
        return 0

    command = [
        "cdt-train", "--stage", "smoke",
        "--config", "configs/model_qwen3vl2b.yaml",
        "--run-name", "phase2_smoke",
        "--steps", str(args.steps),
        "--resolutions", args.resolutions,
        "--no-wandb",
    ]
    if args.backend:
        command += ["--backend", args.backend]
    if args.no_resume_test:
        command += ["--no-resume-test"]

    staging = REPO_ROOT / ".kaggle_dataset"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    n = _copy_sources(staging)
    sha = _git_sha()
    print(f"staged {n} files from {REPO_ROOT.name} @ {sha}")
    print("command:", " ".join(command))

    if args.dry_run:
        print(f"dry run: staged at {staging}, nothing pushed")
        return 0

    print("\npushing code dataset ...")
    ds = push_dataset(api, staging, notes=f"{sha} {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  dataset: {ds}")

    print("pushing kernel ...")
    push_kernel(api, ds, command, title="chartqa-dt run", gpu=not args.no_gpu)
    print(f"  kernel: https://www.kaggle.com/code/{kernel}")

    print("\nwaiting (Ctrl-C is safe; re-attach with --status) ...")
    status = wait_for(api, kernel, timeout_min=args.timeout_min)
    print(f"\nfinal status: {status}")
    fetch_output(api, kernel, REPO_ROOT / "outputs" / "kaggle")
    return 0 if "complete" in status.lower() else 1


if __name__ == "__main__":
    sys.exit(main())
