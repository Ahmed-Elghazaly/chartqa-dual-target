"""Verify every credential this project needs, and say precisely what is wrong.

Run:  python scripts/check_credentials.py

Why this exists
---------------
Diagnosing a credential by whether a command "works" is unreliable, because
several Kaggle endpoints return 200 to completely unauthenticated requests.
During setup, `datasets/list` returned 200 with no credentials at all, with
deliberately invalid credentials, and with the real token — which made a
rejected token look like a partially-working one and sent the diagnosis in the
wrong direction entirely.

So every check here calls an endpoint that genuinely requires authentication,
and the Kaggle check includes a negative control: the same call with junk
credentials, which MUST fail. If the control passes, the check is meaningless
and says so.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "


def _print(status: str, name: str, detail: str = "") -> None:
    print(f"[{status}] {name:<28} {detail}")


# --------------------------------------------------------------------------- #
# Kaggle
# --------------------------------------------------------------------------- #

def _kaggle_token() -> tuple[str | None, str]:
    """Return (token, where_it_came_from).

    Kaggle has two token systems and they are NOT interchangeable:

    * **New access token** — looks like ``KGAT_...``. Goes in
      ``~/.kaggle/access_token`` (raw text) or the ``KAGGLE_API_TOKEN``
      environment variable. Created by "Generate New Token" on
      https://www.kaggle.com/settings/api
    * **Legacy API key** — 32 hex characters, goes in ``~/.kaggle/kaggle.json``
      as ``{"username": ..., "key": ...}``. Created by "Create Legacy API Key".

    Putting a ``KGAT_`` token into ``kaggle.json``'s ``key`` field fails on every
    authenticated endpoint, because the client then sends it as HTTP Basic
    ``username:key`` instead of as a bearer token.
    """
    if tok := os.environ.get("KAGGLE_API_TOKEN"):
        return tok.strip(), "KAGGLE_API_TOKEN env var"
    p = Path.home() / ".kaggle" / "access_token"
    if p.is_file() and (tok := p.read_text(encoding="utf-8").strip()):
        return tok, str(p)
    j = Path.home() / ".kaggle" / "kaggle.json"
    if j.is_file():
        try:
            key = json.loads(j.read_text(encoding="utf-8")).get("key", "").strip()
        except json.JSONDecodeError:
            return None, "kaggle.json is not valid JSON"
        if key.startswith("KGAT_"):
            return key, f"{j} (WRONG FILE for a KGAT_ token)"
        if key:
            return key, f"{j} (legacy key)"
    return None, "not found"


def _kaggle_call(token: str, *, legacy_user: str | None = None, retries: int = 2) -> int:
    """Hit an endpoint that genuinely requires auth. Returns the HTTP status.

    Uses ``requests`` rather than ``urllib`` on purpose: ``requests`` ships its
    own CA bundle via ``certifi``. A python.org install on macOS has no system
    trust store wired up, so raw ``urllib`` raises CERTIFICATE_VERIFY_FAILED —
    which surfaces as a connection error and is easily misread as a rejected
    credential. That exact confusion cost real time during setup.

    Retries transient socket errors: Kaggle briefly throttles a client that has
    just presented bad credentials.
    """
    import requests

    url = "https://www.kaggle.com/api/v1/competitions/list?page=1"
    if legacy_user:
        blob = base64.b64encode(f"{legacy_user}:{token}".encode()).decode()
        headers = {"Authorization": f"Basic {blob}"}
    else:
        headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(retries + 1):
        try:
            return requests.get(url, headers=headers, timeout=30).status_code
        except requests.RequestException:
            if attempt == retries:
                return -1
            time.sleep(2 * (attempt + 1))
    return -1


def check_kaggle() -> bool:
    token, where = _kaggle_token()
    if not token:
        _print(BAD, "Kaggle", "no token found")
        print("        Create one at https://www.kaggle.com/settings/api -> 'Generate New Token',")
        print("        then:  printf '%s' 'KGAT_...' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token")
        return False

    if "WRONG FILE" in where:
        _print(BAD, "Kaggle", "KGAT_ token is in kaggle.json, which cannot work")
        print("        A KGAT_ token is a bearer token, not a legacy username:key pair.")
        print("        Fix:  printf '%s' \"$(python3 -c \"import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['key'])\")\" > ~/.kaggle/access_token")
        print("        chmod 600 ~/.kaggle/access_token")
        return False

    status = _kaggle_call(token)
    if status != 200:
        _print(BAD, "Kaggle", f"HTTP {status} using token from {where}")
        return False

    # Negative control, run AFTER the real check so a throttle triggered by bad
    # credentials cannot corrupt the real result. Junk must be rejected, or the
    # endpoint does not require auth and the whole check proves nothing.
    control = _kaggle_call("KGAT_definitely_not_a_real_token_0000", retries=0)
    if control == 200:
        _print(WARN, "Kaggle", "negative control PASSED — endpoint does not require auth; result is meaningless")
        return False

    _print(OK, "Kaggle", f"authenticated (token from {where}; control rejected with HTTP {control})")
    return True


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #

def check_hf() -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from chartqa_dt.env import load_dotenv
    from chartqa_dt.hub import get_token

    load_dotenv()
    token = get_token()
    if not token:
        _print(BAD, "Hugging Face", "no token (set HF_TOKEN in .env)")
        return False
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).whoami()
    except Exception as exc:  # noqa: BLE001
        _print(BAD, "Hugging Face", f"{type(exc).__name__}: {str(exc)[:80]}")
        return False

    auth = info.get("auth", {}).get("accessToken", {})
    perms: list[str] = []
    for scope in auth.get("fineGrained", {}).get("scoped", []):
        perms += scope.get("permissions", [])
    can_write = "repo.write" in perms or auth.get("role") == "write"
    _print(OK if can_write else WARN, "Hugging Face",
           f"user={info.get('name')} role={auth.get('role')} write={can_write}")
    if not can_write:
        print("        A write-capable token is required to push adapters and checkpoints.")
    return can_write


# --------------------------------------------------------------------------- #
# GitHub (via the gh CLI — no separate token needed)
# --------------------------------------------------------------------------- #

def check_github() -> bool:
    try:
        r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        _print(BAD, "GitHub (gh CLI)", "gh not installed")
        return False
    out = r.stdout + r.stderr
    if "Logged in" not in out:
        _print(BAD, "GitHub (gh CLI)", "not logged in — run: gh auth login")
        return False
    account = next((ln.split("account")[-1].strip() for ln in out.splitlines() if "Logged in" in ln), "?")
    has_repo = "'repo'" in out or " repo," in out or "repo'" in out
    _print(OK if has_repo else WARN, "GitHub (gh CLI)",
           f"account={account} repo_scope={has_repo}")
    if not has_repo:
        print("        Need the 'repo' scope: gh auth refresh -h github.com -s repo")
    return has_repo


def main() -> int:
    print("Credential check\n" + "-" * 62)
    results = {"kaggle": check_kaggle(), "huggingface": check_hf(), "github": check_github()}
    print("-" * 62)
    missing = [k for k, v in results.items() if not v]
    if missing:
        print(f"NOT READY: {', '.join(missing)}")
        return 1
    print("All credentials present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
