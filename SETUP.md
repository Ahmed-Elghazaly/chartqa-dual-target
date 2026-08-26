# Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/check_credentials.py     # verifies all three credentials
```

`check_credentials.py` is the single source of truth. It only calls endpoints that genuinely
require authentication, and it runs a **negative control** — the same call with a junk token,
which must be rejected. Without that control the check would be worthless; see the Kaggle note
below for why.

---

## Kaggle — two token systems, and they are not interchangeable

This is the one that costs people an afternoon.

| | New access token | Legacy API key |
|---|---|---|
| Looks like | `KGAT_…` (37 chars) | 32 hex characters |
| Created by | **"Generate New Token"** at [kaggle.com/settings/api](https://www.kaggle.com/settings/api) | **"Create Legacy API Key"**, same page |
| Goes in | `~/.kaggle/access_token` (raw text, no JSON) **or** `KAGGLE_API_TOKEN` | `~/.kaggle/kaggle.json` as `{"username":…, "key":…}` |
| Sent as | `Authorization: Bearer <token>` | HTTP Basic `username:key` |

**A `KGAT_` token placed in `kaggle.json` fails on every authenticated endpoint.** The client
sends it as Basic `username:key`, which is not what a bearer token is. There is no
permissions/scope UI to fix — the token is fine, the file is wrong.

```bash
printf '%s' 'KGAT_your_token_here' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

If you already have it in `kaggle.json`, move it across:

```bash
python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['key'],end='')" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

### Two traps that make a working token look broken

1. **Some Kaggle endpoints return HTTP 200 to unauthenticated requests.**
   `GET /api/v1/datasets/list` returns 200 with no credentials, with deliberately invalid
   credentials, and with a valid token — identically. During setup this made a wholly rejected
   token look like a partially-working one ("datasets work, kernels don't → must be a missing
   scope"), which was completely wrong. Only ever test auth against an endpoint that requires it,
   and always run the junk-credential control.

2. **A python.org Python on macOS has no CA trust store.** Raw `urllib` raises
   `CERTIFICATE_VERIFY_FAILED`, which surfaces as a connection error and reads like a rejected
   credential. `requests` bundles `certifi` and works. Every network call in this project uses
   `requests` for that reason. If you hit it elsewhere, run
   `/Applications/Python\ 3.11/Install\ Certificates.command`.

**Phone verification** is separately required before Kaggle notebooks may use a GPU or the
internet: [kaggle.com/settings](https://www.kaggle.com/settings) → Phone Verification.

---

## Hugging Face

A **write**-capable token, used for the private artifact repo that checkpoints are pushed to.
Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens); a fine-grained
token needs `repo.write` on your own namespace.

```bash
printf 'HF_TOKEN=hf_your_token_here\n' > .env      # .env is gitignored, and CI fails if one enters history
```

## GitHub

No separate token needed if the `gh` CLI is authenticated with the `repo` scope:

```bash
gh auth status                                     # expect: Logged in, scopes include 'repo'
gh auth refresh -h github.com -s repo              # only if 'repo' is missing
```

For a Kaggle or Colab notebook to clone the **private** repo it needs a token in its own secret
store; `gh auth token` prints a usable one. Add it as `GITHUB_TOKEN` under Kaggle *Add-ons →
Secrets* or the Colab 🔑 panel, along with `GITHUB_USER` and `HF_TOKEN`.

---

## Free-GPU hosts

| | Kaggle | Colab |
|---|---|---|
| Accelerator | GPU T4 ×2 (one is used) | Runtime → Change runtime type → T4 |
| Internet | must be **On** (needs phone verification) | on by default |
| Session budget | ~30 GPU-hours/week, ~12 h sessions | shorter, less predictable |
| Notebook | `notebooks/kaggle_run.ipynb` | `notebooks/colab_run.ipynb` |

Everything long is resumable and pushes checkpoints to the private HF repo on every save, because
these sessions are killed without warning.
