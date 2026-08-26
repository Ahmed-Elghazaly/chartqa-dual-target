"""Report CI status for the CURRENT commit, not for whatever ran last.

Written after reporting "CI green" several times from a stale spot-check while
the pipeline had in fact been failing for eight consecutive runs. The failure was
an environment bug in the workflow rather than in the project, but the reporting
error was real and is exactly the stale-evidence mistake this project keeps
finding elsewhere.

Run:  python scripts/check_ci.py
"""

from __future__ import annotations

import json
import subprocess
import sys


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    local_dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                      text=True, check=True).stdout.strip())
    try:
        runs = json.loads(gh("run", "list", "--limit", "30", "--json",
                             "databaseId,conclusion,status,headSha,displayTitle"))
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        print(f"cannot query GitHub: {exc}")
        return 2

    print(f"HEAD {head[:12]}{'  (working tree dirty)' if local_dirty else ''}")
    for_head = [r for r in runs if r["headSha"] == head]
    if not for_head:
        print("  no CI run for this commit yet — push first, or it is still queueing.")
    for r in for_head:
        print(f"  {r['conclusion'] or r['status']:<12} {r['databaseId']}  {r['displayTitle'][:60]}")

    recent = runs[:10]
    tally: dict[str, int] = {}
    for r in recent:
        key = r["conclusion"] or r["status"]
        tally[key] = tally.get(key, 0) + 1
    print(f"\nlast {len(recent)} runs: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    failing = [r for r in recent if r["conclusion"] == "failure"]
    if failing:
        print(f"\n{len(failing)} of the last {len(recent)} runs FAILED. Inspect with:")
        print(f"  gh run view {failing[0]['databaseId']} --log-failed")
        return 1
    print("\nrecent CI is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
