"""``cdt-data`` — download, verify, load and mix the datasets (PLAN Phase 3)."""

from __future__ import annotations

from pathlib import Path

from chartqa_dt.cli._common import base_parser, setup


def main() -> None:
    p = base_parser("cdt-data",
                    "Download, hash-verify, load, deduplicate and mix the datasets.")
    p.add_argument("command", nargs="?", default="status",
                   choices=["status", "download", "verify", "audit", "dev"],
                   help="status: show what is present; download: fetch archives; "
                        "verify: re-hash against MANIFEST.json; audit: the 200-row "
                        "RefChartQA box audit; dev: materialise a ~200-example subset")
    p.add_argument("--datasets", type=str, default="chartqa",
                   help="comma-separated: chartqa, refchartqa, chartqapro")
    ctx = setup(p)

    from chartqa_dt.data.download import (
        MANIFEST_PATH,
        fetch_archive,
        load_manifest,
        materialise_dev_subset,
        verify_manifest,
    )
    from chartqa_dt.data.sources import SOURCES, ArchiveSpec

    keys = [k.strip() for k in ctx.args.datasets.split(",") if k.strip()]
    unknown = set(keys) - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown datasets: {sorted(unknown)}; expected {sorted(SOURCES)}")
    root = Path(ctx.env.data_root)
    command = ctx.args.command

    if command == "status":
        manifest = load_manifest()
        print(f"\ndata_root: {root}")
        print(f"manifest : {MANIFEST_PATH} "
              f"({len(manifest['archives'])} archive(s) recorded)")
        for key, entry in sorted(manifest["archives"].items()):
            print(f"  {key:<12} {entry['size_bytes']:>13,} B  "
                  f"sha256 {entry['sha256'][:16]}…  rev {entry['revision'][:12]}")
        if not manifest["archives"]:
            print("  nothing downloaded yet — run: cdt-data download")
        return

    if command == "download":
        for key in keys:
            spec = SOURCES[key]
            if not isinstance(spec, ArchiveSpec):
                print(f"{key}: read through `datasets`, no archive to fetch. "
                      f"Use `cdt-data dev` for a subset.")
                continue
            result = fetch_archive(spec, data_root=root)
            print(f"{key}: {result.size_bytes:,} B  sha256 {result.sha256[:16]}…  "
                  f"{'cached' if result.cached else 'downloaded'}")
        return

    if command == "verify":
        status = verify_manifest(data_root=root)
        for key, state in sorted(status.items()):
            print(f"  {key:<12} {state}")
        if any(v == "MISMATCH" for v in status.values()):
            raise SystemExit("a recorded archive no longer hashes to its manifest entry")
        return

    if command == "dev":
        for key in keys:
            path = materialise_dev_subset(key, data_root=root)
            print(f"{key}: dev subset at {path}")
        return

    if command == "audit":
        import sys

        from scripts.refchartqa_audit import main as run

        sys.argv = ["cdt-data-audit"]
        run()
        return


if __name__ == "__main__":
    main()
