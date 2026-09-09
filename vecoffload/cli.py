"""
Command-line entry point.

    python -m vecoffload --list
    python -m vecoffload ablation
    python -m vecoffload all --outdir results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from .experiments import REGISTRY
from .latex_export import EXPORTERS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="vecoffload",
        description="Reproduce the experiments from the vehicular edge "
                    "offloading manuscript.")
    ap.add_argument("experiment", nargs="?", default="all",
                    help="experiment name, or 'all' (default)")
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    ap.add_argument("--outdir", default="results", help="directory for CSV output")
    ap.add_argument("--events", type=int, default=None,
                    help="override the number of decision epochs (faster smoke runs)")
    ap.add_argument("--latex", action="store_true",
                    help="also emit LaTeX rows / pgfplots coordinates")
    args = ap.parse_args(argv)

    if args.list:
        print("Available experiments:")
        for name in REGISTRY:
            print("  " + name)
        return 0

    names = list(REGISTRY) if args.experiment == "all" else [args.experiment]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"choose from: {', '.join(REGISTRY)}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    for name in names:
        fn = REGISTRY[name]
        kwargs = {}
        if args.events is not None and "n_events" in fn.__code__.co_varnames:
            kwargs["n_events"] = args.events
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        df = fn(**kwargs)
        print(df.round(3).to_string(index=False))
        print(f"[{time.time() - t0:.1f}s]")
        df.to_csv(outdir / f"{name}.csv", index=False)
        if args.latex and name in EXPORTERS:
            block = EXPORTERS[name](df)
            (outdir / f"{name}.tex.txt").write_text(block + "\n")
            print("\n-- LaTeX --")
            print(block)

    print(f"\nCSV written to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
