#!/usr/bin/env python3
"""
Promote a completed eval run to the stored baseline.

  python scripts/save_baseline.py                           # 100-scene → artifacts/baseline
  python scripts/save_baseline.py --src artifacts/ci_eval --out artifacts/ci_baseline
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_METRICS_GLOB = "metrics_*.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote eval metrics to baseline")
    ap.add_argument("--src", type=Path, default=Path("artifacts/eval"),
                    help="Source eval directory (default: artifacts/eval)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/baseline"),
                    help="Destination baseline directory (default: artifacts/baseline)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing baseline without prompting")
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"ERROR: {args.src} does not exist — run scripts/run_eval.py first")

    csv_files = list(args.src.glob(_METRICS_GLOB))
    if not csv_files:
        sys.exit(f"ERROR: no {_METRICS_GLOB} files found in {args.src}")

    if args.out.exists() and not args.force:
        reply = input(f"  Overwrite existing baseline at {args.out}? [y/N] ").strip().lower()
        if reply != "y":
            print("  Aborted.")
            return

    args.out.mkdir(parents=True, exist_ok=True)
    for f in csv_files:
        shutil.copy2(f, args.out / f.name)
        print(f"  copied  {f.name}")

    # Also copy overall
    overall = args.src / "metrics_overall.csv"
    if overall.exists():
        shutil.copy2(overall, args.out / overall.name)
        print(f"  copied  {overall.name}")

    print(f"\n  Baseline saved to {args.out}/  ({len(csv_files)} files)")
    print("  Commit artifacts/baseline/ to lock in this reference point.")


if __name__ == "__main__":
    main()
