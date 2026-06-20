#!/usr/bin/env python3
"""
Regression gate CLI — compare current eval metrics against stored baseline.

Exit code: 0 = all checks pass, 1 = regression detected.

  python scripts/run_gate.py
  python scripts/run_gate.py --baseline artifacts/baseline --current artifacts/eval
  python scripts/run_gate.py --baseline artifacts/ci_baseline --current artifacts/ci_eval -v
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gates.gate_runner import run_gate
from gates.thresholds import GateThresholds


def main() -> None:
    ap = argparse.ArgumentParser(description="PerceptorGuard regression gate")
    ap.add_argument("--baseline", type=Path, default=Path("artifacts/baseline"),
                    help="Directory with baseline metrics CSVs")
    ap.add_argument("--current", type=Path, default=Path("artifacts/eval"),
                    help="Directory with current metrics CSVs")
    ap.add_argument("--thresholds", type=Path, default=Path("configs/gate_thresholds.yml"),
                    help="Gate threshold YAML (default: configs/gate_thresholds.yml)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print all checks, not just failures")
    args = ap.parse_args()

    for d, label in [(args.baseline, "baseline"), (args.current, "current")]:
        if not d.exists():
            sys.exit(f"ERROR: {label} directory not found: {d}")
        if not (d / "metrics_overall.csv").exists():
            sys.exit(f"ERROR: {d}/metrics_overall.csv missing — run eval first")

    thresholds = GateThresholds.from_yaml(args.thresholds)
    passed = run_gate(args.baseline, args.current, thresholds, verbose=args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
