"""
Orchestrates the regression gate: load thresholds, compare metrics,
print a structured report, return pass/fail.
"""
from __future__ import annotations

from pathlib import Path

from gates.comparator import RegressionResult, compare_metrics
from gates.thresholds import GateThresholds

_W = 74


def run_gate(
    baseline_dir: Path,
    current_dir: Path,
    thresholds: GateThresholds | None = None,
    verbose: bool = False,
) -> bool:
    """
    Compare *current_dir* metrics against *baseline_dir*.
    Returns True if all checks pass (no regression).
    Prints a human-readable report to stdout.
    """
    if thresholds is None:
        thresholds = GateThresholds.from_yaml()

    results = compare_metrics(baseline_dir, current_dir, thresholds)
    failures = [r for r in results if not r.passed]
    passes   = [r for r in results if r.passed]

    print("═" * _W)
    print("  PerceptorGuard Regression Gate")
    print(f"  baseline : {baseline_dir}")
    print(f"  current  : {current_dir}")
    print("═" * _W)

    if failures:
        print(f"\n  FAILED — {len(failures)} regression(s) detected\n")
        for r in sorted(failures, key=lambda x: (x.slice_name, x.metric)):
            delta_str = f"{r.delta:+.4f}"
            if r.metric == "fp_count":
                print(f"  ✗ {r.slice_name} / {r.metric}")
                print(f"      baseline={r.baseline:.0f}  current={r.current:.0f}  "
                      f"ceil={r.floor:.0f}  delta={r.current - r.baseline:+.0f}")
            else:
                print(f"  ✗ {r.slice_name} / {r.metric}")
                print(f"      baseline={r.baseline:.4f}  current={r.current:.4f}  "
                      f"floor={r.floor:.4f}  delta={delta_str}")
    else:
        print(f"\n  PASSED — all {len(passes)} checks within threshold\n")

    if verbose:
        print(f"\n  {'─' * (_W - 2)}")
        print(f"  ALL CHECKS ({len(results)} total)\n")
        for r in results:
            sym = "✓" if r.passed else "✗"
            if r.metric == "fp_count":
                print(f"  {sym} {r.slice_name}/{r.metric}: "
                      f"{r.baseline:.0f} → {r.current:.0f} (ceil {r.floor:.0f})")
            else:
                print(f"  {sym} {r.slice_name}/{r.metric}: "
                      f"{r.baseline:.4f} → {r.current:.4f} (floor {r.floor:.4f})")

    print("\n" + "═" * _W)
    return len(failures) == 0
