"""
Ranked failure-mode report: frequency, conditions, and cluster insights.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_W = 74
_TOP_N = 3


def _pct(n: int, total: int) -> str:
    return f"{n/total:.1%}" if total else "0.0%"


def print_triage(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    model_name: str = "",
) -> None:
    failures = df[df["failure_mode"] != "true_positive"]
    total_failures = len(failures)
    total_gt = int(df[df["match_type"].isin(["tp", "fn"])].shape[0])

    print("═" * _W)
    print("  PerceptorGuard  —  Failure Triage Report")
    if model_name:
        print(f"  model={model_name}")
    print(f"  total failures={total_failures}  (GT objects={total_gt})")
    print("═" * _W)

    # ── Ranked failure modes ──────────────────────────────────────────────────
    mode_counts = failures["failure_mode"].value_counts()
    print(f"\n  RANKED FAILURE MODES")
    print(f"  {'rank':<5}  {'failure_mode':<22}  {'count':>6}  {'% of failures':>13}  {'% of GT':>9}")
    for rank, (mode, cnt) in enumerate(mode_counts.items(), 1):
        print(f"  {rank:<5}  {mode:<22}  {cnt:>6}  {_pct(cnt, total_failures):>13}  "
              f"{_pct(cnt, total_gt):>9}")

    # ── Top-3 conditions per mode ─────────────────────────────────────────────
    print(f"\n  TOP-{_TOP_N} CONDITIONS PER FAILURE MODE")
    for mode in mode_counts.index[:_TOP_N]:
        grp = failures[failures["failure_mode"] == mode]
        print(f"\n  [{mode}]  n={len(grp)}")

        if "profile" in grp.columns:
            top = grp["profile"].value_counts().head(3)
            print(f"    profiles:  " + "  ".join(f"{k}({v})" for k, v in top.items()))

        if "gt_tier" in grp.columns and grp["gt_tier"].notna().any():
            top = grp["gt_tier"].value_counts().head(3)
            print(f"    tiers:     " + "  ".join(f"{k}({v})" for k, v in top.items()))

        if "distance_bin" in grp.columns:
            top = grp["distance_bin"].value_counts().head(3)
            print(f"    distance:  " + "  ".join(f"{k}({v})" for k, v in top.items()))

        if "lighting_bin" in grp.columns:
            top = grp["lighting_bin"].value_counts().head(3)
            print(f"    lighting:  " + "  ".join(f"{k}({v})" for k, v in top.items()))

        if "gt_class" in grp.columns and grp["gt_class"].notna().any():
            top = grp["gt_class"].value_counts().head(3)
            print(f"    classes:   " + "  ".join(f"{k}({v})" for k, v in top.items()))

    # ── Cluster insights ──────────────────────────────────────────────────────
    if not cluster_summary.empty:
        print(f"\n  CLUSTER INSIGHTS  ({len(cluster_summary)} clusters)")
        print(f"  {'#':<3}  {'dominant_mode':<22}  {'count':>6}  {'mode%':>6}  "
              f"{'dist':>5}  {'light':>5}  {'objs':>5}  {'profile':<12}")
        for _, row in cluster_summary.iterrows():
            print(f"  {int(row.cluster):<3}  {row.dominant_failure_mode:<22}  "
                  f"{int(row['count']):>6}  {row.mode_pct:>6.1%}  "
                  f"{row.mean_camera_distance:>5.1f}  {row.mean_ambient_light:>5.2f}  "
                  f"{row.mean_num_objects:>5.1f}  {str(row.top_profile):<12}")

    print("\n" + "═" * _W)


def save_triage(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "failures_classified.csv", index=False)
    if not cluster_summary.empty:
        cluster_summary.to_csv(out_dir / "cluster_summary.csv", index=False)
    print(f"\n  Saved to {out_dir}/")
    print(f"    failures_classified.csv  ({len(df)} rows)")
    if not cluster_summary.empty:
        print(f"    cluster_summary.csv  ({len(cluster_summary)} clusters)")
