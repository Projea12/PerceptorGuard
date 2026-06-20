"""Formatted console report and CSV persistence."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_W = 74


def _bar(recall: float, width: int = 20) -> str:
    if recall != recall:   # nan
        return " " * width
    filled = round(recall * width)
    return "█" * filled + "░" * (width - filled)


def _pct(v: float) -> str:
    return f"{v:.1%}" if v == v else "  n/a "


def _delta(v: float, baseline: float) -> str:
    if v != v or baseline != baseline or baseline == 0:
        return ""
    d = (v - baseline) / baseline
    return f"  ({d:+.0%} vs baseline)"


def print_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    overall: dict,
    model_name: str,
    iou_threshold: float,
    op_conf: float = 0.25,
) -> None:
    n_scenes = df["scene_id"].nunique()
    n_gt = int(df[df["match_type"].isin(["tp", "fn"])].shape[0])

    print("═" * _W)
    print(f"  PerceptorGuard Eval Report")
    print(f"  model={model_name}  |  scenes={n_scenes}  |  GT objects={n_gt}")
    print(f"  IoU threshold={iou_threshold}  |  operating-point conf≥{op_conf}")
    print("═" * _W)

    o = overall
    print(f"\n  OVERALL")
    print(f"    TP={o['tp']}  FP={o['fp']}  FN={o['fn']}")
    print(f"    Precision={_pct(o['precision'])}  Recall={_pct(o['recall'])}"
          f"  F1={_pct(o['f1'])}  mAP@0.5={_pct(o['map'])}")

    # ── Tier ──────────────────────────────────────────────────────────────────
    if "gt_tier" in tables:
        t = tables["gt_tier"]
        print(f"\n  BY TIER  (key split: easy=COCO vocab, hard=off-vocabulary)")
        print(f"  {'tier':<8}  {'Recall':>7}  {'Precision':>10}  {'F1':>6}  "
              f"{'TP':>5}  {'FP':>5}  {'FN':>5}")
        for tier, row in t.iterrows():
            print(f"  {str(tier):<8}  {_pct(row.recall):>7}  {_pct(row.precision):>10}"
                  f"  {_pct(row.f1):>6}  {row.tp:>5}  {row.fp:>5}  {row.fn:>5}")

    # ── Profile ────────────────────────────────────────────────────────────────
    if "profile" in tables:
        t = tables["profile"]
        baseline_recall = t.loc["baseline", "recall"] if "baseline" in t.index else float("nan")
        print(f"\n  BY PROFILE")
        print(f"  {'profile':<12}  {'Recall':>7}  {'F1':>6}  bar (recall)"
              f"  {'TP':>5}  {'FP':>5}  {'FN':>5}  Δ vs baseline")
        for prof, row in t.iterrows():
            d = _delta(row.recall, baseline_recall) if str(prof) != "baseline" else ""
            print(f"  {str(prof):<12}  {_pct(row.recall):>7}  {_pct(row.f1):>6}  "
                  f"[{_bar(row.recall)}]  {row.tp:>5}  {row.fp:>5}  {row.fn:>5}{d}")

    # ── Distance ───────────────────────────────────────────────────────────────
    if "distance_bin" in tables:
        t = tables["distance_bin"]
        print(f"\n  BY CAMERA DISTANCE")
        print(f"  {'bin':<20}  {'Recall':>7}  {'F1':>6}  bar (recall)  "
              f"{'TP':>5}  {'FP':>5}  {'FN':>5}")
        for val, row in t.iterrows():
            print(f"  {str(val):<20}  {_pct(row.recall):>7}  {_pct(row.f1):>6}  "
                  f"[{_bar(row.recall)}]  {row.tp:>5}  {row.fp:>5}  {row.fn:>5}")

    # ── Lighting ───────────────────────────────────────────────────────────────
    if "lighting_bin" in tables:
        t = tables["lighting_bin"]
        print(f"\n  BY LIGHTING (ambient coefficient)")
        print(f"  {'bin':<24}  {'Recall':>7}  {'F1':>6}  bar (recall)  "
              f"{'TP':>5}  {'FP':>5}  {'FN':>5}")
        for val, row in t.iterrows():
            print(f"  {str(val):<24}  {_pct(row.recall):>7}  {_pct(row.f1):>6}  "
                  f"[{_bar(row.recall)}]  {row.tp:>5}  {row.fp:>5}  {row.fn:>5}")

    # ── Clutter ────────────────────────────────────────────────────────────────
    if "clutter_bin" in tables:
        t = tables["clutter_bin"]
        print(f"\n  BY CLUTTER DENSITY")
        print(f"  {'bin':<18}  {'Recall':>7}  {'F1':>6}  bar (recall)  "
              f"{'TP':>5}  {'FP':>5}  {'FN':>5}")
        for val, row in t.iterrows():
            print(f"  {str(val):<18}  {_pct(row.recall):>7}  {_pct(row.f1):>6}  "
                  f"[{_bar(row.recall)}]  {row.tp:>5}  {row.fp:>5}  {row.fn:>5}")

    # ── Per-class ──────────────────────────────────────────────────────────────
    if "class" in tables:
        t = tables["class"]
        print(f"\n  BY CLASS")
        print(f"  {'class':<14}  {'tier':<6}  {'Recall':>7}  {'AP@0.5':>7}  "
              f"{'TP':>5}  {'FP':>5}  {'FN':>5}  bar (recall)")
        for cls, row in t.iterrows():
            tier_tag = str(row.get("tier", "?"))
            print(f"  {str(cls):<14}  {tier_tag:<6}  {_pct(row.recall):>7}  "
                  f"{_pct(row.ap):>7}  {row.tp:>5}  {row.fp:>5}  {row.fn:>5}  "
                  f"[{_bar(row.recall)}]")

    print("\n" + "═" * _W)


def save_tables(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    overall: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "matches.csv", index=False)
    for name, tbl in tables.items():
        tbl.to_csv(out_dir / f"metrics_{name}.csv")
    # Overall as single-row CSV
    pd.DataFrame([{k: v for k, v in overall.items() if k != "class_ap"}]).to_csv(
        out_dir / "metrics_overall.csv", index=False
    )
    print(f"\n  Saved to {out_dir}/")
    print(f"    matches.csv  ({len(df)} rows)")
    for name in tables:
        print(f"    metrics_{name}.csv")
