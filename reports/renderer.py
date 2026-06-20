"""
Jinja2-based report renderer — produces HTML and/or Markdown output.

Globals injected into every template:
  pct(v)                  → "12.3%" or "n/a"
  bar(v)                  → inline HTML recall bar
  rc_class(v)             → CSS class for recall colouring
  delta_vs_baseline(v, b) → coloured delta string
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gates.comparator import RegressionResult

_TMPL_DIR = Path(__file__).parent / "templates"


def _pct(v) -> str:
    try:
        f = float(v)
        if math.isnan(f):
            return "n/a"
        return f"{f:.1%}"
    except (TypeError, ValueError):
        return "n/a"


def _rc_class(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "rc-zero"
    if math.isnan(f) or f == 0.0:
        return "rc-zero"
    if f < 0.2:
        return "rc-low"
    if f < 0.6:
        return "rc-mid"
    return "rc-good"


def _bar_html(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = 0.0
    if math.isnan(f):
        f = 0.0
    pct = max(0.0, min(1.0, f))
    if pct == 0.0:
        cls = "bar-zero"
        width = "2px"
    elif pct < 0.3:
        cls = "bar-low"
        width = f"{pct * 100:.0f}%"
    elif pct < 0.7:
        cls = "bar-mid"
        width = f"{pct * 100:.0f}%"
    else:
        cls = "bar-good"
        width = f"{pct * 100:.0f}%"
    return (f'<div class="bar-wrap">'
            f'<div class="bar-bg"><div class="bar-fill {cls}" style="width:{width}"></div></div>'
            f'</div>')


def _delta_str(v, baseline) -> str:
    try:
        fv, fb = float(v), float(baseline)
    except (TypeError, ValueError):
        return ""
    if math.isnan(fv) or math.isnan(fb):
        return ""
    d = fv - fb
    if abs(d) < 1e-6:
        return '<span class="delta-neu">—</span>'
    cls = "delta-pos" if d > 0 else "delta-neg"
    return f'<span class="{cls}">{d:+.1%}</span>'


def render(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    overall: dict,
    model_name: str,
    dataset_dir: Path | None = None,
    gate_results: "list[RegressionResult] | None" = None,
    failure_df: pd.DataFrame | None = None,
    out_dir: Path = Path("artifacts/report"),
    formats: tuple[str, ...] = ("html", "md"),
    iou_threshold: float = 0.5,
    op_conf: float = 0.25,
) -> dict[str, Path]:
    from jinja2 import Environment, FileSystemLoader

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build template context ────────────────────────────────────────────────
    n_scenes = int(df["scene_id"].nunique())
    n_gt = int(df[df["match_type"].isin(["tp", "fn"])].shape[0])
    n_classes = len(tables.get("class", pd.DataFrame()))

    gate_passed = None
    gate_failures: list = []
    if gate_results is not None:
        gate_failures = [r for r in gate_results if not r.passed]
        gate_passed = len(gate_failures) == 0

    # Failure modes from triage
    failure_modes: list[tuple[str, int, float]] = []
    if failure_df is not None and "failure_mode" in failure_df.columns:
        fail_only = failure_df[failure_df["failure_mode"] != "true_positive"]
        total_f = len(fail_only)
        for mode, cnt in fail_only["failure_mode"].value_counts().items():
            failure_modes.append((str(mode), int(cnt), cnt / total_f if total_f else 0.0))

    # Baseline recall for profile delta
    profile_table = tables.get("profile")
    baseline_recall = 0.0
    if profile_table is not None and "baseline" in profile_table.index:
        try:
            baseline_recall = float(profile_table.loc["baseline", "recall"])
        except Exception:
            pass

    # Annotated failure images
    annotated_scenes = []
    if dataset_dir is not None and dataset_dir.exists():
        from reports.annotator import select_failure_scenes
        annotated_scenes = select_failure_scenes(df, df, dataset_dir, top_n=4)

    ctx = dict(
        model_name=model_name,
        date=datetime.date.today().isoformat(),
        n_scenes=n_scenes,
        n_gt=n_gt,
        n_classes=n_classes,
        iou_threshold=iou_threshold,
        op_conf=op_conf,
        overall=overall,
        tier_table=tables.get("gt_tier"),
        profile_table=profile_table,
        class_table=tables.get("class"),
        tables={
            "distance_table": tables.get("distance_bin"),
            "lighting_table": tables.get("lighting_bin"),
            "clutter_table":  tables.get("clutter_bin"),
        },
        gate_results=gate_results,
        gate_passed=gate_passed,
        gate_failures=gate_failures,
        failure_modes=failure_modes,
        baseline_recall=baseline_recall,
        annotated_scenes=annotated_scenes,
        # template helpers
        pct=_pct,
        bar=_bar_html,
        rc_class=_rc_class,
        delta_vs_baseline=_delta_str,
    )

    env = Environment(loader=FileSystemLoader(str(_TMPL_DIR)), autoescape=False)

    outputs: dict[str, Path] = {}

    if "html" in formats:
        tmpl = env.get_template("report.html.j2")
        out_path = out_dir / "report.html"
        out_path.write_text(tmpl.render(**ctx), encoding="utf-8")
        outputs["html"] = out_path

    if "md" in formats:
        tmpl = env.get_template("report.md.j2")
        out_path = out_dir / "report.md"
        out_path.write_text(tmpl.render(**ctx), encoding="utf-8")
        outputs["md"] = out_path

    return outputs
