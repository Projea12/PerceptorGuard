"""
Optional experiment tracking: W&B and MLflow.

Both integrations are guarded by try/import so the report pipeline runs
whether or not either package is installed.  Pass tracker="none" to skip.
"""
from __future__ import annotations

import datetime


def log_run(
    overall: dict,
    tables: dict,
    model_name: str,
    iou_threshold: float = 0.5,
    gate_passed: bool | None = None,
    tracker: str = "none",   # "wandb" | "mlflow" | "none"
    project: str = "perceptorguard",
    run_name: str | None = None,
) -> bool:
    """
    Log metrics to the requested tracker.  Returns True if logging succeeded.
    """
    if tracker == "wandb":
        return _log_wandb(overall, tables, model_name, iou_threshold,
                          gate_passed, project, run_name)
    if tracker == "mlflow":
        return _log_mlflow(overall, tables, model_name, iou_threshold,
                           gate_passed, project, run_name)
    return False   # tracker="none"


def _log_wandb(overall, tables, model_name, iou_threshold,
               gate_passed, project, run_name) -> bool:
    try:
        import wandb  # type: ignore
    except ImportError:
        print("  [tracker] wandb not installed — skipping (pip install wandb)")
        return False

    run = wandb.init(
        project=project,
        name=run_name or f"{model_name}_{datetime.date.today()}",
        config={
            "model": model_name,
            "iou_threshold": iou_threshold,
        },
        reinit=True,
    )

    metrics: dict[str, float] = {
        "overall/mAP": float(overall.get("map", 0) or 0),
        "overall/recall": float(overall.get("recall", 0) or 0),
        "overall/precision": float(overall.get("precision", 0) or 0),
        "overall/tp": float(overall.get("tp", 0)),
        "overall/fp": float(overall.get("fp", 0)),
        "overall/fn": float(overall.get("fn", 0)),
    }
    if gate_passed is not None:
        metrics["gate/passed"] = float(gate_passed)

    # per-class AP
    if "class" in tables:
        for cls, row in tables["class"].iterrows():
            ap = row.get("ap", 0)
            if ap == ap:  # not NaN
                metrics[f"class/{cls}/ap"] = float(ap)
            rec = row.get("recall", 0)
            if rec == rec:
                metrics[f"class/{cls}/recall"] = float(rec)

    wandb.log(metrics)
    run.finish()
    print(f"  [tracker] W&B run logged: {run.url}")
    return True


def _log_mlflow(overall, tables, model_name, iou_threshold,
                gate_passed, project, run_name) -> bool:
    try:
        import mlflow  # type: ignore
    except ImportError:
        print("  [tracker] mlflow not installed — skipping (pip install mlflow)")
        return False

    mlflow.set_experiment(project)
    with mlflow.start_run(run_name=run_name or f"{model_name}_{datetime.date.today()}"):
        mlflow.log_params({
            "model": model_name,
            "iou_threshold": iou_threshold,
        })
        mlflow.log_metrics({
            "mAP": float(overall.get("map", 0) or 0),
            "recall": float(overall.get("recall", 0) or 0),
            "precision": float(overall.get("precision", 0) or 0),
            "tp": float(overall.get("tp", 0)),
            "fp": float(overall.get("fp", 0)),
            "fn": float(overall.get("fn", 0)),
        })
        if gate_passed is not None:
            mlflow.log_metric("gate_passed", float(gate_passed))

        if "class" in tables:
            for cls, row in tables["class"].iterrows():
                ap = row.get("ap", 0)
                if ap == ap:
                    mlflow.log_metric(f"ap_{cls.replace(' ', '_')}", float(ap))

    print(f"  [tracker] MLflow run logged (experiment: {project})")
    return True
