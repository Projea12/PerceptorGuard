"""
PerceptorGuard CLI — entry point for the pip-installed tool.

Commands:
  perceptorguard eval    — run evaluation against a dataset
  perceptorguard report  — generate HTML + Markdown report from eval output
  perceptorguard gate    — run regression gate against stored baseline
  perceptorguard triage  — classify and cluster failures
  perceptorguard generate — generate synthetic dataset (requires [synthetic])
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the repo root is on the path whether installed or run from source
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _cmd_eval(args: argparse.Namespace) -> int:
    from scripts.run_eval import main as _main
    sys.argv = ["perceptorguard eval"] + _forward(args)
    _main()
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from scripts.generate_report import main as _main
    sys.argv = ["perceptorguard report"] + _forward(args)
    _main()
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from scripts.run_gate import main as _main
    sys.argv = ["perceptorguard gate"] + _forward(args)
    _main()
    return 0


def _cmd_triage(args: argparse.Namespace) -> int:
    from scripts.triage import main as _main
    sys.argv = ["perceptorguard triage"] + _forward(args)
    _main()
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    try:
        from scripts.generate import main as _main
    except SystemExit:
        return 1
    sys.argv = ["perceptorguard generate"] + _forward(args)
    _main()
    return 0


def _forward(args: argparse.Namespace) -> list[str]:
    """Convert parsed namespace back to argv for the delegated script."""
    forwarded = []
    for k, v in vars(args).items():
        if k == "command" or v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                forwarded.append(flag)
        else:
            forwarded += [flag, str(v)]
    return forwarded


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="perceptorguard",
        description="PerceptorGuard — perception model evaluation harness",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── eval ──────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Run evaluation against a dataset")
    p_eval.add_argument("--dataset", type=Path, default=Path("artifacts/dataset"))
    p_eval.add_argument("--out",     type=Path, default=Path("artifacts/eval"))
    p_eval.add_argument("--model",   default="yolov8n.pt")
    p_eval.add_argument("--iou",     type=float, default=0.5)
    p_eval.add_argument("--imgsz",   type=int,   default=640)
    p_eval.add_argument("--no-save", action="store_true")

    # ── report ────────────────────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Generate HTML + Markdown report")
    p_rep.add_argument("--dataset",   type=Path, default=Path("artifacts/dataset"))
    p_rep.add_argument("--eval",      type=Path, default=Path("artifacts/eval"))
    p_rep.add_argument("--baseline",  type=Path, default=None)
    p_rep.add_argument("--triage",    type=Path, default=None)
    p_rep.add_argument("--out",       type=Path, default=Path("artifacts/report"))
    p_rep.add_argument("--model",     default="yolov8n.pt")
    p_rep.add_argument("--iou",       type=float, default=0.5)
    p_rep.add_argument("--formats",   default="html,md")
    p_rep.add_argument("--tracker",   default="none",
                       choices=["none", "wandb", "mlflow"])
    p_rep.add_argument("--no-images", action="store_true")

    # ── gate ──────────────────────────────────────────────────────────────────
    p_gate = sub.add_parser("gate", help="Run regression gate against baseline")
    p_gate.add_argument("--baseline", type=Path, default=Path("artifacts/baseline"))
    p_gate.add_argument("--current",  type=Path, default=Path("artifacts/eval"))
    p_gate.add_argument("--thresholds", type=Path,
                        default=Path("configs/gate_thresholds.yml"))
    p_gate.add_argument("-v", "--verbose", action="store_true")

    # ── triage ────────────────────────────────────────────────────────────────
    p_tri = sub.add_parser("triage", help="Classify and cluster failures")
    p_tri.add_argument("--matches",  type=Path,
                       default=Path("artifacts/eval/matches.csv"))
    p_tri.add_argument("--out",      type=Path, default=Path("artifacts/triage"))
    p_tri.add_argument("--iou",      type=float, default=0.5)
    p_tri.add_argument("--clusters", type=int,   default=5)
    p_tri.add_argument("--no-cluster", action="store_true")
    p_tri.add_argument("--no-save",    action="store_true")
    p_tri.add_argument("--model",    default="")

    # ── generate (synthetic, requires [synthetic] extra) ──────────────────────
    p_gen = sub.add_parser(
        "generate",
        help="Generate synthetic dataset (requires pip install perceptorguard[synthetic])",
    )
    p_gen.add_argument("--count",  type=int,  default=100)
    p_gen.add_argument("--seed",   type=int,  default=0)
    p_gen.add_argument("--out",    type=Path, default=Path("artifacts/dataset"))
    p_gen.add_argument("--width",  type=int,  default=640)
    p_gen.add_argument("--height", type=int,  default=480)

    args = parser.parse_args()

    dispatch = {
        "eval":     _cmd_eval,
        "report":   _cmd_report,
        "gate":     _cmd_gate,
        "triage":   _cmd_triage,
        "generate": _cmd_generate,
    }

    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
