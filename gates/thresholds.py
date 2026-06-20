"""Gate threshold configuration — loaded from configs/gate_thresholds.yml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "gate_thresholds.yml"


@dataclass
class GateThresholds:
    recall_slack: float = 0.02
    precision_slack: float = 0.05
    map_slack: float = 0.005
    fp_slack_frac: float = 0.30
    hard_floor_map: float = -1.0
    hard_floor_recall: float = -1.0

    @classmethod
    def from_yaml(cls, path: Path = _DEFAULT_CONFIG) -> "GateThresholds":
        if not _HAS_YAML:
            # pyyaml not installed — fall back to simple line parser
            cfg: dict = {}
            for line in Path(path).read_text().splitlines():
                line = line.split("#")[0].strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        cfg[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
            return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def floor(self, baseline: float, slack: float) -> float:
        return baseline - slack
