"""Resolve user dataset class names to model class names.

Usage flow:
    1. load_or_create(user_classes, model_classes, config_path)
       - If config_path exists: load it, warn about any newly unmapped classes.
       - If not: auto-suggest via fuzzy matching, write the file, warn user to review.
    2. class_map.apply_to_gts(gts_by_filename)
       - Remaps every GroundTruth.class_name to the model name.
       - GTs whose class has no mapping (mapped to null) are dropped with a warning.

The persisted YAML is the source of truth. Edit it directly to correct mistakes.
"""
from __future__ import annotations

import difflib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from scenarios.schemas import GroundTruth


# Auto-suggest accepts a fuzzy match only when similarity exceeds this ratio.
_FUZZY_THRESHOLD = 0.7


@dataclass
class ClassMap:
    """Resolved mapping from user class names to model class names."""

    mapping: dict[str, Optional[str]]  # user_class -> model_class | None

    def resolve(self, user_class: str) -> Optional[str]:
        return self.mapping.get(user_class)

    def apply_to_gts(
        self,
        gts_by_filename: dict[str, list[GroundTruth]],
    ) -> dict[str, list[GroundTruth]]:
        """Return a new dict with every GroundTruth remapped to model class names.

        GTs whose user class maps to None are dropped; a single warning is emitted
        listing all dropped class names so the user knows what was excluded.
        """
        dropped_classes: set[str] = set()
        result: dict[str, list[GroundTruth]] = {}

        for fname, gts in gts_by_filename.items():
            remapped: list[GroundTruth] = []
            for gt in gts:
                model_class = self.mapping.get(gt.class_name)
                if model_class is None:
                    dropped_classes.add(gt.class_name)
                    continue
                remapped.append(gt.model_copy(update={"class_name": model_class}))
            if remapped:
                result[fname] = remapped

        if dropped_classes:
            names = ", ".join(sorted(dropped_classes))
            warnings.warn(
                f"Dropped GTs for unmapped classes: {names}. "
                "Edit configs/class_map.yml to add a mapping for these classes.",
                stacklevel=2,
            )

        return result


def load_or_create(
    user_classes: set[str],
    model_classes: set[str],
    config_path: Path,
) -> ClassMap:
    """Load an existing class map or auto-suggest one and write it to disk.

    Args:
        user_classes: Class names found in the user's GT annotations.
        model_classes: Class names the model can output (e.g. COCO 80 classes).
        config_path: Where to read/write the YAML mapping file.

    Returns:
        ClassMap ready to apply to GTs.
    """
    config_path = Path(config_path)

    if config_path.exists():
        return _load(user_classes, config_path)

    mapping = _auto_suggest(user_classes, model_classes)
    _write(mapping, config_path)

    unresolved = [k for k, v in mapping.items() if v is None]
    if unresolved:
        warnings.warn(
            f"Could not auto-map {len(unresolved)} class(es): "
            f"{', '.join(sorted(unresolved))}. "
            f"Edit {config_path} to add mappings for these classes.",
            stacklevel=2,
        )

    suggested = {k: v for k, v in mapping.items() if v is not None}
    if suggested:
        lines = "\n".join(f"  {k!r} → {v!r}" for k, v in sorted(suggested.items()))
        warnings.warn(
            f"Auto-suggested class mappings written to {config_path}:\n{lines}\n"
            "Review and edit the file if any mapping is incorrect.",
            stacklevel=2,
        )

    return ClassMap(mapping=mapping)


def _load(user_classes: set[str], config_path: Path) -> ClassMap:
    """Load mapping from YAML; warn about user classes missing from the file."""
    raw = yaml.safe_load(config_path.read_text()) or {}
    saved: dict[str, Optional[str]] = {
        str(k): (str(v) if v is not None else None)
        for k, v in (raw.get("mappings") or {}).items()
    }

    missing = user_classes - set(saved.keys())
    if missing:
        warnings.warn(
            f"{len(missing)} class(es) in your data have no entry in {config_path}: "
            f"{', '.join(sorted(missing))}. "
            "Add them to the mappings section; they are excluded until then.",
            stacklevel=2,
        )
        for cls in missing:
            saved[cls] = None

    return ClassMap(mapping=saved)


def _auto_suggest(
    user_classes: set[str],
    model_classes: set[str],
) -> dict[str, Optional[str]]:
    """Produce a best-effort mapping using exact then fuzzy matching."""
    model_lower: dict[str, str] = {m.lower(): m for m in model_classes}
    mapping: dict[str, Optional[str]] = {}

    for user_cls in sorted(user_classes):
        # 1. Exact match (case-insensitive)
        if user_cls.lower() in model_lower:
            mapping[user_cls] = model_lower[user_cls.lower()]
            continue

        # 2. Fuzzy match
        candidates = difflib.get_close_matches(
            user_cls.lower(),
            model_lower.keys(),
            n=1,
            cutoff=_FUZZY_THRESHOLD,
        )
        if candidates:
            mapping[user_cls] = model_lower[candidates[0]]
        else:
            mapping[user_cls] = None  # explicit null — user must resolve

    return mapping


def _write(mapping: dict[str, Optional[str]], config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# PerceptorGuard class mapping\n"
        "# Maps your dataset class names to the names your model outputs.\n"
        "# null means the class has no model equivalent and will be excluded.\n"
        "# Edit this file if any auto-suggested mapping is wrong.\n"
    )

    body = yaml.dump(
        {"mappings": {k: mapping[k] for k in sorted(mapping)}},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    )

    config_path.write_text(header + body)
