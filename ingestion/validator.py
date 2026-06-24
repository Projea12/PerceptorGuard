"""Validation gate — runs before any matching or metric computation.

All 10 rules are checked here. Any hard-stop rule raises an EvalInputError
subclass with a precise message telling the user exactly what is wrong and
how to fix it. Nothing downstream ever sees invalid data.

Rule summary:
  1  GT file has at least one image                         hard stop
  2  Predictions file has at least one detection            hard stop
  3  No duplicate image IDs in GT file                      hard stop
  4  No exact duplicate prediction entries                  hard stop
  5  Every GT image ID exists in predictions                hard stop
  6  Every prediction image exists in GT                    hard stop
  7  Images in GT with no annotated boxes                   warning only
  8  GT images exist on disk          (if --images given)   hard stop
  9  Pred images exist on disk        (if --images given)   hard stop
  10 Images directory is not empty    (if --images given)   hard stop

Rule 4 (exact duplicate predictions) — hard stop.
Multiple predictions per image is normal. Exact copies of the same prediction
(same image, class, box, confidence) are not — they mean the inference script
processed the same image more than once and saved all results. This silently
inflates FP counts and collapses Precision.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.coco_gt import CocoGTDataset
from ingestion.exceptions import (
    DuplicateIDError,
    FileEmptyError,
    FileMismatchError,
    MissingImageError,
)


@dataclass
class ValidationSummary:
    """Audit trail produced by a successful validation pass."""
    gt_image_count: int
    pred_image_count: int
    matched_image_count: int
    gt_box_count: int
    images_with_no_boxes: int
    duplicate_ids_found: int
    disk_verified: bool
    images_verified_on_disk: int


def validate_eval_inputs(
    gt_ds: CocoGTDataset,
    preds_by_filename: dict,
    images_dir: Path | None = None,
) -> ValidationSummary:
    """Validate GT and predictions before any matching begins.

    Args:
        gt_ds:             Loaded CocoGTDataset.
        preds_by_filename: Loaded predictions dict (filename -> list[Detection]).
        images_dir:        Optional image directory (enables Rules 8, 9, 10).

    Returns:
        ValidationSummary for inclusion in the evaluation report audit trail.

    Raises:
        FileEmptyError:    Rules 1, 2, 10.
        DuplicateIDError:  Rule 3.
        FileMismatchError: Rules 5, 6.
        MissingImageError: Rules 8, 9.
    """
    gt_filenames: set[str] = set(gt_ds.filename_by_id.values())
    pred_filenames: set[str] = set(preds_by_filename.keys())

    # ── Rule 1: GT file must not be empty ────────────────────────────────────
    if len(gt_filenames) == 0:
        raise FileEmptyError(
            "GT annotations file contains no images.\n\n"
            "  Images found: 0\n\n"
            "  Cannot evaluate against an empty annotation file.\n"
            "  Check that you passed the correct GT file.\n\n"
            "  Stopping. No metrics produced."
        )

    # ── Rule 2: Predictions file must not be empty ───────────────────────────
    if len(pred_filenames) == 0:
        raise FileEmptyError(
            f"Predictions file contains no valid detections.\n\n"
            f"  GT images:          {len(gt_filenames)}\n"
            f"  Valid predictions:  0\n\n"
            f"  Either the predictions file is empty or all predictions were\n"
            f"  filtered out (unknown image IDs, unknown categories, bad scores).\n"
            f"  Check that your predictions file matches the GT image set.\n\n"
            f"  Stopping. No metrics produced."
        )

    # ── Rule 3: No duplicate image IDs in GT ─────────────────────────────────
    if gt_ds.duplicate_image_ids:
        dupes = sorted(gt_ds.duplicate_image_ids)
        dupes_preview = ", ".join(str(d) for d in dupes[:10])
        if len(dupes) > 10:
            dupes_preview += f" ... and {len(dupes) - 10} more"
        raise DuplicateIDError(
            f"Duplicate image IDs detected in GT annotations file.\n\n"
            f"  Duplicate IDs ({len(dupes)} total): {dupes_preview}\n\n"
            f"  Duplicate IDs cause ambiguous GT-to-prediction matching.\n"
            f"  Each image ID must appear exactly once in the 'images' array.\n\n"
            f"  How to fix:\n"
            f"    Deduplicate the 'images' array in your annotations file,\n"
            f"    keeping only the first occurrence of each image ID.\n\n"
            f"  Stopping. No metrics produced."
        )

    # ── Rule 4: No exact duplicate predictions ───────────────────────────────
    seen_fingerprints: set[tuple] = set()
    duplicate_preds: list[tuple] = []
    for fname, detections in preds_by_filename.items():
        for det in detections:
            fingerprint = (
                fname,
                det.class_id,
                round(det.box.x_min, 4),
                round(det.box.y_min, 4),
                round(det.box.x_max, 4),
                round(det.box.y_max, 4),
                round(det.confidence, 6),
            )
            if fingerprint in seen_fingerprints:
                duplicate_preds.append(fingerprint)
            else:
                seen_fingerprints.add(fingerprint)

    if duplicate_preds:
        sample = duplicate_preds[:3]
        sample_lines = "\n".join(
            f"    image={fp[0]}  class_id={fp[1]}  "
            f"box=[{fp[2]},{fp[3]},{fp[4]},{fp[5]}]  score={fp[6]}"
            for fp in sample
        )
        more = f"\n    ... and {len(duplicate_preds) - 3} more" if len(duplicate_preds) > 3 else ""
        raise DuplicateIDError(
            f"Exact duplicate predictions detected in predictions file.\n\n"
            f"  Duplicate entries ({len(duplicate_preds)} total):\n"
            f"{sample_lines}{more}\n\n"
            f"  This means your inference script processed the same image\n"
            f"  more than once and saved all results.\n"
            f"  Every duplicate becomes a False Positive, collapsing Precision.\n\n"
            f"  How to fix:\n"
            f"    Deduplicate your predictions file before running evaluation.\n"
            f"    Each unique (image, class, box, score) must appear only once.\n\n"
            f"  Stopping. No metrics produced."
        )

    # ── Rules 5 & 6: Image sets must match exactly ───────────────────────────
    gt_only = gt_filenames - pred_filenames      # GT images with no predictions
    pred_only = pred_filenames - gt_filenames    # Pred images with no GT

    if gt_only or pred_only:
        overlap = gt_filenames & pred_filenames
        lines = [
            "GT and predictions do not cover the same image set.\n",
            f"  Images in GT:              {len(gt_filenames)}",
            f"  Images in predictions:     {len(pred_filenames)}",
            f"  Images in common:          {len(overlap)}",
        ]

        if gt_only:
            sample = sorted(gt_only)[:5]
            sample_str = "\n    ".join(sample)
            if len(gt_only) > 5:
                sample_str += f"\n    ... and {len(gt_only) - 5} more"
            lines += [
                f"  Images in GT only:         {len(gt_only)}",
                f"    (these would be counted as {len(gt_only)} images worth of "
                f"False Negatives,",
                f"     collapsing Recall to near 0% — meaningless metrics)",
                f"    Sample: {sample[0]}" if sample else "",
            ]

        if pred_only:
            sample = sorted(pred_only)[:5]
            sample_str = "\n    ".join(sample)
            if len(pred_only) > 5:
                sample_str += f"\n    ... and {len(pred_only) - 5} more"
            lines += [
                f"  Images in predictions only: {len(pred_only)}",
                f"    (these would inflate False Positives — no GT to match against)",
            ]

        if len(overlap) == 0:
            lines += [
                "\n  Zero images overlap. You have likely passed completely wrong files.",
            ]

        lines += [
            "\n  How to fix — choose one:",
            "    Option 1: Run inference on ALL images in your GT file,",
            "              then re-run evaluation.",
            "    Option 2: Filter your GT annotations to only the images",
            "              you ran inference on, then re-run evaluation.",
            "\n  Stopping. No metrics produced.",
        ]
        raise FileMismatchError("\n".join(lines))

    # ── Rule 7: Images with no GT boxes — warning only ───────────────────────
    images_with_boxes = set(gt_ds.gts_by_filename.keys())
    images_without_boxes = gt_filenames - images_with_boxes
    if images_without_boxes:
        warnings.warn(
            f"{len(images_without_boxes)} image(s) in GT have no annotated boxes. "
            f"These may be intentional negative examples. "
            f"They contribute 0 GT boxes to evaluation.",
            stacklevel=2,
        )

    # ── Rules 8, 9, 10: Disk checks (only when images_dir is provided) ───────
    disk_verified = False
    images_verified = 0

    if images_dir is not None:
        images_dir = Path(images_dir)

        # Rule 10: directory must not be empty
        image_files = list(images_dir.iterdir()) if images_dir.exists() else []
        if not image_files:
            raise MissingImageError(
                f"Images directory is empty or does not exist.\n\n"
                f"  Path provided: {images_dir}\n"
                f"  Files found:   0\n\n"
                f"  Check that you passed the correct --images directory.\n\n"
                f"  Stopping. No metrics produced."
            )

        # Rule 8: GT images must exist on disk
        missing_gt = [
            fname for fname in gt_filenames
            if not (images_dir / fname).exists()
            and not (images_dir / Path(fname).name).exists()
        ]
        if missing_gt:
            sample = sorted(missing_gt)[:5]
            more = f" ... and {len(missing_gt) - 5} more" if len(missing_gt) > 5 else ""
            raise MissingImageError(
                f"{len(missing_gt)} GT image(s) not found on disk.\n\n"
                f"  Images directory: {images_dir}\n"
                f"  Missing ({len(missing_gt)} total):\n"
                + "\n".join(f"    {f}" for f in sample) + more + "\n\n"
                f"  Check that --images points to the directory containing\n"
                f"  the images referenced in your GT annotations file.\n\n"
                f"  Stopping. No metrics produced."
            )

        # Rule 9: Pred images must exist on disk
        missing_pred = [
            fname for fname in pred_filenames
            if not (images_dir / fname).exists()
            and not (images_dir / Path(fname).name).exists()
        ]
        if missing_pred:
            sample = sorted(missing_pred)[:5]
            more = f" ... and {len(missing_pred) - 5} more" if len(missing_pred) > 5 else ""
            raise MissingImageError(
                f"{len(missing_pred)} prediction image(s) not found on disk.\n\n"
                f"  Images directory: {images_dir}\n"
                f"  Missing ({len(missing_pred)} total):\n"
                + "\n".join(f"    {f}" for f in sample) + more + "\n\n"
                f"  Check that --images points to the directory containing\n"
                f"  the images your model ran inference on.\n\n"
                f"  Stopping. No metrics produced."
            )

        disk_verified = True
        images_verified = len(gt_filenames)

    # ── All rules passed — return audit summary ───────────────────────────────
    gt_box_count = sum(len(v) for v in gt_ds.gts_by_filename.values())

    return ValidationSummary(
        gt_image_count=len(gt_filenames),
        pred_image_count=len(pred_filenames),
        matched_image_count=len(gt_filenames & pred_filenames),
        gt_box_count=gt_box_count,
        images_with_no_boxes=len(images_without_boxes),
        duplicate_ids_found=0,
        disk_verified=disk_verified,
        images_verified_on_disk=images_verified,
    )


def format_audit_trail(summary: ValidationSummary, images_dir: Path | None = None) -> str:
    """Return the validation audit trail string for inclusion in reports."""
    lines = [
        "EVALUATION SCOPE — VALIDATED",
        "─" * 50,
        f"Images validated:          {summary.gt_image_count} / {summary.gt_image_count} (100% match)",
        f"GT boxes in scope:         {summary.gt_box_count}",
        f"Duplicate IDs found:       {summary.duplicate_ids_found}",
        f"Images with no GT boxes:   {summary.images_with_no_boxes}"
        + (" (negative examples — expected)" if summary.images_with_no_boxes else ""),
    ]
    if images_dir is not None:
        lines.append(f"Images on disk verified:   {summary.images_verified_on_disk} / {summary.gt_image_count}")
    else:
        lines.append(
            "Image disk verification:   SKIPPED (no --images provided)\n"
            "                           Lighting and clutter slices will not be computed."
        )
    lines += [
        "─" * 50,
        "Validation status:         PASSED — all rules confirmed",
    ]
    return "\n".join(lines)
