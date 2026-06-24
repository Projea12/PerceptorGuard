"""Custom exceptions for PerceptorGuard input validation.

Hierarchy:
    EvalInputError          ← base; CLI catches this for clean output
      FileMismatchError     ← GT vs predictions image set mismatch
      FileEmptyError        ← empty GT or predictions file
      DuplicateIDError      ← duplicate image IDs in GT file
      MissingImageError     ← images referenced but not found on disk
"""
from __future__ import annotations


class EvalInputError(Exception):
    """Base class for all input validation failures.

    Raised before any metric is computed so no corrupt numbers are produced.
    The CLI catches this and prints a clean error without a Python traceback.
    """


class FileMismatchError(EvalInputError):
    """GT and predictions do not cover the same set of images."""


class FileEmptyError(EvalInputError):
    """GT or predictions file contains no usable data."""


class DuplicateIDError(EvalInputError):
    """Duplicate image IDs detected in the GT annotations file."""


class MissingImageError(EvalInputError):
    """Images referenced by GT or predictions are not found on disk."""
