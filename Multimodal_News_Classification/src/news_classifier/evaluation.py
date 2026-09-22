"""Evaluation helpers for unconstrained and hierarchy-consistent predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def tune_per_label_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Choose one threshold per label using validation-set F1 only."""

    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have identical shapes.")
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.05)

    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for label_index in range(y_true.shape[1]):
        scores = [
            f1_score(
                y_true[:, label_index],
                probabilities[:, label_index] >= threshold,
                zero_division=0,
            )
            for threshold in grid
        ]
        thresholds[label_index] = float(grid[int(np.argmax(scores))])
    return thresholds


def decode_to_valid_patterns(
    probabilities: np.ndarray,
    valid_patterns: np.ndarray,
) -> np.ndarray:
    """Return the most likely label pattern observed in the training data.

    The score is the Bernoulli log-likelihood of every valid pattern under each
    vector of independently predicted label probabilities.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid_patterns = np.asarray(valid_patterns, dtype=np.int8)
    if probabilities.ndim != 2 or valid_patterns.ndim != 2:
        raise ValueError("probabilities and valid_patterns must be 2-D arrays.")
    if probabilities.shape[1] != valid_patterns.shape[1]:
        raise ValueError("probabilities and valid_patterns must share label width.")

    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    scores = (
        np.log(clipped)[:, None, :] * valid_patterns[None, :, :]
        + np.log1p(-clipped)[:, None, :] * (1 - valid_patterns)[None, :, :]
    ).sum(axis=2)
    return valid_patterns[np.argmax(scores, axis=1)].astype(np.int8)


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute explicit, non-ambiguous multi-label metrics."""

    return {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Return support, precision, recall, F1, and average precision per label."""

    rows = []
    for label_index in range(y_true.shape[1]):
        rows.append(
            {
                "label": f"label_{label_index}",
                "support": int(y_true[:, label_index].sum()),
                "precision": precision_score(
                    y_true[:, label_index],
                    y_pred[:, label_index],
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true[:, label_index],
                    y_pred[:, label_index],
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true[:, label_index],
                    y_pred[:, label_index],
                    zero_division=0,
                ),
                "average_precision": average_precision_score(
                    y_true[:, label_index],
                    probabilities[:, label_index],
                ),
            }
        )
    return pd.DataFrame(rows)
