"""Scores for a probability, plus the translation into expected points.

Log loss and Brier score both punish confident wrong answers. Accuracy does
not: a model that says 51% on every shot can look accurate and still be
useless for expected points. These two are the numbers to quote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def clip_probability(p: np.ndarray | pd.Series) -> np.ndarray:
    """Keep log(p) finite if a model ever emits an exact 0 or 1."""
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def expected_points(p_make: np.ndarray | pd.Series, point_value: np.ndarray | pd.Series) -> np.ndarray:
    """Expected points of a shot: make probability times 2 or 3.

    A 40% three is worth 1.20 points. A 50% long two is worth 1.00. Shot
    selection uses this currency, not raw FG%, or midrange shots look fine.
    """
    return np.asarray(p_make, dtype=float) * np.asarray(point_value, dtype=float)


def actual_points(made: np.ndarray | pd.Series, point_value: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(made, dtype=float) * np.asarray(point_value, dtype=float)


def binary_metrics(y_true: np.ndarray | pd.Series, p_make: np.ndarray | pd.Series) -> dict[str, float]:
    """Log loss, Brier, AUC, and a simple calibration check (mean p vs mean y)."""
    y = np.asarray(y_true, dtype=int)
    p = clip_probability(p_make)
    if len(y) != len(p):
        raise ValueError("y_true and p_make must be the same length")
    if len(y) == 0:
        raise ValueError("Cannot score an empty prediction vector")
    metrics = {
        "n": float(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": float(p.mean()),
        "mean_actual": float(y.mean()),
        "ece": expected_calibration_error(y, p),
    }
    # AUC is undefined if the test slice is all makes or all misses.
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def expected_calibration_error(y_true: np.ndarray, p_make: np.ndarray, n_bins: int = 10) -> float:
    """Average |predicted − actual| across equal-width probability bins, weighted by bin size."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_make, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total = len(y)
    error = 0.0
    for bin_id in range(n_bins):
        mask = index == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def reliability_table(y_true: np.ndarray | pd.Series, p_make: np.ndarray | pd.Series, n_bins: int = 10) -> list[dict]:
    """One row per probability bin: what the model said, and what happened."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_make, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = index == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_left": float(edges[bin_id]),
                "bin_right": float(edges[bin_id + 1]),
                "n": count,
                "mean_predicted": float(p[mask].mean()),
                "empirical_fg": float(y[mask].mean()),
            }
        )
    return rows
