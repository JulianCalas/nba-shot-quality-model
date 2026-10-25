"""Shrink noisy player (or team) averages toward the league mean.

A player who is +15 points per 100 shots on 40 attempts has not revealed a
new law of basketball. Most of that number is variance. Empirical Bayes
shrinkage pulls the average toward the league mean, harder when the sample
is small and harder when shot-to-shot variance is large relative to how
much players truly differ.

The model, in one line: a player's observed mean is their true mean plus
noise. True means scatter around the league with variance ``tau^2``. One
shot's noise has variance ``sigma^2``. The prior strength

    k = sigma^2 / tau^2

is in units of shots. A player with ``n = k`` attempts is weighted halfway
between their own average and the league. This is the normal-normal
shrinkage formula, estimated with a method of moments rather than a prior
someone picked because it looked strict.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def estimate_shrinkage(values: np.ndarray | pd.Series, groups: np.ndarray | pd.Series, min_n: int = 30) -> dict:
    """Estimate ``sigma^2``, ``tau^2``, ``k``, and the grand mean.

    ``values`` is one number per shot (a point residual, or expected points).
    ``groups`` is the player or team id on that shot.
    """
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "group": np.asarray(groups)})
    if frame.empty:
        raise ValueError("Cannot estimate shrinkage on an empty series")
    grand = float(frame["value"].mean())
    stats = frame.groupby("group")["value"].agg(mean="mean", n="count", var="var")
    usable = stats.loc[stats["n"] >= 2]
    dof = float((usable["n"] - 1).sum())
    if dof <= 0:
        sigma2 = float(frame["value"].var(ddof=1)) if len(frame) > 1 else 1.0
    else:
        sigma2 = float(((usable["n"] - 1) * usable["var"]).sum() / dof)
    sigma2 = max(sigma2, 1e-8)

    qualified = stats.loc[stats["n"] >= min_n]
    if qualified.empty:
        # Not enough groups to see real differences. Shrink everything hard.
        return {"sigma2": sigma2, "tau2": 0.0, "k": math.inf, "prior_mean": grand, "n_groups": 0}

    # For a mean built from n shots, E[(mean - grand)^2] ≈ tau^2 + sigma^2 / n.
    deviation = (qualified["mean"] - grand) ** 2
    tau2 = float(deviation.mean() - (sigma2 / qualified["n"]).mean())
    tau2 = max(0.0, tau2)
    k = math.inf if tau2 <= 1e-12 else float(sigma2 / tau2)
    return {
        "sigma2": sigma2,
        "tau2": tau2,
        "k": k,
        "prior_mean": grand,
        "n_groups": int(len(qualified)),
    }


def apply_shrinkage(
    observed_mean: np.ndarray | pd.Series,
    n: np.ndarray | pd.Series,
    prior_mean: float,
    k: float,
) -> np.ndarray:
    """Pull ``observed_mean`` toward ``prior_mean``. Weight is ``n / (n + k)``."""
    observed = np.asarray(observed_mean, dtype=float)
    samples = np.asarray(n, dtype=float)
    if np.any(samples < 0):
        raise ValueError("Sample size n cannot be negative")
    if math.isinf(k):
        return np.full(observed.shape, float(prior_mean))
    if k < 0:
        raise ValueError("Prior strength k cannot be negative")
    weight = samples / (samples + k)
    return weight * observed + (1.0 - weight) * float(prior_mean)


def shrinkage_weight(n: np.ndarray | pd.Series | float, k: float) -> np.ndarray:
    """How much of the player's own average survives. 0 = fully shrunk, 1 = raw."""
    samples = np.asarray(n, dtype=float)
    if math.isinf(k):
        return np.zeros_like(samples, dtype=float)
    return samples / (samples + k)
