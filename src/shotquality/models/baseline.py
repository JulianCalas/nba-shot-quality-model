"""Zone-average baseline.

Every shot is given the make rate of its zone in the training data. If a zone
never appeared in training, the shot gets the overall training make rate.

This is the model a careful analyst could build in a spreadsheet. Logistic
regression and the boosted model have to beat it on the holdout season to
justify their complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ZoneBaseline:
    def __init__(self) -> None:
        self.zone_rates_: dict[str, float] = {}
        self.global_rate_: float = 0.5

    def fit(self, zones: pd.Series, made: pd.Series) -> "ZoneBaseline":
        if len(zones) != len(made):
            raise ValueError("zones and made must be the same length")
        if len(zones) == 0:
            raise ValueError("Cannot fit a zone baseline on zero shots")
        frame = pd.DataFrame({"zone": zones.to_numpy(), "made": made.to_numpy(dtype=float)})
        self.global_rate_ = float(frame["made"].mean())
        self.zone_rates_ = frame.groupby("zone")["made"].mean().astype(float).to_dict()
        return self

    def predict_proba(self, zones: pd.Series) -> np.ndarray:
        """Return P(make) for each shot. Unknown zones use the global rate."""
        if not self.zone_rates_:
            raise RuntimeError("ZoneBaseline.predict_proba called before fit")
        rates = zones.map(self.zone_rates_).astype(float)
        return rates.fillna(self.global_rate_).to_numpy(dtype=float)
