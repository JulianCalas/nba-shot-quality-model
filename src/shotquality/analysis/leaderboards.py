"""Player and team boards: selection, shot making, shrinkage, and standard errors.

Shot making is points over expected (actual points minus expected points).
Shot selection is expected points relative to the league on the same season.
Both are reported per 100 shots so a reader can compare a +4 to a +4.

Standard errors use the shot-level residual standard deviation, not a
bootstrap. For a player with n shots, SE(mean residual) = sigma / sqrt(n).
That is the right scale for "is this gap bigger than noise?"
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from shotquality.analysis.sql_queries import player_aggregate, team_aggregate
from shotquality.config import MIN_FGA
from shotquality.models.shrinkage import apply_shrinkage, estimate_shrinkage


def _per_100(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def build_leaderboards(connection, scored: pd.DataFrame, min_fga: int = MIN_FGA) -> dict:
    """SQL does the counting. Python estimates the prior and shrinks the means."""
    players = player_aggregate(connection)
    teams = team_aggregate(connection)

    residual = scored["actual_pts"].to_numpy(dtype=float) - scored["xpts"].to_numpy(dtype=float)
    making = estimate_shrinkage(residual, scored["player_id"].to_numpy(), min_n=30)
    selection = estimate_shrinkage(scored["xpts"].to_numpy(dtype=float), scored["player_id"].to_numpy(), min_n=30)

    league_xpts = float(scored["xpts"].mean())
    league_points = float(scored["actual_pts"].mean())
    sigma_resid = math.sqrt(making["sigma2"])
    sigma_xpts = math.sqrt(selection["sigma2"])

    players["poe_per_100"] = _per_100(players["poe_per_shot"])
    players["poe_per_shot_shrunk"] = apply_shrinkage(
        players["poe_per_shot"], players["fga"], making["prior_mean"], making["k"]
    )
    players["poe_per_100_shrunk"] = _per_100(players["poe_per_shot_shrunk"])
    players["xpts_per_shot_shrunk"] = apply_shrinkage(
        players["xpts_per_shot"], players["fga"], selection["prior_mean"], selection["k"]
    )
    players["selection_per_100"] = _per_100(players["xpts_per_shot"] - league_xpts)
    players["selection_per_100_shrunk"] = _per_100(players["xpts_per_shot_shrunk"] - league_xpts)
    players["poe_se_per_100"] = sigma_resid * 100.0 / np.sqrt(players["fga"].to_numpy(dtype=float))
    players["selection_se_per_100"] = sigma_xpts * 100.0 / np.sqrt(players["fga"].to_numpy(dtype=float))
    players["fg_over_xfg"] = players["fg_pct"] - players["xfg"]
    players["qualified"] = players["fga"] >= min_fga
    players = players.sort_values(["qualified", "poe_per_100_shrunk"], ascending=[False, False]).reset_index(drop=True)

    # Teams are not shrunk with the player prior. A team mean is thousands of
    # shots, and team-to-team variance is a different parameter. The standard
    # error is the honest uncertainty; shrinkage would barely move them.
    teams["poe_per_100"] = _per_100(teams["poe_per_shot"])
    teams["selection_per_100"] = _per_100(teams["xpts_per_shot"] - league_xpts)
    teams["poe_se_per_100"] = sigma_resid * 100.0 / np.sqrt(teams["fga"].to_numpy(dtype=float))
    teams = teams.sort_values("poe_per_100", ascending=False).reset_index(drop=True)

    league = {
        "shots": int(len(scored)),
        "fg_pct": float(scored["shot_made"].mean()),
        "xfg": float(scored["p_make"].mean()),
        "points_per_shot": league_points,
        "xpts_per_shot": league_xpts,
        "min_fga": int(min_fga),
        "qualified_players": int(players["qualified"].sum()),
        "players": int(len(players)),
    }
    return {
        "players": players,
        "teams": teams,
        "making_shrinkage": making,
        "selection_shrinkage": selection,
        "league": league,
        "sigma_residual": sigma_resid,
    }
