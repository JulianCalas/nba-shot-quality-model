"""Compact shot extract for the Streamlit app.

The full scored table is hundreds of thousands of rows and stays out of git.
The app only needs, for each player it can select, the court coordinates,
the zone, and the three model probabilities. Leaderboards are small and are
copied whole so every player and team on those boards is still available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shotquality.config import (
    APP_PLAYER_BOARD,
    APP_SCORED_SHOTS,
    APP_TEAM_BOARD,
    DATA_APP,
    PLAYER_BOARD,
    SCORED_SHOTS,
    TEAM_BOARD,
)

# Columns the app and the shot chart actually read. Anything else is dropped.
APP_SHOT_COLUMNS = [
    "player_name",
    "loc_x_ft",
    "loc_y_ft",
    "shot_zone_basic",
    "shot_made",
    "point_value",
    "actual_pts",
    "p_zone",
    "p_logistic",
    "p_lightgbm",
]

_FLOAT_COLUMNS = ("loc_x_ft", "loc_y_ft", "p_zone", "p_logistic", "p_lightgbm")
# Stay well under the 25 MB budget for the whole data/app directory.
_SHOT_FLOAT32_ABOVE_BYTES = 12 * 1024 * 1024
_APP_BUDGET_BYTES = 25 * 1024 * 1024


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing {missing}")


def _shots_for_offered_players(scored: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Keep every shot from every player the app can select."""
    _require_columns(scored, APP_SHOT_COLUMNS, "scored shots")
    _require_columns(players, ["player_name"], "player leaderboard")
    names = players["player_name"].dropna().astype(str)
    if names.duplicated().any():
        dupes = sorted(names.loc[names.duplicated()].unique())
        raise ValueError(f"player leaderboard has duplicate names: {dupes[:5]}")
    offered = set(names)
    slim = scored.loc[scored["player_name"].astype(str).isin(offered), APP_SHOT_COLUMNS].copy()
    present = set(slim["player_name"].astype(str).unique())
    missing = sorted(offered - present)
    if missing:
        raise ValueError(
            f"{len(missing)} players on the leaderboard have no scored shots, for example {missing[:5]}"
        )
    return slim.reset_index(drop=True)


def _maybe_narrow(shots: pd.DataFrame, path: Path) -> str:
    """Write zstd parquet, narrowing floats only if the file is still large."""
    shots.to_parquet(path, index=False, compression="zstd")
    if path.stat().st_size <= _SHOT_FLOAT32_ABOVE_BYTES:
        return "float64"
    narrow = shots.copy()
    for column in _FLOAT_COLUMNS:
        narrow[column] = narrow[column].astype("float32")
    narrow.to_parquet(path, index=False, compression="zstd")
    return "float32"


def export_app_dataset(
    scored: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
) -> dict:
    """Write ``data/app`` from the full processed tables.

    Pass the in-memory frames from the pipeline, or omit them to read the
    parquet files already written under ``data/processed``.
    """
    if scored is None:
        scored = pd.read_parquet(SCORED_SHOTS)
    if players is None:
        players = pd.read_parquet(PLAYER_BOARD)
    if teams is None:
        teams = pd.read_parquet(TEAM_BOARD)
    _require_columns(teams, ["team_name"], "team leaderboard")
    if teams["team_name"].duplicated().any():
        raise ValueError("team leaderboard has duplicate team names")

    shots = _shots_for_offered_players(scored, players)
    DATA_APP.mkdir(parents=True, exist_ok=True)
    dtype = _maybe_narrow(shots, APP_SCORED_SHOTS)
    players.to_parquet(APP_PLAYER_BOARD, index=False, compression="zstd")
    teams.to_parquet(APP_TEAM_BOARD, index=False, compression="zstd")

    written = [APP_SCORED_SHOTS, APP_PLAYER_BOARD, APP_TEAM_BOARD]
    sizes = {path.name: path.stat().st_size for path in written}
    total = sum(sizes.values())
    if total >= _APP_BUDGET_BYTES:
        raise RuntimeError(
            f"data/app is {total / 1_048_576:.1f} MB, which is not under the 25 MB budget"
        )
    return {
        "shots": int(len(shots)),
        "players": int(len(players)),
        "teams": int(len(teams)),
        "shot_dtype": dtype,
        "bytes": sizes,
        "total_bytes": total,
    }
