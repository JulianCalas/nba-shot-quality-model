"""Paths, seasons, and the few analysis choices that should stay in one place."""

from __future__ import annotations

from pathlib import Path

# Ending years in the sportsdataverse file names.
# shots_2024.parquet is the 2023-24 season (game ids begin with 00223).
MIRROR_END_YEARS: tuple[int, ...] = (2024, 2025, 2026)
MIRROR_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/"
    "releases/download/nba_stats_shots/shots_{end_year}.parquet"
)

# Train on two full seasons, test on the next one. See features.split_shots.
TRAIN_SEASONS: tuple[str, ...] = ("2023-24", "2024-25")
TEST_SEASON = "2025-26"
# Share of game ids in the latest training season held out for early stopping.
VAL_GAME_FRACTION = 0.20

# Leaderboards hide small samples. Shrinkage does the rest of the work.
MIN_FGA = 300
# "About average" band for the plain-language player blurb, in points per 100 shots.
PROFILE_BAND = 2.0

# Shots from 40 feet and beyond are almost all end-of-clock heaves.
HEAVE_DISTANCE_FT = 40.0
# Game clock, not shot clock. The public feed does not have the shot clock.
LATE_CLOCK_SECONDS = 4.0

RANDOM_SEED = 42


def find_repo_root() -> Path:
    """Find the repository root whether the code is imported from src or an editable install."""
    file_guess = Path(__file__).resolve().parents[2]
    if (file_guess / "src" / "shotquality").exists():
        return file_guess
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src" / "shotquality").exists():
            return candidate
    return file_guess


ROOT = find_repo_root()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

RAW_SHOTS = DATA_PROCESSED / "shots.parquet"
SCORED_SHOTS = DATA_PROCESSED / "scored_shots.parquet"
PLAYER_BOARD = DATA_PROCESSED / "player_leaderboard.parquet"
TEAM_BOARD = DATA_PROCESSED / "team_leaderboard.parquet"
METRICS_PATH = DATA_PROCESSED / "metrics.json"
DATABASE_PATH = DATA_PROCESSED / "shots.duckdb"
