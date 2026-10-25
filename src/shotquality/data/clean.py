"""Clean the common shot table: names, coordinates, zones, heaves.

Nothing in this module uses whether the shot went in, except to keep that
column as the label. Zone and action labels are functions of where the shot
was and how it was described.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from shotquality.config import HEAVE_DISTANCE_FT, LATE_CLOCK_SECONDS
from shotquality.features.build import assign_action_group, assign_zones

_CLOCK = re.compile(r"^PT(\d+)M(\d+(?:\.\d+)?)S$")

# A full NBA regular season is 30 teams x 82 games / 2 = 1,230 games.
FULL_SEASON_GAMES = 1230


def parse_clock(value: object) -> float:
    """Parse an NBA clock like ``PT11M41.00S`` into seconds remaining.

    The feed uses an ISO-8601 duration, not ``11:41``. Unrecognized values
    come back as NaN and are dropped later, rather than crashing a 600k-row load.
    """
    match = _CLOCK.match(str(value).strip())
    if match is None:
        return float("nan")
    return int(match.group(1)) * 60 + float(match.group(2))


def season_label_from_game_id(game_id: object) -> str:
    """``0022300001`` -> ``2023-24``.

    Regular-season game ids are ``002`` + the starting year (``23``) + the
    game number. The label is the season a fan would say out loud.
    """
    gid = str(game_id).replace(".0", "").zfill(10)
    start = int(gid[3:5])
    end = (start + 1) % 100
    return f"20{start:02d}-{end:02d}"


def _lookup_names(player_ids: pd.Series, raw_names: pd.Series) -> pd.Series:
    """Fill full names from the local nba_api player list.

    The mirror only stores a last name (``Turner``). The id is the stable key.
    ``nba_api.stats.static.players`` is a file shipped with the package, not a
    network call. Anyone missing from that file keeps the raw name plus the id
    so two players who share a last name cannot be merged by accident.
    """
    from nba_api.stats.static import players

    lookup = {int(player["id"]): player["full_name"] for player in players.get_players()}
    names = player_ids.map(lookup)
    missing = names.isna()
    names.loc[missing] = raw_names.loc[missing].astype(str) + " (" + player_ids.loc[missing].astype(str) + ")"
    return names


def _lookup_teams(team_ids: pd.Series, tricodes: pd.Series) -> pd.Series:
    from nba_api.stats.static import teams

    lookup = {int(team["id"]): team["full_name"] for team in teams.get_teams()}
    names = team_ids.map(lookup)
    missing = names.isna()
    fallback = tricodes.loc[missing].astype(str)
    fallback = fallback.where(fallback.str.len() > 0, "Unknown")
    names.loc[missing] = fallback
    return names


def build_shot_table(common: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the modeling table and a small data-quality report.

    Heaves stay in the table with ``is_heave = 1`` so they can be counted.
    The model and the leaderboards drop them. They are not shot selection in
    any ordinary sense: a 70-foot shot at the buzzer is a football play.
    """
    required = {
        "game_id",
        "season",
        "period",
        "seconds_left",
        "team_id",
        "team_tricode",
        "player_id",
        "player_name_raw",
        "action_raw",
        "shot_made",
        "shot_value",
        "shot_distance",
        "loc_x",
        "loc_y",
        "data_source",
    }
    missing = required - set(common.columns)
    if missing:
        raise ValueError(f"Common shot frame is missing columns: {sorted(missing)}")

    frame = common.copy()
    starting_rows = int(len(frame))
    frame = frame.loc[frame["shot_value"].isin([2, 3])].copy()
    dropped_value = starting_rows - int(len(frame))

    # Coordinates in the feed are tenths of a foot, basket at (0, 0).
    frame["loc_x_ft"] = frame["loc_x"].astype(float) / 10.0
    frame["loc_y_ft"] = frame["loc_y"].astype(float) / 10.0
    frame["shot_distance"] = frame["shot_distance"].astype(float)
    frame["seconds_left"] = frame["seconds_left"].astype(float)

    before_na = int(len(frame))
    frame = frame.dropna(subset=["seconds_left", "shot_distance", "loc_x_ft", "loc_y_ft", "player_id"])
    dropped_na = before_na - int(len(frame))

    dedupe_key = [
        "game_id",
        "period",
        "seconds_left",
        "player_id",
        "loc_x",
        "loc_y",
        "shot_distance",
        "shot_made",
        "action_raw",
    ]
    before_dupes = int(len(frame))
    frame = frame.drop_duplicates(dedupe_key, keep="first")
    dropped_dupes = before_dupes - int(len(frame))

    zones = assign_zones(
        frame["loc_x_ft"],
        frame["loc_y_ft"],
        frame["shot_distance"],
        frame["shot_value"],
    )
    frame["shot_zone_basic"] = zones["shot_zone_basic"].to_numpy()
    frame["shot_zone_range"] = zones["shot_zone_range"].to_numpy()
    frame["action_group"] = assign_action_group(frame["action_raw"]).to_numpy()
    frame["shot_type"] = np.where(frame["shot_value"].to_numpy() == 3, "3PT", "2PT")
    frame["point_value"] = frame["shot_value"].astype(int)
    frame["shot_made"] = frame["shot_made"].astype(int)
    # actual_pts is the outcome, in points. It is not a model feature.
    frame["actual_pts"] = frame["shot_made"] * frame["point_value"]
    frame["is_heave"] = (frame["shot_distance"] >= HEAVE_DISTANCE_FT).astype(int)
    frame["late_clock"] = (frame["seconds_left"] <= LATE_CLOCK_SECONDS).astype(int)
    frame["period"] = frame["period"].astype(int)
    frame["player_name"] = _lookup_names(frame["player_id"], frame["player_name_raw"])
    frame["team_name"] = _lookup_teams(frame["team_id"], frame["team_tricode"])

    frame = frame.sort_values(
        ["season", "game_id", "period", "seconds_left", "player_id"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    frame.insert(0, "shot_id", np.arange(1, len(frame) + 1))

    keep = [
        "shot_id",
        "game_id",
        "season",
        "period",
        "seconds_left",
        "late_clock",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "action_raw",
        "action_group",
        "shot_type",
        "shot_value",
        "point_value",
        "shot_made",
        "actual_pts",
        "shot_distance",
        "loc_x",
        "loc_y",
        "loc_x_ft",
        "loc_y_ft",
        "shot_zone_basic",
        "shot_zone_range",
        "is_heave",
        "data_source",
    ]
    frame = frame.loc[:, keep]

    season_rows = []
    for season, part in frame.groupby("season", sort=True):
        season_rows.append(
            {
                "season": season,
                "shots": int(len(part)),
                "games": int(part["game_id"].nunique()),
                "players": int(part["player_id"].nunique()),
                "teams": int(part["team_id"].nunique()),
                "fg_pct": float(part["shot_made"].mean()),
                "heaves": int(part["is_heave"].sum()),
                "full_season_games": int(part["game_id"].nunique()) == FULL_SEASON_GAMES,
            }
        )

    report = {
        "rows_in": starting_rows,
        "dropped_bad_shot_value": int(dropped_value),
        "dropped_missing_location_or_clock": int(dropped_na),
        "dropped_duplicates": int(dropped_dupes),
        "rows_out": int(len(frame)),
        "heaves": int(frame["is_heave"].sum()),
        "data_source": sorted(frame["data_source"].unique().tolist()),
        "seasons": season_rows,
        "unmatched_player_names": int(frame["player_name"].str.contains(r"\(\d+\)$", regex=True).sum()),
    }
    return frame, report
