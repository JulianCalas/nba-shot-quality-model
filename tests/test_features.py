"""Zones, action groups, the feature list, and the season split."""

import numpy as np
import pandas as pd

from shotquality.data.clean import parse_clock, season_label_from_game_id
from shotquality.features.build import (
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    assign_action_group,
    assign_zones,
    design_matrix,
    split_shots,
)


def _zones(x, y, distance, value):
    frame = assign_zones(
        pd.Series([x]),
        pd.Series([y]),
        pd.Series([distance]),
        pd.Series([value]),
    )
    return frame.loc[0, "shot_zone_basic"], frame.loc[0, "shot_zone_range"]


def test_restricted_area_paint_and_midrange():
    assert _zones(0, 0, 0, 2)[0] == "Restricted Area"
    assert _zones(0, 10, 10, 2)[0] == "In The Paint (Non-RA)"
    assert _zones(0, 18, 18, 2)[0] == "Mid-Range"


def test_three_point_zones_and_heave():
    assert _zones(-23, 2, 23, 3)[0] == "Left Corner 3"
    assert _zones(23, 2, 23, 3)[0] == "Right Corner 3"
    assert _zones(0, 25, 25, 3)[0] == "Above the Break 3"
    basic, zone_range = _zones(0, 50, 50, 3)
    assert basic == "Backcourt"
    assert zone_range == "Back Court Shot"


def test_distance_bins():
    assert _zones(0, 6, 6, 2)[1] == "Less Than 8 ft."
    assert _zones(0, 10, 10, 2)[1] == "8-16 ft."
    assert _zones(0, 18, 18, 2)[1] == "16-24 ft."
    assert _zones(0, 25, 25, 3)[1] == "24+ ft."


def test_action_group_priority():
    labels = assign_action_group(
        pd.Series(
            [
                "Cutting Dunk Shot",
                "Tip Dunk Shot",
                "Tip Layup Shot",
                "Driving Floating Jump Shot",
                "Finger Roll Layup Shot",
                "Turnaround Hook Shot",
                "Step Back Jump shot",
                "Jump Shot",
            ]
        )
    )
    assert list(labels) == ["dunk", "dunk", "tip", "floater", "layup", "hook", "jumper", "jumper"]


def test_features_exclude_outcome_and_player_identity():
    assert set(FEATURE_COLUMNS).isdisjoint(FORBIDDEN_COLUMNS)
    assert "shot_made" not in FEATURE_COLUMNS
    assert "player_id" not in FEATURE_COLUMNS
    assert "actual_pts" not in FEATURE_COLUMNS


def test_design_matrix_is_only_the_allowed_columns():
    frame = pd.DataFrame(
        {
            "shot_distance": [10.0],
            "loc_x_ft": [1.0],
            "loc_y_ft": [10.0],
            "seconds_left": [400.0],
            "late_clock": [0],
            "period": [1],
            "shot_zone_basic": ["Mid-Range"],
            "shot_zone_range": ["8-16 ft."],
            "action_group": ["jumper"],
            "shot_type": ["2PT"],
            "shot_made": [1],
            "player_id": [99],
        }
    )
    matrix = design_matrix(frame, cat_as="string")
    assert list(matrix.columns) == FEATURE_COLUMNS
    assert "shot_made" not in matrix.columns


def test_clock_and_season_label():
    assert parse_clock("PT11M41.00S") == 701
    assert parse_clock("PT00M00.40S") == 0.4
    assert np.isnan(parse_clock("11:41"))
    assert season_label_from_game_id("0022300001") == "2023-24"
    assert season_label_from_game_id("0022500999") == "2025-26"
    assert season_label_from_game_id(22300001) == "2023-24"


def _split_frame():
    rows = []
    for season, yy, n_games in [("2023-24", 23, 6), ("2024-25", 24, 10), ("2025-26", 25, 6)]:
        for game in range(1, n_games + 1):
            for shot in range(4):
                rows.append(
                    {
                        "season": season,
                        "game_id": f"002{yy}{game:05d}",
                        "shot_made": shot % 2,
                    }
                )
    return pd.DataFrame(rows)


def test_split_keeps_games_on_one_side_and_holds_out_the_last_season():
    parts = split_shots(_split_frame())
    assert set(parts["test"]["season"]) == {"2025-26"}
    assert set(parts["train"]["season"]) <= {"2023-24", "2024-25"}
    assert set(parts["fit"]["game_id"]).isdisjoint(parts["val"]["game_id"])
    assert set(parts["train"]["game_id"]).isdisjoint(parts["test"]["game_id"])
    assert set(parts["val"]["season"]) == {"2024-25"}
    fit_last = parts["fit"].loc[parts["fit"]["season"] == "2024-25", "game_id"]
    assert fit_last.max() < parts["val"]["game_id"].min()
