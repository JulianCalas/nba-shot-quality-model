"""SQL aggregates, the shot-chart normalizer, and the plain-language profile."""

import pandas as pd

from shotquality.analysis.profile import profile_sentence
from shotquality.analysis.sql_queries import player_aggregate, season_summary, zone_profile
from shotquality.data.database import build_database
from shotquality.data.pull import shotchart_detail_to_common


def test_profile_sentence_four_quadrants_and_the_tie():
    assert "higher-value" in profile_sentence(5, 5) and "makes more" in profile_sentence(5, 5)
    assert "good looks" in profile_sentence(5, -5)
    assert "tougher than average" in profile_sentence(-5, 5)
    assert "lower-value" in profile_sentence(-5, -5) and "below expectation" in profile_sentence(-5, -5)
    assert profile_sentence(0.2, -0.2) == "About average in both shot selection and shot making."


def test_sql_season_zone_and_player_counts(tmp_path):
    shots = pd.DataFrame(
        {
            "season": ["2023-24", "2023-24", "2025-26"],
            "game_id": ["g1", "g1", "g2"],
            "player_id": [1, 2, 1],
            "shot_made": [1, 0, 1],
            "shot_type": ["2PT", "3PT", "2PT"],
            "shot_zone_basic": ["Restricted Area", "Mid-Range", "Restricted Area"],
            "point_value": [2, 3, 2],
            "actual_pts": [2, 0, 2],
            "shot_distance": [2.0, 16.0, 3.0],
            "is_heave": [0, 0, 1],
        }
    )
    predictions = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "player_name": ["A", "A", "B"],
            "team_id": [10, 10, 11],
            "team_name": ["Home", "Home", "Away"],
            "shot_made": [1, 0, 1],
            "actual_pts": [2.0, 0.0, 3.0],
            "p_make": [0.5, 0.4, 0.6],
            "xpts": [1.0, 0.8, 1.8],
        }
    )
    connection = build_database(shots, predictions, tmp_path / "tiny.duckdb")
    seasons = season_summary(connection)
    assert set(seasons["season"]) == {"2023-24", "2025-26"}
    assert int(seasons.loc[seasons["season"] == "2023-24", "shots"].iloc[0]) == 2
    zones = zone_profile(connection)
    # The heave is excluded from the zone profile.
    assert int(zones["fga"].sum()) == 2
    players = player_aggregate(connection)
    player_a = players.loc[players["player_id"] == 1].iloc[0]
    assert int(player_a["fga"]) == 2
    assert player_a["team_name"] == "Home"
    assert abs(player_a["poe_total"] - ((2 - 1) + (0 - 0.8))) < 1e-9
    connection.close()


def test_shotchart_detail_normalizer_does_not_keep_the_outcome_as_a_feature_name():
    raw = pd.DataFrame(
        {
            "GAME_ID": ["0022300001"],
            "TEAM_ID": [1610612738],
            "TEAM_NAME": ["Boston Celtics"],
            "PLAYER_ID": [1628369],
            "PLAYER_NAME": ["Jayson Tatum"],
            "PERIOD": [4],
            "MINUTES_REMAINING": [0],
            "SECONDS_REMAINING": [12],
            "ACTION_TYPE": ["Step Back Jump shot"],
            "SHOT_TYPE": ["3PT Field Goal"],
            "SHOT_DISTANCE": [26],
            "LOC_X": [10],
            "LOC_Y": [250],
            "SHOT_MADE_FLAG": [1],
            "EVENT_TYPE": ["Made Shot"],
        }
    )
    common = shotchart_detail_to_common(raw)
    assert common.loc[0, "season"] == "2023-24"
    assert common.loc[0, "seconds_left"] == 12
    assert common.loc[0, "shot_value"] == 3
    assert common.loc[0, "shot_made"] == 1
    assert common.loc[0, "action_raw"] == "Step Back Jump shot"
    assert "EVENT_TYPE" not in common.columns
    assert "SHOT_MADE_FLAG" not in common.columns
