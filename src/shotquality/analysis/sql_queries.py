"""The SQL that answers descriptive questions.

DuckDB speaks ordinary SQL. ``QUALIFY`` is the one non-standard clause: it
filters on a window function without a second subquery, which is the clean
way to keep each player's most-used team.
"""

from __future__ import annotations

import duckdb
import pandas as pd

# How many shots, games, and what make rate, by season?
SEASON_SQL = """
SELECT
    season,
    COUNT(*) AS shots,
    COUNT(DISTINCT game_id) AS games,
    COUNT(DISTINCT player_id) AS players,
    AVG(shot_made) AS fg_pct,
    AVG(CASE WHEN shot_type = '3PT' THEN 1.0 ELSE 0.0 END) AS three_rate,
    SUM(is_heave) AS heaves
FROM shots
GROUP BY season
ORDER BY season
"""

# The zone baseline, computed in SQL so it can be checked against the model.
# Heaves are out: a 2% backcourt make rate is not a zone anyone "selects."
ZONE_SQL = """
SELECT
    shot_zone_basic,
    COUNT(*) AS fga,
    AVG(shot_made) AS fg_pct,
    AVG(shot_distance) AS avg_distance_ft,
    AVG(point_value) AS avg_point_value,
    AVG(actual_pts) AS points_per_shot
FROM shots
WHERE is_heave = 0
GROUP BY shot_zone_basic
ORDER BY fga DESC
"""

ACTION_SQL = """
SELECT
    action_group,
    COUNT(*) AS fga,
    AVG(shot_made) AS fg_pct,
    AVG(shot_distance) AS avg_distance_ft
FROM shots
WHERE is_heave = 0
GROUP BY action_group
ORDER BY fg_pct DESC
"""

# Shot mix, not shot making. A team can shoot a lot of midrange and still
# make them; this query only describes the diet of shots.
TEAM_MIX_SQL = """
SELECT
    team_name,
    COUNT(*) AS fga,
    AVG(shot_made) AS fg_pct,
    AVG(CASE WHEN shot_zone_basic = 'Restricted Area' THEN 1.0 ELSE 0.0 END) AS rim_rate,
    AVG(CASE WHEN shot_zone_basic = 'Mid-Range' THEN 1.0 ELSE 0.0 END) AS midrange_rate,
    AVG(CASE WHEN shot_type = '3PT' THEN 1.0 ELSE 0.0 END) AS three_rate
FROM shots
WHERE is_heave = 0 AND season = ?
GROUP BY team_name
ORDER BY midrange_rate DESC
"""

# Holdout-season counting stats. Shrinkage is applied in Python afterward
# because the prior strength depends on shot-level variance, which is awkward
# SQL and easy to get wrong there.
PLAYER_AGG_SQL = """
WITH team_counts AS (
    SELECT player_id, team_name, COUNT(*) AS n
    FROM predictions
    GROUP BY player_id, team_name
),
primary_team AS (
    SELECT player_id, team_name
    FROM team_counts
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY player_id
        ORDER BY n DESC, team_name
    ) = 1
),
agg AS (
    SELECT
        player_id,
        MAX(player_name) AS player_name,
        COUNT(*) AS fga,
        SUM(shot_made) AS fgm,
        AVG(shot_made) AS fg_pct,
        AVG(p_make) AS xfg,
        SUM(actual_pts) AS points,
        SUM(xpts) AS xpoints,
        AVG(xpts) AS xpts_per_shot,
        AVG(actual_pts) AS points_per_shot,
        AVG(actual_pts - xpts) AS poe_per_shot,
        COUNT(DISTINCT team_id) AS n_teams
    FROM predictions
    GROUP BY player_id
)
SELECT
    agg.*,
    primary_team.team_name,
    agg.points - agg.xpoints AS poe_total
FROM agg
LEFT JOIN primary_team USING (player_id)
ORDER BY agg.fga DESC
"""

TEAM_AGG_SQL = """
SELECT
    team_id,
    team_name,
    COUNT(*) AS fga,
    AVG(shot_made) AS fg_pct,
    AVG(p_make) AS xfg,
    AVG(xpts) AS xpts_per_shot,
    AVG(actual_pts) AS points_per_shot,
    SUM(actual_pts - xpts) AS poe_total,
    AVG(actual_pts - xpts) AS poe_per_shot
FROM predictions
GROUP BY team_id, team_name
ORDER BY poe_per_shot DESC
"""


def season_summary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(SEASON_SQL).df()


def zone_profile(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(ZONE_SQL).df()


def action_profile(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(ACTION_SQL).df()


def team_shot_mix(connection: duckdb.DuckDBPyConnection, season: str) -> pd.DataFrame:
    return connection.execute(TEAM_MIX_SQL, [season]).df()


def player_aggregate(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(PLAYER_AGG_SQL).df()


def team_aggregate(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(TEAM_AGG_SQL).df()


def descriptive_bundle(connection: duckdb.DuckDBPyConnection, mix_season: str) -> dict[str, pd.DataFrame]:
    return {
        "seasons": season_summary(connection),
        "zones": zone_profile(connection),
        "actions": action_profile(connection),
        "team_mix": team_shot_mix(connection, mix_season),
    }
