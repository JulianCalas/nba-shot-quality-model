"""Shot features, the column list the model is allowed to see, and the split.

Leakage rules, stated here so they are hard to "clean up" later:

- Do not feed the model ``shot_made``, ``actual_pts``, the description text,
  or the raw "Made Shot" / "Missed Shot" field.
- Do not feed it ``player_id`` or ``team_id``. A shot-quality model that knows
  who shot is partly a player model, and "points over expected" collapses
  toward zero because the expectation already memorized the shooter.
- Action text is allowed. "Dunk" is known before the ball reaches the rim.
  It is grouped into a few families so a step-back does not get its own
  personal make rate. See ``assign_action_group``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from shotquality.config import TEST_SEASON, TRAIN_SEASONS, VAL_GAME_FRACTION

# Court geometry in feet, basket at (0, 0), y increasing toward half court.
# These match the NBA stats coordinate frame (the same one shot charts use):
# baseline at y = -4.75, free-throw line at y = 14.25, paint 16 feet wide.
RESTRICTED_RADIUS_FT = 4.0
PAINT_HALF_WIDTH_FT = 8.0
BASELINE_Y_FT = -4.75
FREE_THROW_Y_FT = 14.25
CORNER_THREE_X_FT = 22.0
THREE_POINT_RADIUS_FT = 23.75
# Where the straight corner line meets the 23.75-foot arc:
#   y = sqrt(23.75^2 - 22^2) ≈ 8.95 ft.
CORNER_THREE_Y_FT = math.sqrt(THREE_POINT_RADIUS_FT**2 - CORNER_THREE_X_FT**2)

ZONE_BASIC_LEVELS = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
    "Backcourt",
]
ZONE_RANGE_LEVELS = [
    "Less Than 8 ft.",
    "8-16 ft.",
    "16-24 ft.",
    "24+ ft.",
    "Back Court Shot",
]
ACTION_LEVELS = ["dunk", "layup", "floater", "tip", "hook", "jumper", "other"]
SHOT_TYPE_LEVELS = ["2PT", "3PT"]

# The model sees these columns and nothing else.
NUMERIC_FEATURES = [
    "shot_distance",
    "loc_x_ft",
    "loc_y_ft",
    "seconds_left",
    "late_clock",
    "period",
]
CATEGORICAL_FEATURES = [
    "shot_zone_basic",
    "shot_zone_range",
    "action_group",
    "shot_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Outcome columns, and identity columns that would let a tree memorize a player.
FORBIDDEN_COLUMNS = [
    "shot_made",
    "shot_result",
    "actual_pts",
    "description",
    "score_home",
    "score_away",
    "p_make",
    "p_zone",
    "p_logistic",
    "p_lightgbm",
    "xpts",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "action_raw",
]


def assign_zones(
    loc_x_ft: pd.Series,
    loc_y_ft: pd.Series,
    shot_distance: pd.Series,
    shot_value: pd.Series,
) -> pd.DataFrame:
    """Rebuild the NBA's basic zones from coordinates.

    The mirror does not include the official zone labels. Two-point vs three
    is taken from ``shot_value`` (the value of the attempt, not whether it
    went in). On the 2023-24 regular season that flag agrees with the
    geometric three-point line on 99.99% of shots. Corner vs above-the-break,
    paint vs midrange, and the restricted area are then pure geometry.

    A shot inside 4 feet is restricted area even if it is also inside the
    paint. Heaves (40+ feet) are labeled Backcourt and later dropped from
    the model, so they do not become their own "zone average."
    """
    x = loc_x_ft.to_numpy(dtype=float)
    y = loc_y_ft.to_numpy(dtype=float)
    distance = shot_distance.to_numpy(dtype=float)
    value = shot_value.to_numpy()
    radius = np.hypot(x, y)

    basic = np.full(len(x), "Mid-Range", dtype=object)
    basic[radius <= RESTRICTED_RADIUS_FT] = "Restricted Area"
    in_paint = (
        (np.abs(x) <= PAINT_HALF_WIDTH_FT)
        & (y >= BASELINE_Y_FT)
        & (y <= FREE_THROW_Y_FT)
        & (radius > RESTRICTED_RADIUS_FT)
    )
    basic[in_paint] = "In The Paint (Non-RA)"

    is_three = value == 3
    corner = is_three & (y <= CORNER_THREE_Y_FT) & (np.abs(x) >= CORNER_THREE_X_FT)
    basic[is_three & ~corner] = "Above the Break 3"
    basic[corner & (x < 0)] = "Left Corner 3"
    basic[corner & (x >= 0)] = "Right Corner 3"
    basic[distance >= 40] = "Backcourt"

    zone_range = np.full(len(x), "24+ ft.", dtype=object)
    zone_range[distance < 8] = "Less Than 8 ft."
    zone_range[(distance >= 8) & (distance < 16)] = "8-16 ft."
    zone_range[(distance >= 16) & (distance < 24)] = "16-24 ft."
    zone_range[distance >= 24] = "24+ ft."
    zone_range[distance >= 40] = "Back Court Shot"

    return pd.DataFrame(
        {"shot_zone_basic": basic, "shot_zone_range": zone_range},
        index=loc_x_ft.index,
    )


def assign_action_group(actions: pd.Series) -> pd.Series:
    """Collapse ~50 action strings into families a model can learn.

    Order is the priority. "Tip Dunk Shot" is a dunk, not a tip.
    "Driving Floating Jump Shot" is a floater, not a generic jumper.
    Step-backs, fadeaways, and pull-ups all stay in ``jumper``. Giving each
    move its own bucket would move shot-making skill into the "quality" side
    of the ledger: the model would expect a specialist's pet move to go in,
    and points over expected would stop measuring that specialty.
    """
    text = actions.fillna("").str.lower()
    conditions = [
        text.str.contains("dunk", regex=False).to_numpy(),
        text.str.contains("hook", regex=False).to_numpy(),
        text.str.contains("tip", regex=False).to_numpy(),
        text.str.contains("float", regex=False).to_numpy(),
        (text.str.contains("layup", regex=False) | text.str.contains("finger roll", regex=False)).to_numpy(),
        (text.str.contains("jump", regex=False) | text.str.contains("shot", regex=False)).to_numpy(),
    ]
    choices = ["dunk", "hook", "tip", "floater", "layup", "jumper"]
    grouped = np.select(conditions, choices, default="other")
    return pd.Series(grouped, index=actions.index)


def design_matrix(frame: pd.DataFrame, cat_as: str = "category") -> pd.DataFrame:
    """The only columns a model is allowed to fit on.

    ``cat_as='category'`` is what LightGBM wants. Logistic regression gets
    plain strings so one-hot encoding does not depend on pandas category
    internals.
    """
    leaked = set(FEATURE_COLUMNS) & set(FORBIDDEN_COLUMNS)
    if leaked:
        raise RuntimeError(f"Feature list includes forbidden columns: {sorted(leaked)}")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Shot table is missing features: {missing}")

    matrix = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        matrix[column] = matrix[column].astype(float)
    if cat_as == "category":
        levels = {
            "shot_zone_basic": ZONE_BASIC_LEVELS,
            "shot_zone_range": ZONE_RANGE_LEVELS,
            "action_group": ACTION_LEVELS,
            "shot_type": SHOT_TYPE_LEVELS,
        }
        for column, allowed in levels.items():
            matrix[column] = pd.Categorical(matrix[column], categories=allowed)
    elif cat_as == "string":
        for column in CATEGORICAL_FEATURES:
            matrix[column] = matrix[column].astype(str)
    else:
        raise ValueError("cat_as must be 'category' or 'string'")
    return matrix


def split_shots(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by season, then hold out late game ids for early stopping.

    Test is the entire 2025-26 regular season. Those games did not exist when
    the training seasons were played, so a random row split is the wrong tool:
    it would put two shots from the same game, the same night, on both sides.

    Validation is the last 20% of game ids in 2024-25. Game ids follow the
    original schedule. A postponed game keeps its id, so this is not a perfect
    calendar split. It is only used to decide when the boosted model should
    stop adding trees. The reported test numbers never use this slice's labels
    until the final refit, which still cannot see 2025-26.
    """
    unknown = sorted(set(frame["season"].unique()) - set(TRAIN_SEASONS) - {TEST_SEASON})
    if unknown:
        raise ValueError(f"Unexpected seasons in the modeling frame: {unknown}")

    train = frame.loc[frame["season"].isin(TRAIN_SEASONS)].copy()
    test = frame.loc[frame["season"] == TEST_SEASON].copy()
    if train.empty or test.empty:
        raise ValueError("Train or test season has no shots. Check the filters.")

    last_season = TRAIN_SEASONS[-1]
    last = train.loc[train["season"] == last_season]
    games = np.sort(last["game_id"].unique())
    if len(games) < 5:
        raise ValueError(f"Need at least 5 games in {last_season} to hold out a validation slice.")
    cut = int(round(len(games) * (1.0 - VAL_GAME_FRACTION)))
    cut = min(max(cut, 1), len(games) - 1)
    val_games = set(games[cut:])
    val_mask = train["game_id"].isin(val_games)
    fit = train.loc[~val_mask].copy()
    val = train.loc[val_mask].copy()

    overlap = set(train["game_id"]) & set(test["game_id"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} game ids appear in both train and test.")
    if set(fit["game_id"]) & set(val["game_id"]):
        raise RuntimeError("A game is in both the fit and validation slices.")

    return {"fit": fit, "val": val, "train": train, "test": test}
