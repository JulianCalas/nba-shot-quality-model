"""Baseline, expected points, shrinkage, and a tiny end-to-end fit."""

import numpy as np
import pandas as pd

from shotquality.models.baseline import ZoneBaseline
from shotquality.models.evaluate import actual_points, binary_metrics, expected_points
from shotquality.models.shrinkage import apply_shrinkage, estimate_shrinkage, shrinkage_weight
from shotquality.models.train import train_models


def test_zone_baseline_uses_only_the_rows_it_is_fit_on():
    zones = pd.Series(["Rim", "Rim", "Mid"])
    made = pd.Series([1, 1, 0])
    model = ZoneBaseline().fit(zones, made)
    # A test set where every rim shot misses. The baseline must stay at the training rate.
    predicted = model.predict_proba(pd.Series(["Rim", "Rim", "Nowhere"]))
    assert predicted[0] == 1.0
    assert predicted[1] == 1.0
    assert predicted[2] == model.global_rate_


def test_expected_points_uses_two_and_three():
    points = expected_points([0.4, 0.5], [3, 2])
    assert abs(points[0] - 1.2) < 1e-9
    assert abs(points[1] - 1.0) < 1e-9
    assert actual_points([1, 0], [3, 2]).tolist() == [3.0, 0.0]
    # A made three that the model called a 50% shot is +1.5 points over expected.
    residual = actual_points([1], [3]) - expected_points([0.5], [3])
    assert residual[0] == 1.5


def test_log_loss_prefers_a_sharp_correct_probability():
    y = np.array([1, 0, 1, 0])
    sharp = binary_metrics(y, np.array([0.9, 0.1, 0.8, 0.2]))
    flat = binary_metrics(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert sharp["log_loss"] < flat["log_loss"]
    assert sharp["brier"] < flat["brier"]


def test_shrinkage_weight_and_direction():
    assert shrinkage_weight(10, 100) == 10 / 110
    assert apply_shrinkage([1.0], [10], prior_mean=0.0, k=100)[0] == 10 / 110
    large = apply_shrinkage([1.0], [10_000], prior_mean=0.0, k=100)[0]
    small = apply_shrinkage([1.0], [10], prior_mean=0.0, k=100)[0]
    assert large > small
    assert abs(large - 1.0) < abs(small - 1.0)
    assert apply_shrinkage([1.0], [10], prior_mean=0.0, k=float("inf"))[0] == 0.0


def test_estimated_prior_shrinks_small_samples_more():
    rng = np.random.default_rng(0)
    values = []
    groups = []
    for player, talent in enumerate(rng.normal(0, 0.25, size=40)):
        draws = talent + rng.normal(0, 1, size=150)
        values.append(draws)
        groups.append(np.full(150, player))
    estimate = estimate_shrinkage(np.concatenate(values), np.concatenate(groups), min_n=30)
    assert estimate["sigma2"] > 0.5
    assert estimate["k"] > 1
    shrunk_small = apply_shrinkage([1.0], [5], estimate["prior_mean"], estimate["k"])[0]
    shrunk_large = apply_shrinkage([1.0], [5000], estimate["prior_mean"], estimate["k"])[0]
    assert abs(shrunk_large - 1.0) < abs(shrunk_small - 1.0)


def _synthetic_shots(n_games: int = 8, shots_per_game: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    shot_id = 1
    for season, yy in [("2023-24", 23), ("2024-25", 24), ("2025-26", 25)]:
        for game in range(1, n_games + 1):
            for _ in range(shots_per_game):
                distance = float(rng.choice([2, 4, 8, 15, 23, 26]))
                is_three = distance >= 23
                # Close shots go in more often. The model has a real signal to find.
                probability = float(np.clip(0.72 - 0.012 * distance, 0.25, 0.8))
                made = int(rng.random() < probability)
                point_value = 3 if is_three else 2
                if distance <= 4:
                    zone, zone_range, action = "Restricted Area", "Less Than 8 ft.", "layup"
                elif distance < 16:
                    zone, zone_range, action = "In The Paint (Non-RA)", "8-16 ft.", "floater"
                elif not is_three:
                    zone, zone_range, action = "Mid-Range", "16-24 ft.", "jumper"
                else:
                    zone, zone_range, action = "Above the Break 3", "24+ ft.", "jumper"
                rows.append(
                    {
                        "shot_id": shot_id,
                        "game_id": f"002{yy}{game:05d}",
                        "season": season,
                        "period": 1,
                        "seconds_left": 300.0,
                        "late_clock": 0,
                        "team_id": 1,
                        "team_name": "Test Team",
                        "player_id": int(rng.integers(1, 12)),
                        "player_name": "Test Player",
                        "action_raw": action,
                        "action_group": action,
                        "shot_type": "3PT" if is_three else "2PT",
                        "shot_value": point_value,
                        "point_value": point_value,
                        "shot_made": made,
                        "actual_pts": made * point_value,
                        "shot_distance": distance,
                        "loc_x_ft": 0.0,
                        "loc_y_ft": distance,
                        "shot_zone_basic": zone,
                        "shot_zone_range": zone_range,
                        "is_heave": 0,
                    }
                )
                shot_id += 1
    return pd.DataFrame(rows)


def test_train_models_scores_only_the_holdout_and_returns_probabilities():
    trained = train_models(_synthetic_shots(), lgbm_n_estimators=40, lgbm_learning_rate=0.1, min_child_samples=1)
    scored = trained["scored"]
    assert len(scored) == trained["n_test"]
    assert set(scored["season"]) == {"2025-26"}
    for column in ("p_zone", "p_logistic", "p_lightgbm", "p_make"):
        assert scored[column].between(0, 1).all()
    # A made three's expected points cannot exceed 3.
    assert scored["xpts"].max() <= 3
    assert trained["primary"] in {"zone", "logistic", "lightgbm"}
    assert trained["test"]["zone"]["log_loss"] > 0
