"""Fit the three models and score the holdout season.

Model choice is locked on the validation slice (lowest log loss) before the
holdout season is scored. All three models are then refit on the full
training seasons and scored once on 2025-26. The refit is what gets saved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shotquality.config import RANDOM_SEED
from shotquality.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES, design_matrix, split_shots
from shotquality.models.baseline import ZoneBaseline
from shotquality.models.evaluate import binary_metrics, expected_points, reliability_table

# Columns written to the scored holdout file. The app and the SQL both read these.
SCORED_COLUMNS = [
    "shot_id",
    "game_id",
    "season",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "period",
    "seconds_left",
    "shot_distance",
    "loc_x_ft",
    "loc_y_ft",
    "shot_zone_basic",
    "shot_zone_range",
    "action_group",
    "shot_type",
    "shot_made",
    "point_value",
    "actual_pts",
    "p_zone",
    "p_logistic",
    "p_lightgbm",
    "p_make",
    "xpts",
    "primary_model",
]


def make_logistic() -> Pipeline:
    """L2 logistic regression on scaled numbers plus one-hot categories.

    ``handle_unknown='ignore'`` means a category that shows up only in the
    test season contributes nothing, instead of crashing. With our closed
    zone and action lists that should not happen; it is still the right default.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=400,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def _lgbm(n_estimators: int, learning_rate: float, min_child_samples: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        min_child_samples=min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=1,
        deterministic=True,
        verbosity=-1,
        force_row_wise=True,
    )


def _positive_proba(model, matrix: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(matrix)
    # LightGBM and sklearn both return columns ordered by class label.
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)
    return proba[:, classes.index(1)]


def _feature_importance(model: LGBMClassifier) -> list[dict]:
    booster = model.booster_
    names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1]
    total = float(gain.sum()) or 1.0
    rows = []
    for index in order:
        rows.append({"feature": names[index], "gain": float(gain[index]), "share": float(gain[index] / total)})
    return rows


def train_models(
    shots: pd.DataFrame,
    lgbm_n_estimators: int = 600,
    lgbm_learning_rate: float = 0.05,
    min_child_samples: int = 200,
) -> dict:
    """Fit zone, logistic, and LightGBM. Return models, metrics, and holdout scores.

    ``shots`` may include heaves; they are removed here. Passing the full
    clean table is the supported way to call this.
    """
    if "is_heave" not in shots.columns:
        raise KeyError("shots is missing is_heave. Build the table with data.clean.build_shot_table.")
    model_frame = shots.loc[shots["is_heave"] == 0].copy()
    parts = split_shots(model_frame)
    fit_df, val_df, train_df, test_df = parts["fit"], parts["val"], parts["train"], parts["test"]

    # --- validation fits (test season is not touched) ---
    zone_val_model = ZoneBaseline().fit(fit_df["shot_zone_basic"], fit_df["shot_made"])
    logistic_val_model = make_logistic()
    logistic_val_model.fit(design_matrix(fit_df, cat_as="string"), fit_df["shot_made"].to_numpy(dtype=int))

    lgbm_val_model = _lgbm(lgbm_n_estimators, lgbm_learning_rate, min_child_samples)
    lgbm_val_model.fit(
        design_matrix(fit_df, cat_as="category"),
        fit_df["shot_made"].to_numpy(dtype=int),
        eval_X=design_matrix(val_df, cat_as="category"),
        eval_y=val_df["shot_made"].to_numpy(dtype=int),
        eval_metric="binary_logloss",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    best_iteration = int(lgbm_val_model.best_iteration_ or lgbm_n_estimators)
    best_iteration = max(best_iteration, 1)

    val_predictions = {
        "zone": zone_val_model.predict_proba(val_df["shot_zone_basic"]),
        "logistic": _positive_proba(logistic_val_model, design_matrix(val_df, cat_as="string")),
        "lightgbm": _positive_proba(lgbm_val_model, design_matrix(val_df, cat_as="category")),
    }
    y_val = val_df["shot_made"].to_numpy(dtype=int)
    validation = {name: binary_metrics(y_val, proba) for name, proba in val_predictions.items()}
    primary = min(validation, key=lambda name: validation[name]["log_loss"])

    # --- refit on every training-season shot, then score the holdout once ---
    zone_model = ZoneBaseline().fit(train_df["shot_zone_basic"], train_df["shot_made"])
    logistic_model = make_logistic()
    logistic_model.fit(design_matrix(train_df, cat_as="string"), train_df["shot_made"].to_numpy(dtype=int))
    lgbm_model = _lgbm(best_iteration, lgbm_learning_rate, min_child_samples)
    lgbm_model.fit(
        design_matrix(train_df, cat_as="category"),
        train_df["shot_made"].to_numpy(dtype=int),
        categorical_feature=CATEGORICAL_FEATURES,
    )

    test_predictions = {
        "zone": zone_model.predict_proba(test_df["shot_zone_basic"]),
        "logistic": _positive_proba(logistic_model, design_matrix(test_df, cat_as="string")),
        "lightgbm": _positive_proba(lgbm_model, design_matrix(test_df, cat_as="category")),
    }
    y_test = test_df["shot_made"].to_numpy(dtype=int)
    test_metrics = {name: binary_metrics(y_test, proba) for name, proba in test_predictions.items()}

    scored = test_df.loc[:, [column for column in SCORED_COLUMNS if column in test_df.columns]].copy()
    scored["p_zone"] = test_predictions["zone"]
    scored["p_logistic"] = test_predictions["logistic"]
    scored["p_lightgbm"] = test_predictions["lightgbm"]
    scored["primary_model"] = primary
    scored["p_make"] = scored[f"p_{primary}"].to_numpy()
    scored["xpts"] = expected_points(scored["p_make"], scored["point_value"])
    scored = scored.loc[:, SCORED_COLUMNS].reset_index(drop=True)

    return {
        "models": {"zone": zone_model, "logistic": logistic_model, "lightgbm": lgbm_model},
        "primary": primary,
        "validation": validation,
        "test": test_metrics,
        "reliability": {name: reliability_table(y_test, proba) for name, proba in test_predictions.items()},
        "scored": scored,
        "importance": _feature_importance(lgbm_model),
        "best_iteration": best_iteration,
        "n_fit": int(len(fit_df)),
        "n_val": int(len(val_df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_fit_games": int(fit_df["game_id"].nunique()),
        "n_val_games": int(val_df["game_id"].nunique()),
        "n_test_games": int(test_df["game_id"].nunique()),
    }
