"""End-to-end run: raw shots -> database, models, leaderboards, figures.

The notebook calls the same functions. This module is the one-command version.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from shotquality.config import (
    DATABASE_PATH,
    FIGURES,
    METRICS_PATH,
    MIN_FGA,
    MODELS,
    PLAYER_BOARD,
    RAW_SHOTS,
    SCORED_SHOTS,
    TEAM_BOARD,
    TEST_SEASON,
)
from shotquality.data.clean import build_shot_table
from shotquality.data.database import build_database
from shotquality.data.pull import load_common_frame
from shotquality.analysis.leaderboards import build_leaderboards
from shotquality.analysis.sql_queries import descriptive_bundle
from shotquality.models.train import train_models
from shotquality.viz.charts import (
    plot_calibration,
    plot_feature_importance,
    plot_model_comparison,
    plot_player_chart,
    plot_player_making,
    plot_selection_vs_making,
    plot_shot_density,
    plot_team_making,
    plot_value_by_distance,
    plot_xfg_heatmap,
    plot_xpts_heatmap,
    plot_zone_rates,
)

logger = logging.getLogger(__name__)


def _json_ready(value):
    """Convert numpy scalars so metrics.json is plain JSON."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_ready(value.item())
        except ValueError:
            pass
    return value


def _board_records(frame: pd.DataFrame, sort_col: str, n: int = 8, ascending: bool = False) -> list[dict]:
    columns = [
        column
        for column in [
            "player_name",
            "team_name",
            "fga",
            "fg_pct",
            "xfg",
            "xpts_per_shot",
            "selection_per_100_shrunk",
            "selection_per_100",
            "poe_per_100",
            "poe_per_100_shrunk",
            "poe_total",
            "poe_se_per_100",
            "n_teams",
        ]
        if column in frame.columns
    ]
    view = frame.sort_values(sort_col, ascending=ascending).head(n)
    records = json.loads(view.loc[:, columns].to_json(orient="records"))
    return records


def run_pipeline(source: str = "auto") -> dict:
    """Fit everything and write the files the notebook, app, and README use."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Loading shots (%s)", source)
    common = load_common_frame(source=source)
    shots, quality = build_shot_table(common)
    logger.info(
        "Clean table: %s shots, %s heaves, %s duplicate rows removed",
        quality["rows_out"],
        quality["heaves"],
        quality["dropped_duplicates"],
    )

    RAW_SHOTS.parent.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(RAW_SHOTS, index=False)

    logger.info("Training models")
    trained = train_models(shots)
    scored = trained["scored"]
    scored.to_parquet(SCORED_SHOTS, index=False)
    logger.info(
        "Primary model on validation log loss: %s. Test log loss zone %.4f, logistic %.4f, lightgbm %.4f",
        trained["primary"],
        trained["test"]["zone"]["log_loss"],
        trained["test"]["logistic"]["log_loss"],
        trained["test"]["lightgbm"]["log_loss"],
    )

    connection = build_database(shots, scored, DATABASE_PATH)
    descriptive = descriptive_bundle(connection, TEST_SEASON)
    boards = build_leaderboards(connection, scored, min_fga=MIN_FGA)
    boards["players"].to_parquet(PLAYER_BOARD, index=False)
    boards["teams"].to_parquet(TEAM_BOARD, index=False)
    connection.close()

    MODELS.mkdir(parents=True, exist_ok=True)
    zone_model = trained["models"]["zone"]
    (MODELS / "zone_baseline.json").write_text(
        json.dumps(
            {"global_rate": zone_model.global_rate_, "zone_rates": zone_model.zone_rates_},
            indent=2,
        )
    )
    joblib.dump(trained["models"]["logistic"], MODELS / "logistic.joblib")
    joblib.dump(trained["models"]["lightgbm"], MODELS / "lightgbm.joblib")

    _write_figures(shots, scored, boards, trained)

    qualified = boards["players"].loc[boards["players"]["qualified"]]
    example = qualified.sort_values("fga", ascending=False).iloc[0]
    metrics = {
        "data_quality": quality,
        "primary_model": trained["primary"],
        "selection_rule": "Lowest log loss on the 2024-25 validation games, before the 2025-26 test season was scored.",
        "best_iteration": trained["best_iteration"],
        "n_fit": trained["n_fit"],
        "n_val": trained["n_val"],
        "n_train": trained["n_train"],
        "n_test": trained["n_test"],
        "n_fit_games": trained["n_fit_games"],
        "n_val_games": trained["n_val_games"],
        "n_test_games": trained["n_test_games"],
        "validation": trained["validation"],
        "test": trained["test"],
        "reliability": trained["reliability"],
        "feature_importance": trained["importance"],
        "shrinkage_making": boards["making_shrinkage"],
        "shrinkage_selection": boards["selection_shrinkage"],
        "sigma_residual": boards["sigma_residual"],
        "league_test": boards["league"],
        "zones": json.loads(descriptive["zones"].to_json(orient="records")),
        "actions": json.loads(descriptive["actions"].to_json(orient="records")),
        "top_makers": _board_records(qualified, "poe_per_100_shrunk", ascending=False),
        "bottom_makers": _board_records(qualified, "poe_per_100_shrunk", ascending=True),
        "top_selection": _board_records(qualified, "selection_per_100_shrunk", ascending=False),
        "bottom_selection": _board_records(qualified, "selection_per_100_shrunk", ascending=True),
        "top_teams": _board_records(boards["teams"], "poe_per_100", ascending=False),
        "bottom_teams": _board_records(boards["teams"], "poe_per_100", ascending=True),
        "best_team_selection": _board_records(boards["teams"], "selection_per_100", ascending=False, n=5),
        "worst_team_selection": _board_records(boards["teams"], "selection_per_100", ascending=True, n=5),
        "example_player": json.loads(example.to_json()),
    }
    METRICS_PATH.write_text(json.dumps(_json_ready(metrics), indent=2))
    logger.info("Wrote %s", METRICS_PATH)
    plt.close("all")
    return metrics


def _write_figures(shots: pd.DataFrame, scored: pd.DataFrame, boards: dict, trained: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_zone_rates(shots, FIGURES / "zone_make_rates.png")
    plot_shot_density(shots, FIGURES / "shot_density.png")
    plot_xfg_heatmap(scored, FIGURES / "xfg_heatmap.png")
    plot_xpts_heatmap(scored, FIGURES / "xpts_heatmap.png")
    plot_calibration(scored, FIGURES / "calibration.png")
    plot_model_comparison(trained["test"], FIGURES / "model_comparison.png")
    plot_value_by_distance(scored, FIGURES / "value_by_distance.png")
    plot_feature_importance(trained["importance"], FIGURES / "feature_importance.png")
    plot_player_making(boards["players"], FIGURES / "player_shot_making.png")
    plot_team_making(boards["teams"], FIGURES / "team_shot_making.png")
    plot_selection_vs_making(boards["players"], FIGURES / "selection_vs_making.png")

    qualified = boards["players"].loc[boards["players"]["qualified"]]
    example = qualified.sort_values("fga", ascending=False).iloc[0]
    player_shots = scored.loc[scored["player_id"] == example["player_id"]]
    subtitle = (
        f"{int(example['fga'])} FGA  ·  FG% {example['fg_pct']:.1%}  ·  xFG% {example['xfg']:.1%}"
        f"  ·  {example['poe_per_100_shrunk']:+.1f} pts / 100 shrunk"
    )
    plot_player_chart(
        player_shots,
        FIGURES / "example_player_chart.png",
        title=f"{example['player_name']}  ·  {example['season'] if 'season' in example.index else TEST_SEASON}",
        subtitle=subtitle,
    )
    plt.close("all")
