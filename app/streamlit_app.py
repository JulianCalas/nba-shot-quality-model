"""Interactive shot-quality check for one player.

From the repository root:

    streamlit run app/streamlit_app.py --server.port 8741

The app reads the holdout-season scores written by ``scripts/run_pipeline.py``
when those files are present. On a fresh clone it reads the compact extract in
``data/app`` instead. It does not refit the model and it does not download
data at startup. Color on the chart is expected FG%, not make/miss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotquality.analysis.profile import profile_sentence  # noqa: E402
from shotquality.config import (  # noqa: E402
    APP_PLAYER_BOARD,
    APP_SCORED_SHOTS,
    METRICS_PATH,
    PLAYER_BOARD,
    PROFILE_BAND,
    SCORED_SHOTS,
)
from shotquality.viz.charts import plot_player_chart  # noqa: E402

MODEL_COLUMNS = {
    "Gradient boosting": "p_lightgbm",
    "Logistic regression": "p_logistic",
    "Zone average": "p_zone",
}


def artifact_source() -> str | None:
    """Prefer the full pipeline output, then the committed compact extract."""
    if SCORED_SHOTS.is_file() and PLAYER_BOARD.is_file() and METRICS_PATH.is_file():
        return "processed"
    if APP_SCORED_SHOTS.is_file() and APP_PLAYER_BOARD.is_file() and METRICS_PATH.is_file():
        return "app"
    return None


@st.cache_data(show_spinner=False)
def load_artifacts(source: str):
    """Read local parquet only. ``source`` is ``processed`` or ``app``."""
    if source == "processed":
        scored_path, player_path = SCORED_SHOTS, PLAYER_BOARD
    elif source == "app":
        scored_path, player_path = APP_SCORED_SHOTS, APP_PLAYER_BOARD
    else:
        raise ValueError(f"unknown artifact source: {source}")
    scored = pd.read_parquet(scored_path)
    players = pd.read_parquet(player_path)
    metrics = json.loads(METRICS_PATH.read_text())
    return scored, players, metrics


def _missing_message() -> None:
    st.error(
        "No shot table found. Expected data/processed from "
        "`python scripts/run_pipeline.py`, or the committed extract in data/app."
    )


def main() -> None:
    st.set_page_config(page_title="NBA shot quality", layout="wide")
    st.title("NBA shot quality")
    st.caption(
        "Given a shot's location, type, and clock, how often should it go in, and how many points is it worth? "
        "The chart is the 2025-26 regular season, scored by a model that never saw that season."
    )

    source = artifact_source()
    if source is None:
        _missing_message()
        return

    scored, players, metrics = load_artifacts(source)
    primary = metrics["primary_model"]
    # Default the selector to whichever model won on the validation season.
    default_label = next(label for label, column in MODEL_COLUMNS.items() if column == f"p_{primary}")

    names = sorted(players["player_name"].unique())
    example = metrics.get("example_player", {}).get("player_name")
    default_index = names.index(example) if example in names else 0

    with st.sidebar:
        st.header("Player")
        player_name = st.selectbox("Shooter", names, index=default_index)
        model_label = st.selectbox("Model", list(MODEL_COLUMNS), index=list(MODEL_COLUMNS).index(default_label))
        st.markdown(
            "Expected FG% is the model's probability before the shot. "
            "Points over expected are actual points minus that probability times 2 or 3."
        )
        st.markdown(
            f"Leaderboard shrinkage uses **{default_label.lower()}**, chosen for lowest validation log loss. "
            "A player needs 300 field-goal attempts to be called qualified."
        )

    row = players.loc[players["player_name"] == player_name].iloc[0]
    shots = scored.loc[scored["player_name"] == player_name].copy()
    p_col = MODEL_COLUMNS[model_label]
    shots["p_selected"] = shots[p_col]
    shots["xpts_selected"] = shots["p_selected"] * shots["point_value"]
    shots["poe"] = shots["actual_pts"] - shots["xpts_selected"]

    fga = int(len(shots))
    fg = float(shots["shot_made"].mean()) if fga else float("nan")
    xfg = float(shots["p_selected"].mean()) if fga else float("nan")
    xpts = float(shots["xpts_selected"].mean()) if fga else float("nan")
    poe_per_100 = float(shots["poe"].mean() * 100) if fga else float("nan")
    league_xpts = float(metrics["league_test"]["xpts_per_shot"])
    selection_per_100 = (xpts - league_xpts) * 100 if fga else float("nan")

    if not bool(row["qualified"]):
        st.warning(
            f"{player_name} has {fga} attempts in the holdout season. "
            "Under 300, the average moves around a lot. Read the direction, not the rank."
        )

    left, right = st.columns([1.15, 0.85])
    with left:
        subtitle = (
            f"{row['team_name']}  ·  {fga} FGA  ·  FG% {fg:.1%}  ·  xFG% {xfg:.1%}  ·  "
            f"{poe_per_100:+.1f} pts per 100 vs this model"
        )
        fig = plot_player_chart(
            shots,
            path=None,
            p_col="p_selected",
            title=f"{player_name}  ·  2025-26",
            subtitle=subtitle,
        )
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("FG%", f"{fg:.1%}")
        c2.metric("xFG%", f"{xfg:.1%}")
        c3.metric("xPTS / shot", f"{xpts:.2f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Selection / 100", f"{selection_per_100:+.1f}")
        c5.metric("Making / 100", f"{poe_per_100:+.1f}")
        if model_label == default_label:
            c6.metric("Making / 100, shrunk", f"{row['poe_per_100_shrunk']:+.1f}")
        else:
            c6.metric("Attempts", f"{fga}")
        # The sentence uses the shrunk rates on the leaderboard model so a
        # 300-shot blip is not narrated as a skill. Other models show the raw gap.
        if model_label == default_label:
            sentence_selection = float(row["selection_per_100_shrunk"])
            sentence_making = float(row["poe_per_100_shrunk"])
        else:
            sentence_selection = selection_per_100
            sentence_making = poe_per_100
        st.markdown(profile_sentence(sentence_selection, sentence_making, band=PROFILE_BAND))
        st.caption(
            "Selection is expected points per 100 shots versus the league. "
            "Making is actual points minus expected points, per 100 shots. "
            "About average means inside ±2 points per 100."
        )

        zone = (
            shots.groupby("shot_zone_basic", as_index=False)
            .agg(
                FGA=("shot_made", "size"),
                **{
                    "FG%": ("shot_made", "mean"),
                    "xFG%": ("p_selected", "mean"),
                    "PTS/shot": ("actual_pts", "mean"),
                    "xPTS": ("xpts_selected", "mean"),
                },
            )
            .sort_values("FGA", ascending=False)
        )
        zone["POE / 100"] = (zone["PTS/shot"] - zone["xPTS"]) * 100
        st.dataframe(
            zone,
            hide_index=True,
            width="stretch",
            column_config={
                "shot_zone_basic": st.column_config.TextColumn("Zone"),
                "FG%": st.column_config.NumberColumn(format="%.1%"),
                "xFG%": st.column_config.NumberColumn(format="%.1%"),
                "PTS/shot": st.column_config.NumberColumn(format="%.2f"),
                "xPTS": st.column_config.NumberColumn(format="%.2f"),
                "POE / 100": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

    st.caption(
        "No defender distance, contest, or shot clock in the public feed. "
        "A make the model did not expect can be skill, a mismatch, or one season of noise. "
        f"Holdout shots: {metrics['league_test']['shots']:,}."
    )


main()
