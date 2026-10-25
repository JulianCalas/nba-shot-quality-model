"""Figures for the README and the notebook. Each function saves a PNG and returns the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from shotquality.viz.court import draw_half_court
from shotquality.viz.style import CRIMSON, INK, MUTED, NAVY, TEAL, VALUE_CMAP, XFG_CMAP, apply_style

ZONE_ORDER = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
]

MODEL_COLORS = {"zone": MUTED, "logistic": NAVY, "lightgbm": TEAL}
MODEL_LABELS = {
    "zone": "Zone average",
    "logistic": "Logistic regression",
    "lightgbm": "Gradient boosting",
}


def save_figure(fig: plt.Figure, path: Path | None) -> plt.Figure:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    return fig


def plot_zone_rates(shots: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Make rate by zone, with the share of attempts on the bar. This is the baseline, drawn."""
    apply_style()
    data = shots.loc[shots["is_heave"] == 0]
    grouped = (
        data.groupby("shot_zone_basic")
        .agg(fga=("shot_made", "size"), fg=("shot_made", "mean"))
        .reindex(ZONE_ORDER)
        .dropna(subset=["fga"])
    )
    share = grouped["fga"] / grouped["fga"].sum()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.barh(grouped.index[::-1], grouped["fg"].to_numpy()[::-1], color=NAVY, height=0.72)
    for bar, fg, pct in zip(bars, grouped["fg"].to_numpy()[::-1], share.to_numpy()[::-1]):
        ax.text(
            fg + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{fg:.1%}   ·   {pct:.0%} of shots",
            va="center",
            ha="left",
            color=MUTED,
            fontsize=9,
        )
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Field-goal percentage")
    ax.set_title("Make rate by shot zone, regular season")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_shot_density(shots: pd.DataFrame, path: Path | None = None, season_label: str = "2023-24 to 2025-26") -> plt.Figure:
    apply_style()
    data = shots.loc[shots["is_heave"] == 0]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    draw_half_court(ax)
    hexbin = ax.hexbin(
        data["loc_x_ft"],
        data["loc_y_ft"],
        gridsize=36,
        cmap="YlOrBr",
        mincnt=30,
        linewidths=0.15,
        edgecolors="white",
        extent=(-25, 25, -5, 42),
        zorder=2,
    )
    colorbar = fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Shots")
    colorbar.outline.set_visible(False)
    ax.set_title(f"Where shots are taken\n{season_label}, heaves excluded")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_xfg_heatmap(scored: pd.DataFrame, path: Path | None = None, p_col: str = "p_make") -> plt.Figure:
    """Average predicted make probability of the shots that were actually taken there.

    Rim hexes are high because they include layups and dunks, not because a
    contested jumper from four feet is a 65% shot.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    draw_half_court(ax)
    hexbin = ax.hexbin(
        scored["loc_x_ft"],
        scored["loc_y_ft"],
        C=scored[p_col],
        reduce_C_function=np.mean,
        gridsize=34,
        cmap=XFG_CMAP,
        mincnt=25,
        linewidths=0.2,
        edgecolors="white",
        vmin=0.32,
        vmax=0.70,
        extent=(-25, 25, -5, 42),
        zorder=2,
    )
    colorbar = fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Expected FG%")
    colorbar.outline.set_visible(False)
    season = scored["season"].iloc[0] if len(scored) else ""
    ax.set_title(f"Expected FG% of shots from each spot\n{season} holdout · shot types that were actually taken")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_xpts_heatmap(scored: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Expected points, centered on the league average so above-average regions read as teal."""
    apply_style()
    center = float(scored["xpts"].mean())
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    draw_half_court(ax)
    norm = TwoSlopeNorm(vmin=min(0.85, center - 0.05), vcenter=center, vmax=max(1.35, center + 0.05))
    hexbin = ax.hexbin(
        scored["loc_x_ft"],
        scored["loc_y_ft"],
        C=scored["xpts"],
        reduce_C_function=np.mean,
        gridsize=34,
        cmap=VALUE_CMAP,
        norm=norm,
        mincnt=25,
        linewidths=0.2,
        edgecolors="white",
        extent=(-25, 25, -5, 42),
        zorder=2,
    )
    colorbar = fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Expected points")
    colorbar.outline.set_visible(False)
    season = scored["season"].iloc[0] if len(scored) else ""
    ax.set_title(
        f"Expected points per shot\n{season} holdout · teal is above the league average ({center:.2f})"
    )
    fig.tight_layout()
    return save_figure(fig, path)


def plot_player_chart(
    player_shots: pd.DataFrame,
    path: Path | None = None,
    p_col: str = "p_make",
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """One player's shots, colored by expected FG% rather than make or miss."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    draw_half_court(ax)
    if len(player_shots) == 0:
        ax.set_title(title or "No shots")
        return save_figure(fig, path)
    ax.scatter(
        player_shots["loc_x_ft"],
        player_shots["loc_y_ft"],
        c=player_shots[p_col],
        cmap=XFG_CMAP,
        vmin=0.32,
        vmax=0.70,
        s=16,
        alpha=0.9,
        linewidths=0.2,
        edgecolors="#1c2331",
        zorder=2,
    )
    colorbar = fig.colorbar(ax.collections[-1], ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Expected FG%")
    colorbar.outline.set_visible(False)
    ax.set_title(title or "Shot chart")
    if subtitle:
        ax.text(0.0, -5.6, subtitle, ha="center", va="top", color=MUTED, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_calibration(scored: pd.DataFrame, path: Path | None = None, n_bins: int = 10) -> plt.Figure:
    """When the model says 40%, do shots go in about 40% of the time?"""
    # Local import so the Streamlit app does not need scikit-learn installed.
    from sklearn.calibration import calibration_curve

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    y = scored["shot_made"].to_numpy()
    for name, column in [("zone", "p_zone"), ("logistic", "p_logistic"), ("lightgbm", "p_lightgbm")]:
        fraction, mean_pred = calibration_curve(y, scored[column], n_bins=n_bins, strategy="uniform")
        ax.plot(
            mean_pred,
            fraction,
            marker="o",
            lw=2,
            color=MODEL_COLORS[name],
            label=MODEL_LABELS[name],
        )
    ax.plot([0, 1], [0, 1], color="#c5bfb4", lw=1, linestyle="--", label="Perfect calibration")
    ax.set_xlim(0.15, 0.97)
    ax.set_ylim(0.15, 0.97)
    ax.set_xlabel("Predicted make probability")
    ax.set_ylabel("Shots that actually went in")
    ax.set_title("Calibration on the holdout season")
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_model_comparison(test_metrics: dict, path: Path | None = None) -> plt.Figure:
    """Lower is better. The axis starts at zero so a small gain does not look huge."""
    apply_style()
    names = ["zone", "logistic", "lightgbm"]
    labels = [MODEL_LABELS[name] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    for ax, key, title in zip(axes, ["log_loss", "brier"], ["Log loss  ·  lower is better", "Brier score  ·  lower is better"]):
        values = [test_metrics[name][key] for name in names]
        bars = ax.bar(labels, values, color=[MODEL_COLORS[name] for name in names], width=0.72)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(values) * 1.22)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.03,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
        ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_value_by_distance(scored: pd.DataFrame, path: Path | None = None, min_n: int = 200) -> plt.Figure:
    """Actual vs expected, by distance. The three-point line should show up as a jump in points, not in FG%."""
    apply_style()
    data = scored.loc[scored["shot_distance"] <= 30].copy()
    data["distance_ft"] = data["shot_distance"].round().astype(int)
    grouped = data.groupby("distance_ft").agg(
        n=("shot_made", "size"),
        fg=("shot_made", "mean"),
        xfg=("p_make", "mean"),
        pps=("actual_pts", "mean"),
        xpts=("xpts", "mean"),
    )
    grouped = grouped.loc[grouped["n"] >= min_n]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    axes[0].plot(grouped.index, grouped["fg"], color=INK, lw=2, label="Actual FG%")
    axes[0].plot(grouped.index, grouped["xfg"], color=TEAL, lw=2, label="Model xFG%")
    axes[0].set_ylabel("Make probability")
    axes[0].set_title("Make rate")
    axes[1].plot(grouped.index, grouped["pps"], color=INK, lw=2, label="Actual points")
    axes[1].plot(grouped.index, grouped["xpts"], color=TEAL, lw=2, label="Expected points")
    axes[1].set_ylabel("Points per shot")
    axes[1].set_title("Point value")
    for ax in axes:
        ax.axvline(22, color="#c5bfb4", lw=1, linestyle="--")
        ax.set_xlabel("Shot distance (feet)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("Holdout season by distance  ·  dashed line is the corner three", y=1.03)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_feature_importance(importance: list[dict] | pd.DataFrame, path: Path | None = None) -> plt.Figure:
    apply_style()
    frame = importance if isinstance(importance, pd.DataFrame) else pd.DataFrame(importance)
    frame = frame.sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(frame["feature"], frame["share"], color=NAVY, height=0.7)
    ax.set_xlabel("Share of total gain")
    ax.set_title("What the boosted model splits on")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_player_making(players: pd.DataFrame, path: Path | None = None, n_each: int = 10) -> plt.Figure:
    """Top and bottom qualified players by shrunk points over expected per 100 shots."""
    apply_style()
    qualified = players.loc[players["qualified"]].copy()
    top = qualified.nlargest(n_each, "poe_per_100_shrunk")
    bottom = qualified.nsmallest(n_each, "poe_per_100_shrunk")
    board = pd.concat([bottom, top]).drop_duplicates("player_id").sort_values("poe_per_100_shrunk")
    colors = np.where(board["poe_per_100_shrunk"] >= 0, TEAL, CRIMSON)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.barh(
        board["player_name"],
        board["poe_per_100_shrunk"],
        xerr=board["poe_se_per_100"],
        color=colors,
        height=0.72,
        ecolor="#b7b1a6",
        error_kw={"lw": 0.8, "capsize": 2},
    )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Shrunk points over expected per 100 shots\nWhiskers are the standard error of the raw average")
    ax.set_title("Shot making over expected  ·  holdout season, 300+ FGA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_team_making(teams: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    apply_style()
    board = teams.sort_values("poe_per_100")
    colors = np.where(board["poe_per_100"] >= 0, TEAL, CRIMSON)
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.barh(
        board["team_name"],
        board["poe_per_100"],
        xerr=board["poe_se_per_100"],
        color=colors,
        height=0.72,
        ecolor="#b7b1a6",
        error_kw={"lw": 0.8, "capsize": 2},
    )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Points over expected per 100 shots\nWhiskers are the standard error of the average")
    ax.set_title("Team shot making over expected  ·  holdout season")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_selection_vs_making(players: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Each qualified player is a dot. Axes are both points per 100 shots."""
    apply_style()
    data = players.loc[players["qualified"]].copy()
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.scatter(
        data["selection_per_100_shrunk"],
        data["poe_per_100_shrunk"],
        s=28,
        color=NAVY,
        alpha=0.75,
        linewidths=0,
        zorder=2,
    )
    ax.axvline(0, color="#c5bfb4", lw=1)
    ax.axhline(0, color="#c5bfb4", lw=1)
    labeled = _players_to_label(data)
    for row in labeled.itertuples(index=False):
        ax.annotate(
            row.player_name.split(" ")[-1],
            (row.selection_per_100_shrunk, row.poe_per_100_shrunk),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("Shot selection: expected points per 100 vs league (shrunk)")
    ax.set_ylabel("Shot making: points over expected per 100 (shrunk)")
    ax.set_title("Selection vs finishing  ·  qualified players, holdout season")
    # Corner notes, placed in data coordinates after limits settle.
    ax.text(0.02, 0.98, "tough shots,\nmakes them", transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=8)
    ax.text(0.98, 0.98, "good looks,\nmakes them", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8)
    ax.text(0.02, 0.02, "tough shots,\nmisses extra", transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=8)
    ax.text(0.98, 0.02, "good looks,\nmisses extra", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, path)


def _players_to_label(players: pd.DataFrame, n_making: int = 5, n_selection: int = 3) -> pd.DataFrame:
    """Label the extremes only, so the scatter stays readable."""
    pieces = [
        players.nlargest(n_making, "poe_per_100_shrunk"),
        players.nsmallest(n_making, "poe_per_100_shrunk"),
        players.nlargest(n_selection, "selection_per_100_shrunk"),
        players.nsmallest(n_selection, "selection_per_100_shrunk"),
    ]
    return pd.concat(pieces).drop_duplicates("player_id")
