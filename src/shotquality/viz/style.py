"""Shared colors. The figures should look like a scouting memo, not a default stylesheet."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#1c2331"
MUTED = "#5e6977"
NAVY = "#1f4b73"
TEAL = "#0e7c66"
CRIMSON = "#b4333a"
SAND = "#f4efe4"
LINE = "#2c3544"
COURT = "#fbf7f1"
GRID = "#e6e1d8"

# Sequential scale for make probability: low navy, high amber.
XFG_CMAP = LinearSegmentedColormap.from_list(
    "xfg",
    ["#243652", "#4f6d8c", "#d9c7a5", "#e09f3e", "#c4491d"],
)

# Points relative to average: below average in crimson, above in teal.
VALUE_CMAP = LinearSegmentedColormap.from_list(
    "value",
    [CRIMSON, "#f7f1e6", TEAL],
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
