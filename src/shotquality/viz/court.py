"""Half-court drawing in feet.

The basket is (0, 0). y increases toward half court. Multiply the NBA feed's
``x_legacy`` / ``y_legacy`` by 0.1 and the dots land on these lines. The
baseline sits at y = -4.75 in this convention (the usual shot-chart frame),
which is what lines up with the data, rather than a from-scratch blueprint.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle, Rectangle

from shotquality.features.build import (
    BASELINE_Y_FT,
    CORNER_THREE_X_FT,
    CORNER_THREE_Y_FT,
    FREE_THROW_Y_FT,
    PAINT_HALF_WIDTH_FT,
    RESTRICTED_RADIUS_FT,
    THREE_POINT_RADIUS_FT,
)
from shotquality.viz.style import COURT, LINE, apply_style

# Half court is 47 feet from baseline to midcourt. Baseline is at -4.75,
# so the midcourt line is at 42.25.
HALFCOURT_Y_FT = BASELINE_Y_FT + 47.0
COURT_HALF_WIDTH_FT = 25.0


def draw_half_court(ax: Axes, line_color: str = LINE, lw: float = 1.35, face_color: str = COURT) -> Axes:
    """Draw the offensive half court on ``ax`` and fix the limits."""
    apply_style()
    ax.set_facecolor(face_color)

    def add(patch) -> None:
        ax.add_patch(patch)

    # Floor, so hex bins never paint outside the lines.
    add(
        Rectangle(
            (-COURT_HALF_WIDTH_FT, BASELINE_Y_FT),
            COURT_HALF_WIDTH_FT * 2,
            HALFCOURT_Y_FT - BASELINE_Y_FT,
            linewidth=0,
            facecolor=face_color,
            zorder=0,
        )
    )
    # Outer lines: baseline, sidelines, half-court line.
    add(
        Rectangle(
            (-COURT_HALF_WIDTH_FT, BASELINE_Y_FT),
            COURT_HALF_WIDTH_FT * 2,
            HALFCOURT_Y_FT - BASELINE_Y_FT,
            linewidth=lw,
            edgecolor=line_color,
            facecolor="none",
            zorder=3,
        )
    )
    # Paint (16 ft wide) and the inner key (12 ft wide).
    paint_height = FREE_THROW_Y_FT - BASELINE_Y_FT
    add(
        Rectangle(
            (-PAINT_HALF_WIDTH_FT, BASELINE_Y_FT),
            PAINT_HALF_WIDTH_FT * 2,
            paint_height,
            linewidth=lw,
            edgecolor=line_color,
            facecolor="none",
            zorder=3,
        )
    )
    add(
        Rectangle(
            (-6, BASELINE_Y_FT),
            12,
            paint_height,
            linewidth=lw,
            edgecolor=line_color,
            facecolor="none",
            zorder=3,
        )
    )
    # Free-throw circle. Solid on the far side, dashed inside the paint.
    add(Arc((0, FREE_THROW_Y_FT), 12, 12, theta1=0, theta2=180, linewidth=lw, edgecolor=line_color, zorder=3))
    add(
        Arc(
            (0, FREE_THROW_Y_FT),
            12,
            12,
            theta1=180,
            theta2=360,
            linewidth=lw,
            edgecolor=line_color,
            linestyle=(0, (2, 2)),
            zorder=3,
        )
    )
    # Restricted area: 4-foot arc above the rim.
    add(
        Arc(
            (0, 0),
            RESTRICTED_RADIUS_FT * 2,
            RESTRICTED_RADIUS_FT * 2,
            theta1=0,
            theta2=180,
            linewidth=lw,
            edgecolor=line_color,
            zorder=3,
        )
    )
    # Rim (18 inches across) and backboard.
    add(Circle((0, 0), radius=0.75, linewidth=lw, edgecolor=line_color, facecolor="none", zorder=4))
    ax.plot([-3, 3], [-0.75, -0.75], color=line_color, lw=lw + 0.4, zorder=4, solid_capstyle="butt")

    # Corner threes, then the arc that connects them.
    ax.plot(
        [-CORNER_THREE_X_FT, -CORNER_THREE_X_FT],
        [BASELINE_Y_FT, CORNER_THREE_Y_FT],
        color=line_color,
        lw=lw,
        zorder=3,
    )
    ax.plot(
        [CORNER_THREE_X_FT, CORNER_THREE_X_FT],
        [BASELINE_Y_FT, CORNER_THREE_Y_FT],
        color=line_color,
        lw=lw,
        zorder=3,
    )
    corner_angle = float(np.degrees(np.arctan2(CORNER_THREE_Y_FT, CORNER_THREE_X_FT)))
    add(
        Arc(
            (0, 0),
            THREE_POINT_RADIUS_FT * 2,
            THREE_POINT_RADIUS_FT * 2,
            theta1=corner_angle,
            theta2=180 - corner_angle,
            linewidth=lw,
            edgecolor=line_color,
            zorder=3,
        )
    )
    # Near half of the center circle.
    add(
        Arc(
            (0, HALFCOURT_Y_FT),
            12,
            12,
            theta1=180,
            theta2=360,
            linewidth=lw,
            edgecolor=line_color,
            zorder=3,
        )
    )

    ax.set_xlim(-26, 26)
    ax.set_ylim(-6.2, HALFCOURT_Y_FT + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
