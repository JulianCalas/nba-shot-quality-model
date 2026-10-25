"""The court helper should draw without needing a display."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shotquality.viz.court import HALFCOURT_Y_FT, draw_half_court


def test_half_court_sets_basket_at_origin_and_draws_lines():
    fig, ax = plt.subplots()
    draw_half_court(ax)
    assert ax.get_xlim()[0] < 0 < ax.get_xlim()[1]
    assert ax.get_ylim()[0] < 0 < HALFCOURT_Y_FT <= ax.get_ylim()[1]
    assert len(ax.patches) > 5
    plt.close(fig)
