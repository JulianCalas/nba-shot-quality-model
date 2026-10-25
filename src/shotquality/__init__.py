"""NBA shot quality: expected points, and who beats that number.

The package is organized the way the analysis actually runs:

- ``data`` loads the shot feed and builds a clean table
- ``features`` turns locations into model inputs, without using the outcome
- ``models`` fits the baseline, the logistic regression, and the boosted model
- ``analysis`` aggregates with SQL and builds leaderboards
- ``viz`` draws the court and the figures
"""

__version__ = "1.0.0"
