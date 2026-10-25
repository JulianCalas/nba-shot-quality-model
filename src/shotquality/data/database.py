"""DuckDB file used for the SQL part of the analysis.

The database is rebuilt from the parquet tables. It is not a second copy of
the raw feed that has to be edited by hand.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def build_database(shots: pd.DataFrame, predictions: pd.DataFrame | None, path: Path) -> duckdb.DuckDBPyConnection:
    """Create (or replace) a DuckDB file with a ``shots`` table and, if given, ``predictions``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = duckdb.connect(str(path))
    connection.register("shots_df", shots)
    connection.execute("CREATE TABLE shots AS SELECT * FROM shots_df")
    if predictions is not None:
        connection.register("pred_df", predictions)
        connection.execute("CREATE TABLE predictions AS SELECT * FROM pred_df")
    return connection
