#!/usr/bin/env python3
"""Download regular-season shot charts into ``data/raw``.

Tries stats.nba.com through ``nba_api`` when you ask for it. If that host does
not answer, or if you pass ``--source mirror``, the script caches the public
SportsDataverse parquet files instead.

    python scripts/pull_shots.py                 # use the cache, or fill it
    python scripts/pull_shots.py --source api    # live ShotChartDetail pull
    python scripts/pull_shots.py --source mirror --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotquality.data.pull import ensure_raw_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "api", "mirror"], default="auto")
    parser.add_argument("--refresh", action="store_true", help="Re-download the mirror even if files exist.")
    args = parser.parse_args()
    source = ensure_raw_files(source=args.source, refresh=args.refresh)
    print(f"Raw shots are ready. Source: {source}")


if __name__ == "__main__":
    main()
