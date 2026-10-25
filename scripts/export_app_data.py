#!/usr/bin/env python3
"""Write the compact ``data/app`` extract the Streamlit app ships with.

Run this after ``scripts/run_pipeline.py``. The pipeline also calls the same
function, so a full run already refreshes these files.

    python scripts/export_app_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotquality.export_app import export_app_dataset  # noqa: E402


def main() -> None:
    info = export_app_dataset()
    total_mb = info["total_bytes"] / 1_048_576
    print(
        f"Wrote data/app: {info['shots']} shots, {info['players']} players, "
        f"{info['teams']} teams, {info['shot_dtype']}, {total_mb:.2f} MB"
    )
    for name, size in info["bytes"].items():
        print(f"  {name:28} {size / 1_048_576:.2f} MB")


if __name__ == "__main__":
    main()
