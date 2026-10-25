#!/usr/bin/env python3
"""Train the shot-quality models and write figures, tables, and the database.

    python scripts/run_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotquality.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "api", "mirror"], default="auto")
    args = parser.parse_args()
    metrics = run_pipeline(source=args.source)
    test = metrics["test"]
    print(f"Primary model: {metrics['primary_model']}")
    for name in ("zone", "logistic", "lightgbm"):
        print(
            f"  {name:10}  log loss {test[name]['log_loss']:.4f}   "
            f"brier {test[name]['brier']:.4f}   ece {test[name]['ece']:.4f}"
        )


if __name__ == "__main__":
    main()
