"""
Synthetic MR1 worker used for visualization-only test trees.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic MR1 worker")
    parser.add_argument("--sleep", type=float, default=12.0)
    args = parser.parse_args()

    remaining = max(0.0, args.sleep)
    while remaining > 0:
        interval = min(0.5, remaining)
        time.sleep(interval)
        remaining -= interval


if __name__ == "__main__":
    main()
