#!/usr/bin/env python3
"""Initialize cell activity output arrays before running parallel compute jobs.

Usage:
    python processing/init_cell_activity.py --zarr output.zarr
"""

import argparse

from init_raw_fluorescence_zarr import init_cell_activity_arrays


def main():
    parser = argparse.ArgumentParser(
        description="Initialize cell activity output arrays"
    )
    parser.add_argument(
        "--zarr",
        type=str,
        required=True,
        help="Path to zarr (must already contain metadata from init_raw_fluorescence_zarr)",
    )
    args = parser.parse_args()
    init_cell_activity_arrays(args.zarr)


if __name__ == "__main__":
    main()
