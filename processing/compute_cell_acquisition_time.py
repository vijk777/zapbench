#!/usr/bin/env python3
"""Compute per-cell acquisition timestamps by averaging over pixels.

Adds cell_acquisition_ms [num_cells, num_timesteps] to the zarr.

Usage:
    python compute_cell_acquisition_time.py --zarr /path/to/output.zarr
"""

import argparse

import numpy as np

from zarr_utils import (
    TIME_CHUNK_SIZE,
    set_thread_limits,
    open_array,
    create_array,
    load_metadata,
    compute_cell_boundaries,
)

set_thread_limits(2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-cell acquisition timestamps"
    )
    parser.add_argument(
        "--zarr",
        type=str,
        required=True,
        help="Path to zarr with acquisition_time and cell_ids",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Zarr: {args.zarr}", flush=True)

    # Load metadata
    metadata = load_metadata(args.zarr)
    num_timesteps = metadata['num_timesteps']
    print(f"Data: {metadata['num_pixels']:,} pixels, {num_timesteps} timesteps", flush=True)

    # Open input arrays
    print("Opening input arrays...", flush=True)
    acquisition_time = open_array(args.zarr, 'acquisition_time')
    cell_ids = open_array(args.zarr, 'cell_ids').read().result()

    num_cells = int(cell_ids.max()) + 1
    print(f"Cells: {num_cells:,}", flush=True)

    # Precompute cell boundaries
    cell_boundaries, pixels_per_cell = compute_cell_boundaries(cell_ids, num_cells)

    # Output array
    cell_acquisition_ms = np.zeros((num_cells, num_timesteps), dtype=np.uint32)

    num_chunks = (num_timesteps + TIME_CHUNK_SIZE - 1) // TIME_CHUNK_SIZE

    print("Computing cell acquisition times...", flush=True)
    for i, t_start in enumerate(range(0, num_timesteps, TIME_CHUNK_SIZE)):
        t_end = min(t_start + TIME_CHUNK_SIZE, num_timesteps)
        print(f"  Processing timesteps [{t_start}, {t_end}) ({i+1}/{num_chunks})", flush=True)

        # Load time chunk
        acq_chunk = acquisition_time[:, t_start:t_end].read().result().astype(np.float64)

        # Aggregate per cell using reduceat
        cell_sums = np.add.reduceat(acq_chunk, cell_boundaries[:-1], axis=0)
        cell_means = cell_sums / pixels_per_cell[:, np.newaxis]

        # Round to integers
        cell_acquisition_ms[:, t_start:t_end] = np.round(cell_means).astype(np.uint32)

    print(f"  Range: [{cell_acquisition_ms.min()}, {cell_acquisition_ms.max()}] ms", flush=True)

    # Write output
    print("Writing cell_acquisition_ms...", flush=True)
    out_arr = create_array(
        args.zarr, 'cell_acquisition_ms',
        (num_cells, num_timesteps), (num_cells, 100), 'uint32'
    )
    out_arr.write(cell_acquisition_ms).result()

    print(f"Done. Added cell_acquisition_ms to {args.zarr}", flush=True)


if __name__ == "__main__":
    main()
