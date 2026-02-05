#!/usr/bin/env python3
"""Compute per-cell acquisition timestamps from pixel-level acquisition times.

For each cell, computes the mean acquisition_time_ms across its constituent pixels.
This is the timestamp-only subset of compute_cell_activity.py, without the expensive
percentile baseline computation.

Usage:
    python processing/compute_cell_timestamps.py --zarr output.zarr
"""

import argparse

import numpy as np

from zarr_utils import (
    CELL_ACTIVITY_CELLS_PER_CHUNK as TARGET_CELLS_PER_CHUNK,
    TIME_CHUNK_SIZE,
    set_thread_limits,
    open_array,
    create_array,
    load_metadata,
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
        help="Path to zarr with acquisition_time_ms, cell_pixel_boundaries",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    metadata = load_metadata(args.zarr)
    num_timesteps = metadata['num_timesteps']
    num_cells = metadata['num_cells']

    print(f"Zarr: {args.zarr}", flush=True)
    print(f"Data: {num_cells:,} cells, {num_timesteps} timesteps", flush=True)

    # Create output array
    print("Creating cell_acquisition_ms output array...", flush=True)
    cell_acq_arr = create_array(
        args.zarr, 'cell_acquisition_ms',
        (num_cells, num_timesteps), (TARGET_CELLS_PER_CHUNK, TIME_CHUNK_SIZE), 'uint16'
    )

    acquisition_time_ms = open_array(args.zarr, 'acquisition_time_ms')
    cell_boundaries = open_array(args.zarr, 'cell_pixel_boundaries').read().result()

    # Process in cell-aligned chunks
    c = 0
    chunk_idx = 0
    while c < num_cells:
        c_end = min(c + TARGET_CELLS_PER_CHUNK, num_cells)
        pixel_start = int(cell_boundaries[c])
        pixel_end = int(cell_boundaries[c_end])
        num_pixels_chunk = pixel_end - pixel_start
        chunk_idx += 1
        print(f"  Chunk {chunk_idx}: cells [{c}, {c_end}), "
              f"pixels [{pixel_start}, {pixel_end}) ({num_pixels_chunk:,} pixels)", flush=True)

        acq = acquisition_time_ms[pixel_start:pixel_end, :].read().result().astype(np.float64)
        local_boundaries = cell_boundaries[c:c_end + 1] - pixel_start

        num_cells_chunk = c_end - c
        cell_acq = np.empty((num_cells_chunk, num_timesteps), dtype=np.uint16)

        for i in range(num_cells_chunk):
            p_start = int(local_boundaries[i])
            p_end = int(local_boundaries[i + 1])
            if p_end > p_start:
                cell_acq[i, :] = np.round(acq[p_start:p_end, :].mean(axis=0)).astype(np.uint16)
            else:
                cell_acq[i, :] = 0

        cell_acq_arr[c:c_end, :].write(cell_acq).result()
        c = c_end

    print(f"\nDone. Wrote cell_acquisition_ms to {args.zarr}", flush=True)


if __name__ == "__main__":
    main()
