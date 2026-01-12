#!/usr/bin/env python3
"""Compute background-corrected cell activity from raw fluorescence data.

Computes F0 (8th percentile per pixel), then aggregates (F - F0) and (F - F0) / F0 per cell.

Usage:
    python processing/compute_cell_activity.py --zarr output.zarr
"""

import argparse

import numpy as np
np.seterr(all='raise')

from zarr_utils import (
    TIME_CHUNK_SIZE,
    set_thread_limits,
    open_array,
    create_array,
    load_metadata,
    save_metadata,
)

set_thread_limits(2)

# Processing parameters
PERCENTILE = 8
PIXEL_CHUNK_SIZE = 100_000  # pixels per chunk for percentile computation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute background-corrected cell activity"
    )
    parser.add_argument(
        "--zarr",
        type=str,
        required=True,
        help="Path to zarr with raw_values, cell_ids, aligned_coords",
    )
    return parser.parse_args()


def compute_percentile_chunked(raw_values, percentile: int, chunk_size: int) -> np.ndarray:
    """Compute percentile over time for each pixel, processing in chunks."""
    num_pixels = raw_values.shape[0]
    F0 = np.empty(num_pixels, dtype=np.float32)

    num_chunks = (num_pixels + chunk_size - 1) // chunk_size

    for i, start in enumerate(range(0, num_pixels, chunk_size)):
        end = min(start + chunk_size, num_pixels)
        print(f"  Computing percentile for pixels [{start:,}, {end:,}) ({i+1}/{num_chunks})", flush=True)

        # Load chunk of pixels, all timesteps
        chunk_data = raw_values[start:end, :].read().result()
        F0[start:end] = np.percentile(chunk_data, percentile, axis=1).astype(np.float32)

    return F0


def aggregate_cell_data(
    raw_values,
    acquisition_time_ms,
    F0: np.ndarray,
    cell_boundaries: np.ndarray,
    pixels_per_cell: np.ndarray,
    num_cells: int,
    num_timesteps: int,
    time_chunk_size: int
) -> tuple:
    """Compute corrected activity and acquisition time, aggregated per cell.

    Streams through time chunks to avoid loading full array.

    Returns:
        cell_activity: [num_cells, num_timesteps] - mean corrected activity (F - F0)
        cell_activity_normalized: [num_cells, num_timesteps] - mean normalized activity (F - F0) / F0
        cell_acquisition_ms: [num_cells, num_timesteps] - mean acquisition time
    """
    # Output arrays
    cell_activity = np.zeros((num_cells, num_timesteps), dtype=np.float32)
    cell_activity_normalized = np.zeros((num_cells, num_timesteps), dtype=np.float32)
    cell_acquisition_ms = np.zeros((num_cells, num_timesteps), dtype=np.uint32)

    num_chunks = (num_timesteps + time_chunk_size - 1) // time_chunk_size

    # Precompute F0 with small epsilon to avoid division by zero
    F0_safe = np.maximum(F0, 1.0)

    for i, t_start in enumerate(range(0, num_timesteps, time_chunk_size)):
        t_end = min(t_start + time_chunk_size, num_timesteps)
        print(f"  Processing timesteps [{t_start}, {t_end}) ({i+1}/{num_chunks})", flush=True)

        # Load time chunks
        F = raw_values[:, t_start:t_end].read().result().astype(np.float32)
        acq = acquisition_time_ms[:, t_start:t_end].read().result().astype(np.float64)

        # Compute corrected activity: S = F - F0
        dF = F - F0[:, np.newaxis]

        # Compute normalized activity: (F - F0) / F0
        dF_norm = dF / F0_safe[:, np.newaxis]

        # Aggregate per cell using reduceat (boundaries must be int64 for reduceat)
        bounds = cell_boundaries[:-1].astype(np.int64)
        activity_sums = np.add.reduceat(dF, bounds, axis=0)
        cell_activity[:, t_start:t_end] = activity_sums / pixels_per_cell[:, np.newaxis]

        activity_norm_sums = np.add.reduceat(dF_norm, bounds, axis=0)
        cell_activity_normalized[:, t_start:t_end] = activity_norm_sums / pixels_per_cell[:, np.newaxis]

        acq_sums = np.add.reduceat(acq, bounds, axis=0)
        cell_acquisition_ms[:, t_start:t_end] = np.round(
            acq_sums / pixels_per_cell[:, np.newaxis]
        ).astype(np.uint32)

    return cell_activity, cell_activity_normalized, cell_acquisition_ms


def main():
    args = parse_args()

    print(f"Zarr: {args.zarr}", flush=True)

    # Load metadata
    metadata = load_metadata(args.zarr)
    num_timesteps = metadata['num_timesteps']
    print(f"Data: {metadata['num_pixels']:,} pixels, {num_timesteps} timesteps", flush=True)

    # Open input arrays
    print("Opening input arrays...", flush=True)
    raw_values = open_array(args.zarr, 'raw_values')
    acquisition_time_ms = open_array(args.zarr, 'acquisition_time_ms')

    num_cells = metadata['num_cells']
    print(f"Cells: {num_cells:,}", flush=True)

    # Step 1: Compute F0 (8th percentile per pixel)
    print(f"\nStep 1: Computing F0 ({PERCENTILE}th percentile)...", flush=True)
    F0 = compute_percentile_chunked(raw_values, PERCENTILE, PIXEL_CHUNK_SIZE)
    print(f"  F0 range: [{F0.min():.1f}, {F0.max():.1f}]", flush=True)

    # Step 2: Compute corrected activity and acquisition time, aggregate per cell
    print("\nStep 2: Computing cell activity and acquisition times...", flush=True)
    cell_boundaries = open_array(args.zarr, 'cell_pixel_boundaries').read().result()
    pixels_per_cell = np.diff(cell_boundaries).astype(np.float32)
    pixels_per_cell = np.maximum(pixels_per_cell, 1.0)  # Avoid division by zero for empty cells
    cell_activity, cell_activity_normalized, cell_acquisition_ms = aggregate_cell_data(
        raw_values, acquisition_time_ms, F0,
        cell_boundaries, pixels_per_cell,
        num_cells, num_timesteps, TIME_CHUNK_SIZE
    )
    print(f"  cell_activity range: [{cell_activity.min():.1f}, {cell_activity.max():.1f}]", flush=True)
    print(f"  cell_activity_normalized range: [{cell_activity_normalized.min():.3f}, {cell_activity_normalized.max():.3f}]", flush=True)
    print(f"  cell_acquisition_ms range: [{cell_acquisition_ms.min()}, {cell_acquisition_ms.max()}] ms", flush=True)

    # Write outputs to existing zarr
    print("\nWriting outputs...", flush=True)

    num_pixels = len(F0)

    # baseline_F0
    print("  Writing baseline_F0...", flush=True)
    F0_arr = create_array(args.zarr, 'baseline_F0', (num_pixels,), (num_pixels,), 'float32')
    F0_arr.write(F0).result()

    # cell_activity
    print("  Writing cell_activity...", flush=True)
    cell_activity_arr = create_array(
        args.zarr, 'cell_activity',
        (num_cells, num_timesteps), (num_cells, 100), 'float32'
    )
    cell_activity_arr.write(cell_activity).result()

    # cell_activity_normalized
    print("  Writing cell_activity_normalized...", flush=True)
    cell_activity_norm_arr = create_array(
        args.zarr, 'cell_activity_normalized',
        (num_cells, num_timesteps), (num_cells, 100), 'float32'
    )
    cell_activity_norm_arr.write(cell_activity_normalized).result()

    # cell_acquisition_ms
    print("  Writing cell_acquisition_ms...", flush=True)
    cell_acq_arr = create_array(
        args.zarr, 'cell_acquisition_ms',
        (num_cells, num_timesteps), (num_cells, 100), 'uint32'
    )
    cell_acq_arr.write(cell_acquisition_ms).result()

    # Update metadata
    metadata['baseline_percentile'] = PERCENTILE
    metadata['num_cells'] = num_cells
    save_metadata(args.zarr, metadata)

    print(f"\nDone. Added baseline_F0, cell_activity, cell_activity_normalized, cell_acquisition_ms to {args.zarr}", flush=True)


if __name__ == "__main__":
    main()
