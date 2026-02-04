#!/usr/bin/env python3
"""Compute background-corrected cell activity from raw fluorescence data.

Computes F0 (8th percentile per pixel with rolling window), applies per-cell median
smoothing (spatial filtering approximation), then computes (F - F0) and (F - F0) / F0.

Memory-efficient: streams through cell-aligned pixel chunks, writing cell activity
incrementally. Peak memory is O(pixels_in_chunk × num_timesteps).

Usage:
    python processing/compute_cell_activity.py --zarr output.zarr
"""

import argparse

import numpy as np
np.seterr(all='raise')
from scipy.ndimage import percentile_filter

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
PERCENTILE_WINDOW_RADIUS = 400  # ±400 frames (~6 minutes) for rolling baseline
CLIP_MIN = -0.25  # Clipping bounds for normalized activity
CLIP_MAX = 1.5
TARGET_CELLS_PER_CHUNK = 1000  # Target number of cells per processing chunk


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


def compute_cell_chunks(cell_boundaries: np.ndarray, target_cells: int) -> list:
    """Compute cell-aligned chunks for streaming.

    Returns list of (cell_start, cell_end, pixel_start, pixel_end) tuples.
    Each chunk contains complete cells only.
    """
    num_cells = len(cell_boundaries) - 1
    chunks = []

    cell_start = 0
    while cell_start < num_cells:
        cell_end = min(cell_start + target_cells, num_cells)
        pixel_start = cell_boundaries[cell_start]
        pixel_end = cell_boundaries[cell_end]
        chunks.append((cell_start, cell_end, pixel_start, pixel_end))
        cell_start = cell_end

    return chunks


def process_cell_chunk(
    raw_values,
    acquisition_time_ms,
    pixel_start: int,
    pixel_end: int,
    cell_start: int,
    cell_end: int,
    cell_boundaries: np.ndarray,
    percentile: int,
    window_radius: int,
    clip_min: float,
    clip_max: float,
) -> tuple:
    """Process a chunk of cells: compute F0, apply per-cell smoothing, aggregate.

    Args:
        raw_values: TensorStore array [num_pixels, num_timesteps]
        acquisition_time_ms: TensorStore array [num_pixels, num_timesteps]
        pixel_start, pixel_end: Pixel range for this chunk
        cell_start, cell_end: Cell range for this chunk
        cell_boundaries: Full cell boundaries array
        percentile: Percentile for baseline (e.g., 8)
        window_radius: Rolling window radius (±frames)
        clip_min, clip_max: Clipping bounds for normalized activity

    Returns:
        cell_activity: [num_cells_in_chunk, num_timesteps]
        cell_activity_normalized: [num_cells_in_chunk, num_timesteps]
        cell_acquisition_ms: [num_cells_in_chunk, num_timesteps]
    """
    num_cells_chunk = cell_end - cell_start
    window_size = 2 * window_radius + 1

    # Load raw values for this pixel chunk (all timesteps)
    F = raw_values[pixel_start:pixel_end, :].read().result().astype(np.float32)
    num_timesteps = F.shape[1]

    # Step 1: Compute rolling percentile F0 for each pixel
    F0 = percentile_filter(
        F,
        percentile=percentile,
        size=(1, window_size),
        mode='reflect'
    )

    # Step 2: Apply per-cell median smoothing to F0
    # For each cell, replace each pixel's F0 with median F0 across all pixels in cell
    local_boundaries = cell_boundaries[cell_start:cell_end + 1] - pixel_start
    for c in range(num_cells_chunk):
        p_start = local_boundaries[c]
        p_end = local_boundaries[c + 1]
        if p_end > p_start:
            # Compute median F0 across pixels in this cell, for each timestep
            cell_median_F0 = np.median(F0[p_start:p_end, :], axis=0, keepdims=True)
            F0[p_start:p_end, :] = cell_median_F0

    # Step 3: Compute dF and dF/F0
    F0_safe = np.maximum(F0, 1.0)
    dF = F - F0
    dF_norm = np.clip(dF / F0_safe, clip_min, clip_max)

    # Step 4: Aggregate to cells
    cell_activity = np.empty((num_cells_chunk, num_timesteps), dtype=np.float32)
    cell_activity_normalized = np.empty((num_cells_chunk, num_timesteps), dtype=np.float32)
    cell_acquisition_ms = np.empty((num_cells_chunk, num_timesteps), dtype=np.uint16)

    # Load acquisition times
    acq = acquisition_time_ms[pixel_start:pixel_end, :].read().result().astype(np.float64)

    for c in range(num_cells_chunk):
        p_start = local_boundaries[c]
        p_end = local_boundaries[c + 1]
        num_pixels = p_end - p_start
        if num_pixels > 0:
            cell_activity[c, :] = dF[p_start:p_end, :].mean(axis=0)
            cell_activity_normalized[c, :] = dF_norm[p_start:p_end, :].mean(axis=0)
            cell_acquisition_ms[c, :] = np.round(acq[p_start:p_end, :].mean(axis=0)).astype(np.uint16)
        else:
            cell_activity[c, :] = 0
            cell_activity_normalized[c, :] = 0
            cell_acquisition_ms[c, :] = 0

    return cell_activity, cell_activity_normalized, cell_acquisition_ms


def main():
    args = parse_args()

    print(f"Zarr: {args.zarr}", flush=True)

    # Load metadata
    metadata = load_metadata(args.zarr)
    num_timesteps = metadata['num_timesteps']
    num_pixels = metadata['num_pixels']
    num_cells = metadata['num_cells']
    print(f"Data: {num_pixels:,} pixels, {num_timesteps} timesteps, {num_cells:,} cells", flush=True)

    # Open input arrays
    print("Opening input arrays...", flush=True)
    raw_values = open_array(args.zarr, 'raw_values')
    acquisition_time_ms = open_array(args.zarr, 'acquisition_time_ms')
    cell_boundaries = open_array(args.zarr, 'cell_pixel_boundaries').read().result()

    # Compute cell-aligned chunks
    chunks = compute_cell_chunks(cell_boundaries, TARGET_CELLS_PER_CHUNK)
    print(f"Processing {len(chunks)} cell-aligned chunks (~{TARGET_CELLS_PER_CHUNK} cells each)", flush=True)

    # Create output arrays
    print("Creating output arrays...", flush=True)
    cell_activity_arr = create_array(
        args.zarr, 'cell_activity',
        (num_cells, num_timesteps), (TARGET_CELLS_PER_CHUNK, TIME_CHUNK_SIZE), 'float32'
    )
    cell_activity_norm_arr = create_array(
        args.zarr, 'cell_activity_normalized',
        (num_cells, num_timesteps), (TARGET_CELLS_PER_CHUNK, TIME_CHUNK_SIZE), 'float32'
    )
    cell_acq_arr = create_array(
        args.zarr, 'cell_acquisition_ms',
        (num_cells, num_timesteps), (TARGET_CELLS_PER_CHUNK, TIME_CHUNK_SIZE), 'uint16'
    )

    # Process each chunk
    print(f"\nProcessing cells (F0: {PERCENTILE}th percentile, ±{PERCENTILE_WINDOW_RADIUS} frames)...", flush=True)
    for i, (cell_start, cell_end, pixel_start, pixel_end) in enumerate(chunks):
        num_pixels_chunk = pixel_end - pixel_start
        print(f"  Chunk {i+1}/{len(chunks)}: cells [{cell_start}, {cell_end}), "
              f"pixels [{pixel_start}, {pixel_end}) ({num_pixels_chunk:,} pixels)", flush=True)

        cell_activity, cell_activity_normalized, cell_acquisition_ms = process_cell_chunk(
            raw_values, acquisition_time_ms,
            pixel_start, pixel_end,
            cell_start, cell_end,
            cell_boundaries,
            PERCENTILE, PERCENTILE_WINDOW_RADIUS,
            CLIP_MIN, CLIP_MAX,
        )

        # Write chunk results immediately
        cell_activity_arr[cell_start:cell_end, :].write(cell_activity).result()
        cell_activity_norm_arr[cell_start:cell_end, :].write(cell_activity_normalized).result()
        cell_acq_arr[cell_start:cell_end, :].write(cell_acquisition_ms).result()

    # Update metadata
    metadata['baseline_percentile'] = PERCENTILE
    metadata['baseline_window_radius'] = PERCENTILE_WINDOW_RADIUS
    metadata['spatial_smoothing'] = 'per_cell_median'
    metadata['normalized_clip_range'] = [CLIP_MIN, CLIP_MAX]
    save_metadata(args.zarr, metadata)

    print(f"\nDone. Wrote cell_activity, cell_activity_normalized, cell_acquisition_ms to {args.zarr}", flush=True)


if __name__ == "__main__":
    main()
