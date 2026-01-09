#!/usr/bin/env python3
"""Compute background-corrected cell activity from raw fluorescence data.

Pipeline:
1. Compute F0 (10th percentile) per pixel over time
2. Fit smooth spatial field A_hat via binning and Gaussian smoothing
3. Compute corrected activity S = max(0, F - A_hat) and aggregate per cell

Adds the following arrays to the existing zarr:
- baseline_F0: [num_pixels] - 10th percentile per pixel
- baseline_A_hat: [num_pixels] - smooth spatial field at each pixel
- baseline_A_hat_grid: [nx, ny, nz] - binned/smoothed grid for inspection
- cell_activity: [num_cells, num_timesteps] - mean corrected activity per cell

Usage:
    python compute_cell_activity.py --zarr /path/to/output.zarr
"""

import argparse
import json
import os

import numpy as np
import tensorstore as ts
from scipy.ndimage import gaussian_filter

# Limit threads to avoid oversubscription on cluster
os.environ['BLOSC_NTHREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['NUMEXPR_MAX_THREADS'] = '2'

# Spatial binning parameters
BIN_STRIDE_X = 64
BIN_STRIDE_Y = 64
BIN_STRIDE_Z = 2
GAUSSIAN_SIGMA = 2

# Processing parameters
PERCENTILE = 10
PIXEL_CHUNK_SIZE = 100_000  # pixels per chunk for percentile computation
TIME_CHUNK_SIZE = 100  # timesteps per chunk for aggregation


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


def open_array(zarr_path: str, name: str):
    """Open a zarr array for reading."""
    return ts.open({
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(zarr_path, name),
        },
        'open': True,
    }).result()


def create_array(zarr_path: str, name: str, shape: tuple, chunks: tuple, dtype: str):
    """Create a zarr3 array."""
    spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(zarr_path, name),
        },
        'metadata': {
            'shape': list(shape),
            'chunk_grid': {
                'name': 'regular',
                'configuration': {'chunk_shape': list(chunks)}
            },
            'data_type': dtype,
            'codecs': [
                {'name': 'transpose', 'configuration': {'order': list(range(len(shape) - 1, -1, -1))}},
                {'name': 'bytes', 'configuration': {'endian': 'little'}},
                {'name': 'blosc', 'configuration': {'cname': 'zstd', 'clevel': 4, 'shuffle': 'shuffle'}}
            ],
        },
        'create': True,
        'delete_existing': True,
    }
    return ts.open(spec).result()


def compute_percentile_chunked(raw_values, percentile: int, chunk_size: int) -> np.ndarray:
    """Compute percentile over time for each pixel, processing in chunks."""
    num_pixels, num_timesteps = raw_values.shape
    F0 = np.empty(num_pixels, dtype=np.float32)

    num_chunks = (num_pixels + chunk_size - 1) // chunk_size

    for i, start in enumerate(range(0, num_pixels, chunk_size)):
        end = min(start + chunk_size, num_pixels)
        print(f"  Computing percentile for pixels [{start:,}, {end:,}) ({i+1}/{num_chunks})", flush=True)

        # Load chunk of pixels, all timesteps
        chunk_data = raw_values[start:end, :].read().result()
        F0[start:end] = np.percentile(chunk_data, percentile, axis=1).astype(np.float32)

    return F0


def fit_smooth_spatial_field(F0: np.ndarray, aligned_coords: np.ndarray, volume_shape: tuple) -> tuple:
    """Fit smooth spatial field via binning and Gaussian smoothing.

    Returns:
        A_hat: [num_pixels] - interpolated smooth field at each pixel
        A_hat_grid: [nx, ny, nz] - the smoothed binned grid
    """
    num_pixels = len(F0)
    x = aligned_coords[:, 0]
    y = aligned_coords[:, 1]
    z = aligned_coords[:, 2]

    # Compute bin indices
    bx = x // BIN_STRIDE_X
    by = y // BIN_STRIDE_Y
    bz = z // BIN_STRIDE_Z

    # Grid dimensions
    nx = (volume_shape[0] + BIN_STRIDE_X - 1) // BIN_STRIDE_X
    ny = (volume_shape[1] + BIN_STRIDE_Y - 1) // BIN_STRIDE_Y
    nz = (volume_shape[2] + BIN_STRIDE_Z - 1) // BIN_STRIDE_Z

    print(f"  Binning into grid [{nx}, {ny}, {nz}]", flush=True)

    # Accumulate sum and count per bin
    grid_sum = np.zeros((nx, ny, nz), dtype=np.float64)
    grid_count = np.zeros((nx, ny, nz), dtype=np.int64)

    # Use np.add.at for efficient accumulation
    np.add.at(grid_sum, (bx, by, bz), F0)
    np.add.at(grid_count, (bx, by, bz), 1)

    # Compute mean per bin (avoid division by zero)
    mask = grid_count > 0
    grid_mean = np.zeros_like(grid_sum, dtype=np.float32)
    grid_mean[mask] = grid_sum[mask] / grid_count[mask]

    # Fill empty bins with nearest neighbor (simple approach: use median of non-empty)
    if not mask.all():
        grid_mean[~mask] = np.median(grid_mean[mask])

    print(f"  Applying Gaussian smoothing (sigma={GAUSSIAN_SIGMA})", flush=True)
    A_hat_grid = gaussian_filter(grid_mean.astype(np.float64), sigma=GAUSSIAN_SIGMA).astype(np.float32)

    # Interpolate back to pixel coordinates using nearest bin (fast)
    # Clamp bin indices to valid range
    bx_clamped = np.clip(bx, 0, nx - 1)
    by_clamped = np.clip(by, 0, ny - 1)
    bz_clamped = np.clip(bz, 0, nz - 1)

    print(f"  Interpolating to {num_pixels:,} pixels", flush=True)
    A_hat = A_hat_grid[bx_clamped, by_clamped, bz_clamped]

    return A_hat, A_hat_grid


def compute_cell_activity(
    raw_values,
    A_hat: np.ndarray,
    cell_ids: np.ndarray,
    num_cells: int,
    time_chunk_size: int
) -> np.ndarray:
    """Compute corrected activity and aggregate per cell.

    Streams through time chunks to avoid loading full array.
    """
    num_pixels, num_timesteps = raw_values.shape

    # Precompute cell boundaries (cell_ids are sorted)
    cell_boundaries = np.searchsorted(cell_ids, np.arange(num_cells + 1))
    pixels_per_cell = np.diff(cell_boundaries).astype(np.float32)

    # Handle cells with zero pixels (shouldn't happen but be safe)
    pixels_per_cell = np.maximum(pixels_per_cell, 1)

    # Output array
    cell_activity = np.zeros((num_cells, num_timesteps), dtype=np.float32)

    num_chunks = (num_timesteps + time_chunk_size - 1) // time_chunk_size

    for i, t_start in enumerate(range(0, num_timesteps, time_chunk_size)):
        t_end = min(t_start + time_chunk_size, num_timesteps)
        print(f"  Processing timesteps [{t_start}, {t_end}) ({i+1}/{num_chunks})", flush=True)

        # Load time chunk
        F = raw_values[:, t_start:t_end].read().result().astype(np.float32)

        # Compute corrected activity: S = max(0, F - A_hat)
        S = np.maximum(0, F - A_hat[:, np.newaxis])

        # Aggregate per cell using reduceat
        cell_sums = np.add.reduceat(S, cell_boundaries[:-1], axis=0)
        cell_activity[:, t_start:t_end] = cell_sums / pixels_per_cell[:, np.newaxis]

    return cell_activity


def main():
    args = parse_args()

    print(f"Zarr: {args.zarr}", flush=True)

    # Load metadata
    metadata_path = os.path.join(args.zarr, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    num_pixels = metadata['num_pixels']
    num_timesteps = metadata['num_timesteps']
    print(f"Data: {num_pixels:,} pixels, {num_timesteps} timesteps", flush=True)

    # Open input arrays
    print("Opening input arrays...", flush=True)
    raw_values = open_array(args.zarr, 'raw_values')
    cell_ids = open_array(args.zarr, 'cell_ids').read().result()
    aligned_coords = open_array(args.zarr, 'aligned_coords').read().result()

    num_cells = int(cell_ids.max()) + 1
    print(f"Cells: {num_cells:,}", flush=True)

    # Volume shape (from init script constants)
    volume_shape = (2048, 1328, 72)

    # Step 1: Compute F0 (10th percentile per pixel)
    print("\nStep 1: Computing F0 (10th percentile)...", flush=True)
    F0 = compute_percentile_chunked(raw_values, PERCENTILE, PIXEL_CHUNK_SIZE)
    print(f"  F0 range: [{F0.min():.1f}, {F0.max():.1f}]", flush=True)

    # Step 2: Fit smooth spatial field
    print("\nStep 2: Fitting smooth spatial field...", flush=True)
    A_hat, A_hat_grid = fit_smooth_spatial_field(F0, aligned_coords, volume_shape)
    print(f"  A_hat range: [{A_hat.min():.1f}, {A_hat.max():.1f}]", flush=True)

    # Step 3: Compute corrected activity and aggregate per cell
    print("\nStep 3: Computing cell activity...", flush=True)
    cell_activity = compute_cell_activity(
        raw_values, A_hat, cell_ids, num_cells, TIME_CHUNK_SIZE
    )
    print(f"  cell_activity range: [{cell_activity.min():.1f}, {cell_activity.max():.1f}]", flush=True)

    # Write outputs to existing zarr
    print("\nWriting outputs...", flush=True)

    # baseline_F0
    print("  Writing baseline_F0...", flush=True)
    F0_arr = create_array(args.zarr, 'baseline_F0', (num_pixels,), (num_pixels,), 'float32')
    F0_arr.write(F0).result()

    # baseline_A_hat
    print("  Writing baseline_A_hat...", flush=True)
    A_hat_arr = create_array(args.zarr, 'baseline_A_hat', (num_pixels,), (num_pixels,), 'float32')
    A_hat_arr.write(A_hat).result()

    # baseline_A_hat_grid
    print("  Writing baseline_A_hat_grid...", flush=True)
    A_hat_grid_arr = create_array(
        args.zarr, 'baseline_A_hat_grid',
        A_hat_grid.shape, A_hat_grid.shape, 'float32'
    )
    A_hat_grid_arr.write(A_hat_grid).result()

    # cell_activity
    print("  Writing cell_activity...", flush=True)
    cell_activity_arr = create_array(
        args.zarr, 'cell_activity',
        (num_cells, num_timesteps), (num_cells, 100), 'float32'
    )
    cell_activity_arr.write(cell_activity).result()

    # Update metadata
    metadata['baseline_percentile'] = PERCENTILE
    metadata['baseline_bin_stride'] = [BIN_STRIDE_X, BIN_STRIDE_Y, BIN_STRIDE_Z]
    metadata['baseline_gaussian_sigma'] = GAUSSIAN_SIGMA
    metadata['num_cells'] = num_cells

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Added baseline_F0, baseline_A_hat, baseline_A_hat_grid, cell_activity to {args.zarr}", flush=True)


if __name__ == "__main__":
    main()
