#!/usr/bin/env python3
"""Initialize zarr for raw fluorescence extraction.

Creates output zarr structure and writes static arrays from segmentation.
Must be run once before submitting batch extraction jobs.

Usage:
    python processing/init_raw_fluorescence_zarr.py --output-zarr output.zarr
"""

import argparse
import json
import os

import numpy as np
import tensorstore as ts

from zarr_utils import (
    SIZE_T,
    STRIDE_X, STRIDE_Y, STRIDE_Z,
    PIXEL_CHUNK_SIZE,
    create_array,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize zarr for raw fluorescence extraction"
    )
    parser.add_argument(
        "--output-zarr",
        type=str,
        required=True,
        help="Path to output zarr directory",
    )
    parser.add_argument(
        "--gs-uri",
        type=str,
        default="gs://zapbench-release/volumes/20240930",
        help="GCS URI for source data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for time chunks (should match extraction job batch size)",
    )
    return parser.parse_args()


def load_segmentation(gs_uri: str) -> np.ndarray:
    """Load segmentation volume from GCS."""
    print(f"Loading segmentation from {gs_uri}/segmentation...", flush=True)
    ds = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': f'{gs_uri}/segmentation'
    }).result()
    segmentation = ds.read().result()
    print(f"  Shape: {segmentation.shape}, dtype: {segmentation.dtype}", flush=True)
    return segmentation


def extract_and_sort_coordinates(segmentation: np.ndarray):
    """Extract labeled voxel coordinates and sort by cell ID.

    Returns:
        xi, yi, zi: Aligned coordinates (sorted by cell_id)
        gx, gy, gz: Flow field grid coordinates (sorted by cell_id)
        cell_ids: Cell IDs (sorted, 0-indexed)
        num_cells: Number of cells
    """
    print("Extracting labeled voxel coordinates...", flush=True)
    xi, yi, zi = np.where(segmentation > 0)
    cell_ids = segmentation[xi, yi, zi].astype(np.uint64) - 1  # 0-indexed

    num_pixels = len(xi)
    num_cells = int(cell_ids.max() + 1)
    print(f"  Found {num_pixels:,} labeled pixels across {num_cells:,} cells", flush=True)

    print("Sorting by cell ID for contiguous grouping...", flush=True)
    sort_idx = np.argsort(cell_ids)
    xi = xi[sort_idx].astype(np.int32)
    yi = yi[sort_idx].astype(np.int32)
    zi = zi[sort_idx].astype(np.int32)
    cell_ids = cell_ids[sort_idx]

    print("Computing grid coordinates...", flush=True)
    gx = (xi // STRIDE_X).astype(np.int32)
    gy = (yi // STRIDE_Y).astype(np.int32)
    gz = (zi // STRIDE_Z).astype(np.int32)

    return xi, yi, zi, gx, gy, gz, cell_ids, num_cells


def create_output_zarr(output_path: str, num_pixels: int, num_cells: int, batch_size: int):
    """Create output zarr with proper structure and chunking."""
    print(f"Creating output zarr at {output_path}...", flush=True)
    os.makedirs(output_path, exist_ok=True)

    arrays = {}

    # Time-varying arrays: chunk along both dimensions
    pixel_chunk = min(PIXEL_CHUNK_SIZE, num_pixels)

    print(f"  Creating raw_values [{num_pixels}, {SIZE_T}], chunks=[{pixel_chunk}, {batch_size}]", flush=True)
    arrays['raw_values'] = create_array(
        output_path, 'raw_values',
        shape=(num_pixels, SIZE_T),
        chunks=(pixel_chunk, batch_size),
        dtype='uint16',
    )

    print(f"  Creating raw_z [{num_pixels}, {SIZE_T}], chunks=[{pixel_chunk}, {batch_size}]", flush=True)
    arrays['raw_z'] = create_array(
        output_path, 'raw_z',
        shape=(num_pixels, SIZE_T),
        chunks=(pixel_chunk, batch_size),
        dtype='int8',
    )

    print(f"  Creating acquisition_time_ms [{num_pixels}, {SIZE_T}], chunks=[{pixel_chunk}, {batch_size}]", flush=True)
    arrays['acquisition_time_ms'] = create_array(
        output_path, 'acquisition_time_ms',
        shape=(num_pixels, SIZE_T),
        chunks=(pixel_chunk, batch_size),
        dtype='uint16',
    )

    # Static arrays: single chunk
    print(f"  Creating cell_ids [{num_pixels}]", flush=True)
    arrays['cell_ids'] = create_array(
        output_path, 'cell_ids',
        shape=(num_pixels,),
        chunks=(num_pixels,),
        dtype='uint64',
    )

    print(f"  Creating aligned_coords [{num_pixels}, 3]", flush=True)
    arrays['aligned_coords'] = create_array(
        output_path, 'aligned_coords',
        shape=(num_pixels, 3),
        chunks=(num_pixels, 3),
        dtype='int32',
    )

    print(f"  Creating grid_coords [{num_pixels}, 3]", flush=True)
    arrays['grid_coords'] = create_array(
        output_path, 'grid_coords',
        shape=(num_pixels, 3),
        chunks=(num_pixels, 3),
        dtype='int32',
    )

    # Cell index for efficient per-cell access
    print(f"  Creating cell_pixel_boundaries [{num_cells + 1}]", flush=True)
    arrays['cell_pixel_boundaries'] = create_array(
        output_path, 'cell_pixel_boundaries',
        shape=(num_cells + 1,),
        chunks=(num_cells + 1,),
        dtype='uint64',
    )

    return arrays


def write_metadata(output_path: str, gs_uri: str, batch_size: int, num_pixels: int, num_cells: int, pixel_chunk: int):
    """Write metadata JSON file."""
    metadata = {
        'gs_uri': gs_uri,
        'batch_size': batch_size,
        'num_pixels': num_pixels,
        'num_cells': num_cells,
        'num_timesteps': SIZE_T,
        'pixel_chunk_size': pixel_chunk,
        'stride_x': STRIDE_X,
        'stride_y': STRIDE_Y,
        'stride_z': STRIDE_Z,
    }
    metadata_path = os.path.join(output_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote metadata to {metadata_path}", flush=True)


def write_static_arrays(arrays, xi, yi, zi, gx, gy, gz, cell_ids, num_cells):
    """Write static arrays to zarr."""
    print("Writing static arrays...", flush=True)

    print("  Writing cell_ids...", flush=True)
    arrays['cell_ids'].write(cell_ids).result()

    print("  Writing aligned_coords...", flush=True)
    aligned_coords = np.stack([xi, yi, zi], axis=1)
    arrays['aligned_coords'].write(aligned_coords).result()

    print("  Writing grid_coords...", flush=True)
    grid_coords = np.stack([gx, gy, gz], axis=1)
    arrays['grid_coords'].write(grid_coords).result()

    print("  Writing cell_pixel_boundaries...", flush=True)
    # Compute boundaries: cell i spans pixels[boundaries[i]:boundaries[i+1]]
    cell_pixel_boundaries = np.searchsorted(cell_ids, np.arange(num_cells + 1)).astype(np.uint64)
    arrays['cell_pixel_boundaries'].write(cell_pixel_boundaries).result()

    print("Static arrays written successfully.", flush=True)


def main():
    args = parse_args()

    # Load segmentation
    segmentation = load_segmentation(args.gs_uri)

    # Extract and sort coordinates
    xi, yi, zi, gx, gy, gz, cell_ids, num_cells = extract_and_sort_coordinates(segmentation)
    num_pixels = len(xi)

    # Create output zarr
    arrays = create_output_zarr(args.output_zarr, num_pixels, num_cells, args.batch_size)
    pixel_chunk = min(PIXEL_CHUNK_SIZE, num_pixels)

    # Write metadata
    write_metadata(args.output_zarr, args.gs_uri, args.batch_size, num_pixels, num_cells, pixel_chunk)

    # Write static arrays
    write_static_arrays(arrays, xi, yi, zi, gx, gy, gz, cell_ids, num_cells)

    print(f"\nInitialization complete: {args.output_zarr}", flush=True)
    print(f"  {num_pixels:,} pixels across {num_cells:,} cells, {SIZE_T} timesteps", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"\nNext step: Submit batch extraction jobs with:", flush=True)
    print(f"  python extract_raw_fluorescence.py --start-t <start> --end-t <end> --output-zarr {args.output_zarr}", flush=True)


if __name__ == "__main__":
    main()
