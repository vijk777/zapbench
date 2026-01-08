#!/usr/bin/env python3
"""Initialize zarr for raw fluorescence extraction.

Creates the output zarr structure and writes static arrays (cell_ids, aligned_coords,
grid_coords). Must be run once before submitting batch extraction jobs.

Usage:
    python init_raw_fluorescence_zarr.py --output-zarr /path/to/output.zarr

    # With custom source:
    python init_raw_fluorescence_zarr.py --output-zarr /path/to/output.zarr \
        --gs-uri gs://zapbench-release/volumes/20240930

    # With custom batch size (affects chunking):
    python init_raw_fluorescence_zarr.py --output-zarr /path/to/output.zarr --batch-size 100
"""

import argparse
import json
import os
import numpy as np
import tensorstore as ts


# Volume dimensions
SIZE_X, SIZE_Y, SIZE_Z, SIZE_T = 2048, 1328, 72, 7879

# Flow field grid strides (aligned space pixels per grid point)
STRIDE_X = 16
STRIDE_Y = 16
STRIDE_Z = 2

# Chunk size along pixel dimension (must keep chunk < 2GB for Blosc)
# 1M pixels * 100 timesteps * 2 bytes = 200MB per chunk
PIXEL_CHUNK_SIZE = 1_000_000


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
    """
    print("Extracting labeled voxel coordinates...", flush=True)
    xi, yi, zi = np.where(segmentation > 0)
    cell_ids = segmentation[xi, yi, zi].astype(np.uint64) - 1  # 0-indexed

    num_pixels = len(xi)
    num_cells = cell_ids.max() + 1
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

    return xi, yi, zi, gx, gy, gz, cell_ids


def create_tensorstore_array(path: str, name: str, shape: tuple, chunks: tuple, dtype: str):
    """Create a zarr3 array using tensorstore."""
    spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(path, name),
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


def create_output_zarr(output_path: str, num_pixels: int, batch_size: int):
    """Create output zarr with proper structure and chunking."""
    print(f"Creating output zarr at {output_path}...", flush=True)
    os.makedirs(output_path, exist_ok=True)

    arrays = {}

    # Time-varying arrays: chunk along both dimensions
    # Pixel chunks must be small enough to stay under Blosc 2GB limit
    pixel_chunk = min(PIXEL_CHUNK_SIZE, num_pixels)

    print(f"  Creating raw_values [{num_pixels}, {SIZE_T}], chunks=[{pixel_chunk}, {batch_size}]", flush=True)
    arrays['raw_values'] = create_tensorstore_array(
        output_path, 'raw_values',
        shape=(num_pixels, SIZE_T),
        chunks=(pixel_chunk, batch_size),
        dtype='uint16',
    )

    print(f"  Creating raw_z [{num_pixels}, {SIZE_T}], chunks=[{pixel_chunk}, {batch_size}]", flush=True)
    arrays['raw_z'] = create_tensorstore_array(
        output_path, 'raw_z',
        shape=(num_pixels, SIZE_T),
        chunks=(pixel_chunk, batch_size),
        dtype='int16',
    )

    print(f"  Creating acquisition_time [{num_pixels}, {SIZE_T}], chunks=[{pixel_chunk}, {batch_size}]", flush=True)
    arrays['acquisition_time'] = create_tensorstore_array(
        output_path, 'acquisition_time',
        shape=(num_pixels, SIZE_T),
        chunks=(pixel_chunk, batch_size),
        dtype='uint32',
    )

    # Static arrays: single chunk
    print(f"  Creating cell_ids [{num_pixels}]", flush=True)
    arrays['cell_ids'] = create_tensorstore_array(
        output_path, 'cell_ids',
        shape=(num_pixels,),
        chunks=(num_pixels,),
        dtype='uint64',
    )

    print(f"  Creating aligned_coords [{num_pixels}, 3]", flush=True)
    arrays['aligned_coords'] = create_tensorstore_array(
        output_path, 'aligned_coords',
        shape=(num_pixels, 3),
        chunks=(num_pixels, 3),
        dtype='int32',
    )

    print(f"  Creating grid_coords [{num_pixels}, 3]", flush=True)
    arrays['grid_coords'] = create_tensorstore_array(
        output_path, 'grid_coords',
        shape=(num_pixels, 3),
        chunks=(num_pixels, 3),
        dtype='int32',
    )

    return arrays


def write_metadata(output_path: str, gs_uri: str, batch_size: int, num_pixels: int, pixel_chunk: int):
    """Write metadata JSON file."""
    metadata = {
        'gs_uri': gs_uri,
        'batch_size': batch_size,
        'num_pixels': num_pixels,
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


def write_static_arrays(arrays, xi, yi, zi, gx, gy, gz, cell_ids):
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

    print("Static arrays written successfully.", flush=True)


def main():
    args = parse_args()

    # Load segmentation
    segmentation = load_segmentation(args.gs_uri)

    # Extract and sort coordinates
    xi, yi, zi, gx, gy, gz, cell_ids = extract_and_sort_coordinates(segmentation)
    num_pixels = len(xi)

    # Create output zarr
    arrays = create_output_zarr(args.output_zarr, num_pixels, args.batch_size)
    pixel_chunk = min(PIXEL_CHUNK_SIZE, num_pixels)

    # Write metadata
    write_metadata(args.output_zarr, args.gs_uri, args.batch_size, num_pixels, pixel_chunk)

    # Write static arrays
    write_static_arrays(arrays, xi, yi, zi, gx, gy, gz, cell_ids)

    print(f"\nInitialization complete: {args.output_zarr}", flush=True)
    print(f"  {num_pixels:,} pixels, {SIZE_T} timesteps", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"\nNext step: Submit batch extraction jobs with:", flush=True)
    print(f"  python extract_raw_fluorescence.py --start-t <start> --end-t <end> --output-zarr {args.output_zarr}", flush=True)


if __name__ == "__main__":
    main()
