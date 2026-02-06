#!/usr/bin/env python3
"""Compute per-cell ephys sample indices using flow-corrected z-coordinates.

For each cell and timestep, computes the ephys sample index by:
1. Computing the cell's centroid from segmentation
2. Applying flow field correction to get the corrected z-coordinate
3. Looking up the ephys sample index from imaging_sample_index

The output is appended to ephys.zarr as cell_ephys_index array.

Usage:
    python processing/compute_cell_ephys_index.py
"""

import json
import os
import time

import numpy as np
np.seterr(all='raise')
import tensorstore as ts
from scipy.ndimage import map_coordinates

# Flow field grid strides (aligned space pixels per grid point)
STRIDE_X = 16
STRIDE_Y = 16
STRIDE_Z = 2

# GCS URI for source data
GS_URI = "gs://zapbench-release/volumes/20240930"

# Local ephys zarr path
EPHYS_ZARR = "ephys.zarr"

# Chunk size for output array (match traces format)
CHUNK_SIZE = 512

# Retry parameters for GCS reads
MAX_RETRIES = 5
TIMEOUT_S = 30


def read_with_retry(store, label=""):
    """Read from tensorstore with retry logic for GCS timeouts."""
    for attempt in range(MAX_RETRIES):
        future = store.read()
        try:
            return future.result(timeout=TIMEOUT_S)
        except TimeoutError:
            print(f"    {label} timeout (attempt {attempt + 1}/{MAX_RETRIES}), retrying...", flush=True)
            time.sleep(2 ** attempt)
    raise TimeoutError(f"{label} failed after {MAX_RETRIES} attempts")


def load_segmentation() -> np.ndarray:
    """Load segmentation volume from GCS."""
    print(f"Loading segmentation from {GS_URI}/segmentation...", flush=True)
    ds = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': f'{GS_URI}/segmentation'
    }).result()
    segmentation = read_with_retry(ds, label="segmentation")
    print(f"  Shape: {segmentation.shape}, dtype: {segmentation.dtype}", flush=True)
    return segmentation


def compute_cell_centroids(segmentation: np.ndarray):
    """Compute centroid (mean x, y, z) for each cell.

    Returns:
        centroids: Array of shape (num_cells, 3) with (x, y, z) centroids
        num_cells: Number of cells
    """
    print("Computing cell centroids...", flush=True)

    # Extract labeled voxel coordinates
    xi, yi, zi = np.where(segmentation > 0)
    cell_ids = segmentation[xi, yi, zi].astype(np.int64) - 1  # 0-indexed

    num_pixels = len(xi)
    num_cells = int(cell_ids.max() + 1)
    print(f"  Found {num_pixels:,} labeled pixels across {num_cells:,} cells", flush=True)

    # Compute centroids using bincount
    centroids = np.zeros((num_cells, 3), dtype=np.float64)
    counts = np.bincount(cell_ids, minlength=num_cells).astype(np.float64)

    centroids[:, 0] = np.bincount(cell_ids, weights=xi.astype(np.float64), minlength=num_cells) / counts
    centroids[:, 1] = np.bincount(cell_ids, weights=yi.astype(np.float64), minlength=num_cells) / counts
    centroids[:, 2] = np.bincount(cell_ids, weights=zi.astype(np.float64), minlength=num_cells) / counts

    print(f"  Centroid z range: [{centroids[:, 2].min():.1f}, {centroids[:, 2].max():.1f}]", flush=True)

    return centroids, num_cells


def open_flow_fields():
    """Open flow fields array from GCS and return store with chunk info."""
    print(f"Opening flow fields from {GS_URI}/flow_fields...", flush=True)
    ds = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': f'{GS_URI}/flow_fields'
    }).result()

    # Read chunk shape from zarr metadata
    kvstore = ts.KvStore.open(f'{GS_URI}/flow_fields/').result()
    zarr_json_bytes = kvstore.read('zarr.json').result().value
    zarr_meta = json.loads(zarr_json_bytes)
    time_chunk = zarr_meta['chunk_grid']['configuration']['chunk_shape'][-1]

    print(f"  Shape: {ds.shape}, time chunk: {time_chunk}", flush=True)
    return ds, time_chunk


def load_imaging_sample_index() -> np.ndarray:
    """Load imaging_sample_index from ephys zarr."""
    print(f"Loading imaging_sample_index from {EPHYS_ZARR}...", flush=True)
    ds = ts.open({
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(EPHYS_ZARR, 'imaging_sample_index'),
        },
        'open': True,
    }).result()
    data = ds.read().result()
    print(f"  Shape: {data.shape}", flush=True)
    return data


def create_output_array(num_timesteps: int, num_cells: int):
    """Create cell_ephys_index output array."""
    print(f"Creating cell_ephys_index array [{num_timesteps}, {num_cells}]...", flush=True)

    spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(EPHYS_ZARR, 'cell_ephys_index'),
        },
        'metadata': {
            'shape': [num_timesteps, num_cells],
            'chunk_grid': {
                'name': 'regular',
                'configuration': {
                    'chunk_shape': [CHUNK_SIZE, CHUNK_SIZE],
                },
            },
            'chunk_key_encoding': {'name': 'default'},
            'data_type': 'int32',
            'codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}}],
        },
        'create': True,
        'delete_existing': True,
    }
    return ts.open(spec).result()


def update_metadata(num_cells: int):
    """Update ephys zarr metadata with cell_ephys_index info."""
    zarr_json_path = os.path.join(EPHYS_ZARR, 'zarr.json')
    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)

    metadata['attributes']['cell_ephys_index'] = {
        'num_cells': num_cells,
        'description': 'Ephys sample index for each cell at each timepoint, computed from flow-corrected cell centroids',
    }

    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated {zarr_json_path}", flush=True)


def main():
    # Load segmentation and compute cell centroids
    segmentation = load_segmentation()
    centroids, num_cells = compute_cell_centroids(segmentation)
    del segmentation  # Free memory

    # Precompute flow field grid coordinates for centroids
    # Flow field interpolation uses [z/STRIDE_Z, y/STRIDE_Y, x/STRIDE_X] ordering
    grid_coords = np.array([
        centroids[:, 2] / STRIDE_Z,  # z
        centroids[:, 1] / STRIDE_Y,  # y
        centroids[:, 0] / STRIDE_X,  # x
    ])

    # Load imaging_sample_index
    imaging_sample_index = load_imaging_sample_index()
    num_timesteps, num_z = imaging_sample_index.shape

    # Open flow fields
    ds_flow, time_chunk = open_flow_fields()

    # Create output array
    out_array = create_output_array(num_timesteps, num_cells)

    # Process in batches aligned with flow field chunks
    print(f"\nProcessing {num_timesteps} timesteps in chunks of {time_chunk}...", flush=True)
    start_time = time.time()

    # Allocate output buffer
    cell_ephys_index = np.empty((num_timesteps, num_cells), dtype=np.int32)

    num_chunks = (num_timesteps + time_chunk - 1) // time_chunk
    for chunk_idx in range(num_chunks):
        t_start = chunk_idx * time_chunk
        t_end = min(t_start + time_chunk, num_timesteps)

        elapsed = time.time() - start_time
        rate = t_start / elapsed if elapsed > 0 else 0
        eta = (num_timesteps - t_start) / rate if rate > 0 else 0
        print(f"  chunk {chunk_idx+1}/{num_chunks}: t=[{t_start}, {t_end}) "
              f"({rate:.1f} t/s, ETA: {eta/60:.1f} min)", flush=True)

        # Read flow field chunk: shape [3, fz, fy, fx, chunk_size]
        flow_chunk = read_with_retry(ds_flow[:, :, :, :, t_start:t_end], label=f"flow chunk {chunk_idx}")

        for i, t in enumerate(range(t_start, t_end)):
            # Interpolate z-offset at cell centroids using cubic spline
            # Flow field component 2 is the z-offset
            z_offset = map_coordinates(flow_chunk[2, :, :, :, i], grid_coords, order=3, mode='nearest')

            # Compute corrected z: centroid_z + z_offset/4.0
            # (z_offset is scaled by 4x relative to aligned space)
            corrected_z = centroids[:, 2] + z_offset / 4.0

            # Verify corrected z is in valid range
            assert (corrected_z >= 0).all() and (corrected_z < 72).all(), \
                f"corrected_z out of range [0, 72) at t={t}: min={corrected_z.min()}, max={corrected_z.max()}"

            # Round to nearest integer for indexing
            corrected_z_int = np.round(corrected_z).astype(np.int32)

            # Look up ephys sample index for each cell
            cell_ephys_index[t, :] = imaging_sample_index[t, corrected_z_int]

    elapsed = time.time() - start_time
    print(f"\nProcessed {num_timesteps} timesteps in {elapsed/60:.1f} minutes", flush=True)

    # Write output
    print("Writing cell_ephys_index to zarr...", flush=True)
    out_array.write(cell_ephys_index).result()

    # Update metadata
    update_metadata(num_cells)

    print(f"\nDone. Added cell_ephys_index to {EPHYS_ZARR}", flush=True)
    print(f"  Shape: [{num_timesteps}, {num_cells}]", flush=True)
    print(f"  Sample index range: [{cell_ephys_index.min()}, {cell_ephys_index.max()}]", flush=True)


if __name__ == "__main__":
    main()
