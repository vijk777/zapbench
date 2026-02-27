#!/usr/bin/env python3
"""Compute cell centroid positions in pixel coordinates.

Loads the 3D segmentation mask and computes the centroid [x, y, z] for each
cell. The result is written to cell_position_xyz.zarr as a single zarr v3
array of shape [num_cells, 3] (float32).

Usage:
    python processing/compute_cell_positions.py
"""

import json
import os
import time

import numpy as np
import tensorstore as ts

# GCS URI for source data
GS_URI = "gs://zapbench-release/volumes/20240930"

# Output path
OUTPUT_ZARR = "cell_position_xyz.zarr"

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
            print(f"  {label} timeout (attempt {attempt + 1}/{MAX_RETRIES}), retrying...", flush=True)
            time.sleep(2 ** attempt)
    raise TimeoutError(f"{label} failed after {MAX_RETRIES} attempts")


def load_segmentation():
    """Load segmentation volume from GCS. Returns array and its shape."""
    print(f"Loading segmentation from {GS_URI}/segmentation...", flush=True)
    ds = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': f'{GS_URI}/segmentation',
    }).result()

    # Read raw volume zarr metadata for voxel size info
    kvstore = ts.KvStore.open(f'{GS_URI}/raw/').result()
    zarr_json_bytes = kvstore.read('zarr.json').result().value
    raw_meta = json.loads(zarr_json_bytes)

    segmentation = read_with_retry(ds, label="segmentation")
    print(f"  Shape: {segmentation.shape}, dtype: {segmentation.dtype}", flush=True)
    return segmentation, raw_meta


def compute_centroids(segmentation):
    """Compute centroid [x, y, z] for each cell.

    Args:
        segmentation: 3D array (X, Y, Z) with 1-indexed cell IDs.

    Returns:
        centroids: float32 array of shape [num_cells, 3].
        num_cells: int.
    """
    print("Computing cell centroids...", flush=True)

    xi, yi, zi = np.where(segmentation > 0)
    cell_ids = segmentation[xi, yi, zi].astype(np.int64) - 1  # 0-indexed

    num_pixels = len(xi)
    num_cells = int(cell_ids.max() + 1)
    print(f"  {num_pixels:,} labeled voxels across {num_cells:,} cells", flush=True)

    counts = np.bincount(cell_ids, minlength=num_cells).astype(np.float64)
    centroids = np.zeros((num_cells, 3), dtype=np.float64)
    centroids[:, 0] = np.bincount(cell_ids, weights=xi.astype(np.float64), minlength=num_cells) / counts
    centroids[:, 1] = np.bincount(cell_ids, weights=yi.astype(np.float64), minlength=num_cells) / counts
    centroids[:, 2] = np.bincount(cell_ids, weights=zi.astype(np.float64), minlength=num_cells) / counts

    print(f"  x range: [{centroids[:, 0].min():.1f}, {centroids[:, 0].max():.1f}]", flush=True)
    print(f"  y range: [{centroids[:, 1].min():.1f}, {centroids[:, 1].max():.1f}]", flush=True)
    print(f"  z range: [{centroids[:, 2].min():.1f}, {centroids[:, 2].max():.1f}]", flush=True)

    return centroids.astype(np.float32), num_cells


def parse_voxel_size(raw_meta):
    """Extract voxel size [x, y, z] in nm from raw volume zarr metadata.

    The raw zarr.json stores dimension_units as e.g. ["406 nm", "406 nm", "4000 nm", "0.9141 s"]
    for dimensions [x, y, z, t]. We parse the spatial dimensions (first 3).
    """
    dimension_units = raw_meta.get('attributes', {}).get('dimension_units', [])
    voxel_size_nm = []
    for unit_str in dimension_units[:3]:  # x, y, z only
        parts = unit_str.strip().split()
        value = float(parts[0])
        unit = parts[1] if len(parts) > 1 else 'nm'
        # Normalize to nanometers
        if unit == 'nm':
            voxel_size_nm.append(value)
        elif unit == 'um' or unit == 'µm':
            voxel_size_nm.append(value * 1000)
        elif unit == 'mm':
            voxel_size_nm.append(value * 1e6)
        else:
            voxel_size_nm.append(value)  # assume nm
    return voxel_size_nm


def write_zarr(centroids, num_cells, volume_shape, raw_meta):
    """Write centroids to a zarr v3 store with metadata."""
    print(f"Writing {OUTPUT_ZARR}...", flush=True)

    # Create the array — single chunk for the whole thing
    spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': OUTPUT_ZARR,
        },
        'metadata': {
            'shape': [num_cells, 3],
            'chunk_grid': {
                'name': 'regular',
                'configuration': {
                    'chunk_shape': [num_cells, 3],
                },
            },
            'data_type': 'float32',
            'codecs': [
                {'name': 'bytes', 'configuration': {'endian': 'little'}},
            ],
        },
        'create': True,
        'delete_existing': True,
    }
    store = ts.open(spec).result()
    store.write(centroids).result()

    # Write metadata into zarr.json attributes
    zarr_json_path = os.path.join(OUTPUT_ZARR, 'zarr.json')
    with open(zarr_json_path, 'r') as f:
        meta = json.load(f)

    meta['attributes'] = {
        'description': 'Cell centroid positions in pixel coordinates [x, y, z]',
        'axes': ['x', 'y', 'z'],
        'num_cells': num_cells,
        'volume_shape_xyz': list(int(s) for s in volume_shape),
    }

    # Add voxel size from raw volume metadata
    voxel_size_nm = parse_voxel_size(raw_meta)
    if voxel_size_nm:
        meta['attributes']['voxel_size_xyz_nm'] = voxel_size_nm
        print(f"  Voxel size (x, y, z): {voxel_size_nm} nm", flush=True)

    with open(zarr_json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Shape: [{num_cells}, 3]", flush=True)
    print(f"  Volume shape (x, y, z): {list(volume_shape)}", flush=True)


def main():
    segmentation, raw_meta = load_segmentation()
    volume_shape = segmentation.shape  # (X, Y, Z)

    centroids, num_cells = compute_centroids(segmentation)
    del segmentation

    write_zarr(centroids, num_cells, volume_shape, raw_meta)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
