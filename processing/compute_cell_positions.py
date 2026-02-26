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

    # Read zarr metadata for any pixel size info
    kvstore = ts.KvStore.open(f'{GS_URI}/segmentation/').result()
    zarr_json_bytes = kvstore.read('zarr.json').result().value
    zarr_meta = json.loads(zarr_json_bytes)

    segmentation = read_with_retry(ds, label="segmentation")
    print(f"  Shape: {segmentation.shape}, dtype: {segmentation.dtype}", flush=True)
    return segmentation, zarr_meta


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


def write_zarr(centroids, num_cells, volume_shape, zarr_meta):
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

    # Propagate any pixel size metadata from the segmentation zarr
    seg_attrs = zarr_meta.get('attributes', {})
    for key in ('pixel_size', 'pixel_size_um', 'voxel_size', 'resolution',
                'pixel_resolution', 'scale'):
        if key in seg_attrs:
            meta['attributes'][key] = seg_attrs[key]

    with open(zarr_json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Shape: [{num_cells}, 3]", flush=True)
    print(f"  Volume shape (x, y, z): {list(volume_shape)}", flush=True)


def main():
    segmentation, zarr_meta = load_segmentation()
    volume_shape = segmentation.shape  # (X, Y, Z)

    centroids, num_cells = compute_centroids(segmentation)
    del segmentation

    write_zarr(centroids, num_cells, volume_shape, zarr_meta)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
