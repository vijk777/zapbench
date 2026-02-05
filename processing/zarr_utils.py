"""Shared utilities for zarr-based fluorescence processing scripts.

Provides common functions for opening/creating zarr arrays, loading metadata,
and computing cell aggregation boundaries.
"""

import json
import os
from typing import Tuple

import numpy as np
import tensorstore as ts

# Volume dimensions
SIZE_X, SIZE_Y, SIZE_Z, SIZE_T = 2048, 1328, 72, 7879
VOLUME_SHAPE = (SIZE_X, SIZE_Y, SIZE_Z)

# Flow field grid strides (aligned space pixels per grid point)
STRIDE_X = 16
STRIDE_Y = 16
STRIDE_Z = 2

# Chunk size along pixel dimension (must keep chunk < 2GB for Blosc)
PIXEL_CHUNK_SIZE = 1_000_000

# Default time chunk size for streaming operations
TIME_CHUNK_SIZE = 100

# Cell activity processing parameters
CELL_ACTIVITY_CELLS_PER_CHUNK = 1000
CELL_ACTIVITY_PERCENTILE = 8
CELL_ACTIVITY_WINDOW_RADIUS = 400  # ±400 frames (~6 minutes) for rolling baseline
CELL_ACTIVITY_CLIP_MIN = -0.25
CELL_ACTIVITY_CLIP_MAX = 1.5


def set_thread_limits(n_threads: int = 2) -> None:
    """Set environment variables to limit thread usage on cluster."""
    os.environ['BLOSC_NTHREADS'] = str(n_threads)
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_MAX_THREADS'] = str(n_threads)


def open_array(zarr_path: str, name: str) -> ts.TensorStore:
    """Open a zarr3 array for reading."""
    return ts.open({
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(zarr_path, name),
        },
        'open': True,
    }).result()


def create_array(
    zarr_path: str,
    name: str,
    shape: Tuple[int, ...],
    chunks: Tuple[int, ...],
    dtype: str
) -> ts.TensorStore:
    """Create a zarr3 array with standard compression settings."""
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


def load_metadata(zarr_path: str) -> dict:
    """Load metadata.json from zarr directory."""
    metadata_path = os.path.join(zarr_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            "Run init_raw_fluorescence_zarr.py first."
        )
    with open(metadata_path, 'r') as f:
        return json.load(f)


def save_metadata(zarr_path: str, metadata: dict) -> None:
    """Save metadata.json to zarr directory."""
    metadata_path = os.path.join(zarr_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def compute_cell_boundaries(
    cell_ids: np.ndarray,
    num_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cell boundaries and pixel counts for sorted cell_ids.

    Args:
        cell_ids: Sorted array of cell IDs for each pixel
        num_cells: Total number of cells

    Returns:
        cell_boundaries: Array of shape [num_cells + 1] with start indices
        pixels_per_cell: Array of shape [num_cells] with pixel counts (min 1)
    """
    cell_boundaries = np.searchsorted(cell_ids, np.arange(num_cells + 1))
    pixels_per_cell = np.diff(cell_boundaries).astype(np.float64)
    pixels_per_cell = np.maximum(pixels_per_cell, 1)  # Avoid division by zero
    return cell_boundaries, pixels_per_cell
