#!/usr/bin/env python3
"""Convert ephys binary file to flat zarr v3 format with imaging alignment."""

import io
import json
import os

import numpy as np
import scipy.signal
from scipy.spatial import KDTree
import tensorstore as ts

from zapbench.constants import (
    EPHYS_COLUMNS,
    EPHYS_SAMPLING_FREQUENCY_HZ,
)

# Source channel index for trial_id in raw data
TRIAL_ID_CHANNEL = 4

# Chunk size for zarr arrays
CHUNK_SIZE = 1_000_000

# Imaging parameters
NUM_TIMEPOINTS = 7879
NUM_Z_SLICES = 72
SAMPLES_PER_Z = 72
SAMPLES_PER_VOLUME = 5484  # = 6kHz x 914ms

# TTL peak detection parameters
TTL_MIN_DISTANCE = 500
TTL_MIN_HEIGHT = 3.55


def load_ephys(file_handle, num_channels=10):
    """Load ephys data from binary file."""
    try:
        data = np.fromfile(file_handle, dtype=np.float32)
    except io.UnsupportedOperation:
        data = np.frombuffer(file_handle.read(), dtype=np.float32)
    if data.size % num_channels:
        raise ValueError(f'Data does not fit in num_channels: {num_channels}')
    return data.reshape((-1, num_channels))


def compute_imaging_sample_index(ttl_data):
    """Compute ephys sample indices for each imaging frame (t, z).

    Uses high TTL peaks as volume sync markers and low TTL peaks for z-plane
    positions. The 72 low peaks per volume correspond to z=0 through z=71.
    Falls back to linear extrapolation for truncated/missing data at the end.

    Note: The ephys trace is truncated mid-volume at t=7871, so indices for
    z>=57 of t=7871 and all of t=7872-7878 may exceed the actual data bounds.

    Args:
        ttl_data: 1D array of TTL signal values.

    Returns:
        Tuple of:
        - imaging_sample_index: 2D array of shape (NUM_TIMEPOINTS, NUM_Z_SLICES)
          containing the ephys sample index corresponding to each imaging frame.
        - high_peaks: 1D array of detected high peak sample indices (volume sync).
    """
    # Find high peaks (volume sync markers)
    high_peaks, _ = scipy.signal.find_peaks(
        ttl_data, distance=TTL_MIN_DISTANCE, height=TTL_MIN_HEIGHT
    )

    # Find low peaks (z-plane markers, 72 per volume for z=0 to z=71)
    low_peaks_all, _ = scipy.signal.find_peaks(ttl_data, distance=50, height=1)

    # Remove low peaks within 5 samples of high peaks using KDTree
    # (high peaks also trigger as low peaks due to lower threshold)
    tree = KDTree(low_peaks_all.reshape(-1, 1))
    inds = tree.query_ball_point(high_peaks.reshape(-1, 1), r=5)
    select = np.ones(len(low_peaks_all), dtype=bool)
    for i in inds:
        for j in i:
            select[j] = False
    low_peaks = low_peaks_all[select]

    print(f'  Found {len(high_peaks)} high peaks (volume sync)')
    print(f'  Found {len(low_peaks)} low peaks (z-planes)')

    # Build (timepoints, z_slices) index array
    imaging_sample_index = np.full((NUM_TIMEPOINTS, NUM_Z_SLICES), -1, dtype=np.int32)

    flat_index = imaging_sample_index.ravel()
    flat_index[:len(low_peaks)] = low_peaks

    incomplete = (imaging_sample_index < 0).any(axis=1)
    assert incomplete.sum() == 8 # 1 partial and 7 missing time points

    num_complete = len(high_peaks) - 1
    assert (imaging_sample_index[:num_complete] > 0).all()
    time_delta_per_z = np.diff(imaging_sample_index[:num_complete], axis=0).mean(axis=0)
    last_row = imaging_sample_index[num_complete-1]
    extrapolated = np.where(
        imaging_sample_index[num_complete:] >= 0,
        imaging_sample_index[num_complete:],
        np.round(np.stack(
        [
            last_row + (i+1)*time_delta_per_z for i in range(NUM_TIMEPOINTS-num_complete)
        ], axis=0
    )).astype(np.int32))
    imaging_sample_index[num_complete:] = extrapolated

    assert (imaging_sample_index >= 0).all()

    return imaging_sample_index, high_peaks


def write_1d_array(output_path, name, data, dtype):
    """Write a 1D array to zarr format."""
    data = np.asarray(data).astype(dtype)
    n_rows = len(data)
    chunk_size = min(CHUNK_SIZE, n_rows)

    spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(output_path, name),
        },
        'metadata': {
            'shape': [n_rows],
            'chunk_grid': {
                'name': 'regular',
                'configuration': {
                    'chunk_shape': [chunk_size],
                },
            },
            'chunk_key_encoding': {'name': 'default'},
            'data_type': dtype,
            'codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}}],
        },
        'create': True,
        'delete_existing': True,
    }
    store = ts.open(spec).result()
    store[:] = data


def write_2d_array(output_path, name, data, dtype):
    """Write a 2D array to zarr format."""
    data = np.asarray(data).astype(dtype)
    shape = data.shape
    # Use reasonable chunk sizes for 2D data
    chunk_shape = [min(1000, shape[0]), shape[1]]

    spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(output_path, name),
        },
        'metadata': {
            'shape': list(shape),
            'chunk_grid': {
                'name': 'regular',
                'configuration': {
                    'chunk_shape': chunk_shape,
                },
            },
            'chunk_key_encoding': {'name': 'default'},
            'data_type': dtype,
            'codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}}],
        },
        'create': True,
        'delete_existing': True,
    }
    store = ts.open(spec).result()
    store[:] = data


def pad_array(arr, padded_size, pad_value):
    """Pad a 1D array to the specified size with a constant value."""
    if len(arr) >= padded_size:
        return arr
    padding = np.full(padded_size - len(arr), pad_value, dtype=arr.dtype)
    return np.concatenate([arr, padding])


def main():
    input_path = '/groups/saalfeld/saalfeldlab/zapbench-release/volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt'
    output_path = 'ephys.zarr'

    print(f'Loading data from {input_path}')
    with open(input_path, 'rb') as f:
        data = load_ephys(f)

    valid_samples = data.shape[0]
    print(f'Raw data shape: {data.shape}')
    print(f'Number of valid samples: {valid_samples}')

    # Extract condition column
    # Raw trial_id is shifted: trial_id=1 is gain (condition 0), trial_id=0 is dark (condition 8)
    # Transform: condition = (trial_id - 1) % 9
    raw_trial_ids = data[:, TRIAL_ID_CHANNEL].astype(np.int8)
    condition_ids = ((raw_trial_ids - 1) % 9).astype(np.int8)

    # Compute imaging sample index from TTL channel
    print('Computing imaging sample index from TTL...')
    ttl_channel_idx = next(
        col['source_channel'] for col in EPHYS_COLUMNS if col['name'] == 'ttl'
    )
    ttl_data = data[:, ttl_channel_idx]
    imaging_sample_index, _ = compute_imaging_sample_index(ttl_data)
    print(f'  imaging_sample_index shape: {imaging_sample_index.shape}')

    # Compute padded size from imaging_sample_index to ensure all indices are valid
    padded_size = int(imaging_sample_index.max()) + 1
    print(f'  Padding arrays from {valid_samples} to {padded_size} samples '
          f'(+{padded_size - valid_samples} samples)')

    # Create output directory
    print(f'Creating zarr at {output_path}')
    os.makedirs(output_path, exist_ok=True)

    # Build columns metadata
    columns_meta = [
        {'name': col['name'], 'dtype': col['dtype'],
         'description': col['description']}
        for col in EPHYS_COLUMNS
    ]
    # Add condition column
    columns_meta.append({
        'name': 'condition',
        'dtype': 'int8',
        'description': 'Condition index (0-8)',
    })

    # Write global metadata
    global_meta = {
        'zarr_format': 3,
        'node_type': 'group',
        'attributes': {
            'sampling_frequency_hz': EPHYS_SAMPLING_FREQUENCY_HZ,
            'valid_samples': valid_samples,
            'padded_samples': padded_size,
            'columns': columns_meta,
            'imaging': {
                'num_timepoints': NUM_TIMEPOINTS,
                'num_z_slices': NUM_Z_SLICES,
                'samples_per_z': SAMPLES_PER_Z,
                'samples_per_volume': SAMPLES_PER_VOLUME,
            },
        },
    }
    with open(os.path.join(output_path, 'zarr.json'), 'w') as f:
        json.dump(global_meta, f, indent=2)
    print('Wrote zarr.json')

    # Write each column as 1D array (padded to ensure all imaging indices are valid)
    for col in EPHYS_COLUMNS:
        col_name = col['name']
        source_channel = col['source_channel']
        dtype = col['dtype']

        col_data = data[:, source_channel]
        # Pad with 0.0 for TTL, last valid value for other columns
        if col_name == 'ttl':
            pad_value = 0.0
        else:
            pad_value = col_data[-1]
        col_data_padded = pad_array(col_data, padded_size, pad_value)
        write_1d_array(output_path, col_name, col_data_padded, dtype)
        print(f'Wrote {col_name} ({dtype}): {len(col_data_padded)} samples')

    # Write condition column (padded with last valid value)
    condition_ids_padded = pad_array(condition_ids, padded_size, condition_ids[-1])
    write_1d_array(output_path, 'condition', condition_ids_padded, 'int8')
    print(f'Wrote condition (int8): {len(condition_ids_padded)} samples')

    # Write imaging sample index as 2D array
    write_2d_array(output_path, 'imaging_sample_index', imaging_sample_index, 'int32')
    print(f'Wrote imaging_sample_index (int32): {imaging_sample_index.shape}')

    print('Done')


if __name__ == '__main__':
    main()
