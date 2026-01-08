#!/usr/bin/env python3
"""Extract raw fluorescence values for a batch of timesteps.

Reads precomputed coordinates from the initialized zarr and extracts raw fluorescence
values for a contiguous range of timesteps. Designed to run as parallel jobs on a cluster.

Usage:
    python extract_raw_fluorescence.py --start-t 0 --end-t 100 --output-zarr /path/to/output.zarr

    # Submit as cluster jobs:
    for start_t in $(seq 0 100 7878); do
        end_t=$((start_t + 100))
        if [ $end_t -gt 7879 ]; then end_t=7879; fi
        bsub -J "extract_t${start_t}" -n 1 -o "logs/t${start_t}.log" -e "logs/t${start_t}.err" \
            python extract_raw_fluorescence.py --start-t $start_t --end-t $end_t \
            --output-zarr /path/to/output.zarr
    done
"""

import argparse
import json
import os
import numpy as np
import tensorstore as ts


# Acquisition timing constants
MS_PER_TIMESTEP = 914  # milliseconds between timesteps
MS_PER_Z = 12  # milliseconds between z planes within a timestep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract raw fluorescence values for a batch of timesteps"
    )
    parser.add_argument(
        "--start-t",
        type=int,
        required=True,
        help="Start timestep (inclusive)",
    )
    parser.add_argument(
        "--end-t",
        type=int,
        required=True,
        help="End timestep (exclusive)",
    )
    parser.add_argument(
        "--output-zarr",
        type=str,
        required=True,
        help="Path to output zarr directory (must be initialized)",
    )
    return parser.parse_args()


def load_metadata(output_path: str) -> dict:
    """Load metadata from initialized zarr."""
    metadata_path = os.path.join(output_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            "Run init_raw_fluorescence_zarr.py first."
        )
    with open(metadata_path, 'r') as f:
        return json.load(f)


def open_output_array(output_path: str, name: str):
    """Open an existing zarr array for reading or writing."""
    return ts.open({
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(output_path, name),
        },
        'open': True,
    }).result()


def open_source_data(gs_uri: str):
    """Open source data arrays from GCS."""
    ds_flow = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': f'{gs_uri}/flow_fields'
    }).result()

    ds_raw = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': f'{gs_uri}/raw/'
    }).result()

    return ds_flow, ds_raw


def main():
    args = parse_args()
    start_t = args.start_t
    end_t = args.end_t
    batch_size = end_t - start_t

    print(f"Extracting raw fluorescence for timesteps [{start_t}, {end_t})")

    # Load metadata
    metadata = load_metadata(args.output_zarr)
    gs_uri = metadata['gs_uri']
    num_pixels = metadata['num_pixels']
    num_timesteps = metadata['num_timesteps']

    # Clamp end_t to valid range (handles last batch in job array)
    if end_t > num_timesteps:
        end_t = num_timesteps
        batch_size = end_t - start_t

    # Validate range
    if start_t < 0 or end_t > num_timesteps or start_t >= end_t:
        raise ValueError(f"Invalid range [{start_t}, {end_t}) for {num_timesteps} timesteps")

    print(f"  Source: {gs_uri}")
    print(f"  Pixels: {num_pixels:,}")
    print(f"  Batch size: {batch_size}")

    # Open source data
    print("Opening source data...")
    ds_flow, ds_raw = open_source_data(gs_uri)

    # Open output arrays
    print("Opening output arrays...")
    out_raw_values = open_output_array(args.output_zarr, 'raw_values')
    out_raw_z = open_output_array(args.output_zarr, 'raw_z')
    out_acq_time = open_output_array(args.output_zarr, 'acquisition_time')

    # Load precomputed coordinates
    print("Loading precomputed coordinates...")
    aligned_coords = open_output_array(args.output_zarr, 'aligned_coords').read().result()
    grid_coords = open_output_array(args.output_zarr, 'grid_coords').read().result()

    xi = aligned_coords[:, 0]
    yi = aligned_coords[:, 1]
    zi = aligned_coords[:, 2]
    gx = grid_coords[:, 0]
    gy = grid_coords[:, 1]
    gz = grid_coords[:, 2]

    # Allocate output buffers
    print("Allocating buffers...")
    raw_values_batch = np.empty((num_pixels, batch_size), dtype=np.uint16)
    raw_z_batch = np.empty((num_pixels, batch_size), dtype=np.int16)
    acq_time_batch = np.empty((num_pixels, batch_size), dtype=np.uint32)

    # Offset scaling factor
    offset_scale = np.array([1, 1, 4], dtype=np.float32)

    # Process each timestep
    print("Processing timesteps...")
    for t_idx, T in enumerate(range(start_t, end_t)):
        if t_idx % 10 == 0:
            print(f"  T={T} ({t_idx + 1}/{batch_size})")

        # Read flow field offsets for all pixels at time T
        # Flow field shape: [3, fz, fy, fx, t]
        offset = ds_flow[:, gz, gy, gx, T].read().result()

        # Compute integer offsets (raw space has different z resolution)
        ioffset = np.round(offset / offset_scale[:, np.newaxis]).astype(np.int32)

        # Compute raw coordinates
        raw_x = xi + ioffset[0]
        raw_y = yi + ioffset[1]
        raw_z = zi + ioffset[2]

        # Read entire raw volume for this timestep
        raw_stack = ds_raw[:, :, :, T].read().result()

        # Sample values at raw coordinates
        raw_values_batch[:, t_idx] = raw_stack[raw_x, raw_y, raw_z]
        raw_z_batch[:, t_idx] = raw_z

        # Compute acquisition time: T * 914ms + raw_z * 12ms
        acq_time_batch[:, t_idx] = T * MS_PER_TIMESTEP + raw_z * MS_PER_Z

    # Write batch to output
    print(f"Writing batch to zarr [:, {start_t}:{end_t}]...")
    out_raw_values[:, start_t:end_t].write(raw_values_batch).result()
    out_raw_z[:, start_t:end_t].write(raw_z_batch).result()
    out_acq_time[:, start_t:end_t].write(acq_time_batch).result()

    print(f"Done. Processed {batch_size} timesteps.")


if __name__ == "__main__":
    main()
