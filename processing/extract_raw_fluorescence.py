#!/usr/bin/env python3
"""Extract raw fluorescence values for a batch of timesteps.

Applies motion correction via cubic spline interpolation of flow fields, then samples
raw fluorescence at corrected coordinates. Designed for parallel cluster jobs.

Usage:
    python processing/extract_raw_fluorescence.py --start-t 0 --end-t 100 --output-zarr output.zarr
"""

import argparse

import numpy as np
import tensorstore as ts
from scipy.ndimage import map_coordinates

from zarr_utils import load_metadata, open_array, STRIDE_X, STRIDE_Y, STRIDE_Z


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

    print(f"Extracting raw fluorescence for timesteps [{start_t}, {end_t})", flush=True)

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

    print(f"  Source: {gs_uri}", flush=True)
    print(f"  Pixels: {num_pixels:,}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)

    # Open source data
    print("Opening source data...", flush=True)
    ds_flow, ds_raw = open_source_data(gs_uri)

    # Open output arrays
    print("Opening output arrays...", flush=True)
    out_raw_values = open_array(args.output_zarr, 'raw_values')
    out_raw_z = open_array(args.output_zarr, 'raw_z')
    out_acq_time = open_array(args.output_zarr, 'acquisition_time_ms')

    # Load precomputed coordinates
    print("Loading precomputed coordinates...", flush=True)
    aligned_coords = open_array(args.output_zarr, 'aligned_coords').read().result()

    xi = aligned_coords[:, 0]
    yi = aligned_coords[:, 1]
    zi = aligned_coords[:, 2]

    # Allocate output buffers
    print("Allocating buffers...", flush=True)
    raw_values_batch = np.empty((num_pixels, batch_size), dtype=np.uint16)
    raw_z_batch = np.empty((num_pixels, batch_size), dtype=np.int16)
    acq_time_batch = np.empty((num_pixels, batch_size), dtype=np.uint32)

    # Process each timestep
    print("Processing timesteps...", flush=True)
    for t_idx, T in enumerate(range(start_t, end_t)):
        print(f"  T={T} ({t_idx + 1}/{batch_size})", flush=True)

        # Read flow field slice for this timestep and interpolate
        # Flow field shape: [3, fz, fy, fx, t] -> slice is [3, fz, fy, fx]
        flow_t = ds_flow[:, :, :, :, T].read().result()

        # Cubic spline interpolation at pixel positions
        coords = np.array([zi / STRIDE_Z, yi / STRIDE_Y, xi / STRIDE_X])
        offset = np.stack([map_coordinates(flow_t[c], coords, order=3, mode='nearest') for c in range(3)])

        # Compute raw coordinates (scale offset for raw space z resolution)
        raw_coords = np.array([
            xi + offset[0],
            yi + offset[1],
            zi + offset[2] / 4.0,  # raw space has 4x lower z resolution
        ])

        # Read entire raw volume for this timestep
        raw_stack = ds_raw[:, :, :, T].read().result()

        # Sample values at raw coordinates (nearest neighbor)
        raw_values_batch[:, t_idx] = map_coordinates(raw_stack, raw_coords, order=0, mode='nearest')
        raw_z_batch[:, t_idx] = np.round(raw_coords[2]).astype(np.int16)

        # Compute acquisition time: T * 914ms + raw_z * 12ms
        acq_time_batch[:, t_idx] = T * MS_PER_TIMESTEP + raw_z_batch[:, t_idx] * MS_PER_Z

    # Write batch to output
    print(f"Writing batch to zarr [:, {start_t}:{end_t}]...", flush=True)
    out_raw_values[:, start_t:end_t].write(raw_values_batch).result()
    out_raw_z[:, start_t:end_t].write(raw_z_batch).result()
    out_acq_time[:, start_t:end_t].write(acq_time_batch).result()

    print(f"Done. Processed {batch_size} timesteps.", flush=True)


if __name__ == "__main__":
    main()
