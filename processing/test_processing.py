#!/usr/bin/env python3
"""Unit tests for fluorescence processing pipeline.

Generates tiny synthetic data (XYZT=10,10,10,5 with 5 cells) and tests
both intermediate functions and end-to-end pipeline.

Usage:
    python test_processing.py
"""

import os
import shutil
import tempfile

import numpy as np
import tensorstore as ts
from absl.testing import absltest

# Import modules under test
import zarr_utils
from zarr_utils import (
    open_array,
    create_array,
    load_metadata,
    save_metadata,
    compute_cell_boundaries,
    set_thread_limits,
)

# Test data dimensions (tiny)
TEST_SIZE_X, TEST_SIZE_Y, TEST_SIZE_Z, TEST_SIZE_T = 10, 10, 10, 5
TEST_NUM_CELLS = 5


def make_synthetic_segmentation():
    """Create synthetic segmentation with 5 small cells.

    Each cell is a 2x2x2 cube at different locations.
    Cell IDs are 1-indexed (0 = background).
    """
    seg = np.zeros((TEST_SIZE_X, TEST_SIZE_Y, TEST_SIZE_Z), dtype=np.uint16)

    cell_positions = [
        (1, 1, 1),
        (4, 1, 1),
        (1, 4, 1),
        (4, 4, 4),
        (7, 7, 7),
    ]

    for cell_id, (x, y, z) in enumerate(cell_positions, start=1):
        seg[x:x+2, y:y+2, z:z+2] = cell_id

    return seg


def make_synthetic_raw_data():
    """Create synthetic raw fluorescence data.

    Returns array of shape [X, Y, Z, T] with predictable values.
    """
    raw = np.zeros((TEST_SIZE_X, TEST_SIZE_Y, TEST_SIZE_Z, TEST_SIZE_T), dtype=np.uint16)

    for t in range(TEST_SIZE_T):
        raw[:, :, :, t] = 100 + t * 10

    # Add extra signal to specific cells
    raw[1:3, 1:3, 1:3, 2:4] += 50  # Cell 1 at t=2,3
    raw[1:3, 4:6, 1:3, 4] += 100   # Cell 3 at t=4

    return raw


class ZarrUtilsTest(absltest.TestCase):
    """Tests for zarr_utils module."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.temp_dir)

    def test_set_thread_limits(self):
        set_thread_limits(4)
        self.assertEqual(os.environ['BLOSC_NTHREADS'], '4')
        self.assertEqual(os.environ['OMP_NUM_THREADS'], '4')
        set_thread_limits(2)

    def test_create_and_open_array(self):
        zarr_path = os.path.join(self.temp_dir, 'test.zarr')
        os.makedirs(zarr_path, exist_ok=True)

        shape = (10, 20)
        chunks = (5, 10)
        arr = create_array(zarr_path, 'test_array', shape, chunks, 'float32')

        data = np.random.rand(*shape).astype(np.float32)
        arr.write(data).result()

        arr2 = open_array(zarr_path, 'test_array')
        result = arr2.read().result()

        np.testing.assert_array_equal(result, data)

    def test_load_save_metadata(self):
        zarr_path = os.path.join(self.temp_dir, 'test.zarr')
        os.makedirs(zarr_path, exist_ok=True)

        metadata = {
            'num_pixels': 1000,
            'num_timesteps': 100,
            'test_key': 'test_value',
        }

        save_metadata(zarr_path, metadata)
        loaded = load_metadata(zarr_path)

        self.assertEqual(loaded, metadata)

    def test_load_metadata_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_metadata(self.temp_dir)

    def test_compute_cell_boundaries(self):
        cell_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.uint64)
        num_cells = 3

        boundaries, pixels_per_cell = compute_cell_boundaries(cell_ids, num_cells)

        np.testing.assert_array_equal(boundaries, [0, 3, 5, 9])
        np.testing.assert_array_equal(pixels_per_cell, [3, 2, 4])

    def test_compute_cell_boundaries_empty_cell(self):
        cell_ids = np.array([0, 0, 2, 2], dtype=np.uint64)
        num_cells = 3

        boundaries, pixels_per_cell = compute_cell_boundaries(cell_ids, num_cells)

        self.assertEqual(pixels_per_cell[1], 1)  # Empty cell gets 1 to avoid div by zero


class ComputeCellActivityFunctionsTest(absltest.TestCase):
    """Tests for functions in compute_cell_activity.py."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.temp_dir)

    def test_compute_percentile_chunked(self):
        from compute_cell_activity import compute_percentile_chunked

        zarr_path = os.path.join(self.temp_dir, 'test.zarr')
        os.makedirs(zarr_path, exist_ok=True)

        data = np.random.randint(0, 1000, size=(100, 20), dtype=np.uint16)
        arr = create_array(zarr_path, 'raw_values', data.shape, data.shape, 'uint16')
        arr.write(data).result()

        raw_values = open_array(zarr_path, 'raw_values')

        F0 = compute_percentile_chunked(raw_values, percentile=10, chunk_size=30)

        expected = np.percentile(data, 10, axis=1).astype(np.float32)
        np.testing.assert_array_almost_equal(F0, expected, decimal=3)

    def test_fit_smooth_spatial_field(self):
        from compute_cell_activity import fit_smooth_spatial_field, BIN_STRIDE_X, BIN_STRIDE_Y, BIN_STRIDE_Z

        num_pixels = 1000
        coords = np.random.randint(0, 100, size=(num_pixels, 3), dtype=np.int32)
        F0 = (100 + coords[:, 2] * 2).astype(np.float32)

        volume_shape = (100, 100, 100)

        A_hat, A_hat_grid = fit_smooth_spatial_field(F0, coords, volume_shape)

        self.assertEqual(len(A_hat), num_pixels)

        expected_grid_shape = (
            (volume_shape[0] + BIN_STRIDE_X - 1) // BIN_STRIDE_X,
            (volume_shape[1] + BIN_STRIDE_Y - 1) // BIN_STRIDE_Y,
            (volume_shape[2] + BIN_STRIDE_Z - 1) // BIN_STRIDE_Z,
        )
        self.assertEqual(A_hat_grid.shape, expected_grid_shape)

        # Verify z gradient is preserved
        low_z_mask = coords[:, 2] < 30
        high_z_mask = coords[:, 2] > 70
        self.assertGreater(A_hat[high_z_mask].mean(), A_hat[low_z_mask].mean())


class EndToEndTest(absltest.TestCase):
    """End-to-end tests of the full pipeline."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.temp_dir)

    def test_extract_and_sort_coordinates(self):
        import init_raw_fluorescence_zarr as init_script

        segmentation = make_synthetic_segmentation()
        xi, yi, zi, gx, gy, gz, cell_ids, num_cells = init_script.extract_and_sort_coordinates(segmentation)

        num_pixels = len(xi)
        self.assertEqual(num_pixels, TEST_NUM_CELLS * 8)  # 5 cells * 2^3 pixels each
        self.assertEqual(num_cells, TEST_NUM_CELLS)
        self.assertEqual(cell_ids.max(), TEST_NUM_CELLS - 1)  # 0-indexed

        # Verify sorted by cell_id
        self.assertTrue(np.all(np.diff(cell_ids) >= 0))

    def test_cell_pixel_boundaries(self):
        """Test that cell_pixel_boundaries correctly indexes into pixel arrays."""
        import init_raw_fluorescence_zarr as init_script

        segmentation = make_synthetic_segmentation()
        xi, yi, zi, gx, gy, gz, cell_ids, num_cells = init_script.extract_and_sort_coordinates(segmentation)
        num_pixels = len(xi)

        # Compute boundaries as the script does
        cell_pixel_boundaries = np.searchsorted(cell_ids, np.arange(num_cells + 1))

        # Verify shape
        self.assertEqual(len(cell_pixel_boundaries), num_cells + 1)
        self.assertEqual(cell_pixel_boundaries[0], 0)
        self.assertEqual(cell_pixel_boundaries[-1], num_pixels)

        # Verify each cell's pixels have correct cell_id
        for cell_id in range(num_cells):
            start = cell_pixel_boundaries[cell_id]
            end = cell_pixel_boundaries[cell_id + 1]
            cell_pixel_ids = cell_ids[start:end]

            # All pixels in this range should belong to this cell
            self.assertTrue(np.all(cell_pixel_ids == cell_id))

            # Each cell has 8 pixels (2x2x2 cube)
            self.assertEqual(end - start, 8)

    def test_full_pipeline(self):
        import init_raw_fluorescence_zarr as init_script
        from compute_cell_activity import (
            compute_percentile_chunked,
            fit_smooth_spatial_field,
            aggregate_cell_data,
        )

        output_zarr = os.path.join(self.temp_dir, 'output.zarr')
        os.makedirs(output_zarr, exist_ok=True)

        segmentation = make_synthetic_segmentation()
        raw_data = make_synthetic_raw_data()

        # Step 1: Initialize
        xi, yi, zi, gx, gy, gz, cell_ids, num_cells = init_script.extract_and_sort_coordinates(segmentation)
        num_pixels = len(xi)

        self.assertEqual(num_pixels, TEST_NUM_CELLS * 8)
        self.assertEqual(num_cells, TEST_NUM_CELLS)

        create_array(output_zarr, 'cell_ids', (num_pixels,), (num_pixels,), 'uint64').write(cell_ids).result()

        aligned_coords = np.stack([xi, yi, zi], axis=1)
        create_array(output_zarr, 'aligned_coords', (num_pixels, 3), (num_pixels, 3), 'int32').write(aligned_coords).result()

        # Write cell_pixel_boundaries
        cell_pixel_boundaries = np.searchsorted(cell_ids, np.arange(num_cells + 1)).astype(np.uint64)
        create_array(output_zarr, 'cell_pixel_boundaries', (num_cells + 1,), (num_cells + 1,), 'uint64').write(cell_pixel_boundaries).result()

        # Step 2: Extract raw values
        raw_values = np.zeros((num_pixels, TEST_SIZE_T), dtype=np.uint16)
        acquisition_time_ms = np.zeros((num_pixels, TEST_SIZE_T), dtype=np.uint32)

        for t in range(TEST_SIZE_T):
            raw_values[:, t] = raw_data[xi, yi, zi, t]
            acquisition_time_ms[:, t] = t * 914 + zi * 12

        create_array(output_zarr, 'raw_values', (num_pixels, TEST_SIZE_T), (num_pixels, TEST_SIZE_T), 'uint16').write(raw_values).result()
        create_array(output_zarr, 'acquisition_time_ms', (num_pixels, TEST_SIZE_T), (num_pixels, TEST_SIZE_T), 'uint32').write(acquisition_time_ms).result()

        save_metadata(output_zarr, {'num_pixels': num_pixels, 'num_timesteps': TEST_SIZE_T})

        # Step 3: Compute cell activity
        raw_values_arr = open_array(output_zarr, 'raw_values')
        acq_time_arr = open_array(output_zarr, 'acquisition_time_ms')

        F0 = compute_percentile_chunked(raw_values_arr, percentile=10, chunk_size=100)
        self.assertEqual(len(F0), num_pixels)

        test_volume_shape = (TEST_SIZE_X, TEST_SIZE_Y, TEST_SIZE_Z)
        A_hat, A_hat_grid = fit_smooth_spatial_field(F0, aligned_coords, test_volume_shape)
        self.assertEqual(len(A_hat), num_pixels)

        num_cells = int(cell_ids.max()) + 1
        cell_boundaries, pixels_per_cell = compute_cell_boundaries(cell_ids, num_cells)

        cell_activity, cell_acquisition_ms = aggregate_cell_data(
            raw_values_arr, acq_time_arr, A_hat,
            cell_boundaries, pixels_per_cell.astype(np.float32),
            num_cells, TEST_SIZE_T, time_chunk_size=10
        )

        self.assertEqual(cell_activity.shape, (TEST_NUM_CELLS, TEST_SIZE_T))
        self.assertEqual(cell_acquisition_ms.shape, (TEST_NUM_CELLS, TEST_SIZE_T))
        self.assertGreaterEqual(cell_activity.min(), 0)

        # Verify cells with extra signal have higher activity
        self.assertGreater(cell_activity[0, 2], cell_activity[0, 0])
        self.assertGreater(cell_activity[2, 4], cell_activity[2, 0])

    def test_cell_activity_correctness(self):
        from compute_cell_activity import aggregate_cell_data

        output_zarr = os.path.join(self.temp_dir, 'output.zarr')
        os.makedirs(output_zarr, exist_ok=True)

        num_pixels = 6
        num_timesteps = 4
        num_cells = 2

        raw_values_data = np.array([
            [100, 100, 100, 100],
            [110, 110, 110, 110],
            [120, 120, 120, 120],
            [200, 200, 200, 200],
            [210, 210, 210, 210],
            [220, 220, 220, 220],
        ], dtype=np.uint16)

        acq_time_data = np.array([
            [0, 100, 200, 300],
            [10, 110, 210, 310],
            [20, 120, 220, 320],
            [30, 130, 230, 330],
            [40, 140, 240, 340],
            [50, 150, 250, 350],
        ], dtype=np.uint32)

        cell_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64)
        A_hat = np.full(num_pixels, 50.0, dtype=np.float32)

        create_array(output_zarr, 'raw_values', (num_pixels, num_timesteps), (num_pixels, num_timesteps), 'uint16').write(raw_values_data).result()
        create_array(output_zarr, 'acquisition_time_ms', (num_pixels, num_timesteps), (num_pixels, num_timesteps), 'uint32').write(acq_time_data).result()

        raw_values_arr = open_array(output_zarr, 'raw_values')
        acq_time_arr = open_array(output_zarr, 'acquisition_time_ms')

        cell_boundaries, pixels_per_cell = compute_cell_boundaries(cell_ids, num_cells)

        cell_activity, cell_acquisition_ms = aggregate_cell_data(
            raw_values_arr, acq_time_arr, A_hat,
            cell_boundaries, pixels_per_cell.astype(np.float32),
            num_cells, num_timesteps, time_chunk_size=10
        )

        # Expected: mean of max(0, [100,110,120] - 50) = mean([50,60,70]) = 60
        expected_activity = np.array([
            [60, 60, 60, 60],
            [160, 160, 160, 160],
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(cell_activity, expected_activity)

        # Expected: mean of [0,10,20]=10, [100,110,120]=110, etc.
        expected_acq = np.array([
            [10, 110, 210, 310],
            [40, 140, 240, 340],
        ], dtype=np.uint32)

        np.testing.assert_array_equal(cell_acquisition_ms, expected_acq)


if __name__ == '__main__':
    absltest.main()
