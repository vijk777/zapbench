# ephys.zarr

Electrophysiology data in zarr v3 format with imaging alignment.

## Contents

### 1D arrays (43,246,155 samples, padded)

| Array | dtype | Description |
|-------|-------|-------------|
| `ephys_ch1` | float32 | Electrophysiology channel 1 - muscular activity |
| `ephys_ch2` | float32 | Electrophysiology channel 2 - muscular activity (better for emf3) |
| `ttl` | float32 | TTL triggers for volume imaging sync |
| `stimParam3` | float32 | Stimulus parameter 3 (meaning varies by condition) |
| `stimParam4` | int8 | Stimulus parameter 4 (meaning varies by condition) |
| `visual_velocity` | float32 | Grating velocity (+ = backward, - = forward) |
| `condition` | int8 | Condition index (0-8) |

### 2D arrays

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `imaging_sample_index` | (7879, 72) | int32 | Ephys sample index for each imaging frame (t, z) |
| `cell_ephys_index` | (7879, 71721) | int32 | Ephys sample index for each cell at each timepoint |

## Key parameters

- **Sampling frequency**: 6000 Hz
- **Valid samples**: 43,206,740 (actual recorded data)
- **Padded samples**: 43,246,155 (to cover extrapolated indices for truncated timepoints)
- **Imaging timepoints**: 7,879
- **Z-slices per volume**: 72
- **Number of cells**: 71,721

## Regenerating ephys.zarr

### Step 1: Convert raw ephys data

```bash
python processing/convert_ephys_to_zarr.py
```

This reads the raw ephys binary file and creates:
- All 1D arrays (ephys_ch1, ephys_ch2, ttl, etc.)
- `imaging_sample_index` (computed from TTL peaks)

**Input**: `/groups/saalfeld/saalfeldlab/zapbench-release/volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt`

**Output**: `ephys.zarr/` with all 1D arrays and `imaging_sample_index`

### Step 2: Compute per-cell ephys indices

```bash
python processing/compute_cell_ephys_index.py
```

This computes flow-corrected ephys indices for each cell and appends `cell_ephys_index` to the zarr.

**Inputs**:
- `ephys.zarr/imaging_sample_index` (from step 1)
- `gs://zapbench-release/volumes/20240930/segmentation` (cell segmentation)
- `gs://zapbench-release/volumes/20240930/flow_fields` (motion correction)

**Output**: `ephys.zarr/cell_ephys_index`

**Runtime**: ~18 minutes (reads 876 flow field chunks from GCS)

## Usage examples

### Load ephys for an imaging frame (t, z)

```python
import tensorstore as ts

idx_store = ts.open({'driver': 'zarr3', 'kvstore': 'ephys.zarr/imaging_sample_index'}).result()
ephys_store = ts.open({'driver': 'zarr3', 'kvstore': 'ephys.zarr/ephys_ch1'}).result()

t, z = 100, 35
sample_idx = idx_store[t, z].read().result()
window = 72  # samples per z-slice (12ms at 6kHz)
ephys_data = ephys_store[sample_idx:sample_idx+window].read().result()
```

### Load ephys for a cell at timepoint t

```python
cell_idx_store = ts.open({'driver': 'zarr3', 'kvstore': 'ephys.zarr/cell_ephys_index'}).result()
ephys_store = ts.open({'driver': 'zarr3', 'kvstore': 'ephys.zarr/ephys_ch1'}).result()

t, cell_id = 100, 5000
sample_idx = cell_idx_store[t, cell_id].read().result()
window = 72
ephys_data = ephys_store[sample_idx:sample_idx+window].read().result()
```

### Query condition using CONDITION_OFFSETS

```python
from zapbench.constants import CONDITION_OFFSETS

# Get timepoint range for taxis condition (index 3)
t_start, t_end = CONDITION_OFFSETS[3], CONDITION_OFFSETS[4]

# Load cell ephys indices for this condition
cell_ephys = cell_idx_store[t_start:t_end, :].read().result()
```
