# SPDX-License-Identifier: MIT

"""
Drilling data loader for curriculum learning stages.

Reads parquet files from a drilling data directory, segments them by
contiguous OPERATION blocks, and produces samples with sensor time series
plus operation/code labels.  Splits are deterministic (by file hash).

Each sample is a dict with keys:
  - sensors: list of 11 sensor channel values (each a list of floats)
  - operation: str  (drilling operation label)
  - code: str       (drilling code label)
  - subcode: str    (drilling subcode, may be empty)
  - sensor_names: list[str]
"""

import glob
import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from functools import lru_cache


SENSOR_COLUMNS = [
    "rot_vel", "stand_pressure", "wob", "torque", "flow_rate",
    "flow_out_perc", "bit_depth", "block_pos", "hook_load",
    "hole_depth", "tank_vol",
]

# Target length after subsampling each window
DEFAULT_SUBSAMPLE = 512
DEFAULT_WINDOW_SIZE = 65536

# Train / val / test split ratios (by file)
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# TEST_FRAC = 0.10 (remainder)


@lru_cache(maxsize=8)
def _read_parquet_file(fpath: str):
    """Read sensor + label columns from a parquet file, cached."""
    cols = SENSOR_COLUMNS + ["OPERATION", "CODE", "SUBCODE"]
    table = pq.read_table(fpath, columns=cols)
    return table.to_pandas()


def _file_split_key(fpath: str) -> float:
    """Deterministic hash → [0, 1) for splitting files into train/val/test."""
    h = hashlib.md5(os.path.basename(fpath).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _subsample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Subsample or zero-pad array of shape [T, C] to [target_len, C]."""
    T = arr.shape[0]
    if T >= target_len:
        indices = np.linspace(0, T - 1, target_len, dtype=int)
        return arr[indices]
    else:
        pad = np.zeros((target_len - T, arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)


def _normalize_channels(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Max-normalise each channel independently.

    Returns (normalised, means, stds) where means/stds are per-channel.
    """
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    normed = (arr - means) / stds
    return normed, means, stds


def _build_samples_from_file(
    fpath: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    subsample: int = DEFAULT_SUBSAMPLE,
) -> List[dict]:
    """Extract windowed samples from a single parquet file."""
    df = _read_parquet_file(fpath)
    df = df.dropna(subset=["OPERATION"])
    if len(df) == 0:
        return []

    sensor_arr = df[SENSOR_COLUMNS].values.astype(np.float32)
    sensor_arr = np.nan_to_num(sensor_arr, nan=0.0)

    # Find contiguous operation segments
    op_changed = df["OPERATION"] != df["OPERATION"].shift()
    group_ids = op_changed.cumsum()

    samples = []
    for _, group in df.groupby(group_ids):
        operation = str(group["OPERATION"].iloc[0])
        code = str(group["CODE"].iloc[0]) if "CODE" in group.columns else ""
        subcode = str(group["SUBCODE"].iloc[0]) if "SUBCODE" in group.columns else ""

        seg_indices = group.index.to_numpy()
        row_start = int(seg_indices[0])
        row_end = int(seg_indices[-1]) + 1
        seg_len = row_end - row_start

        if seg_len < subsample:
            continue

        stride = window_size // 2
        if seg_len < window_size:
            windows = [(row_start, row_end)]
        else:
            windows = [
                (s, s + window_size)
                for s in range(row_start, row_end - window_size + 1, stride)
            ]

        for ws, we in windows:
            chunk = sensor_arr[ws:we]
            chunk = _subsample(chunk, subsample)
            normed, means, stds = _normalize_channels(chunk)

            # Each channel becomes a separate 1D time series
            channel_data = [normed[:, c].tolist() for c in range(normed.shape[1])]

            samples.append({
                "sensors": channel_data,
                "operation": operation,
                "code": code,
                "subcode": subcode,
                "sensor_names": SENSOR_COLUMNS,
                "means": means.tolist(),
                "stds": stds.tolist(),
            })

    return samples


def load_drilling_splits(
    train_dir: str,
    eval_dir: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    subsample: int = DEFAULT_SUBSAMPLE,
    max_files: Optional[int] = None,
) -> Tuple[list, list, list]:
    """Load drilling data and split into train / val / test.

    If eval_dir is given, all files in train_dir → train, eval_dir → split
    into val (50%) + test (50%).

    Otherwise, files in train_dir are split by file-hash into
    80/10/10 train/val/test.

    Returns (train_samples, val_samples, test_samples) where each sample
    is a dict.
    """
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.parquet")))
    if max_files is not None:
        train_files = train_files[:max_files]

    if eval_dir is not None:
        eval_files = sorted(glob.glob(os.path.join(eval_dir, "*.parquet")))
        if max_files is not None:
            eval_files = eval_files[:max_files]

        train_samples = []
        for f in train_files:
            train_samples.extend(_build_samples_from_file(f, window_size, subsample))

        # Split eval files 50/50 into val/test by file hash
        val_samples, test_samples = [], []
        for f in eval_files:
            s = _build_samples_from_file(f, window_size, subsample)
            if _file_split_key(f) < 0.5:
                val_samples.extend(s)
            else:
                test_samples.extend(s)

    else:
        train_samples, val_samples, test_samples = [], [], []
        for f in train_files:
            s = _build_samples_from_file(f, window_size, subsample)
            key = _file_split_key(f)
            if key < TRAIN_FRAC:
                train_samples.extend(s)
            elif key < TRAIN_FRAC + VAL_FRAC:
                val_samples.extend(s)
            else:
                test_samples.extend(s)

    print(f"Drilling splits: train={len(train_samples)}, "
          f"val={len(val_samples)}, test={len(test_samples)}")
    return train_samples, val_samples, test_samples
