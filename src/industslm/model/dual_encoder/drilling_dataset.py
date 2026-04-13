# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
DrillingContrastiveDataset: efficient lazy-loading dataset for drilling parquet data.

Data flow:
  1. __init__: Fast index build — reads only OPERATION+CODE columns (lightweight).
  2. __getitem__: Uses PyArrow to read only the needed row range + sensor columns
     from the parquet file. An LRU cache avoids re-reading the same file.
"""

import glob
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

SENSOR_COLUMNS = [
    "rot_vel", "stand_pressure", "wob", "torque", "flow_rate",
    "flow_out_perc", "bit_depth", "block_pos", "hook_load",
    "hole_depth", "tank_vol",
]


# File-level cache: keeps the N most recently used parquet files in memory.
# Shared across all workers via fork (each worker gets its own cache).
@lru_cache(maxsize=8)
def _read_sensor_columns(fpath: str) -> np.ndarray:
    """Read only sensor columns from a parquet file, returning a numpy array [T, C]."""
    table = pq.read_table(fpath, columns=SENSOR_COLUMNS)
    arr = table.to_pandas().values.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0)


class DrillingContrastiveDataset(Dataset):
    """
    Efficient lazy-loading contrastive dataset for drilling parquet data.

    At init: scans parquet files to build a lightweight index (reads only 2 columns).
    At __getitem__: reads sensor data on-the-fly with LRU caching per file.

    Args:
        data_dir: Path to directory containing parquet files.
        window_size: Number of timesteps per window.
        stride: Stride between consecutive windows.
        subsample: Target subsampled length per window.
        max_files: Limit number of parquet files (for debugging).
        normalize: Whether to max-scale each sensor channel.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 65536,
        stride: Optional[int] = None,
        subsample: int = 512,
        max_files: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.subsample = subsample
        self.normalize = normalize

        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if max_files is not None:
            parquet_files = parquet_files[:max_files]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        self.index: List[Tuple[str, int, int, str, str]] = []
        self._build_index(parquet_files)
        print(f"Indexed {len(self.index)} contrastive pairs from {len(parquet_files)} files.")

    def _build_index(self, parquet_files: List[str]):
        """Build lightweight index by reading only OPERATION+CODE columns."""
        import pandas as pd

        for i, fpath in enumerate(parquet_files):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Indexing file {i+1}/{len(parquet_files)}...")
            try:
                df = pd.read_parquet(fpath, columns=["OPERATION", "CODE"])
            except Exception as e:
                print(f"  Skipping {os.path.basename(fpath)}: {e}")
                continue

            df = df.dropna(subset=["OPERATION"])
            if len(df) == 0:
                continue

            # Find contiguous operation segments
            op_changed = df["OPERATION"] != df["OPERATION"].shift()
            group_ids = op_changed.cumsum()

            for _, group in df.groupby(group_ids):
                operation = str(group["OPERATION"].iloc[0])
                code = str(group["CODE"].iloc[0])
                seg_indices = group.index.to_numpy()
                row_start = int(seg_indices[0])
                row_end = int(seg_indices[-1]) + 1
                seg_len = row_end - row_start

                if seg_len < self.subsample:
                    continue

                if seg_len < self.window_size:
                    self.index.append((fpath, row_start, row_end, operation, code))
                else:
                    for start in range(row_start, row_end - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        self.index.append((fpath, start, end, operation, code))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        fpath, row_start, row_end, operation, code = self.index[idx]

        # Read from cache (or load file into cache if not present)
        file_data = _read_sensor_columns(fpath)
        sensors = file_data[row_start:row_end]  # [T, C] — just a slice, no copy

        T, C = sensors.shape

        # Subsample to target length
        if T >= self.subsample:
            indices = np.linspace(0, T - 1, self.subsample, dtype=int)
            sensors = sensors[indices]
        else:
            pad = np.zeros((self.subsample - T, C), dtype=np.float32)
            sensors = np.concatenate([sensors, pad], axis=0)

        # Normalize: max-scale per channel
        if self.normalize:
            maxvals = np.abs(sensors).max(axis=0, keepdims=True)
            maxvals = np.where(maxvals == 0, 1.0, maxvals)
            sensors = sensors / maxvals

        # [subsample, C] -> [C, subsample]
        ts_tensor = torch.from_numpy(sensors.T.copy())

        return {
            "time_series": ts_tensor,
            "text": operation,
            "code": code,
        }


def collate_contrastive(batch: List[dict], tokenizer) -> dict:
    """
    Collate function for DrillingContrastiveDataset.

    Returns dict with keys: time_series [B, C, T], input_ids [B, L], attention_mask [B, L], codes [B]
    """
    time_series = torch.stack([s["time_series"] for s in batch])
    texts = [s["text"] for s in batch]
    codes = [s["code"] for s in batch]

    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    return {
        "time_series": time_series,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "codes": codes,
    }
