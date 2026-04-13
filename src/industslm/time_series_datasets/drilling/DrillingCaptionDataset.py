# SPDX-License-Identifier: MIT

"""
Stage 2 — Drilling Captioning Dataset.

Given multi-channel drilling sensor data, the model must describe the
drilling operation and sensor characteristics in natural language.
"""

from typing import List, Literal, Tuple

from industslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from industslm.time_series_datasets.QADataset import QADataset
from industslm.time_series_datasets.drilling.drilling_loader import (
    SENSOR_COLUMNS,
    load_drilling_splits,
)

import os

TRAIN_DIR = os.environ.get(
    "DRILLING_TRAIN_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "drilling data", "train")
)
EVAL_DIR = os.environ.get(
    "DRILLING_EVAL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "drilling data", "eval")
)

_SENSOR_LABELS = [
    f"The following is the {name.replace('_', ' ')} sensor data"
    for name in SENSOR_COLUMNS
]


class DrillingCaptionDataset(QADataset):
    """Stage 2: Caption/describe drilling sensor data."""

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[list, list, list]:
        return load_drilling_splits(TRAIN_DIR, EVAL_DIR)

    def _get_answer(self, row) -> str:
        op = row["operation"]
        code = row.get("code", "")
        subcode = row.get("subcode", "")
        parts = [f"The drilling operation is: {op}."]
        if code:
            parts.append(f"Operation code: {code}.")
        if subcode:
            parts.append(f"Sub-code: {subcode}.")
        return " ".join(parts)

    def _get_pre_prompt(self, row) -> str:
        return (
            "You are given multi-channel drilling sensor data from an oil & gas well. "
            "Describe the drilling operation being performed based on the sensor readings.\n"
        )

    def _get_post_prompt(self, row) -> str:
        return "Description:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        prompts = []
        for i, (label, channel) in enumerate(zip(_SENSOR_LABELS, row["sensors"])):
            mean = row["means"][i]
            std = row["stds"][i]
            text = f"{label}, it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text, channel))
        return prompts
