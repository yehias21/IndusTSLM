# SPDX-License-Identifier: MIT

"""
Stage 3 — Drilling Chain-of-Thought Dataset.

Given multi-channel drilling sensor data, the model reasons step-by-step
about the sensor patterns and concludes with the operation classification.
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


def _build_cot_answer(row: dict) -> str:
    """Build a chain-of-thought rationale from sensor stats and the label."""
    op = row["operation"]
    code = row.get("code", "")
    means = row["means"]
    stds = row["stds"]

    lines = []
    lines.append(
        "Looking at the sensor readings, I need to identify the drilling operation."
    )

    # Highlight notable channels
    notable = []
    for i, name in enumerate(SENSOR_COLUMNS):
        if stds[i] > 0.5:
            notable.append(f"{name.replace('_', ' ')} (high variability, std={stds[i]:.2f})")
        elif abs(means[i]) > 1.0:
            notable.append(f"{name.replace('_', ' ')} (mean={means[i]:.2f})")

    if notable:
        lines.append(
            "The most notable sensor channels are: " + ", ".join(notable[:4]) + "."
        )

    if code:
        lines.append(
            f"Based on the combination of these sensor patterns, "
            f"this is consistent with operation code {code}."
        )

    lines.append(f"Answer: {op}")
    return " ".join(lines)


class DrillingCoTDataset(QADataset):
    """Stage 3: Chain-of-thought reasoning about drilling operations."""

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
        return _build_cot_answer(row)

    def _get_pre_prompt(self, row) -> str:
        return (
            "You are given multi-channel drilling sensor data. "
            "Your task is to classify the drilling operation based on analysis of the data.\n\n"
            "Instructions:\n"
            "- Begin by analyzing the sensor patterns without assuming a specific label.\n"
            "- Think step-by-step about what the observed patterns suggest regarding "
            "the drilling activity.\n"
            "- Write your rationale as a single, natural paragraph.\n"
            "- Make sure your last words are 'Answer: <operation>'.\n"
        )

    def _get_post_prompt(self, row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        prompts = []
        for i, (label, channel) in enumerate(zip(_SENSOR_LABELS, row["sensors"])):
            mean = row["means"][i]
            std = row["stds"][i]
            text = f"{label}, it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text, channel))
        return prompts

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["operation"] = row["operation"]
        sample["code"] = row.get("code", "")
        return sample
