# SPDX-License-Identifier: MIT

"""
Stage 1 — Drilling MCQ Dataset.

Given multi-channel drilling sensor data, the model must pick the correct
drilling operation from a set of multiple-choice options.
"""

import random
from typing import List, Literal, Tuple

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.drilling.drilling_loader import (
    SENSOR_COLUMNS,
    load_drilling_splits,
)

# Default data paths — override via environment variables
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

# Will be populated on first load
_ALL_OPERATIONS: List[str] = []
NUM_DISTRACTORS = 3


class DrillingMCQDataset(QADataset):
    """Stage 1: Multiple-choice operation classification from drilling sensors."""

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[list, list, list]:
        train, val, test = load_drilling_splits(TRAIN_DIR, EVAL_DIR)
        global _ALL_OPERATIONS
        _ALL_OPERATIONS = sorted(
            set(s["operation"] for s in train + val + test)
        )
        return train, val, test

    def _get_answer(self, row) -> str:
        return row["_answer_letter"]

    def _get_pre_prompt(self, row) -> str:
        return (
            "You are given multi-channel drilling sensor data. "
            "Your task is to identify the drilling operation being performed.\n\n"
            "Question: Which drilling operation is shown in the following sensor data?\n"
        )

    def _get_post_prompt(self, row) -> str:
        return row["_options_text"] + "\nAnswer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        prompts = []
        for i, (label, channel) in enumerate(zip(_SENSOR_LABELS, row["sensors"])):
            mean = row["means"][i]
            std = row["stds"][i]
            text = f"{label}, it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text, channel))
        return prompts

    def _format_sample(self, row):
        # Build MCQ options
        correct = row["operation"]
        distractors = [op for op in _ALL_OPERATIONS if op != correct]
        if len(distractors) >= NUM_DISTRACTORS:
            chosen = random.sample(distractors, NUM_DISTRACTORS)
        else:
            chosen = distractors
        options = chosen + [correct]
        random.shuffle(options)

        letters = "ABCDEFGHIJ"
        correct_idx = options.index(correct)
        answer_letter = letters[correct_idx]

        options_text = "\n".join(
            f"  {letters[i]}) {op}" for i, op in enumerate(options)
        )

        row["_answer_letter"] = answer_letter
        row["_options_text"] = options_text

        sample = super()._format_sample(row)
        sample["operation"] = correct
        sample["code"] = row.get("code", "")
        return sample
