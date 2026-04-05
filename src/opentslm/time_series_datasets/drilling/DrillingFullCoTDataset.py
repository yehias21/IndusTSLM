# SPDX-License-Identifier: MIT

"""
Stage 5 — Drilling Full CoT Dataset.

The hardest stage: given multi-channel drilling sensor data, the model
must provide a detailed chain-of-thought analysis including operation,
code, and subcode classification plus a qualitative description of
anomalies or notable patterns.
"""

from typing import List, Literal, Tuple

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.drilling.drilling_loader import (
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


def _build_full_cot_answer(row: dict) -> str:
    """Build a detailed CoT rationale covering operation, code, subcode."""
    op = row["operation"]
    code = row.get("code", "")
    subcode = row.get("subcode", "")
    means = row["means"]
    stds = row["stds"]

    lines = []
    lines.append(
        "I will perform a comprehensive analysis of all drilling sensor channels "
        "to identify the operation, code, and any notable patterns."
    )

    # Detailed per-channel analysis
    for i, name in enumerate(SENSOR_COLUMNS):
        friendly = name.replace("_", " ")
        m, s = means[i], stds[i]
        if s > 1.0:
            lines.append(
                f"The {friendly} channel shows very high variability "
                f"(std={s:.2f}, mean={m:.2f}), suggesting active dynamic conditions."
            )
        elif s > 0.3:
            lines.append(
                f"The {friendly} channel has moderate variability "
                f"(std={s:.2f}, mean={m:.2f})."
            )
        elif abs(m) < 0.05 and s < 0.05:
            lines.append(
                f"The {friendly} channel is essentially flat/inactive."
            )

    # Correlations
    depth_channels = ["bit_depth", "hole_depth"]
    depth_indices = [SENSOR_COLUMNS.index(c) for c in depth_channels]
    if all(stds[idx] > 0.3 for idx in depth_indices):
        lines.append(
            "Both bit depth and hole depth are changing, indicating active drilling progression."
        )
    elif all(stds[idx] < 0.1 for idx in depth_indices):
        lines.append(
            "Depth sensors are stable, suggesting a non-drilling operation "
            "(e.g., circulation, tripping, or connection)."
        )

    answer_parts = [f"Operation: {op}"]
    if code:
        answer_parts.append(f"Code: {code}")
    if subcode:
        answer_parts.append(f"Subcode: {subcode}")
    lines.append("Answer: " + ", ".join(answer_parts))

    return " ".join(lines)


class DrillingFullCoTDataset(QADataset):
    """Stage 5: Full CoT reasoning with operation + code + subcode."""

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
        return _build_full_cot_answer(row)

    def _get_pre_prompt(self, row) -> str:
        return (
            "You are given multi-channel drilling sensor data from an oil & gas well. "
            "Provide a comprehensive analysis of the sensor data and classify the "
            "drilling operation, code, and subcode.\n\n"
            "Instructions:\n"
            "- Analyze each sensor channel (mechanical, hydraulic, depth, tank).\n"
            "- Note any anomalies, transitions, or correlations between channels.\n"
            "- Identify whether the well is actively drilling, circulating, tripping, "
            "making connections, or performing another operation.\n"
            "- End with 'Answer: Operation: <op>, Code: <code>, Subcode: <subcode>'.\n"
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
        sample["subcode"] = row.get("subcode", "")
        return sample
