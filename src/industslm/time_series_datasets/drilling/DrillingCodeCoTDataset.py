# SPDX-License-Identifier: MIT

"""
Stage 4 — Drilling Code CoT Dataset.

Given multi-channel drilling sensor data, the model reasons step-by-step
and predicts both the operation AND the operation code.
This is a harder task than Stage 3 which only predicts the operation.
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


def _build_code_cot_answer(row: dict) -> str:
    """Build a CoT rationale that concludes with both operation and code."""
    op = row["operation"]
    code = row.get("code", "")
    means = row["means"]
    stds = row["stds"]

    lines = []
    lines.append(
        "I will analyze each sensor channel to determine the drilling operation and its code."
    )

    # Describe patterns per channel group
    mechanical = ["rot_vel", "wob", "torque", "hook_load"]
    hydraulic = ["stand_pressure", "flow_rate", "flow_out_perc"]
    depth = ["bit_depth", "block_pos", "hole_depth"]
    tank = ["tank_vol"]

    for group_name, group_cols in [
        ("mechanical", mechanical),
        ("hydraulic", hydraulic),
        ("depth", depth),
        ("tank", tank),
    ]:
        group_desc = []
        for col in group_cols:
            idx = SENSOR_COLUMNS.index(col)
            if stds[idx] > 0.5:
                group_desc.append(f"{col.replace('_', ' ')} shows high variability")
            elif abs(means[idx]) < 0.1 and stds[idx] < 0.1:
                group_desc.append(f"{col.replace('_', ' ')} is near-constant")
        if group_desc:
            lines.append(
                f"The {group_name} sensors show: " + "; ".join(group_desc) + "."
            )

    answer_parts = [f"Operation: {op}"]
    if code:
        answer_parts.append(f"Code: {code}")
    lines.append("Answer: " + ", ".join(answer_parts))

    return " ".join(lines)


class DrillingCodeCoTDataset(QADataset):
    """Stage 4: CoT reasoning predicting operation + code from drilling sensors."""

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
        return _build_code_cot_answer(row)

    def _get_pre_prompt(self, row) -> str:
        return (
            "You are given multi-channel drilling sensor data from an oil & gas well. "
            "Your task is to classify both the drilling operation and its specific code "
            "based on analysis of the sensor patterns.\n\n"
            "Instructions:\n"
            "- Analyze the mechanical sensors (rotary velocity, weight on bit, torque, hook load).\n"
            "- Analyze the hydraulic sensors (standpipe pressure, flow rate, flow out percentage).\n"
            "- Analyze the depth sensors (bit depth, block position, hole depth).\n"
            "- Think step-by-step about what these patterns reveal.\n"
            "- End with 'Answer: Operation: <operation>, Code: <code>'.\n"
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
