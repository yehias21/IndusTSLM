# SPDX-License-Identifier: MIT
"""Wrap an existing OpenTSLM QADataset so that every sample carries:
    (a) the original prompt/answer (possibly rewritten for curriculum stage),
    (b) a dict of expert targets used by CoTTFlamingo.compute_loss.

This is the direct analog of CoVT Fig. 4 (four-stage data formatting).
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from industslm.cott.cott_model import (
    CP_TOKEN, FC_TOKEN, SP_TOKEN, SEM_TOKEN, DEFAULT_BUDGET,
)
from industslm.cott.experts import compute_expert_targets


def _repeat(tok: str, n: int) -> str:
    return "".join([tok] * n)


STAGE_STUB = (
    "The change-points of the signal are {cp}, "
    "the forecast is {fc}, the spectral edges are {sp}, "
    "and the patch features are {sem}."
)


def build_stub(budget: Dict[str, int], active: List[str]) -> str:
    parts = {
        "cp":  _repeat(CP_TOKEN,  budget["cp"])  if "cp"  in active else "",
        "fc":  _repeat(FC_TOKEN,  budget["fc"])  if "fc"  in active else "",
        "sp":  _repeat(SP_TOKEN,  budget["sp"])  if "sp"  in active else "",
        "sem": _repeat(SEM_TOKEN, budget["sem"]) if "sem" in active else "",
    }
    return STAGE_STUB.format(**parts)


class CoTTStageWrapper(Dataset):
    """Curriculum-aware wrapper.

    Args:
        base: the underlying OpenTSLM CoT dataset (HAR-CoT, Sleep-CoT, ECG-QA-CoT)
        stage: one of {"comprehension", "generation", "reasoning", "efficient"}
        budget: {"cp": 8, "fc": 4, "sp": 4, "sem": 4}
        horizon: forecast horizon used by the forecast expert
        use_chronos: if False, forecast/semantic targets are zero-filled
                     (useful for smoke-tests and for the CP-only ablation)
    """

    STAGES = ("comprehension", "generation", "reasoning", "efficient")

    def __init__(self, base: Dataset, stage: str,
                 budget: Dict[str, int] = None,
                 horizon: int = 32,
                 use_chronos: bool = True):
        assert stage in self.STAGES
        self.base = base
        self.stage = stage
        self.budget = budget or DEFAULT_BUDGET
        self.horizon = horizon
        self.use_chronos = use_chronos

    def __len__(self):
        return len(self.base)

    # ------------------------------------------------------------------
    def _active_groups(self) -> List[str]:
        if self.stage != "efficient":
            return ["cp", "fc", "sp", "sem"]
        # randomly keep 1..4 groups
        k = random.randint(1, 4)
        return random.sample(["cp", "fc", "sp", "sem"], k)

    def _rewrite(self, sample: Dict) -> Dict:
        active = self._active_groups()
        stub = build_stub(self.budget, active)

        if self.stage == "comprehension":
            # stub inserted after <TS> block; answer unchanged
            sample = {**sample, "pre_prompt":
                      sample["pre_prompt"] + " " + stub}
        elif self.stage == "generation":
            sample = {**sample,
                      "post_prompt":
                      "What are the change-points, forecast, spectrum, "
                      "and patch features of the signal?",
                      "answer": stub}
        elif self.stage == "reasoning":
            sample = {**sample,
                      "answer":
                      f"<think>{stub}</think> "
                      f"<answer>{sample['answer']}</answer>"}
        else:  # efficient
            sample = {**sample,
                      "answer":
                      f"<think>{stub}</think> "
                      f"<answer>{sample['answer']}</answer>"}
        sample["_active"] = active
        return sample

    # ------------------------------------------------------------------
    def _compute_targets(self, sample: Dict) -> Dict[str, torch.Tensor]:
        # Use the first channel of the first series as the canonical window
        # for expert supervision. Multichannel cases (ECG 12 leads, HAR 3-axis)
        # average their per-channel targets — this matches CoVT's practice of
        # giving one target per expert regardless of input richness.
        ts = sample["time_series"]          # [C, L]
        if isinstance(ts, list):
            ts = torch.as_tensor(ts, dtype=torch.float32)
        if ts.dim() == 1:
            ts = ts.unsqueeze(0)
        x = ts.float().mean(0)              # [L]

        tgt = compute_expert_targets(x, horizon=self.horizon,
                                     use_chronos=self.use_chronos)
        # Chronos-2 mid-layer features — we approximate by repeating the patch
        # embedding along N_fc. In a real run, plumb the middle-layer features
        # from experts.get_chronos().
        feats = tgt.semantic.unsqueeze(0).repeat(self.budget["fc"], 1, 1)

        return {
            "raw_series":     x,
            "cp_mask":        tgt.cp_mask,
            "log_stft":       tgt.spectral,
            "forecast":       tgt.forecast,
            "chronos_feats":  feats,
            "semantic":       tgt.semantic,
        }

    def __getitem__(self, i):
        sample = self.base[i]
        sample = self._rewrite(sample)
        sample["cott_targets"] = self._compute_targets(sample)
        return sample
