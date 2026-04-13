# SPDX-License-Identifier: MIT
"""Four-stage CoTT curriculum trainer (analog of CoVT Fig. 4).

Usage:
    python train_cott.py --dataset har --variant cott4 --stages all

This driver is intentionally thin: it re-uses the batching / optimizer /
early-stopping logic that already exists in `curriculum_learning.py`.
We only swap the model for CoTTFlamingo and the dataset for
CoTTStageWrapper, and we pass the expert targets into compute_loss.
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from industslm.cott.cott_model import DEFAULT_BUDGET
from industslm.cott.build import build_cott_flamingo
from industslm.cott.cott_dataset import CoTTStageWrapper
from industslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from industslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from industslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from industslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


DATASETS = {"har": HARCoTQADataset, "sleep": SleepEDFCoTQADataset,
            "ecg": ECGQACoTQADataset}

VARIANT_BUDGETS = {
    "baseline":     {"cp": 0, "fc": 0, "sp": 0, "sem": 0},
    "latent_only":  {"cp": 0, "fc": 0, "sp": 0, "sem": 0},  # handled separately
    "cott1_cp":     {"cp": 8, "fc": 0, "sp": 0, "sem": 0},
    "cott1_fc":     {"cp": 0, "fc": 8, "sp": 0, "sem": 0},
    "cott1_sp":     {"cp": 0, "fc": 0, "sp": 8, "sem": 0},
    "cott1_sem":    {"cp": 0, "fc": 0, "sp": 0, "sem": 8},
    "cott3":        {"cp": 8, "fc": 4, "sp": 0, "sem": 4},
    "cott4":        DEFAULT_BUDGET,
}

STAGE_STEPS = {"comprehension": 4000, "generation": 3000,
               "reasoning": 3000, "efficient": 5000}


def collate(batch):
    base = extend_time_series_to_match_patch_size_and_aggregate(
        [{k: v for k, v in b.items() if k != "cott_targets"} for b in batch]
    )
    targets = [b["cott_targets"] for b in batch]
    for i, b in enumerate(batch):
        base[i]["_active"] = b.get("_active", [])
    return base, targets


def run_stage(model, loader, stage, opt, sched, device, gamma=1.0):
    model.train()
    steps = STAGE_STEPS[stage]
    pbar = tqdm(total=steps, desc=f"stage={stage}")
    it = iter(loader)
    step = 0
    while step < steps:
        try:
            batch, targets = next(it)
        except StopIteration:
            it = iter(loader); continue
        loss = model.compute_loss(batch, cott_targets=targets, gamma=gamma)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched: sched.step()
        pbar.update(1); pbar.set_postfix(loss=float(loss))
        step += 1
    pbar.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), default="har")
    p.add_argument("--variant", choices=list(VARIANT_BUDGETS), default="cott4")
    p.add_argument("--llm", default="Qwen/Qwen3-1.7B",
                   help="HF LLM id. Tested: Qwen3-{0.6B,1.7B,4B}, "
                        "Qwen2.5-{0.5B,1.5B,3B}, Llama-3.2-{1B,3B}, "
                        "google/gemma-3-{270m,1b-pt}")
    p.add_argument("--encoder", choices=["cnn", "chronos"], default="chronos")
    p.add_argument("--chronos_id", default="amazon/chronos-2")
    p.add_argument("--stages", default="all",
                   help='"all" or comma list of {comprehension,generation,reasoning,efficient}')
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="cott_ckpt.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    budget = VARIANT_BUDGETS[args.variant]
    model = build_cott_flamingo(
        llm_id=args.llm, encoder=args.encoder, chronos_id=args.chronos_id,
        budget=budget, horizon=args.horizon, device=device,
    )

    train_base = DATASETS[args.dataset]("train", EOS_TOKEN=model.get_eos_token())

    stages = (list(CoTTStageWrapper.STAGES) if args.stages == "all"
              else args.stages.split(","))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr, weight_decay=1e-2)

    for stage in stages:
        ds = CoTTStageWrapper(train_base, stage=stage, budget=budget,
                              horizon=args.horizon,
                              use_chronos=budget.get("fc", 0) +
                                          budget.get("sem", 0) > 0)
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate)
        run_stage(model, loader, stage, opt, None, device, gamma=args.gamma)

    model.store_to_file(args.out)


if __name__ == "__main__":
    main()
