# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Evaluation script for the contrastive dual-encoder model.

Computes:
  1. Cross-modal retrieval: Recall@1, @10, @100 (TS->Text and Text->TS)
  2. Class-level retrieval: F1@1, F1@10
  3. Zero-shot classification: accuracy using text prompts as class anchors
  4. Embedding visualization (optional t-SNE)
"""

import argparse
import json
import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from industslm.model.dual_encoder.DualEncoder import DualEncoderModel
from industslm.model.dual_encoder.drilling_dataset import (
    DrillingContrastiveDataset,
    collate_contrastive,
)


# Zero-shot prompts per class (from DriMM paper)
ZERO_SHOT_PROMPTS = {
    "TRIP": ["RIH", "POOH"],
    "DRILL": ["DRILL"],
    "CSG": ["CSG", "LINER"],
    "CM": ["TUBING", "TBG"],
    "CMT": ["CEMENT", "CMT", "JOB"],
    "CORE": ["CORE", "CORING"],
    "STKP": ["STUCK", "WORKING STRING"],
    "DRLOUT": ["DRILL OUT"],
    "REAM": ["WASHED DOWN", "REAM"],
    "CIRC": ["CIRCULATE", "CIRC"],
    "BOP": ["BOP", "BLOWOUT PREVENTER"],
    "SAFE": ["SAFETY", "SAFETY MEETING"],
    "SRFEQ": ["SURFACE EQUIPMENT"],
    "WLHD": ["WELLHEAD"],
    "WAIT": ["WAIT", "WAITING"],
    "RIGMT": ["RIG MAINTENANCE"],
}


def load_model(checkpoint_path: str, device: str = "cuda") -> DualEncoderModel:
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = DualEncoderModel(
        ts_encoder_name=config["ts_encoder_name"],
        text_encoder_name=config["text_encoder_name"],
        projection_dim=config["projection_dim"],
        projector_type=config["projector_type"],
        temperature=config["temperature"],
        learnable_temperature=config["learnable_temperature"],
        ts_pooling=config["ts_pooling"],
        text_pooling=config.get("text_pooling", "auto"),
        freeze_text_encoder=False,
        freeze_ts_encoder=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract all TS and text embeddings + metadata from a dataloader."""
    all_ts_emb = []
    all_txt_emb = []
    all_codes = []
    all_texts = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        ts = batch["time_series"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        ts_emb = model.encode_time_series(ts)
        txt_emb = model.encode_text(input_ids, attention_mask)

        all_ts_emb.append(ts_emb.cpu())
        all_txt_emb.append(txt_emb.cpu())
        all_codes.extend(batch["codes"])
        # Decode text back for display
        all_texts.extend(
            model.get_tokenizer().batch_decode(input_ids.cpu(), skip_special_tokens=True)
        )

    ts_emb = torch.cat(all_ts_emb, dim=0)    # [N, D]
    txt_emb = torch.cat(all_txt_emb, dim=0)   # [N, D]
    return ts_emb, txt_emb, all_codes, all_texts


def compute_retrieval_metrics(ts_emb, txt_emb, codes, ks=(1, 10, 100)):
    """Compute pair-level recall and class-level F1 for cross-modal retrieval."""
    N = ts_emb.size(0)

    # Cosine similarity matrix
    sim = ts_emb @ txt_emb.T  # [N, N] — already L2-normalized

    results = {}

    for direction, sims in [("ts2txt", sim), ("txt2ts", sim.T)]:
        # Pair-level recall
        ranks = (sims.argsort(dim=1, descending=True) == torch.arange(N).unsqueeze(1)).nonzero(as_tuple=True)[1]

        for k in ks:
            recall = (ranks < k).float().mean().item() * 100
            results[f"{direction}_recall@{k}"] = recall

        # Class-level precision/recall/F1
        for k in [1, 10]:
            top_k_indices = sims.argsort(dim=1, descending=True)[:, :k]
            correct = 0
            total = 0
            for i in range(N):
                query_code = codes[i]
                retrieved_codes = [codes[j] for j in top_k_indices[i]]
                correct += sum(1 for rc in retrieved_codes if rc == query_code)
                total += k
            precision = correct / total if total > 0 else 0
            results[f"{direction}_class_precision@{k}"] = precision * 100

    # Average both directions
    for k in ks:
        results[f"avg_recall@{k}"] = (
            results[f"ts2txt_recall@{k}"] + results[f"txt2ts_recall@{k}"]
        ) / 2

    for k in [1, 10]:
        results[f"avg_class_precision@{k}"] = (
            results[f"ts2txt_class_precision@{k}"] + results[f"txt2ts_class_precision@{k}"]
        ) / 2

    return results


def compute_zero_shot(model, ts_emb, codes, device):
    """Zero-shot classification using text prompts as class anchors."""
    tokenizer = model.get_tokenizer()

    # Build class embeddings from prompts
    class_names = []
    class_embeddings = []

    for cls_name, prompts in ZERO_SHOT_PROMPTS.items():
        tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        with torch.no_grad():
            emb = model.encode_text(input_ids, attention_mask)  # [num_prompts, D]
            # Average prompts for this class
            cls_emb = emb.mean(dim=0)  # [D]

        class_names.append(cls_name)
        class_embeddings.append(cls_emb.cpu())

    class_emb_matrix = torch.stack(class_embeddings)  # [C, D]
    class_emb_matrix = torch.nn.functional.normalize(class_emb_matrix, dim=-1)

    # Classify each sample
    sim = ts_emb @ class_emb_matrix.T  # [N, C]
    preds = sim.argmax(dim=1)  # [N]

    # Map codes to class indices
    code_to_idx = {name: i for i, name in enumerate(class_names)}

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for i, code in enumerate(codes):
        if code in code_to_idx:
            total += 1
            per_class_total[code] += 1
            if preds[i].item() == code_to_idx[code]:
                correct += 1
                per_class_correct[code] += 1

    accuracy = correct / total * 100 if total > 0 else 0

    per_class_acc = {}
    for cls in sorted(per_class_total.keys()):
        if per_class_total[cls] > 0:
            per_class_acc[cls] = per_class_correct[cls] / per_class_total[cls] * 100

    return {
        "zero_shot_accuracy": accuracy,
        "zero_shot_total": total,
        "zero_shot_correct": correct,
        "per_class_accuracy": per_class_acc,
        "classes_used": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate contrastive dual-encoder model")
    parser.add_argument("--checkpoint", type=str, default="results/contrastive_full/best_model.pt")
    parser.add_argument("--eval_dir", type=str, default=None, help="Eval data directory")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=65536)
    parser.add_argument("--subsample", type=int, default=512)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(args.checkpoint, device)
    tokenizer = model.get_tokenizer()

    # Load eval dataset
    eval_dir = args.eval_dir
    if eval_dir is None:
        # Try default
        for p in [
            "/home/yahia.shaaban/project/drilling data/eval",
            "drilling data/eval",
            "../drilling data/eval",
        ]:
            if os.path.isdir(p):
                eval_dir = p
                break
    if eval_dir is None:
        raise ValueError("No eval_dir specified and default paths not found")

    dataset = DrillingContrastiveDataset(
        data_dir=eval_dir,
        window_size=args.window_size,
        subsample=args.subsample,
        max_files=args.max_files,
    )
    collate_fn = partial(collate_contrastive, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"\nEval samples: {len(dataset)}")
    print(f"Unique codes: {set(s[4] for s in dataset.index)}")

    # Extract embeddings
    ts_emb, txt_emb, codes, texts = extract_embeddings(model, loader, device)
    print(f"Extracted {ts_emb.size(0)} embeddings of dim {ts_emb.size(1)}")

    # 1. Cross-modal retrieval
    print("\n" + "=" * 60)
    print("  CROSS-MODAL RETRIEVAL")
    print("=" * 60)
    retrieval = compute_retrieval_metrics(ts_emb, txt_emb, codes)
    for k, v in sorted(retrieval.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}%")

    # 2. Zero-shot classification
    print("\n" + "=" * 60)
    print("  ZERO-SHOT CLASSIFICATION")
    print("=" * 60)
    zs = compute_zero_shot(model, ts_emb, codes, device)
    print(f"  Overall accuracy: {zs['zero_shot_accuracy']:.2f}% ({zs['zero_shot_correct']}/{zs['zero_shot_total']})")
    print(f"\n  Per-class accuracy:")
    for cls, acc in sorted(zs["per_class_accuracy"].items(), key=lambda x: -x[1]):
        print(f"    {cls:10s}: {acc:.1f}%")

    # Save results
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "retrieval": retrieval,
        "zero_shot": {
            "accuracy": zs["zero_shot_accuracy"],
            "total": zs["zero_shot_total"],
            "correct": zs["zero_shot_correct"],
            "per_class": zs["per_class_accuracy"],
        },
        "num_samples": len(dataset),
        "checkpoint": args.checkpoint,
    }

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
