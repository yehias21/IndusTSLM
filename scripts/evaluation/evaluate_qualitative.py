# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Qualitative retrieval examples: shows ground truth vs top retrieved for both directions.
"""

import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from industslm.model.dual_encoder.DualEncoder import DualEncoderModel
from industslm.model.dual_encoder.drilling_dataset import (
    DrillingContrastiveDataset,
    collate_contrastive,
)


def load_model(checkpoint_path, device="cuda"):
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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def extract_all(model, loader, device):
    ts_embs, txt_embs, codes, texts = [], [], [], []
    for batch in tqdm(loader, desc="Extracting"):
        ts = batch["time_series"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        ts_embs.append(model.encode_time_series(ts).cpu())
        txt_embs.append(model.encode_text(ids, mask).cpu())
        codes.extend(batch["codes"])
        texts.extend(
            model.get_tokenizer().batch_decode(ids.cpu(), skip_special_tokens=True)
        )
    return torch.cat(ts_embs), torch.cat(txt_embs), codes, texts


def show_retrieval_examples(ts_emb, txt_emb, codes, texts, n_examples=5, top_k=5):
    N = ts_emb.size(0)
    sim = ts_emb @ txt_emb.T  # [N, N]

    # --- TS -> Text ---
    print("\n" + "=" * 80)
    print("  TIME SERIES -> TEXT RETRIEVAL (top-{})".format(top_k))
    print("=" * 80)

    # Pick diverse query indices (one per unique code, up to n_examples)
    seen_codes = set()
    query_indices = []
    for i in range(N):
        if codes[i] not in seen_codes and len(query_indices) < n_examples:
            seen_codes.add(codes[i])
            query_indices.append(i)

    for qi in query_indices:
        scores = sim[qi]  # [N]
        top_ids = scores.argsort(descending=True)[:top_k]

        print(f"\n{'─' * 80}")
        print(f"  QUERY (index={qi})")
        print(f"  Code:  {codes[qi]}")
        print(f"  Text:  {texts[qi][:120]}")
        print(f"  {'─' * 76}")

        for rank, ti in enumerate(top_ids):
            match = "MATCH" if codes[ti.item()] == codes[qi] else "     "
            print(
                f"  Rank {rank+1} | Sim={scores[ti]:.4f} | Code={codes[ti.item()]:8s} | {match} | {texts[ti.item()][:100]}"
            )

    # --- Text -> TS ---
    print("\n" + "=" * 80)
    print("  TEXT -> TIME SERIES RETRIEVAL (top-{})".format(top_k))
    print("=" * 80)

    seen_codes2 = set()
    query_indices2 = []
    for i in range(N):
        if codes[i] not in seen_codes2 and len(query_indices2) < n_examples:
            seen_codes2.add(codes[i])
            query_indices2.append(i)

    sim_t2ts = txt_emb @ ts_emb.T  # [N, N]

    for qi in query_indices2:
        scores = sim_t2ts[qi]
        top_ids = scores.argsort(descending=True)[:top_k]

        print(f"\n{'─' * 80}")
        print(f"  QUERY TEXT (index={qi})")
        print(f"  Code:  {codes[qi]}")
        print(f"  Text:  {texts[qi][:120]}")
        print(f"  {'─' * 76}")
        print(f"  Top {top_k} retrieved TIME SERIES (by code & similarity):")

        for rank, ti in enumerate(top_ids):
            match = "MATCH" if codes[ti.item()] == codes[qi] else "     "
            print(
                f"  Rank {rank+1} | Sim={scores[ti]:.4f} | Code={codes[ti.item()]:8s} | {match} | {texts[ti.item()][:100]}"
            )

    # --- Summary stats ---
    print("\n" + "=" * 80)
    print("  SUMMARY: Class match rate in top-k")
    print("=" * 80)

    for k in [1, 5, 10]:
        # TS->Text
        ts2txt_top = sim.argsort(dim=1, descending=True)[:, :k]
        ts2txt_match = sum(
            any(codes[j.item()] == codes[i] for j in ts2txt_top[i])
            for i in range(N)
        ) / N * 100

        # Text->TS
        t2ts_top = sim_t2ts.argsort(dim=1, descending=True)[:, :k]
        t2ts_match = sum(
            any(codes[j.item()] == codes[i] for j in t2ts_top[i])
            for i in range(N)
        ) / N * 100

        print(f"  Top-{k:2d}:  TS->Text class match = {ts2txt_match:.1f}%  |  Text->TS class match = {t2ts_match:.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "results/contrastive_full/best_model.pt"

    eval_dir = "/home/yahia.shaaban/project/drilling data/eval"

    model = load_model(ckpt, device)
    tokenizer = model.get_tokenizer()

    dataset = DrillingContrastiveDataset(
        data_dir=eval_dir, window_size=65536, subsample=512, max_files=10
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=partial(collate_contrastive, tokenizer=tokenizer),
        num_workers=2,
    )

    ts_emb, txt_emb, codes, texts = extract_all(model, loader, device)
    print(f"\n{ts_emb.size(0)} samples, {len(set(codes))} unique codes")

    show_retrieval_examples(ts_emb, txt_emb, codes, texts, n_examples=8, top_k=5)


if __name__ == "__main__":
    main()
