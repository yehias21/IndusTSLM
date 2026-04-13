# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Contrastive pretraining runner for DualEncoderModel (DriMM-style).

Uses HuggingFace Accelerate with optional DeepSpeed Zero (1/2/3).

Usage examples:
  # Single GPU, default (TransformerCNN + RoBERTa)
  python contrastive_pretraining.py --train_dir ../drilling\\ data/train --eval_dir ../drilling\\ data/eval

  # Multi-GPU with Accelerate
  accelerate launch contrastive_pretraining.py --train_dir ... --eval_dir ...

  # DeepSpeed Zero-2
  accelerate launch --use_deepspeed contrastive_pretraining.py --deepspeed zero2 --train_dir ... --eval_dir ...

  # Moirai + Qwen3, frozen text encoder
  python contrastive_pretraining.py --ts_encoder moirai --text_encoder qwen3 --freeze_text_encoder --train_dir ...
"""

import argparse
import json
import os
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from industslm.model.dual_encoder.DualEncoder import DualEncoderModel
from industslm.model.dual_encoder.drilling_dataset import (
    DrillingContrastiveDataset,
    collate_contrastive,
)
from industslm.model.dual_encoder.losses import gather_embeddings
from industslm.model.dual_encoder.negative_mining import create_negative_strategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contrastive pretraining for dual-encoder time series + text model"
    )

    # --- Data ---
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training parquet directory")
    parser.add_argument("--eval_dir", type=str, default=None, help="Path to eval parquet directory")
    parser.add_argument("--window_size", type=int, default=65536, help="Window size for time series segments")
    parser.add_argument("--subsample", type=int, default=512, help="Subsampled length per window")
    parser.add_argument("--max_train_files", type=int, default=None, help="Limit training files (for debugging)")
    parser.add_argument("--max_eval_files", type=int, default=None, help="Limit eval files")

    # --- Model ---
    parser.add_argument("--ts_encoder", type=str, default="transformer_cnn",
                        help="Time series encoder: transformer_cnn, chronos, moirai, moment, or HF model ID")
    parser.add_argument("--text_encoder", type=str, default="qwen3.5",
                        help="Text encoder: qwen3.5, roberta, qwen3, bert, or any HF model ID")
    parser.add_argument("--projection_dim", type=int, default=256, help="Shared embedding dimension")
    parser.add_argument("--projector_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--loss_type", type=str, default="siglip", choices=["infonce", "siglip"],
                        help="Contrastive loss: infonce (CLIP-style) or siglip (per-pair sigmoid)")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive loss temperature")
    parser.add_argument("--learnable_temperature", action="store_true")
    parser.add_argument("--ts_pooling", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--text_pooling", type=str, default="auto", choices=["auto", "cls", "last_token", "mean"])
    parser.add_argument("--freeze_text_encoder", action="store_true", help="Freeze text encoder weights")
    parser.add_argument("--freeze_ts_encoder", action="store_true", help="Freeze TS encoder weights")
    parser.add_argument("--negative_strategy", type=str, default="none",
                        help="Negative mining strategy: none, text_swap, number_perturb, in_batch_hard, "
                             "or compose with '+' e.g. 'text_swap+number_perturb'")

    # --- Training ---
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_fraction", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.005)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # --- Accelerate / DeepSpeed ---
    parser.add_argument("--deepspeed", type=str, default="none",
                        choices=["none", "zero1", "zero2", "zero3"],
                        help="DeepSpeed Zero stage (requires accelerate launch --use_deepspeed)")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="results/contrastive")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint directory")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def build_deepspeed_plugin(zero_stage: str, mixed_precision: str):
    """Build Accelerate DeepSpeedPlugin programmatically."""
    if zero_stage == "none":
        return None

    from accelerate import DeepSpeedPlugin

    stage_map = {"zero1": 1, "zero2": 2, "zero3": 3}
    stage = stage_map[zero_stage]

    ds_config = {
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
        },
        "gradient_clipping": 1.0,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
    }

    if stage >= 2:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    if stage == 3:
        ds_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif mixed_precision == "fp16":
        ds_config["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}

    return DeepSpeedPlugin(hf_ds_config=ds_config)


def train(args):
    from accelerate import Accelerator

    # --- Init Accelerator ---
    ds_plugin = build_deepspeed_plugin(args.deepspeed, args.mixed_precision)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision if ds_plugin is None else None,
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
    )
    is_main = accelerator.is_main_process

    if is_main:
        print("=" * 60)
        print("  Contrastive Pretraining (DriMM-style)")
        print("=" * 60)
        print(f"  TS encoder:    {args.ts_encoder}")
        print(f"  Text encoder:  {args.text_encoder}")
        print(f"  Projection:    {args.projection_dim}D ({args.projector_type})")
        print(f"  Loss:          {args.loss_type}")
        print(f"  Temperature:   {args.temperature} (learnable={args.learnable_temperature})")
        print(f"  Neg strategy:  {args.negative_strategy}")
        print(f"  Freeze TS:     {args.freeze_ts_encoder}")
        print(f"  Freeze Text:   {args.freeze_text_encoder}")
        print(f"  Batch size:    {args.batch_size} x {accelerator.num_processes} GPUs")
        print(f"  DeepSpeed:     {args.deepspeed}")
        print(f"  Mixed prec:    {args.mixed_precision}")
        print()

    # --- Negative mining strategy ---
    neg_strategy = create_negative_strategy(args.negative_strategy)
    if is_main:
        print(f"  Neg mining:    {neg_strategy.name}")

    # --- Model ---
    model = DualEncoderModel(
        ts_encoder_name=args.ts_encoder,
        text_encoder_name=args.text_encoder,
        projection_dim=args.projection_dim,
        projector_type=args.projector_type,
        loss_type=args.loss_type,
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        ts_pooling=args.ts_pooling,
        text_pooling=args.text_pooling,
        freeze_text_encoder=args.freeze_text_encoder,
        freeze_ts_encoder=args.freeze_ts_encoder,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)")
        print()

    # --- Datasets ---
    if is_main:
        print("Loading datasets...")

    train_dataset = DrillingContrastiveDataset(
        data_dir=args.train_dir,
        window_size=args.window_size,
        subsample=args.subsample,
        max_files=args.max_train_files,
    )

    eval_dataset = None
    if args.eval_dir:
        eval_dataset = DrillingContrastiveDataset(
            data_dir=args.eval_dir,
            window_size=args.window_size,
            subsample=args.subsample,
            max_files=args.max_eval_files,
        )

    tokenizer = model.get_tokenizer()

    # Wrap collate with negative mining: mine negatives on raw dicts, then collate
    def collate_with_negatives(batch, tokenizer=tokenizer, strategy=neg_strategy):
        batch = strategy.mine(batch)
        return collate_contrastive(batch, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_with_negatives,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=partial(collate_contrastive, tokenizer=tokenizer),
            num_workers=2,
            pin_memory=True,
        )

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.num_epochs * len(train_loader) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_fraction * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if is_main:
        print(f"  Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"  Eval samples:     {len(eval_dataset)}")
        print(f"  Steps/epoch:      {len(train_loader)}")
        print(f"  Total steps:      {total_steps}")
        print(f"  Warmup steps:     {warmup_steps}")
        print()

    # --- Prepare with Accelerate ---
    if eval_loader is not None:
        model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    # --- Resume ---
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_from and os.path.exists(args.resume_from):
        if is_main:
            print(f"Resuming from {args.resume_from}")
        accelerator.load_state(args.resume_from)
        # Try to load training metadata
        meta_path = os.path.join(args.resume_from, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            start_epoch = meta.get("epoch", 0) + 1
            best_val_loss = meta.get("best_val_loss", float("inf"))

    # --- Output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    loss_log_path = os.path.join(args.output_dir, "loss_history.tsv")

    if is_main and start_epoch == 1:
        with open(loss_log_path, "w") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")
        # Save config
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # --- Training Loop ---
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        prog = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            disable=not is_main,
        )

        for batch in prog:
            with accelerator.accumulate(model):
                ts = batch["time_series"]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                # Forward
                unwrapped = accelerator.unwrap_model(model)
                ts_proj, text_proj = model(ts, input_ids, attention_mask)

                # Gather for multi-GPU contrastive (larger effective batch)
                ts_proj_all = gather_embeddings(ts_proj, accelerator)
                text_proj_all = gather_embeddings(text_proj, accelerator)

                loss = unwrapped.loss_fn(ts_proj_all, text_proj_all)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            num_batches += 1

            if is_main:
                prog.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        avg_train_loss = running_loss / max(num_batches, 1)

        # --- Validation ---
        avg_val_loss = float("nan")
        if eval_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="Validating", disable=not is_main):
                    ts = batch["time_series"]
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]

                    unwrapped = accelerator.unwrap_model(model)
                    ts_proj, text_proj = model(ts, input_ids, attention_mask)
                    loss = unwrapped.loss_fn(ts_proj, text_proj)

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)

        if is_main:
            print(f"Epoch {epoch} -- train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

            # Log
            with open(loss_log_path, "a") as f:
                f.write(f"{epoch}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\n")

        # --- Early stopping & checkpointing ---
        if eval_loader is not None and avg_val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Save best checkpoint
            if is_main:
                print(f"  New best val_loss: {best_val_loss:.4f} -- saving checkpoint")

            ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
            accelerator.save_state(ckpt_dir)

            if is_main:
                # Save training metadata
                with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
                    json.dump({"epoch": epoch, "best_val_loss": best_val_loss}, f)

                # Also save standalone model weights (easier to load for inference)
                unwrapped = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "model_state_dict": unwrapped.state_dict(),
                        "config": unwrapped.config,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    os.path.join(args.output_dir, "best_model.pt"),
                )
        else:
            epochs_no_improve += 1
            if is_main and eval_loader is not None:
                print(f"  No improvement ({epochs_no_improve}/{args.early_stop_patience})")

            if epochs_no_improve >= args.early_stop_patience:
                if is_main:
                    print("\n  Early stopping triggered.")
                break

    # --- Save final model ---
    if is_main:
        print("\nTraining complete.")
        final_dir = os.path.join(args.output_dir, "final_checkpoint")
        accelerator.save_state(final_dir)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "model_state_dict": unwrapped.state_dict(),
                "config": unwrapped.config,
                "epoch": epoch,
            },
            os.path.join(args.output_dir, "final_model.pt"),
        )
        print(f"Final model saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    train(args)
