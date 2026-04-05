# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Unified wrappers for text encoders used in the dual-encoder (contrastive) model.

Supports any HuggingFace model: encoder-only (RoBERTa, BERT) and
decoder-only (Qwen3, LLaMA, GPT-2). Produces a single vector per sample: [B, D].
"""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class TextEncoderWrapper(nn.Module):
    """
    Generic text encoder wrapper for any HuggingFace model.

    Pooling strategies:
      - "cls":        use the [CLS] / first token embedding (good for encoder-only models)
      - "last_token": use the last non-padding token (good for decoder-only / causal LMs)
      - "mean":       mean-pool over non-padding tokens
      - "auto":       auto-detect based on model type
    """

    def __init__(
        self,
        model_id: str = "roberta-base",
        pooling: str = "auto",
        max_length: int = 512,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.model_id = model_id
        self.max_length = max_length

        # Load config to detect model type
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.is_causal = getattr(config, "is_decoder", False) or config.model_type in (
            "gpt2", "gpt_neo", "gpt_neox", "llama", "mistral",
            "qwen2", "qwen3", "qwen3_5", "qwen3.5",
            "phi", "phi3", "gemma", "gemma2", "gemma3", "starcoder2", "falcon",
        )

        # Auto-detect pooling
        if pooling == "auto":
            self.pooling = "last_token" if self.is_causal else "cls"
        else:
            self.pooling = pooling

        # Load model
        load_kwargs = {"trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
        self.output_dim = config.hidden_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, texts: List[str]) -> dict:
        """Tokenize a list of strings, returning input_ids and attention_mask tensors."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, L]
            attention_mask:  [B, L]
        Returns:
            [B, D] pooled text representation
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, L, D]

        if self.pooling == "cls":
            return hidden_states[:, 0]

        elif self.pooling == "last_token":
            # Index of the last non-padding token per sample
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]

        elif self.pooling == "mean":
            # Mean pool over non-padding positions
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            summed = (hidden_states * mask).sum(dim=1)   # [B, D]
            counts = mask.sum(dim=1).clamp(min=1)        # [B, 1]
            return summed / counts

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Shorthand name -> HuggingFace model ID
# ---------------------------------------------------------------------------
_TEXT_ENCODER_ALIASES = {
    "roberta": "roberta-base",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "bert": "bert-base-uncased",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3.5": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "llama-1b": "meta-llama/Llama-3.2-1B",
    "llama-3b": "meta-llama/Llama-3.2-3B",
}


def create_text_encoder(
    model_id: str,
    pooling: str = "auto",
    max_length: int = 512,
    torch_dtype: Optional[torch.dtype] = None,
) -> TextEncoderWrapper:
    """
    Factory function to create a text encoder wrapper.

    Args:
        model_id: A shorthand name (e.g. 'roberta', 'qwen3') or a HuggingFace model ID.
        pooling: Pooling strategy ('auto', 'cls', 'last_token', 'mean').
        max_length: Max sequence length for tokenization.
        torch_dtype: Optional torch dtype for model loading.

    Returns:
        A TextEncoderWrapper instance.
    """
    resolved_id = _TEXT_ENCODER_ALIASES.get(model_id, model_id)
    return TextEncoderWrapper(
        model_id=resolved_id,
        pooling=pooling,
        max_length=max_length,
        torch_dtype=torch_dtype,
    )
