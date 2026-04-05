# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
DualEncoderModel: contrastive dual-encoder for time series and text (DriMM-style).

Composes a time series encoder, a text encoder, projection heads,
and an InfoNCE loss into a single nn.Module.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts_encoder_wrapper import TSEncoderWrapper, create_ts_encoder
from .text_encoder_wrapper import TextEncoderWrapper, create_text_encoder
from .losses import create_loss


class DualEncoderModel(nn.Module):
    """
    Dual-encoder contrastive model aligning time series and text in a shared embedding space.

    Architecture (following DriMM):
      TS input  -> ts_encoder  -> ts_projector  -> L2-norm -> ts_emb   [B, projection_dim]
      Text input -> text_encoder -> text_projector -> L2-norm -> text_emb [B, projection_dim]
      Loss = symmetric InfoNCE(ts_emb, text_emb)
    """

    def __init__(
        self,
        ts_encoder_name: str = "transformer_cnn",
        text_encoder_name: str = "qwen3.5",
        projection_dim: int = 256,
        projector_type: str = "linear",
        loss_type: str = "siglip",
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        ts_pooling: str = "mean",
        text_pooling: str = "auto",
        freeze_text_encoder: bool = False,
        freeze_ts_encoder: bool = False,
        ts_encoder_kwargs: Optional[dict] = None,
        text_encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        # --- Time series encoder ---
        ts_kw = ts_encoder_kwargs or {}
        self.ts_encoder: TSEncoderWrapper = create_ts_encoder(
            ts_encoder_name, pooling=ts_pooling, **ts_kw
        )

        # --- Text encoder ---
        txt_kw = text_encoder_kwargs or {}
        self.text_encoder: TextEncoderWrapper = create_text_encoder(
            text_encoder_name, pooling=text_pooling, **txt_kw
        )

        # --- Projection heads (no device placement — Accelerate handles it) ---
        ts_in_dim = self.ts_encoder.output_dim
        txt_in_dim = self.text_encoder.output_dim

        if projector_type == "linear":
            self.ts_projector = nn.Sequential(
                nn.LayerNorm(ts_in_dim),
                nn.Linear(ts_in_dim, projection_dim),
            )
            self.text_projector = nn.Sequential(
                nn.LayerNorm(txt_in_dim),
                nn.Linear(txt_in_dim, projection_dim),
            )
        elif projector_type == "mlp":
            self.ts_projector = nn.Sequential(
                nn.LayerNorm(ts_in_dim),
                nn.Linear(ts_in_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
            self.text_projector = nn.Sequential(
                nn.LayerNorm(txt_in_dim),
                nn.Linear(txt_in_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            raise ValueError(f"Unknown projector_type: {projector_type}")

        # --- Loss ---
        self.loss_fn = create_loss(
            loss_type=loss_type,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

        # --- Freezing ---
        if freeze_ts_encoder:
            self.ts_encoder.freeze()
        if freeze_text_encoder:
            self.text_encoder.freeze()

        # Store config for checkpointing / logging
        self.config = {
            "ts_encoder_name": ts_encoder_name,
            "text_encoder_name": text_encoder_name,
            "projection_dim": projection_dim,
            "projector_type": projector_type,
            "loss_type": loss_type,
            "temperature": temperature,
            "learnable_temperature": learnable_temperature,
            "ts_pooling": ts_pooling,
            "text_pooling": text_pooling,
            "freeze_text_encoder": freeze_text_encoder,
            "freeze_ts_encoder": freeze_ts_encoder,
        }

    def encode_time_series(self, time_series: torch.Tensor) -> torch.Tensor:
        """Encode time series to L2-normalized projection space. [B, C, T] -> [B, D]"""
        ts_emb = self.ts_encoder(time_series)
        ts_proj = self.ts_projector(ts_emb)
        return F.normalize(ts_proj, dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode text to L2-normalized projection space. [B, L] -> [B, D]"""
        text_emb = self.text_encoder(input_ids, attention_mask)
        text_proj = self.text_projector(text_emb)
        return F.normalize(text_proj, dim=-1)

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both projected embeddings.

        Returns:
            (ts_proj, text_proj) — both [B, projection_dim], L2-normalized.
        """
        ts_proj = self.encode_time_series(time_series)
        text_proj = self.encode_text(input_ids, attention_mask)
        return ts_proj, text_proj

    def compute_loss(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss."""
        ts_proj, text_proj = self.forward(time_series, input_ids, attention_mask)
        return self.loss_fn(ts_proj, text_proj)

    # --- Convenience for inference ---

    @torch.no_grad()
    def get_ts_embeddings(self, time_series: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized TS embeddings for retrieval / zero-shot."""
        self.eval()
        return self.encode_time_series(time_series)

    @torch.no_grad()
    def get_text_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get L2-normalized text embeddings for retrieval / zero-shot."""
        self.eval()
        return self.encode_text(input_ids, attention_mask)

    def get_tokenizer(self):
        """Return the text encoder's tokenizer (useful for dataset collation)."""
        return self.text_encoder.tokenizer
