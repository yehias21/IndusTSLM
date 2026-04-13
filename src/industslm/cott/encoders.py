# SPDX-License-Identifier: MIT
"""Time-series encoders exposing the *patch* interface that Flamingo's
perceiver expects: `forward(x)` returns a tensor of shape `[B, N_patches, D]`.

Two backends:
  * `CNNPatchEncoder` — the original `CNNTokenizer` (trained from scratch).
  * `ChronosPatchEncoder` — wraps Amazon's Chronos-2 (or any Chronos T5
     variant) and exposes its per-token encoder hidden states as patches.

Both inherit from `TimeSeriesEncoderBase` so they can drop straight into
`TimeSeriesFlamingoWithTrainableEncoder`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from industslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase
from industslm.model.encoder.CNNTokenizer import CNNTokenizer


class CNNPatchEncoder(CNNTokenizer):
    """Alias preserved for API symmetry."""


class ChronosPatchEncoder(TimeSeriesEncoderBase):
    """Wraps a Chronos T5 encoder (incl. Chronos-2) as a patch encoder.

    Forward:
        x: [B, L]  (already normalised, single-channel as produced by the
                    OpenTSLM collate pipeline).
    Returns:
        patches: [B, N, D] where N = number of Chronos encoder tokens and
                 D = Chronos `d_model`. D is exposed as `self.output_dim`
                 so the Flamingo wrapper can set `vis_dim=D` correctly.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        freeze: bool = True,
        project_to: int | None = None,
    ):
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        d_model = getattr(cfg, "d_model",
                          getattr(cfg, "hidden_size", 512))
        super().__init__(output_dim=project_to or d_model, dropout=0.0)

        self.model = AutoModel.from_pretrained(model_id,
                                               trust_remote_code=True)
        self._d_chronos = d_model
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        # Optional projection so vis_dim can be kept small (matches CoVT's
        # use of a lightweight projector after the frozen DINOv2 / DepthAny).
        self.project = (nn.Linear(d_model, project_to)
                        if project_to and project_to != d_model else nn.Identity())

    # ------------------------------------------------------------------
    @property
    def d_chronos(self) -> int:
        return self._d_chronos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:                       # [B, L]
            B, L = x.shape
        else:
            raise ValueError(f"Expected [B, L], got {tuple(x.shape)}")

        # Chronos T5 expects inputs_embeds of shape [B, L, d_model]. We
        # broadcast the scalar series into that space, same trick as in
        # `dual_encoder.ts_encoder_wrapper.ChronosWrapper.forward`.
        dtype = next(self.model.parameters()).dtype
        inputs_embeds = x.to(dtype).unsqueeze(-1).expand(-1, -1, self._d_chronos)

        encoder = getattr(self.model, "encoder", self.model)
        with torch.set_grad_enabled(self.training and any(
                p.requires_grad for p in self.model.parameters())):
            out = encoder(inputs_embeds=inputs_embeds)
            hidden = out.last_hidden_state      # [B, L, D]

        return self.project(hidden)
