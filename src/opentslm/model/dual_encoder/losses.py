# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Contrastive losses for dual-encoder models.

Supported:
  - InfoNCE (CLIP-style): softmax cross-entropy over similarity matrix
  - SigLIP: per-pair sigmoid loss — better for fine-grained discrimination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP-style) contrastive loss.

    Given L2-normalized embeddings from two modalities,
    pulls matched pairs together and pushes mismatched pairs apart.
    Uses softmax over the full similarity matrix — the "which is best match?" formulation.
    """

    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = False):
        super().__init__()
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer("log_temperature", torch.tensor(temperature).log())

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp()

    def forward(
        self,
        ts_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ts_embeddings:   [B, D] L2-normalized
            text_embeddings: [B, D] L2-normalized
        Returns:
            Scalar loss.
        """
        logits = (ts_embeddings @ text_embeddings.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_ts2txt = F.cross_entropy(logits, labels)
        loss_txt2ts = F.cross_entropy(logits.T, labels)

        return (loss_ts2txt + loss_txt2ts) / 2


class SigLIPLoss(nn.Module):
    """
    Sigmoid contrastive loss (SigLIP, Zhai et al. 2023).

    Instead of softmax over the full matrix (InfoNCE), computes an independent
    binary sigmoid loss for EVERY pair in the batch. Each pair is asked:
    "is this a match or not?" rather than "which is the best match?"

    Advantages over InfoNCE:
      - Better fine-grained discrimination (important for numerical values)
      - No implicit assumption that all negatives are equally dissimilar
      - More stable with large batch sizes
      - Works better with hard negatives
    """

    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = False, bias: float = 0.0):
        super().__init__()
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer("log_temperature", torch.tensor(temperature).log())

        if learnable_temperature:
            self.bias = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer("bias", torch.tensor(bias))

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp()

    def forward(
        self,
        ts_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ts_embeddings:   [B, D] L2-normalized
            text_embeddings: [B, D] L2-normalized
        Returns:
            Scalar loss.
        """
        # Pairwise similarity: [B, B]
        logits = (ts_embeddings @ text_embeddings.T) / self.temperature + self.bias

        # Labels: +1 for diagonal (matched), -1 for off-diagonal (unmatched)
        B = logits.size(0)
        labels = 2 * torch.eye(B, device=logits.device) - 1  # [B, B] of +1/-1

        # Sigmoid loss: -log(sigmoid(label * logit)) per pair
        # = log(1 + exp(-label * logit))
        loss = -F.logsigmoid(labels * logits)

        return loss.mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_LOSS_REGISTRY = {
    "infonce": InfoNCELoss,
    "siglip": SigLIPLoss,
}


def create_loss(
    loss_type: str = "infonce",
    temperature: float = 0.07,
    learnable_temperature: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a contrastive loss.

    Args:
        loss_type: 'infonce' or 'siglip'
        temperature: Temperature parameter
        learnable_temperature: Whether temperature is learnable
    """
    if loss_type not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(_LOSS_REGISTRY.keys())}")

    return _LOSS_REGISTRY[loss_type](
        temperature=temperature,
        learnable_temperature=learnable_temperature,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Multi-GPU gathering
# ---------------------------------------------------------------------------
def gather_embeddings(embeddings: torch.Tensor, accelerator) -> torch.Tensor:
    """Gather embeddings across all processes for larger effective batch in contrastive loss."""
    if accelerator.num_processes <= 1:
        return embeddings
    return accelerator.gather(embeddings.contiguous())
