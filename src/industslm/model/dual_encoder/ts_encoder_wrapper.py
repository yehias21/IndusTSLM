# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Unified wrappers for time series encoders used in the dual-encoder (contrastive) model.

Each wrapper produces a single vector per sample: [B, D].
Supported backends: TransformerCNN (built-in), Chronos, Moirai, MOMENT.
"""

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class TSEncoderWrapper(nn.Module):
    """Base class for all time series encoder wrappers."""

    output_dim: int  # set by subclass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] multivariate time series (C channels, T timesteps)
               or [B, T] univariate
        Returns:
            [B, D] pooled representation
        """
        ...

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Built-in TransformerCNN encoder
# ---------------------------------------------------------------------------
class TransformerCNNEncoderWrapper(TSEncoderWrapper):
    """
    Wraps the existing TransformerCNNEncoder from OpenTSLM.
    Handles multivariate input by encoding each channel independently
    and mean-pooling across channels, then mean-pooling over patches.
    """

    def __init__(self, pooling: str = "mean", **encoder_kwargs):
        super().__init__()
        from industslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder

        self.encoder = TransformerCNNEncoder(**encoder_kwargs)
        self.output_dim = self.encoder.output_dim
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] or [B, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        B, C, T = x.shape
        # Encode each channel: flatten to [B*C, T], get [B*C, N, D]
        patches = self.encoder(x.reshape(B * C, T))
        # Pool over patch dim -> [B*C, D]
        if self.pooling == "mean":
            pooled = patches.mean(dim=1)
        elif self.pooling == "last":
            pooled = patches[:, -1]
        else:
            pooled = patches.mean(dim=1)

        # Reshape back and pool over channels -> [B, D]
        pooled = pooled.view(B, C, -1).mean(dim=1)
        return pooled


# ---------------------------------------------------------------------------
# Chronos (Amazon)
# ---------------------------------------------------------------------------
class ChronosWrapper(TSEncoderWrapper):
    """
    Wraps a Chronos model from HuggingFace (e.g. amazon/chronos-t5-small).
    Extracts encoder hidden states and mean-pools.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-small",
        pooling: str = "mean",
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )
        # Chronos T5 models: encoder hidden size
        self.output_dim = getattr(config, "d_model", config.hidden_size if hasattr(config, "hidden_size") else 512)
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] — treat each channel independently and average
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, C, T = x.shape
        all_embeds = []
        for c in range(C):
            channel = x[:, c, :]  # [B, T]
            # Chronos T5: use encoder directly
            if hasattr(self.model, "encoder"):
                # Create dummy input_ids from quantized time series
                out = self.model.encoder(inputs_embeds=channel.unsqueeze(-1).expand(-1, -1, self.output_dim))
                hidden = out.last_hidden_state  # [B, T, D]
            else:
                out = self.model(channel.unsqueeze(-1))
                hidden = out.last_hidden_state

            if self.pooling == "mean":
                all_embeds.append(hidden.mean(dim=1))
            else:
                all_embeds.append(hidden[:, -1])

        return torch.stack(all_embeds, dim=1).mean(dim=1)  # [B, D]


# ---------------------------------------------------------------------------
# Moirai (Salesforce)
# ---------------------------------------------------------------------------
class MoiraiWrapper(TSEncoderWrapper):
    """
    Wraps a Moirai model from HuggingFace (e.g. Salesforce/moirai-1.0-R-small).
    Extracts hidden representations and mean-pools.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-1.0-R-small",
        pooling: str = "mean",
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.output_dim = getattr(
            config, "d_model",
            getattr(config, "hidden_size", 512)
        )
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Moirai expects multivariate input; adapt as needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Use the model's forward or encode method
        out = self.model(x)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        return hidden[:, -1]


# ---------------------------------------------------------------------------
# MOMENT (AutonLab)
# ---------------------------------------------------------------------------
class MomentWrapper(TSEncoderWrapper):
    """
    Wraps a MOMENT model from HuggingFace (e.g. AutonLab/MOMENT-1-large).
    Extracts hidden representations and mean-pools.
    """

    def __init__(
        self,
        model_id: str = "AutonLab/MOMENT-1-large",
        pooling: str = "mean",
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.output_dim = getattr(
            config, "d_model",
            getattr(config, "hidden_size", 512)
        )
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.model(x)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        return hidden[:, -1]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
# Shorthand name -> (class, default model_id)
_TS_ENCODER_REGISTRY = {
    "transformer_cnn": (TransformerCNNEncoderWrapper, None),
    "chronos": (ChronosWrapper, "amazon/chronos-t5-small"),
    "chronos-small": (ChronosWrapper, "amazon/chronos-t5-small"),
    "chronos-base": (ChronosWrapper, "amazon/chronos-t5-base"),
    "chronos-large": (ChronosWrapper, "amazon/chronos-t5-large"),
    "moirai": (MoiraiWrapper, "Salesforce/moirai-1.0-R-small"),
    "moirai-small": (MoiraiWrapper, "Salesforce/moirai-1.0-R-small"),
    "moirai-large": (MoiraiWrapper, "Salesforce/moirai-1.0-R-large"),
    "moment": (MomentWrapper, "AutonLab/MOMENT-1-large"),
    "moment-small": (MomentWrapper, "AutonLab/MOMENT-1-small"),
    "moment-large": (MomentWrapper, "AutonLab/MOMENT-1-large"),
}


def create_ts_encoder(name: str, pooling: str = "mean", **kwargs) -> TSEncoderWrapper:
    """
    Factory function to create a time series encoder wrapper.

    Args:
        name: One of the registered names (e.g. 'transformer_cnn', 'chronos', 'moirai', 'moment')
              or a HuggingFace model ID for auto-detection.
        pooling: Pooling strategy ('mean' or 'last').
        **kwargs: Additional keyword args passed to the wrapper constructor.

    Returns:
        A TSEncoderWrapper instance.
    """
    if name in _TS_ENCODER_REGISTRY:
        cls, default_model_id = _TS_ENCODER_REGISTRY[name]
        if default_model_id is not None:
            return cls(model_id=kwargs.pop("model_id", default_model_id), pooling=pooling, **kwargs)
        else:
            return cls(pooling=pooling, **kwargs)

    # Fallback: treat name as a HuggingFace model ID
    name_lower = name.lower()
    if "chronos" in name_lower:
        return ChronosWrapper(model_id=name, pooling=pooling, **kwargs)
    elif "moirai" in name_lower:
        return MoiraiWrapper(model_id=name, pooling=pooling, **kwargs)
    elif "moment" in name_lower:
        return MomentWrapper(model_id=name, pooling=pooling, **kwargs)
    else:
        raise ValueError(
            f"Unknown TS encoder: '{name}'. "
            f"Registered names: {list(_TS_ENCODER_REGISTRY.keys())}. "
            f"Or pass a HuggingFace model ID containing 'chronos', 'moirai', or 'moment'."
        )
