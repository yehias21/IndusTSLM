# SPDX-License-Identifier: MIT
"""Temporal experts producing supervision targets for CoTT.

Each function takes a raw 1-D window `x: Tensor[L]` (already normalised the
same way the OpenTSLM HAR/Sleep/ECG loaders do) and returns a dict of tensors
that the corresponding CoTT decoder reconstructs against.

We pre-compute change-point masks and STFTs on the CPU at dataset-load time
(cached to disk); forecast and semantic targets are produced online from a
frozen Chronos-2 encoder because its forward pass is cheap.

Mapping (CoVT paper §3.3 → here):
    SAM           → change-point segmenter (PELT/RBF)
    DepthAnything → Chronos-2 forecaster (self-distillation)
    PIDINet       → STFT log-magnitude + first difference
    DINOv2        → Chronos-2 patch features
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Change-point expert — supervision only (no learned model; labels from PELT)
# ---------------------------------------------------------------------------

def change_point_mask(x: np.ndarray, max_segments: int = 8,
                      pen: float = 3.0) -> np.ndarray:
    """Return a binary mask of length L with 1 at change-point indices.

    Uses `ruptures` if available; otherwise falls back to a cheap gradient-
    threshold heuristic so the code path is still exercised without the
    optional dependency.
    """
    L = x.shape[-1]
    mask = np.zeros(L, dtype=np.float32)
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf", min_size=max(4, L // (max_segments * 4)))
        algo.fit(x.reshape(-1, 1))
        bkps = algo.predict(pen=pen)[:-1]
        bkps = bkps[:max_segments]
        for b in bkps:
            if 0 < b < L:
                mask[b] = 1.0
    except Exception:
        d = np.abs(np.diff(x, prepend=x[..., :1]))
        thr = np.quantile(d, 1.0 - max_segments / L)
        mask[d >= thr] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Spectral expert — online STFT (no heavy deps; analog of PIDINet edges)
# ---------------------------------------------------------------------------

def log_stft(x: torch.Tensor, n_fft: int = 64, hop: int = 16) -> torch.Tensor:
    """Return log|STFT| with shape (F, T'). Pads x if too short."""
    if x.numel() < n_fft:
        x = F.pad(x, (0, n_fft - x.numel()))
    window = torch.hann_window(n_fft, device=x.device)
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window,
                   return_complex=True, center=True)
    return torch.log1p(S.abs())


def first_difference(x: torch.Tensor) -> torch.Tensor:
    """First temporal derivative (edge-like signal)."""
    return F.pad(x[..., 1:] - x[..., :-1], (1, 0))


# ---------------------------------------------------------------------------
# Forecast + semantic experts — use the Chronos-2 encoder already supported
# by the repo. We import lazily so that CoTT can still be unit-tested on a
# machine without the full Chronos-2 stack.
# ---------------------------------------------------------------------------

_CHRONOS_SINGLETON = None

def get_chronos(device: str = "cuda"):
    global _CHRONOS_SINGLETON
    if _CHRONOS_SINGLETON is None:
        from industslm.model.encoder.TransformerCNNEncoder import (  # noqa: F401
            TransformerCNNEncoder,
        )
        try:
            from chronos import ChronosPipeline  # type: ignore
            pipe = ChronosPipeline.from_pretrained(
                "amazon/chronos-2", device_map=device
            )
            _CHRONOS_SINGLETON = pipe
        except Exception as err:
            raise RuntimeError(
                "Chronos-2 is required for forecast/semantic CoTT experts "
                "but could not be loaded: %r" % (err,)
            )
    return _CHRONOS_SINGLETON


@dataclass
class ExpertTargets:
    cp_mask: torch.Tensor         # [L]
    spectral: torch.Tensor        # [F, T']  log|STFT|
    derivative: torch.Tensor      # [L]
    forecast: torch.Tensor        # [H]   Chronos-2 teacher forecast
    semantic: torch.Tensor        # [P, D] Chronos-2 patch features


def compute_expert_targets(x: torch.Tensor,
                           horizon: int = 32,
                           device: str = "cpu",
                           use_chronos: bool = True) -> ExpertTargets:
    """Compute all four expert targets for a single-channel window.

    `x` : FloatTensor of shape [L], already normalised.
    Returns ExpertTargets where every tensor is on CPU (caller moves to GPU).
    """
    assert x.dim() == 1, f"expected 1-D window, got {tuple(x.shape)}"
    x_np = x.detach().cpu().numpy()

    cp = torch.from_numpy(change_point_mask(x_np)).float()
    sp = log_stft(x.float()).float()
    dx = first_difference(x.float()).float()

    if use_chronos:
        pipe = get_chronos(device)
        with torch.no_grad():
            # forecast = median of Chronos-2 quantile forecast
            q = pipe.predict(x.float().unsqueeze(0), horizon)
            fc = q.median(dim=1).values.squeeze(0).float()
            # semantic features: use the encoder directly (Chronos exposes it)
            feats = pipe.embed(x.float().unsqueeze(0)).squeeze(0).float()
    else:
        fc = torch.zeros(horizon)
        feats = torch.zeros(x.numel() // 16 + 1, 64)

    return ExpertTargets(
        cp_mask=cp, spectral=sp, derivative=dx,
        forecast=fc, semantic=feats,
    )
