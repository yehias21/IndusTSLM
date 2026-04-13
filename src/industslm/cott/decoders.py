# SPDX-License-Identifier: MIT
"""Lightweight decoders that reconstruct each expert's target from the
projected CoTT tokens. Mirrors CoVT §3.3 / §A.2–A.4.

Shapes use the notation:
    B = batch size
    Nₖ = number of CoTT tokens of kind k (e.g. 8 for CP)
    d = LLM hidden size (after projection)
    L = raw-window length, H = forecast horizon, F×T' = STFT shape, P×D = patch features
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # for Hungarian matching


class CoTTProjection(nn.Module):
    """Linear + multi-head cross-attention + linear (CoVT §A.1).

    Maps N LLM hidden states `z ∈ R^{B×N×d_llm}` to expert-space tokens
    `z_p ∈ R^{B×N×d_expert}` via a learnable query that attends over z.
    """

    def __init__(self, d_llm: int, d_expert: int, num_heads: int = 4,
                 num_tokens: int = 8):
        super().__init__()
        self.linear_in = nn.Linear(d_llm, d_expert)
        self.query = nn.Parameter(torch.randn(1, num_tokens, d_expert) * 0.02)
        self.attn = nn.MultiheadAttention(d_expert, num_heads, batch_first=True)
        self.linear_out = nn.Linear(d_expert, d_expert)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        kv = self.linear_in(z)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, kv, kv)
        return self.linear_out(out)


# ---------------------------------------------------------------------------
# 1. Change-point decoder (SAM analog, LISA-style prompt alignment)
# ---------------------------------------------------------------------------

class CPEncoder(nn.Module):
    """Small 1-D CNN that takes the raw window and produces dense features."""

    def __init__(self, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, d // 2, 7, padding=3), nn.GELU(),
            nn.Conv1d(d // 2, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 3, padding=1),
        )

    def forward(self, x):  # [B, L] -> [B, d, L]
        return self.net(x.unsqueeze(1))


class CPDecoder(nn.Module):
    """Each of N_cp projected tokens prompts one candidate change-point mask.
    Hungarian matching against ground-truth mask pushes Dice + Focal loss
    (CoVT Eq. 10–14)."""

    def __init__(self, d_expert: int = 128, num_tokens: int = 8):
        super().__init__()
        self.enc = CPEncoder(d_expert)
        self.num_tokens = num_tokens
        self.mask_head = nn.Conv1d(d_expert, 1, 1)

    def forward(self, tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, d]; x: [B, L]  → masks: [B, N, L] in [0,1]."""
        f = self.enc(x)  # [B, d, L]
        # interaction: broadcast token over time, add features, then conv
        B, N, d = tokens.shape
        t = tokens.unsqueeze(-1)                  # [B, N, d, 1]
        f_exp = f.unsqueeze(1)                    # [B, 1, d, L]
        fused = f_exp + t                         # [B, N, d, L]
        fused = fused.reshape(B * N, d, -1)
        logits = self.mask_head(fused).squeeze(1) # [B*N, L]
        return torch.sigmoid(logits).view(B, N, -1)


def dice_loss(p, g, eps=1e-6):
    num = 2 * (p * g).sum(-1)
    den = p.sum(-1) + g.sum(-1) + eps
    return 1 - num / den


def focal_loss(p, g, gamma=2.0, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return -((1 - p) ** gamma * g * p.log()).mean(-1)


def cp_matching_loss(pred_masks: torch.Tensor, gt_mask: torch.Tensor):
    """pred_masks [B,N,L], gt_mask [B,L]. Construct ≤N GT "segments" around
    each change-point and Hungarian-match against predicted masks."""
    B, N, L = pred_masks.shape
    total = pred_masks.new_zeros(())
    for b in range(B):
        cps = (gt_mask[b] > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if cps.numel() == 0:
            continue
        # build GT masks: gaussian bump at each change-point
        idx = cps[:N]
        grid = torch.arange(L, device=gt_mask.device).float()
        gts = torch.stack([torch.exp(-((grid - c) ** 2) / (2 * 4.0 ** 2))
                           for c in idx])  # [n_cp, L]
        # pad to N rows with zeros (unmatched predictions free-float)
        pad = pred_masks.new_zeros(N - gts.shape[0], L)
        gts = torch.cat([gts, pad], 0)
        # cost: Dice + Focal per pair
        p = pred_masks[b].unsqueeze(1).expand(-1, N, -1)  # [N, N, L]
        g = gts.unsqueeze(0).expand(N, -1, -1)
        cost = dice_loss(p, g) + focal_loss(p, g)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        matched = cost[row, col].mean()
        total = total + matched
    return total / max(B, 1)


# ---------------------------------------------------------------------------
# 2. Forecast decoder (DepthAnything analog, BMM reconstruction)
# ---------------------------------------------------------------------------

class ForecastDecoder(nn.Module):
    """N_fc projected tokens × Chronos-2 mid-layer features → next-H values."""

    def __init__(self, d_expert: int, horizon: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.horizon = horizon
        self.feat_proj = nn.Linear(d_expert, d_expert)

    def forward(self, tokens: torch.Tensor,
                chronos_feats: torch.Tensor) -> torch.Tensor:
        """tokens [B,N,d], chronos_feats [B,N,P,d]  → fc [B,H]."""
        feats = self.feat_proj(chronos_feats)             # [B,N,P,d]
        # BMM:  token · featᵀ → [B,N,P]; softmax mix over P; project to H
        attn = torch.einsum("bnd,bnpd->bnp", tokens, feats)
        attn = attn.softmax(-1)
        # Re-project to the horizon by a fixed DCT basis (no learnable params
        # besides feat_proj, keeping the "N tokens as prompts" design)
        bases = torch.linspace(0, 1, self.horizon, device=tokens.device)
        kern = torch.cos(bases.unsqueeze(0) * torch.arange(
            attn.shape[-1], device=tokens.device).unsqueeze(-1))  # [P,H]
        reconstructed = torch.einsum("bnp,ph->bnh", attn, kern)
        return reconstructed.mean(1)                       # [B,H]


# ---------------------------------------------------------------------------
# 3. Spectral decoder (PIDINet analog, per-token 1×1 conv)
# ---------------------------------------------------------------------------

class SpectralEncoder(nn.Module):
    """2-layer CNN over log|STFT| → per-token feature map."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, d // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(d // 2, d, 3, padding=1),
        )
        self.d = d

    def forward(self, logS):  # [B, F, T'] -> [B, d, F, T']
        return self.net(logS.unsqueeze(1))


class SpectralDecoder(nn.Module):
    def __init__(self, d_expert: int = 64, num_tokens: int = 4):
        super().__init__()
        self.enc = SpectralEncoder(d_expert)
        self.num_tokens = num_tokens

    def forward(self, tokens: torch.Tensor,
                log_stft: torch.Tensor) -> torch.Tensor:
        f = self.enc(log_stft)                 # [B, d, F, T']
        B, N, d = tokens.shape
        # each token is a 1×1 kernel applied to f
        w = tokens.view(B, N, d, 1, 1)
        f_exp = f.unsqueeze(1)                 # [B, 1, d, F, T']
        rec = (f_exp * w).sum(2)               # [B, N, F, T']
        return torch.sigmoid(rec.mean(1))      # [B, F, T']


# ---------------------------------------------------------------------------
# 4. Semantic head (DINOv2 analog, feature-level MSE)
# ---------------------------------------------------------------------------

class SemanticHead(nn.Module):
    def __init__(self, d_expert: int, d_target: int):
        super().__init__()
        self.proj = nn.Linear(d_expert, d_target)

    def forward(self, tokens: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """tokens [B,N,d_e], target [B,P,d_t]. Returns MSE."""
        pred = self.proj(tokens)              # [B, N, d_t]
        # pool target to match token count
        P = target.shape[1]
        if P != pred.shape[1]:
            target = F.adaptive_avg_pool1d(
                target.transpose(1, 2), pred.shape[1]
            ).transpose(1, 2)
        return F.mse_loss(pred, target)
