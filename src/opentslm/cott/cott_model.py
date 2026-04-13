# SPDX-License-Identifier: MIT
"""CoTTFlamingo wraps OpenTSLMFlamingo with four CoTT expert heads.

We (a) register four groups of special tokens in the tokenizer, (b) extract
their hidden states after the LLM forward pass, (c) push them through a
per-group CoTT projection + decoder, and (d) add the reconstruction losses
to the language-model CE loss. Inference can skip (c)–(d) entirely: the
rationale still contains the special tokens, but the decoders are never
called.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.cott.decoders import (
    CoTTProjection, CPDecoder, ForecastDecoder, SpectralDecoder, SemanticHead,
    cp_matching_loss, dice_loss, focal_loss,
)


# The exact strings must match what CoTTStageWrapper emits in prompts.
CP_TOKEN   = "<cp_pad>"
FC_TOKEN   = "<fc_pad>"
SP_TOKEN   = "<sp_pad>"
SEM_TOKEN  = "<sem_pad>"


DEFAULT_BUDGET = {"cp": 8, "fc": 4, "sp": 4, "sem": 4}


class CoTTFlamingo(OpenTSLMFlamingo):
    """CoTT-augmented Flamingo.

    **Do not call this constructor directly** — it would re-build the
    vanilla OpenTSLMFlamingo (Llama + CNN tokenizer) before any CoTT
    customisation is applied. Use :func:`opentslm.cott.build.build_cott_flamingo`
    instead, which lets you pick the backbone LLM (Llama / Gemma / Qwen2 /
    Qwen3) and the time-series encoder (CNN / Chronos-2), and wires the
    four expert heads correctly.
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "Use opentslm.cott.build.build_cott_flamingo(...) "
            "to construct a CoTTFlamingo; the default __init__ is disabled."
        )

    # ------------------------------------------------------------------
    def _gather_thought_hiddens(self, hidden_states: torch.Tensor,
                                input_ids: torch.Tensor,
                                kind: str) -> Optional[torch.Tensor]:
        """Pull the hidden states corresponding to a given thought-token kind.

        Returns [B, N_k, d_llm] or None if the batch contains no such tokens
        (e.g. efficient-reasoning dropout removed the group).
        """
        mask = input_ids == self.token_ids[kind]
        # Expect exactly budget[kind] tokens per row when the group is present.
        counts = mask.sum(-1)
        if (counts == 0).all():
            return None
        out = []
        for b in range(hidden_states.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                out.append(None)
            else:
                out.append(hidden_states[b, idx])
        # Drop Nones, stack. Callers that rely on per-sample alignment should
        # instead use _batched_gather_thought_hiddens (see training loop).
        good = [o for o in out if o is not None]
        if not good:
            return None
        return torch.stack(good, 0)

    # ------------------------------------------------------------------
    def compute_loss(self, batch: List[Dict[str, any]],
                     cott_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
                     loss_weights: Optional[Dict[str, float]] = None,
                     gamma: float = 1.0) -> torch.Tensor:
        """LM CE loss + Σ λ_k · L_k (reconstruction).

        `cott_targets[i]` holds per-sample expert targets (from
        `experts.compute_expert_targets`). If None or empty, we degrade to
        plain OpenTSLM training (the Text-CoT baseline).
        """
        loss_weights = loss_weights or {"cp": 1.0, "fc": 1.0,
                                         "sp": 1.0, "sem": 1.0}

        # 1) run the underlying Flamingo forward exactly as in OpenTSLMFlamingo
        input_ids, images, attention_mask, labels = self.pad_and_apply_batch(
            batch, include_labels=False,
        )
        out = self.model(
            vision_x=images, lang_x=input_ids,
            attention_mask=attention_mask, labels=labels,
            output_hidden_states=True,
        )
        ce_loss = out[0]
        if cott_targets is None:
            return ce_loss

        hs = out.hidden_states[-1]     # last layer, [B, T, d]

        # 2) per-expert reconstruction
        total_rec = hs.new_zeros(())
        for kind in ["cp", "fc", "sp", "sem"]:
            if loss_weights.get(kind, 0.0) == 0.0:
                continue
            z = self._gather_thought_hiddens(hs, input_ids, kind)
            if z is None:
                continue
            zp = self.proj[kind](z)          # [B', N_k, d_expert]

            # Build the batched targets for the samples that contained tokens
            targets = self._stack_targets(cott_targets, input_ids, kind)
            if targets is None:
                continue

            if kind == "cp":
                masks = self.cp_decoder(zp, targets["raw_series"])  # [B,N,L]
                rec = cp_matching_loss(masks, targets["cp_mask"])
            elif kind == "fc":
                fc = self.fc_decoder(zp, targets["chronos_feats"])  # [B,H]
                rec = F.l1_loss(fc, targets["forecast"])
            elif kind == "sp":
                pred_stft = self.sp_decoder(zp, targets["log_stft"])
                rec = F.l1_loss(pred_stft, targets["log_stft"]) \
                      + F.l1_loss(pred_stft.diff(dim=-1),
                                  targets["log_stft"].diff(dim=-1))
            else:  # sem
                rec = self.sem_head(zp, targets["semantic"])

            total_rec = total_rec + loss_weights[kind] * rec

        return ce_loss + gamma * total_rec

    # ------------------------------------------------------------------
    def _stack_targets(self, cott_targets, input_ids, kind):
        """Collect only the samples whose prompts contained the `kind` group,
        then stack their targets into a single batched dict."""
        keep = [
            i for i in range(input_ids.shape[0])
            if (input_ids[i] == self.token_ids[kind]).any().item()
        ]
        if not keep:
            return None

        def _get(i, key):
            return cott_targets[i][key].to(input_ids.device)

        d = {}
        if kind == "cp":
            d["raw_series"] = torch.stack([_get(i, "raw_series") for i in keep])
            d["cp_mask"]    = torch.stack([_get(i, "cp_mask")    for i in keep])
        elif kind == "fc":
            d["chronos_feats"] = torch.stack(
                [_get(i, "chronos_feats") for i in keep])
            d["forecast"] = torch.stack([_get(i, "forecast") for i in keep])
        elif kind == "sp":
            d["log_stft"] = torch.stack([_get(i, "log_stft") for i in keep])
        elif kind == "sem":
            d["semantic"] = torch.stack([_get(i, "semantic") for i in keep])
        return d
