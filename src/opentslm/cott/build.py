# SPDX-License-Identifier: MIT
"""`build_cott_flamingo` — one-shot constructor for a CoTT-Flamingo model
from scratch, analogous to `CurriculumTrainer._initialize_model` in
`curriculum_learning.py` but with:

  * a pluggable time-series encoder (CNN tokenizer *or* Chronos-2),
  * cross-attention layers correctly injected for Llama / Gemma / Qwen2 /
    Qwen3 backbones,
  * the four CoTT special tokens registered and the four expert heads
    attached.

Usage
-----
>>> from opentslm.cott.build import build_cott_flamingo
>>> model = build_cott_flamingo(
...     llm_id="Qwen/Qwen3-1.7B",        # or "meta-llama/Llama-3.2-1B"
...     encoder="chronos",               # or "cnn"
...     chronos_id="amazon/chronos-2",
...     device="cuda",
... )
>>> loss = model.compute_loss(batch, cott_targets=targets)

Then drop it into `curriculum_learning.CurriculumTrainer` by setting
`self.model_type = "CoTTFlamingo"` and calling this function from
`_initialize_model()` (patch shown at the bottom of this file).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from open_flamingo.src.flamingo_lm import FlamingoLMMixin, FlamingoLayer
from open_flamingo.src.utils import extend_instance

from opentslm.model.llm.TimeSeriesFlamingoWithTrainableEncoder import (
    TimeSeriesFlamingoWithTrainableEncoder,
)
from opentslm.cott.cott_model import (
    CoTTFlamingo, CP_TOKEN, FC_TOKEN, SP_TOKEN, SEM_TOKEN, DEFAULT_BUDGET,
)
from opentslm.cott.encoders import CNNPatchEncoder, ChronosPatchEncoder
from opentslm.cott.decoders import (
    CoTTProjection, CPDecoder, ForecastDecoder, SpectralDecoder, SemanticHead,
)


# ---------------------------------------------------------------------------
# Decoder-layer-name detection extended with Qwen2 / Qwen3 support.
# Flamingo relies on this to know where to inject gated cross-attention
# blocks. Both Qwen2 (Qwen2ForCausalLM) and Qwen3 (Qwen3ForCausalLM) keep the
# decoder stack at `model.layers`, identical to Llama.
# ---------------------------------------------------------------------------
_KNOWN_DECODER_LAYERS_ATTR = {
    "opt":      "model.decoder.layers",
    "gptj":     "transformer.h",
    "pythia":   "gpt_neox.layers",
    "llama":    "model.layers",
    "mpt":      "transformer.blocks",
    "gemma":    "model.layers",
    "gemma2":   "model.layers",
    "gemma3":   "model.layers",
    "medgemma": "model.layers",
    "qwen":     "model.layers",
    "qwen2":    "model.layers",
    "qwen3":    "model.layers",
}


def _infer_decoder_layers_attr_name(model: nn.Module) -> str:
    name = model.__class__.__name__.lower()
    if "gemma3" in name and "conditionalgeneration" in name:
        return "language_model.layers"
    for k, v in _KNOWN_DECODER_LAYERS_ATTR.items():
        if k in name:
            return v
    raise ValueError(
        f"Unknown LLM class {model.__class__.__name__}; "
        f"add it to _KNOWN_DECODER_LAYERS_ATTR."
    )


# Re-apply the `attention_type` property patch from OpenTSLMFlamingo so that
# newer Transformers versions don't break Qwen's attention-type plumbing.
if not hasattr(FlamingoLayer, "attention_type"):
    FlamingoLayer.attention_type = property(
        lambda self: getattr(self.decoder_layer, "attention_type", None)
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_cott_flamingo(
    llm_id: str = "Qwen/Qwen3-1.7B",
    encoder: str = "chronos",
    chronos_id: str = "amazon/chronos-2",
    budget: Dict[str, int] | None = None,
    horizon: int = 32,
    cross_attn_every_n_layers: int = 1,
    freeze_encoder: bool = True,
    freeze_lm_embeddings: bool = False,
    gradient_checkpointing: bool = False,
    device: str = "cuda",
) -> CoTTFlamingo:
    """Assemble a CoTT-Flamingo from scratch.

    Mirrors `CurriculumTrainer._initialize_model()` for the `OpenTSLMFlamingo`
    branch, but:
      (1) chooses a time-series encoder by name,
      (2) uses an extended decoder-layer-name table so Qwen2/Qwen3 work,
      (3) wires the CoTT special tokens + expert heads onto the resulting
          Flamingo object.
    """
    budget = budget or DEFAULT_BUDGET

    # 1. build tokenizer + LLM -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        llm_id, trust_remote_code=True,
    )
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        llm_id, trust_remote_code=True,
        device_map={"": device},
        attn_implementation="eager",           # required by open_flamingo
    )

    # Flamingo special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"

    # 2. patch LLM → FlamingoLM -----------------------------------------------
    extend_instance(lang_encoder, FlamingoLMMixin)
    lang_encoder.set_decoder_layers_attr_name(
        _infer_decoder_layers_attr_name(lang_encoder)
    )
    lang_encoder.resize_token_embeddings(len(tokenizer))

    if (hasattr(lang_encoder.config, "text_config")
            and hasattr(lang_encoder.config.text_config, "hidden_size")
            and not hasattr(lang_encoder.config, "hidden_size")):
        lang_encoder.config.hidden_size = (
            lang_encoder.config.text_config.hidden_size
        )

    # 3. build time-series encoder -------------------------------------------
    if encoder == "cnn":
        ts_encoder = CNNPatchEncoder().to(device)
        vis_dim = ts_encoder.output_dim
    elif encoder == "chronos":
        ts_encoder = ChronosPatchEncoder(
            model_id=chronos_id, freeze=freeze_encoder
        ).to(device)
        vis_dim = ts_encoder.output_dim
    else:
        raise ValueError(f"Unknown encoder {encoder!r}")

    # 4. assemble Flamingo ----------------------------------------------------
    flamingo = TimeSeriesFlamingoWithTrainableEncoder(
        ts_encoder,
        lang_encoder,
        tokenizer.encode("<|endofchunk|>")[-1],
        tokenizer.encode("<image>")[-1],
        vis_dim=vis_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        gradient_checkpointing=gradient_checkpointing,
    )
    flamingo.to(dtype=lang_encoder.dtype)

    # freeze everything, then unfreeze perceiver + gated XA + (optionally)
    # LM input embeddings + the trainable part of the TS encoder.
    flamingo.requires_grad_(False)
    flamingo.perceiver.requires_grad_(True)
    flamingo.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        flamingo.lang_encoder.get_input_embeddings().requires_grad_(True)
    if not freeze_encoder:
        flamingo.vision_encoder.requires_grad_(True)

    # 5. lift into CoTTFlamingo WITHOUT re-running OpenTSLMFlamingo.__init__
    #    (which would rebuild a CNNTokenizer + LLM from scratch).
    cott = CoTTFlamingo.__new__(CoTTFlamingo)
    # Populate the TimeSeriesLLM base attributes manually.
    cott.device = device
    cott.model = flamingo
    cott.llm = flamingo
    cott.text_tokenizer = tokenizer
    cott.budget = budget
    cott.horizon = horizon

    # Register CoTT special tokens now that the tokenizer/LLM exist.
    tokenizer.add_special_tokens({"additional_special_tokens":
        [CP_TOKEN, FC_TOKEN, SP_TOKEN, SEM_TOKEN]})
    flamingo.lang_encoder.resize_token_embeddings(len(tokenizer))
    cott.token_ids = {
        "cp":  tokenizer.convert_tokens_to_ids(CP_TOKEN),
        "fc":  tokenizer.convert_tokens_to_ids(FC_TOKEN),
        "sp":  tokenizer.convert_tokens_to_ids(SP_TOKEN),
        "sem": tokenizer.convert_tokens_to_ids(SEM_TOKEN),
    }

    # Projections + decoders sized against this LLM's hidden dim.
    d_llm = flamingo.lang_encoder.config.hidden_size
    cott.proj = nn.ModuleDict({
        "cp":  CoTTProjection(d_llm, 128,      num_tokens=budget["cp"]),
        "fc":  CoTTProjection(d_llm, vis_dim,  num_tokens=budget["fc"]),
        "sp":  CoTTProjection(d_llm, 64,       num_tokens=budget["sp"]),
        "sem": CoTTProjection(d_llm, 128,      num_tokens=budget["sem"]),
    }).to(device)
    cott.cp_decoder = CPDecoder(128, budget["cp"]).to(device)
    cott.fc_decoder = ForecastDecoder(vis_dim, horizon, budget["fc"]).to(device)
    cott.sp_decoder = SpectralDecoder(64, budget["sp"]).to(device)
    cott.sem_head   = SemanticHead(128, vis_dim).to(device)

    return cott


# ---------------------------------------------------------------------------
# Drop-in patch for curriculum_learning.CurriculumTrainer
# ---------------------------------------------------------------------------
def register_cott_in_curriculum_trainer():
    """Monkey-patch `CurriculumTrainer._initialize_model` so that
    `model_type="CoTTFlamingo"` works without editing the upstream file.

    Call this once at the top of your training script:

        from opentslm.cott.build import register_cott_in_curriculum_trainer
        register_cott_in_curriculum_trainer()
        trainer = CurriculumTrainer(model_type="CoTTFlamingo",
                                    llm_id="Qwen/Qwen3-1.7B", ...)
        trainer.run_curriculum(...)
    """
    from curriculum_learning import CurriculumTrainer  # local import: it's
                                                        # a script, not a pkg

    _orig = CurriculumTrainer._initialize_model

    def _initialize_model(self):
        if getattr(self, "model_type", None) == "CoTTFlamingo":
            return build_cott_flamingo(
                llm_id=self.llm_id,
                encoder=getattr(self, "ts_encoder", "chronos"),
                chronos_id=getattr(self, "chronos_id", "amazon/chronos-2"),
                budget=getattr(self, "cott_budget", DEFAULT_BUDGET),
                horizon=getattr(self, "cott_horizon", 32),
                gradient_checkpointing=self.gradient_checkpointing,
                device=self.device,
            ).to(self.device)
        return _orig(self)

    CurriculumTrainer._initialize_model = _initialize_model
