# Chain-of-Temporal-Thought (CoTT): Teaching TSLMs to *See* Time-Series Structure Before Answering

**Workshop submission — methodology & experimental design.**

## 1. Motivation and mapping

CoVT (Qin et al., 2025) identifies two failure modes of text-only CoT for Vision-Language Models:
(i) linguistic bottleneck — continuous spatial/geometric structure is lost when verbalized; (ii) supervision is dominated by final text answers, giving the model little incentive to internalize dense perceptual cues. They fix this by inserting *continuous* tokens inside `<think>...</think>`, each group aligned with a lightweight perceptual expert (SAM, DepthAnything v2, PIDINet, DINOv2).

OpenTSLM (Langer et al., 2026) shows the same two failure modes for time-series: text-only CoT "can misdirect and even degrade" performance (their Fig. 6), and rationales are judged by cardiologists to contain plausible-but-wrong pattern recognition in 34.5% of ECG cases. Time-series reasoning is inherently continuous, high-dimensional, and multi-scale — exactly the regime where CoVT claims gains.

**CoTT applies CoVT verbatim, swapping the four vision experts for four temporal experts that cover the perception axes a clinician/engineer uses on a trace.** The rest — soft prompt placement, curriculum, decoder-level alignment, reconstruction-based distillation — transfers one-to-one.

| CoVT expert | Perception axis | CoTT expert | Supervision signal | Alignment |
|---|---|---|---|---|
| SAM (segmentation) | instance / region | **Change-point segmenter** (PELT over the same window) | per-step binary CP mask, ≤K segments | prompt-level (decoder) |
| DepthAnything v2 | 3D dense scalar | **Forecaster** (Chronos-2, frozen) | next-H values of the series | prompt-level (decoder via BMM) |
| PIDINet (edges) | transient boundaries | **Spectral / derivative** | log-magnitude STFT + Δx derivative | 1×1-conv over STFT features |
| DINOv2 | patch semantics | **TSFM patch embeddings** (Chronos-2, frozen) | patch features | feature-level (MSE) |

Token budget matches CoVT's ≈20: **8 change-point + 4 forecast + 4 spectral + 4 semantic = 20**.

## 2. Architecture

Base: `OpenTSLMFlamingo` with `Llama-3.2-1B`. We keep this base because the paper shows it has nearly constant VRAM vs. sequence length (crucial once we add decoder heads) and leads on ECG-QA-CoT. We add a **CoTT head module** consisting of:

- 4 pools of *learnable thought-token ids* (`<cp_pad>×8`, `<fc_pad>×4`, `<sp_pad>×4`, `<sem_pad>×4`) registered as special tokens in the tokenizer. Their LLM embeddings are trainable (like CoVT).
- 4 projection layers (linear + MHA + linear, following CoVT §A.1) mapping the LLM hidden states at those positions into the corresponding expert's prompt/feature space.
- 4 lightweight decoders (see §3) that consume the projected tokens plus a frozen expert-encoder feature map `f` from the *same* input window.

During training we teacher-force the whole chain; losses are computed on the LLM hidden states at the CoTT positions (teacher-forced → positionally aligned). During inference the decoders are **optional**; reasoning happens in latent space.

### Where the tokens live in the prompt

Inherit OpenTSLM's `<TS>…<|endofchunk|>` scheme. For Stage 3/4 the answer template is:

```
<think>The change-points of the signal are <cp_pad>×8,
the forecast is <fc_pad>×4, the spectral edges are <sp_pad>×4,
and the patch features are <sem_pad>×4.</think>
<answer>walking</answer>
```

## 3. Alignment per expert (mirrors CoVT §3.3)

Let `Tᵏ ∈ R^{Nₖ×d_llm}` be the LLM hidden states at the k-th token group's positions, `proj_k` a CoTT projection layer, and `f_k` the encoder feature map of the expert on the *raw* input window. Supervision targets are computed online from the raw series; no curated labels required.

1. **Change-point tokens (k=cp, N=8).** Supervision = binary CP mask `M ∈ {0,1}^L` from `ruptures.Pelt(model="rbf").predict(pen=β)` run once per sample at load time. Decoder: a 3-layer 1D-UNet-like block conditioned on each of the 8 projected tokens via cross-attention over `f_cp` (a small 1D-CNN encoder of the raw series, trained end-to-end since no pretrained CP model is used). Each token prompts one candidate segment; Hungarian matching to ground-truth segments; Dice + Focal loss exactly as in LISA/CoVT §A.2.

2. **Forecast tokens (k=fc, N=4).** Expert: Chronos-2 encoder (frozen, already supported by the repo via `Chronos2TimeSeriesEncoder`). Using the last context window, we take 4 mid-layer feature maps `F^fc_i ∈ R^{P×d_chronos}`. Each projected token interacts with its feature map via batch-matrix-multiplication to reconstruct the next-H future values; ensemble by mean (CoVT Eq. 3). Loss: L1 against Chronos-2's own *teacher* forecast (self-distillation), so no external label is needed.

3. **Spectral/derivative tokens (k=sp, N=4).** Expert features = STFT magnitude of the raw window, cached at load time; `|STFT|` is fed through a 2-layer CNN giving `F^sp_i`. Each projected token acts as a 1×1 conv kernel over `F^sp_i` producing a reconstructed log-magnitude spectrogram; final prediction = sigmoid of mean (CoVT Eq. 20). Loss: L1 on log-magnitude + L1 on first temporal difference `Δx` (covers "edges").

4. **Semantic tokens (k=sem, N=4).** Target = mean-pooled Chronos-2 patch features (frozen). Projected tokens are mapped to the same shape; MSE. This is the DINOv2 analog (CoVT §3.3-4).

Joint loss (CoVT Eq. 4):

```
L_total = L_ce(LM) + γ[λ_cp L_cp + λ_fc L_fc + λ_sp L_sp + λ_sem L_sem]
```

We set `γ = 1`, all `λ = 1` for main runs and ablate `γ∈{0, 0.1, 1, 10}`.

## 4. Four-stage curriculum (CoVT §3.4 verbatim)

Exactly as CoVT Fig. 4, but operating on OpenTSLM's CoT datasets.

| Stage | Input template | Objective |
|---|---|---|
| 1. Comprehension | `<TS>` followed by `<cp_pad>…<sem_pad>` then the question | learn semantics of thought tokens; LM loss on answer only |
| 2. Generation | question asks "what are the change-points / forecast / spectrum / features?" | LM loss on the thought-token stub + reconstruction losses |
| 3. Reasoning | answer = `<think>…CoTT chain…</think><answer>label</answer>` | full CE + reconstruction |
| 4. Efficient reasoning | randomly drop 0..3 token groups per sample | same as 3; forces the model to use whichever tokens remain |

Stage lengths mirror the CoVT schedule for "3 tokens" (4k/3k/3k/5k steps). For ECG-QA-CoT the step counts are doubled because the dataset is ~3× larger.

## 5. Datasets

We use the three CoT datasets released with OpenTSLM, *unchanged* — including their 80/10/10 splits:

- **HAR-CoT** — 3-axis accelerometer, 2.56 s @ 50 Hz, 8 classes.
- **Sleep-CoT** — single-channel EEG, 30 s @ 100 Hz, 5 classes.
- **ECG-QA-CoT** — 12-lead ECG, 10 s @ 100 Hz, 42 QA templates.

Expert targets are computed online (STFT, Δx, Chronos-2 features) or pre-cached once (change-point masks via `ruptures`, ~2 h on CPU for HAR).

## 6. Experimental design

### 6.1 Main comparisons (all on the same Llama-3.2-1B + Flamingo base)

| # | Name | `<think>` tokens | Aligned? | Purpose |
|---|---|---|---|---|
| A | Baseline-Flamingo | — (no CoT training, direct label) | — | reproduces OpenTSLM Table 2 |
| B | Text-CoT | text-only rationale | — | reproduces OpenTSLM §4.3 |
| C | Latent-Only | 20 ordinary soft tokens | ✗ | CoVT's "16 empty" ablation |
| D | CoTT-1 (×4) | 8 of one type only | ✓ | isolates each expert |
| E | CoTT-3 | CP + FC + SEM (16 tokens) | ✓ | CoVT's "3 Visual Tokens" analog (main result) |
| F | CoTT-4 | CP + FC + SP + SEM (20 tokens) | ✓ | full model |

**Metrics.** Macro-F1 and accuracy on each test split (identical to OpenTSLM Table 2). Additional: BLEU/ROUGE-L on rationale text, and for ECG-QA the 5-cardiologist rubric from OpenTSLM §4.5 (pattern recognition / clinical reasoning / context integration).

### 6.2 Ablations (run on HAR-CoT only, cheapest)

1. **Token count per expert** — CP ∈ {1, 4, 8, 16, 32}, fixing FC=SEM=4 (CoVT Tab. 4).
2. **Alignment strategy** — decoder-prompt vs. feature-MSE for CP and FC (CoVT Tab. 5).
3. **Curriculum** — Stage 3+4 only vs. all 4 stages (CoVT Tab. 7).
4. **Loss weight γ** ∈ {0, 0.1, 1, 10}.
5. **VRAM scaling** — # CoTT tokens vs. peak memory, re-using OpenTSLM's simulation harness (`scripts/get_memory_use.py`).

### 6.3 Interpretability / qualitative study

For 20 random test samples per dataset we decode each CoTT group and render:
- CP mask overlaid on the raw series,
- forecast vs. held-out continuation,
- STFT reconstruction vs. ground truth,
- nearest-neighbour retrieval in Chronos-2 feature space using the semantic tokens.

### 6.4 Expert review (ECG-QA only)

Replicate OpenTSLM's 5-cardiologist evaluation on 84 ECG-QA samples (2 per template) for the best CoTT model vs. the text-CoT baseline. Same RIME-derived rubric. Report McNemar on paired judgments.

### 6.5 Compute

Main runs on 1×A100 80 GB (paper setting). Estimated total: 3 datasets × 6 models × (4 stages) ≈ 72 training jobs, ~6 h each. Ablations on HAR fit in 36 h.

## 7. Expected outcomes and what would falsify the claim

We expect, following CoVT's pattern:

- **Positive claim.** CoTT-3/4 ≥ Text-CoT by +3–10 F1 on HAR-CoT and Sleep-CoT, and by +2–5 on ECG-QA-CoT, with the largest gain on the dataset most dominated by dense temporal structure (Sleep).
- **Specificity claim.** Each single-expert variant (D) peaks on its "native" sub-task: CP-only best on counting-like sub-tasks, forecast-only best on trend / long-horizon reasoning, spectral-only best on arrhythmia-type ECG questions, semantic-only a flat but non-trivial lift.
- **Latent-alignment matters.** Latent-Only (C) ≤ Text-CoT, reproducing CoVT's finding that raw extra tokens without expert grounding don't help.

A failure of any of these — in particular C ≈ F, or a single-expert uniformly dominating — would indicate that CoTT's gain is driven by extra capacity rather than grounded perception, and the central CoVT→time-series transfer is not supported.

## 8. Deliverables

1. `src/industslm/cott/` — model, heads, decoders, expert targets (§2–3).
2. `train_cott.py` — 4-stage curriculum driver (§4).
3. `evaluate_cott.py` — reuses `evaluation/` harness for metrics + decoded-token plots.
4. Pretrained checkpoints for variants B, C, E, F on all three datasets.
5. Cardiologist review artifacts (same form as OpenTSLM Fig. 15).
