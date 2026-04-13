# IndusTSLM

<img align="right" src="assets/drilling.png" alt="Drilling sensor signatures" width="280">

**Time-Series Language Models for Drilling Data** · Master's thesis, MBZUAI (2026)

Can the multimodal recipes that bridged vision and language also bridge *industrial sensor streams and natural language*? This repo explores that question on real ADNOC drilling data.

Built on [OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM) · In coordination with the [AIQ Intelligence](https://aiqintelligence.ai) team.

<br clear="right">

---

## What's inside

| | Contribution | AIQ collaborators |
|---|---|---|
| **1** | **DriMM** — CLIP-style dual encoder aligning sensor windows with DDR text, extended with hard-negative mining (+8% average). | Sebastiaan Buiting\*, Soumyadipta Sengupta\*, Abdallah Benzine, Amine El Khair |
| **2** | **IndusTSLM** — soft-prompting (LiveDrill) and Flamingo-style cross-attention for DDR generation. | Abdallah Benzine, Amine El Khair |
| **3** | **DrillBench** *(ongoing)* — 7 tasks × 4 groups on Volve + Utah FORGE, with knowledge decoupling. | Abdallah Benzine, Amine El Khair |

\* equal contribution

<p align="center">
   <img src="assets/industslm.png" alt="IndusTSLM Flamingo architecture" width="78%">
</p>

---

## Key takeaways

- Domain-specific CNN encoders beat large general-purpose TSFMs on drilling data.
- Models identify activity *type* well, but numeric fidelity (depths, pressures) stays low — an open problem.
- Commercial LLMs lean heavily on memorized drilling knowledge. Instruction-tuned IndusTSLM closes the gap by actually reading the sensors.

---

## Repo layout

```
IndusTSLM/
├── assets/           Figures used in the README and paper
├── docs/             Thesis PDF + design docs
├── scripts/
│   ├── training/     Training entrypoints (DriMM, curriculum, CoTT)
│   └── evaluation/   Evaluation entrypoints
├── src/industslm/     Python package (models, datasets, utilities)
├── notebooks/        Exploratory analysis
├── demo/             HuggingFace inference demos
└── test/
```

---

## Quick start

```bash
git clone https://github.com/yehias21/IndusTSLM.git
cd IndusTSLM
uv sync --all-groups && source .venv/bin/activate
# or: pip install -r requirements.txt
```

Training (two-stage curriculum):

```bash
# Stage 1: activity classification (warmup)
python scripts/training/curriculum_learning.py --model OpenTSLMFlamingo --stages stage1_classification

# Stage 2: DDR generation
python scripts/training/curriculum_learning.py --model OpenTSLMFlamingo --stages stage2_generation
```

Backbones: Phi-3 mini, Qwen3 4B · Encoders: Chronos-2 or CNN.

---

## Publications

1. **LiveDrill** — NeurIPS 2025 Workshop (Bert2S)
2. **LLMs as Judges for Domain-Specific Text** — NeurIPS 2025 Workshop (LLM Lifecycle)
3. **Streaming Drilling Report Generation** — IEEE Big Data 2025

---

## Citation

```bibtex
@mastersthesis{shaaban2026industslm,
  title  = {IndusTSLM: Exploring Time-Series Language Models for Drilling Data},
  author = {Shaaban, Yahia Salaheldin},
  school = {MBZUAI},
  year   = {2026}
}
```

Full thesis: [`docs/industslm_thesis.pdf`](docs/industslm_thesis.pdf) · License: [MIT](LICENSES/MIT.txt)
