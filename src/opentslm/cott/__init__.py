# SPDX-License-Identifier: MIT
"""Chain-of-Temporal-Thought (CoTT): a time-series analogue of
Chain-of-Visual-Thought (Qin et al., 2025) built on top of OpenTSLM-Flamingo.

The public entry points are:
    - cott.experts.compute_expert_targets
    - cott.decoders.{CPDecoder, ForecastDecoder, SpectralDecoder, SemanticHead}
    - cott.cott_model.CoTTFlamingo
    - cott.cott_dataset.CoTTStageWrapper
"""
from opentslm.cott.cott_model import CoTTFlamingo  # noqa: F401
from opentslm.cott.cott_dataset import CoTTStageWrapper  # noqa: F401
