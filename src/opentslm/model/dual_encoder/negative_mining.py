# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Negative mining strategies for contrastive learning (Strategy Pattern).

Each strategy takes a batch of samples and returns an augmented batch
with additional hard negative pairs. Strategies are composable and
can be swapped or combined via the NegativeMiner interface.

To add your own strategy:
  1. Subclass NegativeMiningStrategy
  2. Implement mine(batch) -> batch
  3. Register it in STRATEGY_REGISTRY
"""

import random
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional

import torch


class NegativeMiningStrategy(ABC):
    """Base class for all negative mining strategies."""

    @abstractmethod
    def mine(self, batch: List[dict]) -> List[dict]:
        """
        Take a batch of samples and return an augmented batch with hard negatives.

        Each sample is a dict with keys: 'time_series', 'text', 'code'.
        The returned batch may be larger (extra negative samples appended)
        or the same size (existing samples modified/reordered).

        Args:
            batch: List of sample dicts from the dataset.
        Returns:
            Augmented batch with hard negatives included.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class NoNegatives(NegativeMiningStrategy):
    """No-op strategy — returns the batch unchanged. Default behavior."""

    def mine(self, batch: List[dict]) -> List[dict]:
        return batch


class TextSwapNegatives(NegativeMiningStrategy):
    """
    Swap text descriptions between samples of DIFFERENT classes.

    Creates hard negatives where the time series stays the same
    but the text is replaced with a plausible but incorrect description.
    Forces the model to learn fine-grained TS-text alignment.
    """

    def __init__(self, swap_ratio: float = 0.5):
        """
        Args:
            swap_ratio: Fraction of batch to create swapped negatives for.
        """
        self.swap_ratio = swap_ratio

    def mine(self, batch: List[dict]) -> List[dict]:
        n_negatives = max(1, int(len(batch) * self.swap_ratio))

        # Group indices by code
        code_to_indices: Dict[str, List[int]] = {}
        for i, sample in enumerate(batch):
            code = sample.get("code", "UNK")
            code_to_indices.setdefault(code, []).append(i)

        codes = list(code_to_indices.keys())
        if len(codes) < 2:
            return batch  # can't swap if only one class

        negatives = []
        for _ in range(n_negatives):
            # Pick a random sample
            idx = random.randint(0, len(batch) - 1)
            src_code = batch[idx].get("code", "UNK")

            # Pick a text from a DIFFERENT class
            other_codes = [c for c in codes if c != src_code]
            if not other_codes:
                continue
            other_code = random.choice(other_codes)
            other_idx = random.choice(code_to_indices[other_code])

            # Create negative: same time series, different text
            neg_sample = {
                "time_series": batch[idx]["time_series"],
                "text": batch[other_idx]["text"],
                "code": f"NEG_{src_code}",  # mark as negative
            }
            negatives.append(neg_sample)

        return batch + negatives


class NumberPerturbationNegatives(NegativeMiningStrategy):
    """
    Perturb numerical values in text to create hard negatives.

    E.g.: "drilled to 8760 ft" → "drilled to 2760 ft"

    Forces the model to be sensitive to numerical values, not just keywords.
    This directly addresses the numerical grounding problem.
    """

    def __init__(self, perturb_ratio: float = 0.3, max_negatives_per_sample: int = 1):
        self.perturb_ratio = perturb_ratio
        self.max_negatives_per_sample = max_negatives_per_sample
        self._number_pattern = re.compile(r'\b(\d+\.?\d*)\b')

    def _perturb_numbers(self, text: str) -> Optional[str]:
        """Replace one random number in the text with a different value."""
        matches = list(self._number_pattern.finditer(text))
        if not matches:
            return None

        match = random.choice(matches)
        original = match.group()

        try:
            val = float(original)
        except ValueError:
            return None

        # Perturb: multiply by a random factor (0.1x to 10x, but not ~1x)
        factor = random.choice([0.1, 0.2, 0.5, 2.0, 5.0, 10.0])
        new_val = val * factor

        # Format like original (int or float)
        if '.' in original:
            new_str = f"{new_val:.{len(original.split('.')[-1])}f}"
        else:
            new_str = str(int(new_val))

        return text[:match.start()] + new_str + text[match.end():]

    def mine(self, batch: List[dict]) -> List[dict]:
        n_candidates = max(1, int(len(batch) * self.perturb_ratio))
        indices = random.sample(range(len(batch)), min(n_candidates, len(batch)))

        negatives = []
        for idx in indices:
            for _ in range(self.max_negatives_per_sample):
                perturbed = self._perturb_numbers(batch[idx]["text"])
                if perturbed and perturbed != batch[idx]["text"]:
                    neg_sample = {
                        "time_series": batch[idx]["time_series"],
                        "text": perturbed,
                        "code": f"NEG_{batch[idx].get('code', 'UNK')}",
                    }
                    negatives.append(neg_sample)

        return batch + negatives


class InBatchHardNegatives(NegativeMiningStrategy):
    """
    Reorder the batch to place hard negatives (same class, different sample)
    adjacent in the similarity matrix.

    Unlike TextSwap and NumberPerturbation which create NEW samples,
    this strategy reorders EXISTING samples so that the hardest negatives
    (same class but different instance) appear in the same mini-batch.
    """

    def mine(self, batch: List[dict]) -> List[dict]:
        # Group by code
        code_to_samples: Dict[str, List[dict]] = {}
        for sample in batch:
            code = sample.get("code", "UNK")
            code_to_samples.setdefault(code, []).append(sample)

        # Interleave: put samples from same class close together
        # This ensures the similarity matrix has hard negatives nearby
        reordered = []
        code_lists = list(code_to_samples.values())
        random.shuffle(code_lists)

        # Round-robin from each class, but keep 2-3 from same class together
        for samples in code_lists:
            random.shuffle(samples)
            reordered.extend(samples)

        return reordered


class CompositeStrategy(NegativeMiningStrategy):
    """
    Compose multiple strategies sequentially.
    Each strategy's output is fed as input to the next.
    """

    def __init__(self, strategies: List[NegativeMiningStrategy]):
        self.strategies = strategies

    def mine(self, batch: List[dict]) -> List[dict]:
        for strategy in self.strategies:
            batch = strategy.mine(batch)
        return batch

    @property
    def name(self) -> str:
        names = [s.name for s in self.strategies]
        return f"Composite({'+'.join(names)})"


# ---------------------------------------------------------------------------
# Registry & Factory
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY: Dict[str, type] = {
    "none": NoNegatives,
    "text_swap": TextSwapNegatives,
    "number_perturb": NumberPerturbationNegatives,
    "in_batch_hard": InBatchHardNegatives,
}


def create_negative_strategy(
    name: str = "none",
    **kwargs,
) -> NegativeMiningStrategy:
    """
    Factory function to create a negative mining strategy.

    Args:
        name: Strategy name. Use '+' to compose multiple strategies.
              E.g.: 'text_swap+number_perturb'
        **kwargs: Passed to strategy constructors.

    Returns:
        A NegativeMiningStrategy instance.
    """
    if "+" in name:
        # Composite: split and create each
        strategy_names = [n.strip() for n in name.split("+")]
        strategies = [create_negative_strategy(n, **kwargs) for n in strategy_names]
        return CompositeStrategy(strategies)

    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown negative mining strategy: '{name}'. "
            f"Registered: {list(STRATEGY_REGISTRY.keys())}. "
            f"Use '+' to compose: 'text_swap+number_perturb'"
        )

    return STRATEGY_REGISTRY[name](**kwargs)


def register_strategy(name: str, cls: type):
    """
    Register a custom negative mining strategy.

    Usage:
        class MyStrategy(NegativeMiningStrategy):
            def mine(self, batch):
                ...

        register_strategy("my_strategy", MyStrategy)
    """
    STRATEGY_REGISTRY[name] = cls
