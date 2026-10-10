"""Uniform sampler — the baseline against which every curriculum is judged.

The uniform distribution is exactly ``1/|X|`` on every step regardless
of stats or step. Still goes through the floor + cap primitive so the
returned :class:`SamplerDistribution` bundle is structurally identical
to the curriculum samplers (epsilon=0, q_max=None is the no-op call).

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§4 Priority 1b (Uniform is the control condition for every Stage D run).
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class UniformSampler(Sampler):
    """Sampler whose distribution is exactly ``1/|X|`` everywhere."""

    name = SamplerName.UNIFORM

    def __init__(self, n_contexts: int):
        """Bind to a fixed bank size; uniform mass per context is ``1/n``."""
        super().__init__(n_contexts)
        self._uniform = np.full(self.n_contexts, 1.0 / self.n_contexts, dtype=np.float64)

    def update(self, stats: SamplerStats, step: int) -> None:
        """No-op — uniform has no state to update."""
        del stats, step

    def distribution(self, step: int) -> SamplerDistribution:
        """Always return the uniform distribution wrapped in the bundle."""
        del step
        return apply_floor_and_cap(self._uniform, epsilon=0.0, q_max=None)
