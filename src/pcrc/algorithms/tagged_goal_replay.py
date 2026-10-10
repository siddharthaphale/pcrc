"""Tagged global replay buffer for sparse Fetch + HER (Priority 3 default).

A single FIFO buffer holds all transitions across all contexts; every
transition carries its ``source_context_id`` tag. The learner draws
minibatches in two stages: first sample context ids according to a
:class:`SamplerDistribution`, then sample transitions tagged with the
chosen context id. This is the single training-path default for every
sparse Fetch run; per-context buffers exist only as a debug adapter
(plan §4 Priority 3 resolved).

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§3.5 (API sketch), §4 Priority 3 (resolved: single training-path
default), §9 risk #4 (memory budget).
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from pcrc.algorithms.her import GoalConditionedTransition
from pcrc.samplers.base import SamplerDistribution


class TaggedGlobalReplay:
    """Global FIFO of :class:`GoalConditionedTransition` with per-context index.

    Construction is cheap (no preallocation); push/pop are O(1) under
    ``collections.deque``. ``sample_context_weighted`` runs in O(B)
    where ``B`` is the requested minibatch size.

    Attributes:
        capacity: maximum number of transitions stored. When exceeded
            the oldest transitions are dropped (FIFO).
    """

    def __init__(self, capacity: int):
        """Allocate the buffer; per-context indices are built lazily."""
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0; got {capacity}")
        self.capacity = int(capacity)
        # Main FIFO ring.
        self._buffer: deque[GoalConditionedTransition] = deque(maxlen=self.capacity)
        # Per-context membership index: source_context_id → deque of
        # transition objects currently in the buffer with that tag.
        # Kept in sync with ``_buffer`` so sample_context_weighted does
        # not have to filter the whole buffer.
        self._by_context: dict[int, deque[GoalConditionedTransition]] = defaultdict(lambda: deque())

    def __len__(self) -> int:
        """Number of transitions currently in the buffer."""
        return len(self._buffer)

    def add(self, transition: GoalConditionedTransition) -> None:
        """Append one transition; FIFO-evict the oldest on overflow."""
        # If at capacity, the deque-maxlen pop happens BEFORE the append
        # below. Mirror the eviction on the per-context index.
        if len(self._buffer) == self.capacity:
            evicted = self._buffer[0]
            ctx_q = self._by_context.get(evicted.source_context_id)
            if ctx_q is not None and ctx_q:
                # Evict the oldest tagged with this context — by
                # construction the FIFO order is preserved across
                # the main buffer and each per-context queue.
                ctx_q.popleft()
        self._buffer.append(transition)
        self._by_context[transition.source_context_id].append(transition)

    def context_ids_in_buffer(self) -> list[int]:
        """Sorted list of context ids with at least one transition stored."""
        return sorted(cid for cid, q in self._by_context.items() if q)

    def n_transitions_for_context(self, context_id: int) -> int:
        """Number of stored transitions tagged with ``context_id``."""
        return len(self._by_context.get(context_id, ()))

    def sample_context_weighted(
        self,
        batch_size: int,
        distribution: SamplerDistribution,
        rng: np.random.Generator,
    ) -> list[GoalConditionedTransition]:
        """Sample contexts per ``distribution.final``, then sample transitions.

        Implementation:
            1. Renormalize ``distribution.final`` over the subset of
               context ids that have at least one transition stored.
            2. Sample ``batch_size`` context ids with replacement
               from the renormalized distribution.
            3. For each chosen context id, sample one transition
               uniformly at random from the per-context queue.

        If the renormalized support is empty (no stored context
        appears in the sampler's support), raises ``ValueError`` —
        the sampler distribution and the buffer disagree about which
        contexts exist, which is a programmer error.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}")
        if len(self._buffer) == 0:
            raise ValueError("TaggedGlobalReplay is empty; cannot sample")

        n_contexts = distribution.final.shape[0]
        # Subset of distribution support that the buffer actually has.
        in_buffer = np.zeros(n_contexts, dtype=np.float64)
        for cid in range(n_contexts):
            if cid in self._by_context and self._by_context[cid]:
                in_buffer[cid] = distribution.final[cid]
        mass = float(in_buffer.sum())
        if mass <= 0:
            raise ValueError(
                "No overlap between SamplerDistribution support and stored "
                "context tags. Buffer contexts: "
                f"{self.context_ids_in_buffer()[:10]} ; distribution support "
                f"({int((distribution.final > 0).sum())} nonzero entries)."
            )
        p = in_buffer / mass

        chosen_contexts = rng.choice(n_contexts, size=batch_size, replace=True, p=p)
        out: list[GoalConditionedTransition] = []
        for cid in chosen_contexts:
            q = self._by_context[int(cid)]
            # Uniform sample within the per-context queue.
            j = int(rng.integers(0, len(q)))
            out.append(q[j])
        return out

    def sample_uniform(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[GoalConditionedTransition]:
        """Uniform sample over all stored transitions (debug / baseline).

        Not used by the Stage D training path — the sampler-weighted
        path above is the canonical interface. Useful for ablations
        and for the §7 baseline that asserts the sampler-weighted
        path actually changes the empirical context distribution.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}")
        if len(self._buffer) == 0:
            raise ValueError("TaggedGlobalReplay is empty; cannot sample")
        idxs = rng.integers(0, len(self._buffer), size=batch_size)
        return [self._buffer[int(i)] for i in idxs]
