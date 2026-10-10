"""Two-stage sampler distribution blending.

Implements the dormant / warmup / active phases of Gu et al. 2026 §3.5
as a distribution-blending primitive over `n_candidates` arms:

  step <  dormant_until                  → uniform
  dormant_until ≤ step < warmup_until    → (1-α(step))·uniform + α(step)·softmax(logits)
  step ≥ warmup_until                    → softmax(logits)

Default ramp is linear:
    α(step) = (step - dormant_until) / (warmup_until - dormant_until)
The unit tests in tests/cpu/test_two_stage_sampler.py are agnostic to
ramp shape; linear is the documented default.

The outer two-stage sampling surface (sampling X̃_t and X_t over a real
task bank) is built on top of this primitive in the actor-curator loop.
"""

from __future__ import annotations

from typing import Literal

import torch

from pcrc.utils.precision_zones import fp32_zone, to_fp32


class TwoStageSampler:
    def __init__(
        self,
        n_candidates: int,
        dormant_until: int,
        warmup_until: int,
        ramp: Literal["linear"] = "linear",
    ):
        if dormant_until < 0 or warmup_until < dormant_until:
            raise ValueError(
                f"need 0 ≤ dormant_until ≤ warmup_until, "
                f"got dormant_until={dormant_until}, warmup_until={warmup_until}"
            )
        if ramp != "linear":
            raise ValueError(f"ramp must be 'linear' (got {ramp!r})")
        self.n_candidates = int(n_candidates)
        self.dormant_until = int(dormant_until)
        self.warmup_until = int(warmup_until)
        self.ramp = ramp

    def get_distribution(
        self,
        step: int,
        curator_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Dormant/warmup/active distribution blend over the candidate batch.

        The softmax over ``n_candidates`` arms (= 500 in production) is
        wrapped in ``fp32_zone``: under bf16 curator outputs, softmax
        over 500 categories with bf16's ~3-decimal mantissa loses
        tail-mass resolution and the OSMD arm-mass diagnostic degrades.
        Return tensor is fp32 regardless of input dtype (downstream
        callers cast to float64 numpy explicitly, see
        the original actor-curator integration).
        """
        if curator_logits.shape != (self.n_candidates,):
            raise ValueError(
                f"curator_logits must be shape ({self.n_candidates},), "
                f"got {tuple(curator_logits.shape)}"
            )

        with fp32_zone():
            (logits_f,) = to_fp32(curator_logits)
            uniform = torch.full(
                (self.n_candidates,),
                1.0 / self.n_candidates,
                dtype=torch.float32,
                device=logits_f.device,
            )

            if step < self.dormant_until:
                return uniform

            if step >= self.warmup_until:
                return torch.softmax(logits_f, dim=0)

            # Warmup blend.
            warmup_len = self.warmup_until - self.dormant_until
            # warmup_len ≥ 1 here (otherwise either dormant or active branch hit).
            alpha = (step - self.dormant_until) / warmup_len
            softmax_curator = torch.softmax(logits_f, dim=0)
            return (1.0 - alpha) * uniform + alpha * softmax_curator
