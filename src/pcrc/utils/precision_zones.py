"""fp32-zone helper for precision-sensitive arithmetic under bf16 / mixed-precision training.

When a model runs under autocast (bf16), tensors derived from parameter math arrive in bf16. A few
compute sites — tight-clip probability ratios ``exp(lp_new - lp_old)``, importance-weighted means,
softmax over many candidates — fall below bf16's ULP and silently collapse to identity / pegged
values under naive bf16.

The fix: enter an fp32 zone right before the fragile math. ``fp32_zone`` disables any active
autocast region; ``to_fp32`` materializes explicit fp32 copies of the inputs so the math runs at
fp32 regardless of the caller's autocast state.

Usage::

    with fp32_zone():
        lpo_f, lpn_f = to_fp32(log_p_old, log_p_new)
        ratio = (lpn_f - lpo_f).clamp(*ratio_clip).exp()

Bit-identity under fp32: if the inputs are already fp32 and no outer autocast is active, the wrap is
a no-op — ``to_fp32`` returns the same tensor (``.float()`` on an fp32 tensor returns ``self``), and
disabling an inactive autocast does nothing — so a call site can be wrapped unconditionally without
perturbing an fp32 baseline.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def fp32_zone() -> Iterator[None]:
    """Disable any active autocast so enclosed math runs at full precision.

    No-op when no surrounding autocast region is active. Pairs with
    ``to_fp32`` for the explicit input upcast. Both device contexts are
    stacked so the zone is robust against either device's autocast.
    """
    with (
        torch.autocast(device_type="cuda", enabled=False),
        torch.autocast(device_type="cpu", enabled=False),
    ):
        yield


def to_fp32(*tensors: Any) -> tuple[Any, ...]:
    """Materialize fp32 copies of input tensors; pass non-tensors through.

    ``Tensor.float()`` is a no-op on a tensor already in fp32 (returns
    ``self``), so this is free under the current fp32 training path.
    """
    return tuple(t.float() if torch.is_tensor(t) else t for t in tensors)
