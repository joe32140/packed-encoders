"""Per-model state attached by `pack()`.

`pack()` patches in place, so the state hangs off the encoder module under one
attribute. Keeping the original forward here makes the patch reversible and gives
`validate()` an oracle (stock HF) to compare the fused path against even after the
swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from packed_encoders.config import ModernBertParams
from packed_encoders.errors import PackedEncodersError

ATTR = "_packed_encoders"


@dataclass
class PatchState:
    params: ModernBertParams
    original_forward: Callable[..., Any]
    # "sdpa" (general fallback) | "flash" | "triton" | "auto" (packed score)
    attention_backend: str = "sdpa"
    graph_runner: Any = None        # graph._GraphRunner | None (kept loose to avoid a cycle)
    graph_enabled: bool = False
    graph_skip_warned: bool = False  # one-time warning when graphs are skipped (autocast/grad)
    packed_graph_runner: Any = None  # graph._PackedGraphRunner | None
    train_graph_runner: Any = None   # train_graph._TrainGraphRunner | None
    train_graph_enabled: bool = False
    validation_report: Any = None
    pieces: Any = None             # immutable ModernBertPieces selection
    execution: Any = None          # its pre-bound operation callables


# Kept separate from ATTR: existing kernels and direct packed_forward callers
# continue to read the original ModernBERT PatchState without another indirection.
INSTALL_ATTR = "_packed_encoders_installation"


def find_installation(target: object):
    from packed_encoders.locate import walk_targets

    for module in walk_targets(target):
        installed = getattr(module, INSTALL_ATTR, None)
        if installed is not None:
            return installed
    return None


def get_installation(target: object):
    installed = find_installation(target)
    if installed is None:
        raise PackedEncodersError("this model has not been patched with packed_encoders.pack()")
    return installed


def get_state(target: object):
    """Return the recorded engine's native runtime state without matching again."""
    return get_installation(target).packed.state
