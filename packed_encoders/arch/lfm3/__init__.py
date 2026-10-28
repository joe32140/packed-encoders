"""LFM3 hybrid backbones: LFM2's gated short convolutions and GQA attention, plus sliding-window
attention layers, a sigmoid output gate per attention head and sandwich norms.

The LFM2 engine and HF entry serve it unchanged (`arch.lfm2`, which reads each of these from the
module); only the class it binds differs. LFM3's modeling code ships with its checkpoints
(`trust_remote_code`) until a transformers release carries it, so `Lfm3Model` is matched by class
name and modeling module, from either source. What `pack()` installs per batch is LFM2's (see
`arch.lfm2`), with one difference: LFM3's convolution masks a lone left-padded row's inputs, so
such a call runs packed too. Grouped rows run through `PackedLfm2.forward_grouped`.
"""

from __future__ import annotations

from torch import nn

from packed_encoders.arch.lfm2 import HFLfm2Adapter, Lfm2Hybrid, Lfm2Report, PackedLfm2, grouped_attention_masks


def _lfm3_text_model(module: object) -> nn.Module | None:
    """`Lfm3Model`, or the one a wrapper holds as `.language_model`. By class name along the MRO and
    a `modeling_lfm3` module, transformers' own or a checkpoint's copy of it."""
    if not isinstance(module, nn.Module):
        return None
    if any(c.__name__ == "Lfm3Model" and c.__module__.rsplit(".", 1)[-1] == "modeling_lfm3"
           for c in type(module).__mro__):
        return module
    lm = getattr(module, "language_model", None)
    return lm if lm is not module and _lfm3_text_model(lm) is lm else None


class HFLfm3Adapter(HFLfm2Adapter):
    name = "hf-lfm3"
    find = staticmethod(_lfm3_text_model)


class Lfm3Hybrid(Lfm2Hybrid):
    name = "lfm3"
    family = "LFM3"
    adapters = (HFLfm3Adapter(),)
    single_row_conv_masked = True


__all__ = ["HFLfm3Adapter", "Lfm3Hybrid", "Lfm2Report", "PackedLfm2", "grouped_attention_masks"]
