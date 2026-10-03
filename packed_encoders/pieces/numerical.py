"""Normalization, projection and activation pieces with PyTorch references."""

import torch
import torch.nn.functional as F

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Contract, Piece

LN = Contract("layer_norm", "[...,H], [H] -> [...,H]",
              "bias-free; population variance; bf16 CUDA or fp32 masters under bf16 autocast")
ADD_LN = Contract("add_layer_norm", "x,residual [...,H], [H] -> sum,norm [...,H]",
                  "sum rounded to input dtype before bias-free LN; two owned outputs; combined residual gradients")
LINEAR = Contract("linear", "[...,I], [O,I] -> [...,O]", "bias-free projection; bf16 CUDA or autocast")
GEGLU = Contract("geglu", "[...,2I] -> [...,I]",
                 "erf GELU(first half) * second half; fp32 intermediates; backward may overwrite saved projection")


def reference_ln(x, weight, eps):
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), None, eps).to(x.dtype)


def reference_add_ln(x, residual, weight, eps):
    summed = x + residual
    return summed, reference_ln(summed, weight, eps)


def reference_linear(x, weight):
    return F.linear(x, weight)


def reference_geglu(proj):
    a, gate = proj.float().chunk(2, dim=-1)
    return (F.gelu(a, approximate="none") * gate).to(proj.dtype)


def check_ln(x, weight, eps):
    if x.ndim not in (2, 3) or weight.shape != (x.shape[-1],) or x.device != weight.device or eps <= 0:
        raise UnsupportedTargetError("LayerNorm requires [T,H]/[B,S,H], a matching [H] scale and positive epsilon")


def check_add_ln(x, residual, weight, eps):
    check_ln(x, weight, eps)
    if x.shape != residual.shape or x.device != residual.device or x.dtype != residual.dtype:
        raise UnsupportedTargetError("residual LayerNorm requires matching residual shape, device and dtype")


def check_linear(x, weight):
    if x.ndim not in (2, 3) or weight.ndim != 2 or x.shape[-1] != weight.shape[1] or x.device != weight.device:
        raise UnsupportedTargetError("linear requires [T,I]/[B,S,I] and matching [O,I] weight")


def check_geglu(proj):
    if proj.ndim not in (2, 3) or proj.shape[-1] % 2:
        raise UnsupportedTargetError("GeGLU requires [...,2I] with activation followed by gate")


def layer_norm() -> Piece:
    from packed_encoders.ops import fused_layer_norm

    return Piece("layer-norm", LN, fused_layer_norm, reference_ln, check_ln)


def add_layer_norm() -> Piece:
    from packed_encoders.ops import fused_add_layer_norm

    return Piece("residual-layer-norm", ADD_LN, fused_add_layer_norm, reference_add_ln, check_add_ln)


def linear(*, dense=False) -> Piece:
    from packed_encoders.ops import _linear

    return Piece("dense-linear" if dense else "linear", LINEAR,
                 reference_linear if dense else _linear, reference_linear, check_linear)


def geglu() -> Piece:
    from packed_encoders.ops import fused_geglu

    return Piece("geglu", GEGLU, fused_geglu, reference_geglu, check_geglu)
