"""Inference pieces for post-norm ("sandwich") residual blocks. References use PyTorch and round
where the eager module chain rounds; the Triton kernels are imported only in the factory."""
import torch
import torch.nn.functional as F

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Contract, Piece

RMS_ROWS = Contract("rms_norm_rows", "x [T,D] or [T,H,D] (unit last stride), scale [D] or None, input multiplier or None -> contiguous",
                    "x times the multiplier rounded to the input dtype first; fp32 norm and optional scale, round once", autograd=False)
POST_ADD = Contract("post_norm_add", "delta,residual [T,D], scale [D], layer scalar [1] or None, next scale [D] or None -> sum[, norm]",
                    "residual + RMSNorm(delta) * scale, times the layer scalar, then RMSNorm(sum) * next scale; each step rounded to the input dtype; owned outputs", autograd=False)
GELU_MUL = Contract("gelu_tanh_mul", "g,u [T,D] (unit last stride) -> [T,D]",
                    "tanh GELU(g) rounded to the input dtype, times u, rounded", autograd=False)


def _require(condition, message):
    if not condition:
        raise UnsupportedTargetError(message)


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


def _norm(x, w, eps):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y if w is None else y * w.float()).to(x.dtype)


def ref_rms(x, w, eps, in_scale=None):
    return _norm(x if in_scale is None else x * in_scale, w, eps).contiguous()


def check_rms(x, w, eps, in_scale=None):
    _require(x.ndim in (2, 3) and x.stride(-1) == 1 and _pow2(x.shape[-1]) and eps > 0 and
             (w is None or (w.shape == (x.shape[-1],) and w.dtype == torch.float32)),
             "RMSNorm rows require [T,D]/[T,H,D] with a power-of-two unit-stride last dim and an fp32 [D] scale")


def ref_post_add(delta, residual, w, eps, scalar=None, w_next=None):
    s = residual + _norm(delta, w, eps)
    if scalar is not None:
        s = s * scalar.to(s.dtype)
    return s if w_next is None else (s, _norm(s, w_next, eps))


def check_post_add(delta, residual, w, eps, scalar=None, w_next=None):
    _require(delta.ndim == 2 and delta.shape == residual.shape and delta.dtype == residual.dtype and
             delta.is_contiguous() and residual.is_contiguous() and _pow2(delta.shape[1]) and eps > 0,
             "post-norm add requires contiguous [T,D] delta and residual with a power-of-two D")
    _require(w is not None, "post-norm add requires an fp32 [D] primary scale")
    for scale in (w, w_next):
        _require(scale is None or (scale.shape == (delta.shape[1],) and scale.dtype == torch.float32),
                 "post-norm scales must be fp32 [D]")
    _require(scalar is None or (scalar.numel() == 1 and scalar.dtype == torch.float32), "layer scalar must be one fp32 value")


def ref_gelu_mul(g, u):
    return (F.gelu(g, approximate="tanh") * u).contiguous()


def check_gelu_mul(g, u):
    _require(g.ndim == 2 and g.shape == u.shape and g.dtype == u.dtype and g.stride(-1) == 1 and u.stride(-1) == 1,
             "tanh GELU gate requires matching [T,D] inputs with unit last stride")


def post_norm_pieces():
    from packed_encoders._kernels import post_norm as k

    return dict(
        rms_norm=Piece("triton-rms-norm-rows", RMS_ROWS, k.rms_norm, ref_rms, check_rms),
        post_norm_add=Piece("triton-post-norm-add", POST_ADD, k.post_norm_add, ref_post_add, check_post_add),
        gelu_mul=Piece("triton-gelu-tanh-mul", GELU_MUL, k.gelu_mul, ref_gelu_mul, check_gelu_mul),
    )
