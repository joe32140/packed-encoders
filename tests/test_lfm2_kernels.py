"""Independent arithmetic checks for LFM2 inference and shared-prefix kernels."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("width", [256, 2048])
@pytest.mark.parametrize("with_residual", [False, True])
def test_inference_norm_preserves_fla_arithmetic_and_inputs(width, with_residual):
    from fla.modules.layernorm import rms_norm as reference
    from packed_encoders._kernels.inference_norm import rms_norm, add_rms_norm

    gen = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(17, width, device="cuda", dtype=torch.bfloat16, generator=gen)
    residual = torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=gen)
    w = torch.randn(width, device="cuda", dtype=torch.float32, generator=gen)
    original_x, original_r = x.clone(), residual.clone()
    with torch.no_grad():
        if with_residual:
            expected, expected_sum = reference(x, w, None, residual=residual, eps=1e-5, prenorm=True)
            actual, summed = add_rms_norm(x, residual, w, 1e-5)
            assert torch.equal(summed, expected_sum)
            assert summed.data_ptr() != residual.data_ptr()
        else:
            expected = reference(x, w, None, eps=1e-5)
            actual = rms_norm(x, w, 1e-5)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
    assert torch.equal(x, original_x)
    assert torch.equal(residual, original_r)


@pytest.mark.parametrize("width", [1, 3, 5])
@pytest.mark.parametrize("strided_y", [False, True])
def test_conv_boundary_correction_keeps_unselected_rows_and_rounding(width, strided_y):
    from packed_encoders._kernels.short_conv import continue_conv
    from packed_encoders.runtime.sharing import _layout

    plan = _layout([12, 9, 16], [0, 12, 21], {0: (8, [0, 1, 2])}, width)
    d, n = 256, len(plan.src)
    gen = torch.Generator(device="cuda").manual_seed(24)
    proj = torch.randn(n, 3 * d, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(d, width, device="cuda", dtype=torch.bfloat16, generator=gen)
    y = torch.randn(n, d * (2 if strided_y else 1), device="cuda", dtype=torch.bfloat16, generator=gen)
    if strided_y:
        y = y[:, ::2]
    expected = y.clone()
    rows, taps = plan.fix_rows.cuda(), plan.fix_taps.cuda()
    if rows.numel():
        values = proj[taps]
        bx = (values[..., :d].float() * values[..., 2 * d:].float()).bfloat16().float()
        conv = torch.einsum("fkc,ck->fc", bx, w.float()).bfloat16().float()
        expected[rows] = (proj[rows, d:2 * d].float() * conv).bfloat16()
    continue_conv(proj, w, y, rows, taps)
    unchanged = torch.ones(n, dtype=torch.bool, device="cuda")
    unchanged[rows] = False
    assert torch.equal(y[unchanged], expected[unchanged])
    if rows.numel():
        assert F.cosine_similarity(y[rows].float(), expected[rows].float(), dim=-1).min() > 0.9999
