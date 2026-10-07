import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kd,vd,width,bias", [(128, 256, 4, False), (128, 256, 4, True),
                                          (33, 65, 3, True), (1024, 2048, 4, False)])
def test_prefix_conv_matches_fp32_and_captures(kd, vd, width, bias):
    from packed_encoders._kernels.prefix_conv import continue_conv
    torch.manual_seed(3)
    channels = 2 * kd + vd
    x = torch.randn(100, channels + 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(channels, device="cuda", dtype=torch.bfloat16) if bias else None
    rows = torch.tensor([5, 23, 65, 82], device="cuda")
    taps = torch.randint(0, 100, (4, width), device="cuda")
    # Strided outputs exercise both the fused and FLA layouts.
    backing = torch.zeros(100, channels + 16, device="cuda", dtype=torch.bfloat16)
    outs = backing[:, :kd], backing[:, kd:2*kd], backing[:, 2*kd:channels]
    def reference():
        acc = (x[taps, :channels].float() * w.T.float()).sum(1)
        return F.silu(acc if b is None else acc + b.float()).bfloat16()
    continue_conv(x, w, b, rows, taps, outs)
    torch.testing.assert_close(backing[rows, :channels], reference(), rtol=0.01, atol=0.015)
    untouched = torch.ones(100, dtype=torch.bool, device="cuda")
    untouched[rows] = False
    assert not backing[untouched].any() and not backing[:, channels:].any()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        continue_conv(x, w, b, rows, taps, outs)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(backing[rows, :channels], reference(), rtol=0.01, atol=0.015)
