import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_prefix_lengths_exact_and_capturable(seed):
    from packed_encoders._kernels.prefix_match import pairwise_prefix_lengths
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, 20, (1025,), generator=g)
    seqs = [base[:1], base[:31], base[:32], base[:33], base[:63], base[:64], base[:65], base]
    seqs += [torch.cat([base[:n], torch.randint(0, 20, (13,), generator=g)]) for n in (31, 64, 1024)]
    seqs += [base.clone()]
    ids = torch.cat(seqs).cuda()
    cu = torch.tensor([0] + list(torch.tensor(list(map(len, seqs))).cumsum(0)), device="cuda")
    want = torch.zeros(len(seqs), len(seqs), dtype=torch.int32)
    for r in range(len(seqs)):
        for p in range(r):
            n = min(len(seqs[r]), len(seqs[p]))
            want[r, p] = (seqs[r][:n] == seqs[p][:n]).int().cumprod(0).sum()
    torch.testing.assert_close(pairwise_prefix_lengths(ids, cu).cpu(), want)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pairwise_prefix_lengths(ids, cu)
    ids.zero_()  # replay must read the new tokens, not specialize on their values
    graph.replay()
    for r in range(len(seqs)):
        for p in range(r):
            want[r, p] = min(len(seqs[r]), len(seqs[p]))
    torch.testing.assert_close(out.cpu(), want)


@pytest.mark.parametrize("rows", [2, 8, 32, 64, 65])
def test_gpu_plan_matches_original(rows):
    from packed_encoders.arch.qwen3_5.sharing import plan_shared_prefixes
    g = torch.Generator().manual_seed(rows)
    prefixes = [torch.randint(0, 1000, (n,), generator=g) for n in (63, 64, 65, 128)]
    seqs = [torch.cat([prefixes[i % 4], torch.randint(0, 1000, (i + 1,), generator=g)]) for i in range(rows)]
    ids, lengths = torch.cat(seqs).cuda(), list(map(len, seqs))
    original = plan_shared_prefixes(ids, lengths, min_prefix=64, conv_width=4, _use_cute=False)
    got = plan_shared_prefixes(ids, lengths, min_prefix=64, conv_width=4)
    if original is None:
        assert got is None
    else:
        for key, expected in vars(original).items():
            actual = getattr(got, key)
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected
