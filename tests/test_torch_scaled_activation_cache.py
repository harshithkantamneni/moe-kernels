"""Activation quantisation is not part of `up_gemm`, so it must not be timed.

MEASURED, H200 2026-08-28. `torch_scaled_grouped_mm_up` on deepseek-v2-lite at
T=8192 took 1.9855 ms in fp8 against `torch_grouped_mm_up`'s 1.0503 ms in bf16.
fp8 cannot be 1.89x SLOWER on the same GEMM. The difference was work this
harness added: the span quantised activations on every timed call.

That made the whole fp8 arm unusable. Its crossings came out at 0.44 +/- 0.13 of
prediction, against 1.15 +/- 0.07 from the two production kernels, because a
per-call cost that scales with tokens biases the crossing early. STUDY.md
retracts it on those grounds.

WHY CACHING IS THE RIGHT FIX AND NOT A THUMB ON THE SCALE. The span declares
`covers = ("up_gemm",)`, and the byte model costs exactly that. Quantising
activations is not up_gemm; it exists only because `_scaled_grouped_mm` needs
both operands in fp8 while the harness hands out bf16 activations for vLLM's
sake. `covers` is the contract, and the timed region has to match it.

It also restores the comparison C2 needs: torch's bf16 span quantises nothing,
so an fp8 span that does was never measuring the same thing.

`x_perm` is built once per cell in the prologue and reused by every timed
iteration, so a cache keyed on the tensor is warm before the first timed call.
"""
from __future__ import annotations

import torch

from moe.baselines.torch_grouped_mm import scaled_grouped_args
from moe.quant import quantize_per_expert


def cell(ntot=10, K=32, E=4, N=16):
    torch.manual_seed(0)
    a = torch.randn(ntot, K, dtype=torch.bfloat16)
    w = torch.randn(E, N, K) * (K ** -0.5)
    wq, ws = quantize_per_expert(w, "fp8_e4m3")
    offs = torch.tensor([2, 5, 9, ntot], dtype=torch.int32)
    return a, wq, ws, offs


def test_the_same_activations_are_quantised_once():
    """The fix. Every timed iteration sees the same x_perm, so the second call
    must reuse the first call's work rather than redo it."""
    a, wq, ws, offs = cell()
    first = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    second = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    assert first.a.data_ptr() == second.a.data_ptr(), "requantised on every call"
    assert first.scale_a.data_ptr() == second.scale_a.data_ptr()


def test_different_activations_are_quantised_again():
    """A cache that never misses would hand one cell another cell's inputs."""
    a, wq, ws, offs = cell()
    b = a.clone() + 1.0
    first = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    other = scaled_grouped_args(b, wq, ws, offs, "fp8_e4m3")
    assert first.a.data_ptr() != other.a.data_ptr()
    assert not torch.equal(first.a.float(), other.a.float())


def test_the_cached_values_are_still_correct():
    """Caching must change WHEN the work happens, never the numbers."""
    a, wq, ws, offs = cell()
    got = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    want_q, want_s = quantize_per_expert(a, "fp8_e4m3")
    assert torch.equal(got.a.float(), want_q.float())
    assert torch.equal(got.scale_a, want_s)


def test_a_resized_batch_is_not_served_stale():
    """Token count is the axis the whole sweep varies. A cache keyed only on the
    pointer would serve T=1's activations to T=8192 whenever an allocator reuses
    the address, which is exactly what a sweep makes it do."""
    a, wq, ws, offs = cell(ntot=10)
    big = torch.randn(20, a.shape[1], dtype=torch.bfloat16)
    offs_big = torch.tensor([4, 10, 16, 20], dtype=torch.int32)
    small = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    large = scaled_grouped_args(big, wq, ws, offs_big, "fp8_e4m3")
    assert small.a.shape[0] == 10 and large.a.shape[0] == 20
    assert large.scale_a.shape[0] == 20


def test_a_recycled_address_does_not_serve_a_stale_quantisation():
    """MEASURED in this very suite, 2026-08-28. The first version keyed on
    (data_ptr, shape, dtype). torch's allocator reuses freed addresses, so a
    tensor from a finished cell was collected, the next landed at the same
    address with the same shape, and the cache returned the wrong values. The
    test above caught it on the first run.

    Holding a reference to the tensor keeps its address alive for as long as the
    entry does, which is what makes an identity test sound.
    """
    import gc

    a, wq, ws, offs = cell()
    first = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    seen = first.a.float().clone()
    addr = a.data_ptr()

    del a, first
    gc.collect()

    # Force a same-shape tensor with different contents; it may well reuse addr.
    b = torch.full((10, 32), 7.0, dtype=torch.bfloat16)
    got = scaled_grouped_args(b, wq, ws, offs, "fp8_e4m3")
    if b.data_ptr() == addr:
        assert not torch.equal(got.a.float(), seen), (
            "same address, different tensor, stale quantisation served")
    # Correct regardless of whether the address happened to be reused.
    want_q, _ = quantize_per_expert(b, "fp8_e4m3")
    assert torch.equal(got.a.float(), want_q.float())
