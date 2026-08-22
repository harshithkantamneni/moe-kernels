"""The correctness gate must reject wrong kernels.

This is the regression for the worst defect this harness has had: the gate was
formerly satisfied by an all-zeros output at every geometry and dtype the sweep
publishes. Two independent reviews found it, and neither of the two existing
"wrong kernel" tests caught it, because both ran only at fp32.

Every case below runs at bf16, fp16 and fp32 through `driver.check_correctness`,
which is the exact code path the driver gates on.
"""
import pytest
import torch

from moe.bench import driver as D
from moe.reference import torch_ref as R
from moe.spec import BenchSpec, MoEConfig, RoutingSpec
from moe.stages import StageSpan, register
from moe.state import MoEState

# Realistically shaped (fan-in scaling, non-power-of-two F, several experts,
# some of which receive no tokens) but small enough to run on a laptop CPU.
# The tolerance model is quantisation-dominated, so it is essentially the same
# number here as at Mixtral geometry; test_tolerance pins that separately.
GEOM = MoEConfig(name="gauntlet", hidden_size=512, intermediate_size=1408,
                 num_experts=8, top_k=2, num_layers=1, verified=True)


def up_gemm(st):
    return R.grouped_gemm_loop(st.x_perm, st.weights.w1, st.expert_offsets,
                               2 * st.spec.model.intermediate_size)


class _Broken(StageSpan):
    covers = ("up_gemm",)
    requires_cuda = False
    dtypes = ("fp32", "fp16", "bf16")


@register
class GaCorrect(_Broken):
    name = "ga_correct"

    def __call__(self, st: MoEState) -> None:
        st.h_up = up_gemm(st)


@register
class GaZeros(_Broken):
    name = "ga_zeros"

    def __call__(self, st: MoEState) -> None:
        st.h_up = torch.zeros_like(up_gemm(st))


@register
class GaScaled15(_Broken):
    name = "ga_scale_1p5"

    def __call__(self, st: MoEState) -> None:
        st.h_up = up_gemm(st) * 1.5


@register
class GaScaled3(_Broken):
    name = "ga_scale_3"

    def __call__(self, st: MoEState) -> None:
        st.h_up = up_gemm(st) * 3.0


@register
class GaSignFlip(_Broken):
    name = "ga_sign_flip"

    def __call__(self, st: MoEState) -> None:
        st.h_up = -up_gemm(st)


@register
class GaGateUpSwapped(_Broken):
    name = "ga_gate_up_swapped"

    def __call__(self, st: MoEState) -> None:
        h = up_gemm(st)
        gate, up = h.chunk(2, dim=-1)
        st.h_up = torch.cat([up, gate], dim=-1)


@register
class GaSkipTail(_Broken):
    name = "ga_skip_tail"

    def __call__(self, st: MoEState) -> None:
        """The dangerous one: skipping the trailing tile is both FASTER and,
        under a loose tolerance, passing. This is the failure mode that would
        directly inflate a published speedup."""
        h = up_gemm(st)
        cut = max(1, int(h.shape[0] * 0.8))
        h[cut:] = 0
        st.h_up = h


BROKEN = ["ga_zeros", "ga_scale_1p5", "ga_scale_3", "ga_sign_flip",
          "ga_gate_up_swapped", "ga_skip_tail"]
DTYPES = ["bf16", "fp16", "fp32"]


def check(impl: str, dtype: str):
    spec = BenchSpec(GEOM, num_tokens=64, dtype=dtype,
                     routing=RoutingSpec("zipf", 1.0), seed=0)
    names = ["ref_router", "ref_permute", impl, "ref_act", "ref_down_gemm",
             "ref_unpermute"]
    from moe.pipeline import build
    cfg = D.RunConfig(device="cpu")
    x, weights = R.make_inputs(spec, device="cpu")
    return D.check_correctness(spec, build(names, spec=spec), x, weights, None, cfg)[0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_correct_kernel_passes(dtype):
    r = check("ga_correct", dtype)
    assert r.passed, (f"a correct kernel must pass at {dtype}: "
                      f"rel={r.rel_err:.3e} tol={r.tol_rel_max:.3e}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("impl", BROKEN)
def test_broken_kernel_is_rejected(impl, dtype):
    r = check(impl, dtype)
    assert not r.passed, (f"{impl} PASSED at {dtype}: rel={r.rel_err:.3e} "
                          f"tol={r.tol_rel_max:.3e}. The gate is vacuous.")


@pytest.mark.parametrize("dtype", DTYPES)
def test_headroom_between_correct_and_broken_is_large(dtype):
    """The correct kernel should sit well inside the budget and the tightest
    broken one well outside, so the gate is not balanced on a knife edge."""
    correct = check("ga_correct", dtype)
    worst_broken = min(check(i, dtype).rel_err for i in BROKEN)
    assert correct.rel_err < correct.tol_rel_max / 4, (
        f"{dtype}: correct kernel uses most of its budget "
        f"({correct.rel_err:.2e} of {correct.tol_rel_max:.2e})")
    assert worst_broken > correct.tol_rel_max * 4, (
        f"{dtype}: the tightest broken kernel ({worst_broken:.2e}) is close to "
        f"the budget ({correct.tol_rel_max:.2e})")
