import pytest
import torch

from moe import pipeline as P
from moe.reference import torch_ref as R
from moe.routing.imbalance import counts_from_offsets
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import StageSpan, register
from moe.state import MoEState

REF = P.reference_pipeline_names()


def run(spec, forced=None, names=REF):
    x, weights = R.make_inputs(spec)
    st = MoEState(spec=spec, weights=weights, x=x)
    if forced is not None:
        st.forced_topk_ids = forced
    P.build(names, spec=spec).run(st, validate_shapes=True)
    return st, x, weights


def test_reference_pipeline_runs_on_cpu(toy_spec):
    st, _, _ = run(toy_spec)
    assert st.y.shape == (toy_spec.num_tokens, toy_spec.model.hidden_size)
    assert torch.isfinite(st.y).all()


def test_offsets_are_valid_csr(toy_spec):
    st, _, _ = run(toy_spec)
    sizes = counts_from_offsets(st.expert_offsets)
    assert sum(sizes) == toy_spec.rows
    assert int(st.expert_offsets[0]) == 0
    assert int(st.expert_offsets[-1]) == toy_spec.rows


def test_perm_index_is_a_permutation(toy_spec):
    st, _, _ = run(toy_spec)
    perm = st.perm_index.long()
    assert torch.equal(torch.sort(perm).values, torch.arange(toy_spec.rows))


def test_permuted_rows_are_expert_contiguous(toy_spec):
    st, _, _ = run(toy_spec)
    flat_experts = st.topk_ids.reshape(-1)[st.perm_index.long()]
    assert torch.equal(flat_experts, torch.sort(flat_experts).values)


def test_x_perm_rows_match_their_source_token(toy_spec):
    st, x, _ = run(toy_spec)
    src = st.perm_index.long() // toy_spec.model.top_k
    assert torch.equal(st.x_perm, x[src])


def test_matches_golden_fp32(toy_spec):
    st, x, weights = run(toy_spec)
    golden = R.golden_forward(toy_spec, weights, x)
    assert torch.allclose(st.y, golden, atol=1e-5, rtol=1e-5)


def test_fused_tiling_agrees_with_unfused(toy_spec):
    """The load-bearing claim of the span architecture."""
    fused_names = ["ref_router", "ref_permute", "ref_up_gemm", "ref_act",
                   "ref_fused_down_scatter"]
    st_a, _, _ = run(toy_spec)
    st_b, _, _ = run(toy_spec, names=fused_names)
    assert torch.allclose(st_a.y, st_b.y, atol=1e-6, rtol=1e-6)


def test_empty_experts_are_handled(toy_spec):
    """Under real skew most experts receive zero tokens. Send everything to
    expert 0 and expert 1 and make sure nothing divides by an empty group."""
    T, k = toy_spec.num_tokens, toy_spec.model.top_k
    forced = torch.zeros((T, k), dtype=torch.int32)
    forced[:, 1] = 1
    st, x, weights = run(toy_spec, forced=forced)
    sizes = counts_from_offsets(st.expert_offsets)
    assert sizes[0] == T and sizes[1] == T
    assert sizes[2:] == [0] * (toy_spec.model.num_experts - 2)
    golden = R.golden_forward(toy_spec, weights, x, forced_topk_ids=forced)
    assert torch.allclose(st.y, golden, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(st.y).all()


def test_single_expert_receives_everything(toy_spec):
    T, k = toy_spec.num_tokens, toy_spec.model.top_k
    forced = torch.full((T, k), 3, dtype=torch.int32)
    st, _, _ = run(toy_spec, forced=forced)
    sizes = counts_from_offsets(st.expert_offsets)
    assert sizes[3] == T * k
    assert torch.isfinite(st.y).all()


@pytest.mark.parametrize("dtype,atol", [("fp32", 1e-5), ("bf16", 3e-2), ("fp16", 5e-3)])
def test_low_precision_paths_stay_within_tolerance(dtype, atol):
    spec = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=64, dtype=dtype,
                     routing=RoutingSpec("uniform"))
    st, x, weights = run(spec)
    golden = R.golden_forward(spec, weights, x)
    err = (st.y.float() - golden.float()).abs().max().item()
    assert err < atol, f"{dtype}: max abs error {err:.3e} exceeded {atol:.1e}"


def test_shape_validation_catches_a_wrong_output_shape(toy_spec):
    @register
    class _BadUpGemm(StageSpan):
        name = "t_bad_up_gemm"
        covers = ("up_gemm",)
        requires_cuda = False
        dtypes = ("fp32",)

        def __call__(self, st: MoEState) -> None:
            n = st.spec.rows
            st.h_up = torch.zeros((n, st.spec.model.intermediate_size))  # missing the 2x

    names = ["ref_router", "ref_permute", "t_bad_up_gemm", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    x, weights = R.make_inputs(toy_spec)
    st = MoEState(spec=toy_spec, weights=weights, x=x)
    with pytest.raises(P.PipelineError, match="t_bad_up_gemm: state.h_up"):
        P.build(names, spec=toy_spec).run(st, validate_shapes=True)


def test_validate_rejects_a_field_name_that_is_not_a_state_field(toy_spec):
    """A typo in a span's `materialises` must raise the module's ValueError,
    which pipeline.run turns into a named PipelineError, rather than an
    AttributeError that escapes as an unclassified crash."""
    x, weights = R.make_inputs(toy_spec)
    st = MoEState(spec=toy_spec, weights=weights, x=x)
    with pytest.raises(ValueError, match="not MoEState fields"):
        st.validate(only=frozenset({"expert_offset"}))


def test_validate_with_an_empty_selection_checks_nothing(toy_spec):
    """`only or STATE_FIELDS` treated an empty writes set as 'check everything',
    inverting the intent."""
    x, weights = R.make_inputs(toy_spec)
    st = MoEState(spec=toy_spec, weights=weights, x=x)
    st.h_up = torch.zeros((3, 3))          # wrong shape, deliberately
    st.validate(only=frozenset())          # must not raise
    with pytest.raises(ValueError, match="state.h_up"):
        st.validate()
