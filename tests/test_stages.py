import pytest

from moe.stages import (CANONICAL_STAGES, contiguous, contract_for,
                        exposed_writes, get, registry)


def test_single_stage_contract_matches_declaration():
    c = contract_for(("up_gemm",))
    assert c.reads == {"x_perm", "expert_offsets"}
    assert c.writes == {"h_up"}


def test_fused_span_hides_its_intermediate():
    # down_gemm + unpermute: y_perm is a register value inside the fused kernel,
    # so it is never exposed to later spans.
    c = contract_for(("down_gemm", "unpermute"))
    assert c.reads == {"h_act", "expert_offsets", "perm_index", "topk_weights"}
    assert "y_perm" in c.writes            # produced internally
    assert exposed_writes(("down_gemm", "unpermute")) == {"y"}


def test_fused_up_and_act():
    c = contract_for(("up_gemm", "act"))
    assert c.reads == {"x_perm", "expert_offsets"}
    assert exposed_writes(("up_gemm", "act")) == {"h_act"}


def test_noncontiguous_span_rejected():
    assert not contiguous(("up_gemm", "down_gemm"))
    with pytest.raises(ValueError, match="not contiguous"):
        contract_for(("up_gemm", "down_gemm"))


def test_reversed_span_rejected():
    with pytest.raises(ValueError, match="not contiguous"):
        contract_for(("act", "up_gemm"))


def test_duplicate_stage_rejected():
    with pytest.raises(ValueError, match="covers a stage twice"):
        contract_for(("act", "act"))


def test_unknown_stage_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        contract_for(("attention",))


def test_reference_spans_are_registered_for_every_stage():
    reg = registry()
    for stage in CANONICAL_STAGES:
        assert f"ref_{stage}" in reg


def test_reference_spans_run_without_cuda():
    for stage in CANONICAL_STAGES:
        assert get(f"ref_{stage}").requires_cuda is False


def test_unknown_span_error_lists_options():
    with pytest.raises(KeyError, match="registered:"):
        get("no_such_span")
