"""A whole-layer number for a real kernel, and `full` stops paying for the slow one.

TWO SEPARATE THINGS, previously one flag.

`include_pipeline_scope` adds an ALL-REFERENCE whole-layer cell: a python loop
over every expert, at up to T=8192, 25 warmup plus 3 trials. Measured on the
published H200 sweep it is 7.337 ms against vLLM's 0.588 at mixtral/T=512, a
12.5x ceiling nobody is trying to hit. Those 882 cells are the several-hour part
of `full` and they measure the reference rather than any kernel, so `full` no
longer asks for them.

`include_framework_pipeline` adds a whole-layer cell built around a REAL kernel:
ref_router plus vllm_fused_experts, timed as one. That is the only way the study
can answer "what does a full MoE layer cost", because every framework span
covers five of six stages and leaves the router out. It costs roughly one extra
cell per spec and is worth having.

The two must not share an `impl` name. Analyses key on that column, so an
all-reference whole-layer row and a vLLM whole-layer row landing under the same
label would average a python loop into a kernel result.
"""
from __future__ import annotations

import pytest

from moe.bench import profiles as PR
from moe.bench.driver import PIPELINE_SCOPE, is_pipeline_scope, pipeline_scope_for


def test_full_no_longer_pays_for_the_all_reference_whole_layer():
    """The several-hour part of the publication sweep, for no kernel data."""
    assert PR.PROFILES["full"].include_pipeline_scope is False


def test_full_does_ask_for_the_framework_whole_layer():
    assert PR.PROFILES["full"].include_framework_pipeline is True


def test_the_two_whole_layer_kinds_get_DIFFERENT_impl_names():
    """The whole point. `compare.py` builds by[(tokens, impl)]; a collision here
    would blend a 7.3 ms python loop into a 0.588 ms kernel series."""
    assert pipeline_scope_for("vllm_fused_experts") != PIPELINE_SCOPE
    assert pipeline_scope_for("sglang_fused_experts") != pipeline_scope_for(
        "vllm_fused_experts")


def test_the_reference_whole_layer_keeps_its_PUBLISHED_name():
    """3,528 rows in each published bf16 arm carry `__pipeline__`. Renaming it
    would orphan them from anything comparing against a future run."""
    assert PIPELINE_SCOPE == "__pipeline__"


def test_the_framework_name_says_which_kernel_it_wrapped():
    got = pipeline_scope_for("vllm_fused_experts")
    assert "vllm_fused_experts" in got
    assert got.startswith(PIPELINE_SCOPE)


@pytest.mark.parametrize("name", [
    PIPELINE_SCOPE,
    "__pipeline__:vllm_fused_experts",
    "__pipeline__:sglang_fused_experts",
])
def test_every_whole_layer_name_is_recognised_as_whole_layer(name):
    """The driver decides span=None from this. A framework whole-layer cell not
    recognised here would be resolved as a span name and raise."""
    assert is_pipeline_scope(name)


@pytest.mark.parametrize("name", [
    "vllm_fused_experts", "torch_grouped_mm_up", "ref_router", "",
])
def test_a_real_span_is_not_mistaken_for_whole_layer(name):
    assert not is_pipeline_scope(name)


def test_the_base_env_does_not_get_framework_whole_layer_cells():
    """base has no framework span to wrap, and the all-reference cell is the
    other flag's job."""
    prof = PR.Profile(
        name="t", models=("toy",), token_counts=(8,), dtypes=("bf16",),
        routings=(PR.RoutingSpec("uniform"),),
        include_pipeline_scope=False, include_framework_pipeline=True)
    got = [impl for _, _, impl in PR.cells(prof, env="base")]
    assert not any(is_pipeline_scope(i) for i in got)


def test_turning_both_off_yields_no_whole_layer_cells_at_all():
    prof = PR.Profile(
        name="t", models=("toy",), token_counts=(8,), dtypes=("bf16",),
        routings=(PR.RoutingSpec("uniform"),),
        include_pipeline_scope=False, include_framework_pipeline=False)
    for env in (None, "base"):
        got = [impl for _, _, impl in PR.cells(prof, env=env)]
        assert not any(is_pipeline_scope(i) for i in got)


def test_the_reference_cell_still_appears_when_asked_for():
    """Removing the default must not remove the capability: it is a real, if
    slow, lower bound and `--include-reference` style use still wants it."""
    prof = PR.Profile(
        name="t", models=("toy",), token_counts=(8,), dtypes=("bf16",),
        routings=(PR.RoutingSpec("uniform"),),
        include_pipeline_scope=True, include_framework_pipeline=False)
    got = [impl for _, _, impl in PR.cells(prof, env="base")]
    assert PIPELINE_SCOPE in got
