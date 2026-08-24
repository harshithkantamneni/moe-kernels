"""Tests for the dry-run plan.

The dry run is what tells you a sweep is valid before you rent a GPU, and its
counting had no test at all while it lived inside the CLI's printing code.
"""
import pytest

from moe.bench import profiles as PR
from moe.spec import RoutingSpec
from moe.stages import StageSpan, register
from moe.state import MoEState

SMOKE = PR.get("smoke")


class _P(StageSpan):
    requires_cuda = False

    def __call__(self, st: MoEState) -> None:  # pragma: no cover - never run
        pass


@register
class PlanUp(_P):
    name = "p_up"
    covers = ("up_gemm",)
    dtypes = ("bf16",)


@register
class PlanFp32Only(_P):
    name = "p_fp32_only"
    covers = ("up_gemm",)
    dtypes = ("fp32",)


@register
class PlanVllm(_P):
    name = "p_vllm_up"
    covers = ("up_gemm",)
    dtypes = ("bf16",)
    env = "vllm"


def test_plan_counts_cells_and_modes():
    p = PR.plan(SMOKE, impl_filter=("p_up",))
    assert p.specs == len(SMOKE.specs())
    assert p.planned == p.specs          # one impl, supports every cell
    assert p.modes == 1                  # smoke: one l2 mode, no graph
    assert p.timing_rows == p.planned * p.modes
    assert p.ok


def test_unsupported_dtype_is_counted_not_reported_as_a_problem():
    """A bf16-only sweep against an fp32-only implementation is a skip, not an
    error. Counting it as a problem would make every dry run exit non-zero."""
    p = PR.plan(SMOKE, impl_filter=("p_fp32_only",))
    assert p.planned == 0
    assert p.unsupported == len(SMOKE.specs())
    assert p.problems == ()
    assert p.ok


def test_env_filter_selects_implementations():
    assert "p_vllm_up" in PR.plan(SMOKE, env="vllm").impls
    assert "p_vllm_up" not in PR.plan(SMOKE, env="base").impls
    assert "p_up" in PR.plan(SMOKE, env="base").impls


def test_no_implementations_is_a_valid_but_empty_plan():
    p = PR.plan(SMOKE, impl_filter=("nonexistent",))
    assert p.impls == () and p.planned == 0 and p.ok


def test_missing_trace_makes_the_plan_not_ok():
    """The dry run must refuse before the session, not discover it mid-sweep."""
    from dataclasses import replace
    prof = replace(SMOKE, routings=(RoutingSpec("trace", trace_id="not-captured"),))
    p = PR.plan(prof, impl_filter=("p_up",), traces=None)
    assert p.missing_traces == ("not-captured",)
    assert not p.ok


def test_present_trace_satisfies_the_plan():
    from dataclasses import replace

    class FakeTraces:
        def __contains__(self, tid):
            return tid == "captured"

    prof = replace(SMOKE, routings=(RoutingSpec("trace", trace_id="captured"),))
    p = PR.plan(prof, impl_filter=("p_up",), traces=FakeTraces())
    assert p.missing_traces == () and p.ok


def test_pipeline_scope_cell_is_emitted_once_not_per_env():
    """Regression: the all-reference pipeline cell is framework-independent and
    is the slowest cell in the matrix. Emitting it per env ran it three times
    for identical rows."""
    from dataclasses import replace
    prof = replace(SMOKE, include_pipeline_scope=True)
    base = [c for c in PR.cells(prof, env="base") if c[2] == "__pipeline__"]
    vllm = [c for c in PR.cells(prof, env="vllm") if c[2] == "__pipeline__"]
    assert len(base) == len(prof.specs())
    assert vllm == []


REF_IMPLS = tuple(f"ref_{s}" for s in
                  __import__("moe.stages", fromlist=["x"]).CANONICAL_STAGES)


@pytest.mark.parametrize("name", sorted(PR.PROFILES))
def test_every_shipped_profile_plans_cleanly(name):
    """Scoped to the reference spans on purpose: the registry is global and
    other test modules register deliberately-broken doubles into it."""
    p = PR.plan(PR.get(name), impl_filter=REF_IMPLS, include_reference=True)
    assert p.problems == (), p.problems
    assert p.planned > 0


def test_profile_cell_isolates_one_launch_for_counter_profiling():
    """`ncu` profiles a launch, not a sweep. The two open questions each need
    exactly one cell, one routing, one timing mode, so that --launch-count 1
    lands on the kernel the question is about rather than whichever cell the
    matrix happened to order first."""
    p = PR.get("profile-cell")
    assert p.l2_modes == (True,), "cold only: two modes means two launches"
    assert p.graph_modes == (False,), "eager only: graph replay is a dead end"
    assert len(p.routings) == 1 and p.routings[0].kind == "uniform"
    assert p.models == ("deepseek-v3",)
    assert p.trials == 1
    plan = PR.plan(p, env="base")
    assert plan.specs == 1, "more than one spec means ncu cannot pick the cell"
    assert plan.modes == 1, "one timing mode, so --launch-count 1 is unambiguous"
