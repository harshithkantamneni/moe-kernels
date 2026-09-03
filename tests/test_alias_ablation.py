"""The alias ablation must be able to see alpha AND to miss its absence.

A pod script that always prints PASS is worth nothing, so most of this file runs
the whole report end to end against timings generated from six STATED laws and
checks that the verdict changes with the law:

  refit       alpha = 0.558 everywhere               -> every gate passes
  retracted   alpha = 0.10, this repo's old value    -> P1 fails, and the report
                                                        NAMES 0.10 as supported
  tempo       alpha = 0.33, TEMPO's value            -> P1 fails, 0.33 named
  folded      alpha = 0.558 but the aliased kernel
              issued half the global loads           -> the ISA gate fails and
                                                        the clean alpha is void
  l2-step     alpha steps on per-expert bytes        -> the pool median is not
                                                        the refit and P1 fails
  noise       a placebo as large as the signal       -> the placebo gate fails
  max-model   the kernel runs at max(L2, HBM), so the
              difference estimator is biased down       -> only the BRACKET
                                                           still contains 0.558
  l2-heavy    the same, with an aliased ladder costing
              45% of a weight read per tile             -> the bracket is too
                                                           wide and the run says
                                                           NOT TESTABLE
  alias-free  the aliased ladder's slope is ZERO       -> every gate passes and
                                                           alpha comes back
  alias-blind the aliased ladder's slope is the NORMAL
              one's: a shared ceiling at 0.61 of the
              read roof, which is what 2026-09-01
              actually measured                        -> headroom, attribution
                                                           and bracket all fail
                                                           and nothing is quoted

THE LAST TWO ARE THE PAIR THAT MATTERS. They are each other's opposite on the
single quantity the whole experiment turns on, the aliased ladder's slope, and a
gate set that cannot give them different verdicts cannot tell "the per-tile cost
IS DRAM" from "this apparatus could not see DRAM". The 2026-09-01 run reported
the second and was read as the first.

The rest pins the things that would let a wrong number look right: the estimator
staying free of the byte model it exists to bypass, the three runtime scalars
never taking the one value Triton would compile in as a constant, the ISA check
comparing within a rung rather than across models with different
specialisations, the control bounding the tile-count slope rather than the first
read, and a numpy bool not being able to hide a failed gate from the verdict.

`scripts/alias_ablation.py` is loaded by path because `scripts/` is not a
package, the same shape `tests/test_alpha_refit.py` and
`tests/test_group_m_sweep.py` need.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]` and fails inside the decorator otherwise.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AB = _load("alias_ablation", "alias_ablation.py")


@pytest.fixture(scope="module")
def design():
    """The design the pod will actually run, built from the shipped defaults."""
    return AB.build_design(AB.parse_args([]))


def _collect(lines: list[str]):
    """A `say` that records. `Report.__call__` takes an OPTIONAL line, so a bare
    `list.append` is not a substitute and fails on every blank line."""
    def say(line: str = "") -> None:
        lines.append(line)
    return say


def run_report(argv, tmp_path, monkeypatch, capsys):
    """`main` end to end, with output confined to a temp directory."""
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    code = AB.main(argv)
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------
# the gates can see an effect, and can miss its absence
# --------------------------------------------------------------------------

def test_the_refit_law_passes_every_gate_and_the_verdict_says_so(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "refit"], tmp_path, monkeypatch, capsys)
    assert code == 0, out
    assert "VERDICT: the ablation agrees with the refit" in out
    assert "[FAIL]" not in out
    assert "[NOT TESTABLE]" not in out


def test_the_retracted_alpha_fails_p1_and_is_named_as_what_the_band_supports(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "retracted"], tmp_path, monkeypatch,
                           capsys)
    assert code == 1
    assert "[FAIL] P1" in out
    assert "they are DISJOINT" in out
    # Naming which candidate the data supports is the deliverable, not a nicety:
    # "not 0.558" and "0.10" are different findings.
    assert "contains this repo, retracted" in out


def test_tempos_alpha_fails_p1_and_is_named_as_what_the_band_supports(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "tempo"], tmp_path, monkeypatch, capsys)
    assert code == 1
    assert "[FAIL] P1" in out
    assert "contains TEMPO arXiv:2608.13057" in out


def test_a_folded_aliased_kernel_fails_however_clean_the_alpha_looks(
        tmp_path, monkeypatch, capsys):
    """The whole hazard in one test.

    The `folded` law plants a PERFECT alpha of 0.558 and halves the aliased
    kernel's global-load count. P1 passes on that data and the run must still be
    refused, because a fold means the aliased time is the optimiser rather than
    the cache and the agreement is a coincidence of the generator.
    """
    code, out = run_report(["--synthetic", "folded"], tmp_path, monkeypatch, capsys)
    # INVALID (3), not CLAIM_FAIL (1). `ISA` is a VALIDITY gate: a folded kernel
    # means the apparatus measured the optimiser rather than the cache, so
    # nothing on the page may be quoted. 1 would say the world disagreed with a
    # prediction, which is a finding, and no finding was made here. `verdict`
    # chose its own codes until 2026-09-02 and returned 1 for any failed gate
    # whatever its kind; it is `exit_codes.classify` over the same gates now.
    assert code == 3
    assert "[FAIL] ISA" in out
    assert "[PASS] P1" in out
    assert "every number below is void" in out


def test_a_placebo_as_large_as_the_signal_fails_before_any_alpha_is_believed(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "noise"], tmp_path, monkeypatch, capsys)
    # INVALID (3), not CLAIM_FAIL (1): `placebo` is VALIDITY, and a placebo as
    # large as the signal voids the page rather than refuting a prediction.
    assert code == 3
    assert "[FAIL] placebo" in out


def test_an_l2_step_law_is_recovered_per_model_even_though_the_pool_hides_it(
        tmp_path, monkeypatch, capsys):
    """P2's mechanism, planted and recovered.

    The law gives alpha 0.95 above L2 and 0.05 below it. The per-model table has
    to show the step with brackets that do not overlap, and the POOLED median
    lands in the middle and is compatible with 0.558 -- which is exactly the
    reading under which 0.10, 0.33 and 0.558 could all be right about different
    pools of shapes, and exactly why the pooled number is not the finding.
    """
    _, out = run_report(["--synthetic", "l2-step"], tmp_path, monkeypatch, capsys)
    block = out.split("## P2")[1]
    above = [line for line in block.splitlines() if "above L2" in line]
    below = [line for line in block.splitlines() if "below L2" in line]
    # ONE ABOVE AND THREE BELOW, and that is the H200 and not a rounding. The
    # plant sat at 50 MiB until 2026-09-03, which is neither card's L2, and it
    # put deepseek-v3's 56 MiB expert on the wrong side of a line the real card
    # puts it below. At the measured 60 MiB only mixtral's 224 MiB expert is
    # above, so the rehearsal now rehearses the classification the pod will make.
    assert len(above) == 1 and len(below) == 3

    def low(line):
        return float(line.split("alpha")[1].split("to")[0])

    assert min(low(line) for line in above) > 0.8
    assert max(low(line) for line in below) < 0.2
    assert "POOLED" in out


def test_a_synthetic_report_can_never_be_read_as_a_measurement(
        tmp_path, monkeypatch, capsys):
    _, out = run_report(["--synthetic", "refit"], tmp_path, monkeypatch, capsys)
    assert "*** SYNTHETIC" in out
    assert "Nothing here was measured" in out


def test_replaying_synthetic_rows_still_announces_that_they_are_synthetic(
        tmp_path, monkeypatch, capsys):
    """`--replay` does not carry `--synthetic`, so provenance has to travel in
    the records or a synthetic report comes back looking like a pod result."""
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    AB.main(["--synthetic", "refit"])
    capsys.readouterr()
    out_dir = next((tmp_path / "alias_ablation").glob("*synthetic-refit"))
    code, out = run_report(["--replay", str(out_dir)], tmp_path, monkeypatch,
                           capsys)
    assert code == 0
    assert "*** SYNTHETIC" in out
    assert "(refit)" in out


# --------------------------------------------------------------------------
# the verdict cannot disagree with the table above it
# --------------------------------------------------------------------------

def test_a_numpy_false_is_a_failed_gate_and_not_a_passing_one():
    """`numpy.bool_(False) is False` is FALSE, so a gate built from a fitted
    number could print FAIL and then be invisible to the `ok is False` scan that
    decides the verdict. Live in `group_m_alpha_sweep.py` for one run."""
    gate = AB.Gate("planted", np.bool_(False), "")
    assert gate.ok is False
    assert gate.label == "FAIL"
    # INVALID (3): `planted` is not `P<digit>`, so `_kind_from_name` reads
    # VALIDITY, and a failed apparatus gate voids the page. The numpy bool is
    # still what this test is about: without `__post_init__`'s coercion the
    # `g.ok is False` scan inside `verdict` still misses the gate, and the
    # printed sentence would agree with the refit over a table with a FAIL in
    # it.
    assert AB.verdict(_collect([]), [gate]) == 3


def test_a_gate_with_no_evidence_is_not_a_refutation():
    """UNKNOWN counts against the gate and is still not CLAIM_FAIL.

    A VALIDITY gate that could not decide leaves the apparatus's soundness
    unshown, which is exactly as unquotable as a failure: INVALID (3). What it
    is NOT is 1, "a pre-registered claim was refuted". `verdict` used to answer
    4 here, and 4 is ERROR: "crashed; an exception nobody planned for", which
    the ledger reads as RETRY with a traceback to go and find.
    """
    lines: list[str] = []
    assert AB.verdict(_collect(lines), [AB.Gate("planted", None, "")]) == 3
    assert "NOT TESTABLE" in "\n".join(lines)


# --------------------------------------------------------------------------
# the estimator, which is the point of the whole exercise
# --------------------------------------------------------------------------

def test_the_estimator_recovers_a_planted_alpha_from_differences_alone():
    for planted in (0.10, 0.33, 0.558, 1.0):
        w = 0.42
        diffs = {n: w * (1.0 + planted * (n - 1)) for n in (1, 2, 4, 8)}
        fit = AB.fit_alpha(diffs)
        assert fit.ok
        assert fit.alpha == pytest.approx(planted, rel=1e-9)
        assert fit.w_ms == pytest.approx(w, rel=1e-9)


def test_alpha_is_slope_over_intercept_so_the_time_unit_cancels():
    """The claim that no bandwidth and no byte count enters the number.

    Scaling every measured time by a constant is exactly what changing the
    clock, the units, or the achieved bandwidth would do, and alpha must not
    move. This is the property `implied_traffic_ratio` has only because
    `alpha_refit` fits a group intercept; here it is arithmetic.
    """
    diffs = {n: 0.42 * (1.0 + 0.558 * (n - 1)) for n in (1, 2, 4, 8)}
    base = AB.fit_alpha(diffs).alpha
    for scale in (1e-3, 7.0, 1e4):
        scaled = AB.fit_alpha({n: d * scale for n, d in diffs.items()})
        assert scaled.alpha == pytest.approx(base, rel=1e-9)


def test_the_one_tile_rung_is_the_only_thing_that_supplies_w():
    """Drop n=1 and the fitted W is an extrapolation, not a measurement.

    The estimator still returns a number, which is why the preflight refuses a
    ladder that does not start at one tile rather than trusting the fit.
    """
    diffs = {n: 0.42 * (1.0 + 0.558 * (n - 1)) for n in (2, 4, 8)}
    assert AB.fit_alpha(diffs).w_ms == pytest.approx(0.42, rel=1e-9)
    args = AB.parse_args(["--tiles", "2,4,8"])
    gates = AB.preflight(AB.build_design(args), 0)
    named = [g for g in gates if "only source of W" in g.name]
    assert named and named[0].ok is False


def test_a_non_positive_w_refuses_to_produce_an_alpha():
    """The aliased variant not being faster means there is no weight read to be
    a fraction OF, and a ratio through a negative intercept is a large confident
    number with no meaning."""
    fit = AB.fit_alpha({1: -0.01, 2: 0.02, 4: 0.08, 8: 0.2})
    assert not fit.ok
    assert "not positive" in fit.why


def test_the_band_brackets_an_alpha_that_was_planted_in_the_data():
    rng = random.Random(0)
    w, planted = 0.42, 0.558
    samples = {}
    for n in (1, 2, 4, 8):
        normal = w * (1.0 + planted * (n - 1)) + 0.01 * n
        samples[n] = {
            "normal": [normal * (1 + rng.gauss(0, 0.003)) for _ in range(9)],
            "aliased": [0.01 * n * (1 + rng.gauss(0, 0.003)) for _ in range(9)],
        }
    band = AB.bootstrap_alpha(samples, draws=400, seed=1)
    assert band is not None
    assert band[0] <= planted <= band[1]
    # and the interval carries the model ambiguity as well as the noise, so it
    # is at least as wide as the bracket a single fit would have reported
    assert band[1] - band[0] >= AB.fit_bracket(samples).width


def test_the_bootstrap_resamples_replicates_and_not_rungs():
    """A rung is a designed level and is not a sample of anything.

    With one replicate per cell there is nothing to resample, so every draw is
    identical and the reported interval must collapse onto the deterministic
    bracket -- all model ambiguity, zero sampling width. A bootstrap that
    resampled RUNGS would produce a wide interval from the same data and would
    be describing the ladder rather than the measurement.
    """
    samples = {n: {"normal": [0.42 * (1 + 0.558 * (n - 1)) + 0.01 * n],
                   "aliased": [0.01 * n]} for n in (1, 2, 4, 8)}
    band = AB.bootstrap_alpha(samples, draws=200, seed=0)
    bracket = AB.fit_bracket(samples).bracket
    assert band is not None
    assert band == pytest.approx(bracket, abs=1e-12)


def test_the_supported_candidate_says_none_rather_than_the_nearest(
):
    """"Supports none of them" is the answer most easily left unsaid."""
    inside, sentence = AB.supported_candidate((0.70, 0.75))
    assert inside == []
    assert "contains NONE of the three" in sentence
    assert "interval-widths away" in sentence
    inside, sentence = AB.supported_candidate((0.52, 0.60))
    assert inside == ["today's refit"]


# --------------------------------------------------------------------------
# the route is independent of the byte model, which is the whole claim
# --------------------------------------------------------------------------

def test_this_script_never_reaches_the_byte_model_or_a_calibrated_bandwidth():
    """The ablation's value is that it shares no machinery with the refit.

    `alpha_refit` fits from `implied_traffic_ratio`, which is time x bandwidth
    over compulsory bytes, and C4 is a confirmed finding that the compulsory
    ruler was wrong. If this script ever imported `bytes_model`, `efficiency`,
    `roofline` or `alpha_refit`'s estimator, the two numbers would stop being
    independent and the second one would stop being worth measuring.
    """
    import ast
    tree = ast.parse((ROOT / "scripts" / "alias_ablation.py").read_text())
    # Over the AST and not over the text: the file DISCUSSES the byte-model
    # route at length in its docstring, which is the point of it, so a substring
    # scan would fail on its own explanation of why it exists.
    modules, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    banned_modules = {"moe.bench.bytes_model", "moe.bench.efficiency",
                      "moe.bench.roofline", "moe.bench.ridge",
                      "moe.bench.crossing"}
    assert not (modules & banned_modules), modules & banned_modules
    banned_names = {"implied_traffic_ratio", "compulsory_bytes",
                    "compulsory_gbps", "load_measured", "span_cost",
                    "fit_alpha_refit", "cell_key", "Observation"}
    assert not (names & banned_names), names & banned_names


def test_the_only_thing_borrowed_from_the_refit_is_the_rival_constants():
    """Borrowed so they cannot drift, and cross-checked so drift is announced."""
    AR = _load("alpha_refit_probe", "alpha_refit.py")
    assert AB.REPO_RETRACTED_ALPHA == AR.REPO_PUBLISHED_ALPHA
    assert AB.TEMPO_ALPHA == AR.TEMPO_ALPHA
    assert "agrees with alpha_refit.py" in AB.cross_check_candidates()
    # and it reads ONLY those two attributes off the estimator
    import ast
    source = (ROOT / "scripts" / "alias_ablation.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "cross_check_candidates")
    read = {n.args[1].value for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr"}
    assert read == {"REPO_PUBLISHED_ALPHA", "TEMPO_ALPHA"}


def test_the_estimator_needs_no_bandwidth_to_produce_a_number():
    """Stated as a test because the docstring's claim is checkable: the fit
    takes milliseconds and nothing else."""
    import inspect
    signature = inspect.signature(AB.fit_alpha)
    assert list(signature.parameters) == ["diffs"]


# --------------------------------------------------------------------------
# the ablation is really an ablation
# --------------------------------------------------------------------------

def test_no_ablation_scalar_is_ever_one_so_triton_cannot_split_the_kernel(design):
    """The trap that would have silently destroyed the experiment.

    Triton compiles an integer kernel argument of exactly 1 in as a constant. If
    any of the three ablation scalars were 1 in the normal variant, NORMAL and
    ALIASED would land in different specialisations, the normal one would fold
    its multiply at compile time, and "same compiled kernel" would be false
    while every table still printed numbers.
    """
    for rung in design.rungs:
        for extent in AB.ALIAS_EXTENTS:
            scalars = AB.ablation_scalars(rung, extent)
            for name, value in scalars["aliased"].items():
                assert value != 1, f"{rung.key} aliased {name} is 1"
                assert value == 0 or value % 16 == 0, (
                    f"{rung.key} aliased {name} = {value} takes neither the "
                    "zero nor the divisible-by-16 specialisation path")
        scalars = AB.ablation_scalars(rung)
        for name, value in scalars["normal"].items():
            assert value != 1, f"{rung.key} {name} is 1"
            assert value % 16 == 0, (
                f"{rung.key} {name} = {value} is not divisible by 16, so it "
                "takes a different specialisation path from the aliased 0")


def test_the_alias_extent_reaches_both_call_sites(design):
    """The extent has TWO call sites and this is the file's own recurring defect.

    `ablation_scalars` tells the kernel what to alias and `check_output` says
    what the aliased output must then equal. A change landing at one of the two
    is the shape of failure this rebuild has hit seven times: fixed at one of
    two call sites, walled at one of two ways in. Here the two directions fail
    differently and the second one is the dangerous direction -- a reference
    updated without the kernel AGREES with an alias that never happened, and the
    correctness gate passes on a run whose D is noise.

    The two are compared through the closed form's own arithmetic rather than by
    reading the source: at `block` the aliased arm walks a whole BLOCK_N x K
    column block and the reference sums `b[0, :BLOCK_N, :]`; at `tile` it
    re-reads one BLOCK_K x BLOCK_N tile K/BLOCK_K times and the reference sums
    that tile and multiplies. The bytes each extent touches are what the L2
    residency preflight is scored on, so all three agree or none do.
    """
    for rung in design.rungs:
        block = AB.ablation_scalars(rung, "block")["aliased"]
        tile = AB.ablation_scalars(rung, "tile")["aliased"]
        assert set(block) == {"stride_be_eff", "stride_bn_blk_eff",
                              "b_k_advance"}
        assert block["stride_be_eff"] == 0 and block["stride_bn_blk_eff"] == 0
        # `block` KEEPS the advance. It is the whole of the 2026-09-03 fix: the
        # aliased arm streams a column block instead of hammering 16 KiB.
        assert block["b_k_advance"] == rung.block_k
        assert all(v == 0 for v in tile.values())
        assert rung.alias_bytes("block") == rung.block_n * rung.k * 2
        assert rung.alias_bytes("tile") == rung.block_k * rung.block_n * 2
        assert rung.alias_bytes("block") > rung.alias_bytes("tile")
    with pytest.raises(ValueError):
        AB.ablation_scalars(design.rungs[0], "no-such-extent")
    with pytest.raises(ValueError):
        design.rungs[0].alias_bytes("no-such-extent")


def test_the_kernel_keeps_both_loads_live_on_the_compute_side():
    """STUDY.md item 4's own requirement, checked in the source.

    A load whose value is never used is the one thing a compiler is certain to
    remove, and it would remove it from BOTH variants equally -- which would not
    trip the ISA gate and would leave D(n) measuring nothing.
    """
    source = (ROOT / "scripts" / "alias_ablation.py").read_text()
    kernel = source.split("def _ablation_kernel")[1].split("return _ablation_kernel")[0]
    assert "a = tl.load(a_ptrs)" in kernel
    assert "b = tl.load(b_ptrs)" in kernel
    assert "acc += tl.dot(a, b)" in kernel
    assert "tl.sum(a.to(tl.float32)" in kernel and "tl.sum(b.to(tl.float32)" in kernel
    assert "tl.store(c_ptrs, acc)" in kernel


def test_the_isa_gate_compares_within_a_rung_and_never_across_models():
    """Two models legitimately compile to different kernels.

    Triton specialises on each rung's own K, N and strides, so pooling every
    reading and comparing digests would fail on every multi-model run for a
    reason that has nothing to do with folding, and a gate that fails for the
    wrong reason gets disabled.
    """
    def reading(variant, digest, loads):
        return {"variant": variant, "source": "test", "digest": digest,
                "counts": {"ld.global": 0, "cp.async": loads,
                           "global_loads": loads}}
    records = [
        {"id": "mixtral|t1", "isa": [reading("normal", "aaa", 64),
                                     reading("aliased", "aaa", 64)]},
        {"id": "deepseek|t1", "isa": [reading("normal", "zzz", 112),
                                      reading("aliased", "zzz", 112)]},
    ]
    assert AB.isa_gate(records).ok is True


def test_the_isa_gate_fails_when_the_aliased_launch_issued_fewer_loads():
    def reading(variant, digest, loads):
        return {"variant": variant, "source": "test", "digest": digest,
                "counts": {"ld.global": 0, "cp.async": loads,
                           "global_loads": loads}}
    records = [{"id": "m|t1", "isa": [reading("normal", "aaa", 64),
                                      reading("aliased", "bbb", 1)]}]
    gate = AB.isa_gate(records)
    assert gate.ok is False
    assert "every number below is void" in gate.detail


def test_the_isa_gate_notices_equal_counts_from_different_code():
    """Equal counts from two compilations is weaker than one compiled kernel,
    and the usual cause is Triton's equal-to-1 argument specialisation."""
    def reading(variant, digest):
        return {"variant": variant, "source": "test", "digest": digest,
                "counts": {"ld.global": 0, "cp.async": 64, "global_loads": 64}}
    gate = AB.isa_gate([{"id": "m|t1", "isa": [reading("normal", "aaa"),
                                               reading("aliased", "bbb")]}])
    assert gate.ok is False
    assert "equal-to-1" in gate.detail


def test_an_unreachable_ptx_is_not_testable_rather_than_a_pass():
    """"Counts equal" and "counts unavailable" are the two states this check
    exists to keep apart."""
    gate = AB.isa_gate([{"id": "m|t1", "isa": []}])
    assert gate.ok is None
    assert "should be quoted until it does" in gate.detail


def test_counting_ld_global_alone_would_have_read_zero_on_a_pipelined_kernel():
    """Triton emits `cp.async.cg.shared.global`, not `ld.global`, at
    num_stages > 1. A fold check on the wrong mnemonic passes silently."""
    ptx = "\n".join(["cp.async.cg.shared.global [%r1], [%rd2], 16;"] * 64
                    + ["st.global.v4.b32 [%rd9], {%f1,%f2,%f3,%f4};"])
    counts = AB.count_ops(ptx)
    assert counts["ld.global"] == 0
    assert counts["cp.async"] == 64
    assert counts["global_loads"] == 64


# --------------------------------------------------------------------------
# the design is the design the question needs
# --------------------------------------------------------------------------

def test_rows_per_expert_is_an_exact_multiple_of_block_m_at_every_rung(design):
    """Padding is EXACTLY zero, so the tile count is the only thing that moves.

    The published rows cannot do this: uniform routing is SAMPLED per replicate,
    so the tile count varies within a cell, which is the correction FINDINGS.md
    had to make to its staircase table.
    """
    for rung in design.rungs:
        assert rung.rows_per_expert % rung.block_m == 0
        assert rung.rows_per_expert // rung.block_m == rung.tiles
        assert rung.total_rows == rung.experts * rung.tiles * rung.block_m


def test_the_activation_confound_is_bounded_at_every_shipped_rung(design):
    """The L2-capacity confound's bound is n*BLOCK_M/N, and it has to be small
    or the aliased variant's freed L2 is worth as much as the traffic it saved."""
    for rung in design.rungs:
        assert rung.activation_fraction == pytest.approx(
            rung.tiles * rung.block_m / rung.n, rel=1e-9)
        assert rung.activation_fraction <= AB.MAX_ACTIVATION_FRACTION


def test_a_control_too_narrow_to_bound_anything_is_refused_before_it_is_paid_for(
        monkeypatch):
    """The first control geometry shipped here was 8 x 1024 x 1024 and streamed
    12.5% as many activation bytes as weight bytes at the top rung. The
    preflight caught it, which is the gate doing its job on its own author."""
    monkeypatch.setattr(AB, "CONTROL_N", 1024)
    monkeypatch.setattr(AB, "CONTROL_K", 1024)
    monkeypatch.setattr(AB, "CONTROL_EXPERTS", 8)
    gates = AB.preflight(AB.build_design(AB.parse_args([])), 0)
    named = [g for g in gates if "activation stream" in g.name]
    assert named and named[0].ok is False


def test_the_control_expert_fits_in_every_l2_this_study_has_run_on():
    """40 MiB is the A100's and 60 the H200's, both read off their calibrations.

    The control's whole point is that its re-read HITS, so it has to fit in the
    SMALLER of the two. This docstring said "the H200's is 50" until 2026-09-03,
    which is the same remembered number `SYNTHETIC_L2_BYTES` carried.
    """
    smaller = AB.measured_card("NVIDIA A100-SXM4-80GB")["l2_bytes"]
    larger = AB.measured_card("NVIDIA H200")["l2_bytes"]
    assert smaller == 40 * 2 ** 20 and larger == 60 * 2 ** 20
    rung = AB.rung_for(AB.CONTROL_MODEL, 8, AB.DEFAULT_BLOCK_M, AB.FIXED_TILE)
    assert rung.per_expert_bytes < smaller
    # and the whole tensor must NOT fit, or there is no first-pass W to divide by
    assert rung.weight_bytes > larger


def test_the_shipped_models_straddle_l2_so_the_mechanism_is_testable(design):
    sizes = sorted(r.per_expert_bytes for r in design.rungs if not r.control)
    for card in ("NVIDIA H200", "NVIDIA A100-SXM4-80GB"):
        l2 = AB.measured_card(card)["l2_bytes"]
        assert sizes[0] < l2 < sizes[-1], (card, sizes)


def test_every_shipped_rung_stays_below_the_ridge_in_dot_mode():
    """`dot` mode is the only mode with arithmetic, so it is the only one where
    a rung can be compute bound. A compute-bound rung pays for extra tiles in
    padded arithmetic rather than traffic and would report a flat alpha."""
    design = AB.build_design(AB.parse_args(["--compute", "dot"]))
    for rung in design.rungs:
        if not rung.control:
            assert rung.arith_intensity < 160.3, rung.key
    assert all(g.ok is not False for g in AB.preflight(design, 0))


def test_the_shipped_defaults_pass_every_preflight_gate(design):
    gates = AB.preflight(design, AB.measured_card("NVIDIA H200")["l2_bytes"])
    assert all(g.ok is not False for g in gates), [
        (g.name, g.detail) for g in gates if g.ok is False]


def test_dot_mode_reports_a_lower_bound_and_refuses_to_answer_p1(
        tmp_path, monkeypatch, capsys):
    """With a real matmul the aliased variant becomes compute bound while the
    normal one stays memory bound, so D(n) loses one copy of the per-tile
    compute cost and the fitted alpha is biased DOWN. A biased number must not
    be allowed to answer the prediction."""
    code, out = run_report(["--compute", "dot", "--synthetic", "refit"],
                           tmp_path, monkeypatch, capsys)
    # CLAIM_FAIL (1): P1 is the one CLAIM here and an UNKNOWN CLAIM means the
    # claim was NOT ESTABLISHED, which is the same side of the gate as a
    # failure. Not 4 = ERROR, which the ledger reads as a crash to retry.
    assert code == 1
    assert "[NOT TESTABLE] P1" in out
    assert "biased LOW" in out


# --------------------------------------------------------------------------
# the control bounds the right quantity
# --------------------------------------------------------------------------

def _result(model, alpha, w, control=False, direct=None):
    fit = AB.AlphaFit(alpha=alpha, w_ms=w, r2=1.0,
                      alpha_direct=alpha if direct is None else direct)
    return AB.ModelResult(model=model, fit=fit, band=fit.bracket,
                          per_expert_mib=16.0, control=control)


def test_the_max_composition_biases_the_difference_estimator_and_the_bracket_saves_it(
        tmp_path, monkeypatch, capsys):
    """The hazard the bracket exists for, planted and survived.

    Under max(L2, HBM) the difference estimator returns (alpha - r)/(1 - r),
    which for a true 0.558 lands near 0.50 at r = 0.12 and would land near 0.02
    at the r an H200 plausibly has. The DIRECT estimator is exact there, and the
    interval between them still has to contain the truth.
    """
    code, out = run_report(["--synthetic", "max-model"], tmp_path, monkeypatch,
                           capsys)
    assert code == 0, out
    row = next(line for line in out.splitlines()
               if line.strip().startswith("deepseek-v3") and "to" in line
               and "MiB/expert" not in line)
    low, high = (float(v) for v in row.split()[1:4:2])
    assert low < AB.REFIT_ALPHA < high
    assert "[PASS] P1" in out


def test_an_aliased_ladder_too_expensive_to_resolve_says_so_and_picks_nothing(
        tmp_path, monkeypatch, capsys):
    """A wide interval refutes nothing. It says the run did not resolve, and
    the difference matters: this project has a standing habit of reading a wide
    number as a finding."""
    code, out = run_report(["--synthetic", "l2-heavy"], tmp_path, monkeypatch,
                           capsys)
    # INVALID (3): `resolution` is VALIDITY, and an interval too wide to resolve
    # leaves the instrument's soundness unshown, which is as unquotable as a
    # failure. Not 4 = ERROR: nothing crashed.
    assert code == 3
    assert "[NOT TESTABLE] resolution" in out
    assert "cheaper aliased ladder" in out


def test_the_two_estimators_bracket_a_planted_alpha_under_both_compositions():
    """The arithmetic the whole bracket rests on, checked at both extremes.

    Under addition D(n) is exactly W(1+alpha(n-1)) and the difference estimator
    is exact while the direct one reads (alpha+r)/(1+r); under max the roles
    swap and the difference estimator reads (alpha-r)/(1-r). Either way the
    interval contains the truth for alpha <= 1.
    """
    w, alpha, r, fixed = 0.40, 0.558, 0.30, 0.01
    for composition in ("add", "max"):
        samples = {}
        for n in (1, 2, 4, 8):
            l2 = r * w * n
            hbm = w * (1.0 + alpha * (n - 1))
            normal = fixed + (l2 + hbm if composition == "add"
                              else max(l2, hbm))
            samples[n] = {"normal": [normal], "aliased": [fixed + l2]}
        fit = AB.fit_bracket(samples)
        assert fit.ok
        assert fit.bracket[0] <= alpha <= fit.bracket[1], (composition, fit)
        exact = fit.alpha if composition == "add" else fit.alpha_direct
        assert exact == pytest.approx(alpha, rel=1e-6), composition
        assert fit.fixed_ms == pytest.approx(fixed, abs=1e-9)


def test_the_fixed_cost_comes_from_the_aliased_ladder_and_is_not_free(
):
    """Taking `fixed` from the aliased ladder's own n=0 intercept is what keeps
    the direct estimator from being "a fitted intercept", which is the thing
    this whole experiment exists to avoid."""
    import ast
    source = (ROOT / "scripts" / "alias_ablation.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "fit_bracket")
    body = ast.unparse(fn)
    assert "line_a[0] - line_a[1]" in body
    assert "line_n[0] - fixed" in body


def test_the_control_gate_bounds_the_tile_slope_and_not_the_first_read():
    """An earlier version compared the control's W against the real models' and
    would have failed a correct run: the control still pays a full first pass
    over its own tensor, so its W is legitimately the same order as theirs. What
    it does not have is an HBM re-read, so the emptiness shows up in the SLOPE.
    """
    big_w = [_result("real", 0.558, 0.42),
             _result(AB.CONTROL_MODEL, -0.13, 0.40, control=True, direct=0.02)]
    assert AB.control_gate(big_w).ok is True
    leaky = [_result("real", 0.558, 0.42),
             _result(AB.CONTROL_MODEL, 0.40, 0.01, control=True, direct=0.45)]
    gate = AB.control_gate(leaky)
    assert gate.ok is False
    assert "not weight traffic" in gate.detail


def test_a_run_without_a_control_says_the_confound_is_unbounded(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "refit", "--no-control"], tmp_path,
                           monkeypatch, capsys)
    # INVALID (3): `control` is VALIDITY and without one the confound is
    # unbounded, so the page cannot be quoted. Not 4 = ERROR.
    assert code == 3
    assert "[NOT TESTABLE] control" in out
    assert "NOT BOUNDED. TLB, page behaviour and code path" in out


def test_the_placebo_gate_ignores_the_control_whose_d_is_zero_by_design():
    """Observed at 542% on the first synthetic pass of this gate: the control's
    D is near zero, so drift/D is a ratio of two noise floors and reads as a
    catastrophic failure on a perfectly clean run."""
    records = [
        {"id": "real|t1", "control": False,
         "ms": {"normal": [1.0], "aliased": [0.1], "placebo": [1.001]}},
        {"id": "control|t1", "control": True,
         "ms": {"normal": [0.1], "aliased": [0.1], "placebo": [0.1001]}},
    ]
    assert AB.placebo_gate(records).ok is True


# --------------------------------------------------------------------------
# running, resuming and replaying
# --------------------------------------------------------------------------

def test_the_kernel_is_built_lazily_and_a_missing_triton_names_the_venv():
    """Everything here runs off-GPU and fails with a message, not a traceback.

    The kernel is built inside a function for exactly this reason: a module-level
    `@triton.jit` would make the whole file unimportable on a laptop, and the
    plan, the prediction and the estimator's self-test all have to run there.
    """
    import ast
    tree = ast.parse((ROOT / "scripts" / "alias_ablation.py").read_text())
    top_level_imports = {a.name for n in tree.body if isinstance(n, ast.Import)
                         for a in n.names}
    assert "triton" not in top_level_imports and "torch" not in top_level_imports
    if importlib.util.find_spec("triton") is None:
        with pytest.raises(AB.CannotRunHere) as excinfo:
            AB.build_kernel()
        assert "vllm venv" in str(excinfo.value)


def test_a_run_without_a_gpu_says_so_and_is_refused_not_invalid(
        tmp_path, monkeypatch, capsys):
    # The no-GPU condition is FORCED rather than inherited from the host. This
    # asserted only that `--run` exits 3, which is true on a laptop because
    # triton is absent and false on the pod because it is not: the base venv
    # there carries triton 3.7.1, so the kernel builds, the run proceeds, and
    # the exit code is a gate verdict instead. A test whose premise is "this
    # machine has no GPU" silently stops testing anything on the only machine
    # the code actually runs on.
    def _no_triton(*a, **kw):
        raise AB.CannotRunHere(
            "triton is not importable in this interpreter. Run inside the vllm "
            "venv on the pod: /workspace/venvs/vllm/bin/python "
            "scripts/alias_ablation.py --run")
    monkeypatch.setattr(AB, "build_kernel", _no_triton)
    code, out = run_report(["--run"], tmp_path, monkeypatch, capsys)
    # REFUSED (2), not INVALID (3). It returned 3 until 2026-09-02, and 3 tells
    # the driver "measured, then a VALIDITY gate failed: there is a directory of
    # cells that must not be scored, and do NOT retry this arm". Nothing was
    # measured, there is no directory, and the arm is free to retry on a box
    # that has a GPU.
    assert code == exit_codes.REFUSED
    assert exit_codes.parse_result_lines(out) == []
    assert "CANNOT RUN HERE" in out
    assert "The plan, the prediction and the preflight above are still valid" in out


def test_a_run_killed_mid_write_still_replays(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    AB.main(["--synthetic", "refit"])
    capsys.readouterr()
    out_dir = next((tmp_path / "alias_ablation").glob("*synthetic-refit"))
    cells = out_dir / "cells.jsonl"
    cells.write_text(cells.read_text() + '{"kind": "rung", "id": "trunc')
    code, out = run_report(["--replay", str(out_dir)], tmp_path, monkeypatch,
                           capsys)
    assert code == 0
    assert "20 rungs read from disk" in out


def test_records_from_another_design_are_ignored_and_named(tmp_path, monkeypatch,
                                                           capsys):
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    AB.main(["--synthetic", "refit"])
    capsys.readouterr()
    out_dir = next((tmp_path / "alias_ablation").glob("*synthetic-refit"))
    cells = out_dir / "cells.jsonl"
    stray = {"kind": "rung", "id": "not-a-model|t99|bm16", "model": "ghost",
             "tiles": 99, "ms": {"normal": [1.0], "aliased": [0.1]}}
    cells.write_text(cells.read_text() + json.dumps(stray) + "\n")
    code, out = run_report(["--replay", str(out_dir)], tmp_path, monkeypatch,
                           capsys)
    assert code == 0
    assert "1 records name a rung this design does not contain" in out
    assert "not-a-model|t99|bm16" in out


def test_a_resumed_run_skips_rungs_already_on_disk(design, tmp_path):
    """A rung is the resume unit, so a Ctrl-C costs one rung and not the sweep."""
    path = tmp_path / "cells.jsonl"
    AB._append(path, {"kind": "rung", "id": design.rungs[0].key, "ms": {}})
    done = {r["id"] for r in AB.read_records(path)}
    assert design.rungs[0].key in done
    assert design.rungs[1].key not in done


def test_the_fingerprint_changes_when_the_control_geometry_changes(monkeypatch):
    """Resume keys on `model|tiles|block_m`, which does not mention the shape,
    so a changed control geometry would silently reuse records measured on a
    different tensor. The fingerprint is what sends a changed design elsewhere.
    """
    before = AB.build_design(AB.parse_args([])).fingerprint
    monkeypatch.setattr(AB, "CONTROL_N", 8192)
    after = AB.build_design(AB.parse_args([])).fingerprint
    assert before != after


def test_the_output_directory_prefers_the_env_then_the_volume_then_the_repo(
        tmp_path, monkeypatch):
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    assert AB.results_root() == tmp_path
    monkeypatch.delenv("MOE_RESULTS_DIR")
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    assert AB.results_root() == tmp_path / "results"
    monkeypatch.setenv("WORKSPACE", str(tmp_path / "absent"))
    assert AB.results_root() == ROOT / "results"


def test_an_unknown_model_is_refused_by_name(capsys):
    with pytest.raises(SystemExit):
        AB.parse_args(["--models", "gpt-9"])
    assert "unknown model" in capsys.readouterr().err


def test_a_synthetic_replay_ignores_the_attached_card(tmp_path, monkeypatch,
                                                      capsys):
    """The planted law must be recovered identically on every machine.

    This is the regression for a defect that survived because the laptop's
    no-GPU fallback and the planted threshold were the same 50 MiB. On an H200
    (60 MiB L2) the classification moved under the report and one model crossed
    sides, so the "recovered the planted law" assertion failed on the only
    hardware the experiment is for. `l2_bytes_here` is made to answer like a
    real card here; the synthetic output must not move.

    The plant is now the H200's own 60 MiB, read out of its calibration file, so
    the two failure modes are separated: the value cannot be a card nobody owns,
    AND it cannot follow whatever card is attached.
    """
    monkeypatch.setattr(AB, "l2_bytes_here", lambda: 60 * 2 ** 20)
    _, on_card = run_report(["--synthetic", "l2-step"], tmp_path, monkeypatch,
                            capsys)
    monkeypatch.setattr(AB, "l2_bytes_here", lambda: 0)
    _, on_laptop = run_report(["--synthetic", "l2-step"], tmp_path, monkeypatch,
                              capsys)

    def sides(out):
        block = out.split("## P2")[1]
        return [ln for ln in block.splitlines() if "L2   alpha" in ln]

    assert sides(on_card) == sides(on_laptop), (
        "the synthetic classification moved with the attached card")
    assert "L2 on the PLANTED threshold, not the attached card: 60.0 MiB" \
        in on_card
    assert len([x for x in sides(on_card) if "above L2" in x]) == 1
    assert len([x for x in sides(on_card) if "below L2" in x]) == 3


# --------------------------------------------------------------------------
# The acceptance check for the exit-code repair: the log and the process say
# the same thing in every mode this file can reach without a GPU.
# --------------------------------------------------------------------------

#: Every off-GPU mode of this script and the code it must return. `no_gpu` marks
#: the row that forces `build_kernel` to refuse, because on a pod triton IS
#: importable and a test whose premise is "this machine has no GPU" stops
#: testing anything on the only machine the code runs on.
OFF_GPU_MODES = [
    # The bare invocation: a plan, a prediction, a preflight, an MDE, and no
    # timings. It returned 0 until a previous slice made it REFUSED.
    ([], exit_codes.REFUSED, False),
    (["--synthetic", "refit"], exit_codes.DONE, False),
    # P1 is the one CLAIM. TEMPO's alpha lands outside the refit band, which is
    # a statement about the world and therefore a RESULT.
    (["--synthetic", "tempo"], exit_codes.CLAIM_FAIL, False),
    # ISA and placebo are VALIDITY: the apparatus was unsound, so the page is
    # void. Both returned 1 -- "a claim was refuted" -- until 2026-09-02.
    (["--synthetic", "folded"], exit_codes.INVALID, False),
    (["--synthetic", "noise"], exit_codes.INVALID, False),
    # `resolution` and `control` read UNKNOWN. UNKNOWN counts against the gate,
    # and on a VALIDITY gate that is INVALID, not the 4 = ERROR this used to
    # answer.
    (["--synthetic", "l2-heavy"], exit_codes.INVALID, False),
    # THE PAIR. In one world the aliased ladder's slope is zero and every gate
    # passes; in the other it is the normal ladder's and three VALIDITY gates
    # fail. Same code, same ladder, opposite verdicts.
    (["--synthetic", "alias-free"], exit_codes.DONE, False),
    (["--synthetic", "alias-blind"], exit_codes.INVALID, False),
    # A step in per-expert bytes puts the pool median on one side of the step
    # and the bracket across it, so `resolution` reads UNKNOWN: a pooled alpha
    # over a step function is not resolvable, which is P2's own point.
    (["--synthetic", "l2-step"], exit_codes.INVALID, False),
    (["--synthetic", "refit", "--no-control"], exit_codes.INVALID, False),
    # An UNKNOWN CLAIM: the biased estimator may not answer P1, so the claim was
    # not established, which is CLAIM_FAIL and not ERROR.
    (["--compute", "dot", "--synthetic", "refit"], exit_codes.CLAIM_FAIL, False),
    (["--run"], exit_codes.REFUSED, True),
]


@pytest.mark.parametrize("argv,code,no_gpu", OFF_GPU_MODES,
                         ids=[" ".join(a) or "bare" for a, _, _ in OFF_GPU_MODES])
def test_the_log_and_the_exit_code_agree_in_every_off_gpu_mode(
        argv, code, no_gpu, tmp_path, monkeypatch, capsys):
    """The whole repair, stated as one property instead of as prose.

    For every mode this file can reach on a laptop, the RESULT lines it printed
    and the integer it returned have to be the same verdict. `classify_text`
    recomputes the code from the log; a log with NO RESULT lines raises
    `NoGatesScored`, and `moe.bench.exit_codes` documents that as exactly what a
    REFUSED log looks like from there, so the two cases are one rule: score
    gates and match `classify`, or score none and return REFUSED.

    Four of these rows used to break it, all through `verdict`, which printed
    one `exit_codes.result_line` per gate and then returned codes of its own
    devising: 1 for any failed gate whatever its kind, 4 for any undecided one.
    `folded` and `noise` fail a VALIDITY gate and were reported as refuted
    claims; `l2-heavy` and `--no-control` leave one UNKNOWN and were reported as
    crashes. `--run` returned 3 = INVALID, which tells the driver a directory of
    unquotable cells exists, from a run that never opened one.
    """
    if no_gpu:
        def _no_triton(*a, **kw):
            raise AB.CannotRunHere(
                "triton is not importable in this interpreter. Run inside the "
                "vllm venv on the pod.")
        monkeypatch.setattr(AB, "build_kernel", _no_triton)
    rc, out = run_report(argv, tmp_path, monkeypatch, capsys)
    assert rc == code, out
    lines = exit_codes.parse_result_lines(out)
    if lines:
        assert exit_codes.classify_text(out) == rc
    else:
        assert rc == exit_codes.REFUSED, (
            "a run that scored no gate printed no RESULT line, so its log "
            "implies REFUSED and nothing else")


# --------------------------------------------------------------------------
# THE VOID OF 2026-09-01, and why it was the apparatus and not the world
# --------------------------------------------------------------------------

#: The medians `results/published/2026-09-01-nvidia_h200-alpha-0558/session/
#: alias_ablation.md` printed, exactly as printed, and the geometries they were
#: measured on. This is the only run this arm has ever had and its numbers are
#: the regression: a design that cannot fail on THESE cannot fail on anything.
VOID_RUN = {
    "mixtral-8x7b": (8, 4096, 28672,
                     {1: (0.7802, 0.7216), 2: (1.5109, 1.4256),
                      4: (2.9488, 2.7855), 8: (5.8056, 5.5032)}),
    "qwen2-57b-a14b": (64, 3584, 5120,
                       {1: (0.9611, 0.9011), 2: (1.8267, 1.7395),
                        4: (3.5848, 3.4587), 8: (7.1366, 6.8955)}),
    "deepseek-v2-lite": (64, 2048, 2816,
                         {1: (0.3461, 0.2933), 2: (0.5992, 0.5695),
                          4: (1.1419, 1.1000), 8: (2.2507, 2.1852)}),
    "deepseek-v3": (256, 7168, 4096,
                    {1: (5.8852, 5.5723), 2: (11.4874, 11.0285),
                     4: (22.7387, 21.9501), 8: (45.3440, 43.7844)}),
}

#: `moe/bench/hardware/measured_nvidia_h200.yaml`, `detail.bandwidth_patterns`,
#: the `read` entry. The file's own note calls it "closest analogue to streaming
#: expert weights".
H200_READ_ROOF = 4469.60368208941e9


def void_records(scale: float = 1.0):
    """The published run as `cells.jsonl` rows, with the aliased arm scalable.

    `scale` multiplies every ALIASED time. At 1.0 these are the numbers that
    were measured. Below 1.0 the aliased ladder gets cheaper, which is what a
    kernel with headroom looks like, and that is how the PASS branch of the two
    new gates is planted: the same rows, the same code path, one factor.
    """
    rows = []
    for model, (experts, k, n, table) in VOID_RUN.items():
        for tiles, (normal, aliased) in table.items():
            rows.append({
                "kind": "rung", "id": f"{model}|t{tiles}|bm16", "model": model,
                "tiles": tiles, "experts": experts, "k": k, "n": n,
                "weight_bytes": experts * n * k * 2,
                "per_expert_bytes": n * k * 2, "control": False,
                "ms": {"normal": [normal], "aliased": [aliased * scale],
                       "placebo": [normal]},
            })
    return rows


def test_the_void_run_fails_headroom_with_its_own_published_numbers():
    """The 2026-09-01 aliased ladder ran at 0.61 of the card's read roof.

    That single number is the whole diagnosis. Both arms issue the same loads
    and share the cost of getting them from L2 into the SM; only the normal arm
    also pays DRAM. A shared path at 0.61 of the DRAM roof is the SLOWER of the
    two, so DRAM had 39% slack in the normal arm, and ablating a resource with
    slack moves the clock by the un-overlapped residue and by nothing else. The
    residue is the 5 to 8% the run reported and voided itself over.

    The ratio is checked to two decimals on the worst model rather than merely
    asserted to be below the limit, because "below a threshold" would also be
    satisfied by a design that was broken in some other way.
    """
    gate = AB.headroom_gate(void_records(), H200_READ_ROOF)
    assert gate.ok is False
    assert gate.kind == exit_codes.VALIDITY
    ratios = []
    for model in VOID_RUN:
        rows = [r for r in void_records() if r["model"] == model]
        ratios.append(AB._aliased_slope_bytes_s(rows) / H200_READ_ROOF)
    assert 0.60 < min(ratios) <= max(ratios) < 0.62, ratios
    assert "0.61" in gate.detail or "0.60" in gate.detail


def test_the_void_runs_d1_is_below_the_floor_for_the_bytes_it_names():
    """D(1) came out three to eleven times FASTER than the card can read.

    D(1) is the denominator of every alpha this experiment produces and its
    claim is that it is the time cost of one full pass over every expert's
    weight block. Those bytes have a floor and the floor is the card's own
    measured read ceiling. Beating it is not a small effect or a noisy one; it
    is impossible, and it means the label on D is wrong. `signal` reported the
    same fact as "5.3% of a pass", which reads as a statement about alpha.
    """
    gate = AB.attribution_gate(void_records(), H200_READ_ROOF)
    assert gate.ok is False
    worst = []
    for experts, k, n, table in VOID_RUN.values():
        normal, aliased = table[1]
        floor_ms = experts * n * k * 2 / H200_READ_ROOF * 1e3
        worst.append((normal - aliased) / floor_ms)
    # 1/10.7 through 1/3.1, the ratios in the module docstring's table.
    assert 0.09 < min(worst) < 0.10, worst
    assert 0.31 < max(worst) < 0.33, worst
    # THE GATE IS ON THE WORST MODEL AND THAT IS THE POINT. deepseek-v2-lite,
    # the smallest tensor, reaches 0.32 of its floor and would pass on its own;
    # the pooled alpha is fitted across all four, so one model clearing a
    # threshold cannot license a page the other three cannot support.
    assert min(worst) < AB.MIN_ATTRIBUTION_RATIO <= max(worst)
    assert AB.MIN_ATTRIBUTION_RATIO == AB.MIN_SIGNAL_FRACTION
    assert AB.MIN_HEADROOM_RATIO == 1.0 / (1.0 - AB.MIN_SIGNAL_FRACTION)


def test_the_same_two_gates_pass_once_the_shared_path_is_faster_than_dram():
    """The FAIL branch above is only a gate if this branch exists.

    One factor on the aliased arm, nothing else changed. At a fifth of its
    measured cost the aliased ladder delivers about three times the read roof,
    DRAM becomes the binding resource in the normal arm, and D(1) reaches its
    floor. Both gates flip, which is what makes the pair a discriminator rather
    than a threshold that happens to sit above the one run there has been.
    """
    rows = void_records(scale=0.2)
    assert AB.headroom_gate(rows, H200_READ_ROOF).ok is True
    assert AB.attribution_gate(rows, H200_READ_ROOF).ok is True


def test_a_card_with_no_calibration_is_unknown_and_therefore_invalid():
    """No roof, no verdict. REFUSE rather than default.

    Both gates are VALIDITY, so UNKNOWN classifies as INVALID and nothing on the
    page may be quoted. Substituting a datasheet pin rate would move every
    headroom and attribution ratio by the gap between a pin rate and an achieved
    ceiling, which on this card is 4916.7 against 4469.6 GB/s, and would do it
    silently.
    """
    for gate in (AB.headroom_gate(void_records(), None),
                 AB.attribution_gate(void_records(), None)):
        assert gate.ok is None
        assert gate.kind == exit_codes.VALIDITY
        assert "calibrat" in gate.detail
    assert exit_codes.classify([("VALIDITY", "UNKNOWN")]) == exit_codes.INVALID


def test_the_bracket_gate_refuses_an_interval_that_is_not_a_bracket():
    """`(alpha - r)/(1 - r)` is below alpha only while r < 1.

    The 2026-09-01 report printed brackets built on an r of 6.4 to 19.1 and one
    of them, 0.535 to 0.969 pooled, was read as containing the refit. At r above
    1 the difference estimator's denominator is negative, both ends of the
    printed interval sit on the same side of the truth, and the interval
    contains nothing in particular while still looking like error bars.
    """
    def result(r):
        fit = AB.AlphaFit(alpha=0.5, w_ms=0.1, alpha_direct=0.6,
                          aliased_slope=r * 0.1)
        return AB.ModelResult(model="m", fit=fit, band=None,
                              per_expert_mib=10.0, control=False)

    assert AB.bracket_gate([result(12.35)]).ok is False
    assert AB.bracket_gate([result(0.45)]).ok is True
    assert AB.bracket_gate([]).ok is None
    assert "12.35" in AB.bracket_gate([result(12.35)]).detail
    # The CONTROL is excluded: its alpha is zero by construction, so its r is a
    # ratio of two noise floors and would fail a clean run.
    control = AB.ModelResult(model="c", fit=result(12.35).fit, band=None,
                             per_expert_mib=16.0, control=True)
    assert AB.bracket_gate([control]).ok is None


# --------------------------------------------------------------------------
# the two planted worlds the whole redesign turns on
# --------------------------------------------------------------------------

def test_the_two_alias_worlds_get_opposite_verdicts(tmp_path, monkeypatch,
                                                    capsys):
    """Aliased slope zero against aliased slope unchanged.

    These are the hypothesis and its negation stated as timings: in one the
    ablation removes a cost and in the other it removes nothing because the cost
    was never on the critical path. A gate set that gives them the same verdict
    is not testing anything, and the gate set that ran on 2026-09-01 gave the
    second one a page of alphas with error bars.
    """
    free_code, free_out = run_report(["--synthetic", "alias-free"], tmp_path,
                                     monkeypatch, capsys)
    blind_code, blind_out = run_report(["--synthetic", "alias-blind"], tmp_path,
                                       monkeypatch, capsys)
    assert free_code == exit_codes.DONE
    assert blind_code == exit_codes.INVALID

    def verdicts(out):
        return {line.name: line.verdict
                for line in exit_codes.parse_result_lines(out)}

    free, blind = verdicts(free_out), verdicts(blind_out)
    named = [name for name in free
             if name.startswith(("headroom", "attribution", "bracket"))]
    assert len(named) == 3, named
    for name in named:
        assert free[name] == exit_codes.PASS, name
        assert blind[name] == exit_codes.FAIL, name
    # The blind world is the one that happened: a shared ceiling at 0.61 of the
    # read roof, which is what five geometries measured to three digits.
    assert "0.61" in blind_out


def test_the_blind_world_reproduces_the_ratio_that_was_measured(tmp_path,
                                                                monkeypatch,
                                                                capsys):
    """`alias-blind` is the 2026-09-01 run and not a caricature of it.

    A planted failure that fails by an order of magnitude proves a gate can
    print FAIL. A planted failure that reproduces the measured ratio proves the
    gate would have caught the run that actually happened, which is the only
    claim worth making about it.
    """
    _, out = run_report(["--synthetic", "alias-blind"], tmp_path, monkeypatch,
                        capsys)
    line = next(ln for ln in out.splitlines()
                if ln.startswith("RESULT: VALIDITY headroom"))
    ratio = float(line.split("a ratio of ")[1].split()[0])
    assert 0.60 <= ratio <= 0.62, line
    signal = next(ln for ln in out.splitlines()
                  if ln.startswith("RESULT: VALIDITY signal"))
    # 5.3 to 8% of a pass is what the published report recorded for D(1).
    percent = float(signal.split("D(1) at ")[1].split("%")[0])
    assert 4.0 <= percent <= 12.0, signal


# --------------------------------------------------------------------------
# the probe: three minutes spent so sixty are not
# --------------------------------------------------------------------------

def _reading(warps, stages, block_k, compute, gbps):
    return {"pinning": {"num_warps": warps, "num_stages": stages,
                        "block_k": block_k, "compute": compute},
            "aliased_bytes_s": gbps * 1e9 if gbps else None, "note": ""}


def test_the_probe_prefers_the_unbiased_pinning_over_a_faster_biased_one():
    """Headroom is the objective; speed is only how headroom is reached.

    Once a pinning has headroom the remaining question is which estimator is
    less biased, and `dot` is the biased one -- `prediction_gate` will not
    answer P1 from it at all. A dot pinning that is 40% faster and cannot answer
    the question loses to a sum pinning that can.
    """
    readings = [_reading(8, 3, 64, "sum", 2700),      # the shipped one: fails
                _reading(8, 4, 128, "sum", 6500),
                _reading(16, 4, 128, "dot", 9000)]
    chosen, why = AB.choose_pinning(readings, H200_READ_ROOF)
    assert chosen["compute"] == "sum" and chosen["num_stages"] == 4
    assert "sum-mode" in why


def test_the_probe_falls_to_dot_only_when_no_sum_pinning_clears():
    readings = [_reading(8, 3, 64, "sum", 2700),
                _reading(16, 4, 128, "dot", 9000)]
    chosen, why = AB.choose_pinning(readings, H200_READ_ROOF)
    assert chosen["compute"] == "dot"
    assert "LOWER BOUND" in why


def test_the_probe_refuses_when_nothing_clears_and_names_the_best_it_saw():
    """A refusal that does not name the best reading is unactionable.

    The next session needs to know whether the ceiling was missed by 5% or by a
    factor of two, because one of those is a pinning problem and the other is
    the end of the ablation route.
    """
    readings = [_reading(8, 3, 64, "sum", 2700), _reading(8, 4, 128, "dot", 2800)]
    chosen, why = AB.choose_pinning(readings, H200_READ_ROOF)
    assert chosen is None
    assert "0.626" in why or "0.627" in why, why
    assert "2800 GB/s" in why
    assert "no ladder run" in why.lower() or "No ladder run" in why


def test_the_probe_refuses_without_a_roof_rather_than_picking_the_fastest():
    chosen, why = AB.choose_pinning([_reading(8, 3, 64, "sum", 9000)], None)
    assert chosen is None and "calibration" in why


def test_the_probe_gate_is_the_one_result_line_a_stopped_arm_prints():
    """A probe that stops the arm SPENT card time, so the log has to say 3.

    `exit_codes.classify_text` recomputes the exit code from the RESULT lines a
    log carries. A stopped probe printing none would leave a log that looks
    REFUSED, which is the code for "free, nothing measured", and the driver
    would book two or three minutes of a rented card as zero.
    """
    gate = AB.probe_gate([_reading(8, 3, 64, "sum", 2700)], None, "nothing cleared")
    assert gate.kind == exit_codes.VALIDITY and gate.ok is False
    assert exit_codes.classify([gate.scored()]) == exit_codes.INVALID
    assert exit_codes.classify_text(gate.result_line()) == exit_codes.INVALID
    ok = AB.probe_gate([_reading(8, 4, 128, "sum", 9000)],
                       {"num_warps": 8}, "cleared")
    assert exit_codes.classify([ok.scored()]) == exit_codes.DONE


def test_the_probe_measures_the_cheapest_geometry_because_a_rate_is_a_rate(design):
    """0.607 to 0.616 across a 20x spread in footprint: the ceiling is a rate.

    So the probe pays for the smallest allocation it can and leaves the rest of
    the hour on the ladder.
    """
    assert AB.probe_model(design) == "deepseek-v2-lite"
    smallest = min(r.weight_bytes for r in design.rungs if not r.control)
    assert smallest == min(r.weight_bytes for r in design.rungs
                           if r.model == "deepseek-v2-lite")


# --------------------------------------------------------------------------
# the ceilings come out of the calibration files, never out of a memory
# --------------------------------------------------------------------------

def test_the_planted_ceilings_are_the_cards_own_and_not_a_remembered_number():
    """`SYNTHETIC_L2_BYTES` was 50 MiB, which is neither card's L2.

    The H200 has 60 (`observed.l2_bytes: 62914560`) and the A100 has 40
    (`41943040`). A rehearsal planted at 50 straddles a cache nobody owns, and
    it put deepseek-v3's 56 MiB expert above a line the real card puts it below.
    """
    h200 = AB.measured_card("NVIDIA H200")
    a100 = AB.measured_card("NVIDIA A100-SXM4-80GB")
    assert h200["l2_bytes"] == 62914560 == 60 * 2 ** 20
    assert a100["l2_bytes"] == 41943040 == 40 * 2 ** 20
    assert AB.SYNTHETIC_L2_BYTES == h200["l2_bytes"]
    assert AB.SYNTHETIC_ROOF_BYTES_S == h200["roof_bytes_s"]
    assert abs(h200["roof_bytes_s"] - H200_READ_ROOF) < 1.0
    assert AB.SYNTHETIC_L2_BYTES != 50 * 2 ** 20
    # The control has to fit inside the SMALLER of the two, or it controls for
    # nothing on the A100.
    control = AB.rung_for(AB.CONTROL_MODEL, 1, 16, AB.FIXED_TILE)
    assert control.per_expert_bytes < a100["l2_bytes"]


def test_an_uncalibrated_card_returns_nothing_rather_than_a_datasheet_peak():
    assert AB.measured_card("NVIDIA GeForce RTX 4090") == {}


def test_the_aliased_arm_must_be_resident_and_the_preflight_can_say_no():
    """The gate has a FAIL branch and here it is.

    At BLOCK_N = 128 the aliased arm walks 1.75 MiB at most, which is a tenth of
    the quarter-L2 limit. At BLOCK_N = 8192 on deepseek-v3 it walks 112 MiB,
    misses, and stops being an ablation of the weight read: it becomes a second
    copy of it, and D would be noise around zero.
    """
    args = AB.parse_args(["--models", "deepseek-v3", "--block-n", "8192"])
    design = AB.build_design(args)
    gates = AB.preflight(design, 60 * 2 ** 20)
    resident = next(g for g in gates if "L2-resident" in g.name)
    assert resident.ok is False and resident.kind == exit_codes.VALIDITY
    assert "112.00 MiB" in resident.detail
    shipped = AB.build_design(AB.parse_args([]))
    ok = next(g for g in AB.preflight(shipped, 60 * 2 ** 20)
              if "L2-resident" in g.name)
    assert ok.ok is True


# --------------------------------------------------------------------------
# paired in time, priced in the driver's units, and crash-safe
# --------------------------------------------------------------------------

def test_the_ladder_is_walked_tile_major_so_the_control_is_paired_in_time(design):
    """Every model at one tile count before any model's next.

    Model-major put the L2-resident control's whole ladder at the END of the
    run, up to forty minutes and one thermal state away from the ladders it is
    supposed to bound, while `control_gate` compared them anyway. It also meant
    an interrupted hour left some models with no ladder at all; tile-major
    leaves every model's ladder complete up to some tile count, which still fits
    a line. The allocation count is identical either way.
    """
    order = AB.measurement_order(design)
    assert len(order) == len(design.rungs)
    assert {r.key for r in order} == {r.key for r in design.rungs}
    tiles = [r.tiles for r in order]
    assert tiles == sorted(tiles)
    first_block = order[:len(design.models)]
    assert {r.model for r in first_block} == set(design.models)
    assert all(r.tiles == min(design.tiles) for r in first_block)


def test_the_cost_is_priced_with_the_iteration_floor_and_named_wall_or_kernel():
    """`time_kernel` has a ten-iteration floor and the big rungs live on it.

    deepseek-v3 at eight tiles is tens of milliseconds a call, so ten calls is a
    four hundred millisecond trial and not the fifty the budget asks for.
    Pricing without the floor under-counts the top of the ladder by an order of
    magnitude, which is how an arm gets booked at eleven minutes and spends
    thirty-six.
    """
    args = AB.parse_args([])
    design = AB.build_design(args)
    with_floor = AB.estimated_kernel_ms(design, args, H200_READ_ROOF)
    budgeted = (len(design.rungs) * 3 * design.replicates
                * (args.warmup + args.cell_budget_ms * args.trials))
    assert with_floor > 1.2 * budgeted, (with_floor, budgeted)

    # The floor is not an aggregate effect. Priced alone, the dearest rung is
    # deepseek-v3 at eight tiles: 120 GB of weight requests a call, 44 ms each,
    # so the 50 ms budget asks for TWO calls a trial and gets ten. That rung
    # alone is a third of the total above.
    top = max(design.rungs, key=lambda r: r.weight_bytes * r.tiles)
    per_call = (top.weight_bytes * top.tiles
                / (H200_READ_ROOF * AB.KERNEL_EFFICIENCY_PRIOR) * 1e3)
    wanted = math.ceil(args.cell_budget_ms / per_call)
    assert wanted < 10, (per_call, wanted)
    floored = args.warmup + 10 * per_call * args.trials
    asked = args.warmup + wanted * per_call * args.trials
    assert floored > 2.5 * asked, (floored, asked)
    assert AB.estimated_kernel_ms(design, args, None) is None
    lines = []
    AB.report_cost(_collect(lines), design, args, H200_READ_ROOF, probing=True)
    text = "\n".join(lines)
    assert "KERNEL" in text and "WALL" in text
    assert "THIS IS NOT A WALL FIGURE" in text
    assert "EXCLUDES" in text
    unpriced = []
    AB.report_cost(_collect(unpriced), design, args, None, probing=False)
    assert "NOT PRICED" in "\n".join(unpriced)


def test_a_crash_exits_error_and_not_the_claim_fail_the_ledger_latches(
        monkeypatch, capsys):
    """Python spends 1 on an escaping exception and the ledger LATCHES 1.

    `scripts/h200_gaps_session.sh` reads 1 as CLAIM_FAIL, marks the arm
    finished, skips it on every resume, and prints its silence in the closing
    summary as a refuted prediction. A crash is the apparatus failing.
    """
    def boom(*a, **kw):
        raise RuntimeError("planted: the apparatus fell over")

    monkeypatch.setattr(AB, "main", boom)
    assert AB._guarded([]) == exit_codes.ERROR
    assert "planted" in capsys.readouterr().err

    def refuse(*a, **kw):
        raise SystemExit("no such model")

    monkeypatch.setattr(AB, "main", refuse)
    assert AB._guarded([]) == exit_codes.REFUSED
    monkeypatch.setattr(AB, "main", lambda argv=None: exit_codes.CLAIM_FAIL)
    assert AB._guarded([]) == exit_codes.CLAIM_FAIL


def test_a_replay_is_scored_against_the_ruler_the_run_was_measured_with(
        tmp_path, monkeypatch, capsys):
    """`--replay` carries no card, and both new gates divide by one.

    Without the plan's own roof a finished, passing run comes back INVALID
    purely because it was re-reported on a laptop, which is the shape of a false
    retraction. The synthetic path writes a plan for exactly this reason.
    """
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    AB.main(["--synthetic", "refit"])
    capsys.readouterr()
    out_dir = next((tmp_path / "alias_ablation").glob("*synthetic-refit"))
    plan = json.loads((out_dir / "plan.json").read_text())
    assert plan["roof_bytes_s"] == AB.SYNTHETIC_ROOF_BYTES_S
    assert plan["l2_bytes"] == AB.SYNTHETIC_L2_BYTES
    assert plan["alias_extent"] == AB.DEFAULT_ALIAS_EXTENT
    code, out = run_report(["--replay", str(out_dir)], tmp_path, monkeypatch,
                           capsys)
    assert code == exit_codes.DONE
    assert "plan.json" in out
    verdicts = {line.name: line.verdict
                for line in exit_codes.parse_result_lines(out)}
    for name, verdict in verdicts.items():
        if name.startswith(("headroom", "attribution")):
            assert verdict == exit_codes.PASS, name
    (out_dir / "plan.json").unlink()
    code, _ = run_report(["--replay", str(out_dir)], tmp_path, monkeypatch,
                         capsys)
    # No plan, no ruler, no verdict: UNKNOWN on a VALIDITY gate is INVALID.
    assert code == exit_codes.INVALID


def test_the_extent_changes_the_run_id_so_two_ladders_cannot_share_a_directory():
    """`block` and `tile` remove different bytes and produce different r.

    The resume key is the rung, which carries no extent, so a second design
    landing in the first one's directory would find every rung present, spend no
    GPU time, and report the other extent's timings under its own label. The
    card, the seed, the flush and the three timing knobs are in the id for the
    same reason.
    """
    ids = set()
    for extent in AB.ALIAS_EXTENTS:
        args = AB.parse_args(["--alias-extent", extent])
        design = AB.build_design(args)
        ids.add(AB.default_run_id(args, "NVIDIA H200", design))
    assert len(ids) == len(AB.ALIAS_EXTENTS)
    pins = set()
    for warps in (4, 8, 16):
        args = AB.parse_args(["--num-warps", str(warps)])
        pins.add(AB.default_run_id(args, "NVIDIA H200", AB.build_design(args)))
    assert len(pins) == 3


def test_the_headroom_and_signal_limits_are_one_number_and_cannot_disagree():
    """Two thresholds over one physical quantity is this repo's defect as arithmetic.

    When DRAM binds in the normal arm and the shared path binds in the aliased
    one, the normal arm's time IS the DRAM time and the aliased arm's is that
    over `h`, so `D(1)/T_normal(1) = 1 - 1/h` exactly. Pick the headroom limit
    independently of `MIN_SIGNAL_FRACTION` and there is a band of `h` in which a
    run clears headroom and is then voided by signal, with nothing on the page
    saying the two limits were inconsistent rather than the kernel wrong. At
    1.15, which is what this file shipped for an afternoon, that band runs from
    1.15 to 1.333 and every run inside it reads as a failed experiment.
    """
    h = AB.MIN_HEADROOM_RATIO
    assert abs((1.0 - 1.0 / h) - AB.MIN_SIGNAL_FRACTION) < 1e-12
    for ratio in (1.0, 1.2, 1.3):
        assert 1.0 - 1.0 / ratio < AB.MIN_SIGNAL_FRACTION, ratio
        assert ratio < h, ratio
    for ratio in (1.4, 2.0, 8.0):
        assert 1.0 - 1.0 / ratio > AB.MIN_SIGNAL_FRACTION, ratio
        assert ratio > h, ratio


def test_out_refuses_to_resume_into_another_designs_directory(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """The run id is one way in and `--out` is the other.

    Every knob that moves a row is in `default_run_id`, so the default path
    cannot resume into another design's directory. `--out` names one outright
    and walks past all of it, and `scripts/pod_session.sh` passes `--out`. The
    resume key is the rung, which carries neither the pinning nor the alias
    extent, so the second design would find every rung present, spend nothing,
    and report the first one's timings under its own label. The probe makes that
    live rather than hypothetical: it can adopt a different pinning on two runs
    of one command.
    """
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    out = tmp_path / "shared"
    AB.main(["--synthetic", "refit", "--out", str(out)])
    capsys.readouterr()
    plan = json.loads((out / "plan.json").read_text())
    code, text = run_report(["--run", "--no-probe", "--out", str(out),
                             "--alias-extent", "tile"], tmp_path, monkeypatch,
                            capsys)
    assert code == exit_codes.REFUSED
    assert plan["fingerprint"] in text
    assert "REFUSED" in text
    assert exit_codes.parse_result_lines(text) == []


# --------------------------------------------------------------------------
# the MDE, at both of its call sites
# --------------------------------------------------------------------------


def test_an_mde_refuses_rather_than_defaults_on_impossible_inputs():
    """Four inputs, four refusals, and the last one is the new call site's.

    `signal_share` divides D, so a zero there is a run whose ablation removed
    nothing, and the answer is not an infinite resolution but no resolution.
    Defaulting it to `MIN_SIGNAL_FRACTION` when the caller passes zero would
    have printed the plan's number over a run that measured the opposite.
    """
    with pytest.raises(ValueError, match="at least one replicate"):
        AB.mde_of_alpha(0.008, 0, 8)
    with pytest.raises(ValueError, match="positive spread"):
        AB.mde_of_alpha(0.0, 9, 8)
    with pytest.raises(ValueError, match="rung above the first"):
        AB.mde_of_alpha(0.008, 9, 1)
    with pytest.raises(ValueError, match="positive signal share"):
        AB.mde_of_alpha(0.008, 9, 8, signal_share=0.0)


def test_the_mde_moves_the_way_its_derivation_says():
    """Each of the four steps is checked by the direction it moves the answer.

    A formula nobody differentiates is a formula nobody has read. Replicates
    enter as `sqrt(2/replicates)`, so four times as many halve it; the spread
    and the signal share enter linearly and oppositely; the top rung enters as
    `(1 + alpha(n-1))/(n-1)`, which falls, and is why the ladder has a top rung
    at all.
    """
    base = AB.mde_of_alpha(0.0077, 9, 8)
    assert AB.mde_of_alpha(0.0077, 36, 8) == pytest.approx(base / 2.0, rel=1e-9)
    assert AB.mde_of_alpha(0.0154, 9, 8) == pytest.approx(base * 2.0, rel=1e-9)
    assert AB.mde_of_alpha(
        0.0077, 9, 8, signal_share=AB.MIN_SIGNAL_FRACTION / 2.0
    ) == pytest.approx(base * 2.0, rel=1e-9)
    assert AB.mde_of_alpha(0.0077, 9, 2) > base
    # The arithmetic itself, once, against a hand computation rather than
    # against the function under test.
    z = 1.959963984540054 + 0.8416212335729143
    want = (z * 0.0077 * math.sqrt(2.0 / 9) / AB.MIN_SIGNAL_FRACTION
            * math.sqrt(2.0) * (1.0 + AB.REFIT_ALPHA * 7) / 7)
    assert base == pytest.approx(want, rel=1e-9)


def test_the_mde_and_the_signal_gate_read_one_share_function():
    """The recurring defect of this rebuild, in the one place it would bite here.

    `signal_gate` scores D(1)'s share of a pass and `report_mde` divides by it.
    Computed twice they drift, and the report would then gate on one number
    while pricing its own resolution on another. `one_tile_signal_shares` is
    the single function; this pins that the gate's printed percentage is the
    same one the measured MDE block prints.
    """
    rows = void_records()
    shares = AB.one_tile_signal_shares(rows)
    worst, rid = shares[0]
    assert worst == pytest.approx(0.05317, abs=5e-5)
    assert rid.startswith("deepseek-v3")
    # sorted worst-first, and the control never contributes.
    assert shares == sorted(shares)
    gate = AB.signal_gate(rows)
    assert gate.ok is False
    assert f"{worst * 100:.1f}%" in gate.detail

    lines: list[str] = []
    AB.report_mde(_collect(lines), AB.build_design(AB.parse_args([])),
                  (("median", 0.0077),), measured=True, signal_share=worst)
    assert f"D(1) is {worst:.1%} of the weakest one-tile pass" in "\n".join(lines)


def test_the_void_run_could_not_have_resolved_the_candidates():
    """The number the 2026-09-01 report never printed, and the one that reads it.

    That run voided itself on `signal` and `form` and was read as a null result
    about DRAM. Put its own worst signal share through the MDE and the reading
    changes: at D(1) = 5.3% of a pass the design could resolve alpha only to
    0.19, against a 0.11 limit and a 0.46 gap between the rival candidates. It
    was not a null. It was an apparatus that could not have separated 0.10 from
    0.558 whatever the answer had been, and no line in the report said so
    because the MDE was stated once, before the rental, against the floor
    `signal_gate` would have enforced.
    """
    worst, _ = AB.one_tile_signal_shares(void_records())[0]
    as_run = AB.mde_of_alpha(AB.MEASURED_SPREAD_MEDIAN, 9, 8, signal_share=worst)
    as_planned = AB.mde_of_alpha(AB.MEASURED_SPREAD_MEDIAN, 9, 8)
    assert as_planned < AB.MAX_BAND_WIDTH < as_run
    assert as_run == pytest.approx(0.1896, abs=5e-4)
    assert as_run / as_planned == pytest.approx(AB.MIN_SIGNAL_FRACTION / worst,
                                                rel=1e-9)


def test_the_cannot_resolve_branch_is_reachable_by_either_lever():
    """Two levers reach the FAIL wording, and they are different diagnoses.

    `tests/test_p7_instrument_and_gates.py` already plants it with a spread the
    corpus does not contain. The second lever is the one this file adds and the
    one 2026-09-01 actually pulled: the same corpus spread with a signal share
    the run measured. Both must reach it, because a wide spread is a noisy box
    and wants more replicates, while a thin signal is an apparatus that did not
    ablate anything and wants a different kernel, and an arm that can only
    express the first will report the second as noise.
    """
    design = AB.build_design(AB.parse_args([]))
    ok: list[str] = []
    AB.report_mde(_collect(ok), design)
    assert "CANNOT resolve them" not in "\n".join(ok)

    noisy: list[str] = []
    AB.report_mde(_collect(noisy), design, (("planted", 0.05),))
    assert "CANNOT resolve them" in "\n".join(noisy)

    thin: list[str] = []
    AB.report_mde(_collect(thin), design, (("median", AB.MEASURED_SPREAD_MEDIAN),),
                  measured=True, signal_share=0.053)
    assert "CANNOT resolve them" in "\n".join(thin)


def test_a_run_whose_ablation_removed_nothing_quotes_no_resolution():
    """Zero share and no spread are both NOT STATED, not a divide.

    `report_mde` is called on whatever the run produced, including a run that
    timed one pass per cell or whose aliased arm was no cheaper at all. Neither
    is an error and neither is a number: the block says which, and the gates
    below say why.
    """
    design = AB.build_design(AB.parse_args([]))
    empty: list[str] = []
    AB.report_mde(_collect(empty), design, (), measured=True, signal_share=0.5)
    assert "NOT STATED" in "\n".join(empty)
    assert "Raise --replicates" in "\n".join(empty)

    flat: list[str] = []
    AB.report_mde(_collect(flat), design, (("median", 0.0077),), measured=True,
                  signal_share=0.0)
    text = "\n".join(flat)
    assert "NOT STATED" in text
    assert "removed nothing measurable" in text


def test_observed_spreads_recover_a_planted_scatter_and_refuse_one_pass():
    """The run's own noise, by the corpus's own reduction, so the two compare.

    `MEASURED_SPREAD_*` are `timing_spread_median` over the published reports,
    which `rescore_published_reports` defines as a within-cell pstdev over p50.
    This is the same reduction over the interleaved passes, which is the
    quantity the MDE derivation actually needs and is never the smaller of the
    two. The published void table carries one number per cell, so it has no
    spread at all, and that is `()` rather than zero.
    """
    rng = random.Random(11)
    rows = [{"id": f"m|t{t}|bm16", "ms": {
        "normal": [10.0 * (1.0 + rng.gauss(0.0, 0.02)) for _ in range(400)],
        "aliased": [5.0 * (1.0 + rng.gauss(0.0, 0.02)) for _ in range(400)]}}
        for t in (1, 2)]
    spreads = dict(AB.observed_spreads(rows))
    assert spreads["median"] == pytest.approx(0.02, abs=0.004)
    assert spreads["worst"] >= spreads["median"]

    assert AB.observed_spreads(void_records()) == ()
    assert AB.observed_spreads([]) == ()
    assert AB.observed_spreads([{"id": "x", "skipped": True, "ms": {
        "normal": [1.0, 1.0, 1.0]}}]) == ()
    # A cell whose median is zero has no relative spread and is dropped rather
    # than dividing.
    assert AB.observed_spreads([{"id": "x", "ms": {"normal": [0.0, 0.0]}}]) == ()


def test_the_mde_is_stated_twice_and_the_second_is_the_runs_own(
        tmp_path, monkeypatch, capsys):
    """The plan sizes the booking; the report says whether the gates are readable.

    Stating only the first is what let 2026-09-01 print a design that resolves
    0.04 above a run that could resolve 0.19. Both blocks must appear, the
    second must carry the run's own spread and its own signal share rather than
    the two corpus constants, and it must sit above the gates it qualifies.
    """
    code, text = run_report(["--synthetic", "refit"], tmp_path, monkeypatch,
                            capsys)
    assert code == exit_codes.DONE
    assert text.count("MDE on alpha") == 4
    plan_at = text.index("## what this design can see (MDE)")
    run_at = text.index("## what this run could see (MDE, on its own spread)")
    assert plan_at < run_at < text.index("## gates")
    plan_block = text[plan_at:run_at]
    run_block = text[run_at:text.index("## gates")]
    assert f"median {AB.MEASURED_SPREAD_MEDIAN:.2%}" in plan_block
    assert "FLOOR" in plan_block
    # The synthetic law draws at 0.4%, well under the corpus's 0.77% floor, so
    # the second block cannot be the first one repeated.
    assert f"spread {AB.MEASURED_SPREAD_MEDIAN:.2%}" not in run_block
    assert "spread 0.3" in run_block or "spread 0.4" in run_block
    assert "Signal MEASURED IN THIS RUN" in run_block
    share = AB.one_tile_signal_shares(
        AB.synthesise(AB.build_design(AB.parse_args([])), "refit", 0))[0][0]
    assert f"D(1) is {share:.1%}" in run_block


def test_the_remedy_named_is_the_lever_that_actually_moved():
    """Nine more hours of replicates do not fix a signal that was never there.

    The two levers into the MDE want opposite next steps and the report has to
    say which. Above the signal floor the scatter is what binds and
    `--replicates` buys resolution as sqrt(n). Below it the ablation removed too
    little to resolve, no replicate count fixes that, and the lever is the alias
    and the pinning. 2026-09-01 was the second case; a block that had said
    "raise --replicates" over its 5.3% would have sent the next rental to buy
    the same answer more precisely.
    """
    design = AB.build_design(AB.parse_args([]))
    thin: list[str] = []
    AB.report_mde(_collect(thin), design, (("median", 0.0077),), measured=True,
                  signal_share=0.053)
    thin_text = "\n".join(thin)
    assert "replicates are NOT the remedy" in thin_text
    assert "the lever is the alias and the pinning" in thin_text

    thick: list[str] = []
    AB.report_mde(_collect(thick), design, (("median", 0.0077),), measured=True,
                  signal_share=0.80)
    thick_text = "\n".join(thick)
    assert "the lever is --replicates" in thick_text
    assert "NOT the remedy" not in thick_text
    # The split is on the gate's own floor and not on a second threshold.
    edge: list[str] = []
    AB.report_mde(_collect(edge), design, (("median", 0.0077),), measured=True,
                  signal_share=AB.MIN_SIGNAL_FRACTION)
    assert "the lever is --replicates" in "\n".join(edge)
