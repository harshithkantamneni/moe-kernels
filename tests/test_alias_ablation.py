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
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    assert code == 1
    assert "[FAIL] ISA" in out
    assert "[PASS] P1" in out
    assert "every number below is void" in out


def test_a_placebo_as_large_as_the_signal_fails_before_any_alpha_is_believed(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "noise"], tmp_path, monkeypatch, capsys)
    assert code == 1
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
    assert len(above) == 2 and len(below) == 2

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
    assert AB.verdict(_collect([]), [gate]) == 1


def test_a_gate_with_no_evidence_is_not_a_refutation():
    lines: list[str] = []
    assert AB.verdict(_collect(lines), [AB.Gate("planted", None, "")]) == 4
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
        scalars = AB.ablation_scalars(rung)
        assert set(scalars["aliased"].values()) == {0}
        for name, value in scalars["normal"].items():
            assert value != 1, f"{rung.key} {name} is 1"
            assert value % 16 == 0, (
                f"{rung.key} {name} = {value} is not divisible by 16, so it "
                "takes a different specialisation path from the aliased 0")


def test_the_aliased_variant_zeroes_every_one_of_the_three_scalars(design):
    """Zeroing only two of the three would leave a live axis of the B address
    and the aliased variant would still stream from HBM."""
    for rung in design.rungs:
        aliased = AB.ablation_scalars(rung)["aliased"]
        assert set(aliased) == {"stride_be_eff", "stride_bn_blk_eff",
                                "b_k_advance"}
        assert all(v == 0 for v in aliased.values())


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
    """40 MiB is the A100's; the H200's is 50. The control's whole point is that
    its re-read HITS, so it has to fit in the smaller of the two."""
    rung = AB.rung_for(AB.CONTROL_MODEL, 8, AB.DEFAULT_BLOCK_M, AB.FIXED_TILE)
    assert rung.per_expert_bytes < 40 * 2 ** 20
    # and the whole tensor must NOT fit, or there is no first-pass W to divide by
    assert rung.weight_bytes > 50 * 2 ** 20


def test_the_shipped_models_straddle_l2_so_the_mechanism_is_testable(design):
    sizes = sorted(r.per_expert_bytes for r in design.rungs if not r.control)
    assert sizes[0] < 50 * 2 ** 20 < sizes[-1]


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
    gates = AB.preflight(design, 50 * 2 ** 20)
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
    assert code == 4
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
    assert code == 4
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
    assert code == 4
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


def test_a_run_without_a_gpu_says_so_and_exits_three(tmp_path, monkeypatch,
                                                     capsys):
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
    assert code == 3
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
    no-GPU fallback and the planted threshold are the same 50 MiB. On an H200
    (60 MiB L2) the classification moved under the report and one model crossed
    sides, so the "recovered the planted law" assertion failed on the only
    hardware the experiment is for. `l2_bytes_here` is made to answer like a
    real card here; the synthetic output must not move.
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
    assert "L2 on the PLANTED threshold, not the attached card: 50.0 MiB" \
        in on_card
    assert len([x for x in sides(on_card) if "above L2" in x]) == 2
    assert len([x for x in sides(on_card) if "below L2" in x]) == 2
