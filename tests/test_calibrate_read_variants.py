"""The C4 read-ceiling probe, and the repository-wide check that found it.

`scripts/calibrate_read_variants.py` carried the LAST private timing loop in the
tree: events created inside the loop, a start recorded on a stream the previous
iteration's synchronise had just drained, a synchronise every iteration, five
isolated warmup calls, no clock. It survived the A7 migration because no test
had ever loaded the file and no test globbed `scripts/` for the SHAPE -- the
acceptance check listed six file names, and this was not one of them. So the
first test here is the glob, over every file in `scripts/` and `moe/`, and it is
the one that must never be narrowed to a list again.

The rest is the C4 verdict itself. The gates are scored off a flattened
`Reading` rather than a `KernelTiming` precisely so every branch is reachable on
a laptop: a gate whose FAIL branch has never been executed is a claim about the
code, not a check on the world.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from moe.bench import exit_codes as EX  # noqa: E402
from moe.bench import timing  # noqa: E402


def _load(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CRV = _load("calibrate_read_variants")


# --------------------------------------------------------------------------
# A7: the shape, over the whole tree, not over a list of names
# --------------------------------------------------------------------------

def _private_timing_loops() -> list[str]:
    """Every `for`/`while` in the repo whose body mentions CUDA event timing.

    The same walk that found this file: `Event(` or `elapsed_time(` INSIDE a
    loop is the A7 shape, because `time_kernel` primes one event pair per
    iteration BEFORE the loop and reads them after it. A false positive here is
    a file that should be reviewed anyway.
    """
    hits = []
    for path in sorted(REPO.glob("scripts/*.py")) + sorted(REPO.glob("moe/**/*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.For | ast.While):
                src = ast.unparse(node)
                if "Event(" in src or "elapsed_time(" in src:
                    hits.append(f"{path.relative_to(REPO)}:{node.lineno}")
    return hits


def test_no_file_in_the_repository_times_inside_its_own_loop():
    """The acceptance check the A7 migration did not have.

    `tests/test_p7_instrument_and_gates.py` asserts that six NAMED files define
    no `time_call`; a seventh file spelling the same loop inline was invisible
    to it. This is the shape, over every file, so the next one cannot hide
    behind a name.
    """
    assert _private_timing_loops() == [], (
        "a private CUDA-event timing loop is back; use timing.time_kernel, "
        "which is queue-deep, flushes L2, samples the clock under load and "
        "returns the three verdicts")


def test_the_read_probe_measures_through_the_one_instrument():
    """Positively, not just by the absence of the old shape.

    Over the CALLS rather than the text: the docstring names the retired loop
    on purpose, and a check that reads prose would either fail on the
    explanation or force the explanation out of the file.
    """
    path = REPO / "scripts" / "calibrate_read_variants.py"
    called = {ast.unparse(node.func)
              for node in ast.walk(ast.parse(path.read_text()))
              if isinstance(node, ast.Call)}
    assert "timing.time_kernel" in called
    assert not [c for c in called if "Event" in c or "elapsed_time" in c]
    assert "def time_it(" not in path.read_text(), "the private loop is retired"


# --------------------------------------------------------------------------
# the gates, both branches of each
# --------------------------------------------------------------------------

H200 = "NVIDIA H200"


def _gates(readings, gpu_name=H200, registered=CRV.REGISTERED_READ_GBPS):
    return {name: (verdict, detail)
            for _kind, name, verdict, detail in CRV.score(readings, gpu_name,
                                                          registered)}


def test_a_formulation_that_buys_the_anomaly_margin_passes_C4():
    """PASS branch. 2.14% is the margin 4483.4 / 4389.4 asks for."""
    gates = _gates(CRV.plant(CRV.ANOMALY_MARGIN + 0.001, "none"))
    assert gates["C4_formulation"][0] == EX.PASS


def test_the_published_pod_ratio_refutes_C4_on_formulation_alone():
    """FAIL branch, and it is the world the 2026-08-27 run actually found.

    4475.6 over 4463.0 is 1.0028: re-formulating buys 0.28% where the anomaly
    needs 2.14%. The old script would have reported the same run as a 1.020x
    "vs current" against a constant from another loop, which is the arithmetic
    this decomposition exists to separate.
    """
    gates = _gates(CRV.plant(4475.6 / 4463.0, "none"))
    assert gates["C4_formulation"][0] == EX.FAIL
    assert "1.0028" in gates["C4_formulation"][1]


def test_a_rate_above_the_bus_is_a_validity_failure_and_not_a_finding():
    gates = _gates(CRV.plant(1.005, "pin"))
    assert gates["pin_rate"][0] == EX.FAIL
    assert EX.classify([(k, v) for k, _n, v, _d in
                        CRV.score(CRV.plant(1.005, "pin"), H200,
                                  CRV.REGISTERED_READ_GBPS)]) == EX.INVALID


def test_an_unregistered_card_gets_no_pin_rate_guard_and_says_so():
    """UNKNOWN, not PASS. The H200's 4916.7 would pass an A100 reading 2.4x its
    own bus, so a guard against another card's constant is inert rather than
    loose. Same rule and same table as `ruler_rebaseline.PIN_RATE_GBPS`."""
    gates = _gates(CRV.plant(1.005, "none"), gpu_name="NVIDIA L4")
    assert gates["pin_rate"][0] == EX.UNKNOWN
    assert "inert" in gates["pin_rate"][1]


@pytest.mark.parametrize("defect,word", [("host-bound", "host-bound"),
                                         ("drift", "clock moved")])
def test_a_ratio_taken_across_a_bad_row_is_refused(defect, word):
    """The formulation ratio is only meaningful between two rows measured at one
    operating point with a deep queue; both failures are planted."""
    gates = _gates(CRV.plant(1.03, defect))
    assert gates["within_run"][0] == EX.FAIL
    assert word in gates["within_run"][1]


@pytest.mark.parametrize("defect", ["level", "unlevelled"])
def test_the_cross_instrument_term_is_not_scored_without_a_level(defect):
    """The two figures come from two loops. Comparing them across a clock nobody
    checked would report a throttled card as an instrument difference, which is
    the 2.1%-margin-on-an-8-16%-bias problem in one line."""
    gates = _gates(CRV.plant(1.03, defect))
    assert gates["C4_instrument"][0] == EX.UNKNOWN
    assert gates["within_run"][0] == EX.PASS, \
        "a level says nothing about a ratio taken inside one run"


def test_the_instrument_term_fires_on_the_gap_the_pod_run_measured():
    """4463.0 here against the registered 4389.4 is 1.68% on ONE formulation,
    which is most of the 2.14% the old verdict attributed to formulation."""
    gates = _gates(CRV.plant(1.0028, "none"))
    assert gates["C4_instrument"][0] == EX.FAIL
    assert "+1.68%" in gates["C4_instrument"][1]


def test_the_instrument_term_passes_when_the_two_loops_agree():
    """PASS branch: the same formulation reproducing to within 0.5%."""
    readings = CRV.plant(1.0028, "none")
    baseline = CRV.pick(readings, CRV.BASELINE)
    gates = _gates(readings, registered=baseline.gbps)
    assert gates["C4_instrument"][0] == EX.PASS


def test_a_run_with_no_baseline_row_scores_nothing_rather_than_guessing():
    readings = [r for r in CRV.plant(1.03, "none") if r.name != CRV.BASELINE]
    gates = _gates(readings)
    assert gates["within_run"][0] == EX.UNKNOWN
    assert gates["C4_formulation"][0] == EX.UNKNOWN
    assert gates["C4_instrument"][0] == EX.UNKNOWN


# --------------------------------------------------------------------------
# the CLI contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("argv,code", [
    (["--self-test", "1.03"], EX.CLAIM_FAIL),
    (["--self-test", "1.0028"], EX.CLAIM_FAIL),
    (["--self-test", "1.03", "--self-test-defect", "pin"], EX.INVALID),
    (["--self-test", "1.03", "--self-test-defect", "host-bound"], EX.INVALID),
    (["--self-test", "1.03", "--self-test-defect", "level"], EX.CLAIM_FAIL),
])
def test_the_log_and_the_exit_code_agree_in_every_off_gpu_mode(argv, code, capsys):
    """One property over every mode this file can reach on a laptop: the code
    `classify_text` recomputes from the RESULT lines is the code the process
    returned. A log that says one thing and an exit code that says another is
    the defect `moe.bench.exit_codes` exists to make visible."""
    rc = CRV.main(argv)
    out = capsys.readouterr().out
    assert rc == code
    assert EX.classify_text(out) == rc


def test_a_planted_world_never_carries_the_instrument_name():
    """A self-test transcript must not be quotable as a measurement of a card."""
    CRV.main(["--self-test", "1.03"])
    readings = CRV.plant(1.03, "none")
    assert all(r.instrument != timing.TIMING_BASIS for r in readings)
    assert all(r.instrument == "synthetic/model-generated/not-measured"
               for r in readings)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine with no GPU")
def test_no_card_is_REFUSED_and_not_a_crash(capsys):
    """REFUSED (2) is free and retryable-by-hand; ERROR would be a crash and
    CLAIM_FAIL would be a measured refutation of C4 from a run that timed
    nothing."""
    rc = CRV.main([])
    assert rc == EX.REFUSED
    assert "REFUSED: no CUDA device" in capsys.readouterr().out


def test_an_unplanned_crash_exits_ERROR_and_never_CLAIM_FAIL(monkeypatch, capsys):
    """ONE is CLAIM_FAIL, a RESULT that is never retried; a crash is not one."""
    def explode(argv=None):
        raise RuntimeError("planted: the allocator gave up halfway")

    monkeypatch.setattr(CRV, "_main", explode)
    rc = CRV.main([])
    err = capsys.readouterr().err
    assert rc == EX.ERROR
    assert rc not in EX.FINISHED_CODES
    assert "planted: the allocator gave up halfway" in err


def _committed_read_reduce() -> float:
    """The H200's reduction read, walked out of its own committed yaml.

    Never retyped: it has read 4469.6, 4471.4 and 4467.7 across three
    calibrations of this card, and each literal for it went red in turn. What
    the tests below are about is WHICH pattern the lookup finds, not what it
    measures.
    """
    import yaml

    from moe.bench import roofline as RF
    doc = yaml.safe_load(
        (RF.HARDWARE_DIR / "measured_nvidia_h200.yaml").read_text())
    by_name = {p["pattern"]: float(p["gbps"])
               for p in doc["detail"]["bandwidth_patterns"]}
    for name in CRV.READ_REDUCE_NAMES:
        if name in by_name:
            return by_name[name]
    raise AssertionError(f"no reduction read in the file: {sorted(by_name)}")


def test_the_registered_read_figure_comes_from_the_committed_calibration():
    """Not from the 2026-08-26 constant, which the fixed reduction shape has
    already moved: the H200's reduction read in
    `moe/bench/hardware/measured_nvidia_h200.yaml` sits well above the 4389.4
    the anomaly was computed against. The sentence names the loop that measured
    it, because the whole point of the term is that two loops are compared.

    IT IS FOUND UNDER THE NAME THE FILE USES. `calibrate` renamed `read` to
    `read_reduce` on 2026-09-02 and this lookup matched only `read` until
    2026-09-09, so against the 2026-09-09 calibration it found nothing, fell
    back to the historical 4389.4, and said there was no committed calibration
    for a card whose file was in the tree. Both names resolve, and the source
    line says which one was read."""
    gbps, source = CRV.registered_read(H200)
    assert gbps == pytest.approx(_committed_read_reduce())
    assert gbps != pytest.approx(CRV.REGISTERED_READ_GBPS, abs=1.0), \
        "the committed calibration is preferred over the 2026-08-26 constant"
    assert "read_reduce" in source and "no committed calibration" not in source
    assert "time_eager" in source
    fallback, why = CRV.registered_read("NVIDIA L4")
    assert fallback == CRV.REGISTERED_READ_GBPS
    assert "no committed calibration" in why


def _terms(readings, registered):
    """The THE TWO TERMS block, rendered against a registered figure of choice.

    `--self-test` pins `registered` to REGISTERED_READ_GBPS, which is the ONE
    value that makes the total's two denominators coincide, so every gate test
    above exercises only the degenerate case. These go through `report` directly
    for that reason.
    """
    gates = CRV.score(readings, H200, registered)
    lines = CRV.report(readings, gates, registered, "planted for this test")
    return {ln.split()[0]: ln for ln in lines
            if ln.startswith(("  formulation", "  instrument", "  total"))}


def test_the_total_is_scored_against_the_denominator_it_actually_carries():
    """THE MIXED-DENOMINATOR TRAP, and it only shows off the degenerate case.

    `formulation * instrument` telescopes to `best / registered_gbps`, and
    `registered_gbps` is the committed calibration's reduction read on the
    H200, not the 4389.4 ANOMALY_MARGIN was built on. Printed against
    ANOMALY_MARGIN, the 2026-08-27 pod world read as 1.0013x "against the
    1.0214 the anomaly needs" -- 2.0 points short when it is 0.18 points
    short, an 11x misstatement of the residual on a 2.14% margin, in a
    transcript that is this file's only artefact. The threshold has to be
    recomputed against the same denominator, which is a RELATION between the
    two denominators and not either one's value: `needed` moves with every
    recalibration of this card and `ANOMALY_MARGIN` never does.
    """
    registered, _src = CRV.registered_read(H200)
    assert registered == pytest.approx(_committed_read_reduce()), "the premise moved"
    terms = _terms(CRV.plant(4475.6 / 4463.0, "none"), registered)
    needed = CRV.ANOMALY_GBPS / registered
    assert 1.0 < needed < CRV.ANOMALY_MARGIN, (needed, CRV.ANOMALY_MARGIN)
    assert f"{needed:.4f}" in terms["total"]
    assert f"{CRV.ANOMALY_MARGIN:.4f}" not in terms["total"]
    assert "4483.4" in terms["total"] and f"{registered:.1f}" in terms["total"]


def test_the_total_line_names_the_absolute_rate_it_is_a_ratio_of():
    """A ratio whose numerator is not printed cannot be re-derived by a reader
    holding a different registered figure, which is exactly the reader this
    file's transcript has."""
    registered, _src = CRV.registered_read(H200)
    terms = _terms(CRV.plant(4475.6 / 4463.0, "none"), registered)
    assert "4475.6" in terms["total"]
    assert "torch.sum(dim=1)" in terms["total"]


def test_the_within_run_gate_still_scores_against_the_within_run_margin():
    """The other half: ANOMALY_MARGIN is right for `C4_formulation`, because
    both ends of THAT ratio are rows from this run. Only the total telescoped
    onto a foreign denominator."""
    registered, _src = CRV.registered_read(H200)
    gates = _gates(CRV.plant(CRV.ANOMALY_MARGIN + 0.001, "none"),
                   registered=registered)
    assert gates["C4_formulation"][0] == EX.PASS
    assert f"{CRV.ANOMALY_MARGIN:.4f}" in gates["C4_formulation"][1]


def test_the_self_test_denominator_is_the_one_that_hides_the_trap():
    """Named so nobody re-derives the coverage hole. Off GPU `registered` is
    pinned to REGISTERED_READ_GBPS, and there ANOMALY_GBPS / registered IS
    ANOMALY_MARGIN, so the CLI transcript can never distinguish the two."""
    degenerate = CRV.ANOMALY_GBPS / CRV.REGISTERED_READ_GBPS
    assert degenerate == pytest.approx(CRV.ANOMALY_MARGIN)
    committed, _src = CRV.registered_read(H200)
    assert CRV.ANOMALY_GBPS / committed != pytest.approx(CRV.ANOMALY_MARGIN)


def test_the_retired_call_count_flags_are_gone():
    """`--iters 50` and `--warmup 5` were CALL COUNTS. `time_kernel` warms for a
    DURATION of delivered GPU load and derives its own iteration count, and
    neither converts, which is why `time_call` refuses rather than wrapping."""
    parser = CRV.build_parser()
    flags = {action.dest for action in parser._actions}
    assert "iters" not in flags and "warmup" not in flags
    assert {"warmup_ms", "target_ms", "trials"} <= flags


def test_a_reference_clock_that_is_not_a_clock_is_REFUSED(capsys):
    """A bad flag is a precondition not met, and it is checked BEFORE the card.

    Falling back to the calibration's clock would score LEVEL against a
    reference the operator did not ask for, in the one file whose subject is
    which apparatus produced which number. Checked before `torch.cuda` so the
    branch is reachable, and so a mistyped flag costs no pod minutes.
    """
    assert CRV.main(["--reference-clock-mhz", "0"]) == EX.REFUSED
    assert "is not a clock" in capsys.readouterr().out


def test_the_measured_the_planted_and_the_annotated_formulations_are_one_set():
    """Three lists of names in one file is three chances to drift.

    The lambdas are never called here, so CPU tensors are enough to read the
    names back out of `variants`.
    """
    import torch as _torch

    measured = tuple(name for name, _fn in CRV.variants(
        _torch.zeros(1), _torch.zeros(1), _torch.zeros(1)))
    assert measured == CRV.VARIANT_NAMES
    assert tuple(CRV.VARIANT_NOTES) == CRV.VARIANT_NAMES
    assert tuple(r.name for r in CRV.plant(1.0, "none")) == CRV.VARIANT_NAMES
    assert CRV.BASELINE == CRV.VARIANT_NAMES[0]
