"""One instrument, one exit-code table, one MDE line, across five scripts.

WHAT THIS FILE IS FOR. `tests/test_run_ids.py` covers the naming half of this
slice; this is the other half, and each section names the audit finding it
closes.

A7, THE INSTRUMENT. `block_m_crossing_sweep.time_call` had five verbatim
copies. Each created its CUDA events INSIDE the timing loop, recorded the start
event on a stream that had just been synchronised -- an idle GPU, so the host's
own enqueue cost sat inside the interval -- synchronised after every iteration,
never flushed L2, and read no clock. The roof every one of their ratios is read
against was measured queue-deep with pre-primed events and a flush. The audit
bounded the exposed host prefix at ~0.18 ms per fused_experts call on the H200
pod and ~0.30 ms on the A100, i.e. alpha biased low 12-16% and 8-11%
respectively at the smallest cells -- different per card, and of the order of
the cross-card effect the study registered. Three of the six copies are in this
slice's files; this file checks they are gone and that what replaced them
records the state it measured in.

A4, THE EXIT CODES. Four scripts refused with 3 while the driver read 3 as
"measured, and its cells must not be quoted", so runs that had spent nothing
were logged as runs whose numbers were invalid, and the driver's closing summary
grepped free text (`floor|sigma`) and printed a REFUSED log's pre-registered
expectations as measured output. The fix is one `RESULT:` line per scored gate
and `exit_codes.classify` over the same gates, so the log and the exit code
cannot disagree.

B14, THE MDE. No arm in this study stated a minimum detectable effect, so a gate
could pass or fail without anyone knowing whether the design could have resolved
the difference either way. Every plan output now prints one, derived from a
spread MEASURED over the published corpus rather than from a prior.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes, timing  # noqa: E402

H200 = "NVIDIA H200"

#: The files this slice owns that carry a timing loop, plus the two whose only
#: instrument change is the shared module they already called.
OWNED = ("tile_sweep", "tuned_vs_fallback", "dtype_tile_confound",
         "ruler_rebaseline", "group_m_alpha_sweep", "alias_ablation")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TILE = _load("tile_sweep")
TVF = _load("tuned_vs_fallback")
GROUP_M = _load("group_m_alpha_sweep")
ALIAS = _load("alias_ablation")


# --------------------------------------------------------------------------
# A7: the private timing loops are gone
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", OWNED)
def test_no_file_in_this_slice_defines_its_own_timing_loop(name):
    """`grep -c "def time_call" scripts/*.py` -> 0 for every file here.

    Both spellings, because the six copies were not identically named: the
    sweep and the anchor called it `time_call` and the three vLLM scripts called
    it `time_calls`, which is why a grep for one of them missed half of them.
    """
    source = (ROOT / "scripts" / f"{name}.py").read_text()
    assert "def time_call(" not in source, name
    assert "def time_calls(" not in source, name


@pytest.mark.parametrize("name", ("tile_sweep", "tuned_vs_fallback",
                                  "dtype_tile_confound", "group_m_alpha_sweep",
                                  "alias_ablation"))
def test_every_measuring_script_calls_the_shared_instrument(name):
    """The other half of the check above: deleting a loop and leaving no timing
    at all would pass it."""
    source = (ROOT / "scripts" / f"{name}.py").read_text()
    assert "time_kernel(" in source, name


@pytest.mark.parametrize("module", (TILE, TVF))
def test_the_timing_columns_are_in_the_csv_header(module):
    """A column the row carries and the header does not is a column csv drops
    silently, which is how a published arm came to have no iteration count."""
    for column in module.TIMING_CSV_COLUMNS:
        assert column in module.CSV_COLUMNS, column
    for column in ("instrument", "warmup_ms", "iters", "trials",
                   "sm_clock_load_mhz", "clock_level_ok", "clock_drift_ok",
                   "l2_flush"):
        assert column in module.CSV_COLUMNS, column
    # And the provenance block, so a row read on its own names its commit.
    for column in ("prov_git_sha", "prov_gpu_name", "prov_instrument"):
        assert column in module.CSV_COLUMNS, column


def test_the_tile_sweep_writes_every_timing_column_it_promises():
    """`timing_columns` and the header are checked against each other, because
    the two drifting apart is the failure the grouping constant exists for."""
    class T:
        instrument = timing.TIMING_BASIS
        warmup_ms, iters, trials = 300.0, 120, 3
        ms_p50, ms_p90, ms_min, ms_std = 1.0, 1.1, 0.9, 0.01
        sm_clock_load_mhz = 1490.0
        clock_level_ok, clock_drift_ok, l2_flush = False, True, True
        host_bound = None

    row = TILE.timing_columns(T())
    assert set(row) == set(TILE.TIMING_CSV_COLUMNS)
    assert row["clock_level_ok"] == "0"
    assert row["clock_drift_ok"] == "1"
    assert row["host_bound"] == "", (
        "an undetermined flag must be EMPTY, not 0: reading it as False drops a "
        "good cell and reading it as True keeps a host-bound one")


def test_one_bad_repeat_makes_a_tuned_arm_say_so():
    """`summarise_timings` folds the repeats so a filter over the arm's row
    cannot be more permissive than a filter over its repeats."""
    class T:
        def __init__(self, level, drift, host):
            self.instrument = timing.TIMING_BASIS
            self.l2_flush = True
            self.iters, self.trials, self.warmup_ms = 100, 3, 300.0
            self.sm_clock_load_mhz = 1500.0
            self.clock_level_ok, self.clock_drift_ok = level, drift
            self.host_bound = host

    def fold(*repeats):
        result = TVF.ArmResult("m", 1, "native", None, "observed")
        TVF.summarise_timings(result, list(repeats))
        return result

    clean = fold(T(True, True, False), T(True, True, False))
    assert (clean.clock_level_ok, clean.clock_drift_ok, clean.host_bound) == (
        True, True, False)
    assert clean.instrument == timing.TIMING_BASIS
    dirty = fold(T(True, True, False), T(False, True, True))
    assert dirty.clock_level_ok is False
    assert dirty.host_bound is True
    unknown = fold(T(True, True, False), T(None, None, None))
    assert unknown.clock_level_ok is None
    assert unknown.host_bound is None


def test_a_resumed_tuned_row_keeps_the_state_it_was_measured_in(tmp_path):
    """A row written before 2026-09-02 carries no instrument, and must come
    back saying so rather than inheriting this run's."""
    cell = TVF.plan_cells(["mixtral-8x7b"], [32], "bf16", H200)[0][0]
    meta = {"run_id": "r", "gpu_name": H200, "vllm_version": "",
            "torch_version": "", "routing": "uniform", "seed": 0, "prov": None}
    store = TVF.Store(tmp_path / "t.csv")
    result = TVF.ArmResult(cell.model, cell.num_tokens, "native", None,
                           "observed", ms_median=1.0, n_samples=3,
                           instrument=timing.TIMING_BASIS, warmup_ms=300.0,
                           iters=120, trials=3, sm_clock_load_mhz=1490.0,
                           clock_level_ok=False, clock_drift_ok=True,
                           l2_flush=True, host_bound=None)
    store.write(result, cell, meta)
    store.close()
    back = TVF.Store(tmp_path / "t.csv").restore(result.key)
    assert back.instrument == timing.TIMING_BASIS
    assert back.iters == 120 and back.trials == 3
    assert back.clock_level_ok is False and back.clock_drift_ok is True
    assert back.host_bound is None


# --------------------------------------------------------------------------
# A4: one RESULT line per scored gate, and the code comes from those gates
# --------------------------------------------------------------------------

def test_the_tile_sweep_gate_pair_discriminates_both_ways():
    """C1 is the paragraph the script used to end on, made scoreable.

    A FLAT curve confirms the thesis and an improvement at a bigger tile refutes
    it, so both branches are planted here; a gate whose FAIL branch nothing has
    ever produced is a gate nobody has tested.
    """
    def rows(*ratios):
        return [{"block_size_m": 16 * 2 ** i, "ms_p50": 1.0, "error": "",
                 "ratio_vs_first": r} for i, r in enumerate(ratios)]

    flat = TILE.build_gates(rows(1.0, 1.01, 0.99, 1.02), band=0.04)
    assert {g.name.split()[0]: g.verdict for g in flat} == {
        "V1": exit_codes.PASS, "C1": exit_codes.PASS}
    assert exit_codes.classify(g.scored() for g in flat) == exit_codes.DONE

    faster = TILE.build_gates(rows(1.0, 1.0, 0.80, 0.78), band=0.04)
    by_name = {g.name.split()[0]: g for g in faster}
    assert by_name["C1"].verdict == exit_codes.FAIL
    assert "does not allow" in by_name["C1"].observed
    assert (exit_codes.classify(g.scored() for g in faster)
            == exit_codes.CLAIM_FAIL)


def test_a_tile_sweep_that_compiled_one_setting_cannot_report_a_flat_curve():
    """V1's FAIL branch, and it is the one that matters.

    A sweep where every setting but one failed to compile prints a perfectly
    flat curve, and a flat curve is what this experiment reads as confirmation.
    UNKNOWN on the claim and FAIL on validity is INVALID, not DONE.
    """
    rows = [{"block_size_m": 16, "ms_p50": 1.0, "error": "", "ratio_vs_first": 1.0},
            {"block_size_m": 64, "ms_p50": None, "error": "OutOfResources"}]
    gates = TILE.build_gates(rows, band=0.04)
    by_name = {g.name.split()[0]: g for g in gates}
    assert by_name["V1"].verdict == exit_codes.FAIL
    assert by_name["C1"].verdict == exit_codes.UNKNOWN
    assert exit_codes.classify(g.scored() for g in gates) == exit_codes.INVALID


def test_the_tile_sweeps_dry_run_prints_no_result_line(tmp_path, capsys):
    """A REFUSED log must carry none, or `classify_text` recomputes a verdict
    for a run that measured nothing."""
    code = TILE.main(["--dry-run", "--card", H200, "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == exit_codes.REFUSED
    assert exit_codes.parse_result_lines(out) == []
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(out)


def test_the_tuned_gate_split_is_validity_then_claim():
    """G0-G3 say whether the page may be believed; G4-G7 are the predictions.

    The split was in the docstring and in the append order and had never been a
    field, so the process exited 1 for either. Now a VALIDITY failure is INVALID
    and a CLAIM failure is a RESULT, which is what the driver acts on.
    """
    kinds = {}
    for name in ("G0 same-layer", "G1 derivation", "G2 override", "G3 placebo",
                 "G4 sign", "G5 size", "G6 tile height", "G7 swizzle"):
        gate = TVF.Gate(name, "p", "r", True, "saw")
        kinds[gate.token] = gate.kind
    assert kinds == {"G0": exit_codes.VALIDITY, "G1": exit_codes.VALIDITY,
                     "G2": exit_codes.VALIDITY, "G3": exit_codes.VALIDITY,
                     "G4": exit_codes.CLAIM, "G5": exit_codes.CLAIM,
                     "G6": exit_codes.CLAIM, "G7": exit_codes.CLAIM}
    validity_fail = [TVF.Gate("G0 same-layer", "p", "r", False, "saw"),
                     TVF.Gate("G5 size", "p", "r", True, "saw")]
    assert (exit_codes.classify(g.scored() for g in validity_fail)
            == exit_codes.INVALID)
    claim_fail = [TVF.Gate("G0 same-layer", "p", "r", True, "saw"),
                  TVF.Gate("G5 size", "p", "r", False, "saw")]
    assert (exit_codes.classify(g.scored() for g in claim_fail)
            == exit_codes.CLAIM_FAIL)


def test_a_tuned_gate_renders_exactly_one_result_line():
    gate = TVF.Gate("G5 size", "the fallback costs at least 15%", "median >= 1.15",
                    False, "median penalty 1.02")
    lines = [ln for ln in gate.render().splitlines()
             if ln.startswith(exit_codes.RESULT_PREFIX)]
    assert lines == ["RESULT: CLAIM G5 FAIL median penalty 1.02"]
    assert gate.render(with_result=False).count("RESULT: ") == 0
    # The human shape the session driver's own test pins must survive.
    assert "[FAIL] G5 size" in gate.render()


@pytest.mark.parametrize("module,name,kind", [
    (GROUP_M, "P1 alpha falls", exit_codes.CLAIM),
    (GROUP_M, "regime: every cell is memory bound", exit_codes.VALIDITY),
    (ALIAS, "P1: the ablation agrees with the refit, alpha = 0.558",
     exit_codes.CLAIM),
    (ALIAS, "correctness: each variant reproduced its closed form",
     exit_codes.VALIDITY),
    (ALIAS, "resolution: the interval can separate the three candidates",
     exit_codes.VALIDITY),
])
def test_the_two_sweeps_split_their_gates_into_validity_and_claim(module, name, kind):
    assert module.Gate(name, True, "detail").kind == kind


@pytest.mark.parametrize("module,names", [
    (GROUP_M, ("regime: every cell is memory bound",
               "regime: the multi-tile rung has re-reads to save")),
    (ALIAS, ("form: one thing", "form: another thing")),
])
def test_two_gates_sharing_a_first_word_get_two_different_tokens(module, names):
    """The collision the slug exists for.

    `group_m_alpha_sweep` really does ship two gates called `regime: ...`. A
    token taken from the first word alone would have given both the same name,
    and the driver keys its summary on that name: one of the two would have
    vanished from every log without anything looking wrong.
    """
    tokens = {module.Gate(name, True, "d").token for name in names}
    assert len(tokens) == 2, tokens
    assert not any(any(c.isspace() for c in t) for t in tokens)


@pytest.mark.parametrize("module", (GROUP_M, ALIAS))
def test_every_sweep_gate_verdict_reaches_a_result_line(module):
    for ok, expected in ((True, exit_codes.PASS), (False, exit_codes.FAIL),
                         (None, exit_codes.UNKNOWN)):
        gate = module.Gate("band holds", ok, "detail")
        parsed = exit_codes.parse_result_lines(gate.result_line())
        assert len(parsed) == 1
        assert parsed[0].verdict == expected
        assert parsed[0].name == "band-holds"


def test_a_synthetic_group_m_run_prints_result_lines_the_driver_can_read(
        tmp_path, monkeypatch, capsys):
    """End to end: the log's RESULT lines must recompute the process's code.

    `--synthetic flat` plants a scalar alpha, which P1 must refute; the run is a
    CLAIM_FAIL and the log has to say so in the one line the driver greps.
    """
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    code = GROUP_M.main(["--synthetic", "flat"])
    out = capsys.readouterr().out
    lines = {r.name: r.verdict for r in exit_codes.parse_result_lines(out)}
    assert lines, "no RESULT line at all; the driver's summary would be empty"
    p1 = [name for name in lines if name.startswith("P1")]
    assert p1 and lines[p1[0]] == exit_codes.FAIL
    assert code == exit_codes.CLAIM_FAIL
    assert exit_codes.classify_text(out) == code
    # And every name is distinct: two gates in this script share a first word
    # ("regime: ..."), so a token taken from the first word alone would have
    # collided and the driver would have seen one of them.
    assert len(lines) == len(exit_codes.parse_result_lines(out))


# --------------------------------------------------------------------------
# A5: provenance in every report these scripts write
# --------------------------------------------------------------------------

def test_the_tuned_plan_carries_the_provenance_block(tmp_path):
    TVF.main(["--plan-only", "--card", H200, "--tokens", "1,32",
              "--out-dir", str(tmp_path)])
    plan = json.loads(next(tmp_path.rglob("plan.json")).read_text())
    for key in ("git_sha", "gpu_name", "ridge_source", "bandwidth_source",
                "instrument"):
        assert key in plan, key
    assert plan["provenance"]["instrument"] == timing.TIMING_BASIS
    assert plan["provenance"]["missing"]["gpu_name"], (
        "off GPU the block must say WHY it has no card, not leave it blank")
    assert plan["card"] == H200


def test_a_synthetic_sweep_leaves_a_provenance_file(tmp_path, monkeypatch):
    """Beside report.md rather than inside it: the markdown is prose a human
    edits and the block is fields a script reads."""
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    ALIAS.main(["--synthetic", "refit"])
    written = list(tmp_path.rglob("provenance.json"))
    assert len(written) == 1
    block = json.loads(written[0].read_text())
    assert block["provenance"]["instrument"] == timing.TIMING_BASIS
    assert block["git_sha"], "a run nobody can attribute to a commit"


def test_the_ruler_keeps_its_provenance_out_of_the_reproducible_report(
        tmp_path, monkeypatch):
    """`report.txt` is compared byte for byte between two corpus-only runs to
    prove the replay reads no hardware, so a UTC stamp in there would break that
    for a reason unrelated to reproducibility. It goes in the JSON."""
    ruler = _load("ruler_rebaseline")
    # CLAIM_FAIL, because the committed corpus refutes C5 with 90 flips. It is
    # the code the log has implied all along; the DONE this asserted until
    # 2026-09-02 was the masking branch, not the corpus answer.
    assert (ruler.main(["--corpus-only", "--out", str(tmp_path)])
            == exit_codes.CLAIM_FAIL)
    text = next(tmp_path.rglob("report.txt")).read_text()
    payload = json.loads(next(tmp_path.rglob("report.json")).read_text())
    # The word appears in the corpus prose (an arm's "provenance:
    # ceilings_disagree" note); what must NOT be in there is the block's own
    # non-reproducible fields.
    assert "git_sha" not in text
    assert "utc" not in text.split("GATES")[0]
    assert payload["provenance"]["instrument"] == timing.TIMING_BASIS
    assert payload["git_sha"]


# --------------------------------------------------------------------------
# B14: the MDE, from a MEASURED spread
# --------------------------------------------------------------------------

def _published_spreads() -> list[float]:
    out = []
    for path in (ROOT / "results" / "published").glob("*/*.report.json"):
        spread = json.loads(path.read_text()).get("timing_spread_median")
        if spread:
            out.append(float(spread))
    return out


@pytest.mark.parametrize("module", (TILE, TVF))
def test_the_spread_constants_are_the_published_corpus(module):
    """Read out of the corpus, so a re-published arm moves the constant."""
    values = _published_spreads()
    assert len(values) >= 20, "the corpus these constants come from is gone"
    assert module.MEASURED_SPREAD_MEDIAN == pytest.approx(
        statistics.median(values), abs=0.0005)
    assert module.MEASURED_SPREAD_MAX == pytest.approx(max(values), abs=0.0005)


def test_the_group_m_power_noise_is_no_longer_below_the_measured_range():
    """S37: it was 0.005, under a corpus that runs 0.0039 to 0.0182, and it was
    described as "the pessimistic end" of a 0.2% figure."""
    values = _published_spreads()
    assert GROUP_M.POWER_NOISE == pytest.approx(statistics.median(values),
                                                abs=0.0005)
    assert GROUP_M.POWER_NOISE_MAX == pytest.approx(max(values), abs=0.0005)
    assert GROUP_M.POWER_NOISE > 0.005
    assert min(values) <= GROUP_M.POWER_NOISE <= max(values)


def test_the_group_m_band_gate_is_relative_to_the_effect_it_must_resolve():
    """S37 again: an absolute 0.15 against an effect of 0.082 called a band
    1.8x the effect "identified", so the interval could contain both zero and
    twice the number being claimed."""
    assert GROUP_M.max_band_width() == pytest.approx(
        GROUP_M.BAND_WIDTH_OVER_EFFECT * GROUP_M.PUBLISHED_GROUP_M_EFFECT)
    assert GROUP_M.max_band_width() <= GROUP_M.PUBLISHED_GROUP_M_EFFECT
    assert GROUP_M.max_band_width() < 0.15


def test_the_group_m_plan_prints_an_mde_at_both_ends_of_the_spread(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    # REFUSED: this invocation prints a plan and times nothing. It returned 0
    # until 2026-09-02, so a driver that asked for the arm and got a bare
    # invocation logged it DONE and never ran it.
    assert GROUP_M.main([]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "MDE" in out
    assert f"{GROUP_M.POWER_NOISE:.2%}" in out
    assert f"{GROUP_M.POWER_NOISE_MAX:.2%}" in out
    assert "RESOLVED" in out or "CANNOT RESOLVE IT" in out


@pytest.mark.parametrize("module", (TILE, TVF))
def test_more_noise_costs_resolution_rather_than_buying_it(module):
    wide = module.mde_of_ratio(module.MEASURED_SPREAD_MAX, 3)
    narrow = module.mde_of_ratio(module.MEASURED_SPREAD_MEDIAN, 3)
    assert wide > narrow
    assert module.mde_of_ratio(module.MEASURED_SPREAD_MEDIAN, 12) < narrow
    with pytest.raises(ValueError):
        module.mde_of_ratio(0.0, 3)
    with pytest.raises(ValueError):
        module.mde_of_ratio(0.01, 0)


def test_the_tuned_mde_line_says_when_the_design_cannot_see_its_own_effect():
    """The branch that matters: a design that cannot resolve the 15% it is
    registered to find has to say so before the box is rented."""
    args = TVF.build_parser().parse_args(["--reps", "1"])
    planted = TVF.render_mde(args, spreads=(("median", 0.0077),
                                            ("planted", 0.09)))
    assert "CANNOT resolve the effect" in planted
    assert "resolves the effect" in planted
    assert "CANNOT resolve" not in TVF.render_mde(args)


def test_the_tile_sweep_band_is_the_mde_and_not_a_round_number():
    """C1's band has to be something the box can show, or "flat" is a claim
    about the threshold rather than about the kernel."""
    assert TILE.improvement_band(3) == pytest.approx(
        TILE.mde_of_ratio(TILE.MEASURED_SPREAD_MAX, 3))
    assert TILE.improvement_band(3) > TILE.improvement_band(30)


# --------------------------------------------------------------------------
# THE SECOND PASS, 2026-09-02. Everything below closes a defect the review of
# the first pass found in the fix itself, which is the shape this slice keeps
# reproducing: the acceptance check passed and the substance did not.
# --------------------------------------------------------------------------

def test_the_alias_rung_has_no_private_event_loop_left():
    """The fourth copy, which had no function name and so survived the grep.

    `grep -c "def time_call"` was 0 across this slice while `measure_rung` still
    created a `torch.cuda.Event` pair inside its replicate loop, recorded the
    start on a stream the previous iteration had just drained, and synchronised
    after every launch. An acceptance check keyed on a NAME cannot see a loop
    that was never given one, so this one is keyed on the shape.
    """
    import ast
    tree = ast.parse((ROOT / "scripts" / "alias_ablation.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "measure_rung")
    body = ast.unparse(fn)
    assert "time_kernel(" in body
    assert "Event(" not in body, (
        "measure_rung creates its own CUDA events again; that is the A7 shape "
        "with a different name")
    assert "elapsed_time(" not in body
    # The two idle-instant clock samples stay, as a comparison against the
    # queue-deep ones, and must not be the only clock evidence.
    assert "ClockState.sample()" in body


def test_one_bad_pass_makes_an_alias_rung_say_so():
    """`fold_timings` folds a rung's calls so a filter over the row cannot be
    more permissive than a filter over the calls it came from."""
    class T:
        def __init__(self, level, drift, host):
            self.instrument = timing.TIMING_BASIS
            self.l2_flush = True
            self.iters, self.trials, self.warmup_ms = 100, 3, 300.0
            self.sm_clock_load_mhz = 1500.0
            self.clock_level_ok, self.clock_drift_ok = level, drift
            self.host_bound = host

    clean = ALIAS.fold_timings([T(True, True, False), T(True, True, False)])
    assert set(clean) == set(ALIAS.TIMING_COLUMNS)
    assert clean["instrument"] == timing.TIMING_BASIS
    assert (clean["clock_level_ok"], clean["clock_drift_ok"],
            clean["host_bound"]) == (True, True, False)
    dirty = ALIAS.fold_timings([T(True, True, False), T(False, True, True)])
    assert dirty["clock_level_ok"] is False
    assert dirty["host_bound"] is True
    unknown = ALIAS.fold_timings([T(True, True, False), T(None, None, None)])
    assert unknown["clock_level_ok"] is None, (
        "an undetermined flag must stay None: reading it as False drops a good "
        "rung and reading it as True keeps a throttled one")
    assert unknown["host_bound"] is None
    assert ALIAS.fold_timings([]) == {}


def test_every_alias_preflight_gate_is_validity_including_the_one_named_p2():
    """The gate that made the explicit kind necessary.

    `P2 is testable: the models straddle L2` asks whether the DESIGN can address
    prediction 2. The name-derived rule read the `P2` and emitted
    `RESULT: CLAIM P2-is-testable-...`, so a design that could not ask the
    question announced itself as a refuted claim about the world.
    """
    design = ALIAS.build_design(ALIAS.parse_args([]))
    gates = ALIAS.preflight(design, 50 * 2 ** 20)
    named = [g for g in gates if g.token.startswith("P2")]
    assert named, "the straddle gate is gone; this test is now vacuous"
    assert all(g.kind == exit_codes.VALIDITY for g in gates), [
        (g.token, g.kind) for g in gates if g.kind != exit_codes.VALIDITY]
    # And the fallback still works for the result gates, both ways.
    assert ALIAS.Gate("P1: the ablation agrees", True, "d").kind == exit_codes.CLAIM
    assert ALIAS.Gate("placebo: nothing moved", True, "d").kind == exit_codes.VALIDITY
    # A kind outside the table is refused rather than carried into a RESULT line.
    with pytest.raises(ValueError):
        ALIAS.Gate("planted", True, "d", kind="MAYBE")


@pytest.mark.parametrize("module,argv", [
    (ALIAS, []),
    (GROUP_M, []),
])
def test_a_sweep_that_measures_nothing_prints_no_result_line_and_refuses(
        module, argv, tmp_path, monkeypatch, capsys):
    """The regression the first pass introduced, both files.

    Moving the preflight RESULT lines into the plan made a bare invocation print
    four or five `RESULT: VALIDITY ... PASS` lines and exit 0, so
    `classify_text` recomputed DONE for a run that had spent nothing. A REFUSED
    log must carry no RESULT line at all, and the process must agree with it.
    """
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    code = module.main(argv)
    out = capsys.readouterr().out
    assert code == exit_codes.REFUSED
    assert exit_codes.parse_result_lines(out) == []
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(out)
    assert "Nothing was measured" in out
    # The verdicts are still PRINTED; they are prose there, which is the whole
    # point of moving the line rather than the table.
    assert "[PASS]" in out


@pytest.mark.parametrize("module,argv", [
    (ALIAS, ["--synthetic", "refit"]),
    (GROUP_M, ["--synthetic", "monotone"]),
])
def test_a_measured_sweep_scores_its_preflight_gates_in_the_result_lines(
        module, argv, tmp_path, monkeypatch, capsys):
    """The other half: moving the lines must not lose the gates.

    A preflight gate that prints no RESULT line anywhere has vanished from the
    driver's view, which is the failure this slice is named against in the
    opposite direction.
    """
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    module.main(argv)
    out = capsys.readouterr().out
    names = {r.name for r in exit_codes.parse_result_lines(out)}
    plan = module.build_design if module is ALIAS else module.build_plan
    design = plan(module.parse_args([]))
    expected = {g.token for g in module.preflight(
        design, 50 * 2 ** 20 if module is ALIAS else 100.0)}
    assert expected <= names, sorted(expected - names)


def test_the_alias_mde_is_derived_and_both_of_its_branches_are_reachable():
    """B14 for the ablation: the plan must price what the ladder could resolve.

    The FAIL branch is planted with a spread the corpus does not contain,
    because on today's corpus this ladder clears `MAX_BAND_WIDTH` at both ends
    and a branch nothing can reach is a branch nobody has read.
    """
    design = ALIAS.build_design(ALIAS.parse_args([]))
    top = max(design.tiles)
    wide = ALIAS.mde_of_alpha(ALIAS.MEASURED_SPREAD_MAX, design.replicates, top)
    narrow = ALIAS.mde_of_alpha(ALIAS.MEASURED_SPREAD_MEDIAN,
                                design.replicates, top)
    assert wide > narrow
    assert narrow <= ALIAS.MAX_BAND_WIDTH
    # More replicates buy resolution; a shorter ladder costs it.
    assert ALIAS.mde_of_alpha(ALIAS.MEASURED_SPREAD_MEDIAN, 36, top) < narrow
    assert ALIAS.mde_of_alpha(ALIAS.MEASURED_SPREAD_MEDIAN,
                              design.replicates, 2) > narrow
    for bad in ((0.0, 9, 8), (0.01, 0, 8), (0.01, 9, 1)):
        with pytest.raises(ValueError):
            ALIAS.mde_of_alpha(*bad)

    lines: list[str] = []

    def say(line: str = "") -> None:
        lines.append(line)

    ALIAS.report_mde(say, design)
    assert any("MDE on alpha" in line for line in lines)
    assert any("resolves the candidates" in line for line in lines)
    assert not any("CANNOT resolve" in line for line in lines)
    lines.clear()
    ALIAS.report_mde(say, design, spreads=(("median", 0.0077), ("planted", 0.4)))
    assert any("CANNOT resolve them" in line for line in lines)


def test_the_alias_spread_constants_are_the_published_corpus():
    values = _published_spreads()
    assert ALIAS.MEASURED_SPREAD_MEDIAN == pytest.approx(
        statistics.median(values), abs=0.0005)
    assert ALIAS.MEASURED_SPREAD_MAX == pytest.approx(max(values), abs=0.0005)


def test_the_ruler_dry_run_prices_what_it_could_see_and_not_only_what_it_costs():
    """B14 for the re-baseline. It printed an estimated GPU time and no MDE.

    Both branches are planted: on the measured spread this apparatus cannot
    resolve gate 2's 0.5% reproduction tolerance, which is a fact about the
    design a reader has to be told, and a tighter spread flips it.
    """
    ruler = _load("ruler_rebaseline")
    wide = ruler.mde_of_pattern(ruler.MEASURED_SPREAD_MAX, 3)
    narrow = ruler.mde_of_pattern(ruler.MEASURED_SPREAD_MEDIAN, 3)
    assert wide > narrow
    assert ruler.mde_of_pattern(ruler.MEASURED_SPREAD_MEDIAN, 12) < narrow
    for bad in ((0.0, 3), (0.01, 0)):
        with pytest.raises(ValueError):
            ruler.mde_of_pattern(*bad)
    measured = "\n".join(ruler.mde_lines())
    assert "MDE" in measured
    assert "CANNOT resolve it" in measured
    tight = "\n".join(ruler.mde_lines(spreads=(("planted", 0.0005),)))
    assert "resolves it" in tight
    assert "CANNOT resolve it" not in tight


def test_the_ruler_dry_run_prints_the_mde_beside_the_cost(tmp_path, capsys):
    ruler = _load("ruler_rebaseline")
    assert ruler.main(["--dry-run", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "estimated GPU time" in out
    assert "MDE." in out
    assert f"{ruler.MEASURED_SPREAD_MEDIAN:.2%}" in out


@pytest.mark.parametrize("module", ("ruler_rebaseline", "dtype_tile_confound"))
def test_no_script_folds_a_claim_failure_into_done(module):
    """The one defect `exit_codes` exists to detect, in the two files that had
    it, checked on the source because the alternative is a pod.

    Both computed `rc = classify(...)` and then `return DONE` when rc was
    CLAIM_FAIL and a flag was absent, under a comment asserting that
    `classify_text` on the log recomputes the code the process returned. The
    masking was obsolete anyway: 1 is in `FINISHED_CODES` and `ledger_state(1)`
    is "CLAIM_FAIL", so the ledger already reads it as a result rather than a
    retry.
    """
    source = (ROOT / "scripts" / f"{module}.py").read_text()
    assert "return exit_codes.DONE" not in source, (
        f"{module} returns DONE from somewhere other than the gate table")
    assert "reported as exit" not in source


def test_the_ruler_corpus_replay_returns_the_code_its_log_implies(tmp_path,
                                                                  capsys):
    """End to end for the same thing: 90 flips refute C5, and the process says
    so."""
    ruler = _load("ruler_rebaseline")
    code = ruler.main(["--corpus-only", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == exit_codes.CLAIM_FAIL
    assert exit_codes.classify_text(out) == code
    assert {r.name: r.verdict for r in exit_codes.parse_result_lines(out)}["C5"] \
        == exit_codes.FAIL


def test_the_tuned_refusals_all_return_refused_and_not_invalid(tmp_path, capsys):
    """R3's defect, in the sibling this slice rewrote and did not fix.

    3 is INVALID: measured, a validity gate failed, and the directory is full of
    cells that must not be scored. Every refusal here returned it, including the
    card refusal this slice added, so a run that spent nothing announced a
    poisoned directory to the driver and the arm was never queued again.
    """
    assert TVF.EXIT_NOT_MEASURED == exit_codes.REFUSED
    assert TVF.EXIT_GATE_FAILED == exit_codes.CLAIM_FAIL
    assert TVF.EXIT_INVALID == exit_codes.INVALID
    code = TVF.main(["--plan-only", "--gpu-name", H200, "--tokens", "1",
                     "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == exit_codes.REFUSED
    assert exit_codes.parse_result_lines(out) == []
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(out)


def test_the_tuned_card_refusal_announces_refused_rather_than_invalid():
    """`require_card_to_measure` is the newest refusal and had the oldest bug."""
    with pytest.raises(TVF.NoCardToLabel):
        TVF.require_card_to_measure(TVF.ASSUMED_CARD)
    TVF.require_card_to_measure(H200)          # the PASS branch of the same gate


def test_an_unnamed_card_buys_no_tuned_side_to_compare_against(tmp_path, capsys):
    """R6's substantive half: the LOOKUP, not the label.

    The lookup decides which tuned file `resolve_tile` reads, so it decides
    whether the premise of this experiment -- that there IS a tuned side to
    price the ladder against -- holds. The two default models have a tuned H200
    file and almost nothing else does, so defaulting the lookup to an H200 made
    the premise true by construction on every machine, under a label that said
    only that the CARD was assumed.
    """
    env = {"gpu_name": None}
    args = TVF.build_parser().parse_args([])
    lookup, card, note = TVF.resolve_lookup_gpu(args, env)
    assert lookup is None
    assert card == TVF.ASSUMED_CARD
    assert "NO CONFIG LOOKUP DEVICE" in note
    # The PASS branch of the same rule: naming a card gives a lookup again.
    named = TVF.build_parser().parse_args(["--gpu-name", H200])
    assert TVF.resolve_lookup_gpu(named, env)[0] == H200

    TVF.main(["--plan-only", "--tokens", "1,32", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "config lookup device NONE" in out
    # Every H200 in the header is part of `ASSUMED NVIDIA H200`; a BARE one
    # would be a run that had quietly decided which card it was.
    head = out.split("NO CARD WAS NAMED")[0].replace(TVF.ASSUMED_CARD, "")
    assert H200 not in head, (
        "an off-GPU run with no card named a bare H200 before the disclaimer")
    plan = json.loads(next(tmp_path.rglob("plan.json")).read_text())
    assert plan["cells"] == [], "the coverage table was true by construction"
    assert plan["card"] == TVF.ASSUMED_CARD


def test_the_alias_run_id_moves_with_every_timing_knob():
    """The knobs `time_kernel` added are swept and set the measured
    milliseconds of every pass, so a re-run at a different one must not land in
    the directory that holds the old numbers."""
    design = ALIAS.build_design(ALIAS.parse_args([]))
    base = ALIAS.default_run_id(ALIAS.parse_args([]), H200, design)
    for flag, value in (("--warmup", "50"), ("--cell-budget-ms", "10"),
                        ("--trials", "7")):
        args = ALIAS.parse_args([flag, value])
        assert ALIAS.default_run_id(args, H200, design) != base, flag
    assert ALIAS.default_run_id(ALIAS.parse_args([]), H200, design) == base
