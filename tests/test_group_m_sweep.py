"""The GROUP_SIZE_M sweep must be able to see an effect AND to miss its absence.

A pod script that always prints PASS is worth nothing, so most of this file runs
the whole report end to end against measurements generated from three STATED
laws and checks that the verdict changes with the law:

  monotone   alpha falls with the swizzle width          -> every gate passes
  flat       alpha is a scalar                           -> P1 fails
  order      alpha is a scalar and a per-setting level
             shift of the real size is applied to BOTH
             the multi-tile and the single-tile rungs    -> P1 and the control
                                                            gate P4 both fail

The rest pins the things that would let a wrong number look right: the defaults
still being in the memory-bound multi-tile window, the fit being invariant to
the bandwidth constant it is quoted against, a numpy bool not being able to hide
a failed gate from the verdict, and the estimator being IMPORTED rather than
re-implemented.

`scripts/group_m_alpha_sweep.py` is loaded by path because `scripts/` is not a
package, the same shape `tests/test_alpha_refit.py` needs.
"""
from __future__ import annotations

import ast
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


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]` and fails inside the decorator otherwise.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GM = _load("group_m_alpha_sweep", "group_m_alpha_sweep.py")
AR = GM.load_alpha_refit()


@pytest.fixture(scope="module")
def default_plan():
    """The plan the pod will actually run, built from the shipped defaults."""
    return GM.build_plan(GM.parse_args([]))


def run_report(argv, tmp_path, monkeypatch, capsys):
    """`main` end to end, with output confined to a temp directory."""
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    code = GM.main(argv)
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------
# the gates can see an effect, and can miss its absence
# --------------------------------------------------------------------------

def test_the_monotone_law_passes_every_gate_and_the_verdict_says_so(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "monotone"], tmp_path, monkeypatch, capsys)
    assert "VERDICT: PREDICTION HELD" in out
    assert "[FAIL]" not in out
    assert code == 0


def test_a_scalar_alpha_fails_the_falling_alpha_gate(tmp_path, monkeypatch, capsys):
    code, out = run_report(["--synthetic", "flat"], tmp_path, monkeypatch, capsys)
    assert "VERDICT: PREDICTION REFUTED" in out
    assert "[FAIL] P1" in out
    assert code == 1


def test_a_launch_order_artefact_fails_the_single_tile_control(
        tmp_path, monkeypatch, capsys):
    """The failure mode the whole control rung exists for.

    Under `order` the time falls by 25% with GROUP_SIZE_M in BOTH rungs, which is
    what an occupancy or wave-quantisation effect looks like. The fit sees
    nothing (a per-setting level is absorbed by the intercept) and the control
    sees everything, which is the pair of signatures that says "not a re-read".
    """
    code, out = run_report(["--synthetic", "order"], tmp_path, monkeypatch, capsys)
    assert "[FAIL] P4" in out
    assert "NOT a weight re-read" in out
    assert code == 1


def test_a_run_with_no_effect_anywhere_leaves_the_control_gate_untestable(
        tmp_path, monkeypatch, capsys):
    """P4 is a RATIO, so two noise floors must not divide into a verdict.

    Under `flat` neither rung moves by more than a tenth of a percent. Calling
    that "the effect is an artefact" would be a confident answer about an effect
    that is not there.
    """
    _, out = run_report(["--synthetic", "flat"], tmp_path, monkeypatch, capsys)
    assert "[NOT TESTABLE] P4" in out


def test_a_synthetic_report_can_never_be_read_as_a_measurement(
        tmp_path, monkeypatch, capsys):
    _, out = run_report(["--synthetic", "monotone"], tmp_path, monkeypatch, capsys)
    assert "*** SYNTHETIC" in out
    directory = next(p for p in tmp_path.rglob("cells.jsonl")).parent
    assert directory.name.endswith("-synthetic-monotone")
    first = json.loads((directory / "cells.jsonl").read_text().splitlines()[0])
    assert first["provenance"] == "synthetic"


def test_replaying_synthetic_rows_still_announces_that_they_are_synthetic(
        tmp_path, monkeypatch, capsys):
    """`--replay` does not carry `--synthetic`, so provenance has to travel.

    Without the check inside the report, a synthetic directory replayed a week
    later prints a clean table with no banner at all and reads as a pod result.
    """
    run_report(["--synthetic", "monotone"], tmp_path, monkeypatch, capsys)
    directory = next(p for p in tmp_path.rglob("cells.jsonl")).parent
    _, out = run_report(["--replay", str(directory)], tmp_path, monkeypatch, capsys)
    assert "*** SYNTHETIC" in out
    assert "nothing here was measured on any hardware" in out


# --------------------------------------------------------------------------
# the verdict cannot disagree with the table above it
# --------------------------------------------------------------------------

def test_a_numpy_false_is_a_failed_gate_and_not_a_passing_one():
    """The bug this pins: `numpy.bool_(False) is False` is False.

    `fit_alpha` answers in numpy, so every comparison built from a fitted alpha
    was a `numpy.bool_`. The gate printed "[FAIL]" and the `g.ok is False` scan
    that decides the verdict skipped it, so one run of this script reported
    PREDICTION HELD directly underneath a failed gate.
    """
    gate = GM.Gate("planted", np.bool_(False), "detail")
    assert gate.ok is False
    assert gate.label == "FAIL"
    assert GM.verdict(lambda *a: None, [gate]) == 1


def test_a_gate_with_no_evidence_is_not_a_refutation():
    assert GM.verdict(lambda *a: None, [GM.Gate("x", None, "no data")]) == 4


# --------------------------------------------------------------------------
# the estimator is imported, never re-implemented
# --------------------------------------------------------------------------

def test_this_script_defines_no_estimator_of_its_own():
    """Two estimators that can disagree would make every alpha here unquotable.

    Checked in the SOURCE rather than by calling, because the failure being
    guarded against is a future edit adding a local fit "just for the control
    arm", which would still pass every behavioural test in this file.
    """
    tree = ast.parse((ROOT / "scripts" / "group_m_alpha_sweep.py").read_text())
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)}
    assert "fit_alpha" not in defined
    assert "_within_group_ssr" not in defined
    assert GM.load_alpha_refit().fit_alpha.__module__ == "alpha_refit"


def test_a_missing_estimator_refuses_rather_than_falling_back(tmp_path):
    with pytest.raises(GM.EstimatorMissing):
        GM.load_alpha_refit(tmp_path / "there-is-no-such-script.py")


def test_a_file_that_is_not_the_estimator_is_refused_by_name(tmp_path):
    impostor = tmp_path / "alpha_refit.py"
    impostor.write_text("def fit_alpha(rows):\n    return 0.42\n")
    with pytest.raises(GM.EstimatorMissing):
        GM.load_alpha_refit(impostor)


def test_the_band_quantile_matches_the_estimators_own_percentile():
    """`band` uses numpy and `alpha_refit.bootstrap_band` uses its own helper.

    They must be the same convention, or two intervals in two reports about the
    same parameter would mean different things.
    """
    ordered = sorted(random.Random(0).random() for _ in range(37))
    for q in (0.05, 0.5, 0.95):
        assert AR._percentile(ordered, q) == pytest.approx(
            float(np.quantile(ordered, q)))


# --------------------------------------------------------------------------
# the arithmetic that decides what is being measured
# --------------------------------------------------------------------------

def test_the_tile_count_is_the_exact_ceiling_sum_and_not_a_reconstruction(
        default_plan):
    """The published pool cannot do this, which is half the reason for a run.

    `crossing.m_tiles_for_row` divides total rows by a stored tile efficiency
    and has to REFUSE once an expert spans several tiles at a block size the
    schema does not store -- the exact regime this sweep lives in. Holding the
    histogram makes the count exact.
    """
    for cell in default_plan.cells[:40]:
        expected = sum(-(-c // cell.block_m) for c in cell.counts)
        assert cell.m_tiles == pytest.approx(expected)


def test_a_cell_is_a_single_tile_control_by_its_realisation_not_its_mean(
        default_plan):
    """A uniform draw at 8 mean rows per expert still puts 14 on one of them.

    A rung classified by its mean would call a cell with a two-tile expert a
    control, and the control is the only thing separating an L2 claim from a
    launch-order one.
    """
    for cell in default_plan.cells:
        assert cell.single_tile == (cell.max_rows <= cell.block_m)


def test_the_activation_confound_is_bounded_by_block_m_over_block_n(default_plan):
    """An extra M-tile costs one pass over the expert's weights (N*K) and one
    activation re-read per N-tile (BLOCK_M*K each). Everything cancels but the
    tile aspect ratio."""
    assert GM.confound_bound(default_plan) == pytest.approx(
        default_plan.block_m / default_plan.fixed_tile["BLOCK_SIZE_N"])
    assert GM.confound_bound(default_plan) <= 0.25


def test_the_span_costed_here_is_the_span_vllm_fused_experts_covers():
    """`VLLM_COVERS` restates a tuple that cannot be imported without vLLM.

    Read out of the baseline's source instead, so a change upstream in this repo
    breaks the test rather than silently costing a different set of stages.
    """
    tree = ast.parse((ROOT / "moe" / "baselines" / "vllm_fused_moe.py").read_text())
    covers = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VllmFusedExperts":
            for statement in node.body:
                if (isinstance(statement, ast.Assign)
                        and getattr(statement.targets[0], "id", "") == "covers"):
                    covers = tuple(ast.literal_eval(statement.value))
    assert covers is not None, "VllmFusedExperts.covers moved"
    assert GM.VLLM_COVERS == covers


# --------------------------------------------------------------------------
# the fit itself
# --------------------------------------------------------------------------

def _planted(alpha: float, xs_by_tokens: dict, noise: float = 0.0, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for tokens, xs in xs_by_tokens.items():
        level = rng.uniform(0.5, 2.0)
        for index, x in enumerate(xs):
            out.append(AR.Observation(
                traffic_ratio=level * (1 + alpha * x) * math.exp(rng.gauss(0, noise)),
                compulsory_bytes=1.0, per_expert_bytes=0.1,
                active_experts=8.0, m_tiles=8.0 + x / 0.1, block_m=16, group_m=1,
                tile_provenance="planted", model="mixtral-8x7b", dtype="bf16",
                gpu="test", impl=GM.VLLM_IMPL, tokens=tokens,
                routing=f"r{index}", l2_flush=False, cuda_graph=False,
                tile_columns=()))
    return out


LADDER = {64: [0.3, 0.5, 0.8], 128: [1.4, 1.7, 2.0], 256: [3.3, 3.7, 4.1],
          448: [6.4, 6.9, 7.4]}


def test_the_band_brackets_an_alpha_that_was_planted_in_the_data():
    rows = _planted(0.558, LADDER, noise=0.005, seed=3)
    interval = GM.band(AR, rows, draws=200, seed=0)
    assert interval is not None
    assert interval[0] < 0.558 < interval[1]


def test_the_band_resamples_routing_realisations_and_not_the_intercept_group():
    """`alpha_refit.bootstrap_band` returns nothing here, and correctly so.

    Its cluster is the intercept group because in the published pool the rows
    inside one are replicates of a single thermal state. In this design they are
    different routing realisations, so the unit has to move down a level or the
    band is drawn over four clusters and means nothing.
    """
    rows = _planted(0.558, {512: [7.0, 7.5, 8.0, 8.5]}, noise=0.005, seed=1)
    assert AR.bootstrap_band(rows, 50, 0) is None      # one intercept group
    assert GM.band(AR, rows, draws=50, seed=0) is not None


def test_the_fitted_alpha_does_not_depend_on_the_bandwidth_it_is_quoted_against():
    """A constant on every ratio is an additive constant in logs, and the group
    intercept absorbs it exactly. So a pod with no calibration on disk still
    fits the same alpha; only the RATIO's printed level moves."""
    cell = GM.build_cell("mixtral-8x7b", 448, "bf16", "zipf:1.0", 0, 16)
    other = GM.build_cell("mixtral-8x7b", 448, "bf16", "hot:0.4", 1, 16)
    third = GM.build_cell("mixtral-8x7b", 128, "bf16", "uniform", 0, 16)
    fourth = GM.build_cell("mixtral-8x7b", 128, "bf16", "hot:0.4", 2, 16)
    fits = []
    for bandwidth in (4.8e12, 3.1e12):
        rows = [GM.observation(AR, c, 5.0 + i, 1, "test", bandwidth, True)
                for i, c in enumerate((cell, other, third, fourth))]
        fits.append(AR.fit_alpha(rows))
    assert fits[0] == pytest.approx(fits[1], abs=1e-9)


def test_permuting_the_response_inside_a_group_collapses_the_fit():
    rows = _planted(0.558, LADDER, noise=0.005, seed=5)
    assert AR.fit_alpha(rows) > 0.4
    assert abs(GM.placebo_alpha(AR, rows, seed=0)) <= GM.PLACEBO_MAX_ALPHA


# --------------------------------------------------------------------------
# the design, which is what the pod session is spent on
# --------------------------------------------------------------------------

def test_no_cell_in_the_shipped_plan_is_compute_bound(default_plan):
    """The trap that would produce a clean, confident, wrong flat line.

    Above the ridge the time is set by padded ARITHMETIC, which GROUP_SIZE_M
    cannot move, so a compute-bound design reports alpha flat across the ladder
    and looks exactly like a refutation of the mechanism.
    """
    limit = GM.MEMORY_BOUND_MARGIN * AR.RIDGE_BAND[0]
    worst = max(c.arith_intensity for c in default_plan.cells)
    assert worst <= limit, f"worst compulsory AI {worst:.1f} exceeds {limit:.1f}"


def test_the_shipped_plan_has_both_a_multi_tile_rung_and_a_single_tile_control(
        default_plan):
    assert len(default_plan.multi) >= GM.MIN_DISCRIMINATING
    assert len(default_plan.control) >= 3
    assert max(c.tiles_per_expert for c in default_plan.multi) >= 4.0


def test_the_predicted_floor_sits_inside_the_swept_ladder(default_plan):
    """P3 is only a test if the knee is a setting the sweep visits."""
    import statistics
    knee = statistics.median([c.tiles_per_expert for c in default_plan.multi])
    assert min(default_plan.group_m) < knee < max(default_plan.group_m)


def test_every_preflight_gate_passes_on_the_shipped_defaults(default_plan):
    gates = GM.preflight(default_plan, AR.RIDGE_BAND[0])
    failed = [g.name for g in gates if g.ok is not True]
    assert not failed, f"the shipped design fails its own preflight: {failed}"


def test_a_compute_bound_design_is_refused_before_anything_is_spent(
        tmp_path, monkeypatch, capsys):
    code, out = run_report(["--tokens", "4096", "--seeds", "1",
                            "--routings", "uniform"], tmp_path, monkeypatch, capsys)
    assert "[FAIL] regime: every cell is memory bound" in out
    assert "refused before spending anything" in out
    assert code == 1


def test_one_token_count_cannot_identify_alpha_and_the_ladder_can(default_plan):
    """Why the batch is a ladder even though the brief said fixed batch.

    One token count is one intercept, so its LEVEL is absorbed exactly and only
    the curvature of log(1 + alpha x) is left. One rung supplies almost none of
    it.
    """
    top = max(default_plan.tokens)
    one_rung = GM.Plan(**{**default_plan.__dict__, "tokens": (top,),
                          "cells": tuple(c for c in default_plan.multi
                                         if c.tokens == top)})
    ladder = GM.power_band(AR, default_plan, 0.558, GM.POWER_NOISE, 200, 0)
    single = GM.power_band(AR, one_rung, 0.558, GM.POWER_NOISE, 200, 0)
    assert ladder is not None and single is not None
    assert (ladder[1] - ladder[0]) <= GM.PUBLISHED_GROUP_M_EFFECT
    assert (single[1] - single[0]) > 4 * (ladder[1] - ladder[0])


# --------------------------------------------------------------------------
# surviving a pod: resume, abort, and a message when there is no GPU
# --------------------------------------------------------------------------

def test_a_run_without_a_gpu_says_so_and_exits_three(tmp_path, monkeypatch, capsys):
    code, out = run_report(["--run"], tmp_path, monkeypatch, capsys)
    assert "CANNOT RUN HERE" in out
    assert code == 3


def test_the_override_probe_blames_an_absent_vllm_rather_than_the_hook():
    """Two different failures, two different fixes, and only one of them is
    "your vLLM is too new"."""
    pytest.importorskip("torch")
    try:
        import vllm  # noqa: F401
    except ImportError:
        with pytest.raises(GM.CannotRunHere, match="not importable"):
            GM.find_override_config()


def test_a_resumed_run_skips_the_cells_already_on_disk(default_plan, tmp_path):
    records = GM.synthesise(default_plan, "monotone", seed=0)[:10]
    path = tmp_path / "cells.jsonl"
    for record in records:
        GM._append(path, record)
    done = {r["id"] for r in GM.read_records(path)}
    assert len(done) == 10
    cell, group_m = default_plan.cells[0], default_plan.group_m[0]
    assert GM._record_id(cell, group_m, 0) in done


def test_a_run_killed_mid_write_still_replays(tmp_path, default_plan):
    """Every cell is flushed and fsynced as it is measured, so an abort costs
    one cell; the cost of that is a half-written last line."""
    path = tmp_path / "cells.jsonl"
    for record in GM.synthesise(default_plan, "monotone", seed=0)[:5]:
        GM._append(path, record)
    with path.open("a") as handle:
        handle.write('{"kind": "cell", "id": "truncated"')
    assert len(GM.read_records(path)) == 5


def test_records_from_another_plan_are_ignored_and_named(
        tmp_path, monkeypatch, capsys):
    """A `--replay` with different flags must not silently report on a subset."""
    run_report(["--synthetic", "monotone"], tmp_path, monkeypatch, capsys)
    directory = next(p for p in tmp_path.rglob("cells.jsonl")).parent
    _, out = run_report(["--replay", str(directory), "--model", "qwen2-57b-a14b"],
                        tmp_path, monkeypatch, capsys)
    assert "records name a cell this plan does not contain" in out


def test_the_output_directory_prefers_the_env_then_the_volume_then_the_repo(
        tmp_path, monkeypatch):
    """The pod's container disk dies with the pod and the network volume does
    not, so this is not a cosmetic preference."""
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path / "chosen"))
    assert GM.results_root() == tmp_path / "chosen"
    monkeypatch.delenv("MOE_RESULTS_DIR")
    monkeypatch.setenv("WORKSPACE", str(tmp_path / "volume"))
    assert GM.results_root() == ROOT / "results"
    (tmp_path / "volume").mkdir()
    assert GM.results_root() == tmp_path / "volume" / "results"


def test_the_paired_table_drops_a_cell_that_never_reached_the_reference_setting(
        default_plan):
    """An aborted run leaves cells timed at some settings and not others.

    Averaging those unpaired would compare a different set of cells at each
    setting and read the composition change as a GROUP_SIZE_M effect.
    """
    cells = {c.key: c for c in default_plan.cells}
    multi = next(c for c in default_plan.cells if not c.single_tile)
    records = [
        GM._record(multi, 1, 0, default_plan, 10.0),
        GM._record(multi, 8, 0, default_plan, 5.0),
    ]
    orphan = next(c for c in default_plan.cells
                  if not c.single_tile and c.key != multi.key)
    records.append(GM._record(orphan, 8, 0, default_plan, 1.0))
    ratios = GM.paired_ratios(records, cells, 1, want_single_tile=False)
    assert ratios[8] == pytest.approx(0.5)


def test_a_swizzle_that_changed_the_answer_fails_the_correctness_gate(default_plan):
    """GROUP_SIZE_M reorders program ids and must return the identical tensor.

    Nothing in a timing table would show otherwise: an illegal-for-the-shape
    forced config still produces numbers, and if it were faster the report would
    read it as a win.
    """
    records = GM.synthesise(default_plan, "monotone", seed=0)[:20]
    assert GM.swizzle_integrity_gate(records).ok is True
    records[7]["matches_reference"] = False
    records[7]["max_abs_diff"] = 0.25
    gate = GM.swizzle_integrity_gate(records)
    assert gate.ok is False
    assert "differed" in gate.detail


def test_an_in_place_fused_experts_is_reported_as_such(default_plan):
    """It would let every later setting time a decayed input, and the only
    symptom would be that later settings look different."""
    records = GM.synthesise(default_plan, "monotone", seed=0)[:20]
    records[3]["input_unchanged"] = False
    gate = GM.swizzle_integrity_gate(records)
    assert gate.ok is False
    assert "mutated in place" in gate.detail


def test_rows_that_predate_the_correctness_check_answer_not_measured(default_plan):
    records = GM.synthesise(default_plan, "monotone", seed=0)[:5]
    for record in records:
        record.pop("matches_reference")
    assert GM.swizzle_integrity_gate(records).ok is None
