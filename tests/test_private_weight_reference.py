"""The private-weight reference arm: its scorer, its refusals and its contract.

WHAT THIS FILE IS FOR. `scripts/private_weight_reference.py` measures alpha as
a ratio of two measured slopes, so the only things that can go wrong off GPU
are the arithmetic, the partition, the relabelling and the gates -- and every
one of those is pure. Nothing here needs a device; everything here is what the
pod run will execute.
"""
from __future__ import annotations

import ast
import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import private_weight_reference as PW  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.bench import weights as WEIGHTS  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

SCRIPT = ROOT / "scripts" / "private_weight_reference.py"
CFG = MODEL_CONFIGS[PW.DEFAULT_MODEL]

#: The modes this file drives the whole script through, and what each must do.
#: `--dry-run` is a PLANNING mode and scores no gate; every other row is a
#: SCORING mode and must print RESULT lines its own exit code can be recomputed
#: from.
OFF_GPU_MODES: tuple[tuple[tuple[str, ...], bool], ...] = (
    (("--dry-run", "--device-memory-gb", "140"), False),
    (("--self-test", "refit"), True),
    (("--self-test", "no-reuse"), True),
    (("--self-test", "issue-bound"), True),
    (("--self-test", "aliased"), True),
    (("--self-test", "machinery"), True),
    (("--self-test", "compute-bound"), True),
    (("--self-test", "noisy-identity"), True),
    (("--self-test", "over-allocated"), True),
    (("--self-test", "holes"), True),
    (("--self-test", "ragged"), True),
)


def run(args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=900,
                          cwd=str(ROOT))


# --------------------------------------------------------------------------
# 1. the exit-code contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("args,scoring", OFF_GPU_MODES)
def test_the_log_recomputes_the_code_the_process_returned(args, scoring):
    """The property, over EVERY off-GPU mode. A script that prints one thing
    and exits another is the defect `moe/bench/exit_codes.py` exists to
    prevent, and it is the only thing the session driver can see."""
    got = run(args)
    lines = exit_codes.parse_result_lines(got.stdout)
    if scoring:
        assert lines, (args, got.stdout[-2000:])
        # A planted world that came out other than registered exits ERROR,
        # which is deliberately OUTSIDE the gate table: the apparatus is
        # broken and no verdict on the page means anything.
        if got.returncode == exit_codes.ERROR:
            assert "SELF-TEST MISMATCH" in got.stdout
        else:
            assert exit_codes.classify_text(got.stdout) == got.returncode, (
                args, got.returncode, got.stdout[-2000:])
    else:
        assert not lines, "a planning mode scored a gate"
        assert got.returncode == exit_codes.REFUSED


def test_every_gate_prints_exactly_one_result_line_and_nothing_else_does():
    got = run(["--self-test", "refit"])
    parsed = exit_codes.parse_result_lines(got.stdout)
    raw = [ln for ln in got.stdout.splitlines() if ln.startswith("RESULT: ")]
    assert len(raw) == len(parsed), "a RESULT line the parser cannot read back"
    assert [r.name for r in parsed] == ["V0", "V1", "V2", "V3", "V4", "V5",
                                        "V6", "C1", "C2"]


def test_a_dry_run_is_refused_and_not_done():
    """REFUSED (2) and not DONE (0). A plan scores no gate, so a log of it has
    no RESULT line, and `classify_text` over it raises `NoGatesScored` -- which
    is what a REFUSED log looks like from there. DONE says "measured; every
    gate PASSED", and a plan measured nothing."""
    got = run(["--dry-run", "--device-memory-gb", "140"])
    assert got.returncode == exit_codes.REFUSED
    assert "RESULT: " not in got.stdout
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(got.stdout)
    assert "estimated GPU time" in got.stdout


def test_an_unplanned_crash_is_error_and_not_claim_fail(monkeypatch):
    """Left to propagate, an unexpected exception exits the interpreter ONE,
    and ONE is CLAIM_FAIL, which is in FINISHED_CODES: the driver would file
    the arm as finished and never re-run it."""
    def boom(_argv=None):
        raise RuntimeError("planted")

    monkeypatch.setattr(PW, "_main", boom)
    rc = PW.main([])
    assert rc == exit_codes.ERROR
    assert rc != exit_codes.CLAIM_FAIL
    assert rc not in exit_codes.FINISHED_CODES
    assert exit_codes.ledger_state(rc) == "RETRY"


# --------------------------------------------------------------------------
# 2. the planted worlds, and that every gate can fail
# --------------------------------------------------------------------------

def test_every_planted_world_returns_its_registration():
    """A self-test that asserts nothing is a smoke test. Each world registers
    the verdict every named gate must return, and a mismatch is ERROR."""
    for name in sorted(PW.WORLDS):
        got = run(["--self-test", name])
        assert "SELF-TEST MISMATCH" not in got.stdout, (name, got.stdout[-3000:])
        assert "SELF-TEST OK" in got.stdout, (name, got.stdout[-3000:])
        assert got.returncode != exit_codes.ERROR, name


def test_the_worlds_separate_into_three_exit_codes():
    """A set of planted worlds that all return the same code has not shown that
    the scorer discriminates anything."""
    codes = {name: run(["--self-test", name]).returncode
             for name in sorted(PW.WORLDS)}
    assert codes["refit"] == exit_codes.DONE
    assert codes["no-reuse"] == exit_codes.CLAIM_FAIL
    assert codes["issue-bound"] == exit_codes.CLAIM_FAIL
    assert codes["aliased"] == exit_codes.INVALID
    assert len(set(codes.values())) == 3, codes


def test_every_gate_has_a_world_that_fails_it():
    """A gate that cannot fail is as useless as one that cannot pass. The set
    of tags that FAIL somewhere is COUNTED from the worlds rather than listed
    here, so a gate added without a world to fail it turns this red."""
    failing: set[str] = set()
    for name in sorted(PW.WORLDS):
        for line in exit_codes.parse_result_lines(run(["--self-test", name]).stdout):
            if line.verdict == exit_codes.FAIL:
                failing.add(line.name)
    every = {line.name for line in
             exit_codes.parse_result_lines(run(["--self-test", "refit"]).stdout)}
    assert failing == every, (
        f"no planted world reaches the FAIL branch of {sorted(every - failing)}")


def test_the_registrations_name_gates_the_report_actually_has():
    """A registration that silently matches nothing is the
    check-that-examined-nothing shape one level up."""
    tags = {line.name for line in
            exit_codes.parse_result_lines(run(["--self-test", "refit"]).stdout)}
    for name, world in sorted(PW.WORLDS.items()):
        assert set(world.expect) <= tags, (name, set(world.expect) - tags)
        assert world.expect, name


# --------------------------------------------------------------------------
# 3. the outcome partition
# --------------------------------------------------------------------------

def test_the_partition_tiles_the_line_with_no_gap_and_no_overlap():
    assert PW.partition_is_total() == ""
    edges = [lo for _n, lo, _hi, _m in PW.OUTCOMES]
    assert edges == sorted(edges)
    assert PW.OUTCOMES[0][1] == 0.0
    assert PW.OUTCOMES[-1][2] == math.inf


def test_the_partition_check_can_fail(monkeypatch):
    """Planted both ways: a check for a condition that is always true passes
    forever."""
    broken = ((*PW.OUTCOMES[0][:2], 0.2, "x"),) + PW.OUTCOMES[1:]
    monkeypatch.setattr(PW, "OUTCOMES", broken)
    assert PW.partition_is_total() != ""


def test_outcome_for_names_every_registered_world():
    assert PW.outcome_for(0.0)[0] == "ISSUE-AND-LATENCY"
    assert PW.outcome_for(PW.RETRACTED_ALPHA)[0] == "ISSUE-AND-LATENCY"
    assert PW.outcome_for(PW.ALPHA)[0] == "REFIT-CONFIRMED"
    assert PW.outcome_for(1.0)[0] == "NO-REUSE"
    assert PW.outcome_for(1e9)[0] == "NO-REUSE"
    # The states that are not bands: a negative ratio is a ladder that got
    # faster with another M-tile, not a small fraction.
    assert PW.outcome_for(-0.2)[0] == "DESCENDING"
    assert PW.outcome_for(math.nan)[0] == "NOT-A-RATIO"


def test_the_band_edges_are_the_studys_own_and_not_a_second_copy():
    """ALPHA_BAND is IMPORTED. A local copy that drifted would put the
    prediction page and the study in different worlds while both printed a
    confident table."""
    import block_m_crossing_sweep as SWEEP
    assert PW.ALPHA_BAND is SWEEP.ALPHA_BAND
    names = [n for n, _lo, _hi, _m in PW.OUTCOMES]
    band = PW.OUTCOMES[names.index("REFIT-CONFIRMED")]
    assert (band[1], band[2]) == SWEEP.ALPHA_BAND


# --------------------------------------------------------------------------
# 4. the relabelling, which is the whole of the private arm's mechanism
# --------------------------------------------------------------------------

def balanced_flat_ids(num_experts: int, rows: int, top_k: int, seed: int = 0):
    import torch
    slots = []
    for e in range(num_experts):
        slots += [e] * rows
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(slots), generator=g)
    flat = torch.tensor(slots, dtype=torch.int32)[perm]
    return flat.reshape(-1, top_k)


@pytest.mark.parametrize("block_m,tiles", [(32, 1), (32, 2), (32, 6), (64, 3)])
def test_every_new_expert_holds_exactly_one_full_m_tile(block_m, tiles):
    """THE CLAIM THE PRIVATE ARM RESTS ON. Each relabelled expert must hold
    exactly BLOCK_M rows, so `moe_align_block_size` builds the SAME number of
    M-tiles as the shared arm with zero padding, and each carries a distinct
    expert index into a distinct weight copy."""
    import torch

    e, k = CFG.num_experts, CFG.top_k
    rows = tiles * block_m
    ids = balanced_flat_ids(e, rows, k)
    out = PW.private_topk_ids(ids, e, block_m, rows)
    counts = torch.bincount(out.reshape(-1).long(),
                            minlength=PW.expert_space(e, tiles))
    assert counts.numel() == PW.expert_space(e, tiles)
    assert int(counts.min()) == block_m and int(counts.max()) == block_m
    # And the relabelling is a pure renaming: the original expert survives as
    # the residue, so a row still reads the weights of its own expert.
    assert bool(((out.reshape(-1).long() % e) == ids.reshape(-1).long()).all())
    assert out.dtype == ids.dtype


def test_the_identity_tread_relabels_to_itself():
    """At one M-tile per expert the private arm IS the shared arm, which is the
    whole content of V6: there is nothing for the three arms to differ by."""
    e, k = CFG.num_experts, CFG.top_k
    ids = balanced_flat_ids(e, PW.DEFAULT_BLOCK_M, k)
    out = PW.private_topk_ids(ids, e, PW.DEFAULT_BLOCK_M, PW.DEFAULT_BLOCK_M)
    assert bool((out == ids).all())


def test_an_imbalanced_histogram_is_refused_and_not_rounded():
    """A histogram off by one row would put one more copy under one expert than
    was allocated, and the kernel would read past the copies that exist."""
    import torch

    e, k = CFG.num_experts, CFG.top_k
    ids = balanced_flat_ids(e, 64, k).clone()
    flat = ids.reshape(-1)
    # Move one slot from expert 0 to expert 1: the total is unchanged and the
    # shapes still line up, which is exactly why this has to be checked.
    first = int(torch.nonzero(flat == 0)[0])
    flat[first] = 1
    with pytest.raises(PW.PrivateWeightRefusal):
        PW.private_topk_ids(ids, e, 32, 64)


def test_a_partly_filled_stack_is_refused():
    e, k = CFG.num_experts, CFG.top_k
    ids = balanced_flat_ids(e, 48, k)
    with pytest.raises(PW.PrivateWeightRefusal):
        PW.private_topk_ids(ids, e, 32, 48)


def test_the_copy_index_is_the_rank_over_the_tile_height():
    assert PW.private_copy_index(0, 32) == 0
    assert PW.private_copy_index(31, 32) == 0
    assert PW.private_copy_index(32, 32) == 1
    assert PW.private_copy_index(191, 32) == 5


# --------------------------------------------------------------------------
# 5. the byte counts, asserted as RELATIONS and never as literals
# --------------------------------------------------------------------------

def test_the_weight_bill_is_the_specs_own_byte_count_times_the_copy_count():
    """A QUANTITY DERIVED FROM A SPEC IS NOT RE-MULTIPLIED HERE. The per-copy
    figure comes from `moe.bench.weights`, which owns the one multiplication,
    and this asserts the RELATION between the two rather than a number that
    would go stale the day a geometry changed."""
    for model in ("mixtral-8x7b", "deepseek-v2-lite", "qwen2-57b-a14b"):
        cfg = MODEL_CONFIGS[model]
        one = WEIGHTS.routed_expert_weight_bytes(cfg, "bf16")
        assert PW.weight_bytes_total(cfg, "bf16", 1) == one
        for copies in (2, 6, 17):
            assert PW.weight_bytes_total(cfg, "bf16", copies) == copies * one


#: A flush buffer of a stated size, so the arithmetic below does not depend on
#: whether the machine running the tests has a device to ask.
PINNED_FLUSH = (256 * 2 ** 20, "pinned by the test suite")


def test_the_memory_plan_fits_or_refuses_against_the_size_it_was_given():
    """`fits` is None when nothing was asked, which is NOT the same answer as
    True: a plan checked against no card is a plan checked against nothing."""
    def plan_for(free, source):
        return PW.memory_plan(CFG, "bf16", 2, 6, 768, free, source,
                              flush=PINNED_FLUSH)

    plan = plan_for(None, "no device")
    assert plan.fits is None
    assert "NOT CHECKED" in "\n".join(plan.lines())
    assert plan_for(140 * 10 ** 9, "hypothetical").fits is True
    small = plan_for(8 * 10 ** 9, "hypothetical")
    assert small.fits is False
    # The relation, not a literal: the predicted peak is the weight bill, the
    # stated allowance on the modelled activation set, and the instrument's own
    # flush buffer, which is a NAMED term and not part of the allowance.
    assert small.predicted_peak_bytes == int(
        small.weight_bytes + small.allowance * small.activation_bytes
        + small.flush_bytes)
    assert small.flush_bytes == PINNED_FLUSH[0]


def test_the_flush_buffer_is_asked_of_the_instrument_and_not_restated():
    """It is 4x this device's L2, the same order as every activation buffer in
    a cell put together. `timing.flush_mb_for_device` owns the rule; this
    asserts the RELATION to it rather than a number that would go stale."""
    size, source = PW.flush_buffer_bytes()
    try:
        from moe.bench import timing
    except Exception:                                     # noqa: BLE001
        pytest.skip("moe.bench.timing does not import here")
    assert size == timing.flush_mb_for_device() * 2 ** 20
    assert "flush_mb_for_device" in source


def test_a_plan_that_does_not_fit_the_named_card_is_refused_before_it_runs():
    got = run(["--dry-run", "--device-memory-gb", "8"])
    assert got.returncode == exit_codes.REFUSED
    assert "do not fit this card" in got.stdout
    assert "RESULT: " not in got.stdout


# --------------------------------------------------------------------------
# 6. the depth arithmetic, which is what chose the default tile
# --------------------------------------------------------------------------

def test_a_taller_tile_runs_out_of_memory_bound_treads_sooner():
    """THE REASON THE DEFAULT TILE IS 32 AND NOT 64. The compute an M-tile does
    scales with the tile height and the weight traffic an extra M-tile costs
    does not, so a LOW-alpha ladder climbs to its roof faster on a taller tile.
    Asserted as an ORDERING, so it survives any ridge this repo calibrates."""
    ridge, bw = 160.0, 4000.0
    depths = [PW.deepest_memory_bound_tread(
        CFG, block_m=bm, alpha=PW.RETRACTED_ALPHA, ridge=ridge,
        bandwidth_gbps=bw, b=2, limit=128) for bm in (16, 32, 64)]
    assert depths[0] > depths[1] > depths[2], depths
    # And in the refit world every one of them is deep enough that the tile
    # choice is not what decides the design.
    rich = [PW.deepest_memory_bound_tread(
        CFG, block_m=bm, alpha=PW.ALPHA, ridge=ridge, bandwidth_gbps=bw, b=2,
        limit=128) for bm in (16, 32, 64)]
    assert all(d >= PW.DEFAULT_TREADS for d in rich), rich


def test_the_discrimination_floor_is_below_the_world_the_default_registers():
    """The design must be able to REPORT its own registered alternative. Below
    the floor the shared ladder is compute bound before the deepest tread, V4
    voids the page, and the pod minutes buy nothing."""
    ridge, bw = 160.0, 4000.0
    floor = PW.discrimination_floor(CFG, block_m=PW.DEFAULT_BLOCK_M,
                                    treads=PW.DEFAULT_TREADS, ridge=ridge,
                                    bandwidth_gbps=bw, b=2)
    assert floor < PW.RETRACTED_ALPHA, floor
    assert PW.depth_refusal(CFG, block_m=PW.DEFAULT_BLOCK_M,
                            treads=PW.DEFAULT_TREADS, ridge=ridge,
                            bandwidth_gbps=bw, b=2) == ""
    # And it FAILS where it should: the tall tile at the same depth cannot
    # report the retracted world at all.
    why = PW.depth_refusal(CFG, block_m=64, treads=PW.DEFAULT_TREADS,
                           ridge=ridge, bandwidth_gbps=bw, b=2)
    assert why and "discrimination floor" in why


def test_the_depth_refusal_reaches_the_command_line():
    got = run(["--dry-run", "--block-m", "64", "--device-memory-gb", "140"])
    assert got.returncode == exit_codes.REFUSED
    assert "cannot separate the worlds it registers" in got.stdout


# --------------------------------------------------------------------------
# 7. the control this design cannot do without
# --------------------------------------------------------------------------

def test_a_model_with_no_identity_tread_is_refused_before_a_pod_is_rented():
    """deepseek-v2-lite routes k=6 over E=64, so rows per expert must be a
    multiple of 3 and `r = 1 x BLOCK_M` is never one at a power-of-two tile.
    The n=1 tread is V6's whole subject, so a model without one has no
    instrument floor at any depth -- which is a REFUSAL (free) and not an
    INVALID (the arm's minutes, spent)."""
    assert PW.identity_tread_refusal(MODEL_CONFIGS["deepseek-v2-lite"], 32)
    assert PW.identity_tread_refusal(MODEL_CONFIGS[PW.DEFAULT_MODEL], 32) == ""
    got = run(["--dry-run", "--model", "deepseek-v2-lite"])
    assert got.returncode == exit_codes.REFUSED
    assert "n = 1 tread" in got.stdout
    assert "RESULT: " not in got.stdout


def test_the_default_model_can_form_the_identity_tread_at_the_default_tile():
    """Asserted as the property, not as a remembered `rows_quantum` of 1."""
    import block_m_crossing_sweep as SWEEP
    assert PW.DEFAULT_BLOCK_M % SWEEP.rows_quantum(CFG) == 0
    assert PW.ladder_treads(CFG, PW.DEFAULT_BLOCK_M, PW.DEFAULT_TREADS)[0] == 1


# --------------------------------------------------------------------------
# 8. the clock rule: DRIFT excludes, NEITHER side does
# --------------------------------------------------------------------------

PLANTED_REFERENCE_MHZ = 1485.0
H200_MEMORY_LOAD_MHZ = 1980.0
SAGGED_MHZ = 1395.0


def _sample(arm, n, rep, ms, *, level_ok=None, side="", drift_ok=None,
            load=None):
    return PW.Sample(
        arm=arm, repeat=rep, block_m=32, tiles=n, rows_per_expert=n * 32,
        tokens=n * 128, copies=(1 if arm == PW.SHARED else n),
        experts_declared=(CFG.num_experts if arm == PW.SHARED
                          else PW.expert_space(CFG.num_experts, n)),
        ms_p50=ms, ms_min=ms, ms_stdev=0.0, iters=100, trials=3,
        warmup_ms=300.0, instrument="planted",
        sm_clock_load_mhz=load, clock_level_ok=level_ok,
        clock_level_side=side, clock_drift_ok=drift_ok, l2_flush=True)


def test_five_planted_clock_rows_travel_through_the_csv_and_are_counted_once(
        tmp_path):
    """FIVE ROWS, NOT FOUR. Level, HIGH, LOW, DRIFT, and HIGH-AND-DRIFT in one
    row: every earlier world in this repository had its drifting row sitting
    level, so a counter filtering on LEVEL alone counted that row twice and
    stayed green. Both sides of a LEVEL failure are KEPT and recorded; DRIFT
    alone excludes."""
    rows = [
        _sample(PW.SHARED, 1, 0, 1.0, load=PLANTED_REFERENCE_MHZ),
        _sample(PW.SHARED, 2, 0, 2.0, level_ok=False, side="high",
                load=H200_MEMORY_LOAD_MHZ, drift_ok=True),
        _sample(PW.SHARED, 3, 0, 3.0, level_ok=False, side="low",
                load=SAGGED_MHZ, drift_ok=True),
        _sample(PW.SHARED, 4, 0, 4.0, drift_ok=False,
                load=PLANTED_REFERENCE_MHZ),
        _sample(PW.SHARED, 5, 0, 5.0, level_ok=False, side="high",
                drift_ok=False, load=H200_MEMORY_LOAD_MHZ),
    ]
    path = tmp_path / "cells.csv"
    store = PW.Store(path, PW.CSV_FIELDS)
    for row in rows:
        store.append(row)
    back = PW.read_samples(path)
    assert len(back) == 5
    kept = [s for s in back if s.usable]
    assert [s.tiles for s in kept] == [1, 2, 3], (
        "a LEVEL side excluded a row; only DRIFT may")
    excluded = [s for s in back if s.excluded]
    assert [s.tiles for s in excluded] == [4, 5]
    assert sum(1 for s in back if s.excluded) == 2, "a row counted twice"
    # None means NOT DETERMINED and never fine: a row with no clock is kept
    # and counted, not thrown away.
    assert back[0].clock_drift_ok is None and back[0].usable


def test_a_failed_level_with_no_side_is_refused_at_construction():
    """The instrument derives the verdict FROM the side, so a failed LEVEL with
    a blank side is a caller that dropped the column; read as LOW it drops
    every boosted row and read as HIGH it admits a sagged card."""
    with pytest.raises(ValueError):
        _sample(PW.SHARED, 1, 0, 1.0, level_ok=False, side="")


def test_only_the_drift_predicate_decides_membership():
    """AST over the file: the ONE place `clock_drift_ok is False` is read. A
    second reader is how a rule comes to be applied at one of two call sites,
    which is this repository's standing defect."""
    tree = ast.parse(SCRIPT.read_text())
    readers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = "\n".join(
                ast.unparse(s) for s in node.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)))
            if "clock_drift_ok is False" in body:
                readers.add(node.name)
    assert readers == {"excluded"}, readers


def test_the_script_reads_the_level_side_it_records():
    """The repo-wide tripwire's own condition, asserted here too so this file
    fails with the reason rather than `tests/test_shell_gates.py` failing with
    a list."""
    text = SCRIPT.read_text()
    assert "clock_level_ok" in text and "clock_level_side" in text


# --------------------------------------------------------------------------
# 9. the store's header guard
# --------------------------------------------------------------------------

def test_the_store_refuses_a_narrower_header_already_on_disk(tmp_path):
    """`DictWriter` writes the fieldnames it was given and never looks at the
    file, so a wider row under a narrower header shifts every field past the
    first difference: `clock_drift_ok` would read the LEVEL side and a FAILED
    drift -- the one rule that excludes -- would come back None."""
    path = tmp_path / "cells.csv"
    narrow = [c for c in PW.CSV_FIELDS if c != "clock_level_side"]
    with path.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=narrow).writeheader()
    with pytest.raises(PW.SchemaCollision) as exc:
        PW.Store(path, PW.CSV_FIELDS)
    assert "clock_level_side" in str(exc.value)


def test_the_store_accepts_the_header_it_wrote(tmp_path):
    """Planted both ways: a guard that refuses everything is not a guard."""
    path = tmp_path / "cells.csv"
    PW.Store(path, PW.CSV_FIELDS).append(_sample(PW.SHARED, 1, 0, 1.0))
    PW.Store(path, PW.CSV_FIELDS).append(_sample(PW.SHARED, 2, 0, 2.0))
    assert len(PW.read_samples(path)) == 2


# --------------------------------------------------------------------------
# 10. the estimator itself
# --------------------------------------------------------------------------

def _noiseless(alpha: float, treads=(1, 2, 3, 4, 5, 6), repeats=3):
    """An exactly affine world: `ms = A + B n`, with the private arm's B one
    full stream and the shared arm's `alpha` of it. No model, no noise -- this
    is a test of the ARITHMETIC, and the planted worlds test the model."""
    out = []
    for rep in range(repeats):
        for n in treads:
            out.append(_sample(PW.SHARED, n, rep, 0.5 + alpha * n))
            out.append(_sample(PW.ALIAS, n, rep, 0.5 + alpha * n))
            out.append(_sample(PW.PRIVATE, n, rep, 0.5 + 1.0 * n))
    return out


#: Planted fractions, each chosen INSIDE a band and never on its edge. A value
#: sitting exactly on a boundary is a test of float rounding and not of the
#: partition: `0.85` is `NO_REUSE_MIN`, and a least-squares fit returns
#: 0.8499999999 for it, which the partition correctly calls the band below.
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.45, 0.558, 0.70, 0.95])
def test_the_ratio_recovers_the_planted_fraction_exactly(alpha):
    edges = {lo for _n, lo, _hi, _m in PW.OUTCOMES} | {PW.NO_REUSE_MIN}
    assert alpha not in edges, "a planted fraction is sitting on a band edge"
    samples = _noiseless(alpha)
    shared = PW.ladder_for(samples, PW.SHARED)
    private = PW.ladder_for(samples, PW.PRIVATE)
    assert shared.slope_ms == pytest.approx(alpha, rel=1e-9)
    assert private.slope_ms == pytest.approx(1.0, rel=1e-9)
    assert shared.slope_ms / private.slope_ms == pytest.approx(alpha, rel=1e-9)
    assert PW.outcome_for(alpha)[0] == PW.outcome_for(
        shared.slope_ms / private.slope_ms)[0]


def test_a_one_point_ladder_has_no_slope_and_says_so():
    """Refused rather than returned as zero: zero is a value a gate would read
    and act on."""
    with pytest.raises(PW.Unmeasurable):
        PW.fit_line([(1, 1.0)])
    with pytest.raises(PW.Unmeasurable):
        PW.fit_line([(3, 1.0), (3, 2.0)])


def test_the_interval_counts_the_draws_that_produced_a_ratio():
    """An interval standing on a third of its draws must not read like one
    standing on all of them."""
    samples = _noiseless(0.558)
    lo, hi, draws = PW.ratio_interval(samples, 200, seed=1)
    assert draws == 200
    assert lo <= 0.558 <= hi
    # A noiseless world has a degenerate interval, which is the correct answer
    # and not a bug: every resample of identical repeats is the same ladder.
    assert hi - lo < 1e-9


def test_the_interval_widens_with_the_scatter_it_is_made_of():
    """The property, not a remembered width: more scatter between repeats is a
    wider interval, and an interval that did not move with it would be
    reporting the model rather than the data."""
    import random
    widths = []
    for spread in (0.002, 0.02):
        rng = random.Random(7)
        rows = []
        for rep in range(9):
            for n in (1, 2, 3, 4, 5, 6):
                rows.append(_sample(PW.SHARED, n, rep,
                                    (0.5 + 0.558 * n) * (1 + rng.gauss(0, spread))))
                rows.append(_sample(PW.PRIVATE, n, rep,
                                    (0.5 + 1.0 * n) * (1 + rng.gauss(0, spread))))
        lo, hi, _ = PW.ratio_interval(rows, 400, seed=3)
        widths.append(hi - lo)
    assert widths[1] > widths[0], widths


# --------------------------------------------------------------------------
# 11. the buffer proof's own bookkeeping
# --------------------------------------------------------------------------

def test_the_proof_count_is_taken_and_not_quoted():
    """THE RECURRING DEFECT, in its exact shape: a proof whose prose says five
    parts and whose loop runs four. The gate reads `PROOF_PARTS` and so does
    this."""
    assert len(PROOF_NAMES := [n for n, _ in PW.PROOF_PARTS]) == 5
    assert len(set(PROOF_NAMES)) == 5
    text = SCRIPT.read_text()
    for name in PROOF_NAMES:
        # Each part is both declared and WRITTEN by the prover.
        assert f'parts["{name}"]' in text, name


def test_a_missing_part_is_unknown_and_never_a_pass():
    full = PW.BufferProof(parts={n: True for n, _ in PW.PROOF_PARTS}, detail={})
    assert full.verdict == exit_codes.PASS
    partial = PW.BufferProof(
        parts={n: True for n, _ in PW.PROOF_PARTS[:-1]}, detail={})
    assert partial.verdict == exit_codes.UNKNOWN
    one_bad = PW.BufferProof(
        parts={n: (i > 0) for i, (n, _) in enumerate(PW.PROOF_PARTS)}, detail={})
    assert one_bad.verdict == exit_codes.FAIL


def test_the_copies_leave_the_builder_bitwise_identical():
    """THE BUG THIS TEST IS NAMED AGAINST. The first version of the builder
    wrote a distinct sentinel into each copy and left it there, so every copy
    differed from every other in one element -- which is exactly the state
    part 5 (`same_layer`) exists to rule out. A CORRECT relabelling would then
    have produced a different output from the shared arm, V2 would have FAILed,
    and the arm would have exited INVALID on a working instrument. Built on the
    `toy` geometry on the CPU so the invariant is checked without a device."""
    import torch

    toy = MODEL_CONFIGS["toy"]
    copies = 4
    w1, w2, delta = PW.build_private_weights(toy, "bf16", copies, seed=0,
                                             device="cpu")
    e = toy.num_experts
    assert w1.shape[0] == PW.expert_space(e, copies)
    for c in range(1, copies):
        assert torch.equal(w1[c * e:(c + 1) * e], w1[:e]), c
        assert torch.equal(w2[c * e:(c + 1) * e], w2[:e]), c
    # And the weights are not all zero, which an `empty` that was never filled
    # would also satisfy the equality above with.
    assert float(w1[:e].abs().max()) > 0.0
    # Off a CUDA device the allocation delta is None -- NOT MEASURED -- so V3
    # reads UNKNOWN rather than FAILing on an allocation of zero.
    assert delta is None


def test_the_sentinel_round_trip_distinguishes_the_copies_and_puts_them_back():
    """The pairing the builder bug came from: a write that tells one copy's
    memory from another's is only sound if the original bytes go back before
    anything compares the two arms' outputs."""
    import torch

    toy = MODEL_CONFIGS["toy"]
    copies = 4
    w1, _w2, _d = PW.build_private_weights(toy, "bf16", copies, seed=0,
                                           device="cpu")
    e = toy.num_experts
    before = w1.clone()
    ok, detail = PW.sentinel_roundtrip(w1, e, copies)
    assert ok, detail
    assert torch.equal(w1, before), "the sentinels were not put back"
    assert f"{copies} distinct" in detail


def test_the_sentinel_check_fails_when_the_copies_are_not_distinct_memory():
    """Planted the other way: a `w1` whose copies ALIAS one another -- every
    copy a stride-0 view of copy 0 -- must not pass the sentinel part. Without
    this the check is a write followed by a read of the same address."""
    import torch

    toy = MODEL_CONFIGS["toy"]
    e, copies = toy.num_experts, 4
    n, h = 2 * toy.intermediate_size, toy.hidden_size
    base = torch.zeros((e, n, h), dtype=torch.bfloat16)
    # Stride 0 on the expert axis: every expert index, and so every copy,
    # names the same bytes. `expand().reshape()` would NOT do this -- reshape
    # on an expanded tensor materialises a real copy -- and a test built that
    # way passes while proving nothing.
    aliased = base.as_strided((copies * e, n, h), (0, h, 1))
    assert aliased[0].data_ptr() == aliased[e].data_ptr()
    ok, detail = PW.sentinel_roundtrip(aliased, e, copies)
    assert not ok, detail


def test_a_planted_proof_says_it_was_planted():
    proof = PW.planted_proof(True)
    assert proof.synthetic is True
    assert "PLANTED" in "\n".join(proof.lines())


# --------------------------------------------------------------------------
# 12. the run id
# --------------------------------------------------------------------------

def _args(**over):
    ns = PW.build_parser().parse_args(
        ["--device-memory-gb", "140", *sum(([k, str(v)] for k, v in over.items()), [])])
    return ns


def test_every_knob_that_moves_a_millisecond_is_in_the_run_id():
    base = PW.default_run_id(_args(), "NVIDIA H200")
    for flag, value in (("--block-m", 64), ("--treads", 5), ("--repeats", 3),
                        ("--warmup", 500.0), ("--cell-budget-ms", 400.0),
                        ("--trials", 5), ("--model", "qwen2-57b-a14b"),
                        ("--num-stages", 3), ("--block-n", 128),
                        ("--group-m", 16), ("--seed", 3)):
        other = PW.default_run_id(_args(**{flag: value}), "NVIDIA H200")
        assert other != base, flag


def test_an_analysis_knob_is_not_in_the_run_id():
    """Two analyses of one sweep belong in one directory: `--ridge`,
    `--bandwidth-gbps` and `--draws` re-score a set of cells, they do not move
    one."""
    base = PW.default_run_id(_args(), "NVIDIA H200")
    for flag, value in (("--ridge", 145.8), ("--bandwidth-gbps", 1799.4),
                        ("--draws", 500), ("--device-memory-gb", 80.0),
                        ("--alpha", 0.3)):
        assert PW.default_run_id(_args(**{flag: value}),
                                 "NVIDIA H200") == base, flag


def test_the_card_is_in_the_run_id_and_a_planted_world_is_prefixed():
    a100 = PW.default_run_id(_args(), "NVIDIA A100-SXM4-80GB")
    h200 = PW.default_run_id(_args(), "NVIDIA H200")
    assert a100 != h200
    assert h200.startswith("nvidia_h200")
    ns = PW.build_parser().parse_args(["--self-test", "refit"])
    planted = PW.default_run_id(ns, "NVIDIA H200")
    assert planted.startswith("synthetic-")
    assert planted != h200


# --------------------------------------------------------------------------
# 13. the sweep's own bookkeeping, checked without a device
# --------------------------------------------------------------------------

def test_the_arm_rotation_is_derived_from_the_arm_tuple_and_covers_it():
    """No arm is first in every triple. The rotation is derived from `ARMS`
    rather than listed a second time, so a fourth arm rotates by itself."""
    firsts = set()
    for rep in range(len(PW.ARMS) * 2):
        order = [PW.ARMS[(i + rep) % len(PW.ARMS)] for i in range(len(PW.ARMS))]
        assert sorted(order) == sorted(PW.ARMS)
        firsts.add(order[0])
    assert firsts == set(PW.ARMS)


def test_the_three_arms_are_declared_once():
    assert PW.ARMS == (PW.SHARED, PW.ALIAS, PW.PRIVATE)
    assert set(PW.ARM_MEANING) == set(PW.ARMS)


def _one_arm_seconds(alpha, *, treads, block_m, repeats, ridge, bandwidth_gbps,
                     b, warmup_ms, trials, cell_budget_ms):
    """One arm's kernel seconds at one alpha, recomputed here from the sweep's
    own model, so the identity below is an arithmetic check and not a restated
    number."""
    import block_m_crossing_sweep as SWEEP
    total = 0.0
    for n in treads:
        ms = SWEEP.model_ms(CFG, n * block_m, block_m, alpha=alpha, ridge=ridge,
                            bandwidth_gbps=bandwidth_gbps, b=b)
        iters = SWEEP.planned_iters(ms, cell_budget_ms)
        total += repeats * (warmup_ms + trials * iters * ms) * 1e-3
    return total


def test_the_cost_estimate_prices_the_private_arm_at_alpha_one():
    """The private arm moves a full stream per M-tile BY CONSTRUCTION, so
    pricing all three at the study's alpha would price a ladder nobody is going
    to run. Asserted as the exact identity, because an ordering would not
    notice it: `time_kernel` holds `--cell-budget-ms` of kernel time per trial,
    so the per-cell cost is nearly flat in the per-call time and the two worlds
    differ by rounding alone."""
    kw = dict(treads=[1, 2, 3, 4, 5, 6], block_m=32, repeats=9, ridge=160.0,
              bandwidth_gbps=4000.0, b=2, warmup_ms=300.0, trials=3,
              cell_budget_ms=200.0)
    for alpha in (PW.RETRACTED_ALPHA, PW.ALPHA, 1.0):
        want = (2 * _one_arm_seconds(alpha, **kw)
                + _one_arm_seconds(1.0, **kw))
        assert PW.estimated_seconds(CFG, alpha=alpha, **kw) == pytest.approx(
            want, rel=1e-12), alpha


def test_the_plan_names_its_kernel_clock_so_the_session_can_read_it():
    """`scripts/h200_gaps_session.sh` derives a booked figure's CLOCK from what
    the arm's own plan prints. A plan that stops saying "excluding compiles and
    allocation" silently re-files the booking as a wall figure."""
    got = run(["--dry-run", "--device-memory-gb", "140"])
    assert "excluding compiles and allocation" in got.stdout
    assert "NOT IN THAT FIGURE" in got.stdout
