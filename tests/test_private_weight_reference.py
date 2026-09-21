"""The private-weight reference arm: its scorer, its refusals and its contract.

WHAT THIS FILE IS FOR. `scripts/private_weight_reference.py` measures alpha as
a ratio of two measured slopes, so the only things that can go wrong off GPU
are the arithmetic, the partition, the relabelling and the gates -- and every
one of those is pure. Nothing here needs a device; everything here is what the
pod run will execute.
"""
from __future__ import annotations

import ast
import contextlib
import csv
import io
import json
import math
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import private_weight_reference as PW  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.bench import timing as TIMING  # noqa: E402
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
    (("--self-test", "clock-split"), True),
    (("--self-test", "alignment-step"), True),
    (("--self-test", "host-bound-probe"), True),
    (("--self-test", "host-bound-controlled"), True),
    (("--self-test", "ratio-path-split"), True),
    (("--self-test", "ratio-step-over-budget"), True),
    (("--self-test", "ratio-step-under-budget"), True),
    (("--self-test", "over-allocated"), True),
    (("--self-test", "holes"), True),
    (("--self-test", "ragged"), True),
    # THE ELEVENTH, which this list did not have. `faster-than-its-ruler` was
    # in `WORLDS` and in the two tests that iterate WORLDS, so it was scored --
    # but the `classify_text(log) == returncode` contract below runs off THIS
    # list, so the one world whose registered failure is C2 never had its log
    # checked against its own exit code. The test under it now asserts this
    # list covers WORLDS, so a twelfth world cannot be added and left out.
    (("--self-test", "faster-than-its-ruler"), True),
)


def test_the_off_gpu_mode_list_covers_every_planted_world():
    """A LIST THAT NAMES THE WORLDS AND MISSES ONE IS THE CHECK THAT EXAMINED
    NOTHING, one level up: the exit-contract test iterates this tuple, so a
    world absent from it is a world whose log and exit code are never compared.
    """
    driven = {argv[1] for argv, _ in OFF_GPU_MODES if argv[0] == "--self-test"}
    assert driven == set(PW.WORLDS), set(PW.WORLDS) ^ driven


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
                                        "V6", "V7", "V8", "C1", "C2"]


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
    confident table.

    IDENTITY IS ASKED AGAINST THE MODULE OBJECT `PW` ITSELF BOUND, and not
    against a fresh `import block_m_crossing_sweep`, which is what this test did
    until it was run inside the whole suite rather than alone. Six other test
    files load `scripts/block_m_crossing_sweep.py` BY PATH under that same
    module name, so whichever ran first owns `sys.modules` and a later plain
    import can hand back a DIFFERENT module object holding an equal but
    distinct `(0.529, 0.588)`. `is` then fails on two files that agree
    perfectly, which is an artefact of the loader and not a drift. So: identity
    against `PW.SWEEP` proves PW typed no second copy, and EQUALITY against
    whatever instance this process resolves proves the value is the study's."""
    import block_m_crossing_sweep as SWEEP
    assert PW.ALPHA_BAND is PW.SWEEP.ALPHA_BAND
    assert PW.ALPHA_BAND == SWEEP.ALPHA_BAND
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
    n_max = 6
    ids = balanced_flat_ids(e, rows, k)
    out = PW.private_topk_ids(ids, e, block_m, rows, n_max)
    counts = torch.bincount(out.reshape(-1).long(),
                            minlength=PW.expert_space(e, n_max))
    assert counts.numel() == PW.expert_space(e, n_max)
    used = counts[counts > 0]
    assert used.numel() == e * tiles
    assert int(used.min()) == block_m and int(used.max()) == block_m
    # EXPERT-FIRST: the original expert is the QUOTIENT and the copy the
    # residue, so a row still reads the weights of its own expert and no slot
    # past `tiles` copies of any expert is touched.
    flat = out.reshape(-1).long()
    assert bool(((flat // n_max) == ids.reshape(-1).long()).all())
    assert int((flat % n_max).max()) == tiles - 1
    assert out.dtype == ids.dtype


def test_the_identity_tread_relabels_to_itself():
    """At one M-tile per expert the private arm IS the shared arm, which is the
    whole content of V6: there is nothing for the three arms to differ by."""
    e, k = CFG.num_experts, CFG.top_k
    ids = balanced_flat_ids(e, PW.DEFAULT_BLOCK_M, k)
    out = PW.private_topk_ids(ids, e, PW.DEFAULT_BLOCK_M, PW.DEFAULT_BLOCK_M,
                              PW.DEFAULT_TREADS)
    assert bool((out == PW.shared_topk_ids(ids, PW.DEFAULT_TREADS)).all())


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
        PW.private_topk_ids(ids, e, 32, 64, 6)


def test_a_partly_filled_stack_is_refused():
    e, k = CFG.num_experts, CFG.top_k
    ids = balanced_flat_ids(e, 48, k)
    with pytest.raises(PW.PrivateWeightRefusal):
        PW.private_topk_ids(ids, e, 32, 48, 6)


def test_a_ladder_deeper_than_the_declaration_is_refused():
    """Expert-first, tile n_max of expert e would be slot (e+1) x n_max: the
    NEXT expert's copy 0, allocated and holding the wrong weights, so nothing
    would crash. Refused at the relabelling instead."""
    e, k = CFG.num_experts, CFG.top_k
    ids = balanced_flat_ids(e, 7 * 32, k)
    with pytest.raises(PW.PrivateWeightRefusal):
        PW.private_topk_ids(ids, e, 32, 7 * 32, 6)


def _slot_order_by_expert(ids, n_max):
    """The expert each M-tile belongs to, in the order `moe_align_block_size`'s
    sort by slot id visits them."""
    import torch
    flat = ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    return (flat[order] // n_max).tolist()


@pytest.mark.parametrize("tiles", [1, 2, 4, 6])
def test_shared_and_private_visit_the_experts_in_the_same_order(tiles):
    """THE ORDER DEFECT. Copy-first ids (`c x E + e`) sorted to expert 0..E-1
    of copy 0, then of copy 1, while the shared arm ran every tile of expert 0
    first -- and ORDER is the lever this study measured moving the per-M-tile
    cost by 30-48%. Expert-first slots must put the private arm's tiles in the
    shared arm's expert order, row for row."""
    e, k, bm, n_max = CFG.num_experts, CFG.top_k, 32, 6
    ids = balanced_flat_ids(e, tiles * bm, k)
    private = PW.private_topk_ids(ids, e, bm, tiles * bm, n_max)
    shared = PW.shared_topk_ids(ids, n_max)
    assert _slot_order_by_expert(private, n_max) == _slot_order_by_expert(
        shared, n_max)


def _align_buffer(numel, declared, block_m):
    """vLLM's `moe_align_block_size` buffer length, the launch grid's EM."""
    return numel + declared * (block_m - 1)


def test_shared_and_private_declare_one_space_so_the_dead_launches_are_a_constant():
    """THE LAUNCH DEFECT. The sorted-id buffer and so the launch grid scale with
    the declaration. Declaring E x n at tread n put ~8 dead M-rows per extra
    tread into the private slope alone. Now both ratio arms declare
    E x n_max at every tread: identical buffers, and a dead-row count that does
    not change with the tread, so it lands in an intercept."""
    e, bm, n_max = CFG.num_experts, 32, 6
    dead = set()
    for n in range(1, n_max + 1):
        numel = e * n * bm
        s = _align_buffer(numel, PW.declared_experts(PW.SHARED, e, n_max), bm)
        p = _align_buffer(numel, PW.declared_experts(PW.PRIVATE, e, n_max), bm)
        assert s == p, n
        dead.add(s - numel)
    assert len(dead) == 1
    assert PW.declared_experts(PW.NATIVE, e, n_max) == e


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
        tokens=n * 128, copies=PW.copies_read(arm, n),
        experts_declared=PW.declared_experts(arm, CFG.num_experts, 6),
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
            out.append(_sample(PW.NATIVE, n, rep, 0.4 + alpha * n))
            out.append(_sample(PW.SHARED, n, rep, 0.5 + alpha * n))
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
        assert torch.equal(w1[c::copies], w1[::copies]), c
        assert torch.equal(w2[c::copies], w2[::copies]), c
    # Distinct experts hold distinct weights: a layout that wrote one expert's
    # draw into every slot would pass the copy equality above.
    assert not torch.equal(w1[PW.copy_slot(0, 0, copies)],
                           w1[PW.copy_slot(1, 0, copies)])
    # And the weights are not all zero, which an `empty` that was never filled
    # would also satisfy the equality above with.
    assert float(w1[::copies].abs().max()) > 0.0
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
    before = w1.clone()
    ok, detail = PW.sentinel_roundtrip(w1, copies)
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
    assert aliased[0].data_ptr() == aliased[1].data_ptr()
    ok, detail = PW.sentinel_roundtrip(aliased, copies)
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
                        ("--group-m", 16), ("--seed", 3),
                        ("--session-tag", "gaps-nvidia_h200-20260918"),
                        ("--declared-copies", 9)):
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
    assert PW.ARMS == (PW.NATIVE, PW.SHARED, PW.PRIVATE)
    assert PW.RATIO_ARMS == (PW.SHARED, PW.PRIVATE)
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
                + _one_arm_seconds(1.0, **kw)
                + PW.probe_seconds(kw["treads"], PW.PROBE_LABELS))
        assert PW.estimated_seconds(CFG, alpha=alpha, **kw) == pytest.approx(
            want, rel=1e-12), alpha


def test_the_plan_names_its_kernel_clock_so_the_session_can_read_it():
    """`scripts/h200_gaps_session.sh` derives a booked figure's CLOCK from what
    the arm's own plan prints. A plan that stops saying "excluding compiles and
    allocation" silently re-files the booking as a wall figure."""
    got = run(["--dry-run", "--device-memory-gb", "140"])
    assert "excluding compiles and allocation" in got.stdout
    assert "NOT IN THAT FIGURE" in got.stdout


def _full_grid(*, drifted=()):
    """Every cell of a 6-tread, 9-repeat, three-arm grid, some of them drifted."""
    rows = []
    for rep in range(9):
        for n in range(1, 7):
            for arm in PW.ARMS:
                drift_ok = (arm, n, rep) not in drifted
                rows.append(_sample(arm, n, rep, 1.0 + n,
                                    drift_ok=drift_ok,
                                    load=PLANTED_REFERENCE_MHZ))
    return rows


def test_one_drifted_cell_does_not_void_a_run_that_met_every_floor():
    """V0 USED TO BE 'ZERO CELLS DRIFTED' AND SAID SO NOWHERE. It required
    `len(usable) == planned`, so one drifted cell in 162 latched INVALID over a
    run whose printed floors -- usable treads per arm, repeats behind every
    tread -- were all met, on a card this apparatus documents as power-capped
    with an endogenous clock and where DRIFT exclusion is the normal membership
    rule. `alias_ablation` is written against exactly this.
    """
    planned = 6 * 9 * len(PW.ARMS)
    treads = list(range(1, 7))
    clean = PW.gate_v0_non_vacuity(_full_grid(), planned=planned,
                                   treads=treads, repeats=9)
    assert clean.verdict == PW.PASS, clean.measured
    one = PW.gate_v0_non_vacuity(_full_grid(drifted={(PW.PRIVATE, 4, 2)}),
                                 planned=planned, treads=treads, repeats=9)
    assert one.verdict == PW.PASS, one.measured
    assert "drift" in one.measured


def test_drift_past_the_registered_ceiling_still_voids_the_page():
    """And the ceiling is a ceiling, not an amnesty: past it the kept set is a
    subsample the card chose and no ladder on the page is the planned grid."""
    planned = 6 * 9 * len(PW.ARMS)
    treads = list(range(1, 7))
    heavy = {(arm, n, rep) for arm in PW.ARMS for n in treads
             for rep in range(9) if (n + rep) % 3 == 0}
    gate = PW.gate_v0_non_vacuity(_full_grid(drifted=heavy), planned=planned,
                                  treads=treads, repeats=9)
    assert gate.verdict == PW.FAIL, gate.measured
    assert f"{100 * PW.MAX_DRIFT_FRACTION:.0f}%" in gate.threshold


def test_a_resume_re_measures_a_drifted_cell_rather_than_calling_it_done():
    """The resume key is `usable`, the same predicate every ladder fits on, and
    not `status == "ok"`, which a drifted cell still is. Keyed on the status a
    second pass skipped the one class of cell it could have fixed, and the only
    recovery left was deleting cells.csv and re-paying the sweep."""
    source = (ROOT / "scripts" / "private_weight_reference.py").read_text()
    body = source.split("done = {(s.arm, s.tiles, s.repeat)")[1][:120]
    assert "s.usable" in body, body


def test_the_interval_is_over_repeats_and_paired_across_the_two_arms():
    """A COMMON PER-REPEAT FACTOR IS NOT UNCERTAINTY IN A RATIO. The arm
    rotation exists to make a governor or thermal walk common to all three arms
    within a repeat; a ratio then divides it out exactly. Resampling the cells
    within each tread independently, and the two arms independently of each
    other, put that factor back as noise: this ladder, whose ratio is 0.558 in
    EVERY repeat, came back as an interval 3.3x the width of ALPHA_BAND, and the
    widening runs in the direction that makes C1's overlap test easier to pass.
    """
    rows = []
    for rep in range(9):
        factor = 1.0 + 0.20 * ((rep % 3) - 1)
        for n in range(1, 7):
            rows.append(_sample(PW.SHARED, n, rep, (0.3 + 0.558 * n) * factor))
            rows.append(_sample(PW.PRIVATE, n, rep, (0.3 + 1.000 * n) * factor))
    lo, hi, drawn = PW.ratio_interval(rows, 500, 0)
    assert drawn == 500
    assert hi - lo < 1e-9, (lo, hi)
    assert abs(lo - 0.558) < 1e-9, lo


def test_a_paired_draw_takes_both_arms_from_the_same_repeats():
    """The pairing, asserted on the mechanism rather than on an interval: one
    list of repeat indices per draw, and `collapse` takes the arm's cells from
    exactly those, a repeat drawn twice counting twice."""
    rows = [_sample(arm, n, rep, 1.0 + n + rep)
            for arm in (PW.SHARED, PW.PRIVATE)
            for n in (1, 2, 3) for rep in (0, 1, 2)]
    assert PW.repeat_indices(rows) == [0, 1, 2]
    points, _spread, _dropped = PW.collapse(rows, PW.SHARED, [1, 1, 1])
    # Every point is repeat 1's own time, because repeat 1 is the whole draw.
    assert points == [(1, 3.0), (2, 4.0), (3, 5.0)], points


# --------------------------------------------------------------------------
# 14. the 2026-09-17 review round: C1's verdict, V6's pair, V7, the device
#     guard, and the buffer proof run end to end against a CPU reference
# --------------------------------------------------------------------------

def test_c1_does_not_pass_a_point_the_page_names_as_another_world():
    """The reviewer's case: 0.600 [0.585, 0.615] overlapped the band, PASSED
    and exited DONE while the page named ABOVE-THE-REFIT-BAND."""
    assert PW.outcome_for(0.600)[0] == "ABOVE-THE-REFIT-BAND"
    assert PW.c1_verdict(0.600, (0.585, 0.615)) == exit_codes.UNKNOWN


def test_c1_does_not_pass_an_interval_that_reaches_an_alternative_world():
    assert PW.c1_verdict(PW.ALPHA, (0.30, 0.80)) == exit_codes.UNKNOWN
    assert PW.c1_verdict(PW.ALPHA, (0.40, 0.90)) == exit_codes.UNKNOWN


def test_c1_passes_a_resolved_point_in_band_and_fails_a_miss():
    assert PW.c1_verdict(PW.ALPHA, (0.54, 0.57)) == exit_codes.PASS
    assert PW.c1_verdict(0.95, (0.93, 0.97)) == exit_codes.FAIL
    assert PW.c1_verdict(0.10, (0.08, 0.12)) == exit_codes.FAIL
    assert PW.c1_verdict(PW.ALPHA, (math.nan, math.nan)) == exit_codes.UNKNOWN


def _pair_world(*, native_n1=None, private_clock=1500.0, shared_clock=1500.0,
                treads=(1, 2, 3)):
    out = []
    for rep in range(3):
        for n in treads:
            out.append(_sample(PW.NATIVE, n, rep,
                               (native_n1 if (n == 1 and native_n1) else 0.5
                                + 0.5 * n), load=1500.0))
            out.append(_sample(PW.SHARED, n, rep, 0.5 + 0.5 * n,
                               load=shared_clock))
            out.append(_sample(PW.PRIVATE, n, rep, 0.5 + 1.0 * n
                               if n > 1 else 1.0, load=private_clock))
    return out


def test_v6_compares_the_same_call_and_records_native_without_scoring_it():
    """NATIVE differs from SHARED by a constant declaration at n=1. Requiring
    all three to agree there would refuse a correct instrument."""
    samples = _pair_world(native_n1=1.5)   # native 50% off at n=1
    gate = PW.gate_v6_identity(samples, identity_tread=1)
    assert gate.verdict == exit_codes.PASS, gate.lines
    assert any("native sits" in line for line in gate.lines)


def test_v7_fails_a_clock_split_and_passes_a_matched_clock():
    assert PW.gate_v7_clock_parity(_pair_world(), treads=[1, 2, 3]).verdict \
        == exit_codes.PASS
    split = _pair_world(private_clock=1500.0 * (1 - 2 * PW.CLOCK_PARITY))
    assert PW.gate_v7_clock_parity(split, treads=[1, 2, 3]).verdict \
        == exit_codes.FAIL


def test_v7_reads_an_unread_clock_as_unknown_and_not_as_a_match():
    samples = _pair_world()
    for s in samples:
        if s.arm == PW.PRIVATE and s.tiles == 2:
            s.sm_clock_load_mhz = None
    assert PW.gate_v7_clock_parity(samples, treads=[1, 2, 3]).verdict \
        == exit_codes.UNKNOWN


def test_v7_leaves_a_tread_no_ratio_arm_reached_to_v0_and_v1():
    samples = [s for s in _pair_world() if s.tiles != 3]
    assert PW.gate_v7_clock_parity(samples, treads=[1, 2, 3]).verdict \
        == exit_codes.PASS


def test_v7_and_v4_read_unknown_when_they_examined_nothing():
    """A check that examined nothing reports no failures: the review's case.
    With no tread reached by both ratio arms V7 has compared no clocks, and
    with no shared or private row V4 has read no roof fraction."""
    natives = [s for s in _pair_world() if s.arm == PW.NATIVE]
    assert PW.gate_v7_clock_parity(natives, treads=[1, 2, 3]).verdict \
        == exit_codes.UNKNOWN
    assert PW.gate_v4_memory_bound([], roof_tflops=600.0,
                                   roof_source="x").verdict == exit_codes.UNKNOWN
    rows = [{"arm": PW.NATIVE, "tiles": 1, "pct_of_roof": 0.99}]
    assert PW.gate_v4_memory_bound(rows, roof_tflops=600.0,
                                   roof_source="x").verdict == exit_codes.UNKNOWN


def test_the_proof_keeps_no_backup_of_a_copy_it_zeroes():
    """The review's severity-one finding: a clone of one copy is a whole
    weight set above the sweep's peak, and V3 scores that peak. The restore
    comes from copy 0, which is never zeroed."""
    import inspect
    src = inspect.getsource(PW.prove_distinct_buffers)
    assert ".clone()" not in src.split("# 3 and 4.")[1], (
        "the zeroing loops clone something")
    assert "copy_(w1[::copies])" in src and "copy_(w2[::copies])" in src


def test_the_device_guard_accepts_a_no_uuid_identity_only_against_itself(
        tmp_path):
    out = tmp_path / "run"
    weak = PW.NO_UUID_PREFIX + "nvidia_h200"
    assert PW.device_guard(out, weak) == ""
    (out / "cells.csv").write_text("x\n")
    assert PW.device_guard(out, weak) == ""
    assert PW.device_guard(out, "GPU-aaaa") != ""
    assert PW.device_guard(out, PW.NO_UUID_PREFIX + "nvidia_a100") != ""


def test_the_device_guard_refuses_a_second_card_and_resumes_the_first(tmp_path):
    out = tmp_path / "run"
    assert PW.device_guard(out, "GPU-aaaa") == ""
    assert (out / PW.DEVICE_FILE).read_text().strip() == "GPU-aaaa"
    (out / "cells.csv").write_text("x\n")
    assert PW.device_guard(out, "GPU-aaaa") == ""
    assert "GPU-bbbb" in PW.device_guard(out, "GPU-bbbb")
    assert PW.device_guard(out, "") != ""


def test_the_device_guard_refuses_cells_of_unknown_provenance(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "cells.csv").write_text("x\n")
    assert PW.device_guard(out, "GPU-aaaa") != ""


def _reference_fused(x, w1, w2, topk_weights, topk_ids):
    """A per-slot reference of vLLM's fused_experts with SwiGLU: every slot
    reads exactly the expert slot its id names, which is the property the
    proof tests. Row-independent by construction, like the Triton kernel."""
    import torch
    f = w2.shape[2]
    y = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            slot = int(topk_ids[t, j])
            h = w1[slot] @ x[t]
            act = torch.nn.functional.silu(h[:f]) * h[f:]
            y[t] += topk_weights[t, j].to(x.dtype) * (w2[slot] @ act)
    return y


def _cpu_proof(*, misroute=None, declared=None):
    """The whole proof on the toy model, off GPU. `misroute(ids, declared)`
    plants a broken relabelling into the private arm; `declared` above the
    three copies read plants the padding `declared_copies_for` adds."""
    import torch
    toy = MODEL_CONFIGS["toy"]
    e, k, bm, copies = toy.num_experts, toy.top_k, 4, 3
    declared = declared or copies
    rows = copies * bm
    w1, w2, _ = PW.build_private_weights(toy, "bf16", declared, seed=0,
                                         device="cpu")
    ids = balanced_flat_ids(e, rows, k, seed=3).to(torch.int64)
    tokens = ids.shape[0]
    x = torch.randn((tokens, toy.hidden_size), dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(1))
    weights = torch.full(ids.shape, 1.0 / k)
    private = PW.private_topk_ids(ids, e, bm, rows, declared)
    if misroute is not None:
        private = misroute(private, declared)
    by_arm = {PW.NATIVE: ids, PW.SHARED: PW.shared_topk_ids(ids, declared),
              PW.PRIVATE: private}
    calls = []

    def call_for(arm):
        a1, a2 = ((w1[::declared], w2[::declared]) if arm == PW.NATIVE
                  else (w1, w2))

        def call():
            calls.append(arm)
            return _reference_fused(x, a1, a2, weights, by_arm[arm])
        return call
    before = (w1.clone(), w2.clone())
    proof = PW.prove_distinct_buffers(call_for, w1, w2, cfg=toy,
                                      copies_read=copies, dtype="bf16",
                                      private_ids=private,
                                      copies_declared=declared)
    return proof, calls, before, (w1, w2)


def test_the_buffer_proof_passes_a_correct_relabelling_end_to_end():
    import torch
    proof, calls, before, after = _cpu_proof()
    assert proof.verdict == exit_codes.PASS, proof.lines()
    assert len(calls) == PW.proof_calls(3, 3)
    # It restores what it zeroed: nothing is left corrupted.
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])


def test_the_buffer_proof_catches_every_tile_past_the_first_reading_copy_one():
    """The reviewer's case, and why the zeroing is one copy at a time: zeroing
    copies 1..n-1 together changes the output for this bug too."""
    def to_copy_one(private, declared):
        c = private % declared
        return private - c + (c > 0).to(private.dtype)
    proof, *_ = _cpu_proof(misroute=to_copy_one)
    assert proof.parts["kernel_read"] is False, proof.lines()
    assert proof.verdict == exit_codes.FAIL


def test_the_buffer_proof_catches_a_relabelling_back_to_copy_zero():
    proof, *_ = _cpu_proof(misroute=lambda p, declared: p - p % declared)
    assert proof.parts["kernel_read"] is False
    assert proof.verdict == exit_codes.FAIL


def test_the_buffer_proof_passes_padding_copies_that_are_never_read():
    """`declared_copies_for` may declare more copies than the deepest tread
    reads. The proof zeroes each of those too and both arms must not move."""
    import torch
    proof, calls, before, after = _cpu_proof(declared=4)
    assert proof.verdict == exit_codes.PASS, proof.lines()
    assert len(calls) == PW.proof_calls(3, 4) == 3 + 6 + 2
    assert "never read" in proof.detail["kernel_read"]
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])


def test_the_buffer_proof_catches_a_tile_reading_a_padding_copy():
    """A relabelling that sends copy 2's tiles to the never-read copy 3:
    zeroing copy 2 then moves nothing (kernel_read) and zeroing copy 3 moves
    the private output (shared_blind). Both parts see it."""
    def to_padding(private, declared):
        return private + ((private % declared) == 2).to(private.dtype)
    proof, *_ = _cpu_proof(misroute=to_padding, declared=4)
    assert proof.parts["kernel_read"] is False, proof.lines()
    assert proof.parts["shared_blind"] is False, proof.lines()
    assert proof.verdict == exit_codes.FAIL


# --------------------------------------------------------------------------
# 15. the alignment kernel: cited, derived, measured, and kept apart
# --------------------------------------------------------------------------

def test_align_path_follows_the_cited_condition():
    lo, hi = PW.ALIGN_SMALL_BATCH_MAX_IDS, PW.ALIGN_SMALL_BATCH_MAX_EXPERTS
    assert PW.align_path(lo - 1, hi) == PW.SMALL_BATCH
    assert PW.align_path(lo, hi) == PW.BLOCK_SCAN
    assert PW.align_path(lo - 1, hi + 1) == PW.BLOCK_SCAN


def test_the_default_ladder_crosses_the_id_bound_and_native_switches_at_four():
    """The fact the whole add-on exists for, asserted on the default design
    rather than remembered: mixtral at BLOCK_M=32 puts 256 n ids on the
    table, so the study's own E=8 call switches kernel between treads 3 and
    4, inside the fit."""
    treads = PW.ladder_treads(CFG, PW.DEFAULT_BLOCK_M, PW.DEFAULT_TREADS)
    counts = [PW.ids_for_tread(CFG, n, PW.DEFAULT_BLOCK_M) for n in treads]
    assert min(counts) < PW.ALIGN_SMALL_BATCH_MAX_IDS <= max(counts)
    census = PW.path_census(CFG, treads, PW.DEFAULT_BLOCK_M,
                            {a: PW.declared_experts(a, CFG.num_experts, 6)
                             for a in PW.ARMS})
    assert census.switch_tread(PW.NATIVE) == 4


def test_declared_copies_pad_past_the_expert_bound_only_when_the_ladder_crosses():
    n, why = PW.declared_copies_for(CFG, [1, 2, 3, 4, 5, 6], 32)
    assert n == 9 and CFG.num_experts * n > PW.ALIGN_SMALL_BATCH_MAX_EXPERTS
    assert "72" in why
    # A ladder on one side of the id bound pads nothing.
    assert PW.declared_copies_for(CFG, [1, 2, 3], 32)[0] == 3
    assert PW.declared_copies_for(CFG, [4, 5, 6], 32)[0] == 6
    # A model already past the expert bound pads nothing.
    v3 = MODEL_CONFIGS["deepseek-v3"]
    assert v3.num_experts > PW.ALIGN_SMALL_BATCH_MAX_EXPERTS
    treads = PW.ladder_treads(v3, 32, 2)
    assert PW.declared_copies_for(v3, treads, 32)[0] == 2
    # The operator may declare more, never fewer than the deepest tread reads.
    assert PW.declared_copies_for(CFG, [1, 2, 3, 4, 5, 6], 32, 12)[0] == 12
    with pytest.raises(PW.PrivateWeightRefusal):
        PW.declared_copies_for(CFG, [1, 2, 3, 4, 5, 6], 32, 5)


def test_the_census_refuses_a_ratio_arm_that_switches_and_lets_native_switch():
    treads = [1, 2, 3, 4, 5, 6]
    tight = PW.path_census(CFG, treads, 32, {PW.NATIVE: 8, PW.SHARED: 48,
                                             PW.PRIVATE: 48})
    assert tight.switch_tread(PW.SHARED) == 4
    assert [r for r in tight.refusals if r.startswith("shared")]
    assert [r for r in tight.refusals if r.startswith("private")]
    assert not [r for r in tight.refusals if r.startswith("native")]
    padded = PW.path_census(CFG, treads, 32, {PW.NATIVE: 8, PW.SHARED: 72,
                                              PW.PRIVATE: 72})
    assert padded.refusals == ()
    assert padded.switch_tread(PW.NATIVE) == 4
    assert padded.switch_tread(PW.SHARED) is None
    assert padded.switch_tread(PW.PRIVATE) is None


def test_the_census_refuses_the_naive_assignment_and_the_buffer_clamp():
    """Two other launch-changing branches in vLLM's own arithmetic, refused
    from the plan: the toy model at one tread has 16 ids, so declaring 64
    experts trips both `16 x 4 <= 64` and `16 < 64`."""
    toy = MODEL_CONFIGS["toy"]
    census = PW.path_census(toy, [1], 4, {PW.NATIVE: toy.num_experts,
                                          PW.SHARED: 64, PW.PRIVATE: 64})
    assert any("naive assignment" in r for r in census.refusals)
    assert any("clamps the sorted-id buffer" in r for r in census.refusals)
    fine = PW.path_census(toy, [1], 4, {a: toy.num_experts for a in PW.ARMS})
    assert fine.refusals == ()


def test_the_census_refuses_a_declaration_the_scan_kernel_refuses():
    census = PW.path_census(CFG, [1], 32, {PW.NATIVE: 8, PW.SHARED: 1024,
                                           PW.PRIVATE: 1024})
    assert any("refuses" in r for r in census.refusals)


def test_leverage_is_the_indicator_regressed_on_n():
    assert PW.leverage([1, 2, 3, 4, 5, 6], 4) == pytest.approx(4.5 / 17.5)
    assert PW.leverage([1, 2, 3, 4, 5, 6], 1) == pytest.approx(0.0)


def test_step_bias_is_the_leverage_over_the_weight_stream():
    assert PW.step_bias(0.01, [1, 2, 3, 4, 5, 6], 4, 0.5) == pytest.approx(
        (4.5 / 17.5) * 0.01 / 0.5)
    assert PW.step_bias(0.01, [1, 2, 3, 4, 5, 6], 4, 0.0) == math.inf


def _series(step_ms: float, split: int, noise=0.0, seed=0):
    import random
    rng = random.Random(seed)
    out = []
    for n in range(1, 7):
        numel = 256 * n
        ms = 0.012 + 8e-6 * numel + (step_ms if n >= split else 0.0)
        out.append((n, numel, ms * (1.0 + rng.gauss(0.0, noise))))
    return out


def test_step_fit_recovers_a_planted_step_and_finds_none_on_a_line():
    fit = PW.step_fit(_series(0.05, 4))
    assert fit.split_tread == 4
    assert fit.step_ms == pytest.approx(0.05, rel=1e-9)
    assert fit.rss_with < 1e-20
    flat = PW.step_fit(_series(0.0, 4))
    assert abs(flat.step_ms) < 1e-9
    assert flat.rss_without < 1e-20
    with pytest.raises(PW.Unmeasurable):
        PW.step_fit(_series(0.0, 4)[:2])


def _probe_from(series_by_label, spread=0.0, host_bound=False):
    cells = []
    for label, series in series_by_label.items():
        for rep in range(3):
            for n, numel, ms in series:
                jitter = (rep - 1) * spread
                cells.append(PW.ProbeCell(label, n, numel, 72, rep,
                                          ms * (1.0 + jitter),
                                          host_bound=host_bound))
    return PW.AlignProbe(tuple(cells), synthetic=True)


def _census():
    return PW.path_census(CFG, [1, 2, 3, 4, 5, 6], 32,
                          {PW.NATIVE: 8, PW.SHARED: 72, PW.PRIVATE: 72})


def test_v8_budget_is_a_bias_on_the_ratio_and_not_a_step_in_microseconds():
    """The same step passes at a deep private slope and fails at a shallow one:
    the gate is on what the step does to the RATIO."""
    treads = [1, 2, 3, 4, 5, 6]
    probe = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.02, 4)}, spread=1e-4)
    assert PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64).verdict == exit_codes.PASS
    assert PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.2).verdict == exit_codes.FAIL


def test_the_planted_probe_puts_natives_step_where_the_census_does():
    census = _census()
    probe = PW.planted_probe(PW.WORLDS["refit"], CFG, block_m=32,
                             treads=[1, 2, 3, 4, 5, 6],
                             declared_by_arm={PW.NATIVE: 8, PW.SHARED: 72,
                                              PW.PRIVATE: 72},
                             census=census, noise=0.002, seed=0)
    native = PW.read_probe(probe, PW.NATIVE, census)
    assert native.real and native.fit.split_tread == 4
    shared = PW.read_probe(probe, PW.SHARED, census)
    assert not shared.real


def test_the_declaration_fit_takes_natives_step_out_of_v5():
    """LOAD-BEARING: in the alignment-step world the raw native - shared slope
    gap carries the step at the design's leverage and would FAIL V5; the fit
    with the step term reads the declaration's per-tile cost as ~0 and PASSES.
    Noise off, so the arithmetic is exact."""
    world = PW.WORLDS["alignment-step"]
    treads = [1, 2, 3, 4, 5, 6]
    samples = PW.planted_samples(world, CFG, block_m=32, treads=treads,
                                 repeats=3, alpha_shared=world.alpha,
                                 ridge=160.0, bandwidth_gbps=4000.0, b=2,
                                 noise=0.0, seed=0, copies_declared=9,
                                 native_switch=4)
    fit = PW.declaration_fit(samples, treads, 4)
    assert fit.step_ms == pytest.approx(world.alignment_step_ms, rel=1e-6)
    assert abs(fit.per_tile_ms) < 1e-9
    native, shared, private = (PW.ladder_for(samples, a) for a in PW.ARMS)
    raw = abs(native.slope_ms - shared.slope_ms) / private.slope_ms
    assert raw == pytest.approx(PW.leverage(treads, 4) * world.alignment_step_ms
                                / private.slope_ms, rel=1e-6)
    assert raw > PW.MACHINERY_BOUND, "the planted step is not load-bearing"
    assert PW.gate_v5_machinery(native, shared, private).verdict == exit_codes.FAIL
    # The fit alone is UNKNOWN: V5 scores b's FAR EDGE, and with no band the
    # point is all there is. The band from the same noise-free samples PASSES.
    assert PW.gate_v5_machinery(native, shared, private, fit).verdict \
        == exit_codes.UNKNOWN
    band, _s, _n = PW.declaration_interval(samples, treads, 4, 50, 0)
    assert PW.gate_v5_machinery(native, shared, private, fit, band).verdict \
        == exit_codes.PASS
    # And with no switch inside the ladder the fit is the plain line.
    line = PW.declaration_fit(samples, treads, None)
    assert line.step_ms is None and line.dof == 4


def test_the_declaration_fit_still_finds_a_real_per_tile_cost_beside_a_step():
    world = PW.WORLDS["machinery"]
    treads = [1, 2, 3, 4, 5, 6]
    samples = PW.planted_samples(
        PW.World("x", "x", {}, machinery_ms_per_tile=0.35, alignment_step_ms=0.4),
        CFG, block_m=32, treads=treads, repeats=3, alpha_shared=world.alpha,
        ridge=160.0, bandwidth_gbps=4000.0, b=2, noise=0.0, seed=0,
        copies_declared=9, native_switch=4)
    fit = PW.declaration_fit(samples, treads, 4)
    assert fit.per_tile_ms == pytest.approx(0.35, rel=1e-6)
    assert fit.step_ms == pytest.approx(0.4, rel=1e-6)


# --------------------------------------------------------------------------
# 12. the step rule's own operating characteristic
# --------------------------------------------------------------------------

#: The per-cell relative noise every step-rule simulation below draws with, and
#: the seed it draws at. BOTH RULES ARE EXACTLY SCALE-FREE IN IT on pure noise:
#: `_series` adds `ms x sigma x z`, `StepFit.step_ms` and `_ols_se`'s residual
#: are linear in the series, and `AlignProbe.spread_ms` is too, so every rate
#: below is a property of the RULE at this geometry and not of any card's
#: microseconds. `test_..._far_less_often_than_the_spread_rule_it_replaced`
#: asserts that invariance rather than trusting it.
STEP_RULE_NOISE = 0.02
STEP_RULE_SEED = 20260917


def _noise_probe(label, repeats, noise, rng, step_ms=0.0, split=4):
    """`repeats` INDEPENDENTLY noised copies of this design's own six-tread
    series, as an `AlignProbe`, so `AlignProbe.series` takes a real median over
    real repeats.

    `_probe_from` cannot be used for this: its jitter is `(rep - 1) x spread`,
    the same factor on every cell of a repeat, so its median series is the
    noiseless one. This one redraws `_series` per repeat, which is what the
    probe does on a pod.
    """
    cells = []
    for rep in range(repeats):
        for n, numel, ms in _series(step_ms, split, noise=noise,
                                    seed=rng.randrange(1 << 30)):
            cells.append(PW.ProbeCell(label, n, numel, 72, rep, ms))
    return PW.AlignProbe(tuple(cells), synthetic=True)


def _old_spread_rule(probe, label, fit):
    """The step rule `read_probe` carried before the six-lens round, verbatim:
    a step is REAL when it exceeds `PROBE_STEP_SIGMA` ACROSS-REPEAT SPREADS of
    the series' own cells. Kept here so the two rules are scored on the same
    worlds, cell for cell."""
    spread = probe.spread_ms(label)
    return (spread is not None and fit.split_tread is not None
            and abs(fit.step_ms) > PW.PROBE_STEP_SIGMA * spread)


def _step_rule_trial(*, repeats, worlds, noise=STEP_RULE_NOISE,
                     seed=STEP_RULE_SEED, step_ms=0.0, split=4):
    """Score both rules on `worlds` simulated probes and return
    `(old_rate, new_rate, found_rate, old_index, new_index)`.

    The rates are the fraction of worlds each rule called REAL; `found_rate` is
    the fraction where the NEW rule resolved a step AND put it at `split`. The
    two indices are the median of `|step_ms| / threshold` under each rule, a
    CONTINUOUS reading of how far the rule is from firing, which is estimated
    far more stably from `worlds` draws than a 5% tail is.
    """
    import random
    import statistics
    rng = random.Random(seed)
    old_hits = new_hits = found = 0
    old_index, new_index = [], []
    for _ in range(worlds):
        probe = _noise_probe(PW.SHARED, repeats, noise, rng, step_ms, split)
        fit = PW.step_fit(probe.series(PW.SHARED))
        if _old_spread_rule(probe, PW.SHARED, fit):
            old_hits += 1
        if fit.resolved():
            new_hits += 1
            if fit.split_tread == split:
                found += 1
        old_index.append(abs(fit.step_ms)
                         / (PW.PROBE_STEP_SIGMA * probe.spread_ms(PW.SHARED)))
        new_index.append(abs(fit.step_ms) / fit.threshold_ms())
    return (old_hits / worlds, new_hits / worlds, found / worlds,
            statistics.median(old_index), statistics.median(new_index))


def _first_disagreement(*, repeats=3, worlds=200, noise=STEP_RULE_NOISE,
                        seed=STEP_RULE_SEED):
    """The first simulated world the OLD rule calls REAL and the NEW one does
    not, as `(probe, fit)`.

    WHY A WITNESS IS NEEDED AT ALL. Every rate in this file is scored on
    `StepFit.resolved` directly, so none of them touches `read_probe`, the
    caller that actually hands V8 its verdict: a `read_probe` reverted to the
    spread would leave all of them green. Only a world the two rules read
    DIFFERENTLY can say which one the caller is using. At this seed the second
    world drawn is already one, and the search is over `worlds` draws so a
    future edit that made disagreements rare would raise rather than pass.
    """
    import random
    rng = random.Random(seed)
    for _ in range(worlds):
        probe = _noise_probe(PW.SHARED, repeats, noise, rng)
        fit = PW.step_fit(probe.series(PW.SHARED))
        if _old_spread_rule(probe, PW.SHARED, fit) and not fit.resolved():
            return probe, fit
    raise AssertionError(f"no world in {worlds} separated the two rules")


def test_the_new_step_rule_fires_on_pure_noise_far_less_often_than_the_spread_rule_it_replaced():
    """`PROBE_STEP_SIGMA`'s docstring claims a MEASURED operating characteristic
    for `StepFit.resolved` and nothing tested it.

    WHAT THIS WOULD HAVE CAUGHT. `read_probe` used to call a step REAL when
    `|step_ms| > PROBE_STEP_SIGMA x AlignProbe.spread_ms`, a threshold on the
    ACROSS-REPEAT spread of a single cell, while `step_fit` had already chosen
    the split as the smallest-RSS of five candidates. Tested as if the split
    had been named in advance, that rule fires on PURE NOISE in 142 of the 600
    worlds below (0.2367). `resolved` prices the same search -- `step_se` from
    `_ols_se`'s own residual, at the Student-t quantile for its own dof
    (`step_quantile`), widened by `selection_penalty(splits_tried)` -- and
    fires in 2 (0.0033). Before the t quantile it used a bare 3 sigma against
    a variance estimated on three degrees of freedom and fired in 42 (0.0700).
    An `_ols_se` that divides its `s2` by `n` instead of `n - k`, or a
    `selection_penalty` that returns 1.0, moves the rate
    assertions below directly; a `read_probe` reverted to the spread leaves
    every one of them green, because they score `resolved` and not the caller,
    and is caught by the witness at the end instead. All three would put V8's
    FAIL and the split handed to `declaration_fit` back on a one-in-four false
    alarm per run.

    HOW THE BOUNDS WERE CHOSEN, so a fixed seed is not a coin flip. At 600
    worlds the binomial standard error of a rate near 0.0033 is
    sqrt(0.0033 x 0.9967 / 600) = 0.0024, so the 0.02 ceiling asserted on the
    new rule sits 7 of those above what it measures here; the old rule's error
    near 0.2367 is sqrt(0.2367 x 0.7633 / 600) = 0.0174, so the 0.15 floor sits
    5.0 below it. Both are bounds on the RATE, not the rate.
    """
    old_rate, new_rate, found_rate, _, _ = _step_rule_trial(
        repeats=3, worlds=600)
    assert old_rate >= 0.15, old_rate
    assert new_rate <= 0.02, new_rate
    # NOT blind, either: POWER is pinned two tests below, at the budget.
    assert found_rate <= new_rate
    # BOTH RATES ARE PROPERTIES OF THE RULE, NOT OF A CARD. `_series` scales
    # every deviation by `noise`, and `step_ms`, `step_se` and `spread_ms` are
    # all homogeneous of degree one in the series, so a ten-fold noisier probe
    # is the same set of verdicts. Allowed one world in 600 for a draw that
    # sits on its own threshold to within float rounding.
    loud_old, loud_new, _, _, _ = _step_rule_trial(
        repeats=3, worlds=600, noise=10 * STEP_RULE_NOISE)
    assert loud_old == pytest.approx(old_rate, abs=1.0 / 600)
    assert loud_new == pytest.approx(new_rate, abs=1.0 / 600)
    # AND THE GATE READS THE NEW RULE, which no rate above can show. At this
    # seed the second world drawn fits a step of +1.00 us that clears 3 of its
    # own spreads (0.96 us) but not its own threshold (1.35 us): the old rule
    # calls it REAL, `resolved` does not, and `read_probe` sides with
    # `resolved`.
    witness, witness_fit = _first_disagreement()
    assert _old_spread_rule(witness, PW.SHARED, witness_fit)
    assert not witness_fit.resolved()
    assert PW.read_probe(witness, PW.SHARED, _census()).real is False


def test_more_probe_repeats_do_not_make_the_new_step_rule_stricter_as_they_made_the_spread_rule():
    """THE DIRECTION, which is the defect the round was really about.

    `AlignProbe.series` is a MEDIAN over `--probe-repeats`, so the noise of the
    number `step_fit` fits falls as the repeats grow; the old threshold,
    `PROBE_STEP_SIGMA x AlignProbe.spread_ms`, is the spread of ONE cell and
    does not move. Measuring more therefore made the old instrument BLINDER:
    on pure noise it fires in 0.2367 of the worlds at 3 repeats, 0.0400 at 5
    and 0.0000 at 9, and its median resolution index `|step_ms| / threshold`
    falls 0.703 -> 0.499 -> 0.336, i.e. a step of a fixed size relative to the
    series' own residual gets HARDER to resolve the more the probe is
    repeated. `StepFit.threshold_ms` is built on `step_se`, which is computed
    from that same median series, so it falls with the repeats the way the
    estimate does: 0.0033 -> 0.0050 -> 0.0000 with an index of
    0.141 -> 0.138 -> 0.141, flat to within 3% across the three.

    WHAT THIS WOULD HAVE CAUGHT. Any threshold that stops reading the fitted
    series and goes back to a per-cell quantity -- `spread_ms`, a fixed
    microsecond floor, `pstdev` of the raw cells -- reintroduces exactly this:
    the V8 UNKNOWN whose remedy line says "more --probe-repeats is what closes
    it" would then be advice that makes the gate less able to close, not more.
    The index bounds are the load-bearing ones: a median over 600 worlds is a
    far steadier statistic than a 5% tail, and the old rule's index ratio
    0.336/0.703 = 0.478 is nowhere near the new rule's 0.141/0.141 = 1.00.
    """
    by_repeats = {r: _step_rule_trial(repeats=r, worlds=600) for r in (3, 5, 9)}
    old = {r: v[0] for r, v in by_repeats.items()}
    new = {r: v[1] for r, v in by_repeats.items()}
    old_index = {r: v[3] for r, v in by_repeats.items()}
    new_index = {r: v[4] for r, v in by_repeats.items()}
    # THE PATHOLOGY: the old rule's false alarms are extinguished by the very
    # repeats that were supposed to sharpen it.
    assert old[3] > old[5] > old[9], old
    assert old[3] >= 0.15 and old[9] <= 0.01, old
    assert old_index[9] < 0.6 * old_index[3], old_index
    # ITS ABSENCE: the new rule neither collapses nor runs away.
    assert all(rate <= 0.02 for rate in new.values()), new
    assert 0.85 <= new_index[9] / new_index[3] <= 1.15, new_index
    assert 0.85 <= new_index[5] / new_index[3] <= 1.15, new_index


def test_a_step_at_the_alignment_budget_is_still_found_by_the_new_rule():
    """POWER, the half of an operating characteristic a stricter rule can buy
    by refusing everything.

    A rule that never fires has no false positives, so the two simulations
    above are only worth having beside this one. The steps here are named in
    the units V8 actually gates on: `step_bias` of 0.010 and 0.015 of the
    ratio, one and one-and-a-half `ALIGN_STEP_RATIO_BUDGET`, converted to
    milliseconds through `leverage` at the six-tread ladder's split of 4 and
    the 0.64 ms weight stream the other V8 tests use. At the same per-cell
    noise the false-positive simulation runs at, `resolved` finds the smaller
    in 383 of 400 worlds (0.9575) and the larger in 399 (0.9975), and puts
    the split at tread 4 every time it fires.

    WHAT THE t QUANTILE COST, stated rather than hidden. Before it, a step
    worth a fifth of the budget was found 0.9525 of the time; now 0.2475, and
    two fifths 0.6525. That is the price of pricing a variance estimated on
    three degrees of freedom, and it does not move a V8 verdict: a step under
    the budget PASSES whether or not it is resolved.

    WHAT THIS WOULD HAVE CAUGHT. `selection_penalty` grows without bound in
    `splits_tried`, and `threshold_ms` multiplies it by `PROBE_STEP_SIGMA`:
    a "safer" edit that raised either -- or an `_ols_se` that overstated
    `step_se` -- would trade this power away silently, and V8 would answer
    UNKNOWN on a design whose step really is over budget, which is the answer
    that costs a pod session. The 0.90 floor sits 5.7 binomial standard errors
    (sqrt(0.9575 x 0.0425 / 400) = 0.0101) below the measured 0.9575.
    """
    treads = [1, 2, 3, 4, 5, 6]
    weight_stream_ms = 0.64
    worth = {frac: frac * PW.ALIGN_STEP_RATIO_BUDGET * weight_stream_ms
             / PW.leverage(treads, 4) for frac in (1.0, 1.5)}
    assert PW.step_bias(worth[1.0], treads, 4, weight_stream_ms) \
        == pytest.approx(0.010)
    assert PW.step_bias(worth[1.5], treads, 4, weight_stream_ms) \
        == pytest.approx(0.015)
    _, small_rate, small_found, _, _ = _step_rule_trial(
        repeats=3, worlds=400, step_ms=worth[1.0])
    assert small_rate >= 0.90, small_rate
    assert small_found >= 0.90, small_found
    _, big_rate, big_found, _, _ = _step_rule_trial(
        repeats=3, worlds=400, step_ms=worth[1.5])
    assert big_rate >= 0.98, big_rate
    assert big_found >= 0.98, big_found


def test_the_selection_penalty_is_sqrt_two_log_k_and_is_what_widens_the_threshold():
    """`selection_penalty` prices the fact that `step_fit` REPORTS THE MINIMUM
    RSS over every candidate split, so the step it hands back is the largest of
    `splits_tried` draws and not one named in advance.

    WHAT THIS WOULD HAVE CAUGHT. The arithmetic has two traps a plausible edit
    walks into: `math.log(1)` is 0.0 and `math.log(0)` raises, so a
    `selection_penalty` written without the `splits_tried > 1` guard either
    zeroes `threshold_ms` -- every step REAL, V8 FAILs on noise -- or throws
    `ValueError` out of the middle of a gate. It has to be 1.0 for a single
    candidate, rising, and it has to reach `threshold_ms` multiplicatively
    beside `PROBE_STEP_SIGMA` rather than being folded into `step_se`.
    """
    assert PW.selection_penalty(0) == 1.0
    assert PW.selection_penalty(1) == 1.0
    assert PW.selection_penalty(5) == pytest.approx(math.sqrt(2.0 * math.log(5)))
    # The closed form beside it only says the two spellings agree, so the shape
    # is pinned at two points by numbers as well, which a reader can check
    # against a table without running this function.
    assert PW.selection_penalty(5) == pytest.approx(1.7941, abs=5e-5)
    assert PW.selection_penalty(12) == pytest.approx(2.2293, abs=5e-5)
    widths = [PW.selection_penalty(k) for k in range(1, 13)]
    assert widths == sorted(widths) and len(set(widths)) == len(widths)
    # A six-tread ladder offers five splits, so 1.79 is the one this design
    # pays, and the threshold is exactly the t quantile at its three dof x se
    # x that penalty.
    fit = PW.step_fit(_series(0.0, 4, noise=STEP_RULE_NOISE, seed=1))
    assert fit.splits_tried == 5 and fit.dof == 3
    assert fit.threshold_ms() == pytest.approx(
        PW.step_quantile(PW.PROBE_STEP_SIGMA, 3) * fit.step_se
        * PW.selection_penalty(5))
    assert fit.threshold_ms(6.0) > 2.0 * fit.threshold_ms(3.0)
    # AND A FIT WITH NO DEGREE OF FREEDOM LEFT IS REFUSED, naming the count,
    # rather than returning `step_se == 0.0` and a `resolved` hard-wired False.
    with pytest.raises(PW.Unmeasurable, match="3 treads cannot resolve a step"):
        PW.step_fit(_series(0.05, 3)[:3])


def test_v8_passes_a_flat_ratio_series_fails_a_real_step_and_doubts_a_noisy_one():
    """The three verdicts V8 can reach from a probe, plus the no-probe UNKNOWN.

    THE THIRD BRANCH IS THE ONE THIS ROUND MOVED. It used to build its doubtful
    probe with `_probe_from(..., spread=5.0)`, but that jitter is
    `(rep - 1) x spread` applied to a whole repeat, so `AlignProbe.series`,
    which takes a MEDIAN over the three repeats, returns the same series for
    any spread at all. The old rule read `AlignProbe.spread_ms` and so was
    fooled by it; `StepFit.resolved` reads the standard error of the fitted
    coefficient on the median series, which that probe leaves bit-identical to
    the resolved one -- it now reads FAIL. The branch therefore carries its
    noise where the new rule looks: in the series itself, through `_series`'s
    own `noise=`. At seed 2 that series fits a step of +0.141 ms at tread 4,
    worth 0.056 of the ratio -- 5.6 times `ALIGN_STEP_RATIO_BUDGET` -- against
    a threshold of 0.272 ms, so the step is 0.52 of what it would have to be:
    over budget, unresolved, and therefore neither shown sound nor unsound.
    """
    treads = [1, 2, 3, 4, 5, 6]
    flat = _probe_from({PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4)})
    assert PW.gate_v8_alignment(flat, treads=treads, census=_census(),
                                weight_stream_ms=0.64).verdict == exit_codes.PASS
    split = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.5, 4)}, spread=1e-4)
    gate = PW.gate_v8_alignment(split, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.FAIL, gate.lines
    # Over budget but the series' OWN residual swallows it: not shown either
    # way. The two conditions are asserted before the verdict, so a future
    # UNKNOWN reached for another reason -- a host-bound cell, an unprobed
    # declaration -- cannot pass this branch by accident.
    noisy = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.1, 4, noise=0.3, seed=2)})
    fit = PW.step_fit(noisy.series(PW.SHARED))
    assert PW.step_bias(fit.step_ms, treads, fit.split_tread, 0.64) \
        > PW.ALIGN_STEP_RATIO_BUDGET
    assert abs(fit.step_ms) < fit.threshold_ms()
    doubted = PW.gate_v8_alignment(noisy, treads=treads, census=_census(),
                                   weight_stream_ms=0.64)
    assert doubted.verdict == exit_codes.UNKNOWN, doubted.lines
    assert any("over budget but NOT resolved" in line for line in doubted.lines)
    assert PW.gate_v8_alignment(None, treads=treads, census=_census(),
                                weight_stream_ms=0.64).verdict == exit_codes.UNKNOWN


# --------------------------------------------------------------------------
# 13. a host-bound probe times the HOST, so V8 says UNKNOWN
# --------------------------------------------------------------------------

def _verdict_probe(verdicts_by_label):
    """An `AlignProbe` whose cells carry EXACTLY the named `(host_bound,
    host_note)` per tread, repeated three times like `_probe_from`.

    `_probe_from` stamps ONE verdict on every cell it makes, so it cannot
    express the case `AlignProbe.host_bound` has to reduce: a declaration
    holding a None cell, a True cell and a False cell at once. Each list is
    one entry per tread of `_series`, in tread order.
    """
    cells = []
    for label, verdicts in verdicts_by_label.items():
        series = _series(0.0, 4)
        for rep in range(3):
            for (n, numel, ms), (hot, note) in zip(series, verdicts, strict=True):
                cells.append(PW.ProbeCell(label, n, numel, 72, rep, ms,
                                          host_bound=hot, host_note=note))
    return PW.AlignProbe(tuple(cells), synthetic=True)


#: The two notes `timing.host_bound_verdict` ITSELF writes, taken from the
#: instrument rather than copied into this file. A note copied by hand drifts
#: the moment the instrument rewords itself, and prose that no code produces is
#: this repository's recurring defect; the real hot note is also 364 characters
#: with `;`, `:` and parentheses inside it, so carrying it end to end is what
#: shows the page passes the instrument's sentence through WHOLE rather than
#: summarising or splitting it.
#:
#: The hot walls are host-bound by the instrument's own arithmetic: three
#: trials of 100 iterations that spent 0.99 s of their 1.00 s wall in the
#: enqueue loop end with a GPU backlog of 0.01 s, which is under the
#: `HOST_BOUND_BACKLOG_ITERS` = 2 iterations of the trial's own 0.01 s
#: per-iteration wall that `host_bound_verdict` thresholds against. The
#: unjudged note is the one it returns for no trials at all.
_HOT_VERDICT, _, _, _HOT_NOTE = TIMING.host_bound_verdict(
    [TIMING.TrialWall(enqueue_s=0.99, wall_s=1.0)] * 3, 100)
_UNJUDGED_VERDICT, _, _, _UNJUDGED_NOTE = TIMING.host_bound_verdict([], 0)

#: One declaration's planted verdicts for `_verdict_probe`, one per tread of
#: `_series`: a cell the instrument could not judge, two it called host-bound,
#: three it cleared. The None cell carries a note of its own, so a reader of
#: `AlignProbe.host_bound` that took the note off the first cell WITH a note
#: rather than off the first HOT cell would report the wrong sentence.
_MIXED_VERDICTS = [(None, _UNJUDGED_NOTE), (True, _HOT_NOTE), (True, _HOT_NOTE),
                   (False, ""), (False, ""), (False, "")]


def test_the_planted_host_bound_notes_are_the_instruments_own_sentences():
    """THREE TESTS BELOW REST ON THIS FIXTURE, so the fixture is checked first:
    the notes they plant have to be the ones `timing.host_bound_verdict`
    returns, not a paraphrase that would let the V8 page carry a sentence the
    instrument never writes. This asserts what the two planted walls MEAN --
    one set host-bound, one unjudgeable -- and that the hot note is the whole
    sentence, remedy clause included, rather than a truncation of it.
    """
    assert (_HOT_VERDICT, _UNJUDGED_VERDICT) == (True, None)
    assert _UNJUDGED_NOTE == "no trials; host-bound not determinable"
    assert _HOT_NOTE.startswith("host-bound: in 3 of 3 trials")
    # The instrument's note ENDS with its own remedy; a note cut short at the
    # first clause is the fake this fixture used to plant.
    assert _HOT_NOTE.endswith("to measure the GPU alone")


def test_v8_is_unknown_on_a_host_bound_probe_even_where_the_series_is_flat_and_the_bias_is_zero():
    """THE HOST-BOUND BRANCH BEATS THE PASS BRANCH, which is the whole point of
    it. `gate_v8_alignment` reads `AlignProbe.host_bound(ratio_label)` BEFORE
    it compares `bias` with `ALIGN_STEP_RATIO_BUDGET`, because an alignment
    call is tens of microseconds -- the size of the host's own enqueue cost --
    so a host-bound cell timed the HOST and its flatness is a fact about the
    wrong machine. Planted identically on both sides: the same flat series, a
    bias of exactly 0.0000, and only `ProbeCell.host_bound` moved. Ordered the
    other way round the gate would have certified the design off a probe that
    never timed the kernel the ratio arms run.
    """
    treads = [1, 2, 3, 4, 5, 6]
    # NATIVE planted FLAT: its switch is due at tread 4 and the probe does not
    # show it, so the positive control fails and cannot rescue the hot probe.
    flat = {PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.0, 4)}
    hot = PW.gate_v8_alignment(_probe_from(flat, host_bound=True),
                               treads=treads, census=_census(),
                               weight_stream_ms=0.64)
    cool = PW.gate_v8_alignment(_probe_from(flat, host_bound=False),
                                treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert hot.verdict == exit_codes.UNKNOWN, hot.lines
    assert cool.verdict == exit_codes.PASS, cool.lines
    # The bias branch AGREED with PASS in both: the same flat series gives a
    # measured bias of 0.0000 either way, so nothing but the host-bound
    # verdict separated UNKNOWN from PASS.
    assert hot.measured == cool.measured
    assert hot.measured.startswith("bias <= 0.0000")


def test_v8_reads_the_host_bound_verdict_before_the_budget_so_an_over_budget_real_step_is_unknown():
    """The other side of the same ordering, and the one a PASS-only test cannot
    reach: a probe whose ratio series carries a REAL 0.5 ms step is FAIL when
    the cells were GPU-bound and UNKNOWN when they were host-bound. A step in
    the host's enqueue cost is not evidence that vLLM switched alignment
    kernel inside the ratio arms' ladder, so V8 must not spend its FAIL on it
    either. If the host-bound test sat in the `elif` chain after `r.real` this
    would still read FAIL and the run would be refused for the wrong reason.
    """
    treads = [1, 2, 3, 4, 5, 6]
    # NO `spread=` HERE, DELIBERATELY, and the neighbouring tests that pass one
    # are passing a dead argument: `_probe_from` scales a WHOLE REPEAT by
    # `(rep - 1) * spread` and `AlignProbe.series` takes the per-tread MEDIAN
    # over repeats, so the middle repeat -- jitter exactly 0 -- IS the series.
    # Measured on this construction: split_tread, step_ms, step_se, rss_with,
    # threshold_ms and resolved() are bit-identical at spread 0.0, 1e-4 and
    # 5.0. Writing 1e-4 here would claim a noisy probe and plant none.
    # NATIVE flat, so the positive control fails and the hot probe stays
    # unscored; with the control confirmed it would be FAIL, which the
    # control test below pins.
    stepped = {PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.5, 4)}
    cool = PW.gate_v8_alignment(_probe_from(stepped, host_bound=False),
                                treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    hot = PW.gate_v8_alignment(_probe_from(stepped, host_bound=True),
                               treads=treads, census=_census(),
                               weight_stream_ms=0.64)
    assert cool.verdict == exit_codes.FAIL, cool.lines
    assert hot.verdict == exit_codes.UNKNOWN, hot.lines
    # Both scored the SAME over-budget real step, so the budget cannot be what
    # separated them: the 0.5 ms step sits at tread 4, whose leverage over
    # these six treads is 4.5/17.5 = 0.2571, and 0.2571 x 0.5 / 0.64 ms of
    # weight stream is a bias of 0.2009 -- twenty times ALIGN_STEP_RATIO_BUDGET.
    assert hot.measured == cool.measured == "bias <= 0.2009, a REAL step"
    assert PW.step_bias(0.5, treads, 4, 0.64) > 20 * PW.ALIGN_STEP_RATIO_BUDGET
    # WHY `cool` IS FAIL AND NOT UNKNOWN, recorded because it is not obvious
    # from the verdict alone: the planted series fits the step term EXACTLY, so
    # the residual is at the rounding floor (RSS 5.1e-31) and `step_se` is
    # 7.0e-16 ms, a number out of rounding rather than out of noise.
    # `resolved()` clears it only because 0.5 ms beats a threshold of
    # 3.8e-15 ms, a margin of 1.3e14. If a floor is ever put under `step_se`,
    # THIS line reports it rather than the bare verdict assert above.
    fit = PW.step_fit(_probe_from(stepped).series(PW.SHARED))
    assert fit.step_se > 0.0
    assert abs(fit.step_ms) > 1e6 * fit.threshold_ms()


def test_v8_is_unknown_when_no_probed_cell_carried_a_host_bound_verdict_at_all():
    """`ProbeCell.host_bound` is `bool | None` and its docstring says None is
    "not determinable", WHICH IS NOT THE SAME AS False. `timing.host_bound_verdict`
    returns None when a trial spanned no wall time or there were no trials, so
    a probe of Nones is an instrument that answered nothing, not an instrument
    that cleared the cells. The gate's `judged == 0` arm exists for that: read
    as False it would have PASSED this flat series and called the design sound
    on a probe with no verdict in it.
    """
    treads = [1, 2, 3, 4, 5, 6]
    # NATIVE flat: an unjudged probe is rescued by the positive control
    # exactly as a host-bound one is, so the control is planted absent here.
    flat = {PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.0, 4)}
    blind = _probe_from(flat, host_bound=None)
    assert blind.host_bound(PW.SHARED) == (0, 0, "")
    gate = PW.gate_v8_alignment(blind, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.UNKNOWN, gate.lines
    assert any("returned no host-bound verdict for any probed cell" in ln
               for ln in gate.lines), gate.lines


def test_align_probe_host_bound_counts_one_declaration_at_a_time_and_notes_a_hot_cell():
    """`AlignProbe.host_bound(label)` filters `self.cells` by `label` first, so
    one declaration's verdicts cannot be counted into another's. It matters
    here because the probe times NATIVE's declaration beside the ratio arms'
    in the same `AlignProbe`, and NATIVE is the arm allowed its switch: a
    reduction over ALL cells would let NATIVE's host-bound cells send V8
    UNKNOWN about a SHARED series the instrument had cleared. The counts are
    3 repeats x the per-tread verdicts in `_MIXED_VERDICTS`, and the note is
    taken off the first HOT cell, not off the first cell that has one.
    """
    probe = _verdict_probe({PW.SHARED: _MIXED_VERDICTS,
                            PW.NATIVE: [(False, "")] * 6})
    per_tread = _MIXED_VERDICTS
    want_hot = 3 * sum(1 for hot, _ in per_tread if hot)
    want_judged = 3 * sum(1 for hot, _ in per_tread if hot is not None)
    assert (want_hot, want_judged) == (6, 15)
    assert probe.host_bound(PW.SHARED) == (want_hot, want_judged, _HOT_NOTE)
    # NATIVE's own 18 cells were all cleared, and SHARED's 6 hot ones did not
    # leak into them.
    assert probe.host_bound(PW.NATIVE) == (0, 3 * 6, "")
    # PRIVATE shares SHARED's declaration and is not probed separately, so it
    # has no cells at all: no verdict, and the note is empty rather than
    # SHARED's.
    assert probe.host_bound(PW.PRIVATE) == (0, 0, "")


def test_v8_is_not_sent_unknown_by_native_cells_the_instrument_called_host_bound():
    """THE GATE-LEVEL HALF of the test above, which the unit assertions cannot
    reach. NATIVE's declaration is probed in the SAME `AlignProbe` as the ratio
    arms' and is not scored by V8 at all, so a `host_bound` that reduced over
    every cell -- the obvious implementation, and the one the label filter
    exists to rule out -- would read 18 hot here and return UNKNOWN about a
    SHARED series the instrument had cleared end to end. That is a refusal to
    quote the ratio bought entirely with cells from the arm V8 does not score.
    """
    treads = [1, 2, 3, 4, 5, 6]
    probe = _verdict_probe({PW.SHARED: [(False, "")] * 6,
                            PW.NATIVE: [(True, _HOT_NOTE)] * 6})
    assert probe.host_bound(PW.NATIVE) == (18, 18, _HOT_NOTE)
    assert probe.host_bound(PW.SHARED) == (0, 18, "")
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.PASS, gate.lines
    # And the page reports the RATIO arms' count, not the probe's: 0 of 18.
    assert ("the instrument called 0 of 18 probed cells HOST-BOUND at this "
            "declaration") in gate.lines, gate.lines
    assert not any(_HOT_NOTE in ln for ln in gate.lines), gate.lines


def test_the_v8_page_names_how_many_cells_were_host_bound_and_the_remedy_for_it():
    """PROSE DRIFT IS THIS REPOSITORY'S RECURRING DEFECT: a gate that returns
    UNKNOWN without saying what it could not time sends a reader back to the
    probe's raw cells. The line must carry the COUNT (so a reader sees whether
    one cell or every cell was host-bound), the instrument's own note, and the
    reason the positive control did not rescue it (NATIVE's switch not seen
    where the census puts it), because neither is guessable from "UNKNOWN".
    """
    treads = [1, 2, 3, 4, 5, 6]
    probe = _verdict_probe({PW.SHARED: _MIXED_VERDICTS,
                            PW.NATIVE: [(False, "")] * 6})
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.UNKNOWN, gate.lines
    assert ("the instrument called 6 of 15 probed cells HOST-BOUND at this "
            f"declaration: {_HOT_NOTE}") in gate.lines, gate.lines
    # The note came off a HOT cell. The unjudged cell sits FIRST in the
    # declaration and carries a note of its own, so a reduction that took the
    # note off the first cell that HAS one would put this sentence on the page
    # instead, and the page would name the wrong reason for the UNKNOWN.
    assert not any(_UNJUDGED_NOTE in ln for ln in gate.lines), gate.lines
    assert any(ln.startswith("POSITIVE CONTROL NOT CONFIRMED: NATIVE's switch "
                             "is due at tread 4 and the host-bound probe "
                             "resolved no step") for ln in gate.lines), gate.lines
    # And a cleared probe says so on the same line rather than staying silent,
    # so "0 of 18" is a positive record that the instrument did look.
    cool = PW.gate_v8_alignment(
        _probe_from({PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4)}),
        treads=treads, census=_census(), weight_stream_ms=0.64)
    assert ("the instrument called 0 of 18 probed cells HOST-BOUND at this "
            "declaration") in cool.lines, cool.lines


# --------------------------------------------------------------------------
# 13b. NATIVE as the host-bound probe's POSITIVE CONTROL
# --------------------------------------------------------------------------

def test_v8_passes_a_host_bound_flat_series_when_natives_switch_resolves_at_the_census_tread():
    """THE RENTED-CARD CASE. On an H200 the probe is expected to be host-bound
    (~30-45 us host against ~6-10 us GPU per call), and V8 used to return
    UNKNOWN for every host-bound probe -- which skipped the sweep, so a rented
    pod produced no ladder at all. NATIVE declares E=8, under the expert
    bound, and the ladder crosses the id bound between treads 3 and 4, so its
    kernel switches at `census.switch_tread(NATIVE)` == 4. A probe that
    RESOLVES that step at tread 4 has shown it can see a kernel switch of this
    op at this size through the host cost; a flat ratio series from it is
    then evidence, and PASS is earned. Pre-change this reads UNKNOWN.
    """
    treads = [1, 2, 3, 4, 5, 6]
    assert _census().switch_tread(PW.NATIVE) == 4
    probe = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.0, 4)}, host_bound=True)
    assert probe.host_bound(PW.SHARED)[:2] == (18, 18)
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.PASS, gate.lines
    assert any(ln.startswith("POSITIVE CONTROL CONFIRMED") for ln in gate.lines)
    # The assumption the control rests on is REGISTERED on the page.
    assert any("ASSUMED: the host's enqueue cost does not itself step at that "
               "tread" in ln for ln in gate.lines), gate.lines


def test_the_confirmed_control_lets_a_host_bound_probe_fail_a_real_ratio_step():
    """Once the control has shown the probe sensitive, the ratio series is
    scored like a GPU-bound one in BOTH directions: a real 0.5 ms step (bias
    0.2009, twenty budgets) is FAIL, not the UNKNOWN it was pre-change."""
    treads = [1, 2, 3, 4, 5, 6]
    probe = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.5, 4)}, host_bound=True)
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.FAIL, gate.lines


def test_natives_step_at_the_wrong_tread_is_not_a_positive_control():
    """The control is a PREDICTION, not "some step somewhere": NATIVE resolved
    at tread 3 when the census says 4 is a probe seeing something other than
    the switch it was shown, and it certifies nothing."""
    treads = [1, 2, 3, 4, 5, 6]
    probe = _probe_from({PW.NATIVE: _series(0.02, 3),
                         PW.SHARED: _series(0.0, 4)}, host_bound=True)
    native = PW.read_probe(probe, PW.NATIVE, _census())
    assert native.real and native.fit.split_tread == 3
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.UNKNOWN, gate.lines
    assert any("resolved its step at tread 3, not 4" in ln for ln in gate.lines)


def test_no_census_switch_means_no_positive_control():
    """A ladder that never crosses the id bound gives NATIVE no switch to show
    the probe, so a host-bound probe stays UNKNOWN and says the control was
    UNAVAILABLE rather than failed."""
    treads = [4, 5, 6, 7]
    census = PW.path_census(CFG, treads, 32,
                            {PW.NATIVE: 8, PW.SHARED: 72, PW.PRIVATE: 72})
    assert census.switch_tread(PW.NATIVE) is None
    line = [(n, 256 * n, 0.012 + 8e-6 * 256 * n) for n in treads]
    probe = _probe_from({PW.NATIVE: line, PW.SHARED: line}, host_bound=True)
    gate = PW.gate_v8_alignment(probe, treads=treads, census=census,
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.UNKNOWN, gate.lines
    assert any(ln.startswith("POSITIVE CONTROL UNAVAILABLE") for ln in gate.lines)


def test_the_host_bound_worlds_separate_on_the_control_alone():
    """The two planted host-bound worlds differ ONLY in whether NATIVE's probe
    step is planted, and that alone moves V8 between UNKNOWN and PASS."""
    a, b = PW.WORLDS["host-bound-probe"], PW.WORLDS["host-bound-controlled"]
    assert a.probe_host_bound and b.probe_host_bound
    assert a.native_probe_step_ms == 0.0 and b.native_probe_step_ms is None
    assert a.expect["V8"] == exit_codes.UNKNOWN and b.expect["V8"] == exit_codes.PASS


# --------------------------------------------------------------------------
# 13c. the step rule at the ladder's own degrees of freedom
# --------------------------------------------------------------------------

def test_step_quantile_matches_the_t_table_and_refuses_no_dof():
    """`step_quantile(3, dof)` is the t quantile with the two-sided tail
    erfc(3/sqrt 2) = 0.0026998. One dof is Cauchy, closed form
    cot(pi p / 2); the rest against published t tables to four figures."""
    p = math.erfc(3.0 / math.sqrt(2.0))
    assert PW.step_quantile(3.0, 1) == pytest.approx(
        1.0 / math.tan(math.pi * p / 2.0), rel=1e-9)
    assert PW.step_quantile(3.0, 2) == pytest.approx(19.2067, abs=5e-4)
    assert PW.step_quantile(3.0, 3) == pytest.approx(9.2189, abs=5e-4)
    assert PW.step_quantile(3.0, 10) == pytest.approx(3.9569, abs=5e-4)
    assert PW.step_quantile(3.0, 100000) == pytest.approx(3.0, abs=1e-3)
    for dof in (1, 2, 3, 7, 30):
        assert PW.t_two_sided_tail(PW.step_quantile(3.0, dof), dof) \
            == pytest.approx(p, rel=1e-8)
    for dof in (0, -1):
        with pytest.raises(PW.Unmeasurable):
            PW.step_quantile(3.0, dof)


def _short_ladder_noise_rate(treads: int, worlds: int = 600) -> float:
    """Pure-noise false-REAL rate of `StepFit.resolved` on the first `treads`
    points of `_series`, 3 independently noised repeats, median series."""
    import random
    rng = random.Random(STEP_RULE_SEED)
    hits = 0
    for _ in range(worlds):
        probe = _noise_probe(PW.SHARED, 3, STEP_RULE_NOISE, rng)
        hits += PW.step_fit(probe.series(PW.SHARED)[:treads]).resolved()
    return hits / worlds


def test_the_step_rule_is_calibrated_at_four_and_five_treads_not_only_six():
    """THE DEFECT. `s2 = RSS/(n-3)` on one or two degrees of freedom judged
    against a bare 3 sigma fired on pure noise 0.428 of the time at 4 treads
    and 0.155 at 5 (the rule before it: 0.233), a rule calibrated at six
    treads only. At the t quantile both sit under 0.02."""
    assert _short_ladder_noise_rate(4) <= 0.02
    assert _short_ladder_noise_rate(5) <= 0.02


def test_ols_se_returns_nan_and_not_zero_when_no_dof_is_left():
    """A zero standard error made every coefficient look exact; NaN cannot be
    mistaken for a measurement."""
    rows = [[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0]]
    coefs, ses = PW._ols_se(rows, [1.0, 2.5, 3.0])
    assert all(math.isfinite(c) for c in coefs)
    assert all(math.isnan(e) for e in ses)


def test_three_treads_are_refused_and_four_still_fail_a_real_step():
    """`--self-test ratio-path-split --treads 3` read V8 UNKNOWN on a planted
    501 us step, 39x the budget, because dof was 0 and `step_se` exactly 0.0.
    Now the command is REFUSED with the count named, before a card is
    touched; at 4 treads, one dof, the same planted step still FAILs V8."""
    three = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test", "ratio-path-split",
         "--treads", "3"], capture_output=True, text=True, cwd=ROOT)
    assert three.returncode == exit_codes.REFUSED, three.stdout
    assert "--treads 3 gives 3 tread(s)" in three.stdout
    assert "leave 0 degrees of freedom" in three.stdout
    four = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test", "ratio-path-split",
         "--treads", "4"], capture_output=True, text=True, cwd=ROOT)
    assert "SELF-TEST OK" in four.stdout, four.stdout[-2000:]
    v8 = [r for r in exit_codes.parse_result_lines(four.stdout)
          if r.name == "V8"]
    assert [r.verdict for r in v8] == [exit_codes.FAIL], v8


# --------------------------------------------------------------------------
# 14. the bias bound on the side the denominator shrinks
# --------------------------------------------------------------------------

def test_step_bias_on_a_positive_step_still_divides_by_the_whole_weight_stream():
    """A regression pin on the half of `step_bias` the correction left alone.
    Splitting the function on the sign of `step_ms` is a chance to move the
    branch that was already right: for `s >= 0` the fitted denominator
    `B_p + L s` GROWS, so `weight_stream_ms` is the whole scale the move is
    taken against and `moved` is never subtracted from it. Two consequences
    are pinned here because only the negative branch may have them: the
    positive branch has no `inf` guard, so a step worth more than the stream
    is a large bias rather than an unmeasurable design, and a `step_ms` of
    exactly 0.0 takes this branch and comes back 0.0 because `moved` is 0,
    not because a guard caught it.
    """
    treads = [1, 2, 3, 4, 5, 6]
    lev = 4.5 / 17.5
    assert PW.leverage(treads, 4) == pytest.approx(lev)
    for step_ms, stream_ms in ((0.01, 0.5), (0.25, 0.64), (4.0, 0.64)):
        assert PW.step_bias(step_ms, treads, 4, stream_ms) == pytest.approx(
            lev * step_ms / stream_ms)
    # 4.0 ms of step is 4.0 x 4.5/17.5 = 1.0286 ms of slope against a 0.64 ms
    # stream: past the point where the NEGATIVE branch returns `inf`, and
    # still a finite 1.607 here.
    big = PW.step_bias(4.0, treads, 4, 0.64)
    assert math.isfinite(big) and big == pytest.approx((4.0 * 4.5 / 17.5) / 0.64)
    assert big == pytest.approx(1.6071, abs=5e-5)
    assert PW.step_bias(0.0, treads, 4, 0.64) == 0.0


def test_step_bias_on_a_negative_step_shrinks_the_denominator_and_doubles_at_half_the_stream():
    """The defect the sign split exists for. A step `s` common to both ratio
    arms enters each straight-line fit as `L s` of slope, so the ratio is
    `(B_s + L s)/(B_p + L s)`, and for `s < 0` the DENOMINATOR falls to
    `B_p - L|s|`. The old single-branch form divided by `weight_stream_ms` on
    both sides, so on the negative side it reported a move against a
    denominator the step had already shrunk. The worked case in the new
    docstring is derived here rather than quoted: set the stream to exactly
    twice `L|s|` and the negative bound is `moved / (2 moved - moved)` = 1.0
    against the positive branch's `moved / (2 moved)` = 0.5, a factor of two.
    """
    treads = [1, 2, 3, 4, 5, 6]
    lev = PW.leverage(treads, 4)
    step_ms = 0.7
    moved = lev * step_ms
    # `L|s|` is exactly half the stream, which is where the shrunk denominator
    # equals the whole one and the two branches stand in a ratio of two.
    stream_ms = 2.0 * moved
    assert PW.step_bias(step_ms, treads, 4, stream_ms) == pytest.approx(0.5)
    assert PW.step_bias(-step_ms, treads, 4, stream_ms) == pytest.approx(1.0)
    assert PW.step_bias(-step_ms, treads, 4, stream_ms) == pytest.approx(
        2.0 * PW.step_bias(step_ms, treads, 4, stream_ms))
    # Strictly larger at every magnitude short of consuming the stream, by
    # exactly the denominator the positive branch does not shrink. The stream
    # is consumed at 0.64 x 17.5/4.5 = 2.4889 ms; 2.0 ms is the largest
    # magnitude in this sweep and still leaves 0.64 - 0.5143 = 0.1257 ms of it.
    stream_ms = 0.64
    assert stream_ms / lev == pytest.approx(2.4889, abs=5e-5)
    for step_ms in (0.001, 0.01, 0.1, 0.5, 1.0, 2.0):
        moved = lev * step_ms
        assert moved < stream_ms
        positive = PW.step_bias(step_ms, treads, 4, stream_ms)
        negative = PW.step_bias(-step_ms, treads, 4, stream_ms)
        assert positive == pytest.approx(moved / stream_ms)
        assert negative == pytest.approx(moved / (stream_ms - moved))
        assert negative > positive


def test_step_bias_is_inf_when_a_negative_step_has_consumed_the_weight_stream():
    """`inf` is a refusal and not a big number. When `L|s|` reaches
    `weight_stream_ms` the fitted denominator `B_p + L s` has been driven to
    zero or through it, and a ratio formed against that denominator is not a
    measurement of anything, so `step_bias` declines to put a number on it.
    The old form returned `L|s| / weight_stream_ms` there, a finite number
    that reads as a bias on a ratio that no longer exists. The guard belongs
    to the negative branch alone, which is pinned by the same magnitude
    positive coming back finite. A `weight_stream_ms` of zero or less is the
    other nothing-left-to-measure state and was already `inf`.
    """
    treads = [1, 2, 3, 4, 5, 6]
    lev = PW.leverage(treads, 4)
    step_ms = 2.0
    moved = lev * step_ms
    # Exactly at the boundary the denominator is 0, and past it it is negative.
    assert PW.step_bias(-step_ms, treads, 4, moved) == math.inf
    assert PW.step_bias(-step_ms, treads, 4, 0.5 * moved) == math.inf
    # The same magnitude with the sign flipped is finite, and is what the old
    # form returned for BOTH signs: moved/moved = 1.0 and moved/(moved/2) = 2.0.
    assert PW.step_bias(step_ms, treads, 4, moved) == pytest.approx(1.0)
    assert PW.step_bias(step_ms, treads, 4, 0.5 * moved) == pytest.approx(2.0)
    for stream_ms in (0.0, -0.1):
        assert PW.step_bias(0.01, treads, 4, stream_ms) == math.inf
        assert PW.step_bias(-0.01, treads, 4, stream_ms) == math.inf


def _slope_of_a_ladder_carrying(step_ms: float, per_tile_ms: float,
                                treads: list[int], split: int,
                                intercept_ms: float) -> float:
    """The slope `fit_line` reads off one arm whose per-tread time is
    `intercept_ms + per_tile_ms n`, with `step_ms` added from `split` up.
    `fit_line` is the script's own ladder estimator (`ladder_for` calls it),
    so what comes back is the slope the ratio is really formed from and not a
    stand-in for it.

    REFUSES a ladder carrying a non-positive time. A large negative step on a
    shallow arm drives cells through zero, and `fit_line` would still return a
    slope for that (its residual term skips `y <= 0` and says nothing), so the
    caller has to keep its planted designs to ladders a card could produce.
    """
    pts = [(n, intercept_ms + per_tile_ms * n + (step_ms if n >= split else 0.0))
           for n in treads]
    assert min(ms for _n, ms in pts) > 0.0, f"a ladder no card could produce: {pts}"
    return PW.fit_line(pts)[1]


def test_step_bias_bounds_what_a_common_step_does_to_a_fitted_ratio_on_both_sides():
    """THE TEST THE CORRECTION IS FOR: the bound checked against the thing it
    bounds, rather than against its own algebra. Two ladders are built with
    per-tile costs `B_p = stream` and `B_s = alpha B_p`, one step common to
    both is planted at tread 4, both are fitted with `fit_line`, and
    `slope(SHARED)/slope(PRIVATE)` is compared with the unstepped `alpha`. The
    move is `L|s| |B_p - B_s| / (B_p |B_p + L s|)`, so the denominator that
    shrinks on the negative side is the FIT's, not a modelling choice.

    `step_bias` holds at every magnitude and both signs. The old form,
    `L|s| / B_p` for both signs, is violated on the negative side, and this
    pins WHERE: `move > L|s|/B_p` reduces to `L|s| > B_s`, so the old formula
    stops bounding once the fraction `L|s|/B_p` passes `alpha`, which is
    exactly when the negative step has eaten the whole numerator slope. Two
    alphas, so the crossing is a property of the arithmetic rather than of one
    planted pair.

    WHAT THIS IS AND IS NOT EVIDENCE OF. The fractions here run to 0.90 of the
    stream, which at this leverage is a step of 2.24 ms: 90x the 24.76 us of
    the reachable case below, so "the old form was not a bound" is arithmetic
    and not a design the probe could hand the gate. The REACHABLE consequence
    of the correction is the verdict flip in the last test in this file, which
    needs 24.76 us.
    """
    treads = [1, 2, 3, 4, 5, 6]
    split = 4
    lev = PW.leverage(treads, split)
    stream_ms = 0.64
    fractions = (0.05, 0.30, 0.50, 0.60, 0.80, 0.90)
    old_form_violations = 0
    for alpha in (0.55, 0.20):
        for fraction in fractions:
            step_ms = fraction * stream_ms / lev
            for signed in (step_ms, -step_ms):
                # 3.0 ms and 2.0 ms of intercept: arbitrary to the slope, which
                # is what is read, but enough that the deepest negative step
                # (2.24 ms, at fraction 0.90) leaves every cell above zero.
                shared = _slope_of_a_ladder_carrying(
                    signed, alpha * stream_ms, treads, split, 3.0)
                private = _slope_of_a_ladder_carrying(
                    signed, stream_ms, treads, split, 2.0)
                # The planted step reaches each fit as exactly `L s` of slope.
                assert shared == pytest.approx(alpha * stream_ms + lev * signed)
                assert private == pytest.approx(stream_ms + lev * signed)
                move = abs(shared / private - alpha)
                bound = PW.step_bias(signed, treads, split, stream_ms)
                assert move <= bound + 1e-12, (alpha, fraction, signed, move, bound)
                old_form = lev * abs(signed) / stream_ms
                broke = move > old_form + 1e-12
                assert broke == (signed < 0.0 and fraction > alpha), (
                    alpha, fraction, signed, move, old_form)
                if broke:
                    old_form_violations += 1
    # Three fractions clear alpha = 0.55 and five clear alpha = 0.20, each on
    # the negative side only: of the 2 x 6 x 2 = 24 planted designs the old
    # form was not a bound on 8.
    assert old_form_violations == 8


def _signed_step_probe(step_ms: float, *, noise: float = 1e-5, seed: int = 3):
    """A three-repeat alignment probe whose SHARED series carries `step_ms` at
    tread 4, on a 0.12 ms + 8 ns/id line.

    The line is ten times `_series`'s 0.012 ms intercept because a NEGATIVE
    step of tens of microseconds on the shallower one drives cells to negative
    times, which no probe can produce. `noise` is a per-cell relative draw,
    there only so `step_fit`'s residual is real and `StepFit.step_se` is a
    standard error rather than the rounding of an exactly-fitted series, which
    is what `ProbeReading.real` is judged against.
    """
    import random
    rng = random.Random(seed)
    cells = []
    for label, step in ((PW.NATIVE, 0.02), (PW.SHARED, step_ms)):
        for rep in range(3):
            for n in range(1, 7):
                numel = 256 * n
                ms = 0.12 + 8e-6 * numel + (step if n >= 4 else 0.0)
                cells.append(PW.ProbeCell(label, n, numel, 72, rep,
                                          ms * (1.0 + rng.gauss(0.0, noise)),
                                          host_bound=False))
    return PW.AlignProbe(tuple(cells), synthetic=True)


def test_v8_fails_a_negative_ratio_arm_step_the_positive_only_bias_scored_inside_budget():
    """The correction changes a VERDICT, not just a printed number. The window
    where it does is arithmetic: the old form passes when
    `L|s|/ws <= ALIGN_STEP_RATIO_BUDGET` and the new one goes over budget when
    `L|s|/(ws - L|s|) > ALIGN_STEP_RATIO_BUDGET`, i.e. when
    `L|s|/ws > 0.01/1.01 = 0.009901`, so every fraction in (0.009901, 0.01] is
    PASS under the old form and, once the step is resolved against its own
    standard error, FAIL under the new one. At ws = 0.64 ms and L = 4.5/17.5
    the fraction 0.00995 is a step of 24.76 us, the tens of microseconds an
    alignment call is, so `gate_v8_alignment` reaching a different verdict here
    is a reachable design and not an arithmetic curiosity. The same magnitude
    with the sign flipped still PASSES, which is the whole content of the fix:
    the gate now reads a bound that knows which side of the fit the denominator
    shrinks on.
    """
    treads = [1, 2, 3, 4, 5, 6]
    stream_ms = 0.64
    lev = PW.leverage(treads, 4)
    budget = PW.ALIGN_STEP_RATIO_BUDGET
    fraction = 0.00995
    assert budget / (1.0 + budget) < fraction <= budget
    step_ms = fraction * stream_ms / lev
    assert step_ms * 1e3 == pytest.approx(24.76, abs=0.01)

    down = _signed_step_probe(-step_ms)
    read = PW.read_probe(down, PW.SHARED, _census())
    assert read.fit.split_tread == 4 and read.real
    old_form = lev * abs(read.fit.step_ms) / stream_ms
    corrected = PW.step_bias(read.fit.step_ms, treads, 4, stream_ms)
    assert old_form <= budget < corrected
    gate = PW.gate_v8_alignment(down, treads=treads, census=_census(),
                                weight_stream_ms=stream_ms)
    assert gate.verdict == exit_codes.FAIL, gate.lines

    up = _signed_step_probe(step_ms)
    assert PW.step_bias(PW.read_probe(up, PW.SHARED, _census()).fit.step_ms,
                        treads, 4, stream_ms) <= budget
    assert PW.gate_v8_alignment(up, treads=treads, census=_census(),
                                weight_stream_ms=stream_ms).verdict == exit_codes.PASS


# --------------------------------------------------------------------------
# 15. the probe's own repeat floor
# --------------------------------------------------------------------------

def _probe_at_repeats(series_by_label, repeats, spread=0.0, host_bound=False):
    """`_probe_from` with the REPEAT COUNT a parameter, which is the one thing
    `--probe-repeats` moves and the one thing `_probe_from`'s `range(3)` holds
    fixed. `host_bound=False` because a cell with no verdict sends V8 down its
    `judged == 0` branch, which is UNKNOWN whatever the step is.
    """
    cells = []
    for label, series in series_by_label.items():
        for rep in range(repeats):
            jitter = (rep - (repeats - 1) / 2.0) * spread
            for n, numel, ms in series:
                cells.append(PW.ProbeCell(label, n, numel, 72, rep,
                                          ms * (1.0 + jitter),
                                          host_bound=host_bound))
    return PW.AlignProbe(tuple(cells), synthetic=True)


@pytest.mark.parametrize("typed", ["1", "0", "-3"])
def test_a_probe_repeat_count_under_the_floor_is_refused_naming_the_flag_typed(typed):
    """THE KNOB HAD NO FLOOR AND ITS FLOOR IS THE GATE'S. `--repeats` and
    `--treads` were both refused at plan time; `--probe-repeats` was accepted
    at any value, so `--probe-repeats 1` bought the whole probe, printed a
    V8 line whose across-repeat spread read NOT DETERMINED, and put an
    unreplicated `StepFit.step_se` under a VALIDITY verdict. The refusal has
    to name the flag the operator typed and the floor, or it sends them
    looking at `--repeats`, which is a different number with a different
    reason.

    AND WHAT IT MUST NOT SAY. The message used to end `V8 ... could not
    reach its FAIL branch at all. A gate that cannot fail is not a gate.`,
    which was true of the spread rule and false of `StepFit.resolved`, whose
    standard error comes off the fit's own residual that a one-repeat series
    has; the test below measures V8 FAILing at one repeat. The replication
    clause is the floor's whole justification now, and the false headline is
    asserted absent.
    """
    got = run(["--probe-repeats", typed, "--dry-run", "--device-memory-gb", "140"])
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:]
    assert (f"REFUSED: --probe-repeats {typed} is below "
            f"{PW.MIN_PROBE_REPEATS}.") in got.stdout, got.stdout[:2000]
    assert "A single pass forms no across-repeat spread" in got.stdout
    assert "V8" in got.stdout
    assert "could not reach its FAIL branch" not in got.stdout
    assert "A gate that cannot fail is not a gate" not in got.stdout
    # Refused BEFORE anything is measured, so no gate is scored: the shape
    # every other plan-time refusal in this file has.
    assert "RESULT: " not in got.stdout


def test_the_probe_repeat_floor_is_read_after_the_treads_floor_and_before_the_repeats_one():
    """ORDER IS BEHAVIOUR HERE, because a refusal that names a flag the
    operator did not type sends them to fix the wrong one. `_main` checks
    `len(treads) < MIN_TREADS`, then `args.probe_repeats < MIN_PROBE_REPEATS`,
    then `args.repeats < MIN_REPEATS`, and each returns immediately -- so with
    two floors broken at once exactly one refusal is printed. Pinned in both
    directions: `--probe-repeats 1 --repeats 1` must report the probe knob and
    say nothing about the ladder's, and `--treads 2 --probe-repeats 1` must
    report the treads knob, which is checked first.
    """
    both = run(["--probe-repeats", "1", "--repeats", "1",
                "--dry-run", "--device-memory-gb", "140"])
    assert both.returncode == exit_codes.REFUSED
    assert f"--probe-repeats 1 is below {PW.MIN_PROBE_REPEATS}." in both.stdout
    assert f"--repeats 1 is below the {PW.MIN_REPEATS}" not in both.stdout
    assert "every ladder here needs" not in both.stdout
    assert both.stdout.count("REFUSED:") == 1, both.stdout[:2000]

    short = len(PW.ladder_treads(CFG, PW.DEFAULT_BLOCK_M, 2))
    assert short < PW.MIN_TREADS, "the shallow ladder must break the tread floor"
    treads_first = run(["--treads", "2", "--probe-repeats", "1",
                        "--dry-run", "--device-memory-gb", "140"])
    assert treads_first.returncode == exit_codes.REFUSED
    assert f"--treads 2 gives {short} tread(s)" in treads_first.stdout
    assert "--probe-repeats" not in treads_first.stdout
    assert treads_first.stdout.count("REFUSED:") == 1, treads_first.stdout[:2000]


def test_the_probe_repeat_floor_is_read_before_the_run_needs_a_calibrated_device():
    """A PLAN-TIME FLOOR HAS TO COST NOTHING, which is a statement about where
    it sits relative to `SWEEP.resolve_ridge`, not about its wording. Without
    `--device-memory-gb` and without `--dry-run` this command reaches
    `resolve_ridge`, which on a machine with no CUDA device refuses with `no
    calibration for this device`. `--probe-repeats 1` must stop BEFORE that,
    on its own refusal, so an operator who mistyped the knob is told about the
    knob rather than about a card. The ridge refusal's absence is the whole
    assertion: were the floor read after the resolve, the same command would
    print it and the flag would never be mentioned.
    """
    early = run(["--probe-repeats", "1"])
    assert early.returncode == exit_codes.REFUSED, early.stdout[-2000:]
    assert f"--probe-repeats 1 is below {PW.MIN_PROBE_REPEATS}." in early.stdout
    assert "no calibration for this device" not in early.stdout
    assert early.stdout.count("REFUSED:") == 1, early.stdout[:2000]
    assert "RESULT: " not in early.stdout


def test_two_probe_repeats_clear_the_floor_and_are_priced_into_the_plan():
    """THE FLOOR IS AT THE LOWEST COUNT THAT STILL FORMS A SPREAD, not at the
    default. `MIN_PROBE_REPEATS` is 2 and `PROBE_REPEATS` is 3, so the default
    run is not refused by its own floor and the operator keeps one step of
    room below it. `--probe-repeats 2` must therefore reach the plan, and the
    plan must PRICE it: `probe_seconds(treads, PROBE_LABELS, repeats)` is `3
    series (one per arm) x 6 treads x repeats x (PROBE_WARMUP_MS + PROBE_TRIALS
    x PROBE_TARGET_MS)` = `3 x 6 x repeats x 140 ms`, which is 5.04 s at two
    repeats and 7.56 s at three.

    THE TWO FIGURES ARE ASSERTED AS FIGURES, not re-derived from the same
    constants the function multiplies, because a test that recomputes
    `probe_seconds`' one line and compares cannot notice the budget constants
    moving underneath the prose above. If PROBE_WARMUP_MS, PROBE_TRIALS or
    PROBE_TARGET_MS changes, 5.04 and 7.56 are what has to be re-derived and
    this docstring is what has to be rewritten.
    """
    assert PW.MIN_PROBE_REPEATS == 2
    assert PW.MIN_PROBE_REPEATS < PW.PROBE_REPEATS

    treads = PW.ladder_treads(CFG, PW.DEFAULT_BLOCK_M, PW.DEFAULT_TREADS)
    assert len(treads) == 6, treads
    at_two = PW.probe_seconds(treads, PW.PROBE_LABELS, PW.MIN_PROBE_REPEATS)
    at_default = PW.probe_seconds(treads, PW.PROBE_LABELS, PW.PROBE_REPEATS)
    assert at_two == pytest.approx(5.04), at_two
    assert at_default == pytest.approx(7.56), at_default
    # A floor on a knob that changed nothing downstream would show up as one
    # figure twice on the plan page.
    assert f"{at_two:.0f}" != f"{at_default:.0f}", (at_two, at_default)

    got = run(["--probe-repeats", str(PW.MIN_PROBE_REPEATS),
               "--dry-run", "--device-memory-gb", "140"])
    assert got.returncode == exit_codes.REFUSED          # the DRY RUN's refusal
    assert "REFUSED: --probe-repeats" not in got.stdout
    assert "reason: --dry-run was given" in got.stdout
    assert f"probe's {at_two:.0f} s" in got.stdout


def test_at_one_probe_repeat_no_spread_is_formed_and_the_step_verdict_is_the_fits_own_residual():
    """WHAT ONE REPEAT ACTUALLY COSTS V8, taken off the code rather than off
    the refusal's wording, because the two do not agree.

    TRUE, and the floor's first clause: `AlignProbe.spread_ms` groups a
    declaration's cells by tread and keeps `pstdev` only where `len(v) > 1`,
    so at one repeat every group has one cell, the median is over an empty
    list and the method returns None. `ProbeReading.lines()` then prints
    `across-repeat spread NOT DETERMINED`, and the operator reads a V8 line
    with no measurement of the instrument's own noise on it.

    NOT TRUE of this code, and the reason this test exists: the refusal used
    to say V8 `could not reach its FAIL branch at all` at one repeat. That held
    of the rule THIS PATCH REPLACED, where `read_probe` set
    `real = (spread is not None and ...)` and a None spread forced UNKNOWN.
    `StepFit.resolved` reads `step_se`, which `_ols_se` takes from the fit's
    own residual over `len(treads) - 3` degrees of freedom, a quantity one
    repeat has, so a one-repeat probe carrying a 500 us step at the ratio
    arms' declaration reaches FAIL. Pinned so the floor's justification is not
    carried forward as a property of the scorer.

    AND THE MIDDLE CLAUSE IS PINNED FOR WHAT IT IS WORTH, which is less than
    it reads. `AlignProbe.series` medians over repeats, so the fit always sees
    `len(treads)` points however often the probe ran: `splits_tried` and the
    degrees of freedom behind `step_se` are the same at 1, 2 and 3 repeats.
    More repeats move the MEDIAN the error is computed on; they never buy the
    error more replication. So the floor's real content is narrow and is the
    first clause alone: below two repeats there is no across-repeat spread to
    print beside the verdict.
    """
    treads = [1, 2, 3, 4, 5, 6]
    census = _census()
    noisy = {PW.NATIVE: _series(0.02, 4, noise=0.004, seed=11),
             PW.SHARED: _series(0.5, 4, noise=0.004, seed=3)}
    one = _probe_at_repeats(noisy, 1)
    two = _probe_at_repeats(noisy, 2, spread=1e-3)

    assert one.spread_ms(PW.SHARED) is None
    assert two.spread_ms(PW.SHARED) is not None
    assert "across-repeat spread NOT DETERMINED" in "\n".join(
        PW.read_probe(one, PW.SHARED, census).lines())

    # The fit sees one point per TREAD at every repeat count, so nothing the
    # knob does reaches the residual the standard error is taken on.
    for probe in (one, two, _probe_at_repeats(noisy, 3, spread=1e-3)):
        f = PW.step_fit(probe.series(PW.SHARED))
        assert len(probe.series(PW.SHARED)) == len(treads)
        assert f.splits_tried == len(treads) - 1
        assert f.step_se > 0.0, "every repeat count still leaves a residual"

    fit = PW.step_fit(one.series(PW.SHARED))
    assert abs(fit.step_ms) > fit.threshold_ms()
    assert fit.resolved()
    assert PW.gate_v8_alignment(one, treads=treads, census=census,
                                weight_stream_ms=0.64).verdict == exit_codes.FAIL
    # The two rules read this same one-repeat probe in OPPOSITE directions,
    # which is the disagreement the refusal's last sentence still describes.
    old_real = (one.spread_ms(PW.SHARED) is not None
                and fit.split_tread is not None
                and abs(fit.step_ms) > PW.PROBE_STEP_SIGMA
                * (one.spread_ms(PW.SHARED) or 0.0))
    assert old_real is False and fit.resolved() is True


def test_the_probe_repeat_refusal_reads_back_through_the_exit_code_contract():
    """THE REFUSAL SHORT-CIRCUITS A MODE THAT WOULD OTHERWISE SCORE GATES, so
    it has to leave a log the contract at the top of this file accepts. A
    scoring mode's log must recompute its own exit code; `--self-test refit`
    is such a mode and returns DONE with RESULT lines. Adding
    `--probe-repeats 1` must turn it into the other shape exactly -- REFUSED,
    no RESULT line, `classify_text` raising `NoGatesScored` -- and not into a
    log that prints a verdict and exits 2, which is the defect
    `moe/bench/exit_codes.py` exists to prevent. The DONE leg is also what
    shows the floor does not fire at the default `--probe-repeats`.
    """
    scored = run(["--self-test", "refit"])
    assert scored.returncode == exit_codes.DONE, scored.stdout[-2000:]
    assert exit_codes.parse_result_lines(scored.stdout)

    refused = run(["--self-test", "refit", "--probe-repeats", "1"])
    assert refused.returncode == exit_codes.REFUSED
    assert "RESULT: " not in refused.stdout
    assert exit_codes.parse_result_lines(refused.stdout) == []
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(refused.stdout)


# --------------------------------------------------------------------------
# report.json carries no bare NaN
# --------------------------------------------------------------------------

#: The three tokens `json.dumps` writes for non-finite floats. Python's own
#: `json.loads` decodes all three; RFC 8259 has a production for none of them,
#: so a document carrying one parses here and nowhere else.
JSON_NON_FINITE_TOKENS: tuple[str, ...] = ("NaN", "Infinity", "-Infinity")

#: A JSON string literal, escapes included, so the scan below can delete every
#: string before looking for a token in VALUE position.
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')

#: The `--draws` the "an interval WAS formed" pass is driven at. `DEFAULT_DRAWS`
#: is 2000 and every draw refits BOTH ladders twice over (`ratio_interval` and
#: `declaration_interval` each take the count), which costs about a second a
#: world; fifteen worlds at the default is about fifteen seconds of an
#: eighteen-minute suite for a document whose SHAPE is reached at any count the
#: percentile rule accepts. MEASURED, not assumed: at 16 draws every world's
#: payload has the same keys, the same formed interval and the same two
#: declaration bands as at 2000. The shipped count is still driven once, at the
#: end of the test, so the configuration the pod runs is not the one this file
#: never runs.
FORMED_DRAWS = 16


def _bare_non_finite_tokens(text: str) -> list[str]:
    """The non-finite tokens standing as VALUES in `text`, strings deleted.

    `parse_constant` is the check that cannot be fooled; this one runs beside
    it only so that a failure NAMES the token. It searches the capitalised
    spellings `json.dumps` writes, and it deletes string literals before
    looking, so what it reports is a token in value position and never a word
    in a note. That is not hypothetical: the document really does carry the
    lowercase spelling inside strings -- `gate_c1_ratio` renders an unformed
    interval through an f-string, where `math.nan` prints as "nan", in its
    `measured` line and again in the basis line under it, and the key
    "provenance" spells those three letters too. A scan relaxed to catch those
    would be one capitalisation away from accepting the bare value it is here
    to find.
    """
    outside_strings = _JSON_STRING.sub('""', text)
    return [token for token in JSON_NON_FINITE_TOKENS
            if token in outside_strings]


def _no_json_constants(token: str):
    """`parse_constant` for `json.loads`, which refuses instead of decoding.

    Python calls this for NaN, Infinity and -Infinity and for no other token,
    so it converts "this document is Python-JSON, not JSON" into a failure.
    This is the ANYWHERE check: it fires on a bare token in any field, not only
    on `ratio_interval`, which is the one field the fix reached.
    """
    raise AssertionError(f"the document carries the bare token {token!r}, "
                         "which no strict JSON parser accepts")


def _payload_for(argv: list[str]) -> tuple[int, dict, str]:
    """`(exit code, the payload `_main` would serialise, the run's log)`.

    A PLANTED WORLD WRITES NO FILE: `_main` guards `report.txt` and
    `report.json` with `if not synthetic`, so `--self-test` never reaches the
    writer and the only way to hold a planted world's document is to take the
    payload `analyse` hands back, which is the object the writer serialises.
    `analyse` is WRAPPED rather than called again with reconstructed arguments,
    so the payload under test is the one the script's own argument assembly
    produced.
    """
    captured: list = []
    real = PW.analyse

    def wrapper(*args, **kwargs):
        report = real(*args, **kwargs)
        captured.append(report)
        return report

    PW.analyse = wrapper
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            rc = PW._main(argv)
    finally:
        PW.analyse = real
    assert captured, f"{argv} never reached analyse:\n{log.getvalue()[-600:]}"
    return rc, captured[0].payload, log.getvalue()


def _analyse(samples, treads: list[int], *, block_m: int = 32,
             draws: int = 10, copies: int = 9):
    """`PW.analyse` over planted cells, with what a planted run hands it.

    The memory plan and the buffer proof `_main` builds for a synthetic world,
    no probe, and the census `analyse` derives itself when it is given none.
    The ridge and bandwidth are the ones the planted-world tests above this
    one use, so the ladders come out at the same slopes.
    """
    mem = PW.memory_plan(CFG, "bf16", 2, copies,
                         PW.SWEEP.tokens_for_rows(CFG, treads[-1] * block_m),
                         140_000_000_000, "HYPOTHETICAL: no card is attached")
    return PW.analyse(
        samples, CFG, block_m=block_m, treads=treads, repeats=3,
        alpha=PW.ALPHA, dtype="bf16", b=2, bandwidth_gbps=4000.0,
        bandwidth_source="this test", ridge=160.0, ridge_source="this test",
        roof_tflops=989.0, roof_source="this test", reference_mhz=None,
        reference_grade="", reference_source="this test", mem=mem,
        proof=PW.planted_proof(True), weight_delta_bytes=mem.weight_bytes,
        high_water_bytes=mem.predicted_peak_bytes, draws=draws, seed=0,
        header=[], card="no card", synthetic=True,
        model_name=PW.DEFAULT_MODEL, pinned={}, copies_declared=copies)


def test_an_interval_that_was_not_formed_is_null_in_the_payload_and_not_nan():
    """`analyse` opens with `interval = (math.nan, math.nan)` and leaves it
    there when `ratio_interval` raises `Unmeasurable`, and the payload used to
    write `list(interval)` into `report.json`: `json.dumps` then emitted the
    bare token `NaN` twice, which Python's own loader accepts and every strict
    parser rejects. `--draws 1` takes that path by `ratio_interval`'s own rule,
    since one draw cannot produce the two ratios a percentile needs, while the
    ratio itself is still fitted -- so what is asserted here is the INTERVAL
    and not an absent ladder.
    """
    rc, payload, log = _payload_for(["--self-test", "refit", "--draws", "1"])
    assert "interval NOT FORMED: only 1 of 1 bootstrap draws" in log
    assert payload["ratio"] is not None, "the ratio itself must still be formed"
    assert payload["ratio_interval"] == [None, None]
    assert payload["ratio_draws"] == 0
    text = json.dumps(payload, indent=2)
    assert _bare_non_finite_tokens(text) == []
    json.loads(text, parse_constant=_no_json_constants)
    # The starved run's OWN verdict, named rather than ignored: the 'refit'
    # world registers C1 PASS, `c1_verdict` cannot score a claim off a
    # non-finite interval, and a planted world that came out other than
    # registered is ERROR and not any code in the gate table. Both of these
    # held BEFORE the fix too; they are here to say which run this is, and the
    # assertions above are the ones the fix moved.
    assert "SELF-TEST MISMATCH  C1: registered PASS, got UNKNOWN" in log
    assert rc == exit_codes.ERROR


@pytest.mark.parametrize("stubbed,expected", [
    ((0.56, math.inf, 7), [0.56, None]),
    ((-math.inf, 0.56, 7), [None, 0.56]),
    ((math.nan, math.inf, 7), [None, None]),
])
def test_analyse_nulls_each_non_finite_end_of_the_interval_on_its_own(
        stubbed, expected):
    """Report assembly is a FUNCTION here so that a serialisation break is a
    unit test rather than a pod finding. Pre-fix it wrote `list(interval)`, so
    an endpoint that came back infinite reached `json.dumps` as the bare token
    `Infinity`. The map has to null THAT END and keep the other: an interval
    flattened to `[null, null]` whenever either end is non-finite would throw
    away a bound the bootstrap did produce. BOTH ENDS ARE DRIVEN, because a map
    written over the second element alone passes a one-sided case and is still
    wrong. `ratio_interval` is stubbed rather than starved because its own
    refusal path yields two non-finite ends at once and so cannot tell a
    per-element map from a whole-value one; the third row is that refusal's own
    shape, kept so the per-element map is shown to agree with it.
    """
    treads = [1, 2, 3, 4, 5, 6]
    world = PW.WORLDS["refit"]
    samples = PW.planted_samples(world, CFG, block_m=32, treads=treads,
                                 repeats=3, alpha_shared=world.alpha,
                                 ridge=160.0, bandwidth_gbps=4000.0, b=2,
                                 noise=0.0, seed=0, copies_declared=9,
                                 native_switch=4)
    real = PW.ratio_interval
    PW.ratio_interval = lambda *a, **k: stubbed
    try:
        payload = _analyse(samples, treads).payload
    finally:
        PW.ratio_interval = real
    assert payload["ratio_interval"] == expected
    assert payload["ratio_draws"] == 7
    text = json.dumps(payload, indent=2)
    assert _bare_non_finite_tokens(text) == []
    json.loads(text, parse_constant=_no_json_constants)


def test_every_planted_worlds_whole_report_document_is_strict_json(tmp_path):
    """THE FIELD THE FIX REACHED IS NOT THE CLAIM. `report.json` is read
    downstream by strict parsers, so what has to hold is that no field emits a
    bare NaN, Infinity or -Infinity -- which is why this drives every world in
    `WORLDS` and checks the WHOLE serialised document with `parse_constant`,
    rather than reading `ratio_interval` back out of a dict.

    Each world is driven twice: once where the bootstrap FORMS an interval and
    the two declaration bands, and once at `--draws 1`, where `ratio_interval`
    and `declaration_interval` both refuse and `analyse` keeps the `(nan, nan)`
    it opened with. The second pass is the one that bites: pre-fix every world
    produced two bare `NaN` tokens there, and the first passed. The two passes
    are asserted to BE two -- an interval and bands on one side, none on the
    other -- so that a future change to the refusal rule cannot quietly turn
    the formed pass into a second starved one and leave the formed half of the
    document unread.

    WHAT THIS DOES NOT ESTABLISH, said plainly: the fix nulls `ratio_interval`
    and nothing else, and no planted world drives any other field non-finite,
    so what is proven here is that these fifteen worlds serialise clean. A
    non-finite `per_tile_band` or `step_band` on real cells would still reach
    `json.dumps` as a bare token; that is an open hole in the script, not a
    hole this test can close.

    The exit code is not what this test reads, and it has no single shape: a
    world whose registration names C1 exits ERROR when starved, since C1
    cannot be scored off an interval that was not formed and a planted world
    that came out other than registered is ERROR, while a world an earlier
    gate already refuses exits INVALID, never having reached C1.
    """
    source = SCRIPT.read_text()
    assert 'write_text(json.dumps(report.payload, indent=2))' in source, (
        "the document serialised below is no longer the one `_main` writes")
    # AND NOTHING IS WRITTEN FOR A PLANTED WORLD, which is why the payload and
    # not a file is what gets read: `--out` is honoured, the directory is
    # created in the measured branch alone, and a self-test leaves it absent.
    out = tmp_path / "runs"
    _payload_for(["--self-test", "refit", "--draws", "1", "--out", str(out)])
    assert not out.exists(), sorted(p.name for p in out.rglob("*"))

    for name in sorted(PW.WORLDS):
        for draws in (FORMED_DRAWS, 1):
            _rc, payload, _log = _payload_for(
                ["--self-test", name, "--draws", str(draws)])
            text = json.dumps(payload, indent=2)
            assert _bare_non_finite_tokens(text) == [], (name, draws)
            json.loads(text, parse_constant=_no_json_constants)
            formed = draws != 1
            where = (name, draws)
            assert (payload["ratio_draws"] > 0) is formed, where
            assert (None not in payload["ratio_interval"]) is formed, where
            decl = payload["declaration_fit"]
            assert (decl["per_tile_band"] is not None) is formed, where
            assert (decl["step_band"] is not None) is formed, where

    # AND ONCE AT THE SHIPPED COUNT. Every pass above trades `DEFAULT_DRAWS`
    # for a count that reaches the same document forty times cheaper; this one
    # run says the count the pod will actually use reaches it too.
    _rc, payload, _log = _payload_for(["--self-test", "refit"])
    assert payload["ratio_draws"] > FORMED_DRAWS, "this pass took --draws"
    text = json.dumps(payload, indent=2)
    assert _bare_non_finite_tokens(text) == []
    json.loads(text, parse_constant=_no_json_constants)


# --------------------------------------------------------------------------
# 16. the page a V8 early exit prints: analyse() over NO samples at all
# --------------------------------------------------------------------------

#: The worlds in which `run_sweep` returns before a weight is allocated, and
#: the V8 verdict each leaves on the page. FAIL ONLY: an UNKNOWN is a
#: statement about the instrument, and the ladder runs past it (see
#: `test_a_v8_unknown_runs_the_sweep_and_only_a_fail_skips_it`).
V8_EARLY_EXITS: tuple[tuple[str, str], ...] = (
    ("ratio-path-split", exit_codes.FAIL),
    ("ratio-step-over-budget", exit_codes.FAIL),
)


def _stand_in_for_vllm(monkeypatch):
    """Put non-importing stand-ins for vLLM's fused-MoE modules into
    `sys.modules` for the length of one test.

    `run_sweep` resolves `SWEEP.find_override`'s `override_config` hook and
    imports `fused_experts` and `MoEActivation` BEFORE it times the alignment
    probe, so its early return cannot be reached on a host where those imports
    fail, which is every host this suite runs on EXCEPT the pod. None of the
    three stand-ins is ever called: the early return happens before the first
    cell. `monkeypatch.setitem` removes them again, so on the pod, where vLLM
    is installed and these shadow it, every other test in this file and every
    later file sees vLLM's own modules back.
    """
    fused = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fused.override_config = lambda *a, **k: None
    fused.fused_experts = lambda *a, **k: None
    activation = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.activation")
    activation.MoEActivation = object
    fused.activation = activation
    layers = types.ModuleType("vllm.model_executor.layers")
    layers.fused_moe = fused
    executor = types.ModuleType("vllm.model_executor")
    executor.layers = layers
    root = types.ModuleType("vllm")
    root.model_executor = executor
    for name, mod in (("vllm", root),
                      ("vllm.model_executor", executor),
                      ("vllm.model_executor.layers", layers),
                      ("vllm.model_executor.layers.fused_moe", fused),
                      ("vllm.model_executor.layers.fused_moe.activation",
                       activation)):
        monkeypatch.setitem(sys.modules, name, mod)


class _ReachedAllocation(Exception):
    """Raised where `run_sweep` resets the peak-memory counter, the first line
    past the V8 early return: reaching it is what "the sweep ran" means off a
    GPU."""


def _skip_the_sweep(monkeypatch, tmp_path, capsys, world_name: str,
                    reach: bool = False):
    """Drive the real `run_sweep` to its V8 early return in `world_name`, and
    hand back what it returned, what it printed and the plan the page needs.

    The probe is the only thing measured before that return, so
    `probe_alignment` is replaced by `planted_probe` for the named world and
    every other line of `run_sweep` is the shipped code. `build_private_weights`
    is replaced by a sentinel that RAISES: running past the early return is
    then an error that names itself rather than a later assertion guessing at
    it, and its silence is this file's proof that no weight was allocated.
    """
    treads = list(range(1, PW.DEFAULT_TREADS + 1))
    block_m = PW.DEFAULT_BLOCK_M
    copies_declared, _why = PW.declared_copies_for(CFG, treads, block_m, None)
    declared_by_arm = {arm: PW.declared_experts(arm, CFG.num_experts,
                                                copies_declared)
                       for arm in PW.ARMS}
    census = PW.path_census(CFG, treads, block_m, declared_by_arm)
    world = PW.WORLDS[world_name]
    args = PW.build_parser().parse_args(["--device-memory-gb", "140"])
    stream_ms = WEIGHTS.weight_stream_ms(CFG, args.dtype, 4000.0)

    def planted(cfg, **kwargs):
        return PW.planted_probe(world, cfg, block_m=block_m, treads=treads,
                                declared_by_arm=declared_by_arm, census=census,
                                noise=0.0, seed=0, weight_stream_ms=stream_ms)

    def never(*a, **k):
        raise AssertionError("build_private_weights ran past the early return")

    _stand_in_for_vllm(monkeypatch)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "unused-cache"))
    monkeypatch.setattr(PW, "probe_alignment", planted)
    monkeypatch.setattr(PW, "build_private_weights", never)

    out_dir = tmp_path / world_name
    out_dir.mkdir(parents=True)
    csv_path = out_dir / "cells.csv"
    store = PW.Store(csv_path, PW.CSV_FIELDS + PW.PROVENANCE_COLUMNS)
    if reach:
        import torch

        def reached(*a, **k):
            raise _ReachedAllocation
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reached)
        with pytest.raises(_ReachedAllocation):
            PW.run_sweep(
                args, CFG, block_m=block_m, treads=treads, pinned={},
                csv_path=csv_path, cache_root=out_dir / "triton-cache",
                store=store, prov=None, dtype=args.dtype,
                copies_declared=copies_declared, census=census,
                stream_ms=stream_ms)
        return types.SimpleNamespace(log=capsys.readouterr().out,
                                     treads=treads, census=census,
                                     stream_ms=stream_ms)
    samples, proof, weight_delta, high_water, probe = PW.run_sweep(
        args, CFG, block_m=block_m, treads=treads, pinned={},
        csv_path=csv_path, cache_root=out_dir / "triton-cache", store=store,
        prov=None, dtype=args.dtype, copies_declared=copies_declared,
        census=census, stream_ms=stream_ms)
    return types.SimpleNamespace(
        samples=samples, proof=proof, weight_delta=weight_delta,
        high_water=high_water, probe=probe, log=capsys.readouterr().out,
        args=args, treads=treads, block_m=block_m, census=census,
        copies_declared=copies_declared, csv_path=csv_path,
        stream_ms=stream_ms)


def _page_after(skipped):
    """`analyse` over a skipped sweep, with the argument shape `_main` hands it
    on exactly this path: the early return's empty `samples`, its empty
    `BufferProof`, `None` for both memory observations, and the probe that
    ended the run. Everything else is what `_main` computes before `run_sweep`
    and does not recompute after it."""
    treads, block_m = skipped.treads, skipped.block_m
    b = PW.dtype_bytes(skipped.args.dtype)
    mem = PW.memory_plan(
        CFG, skipped.args.dtype, b, skipped.copies_declared,
        PW.SWEEP.tokens_for_rows(CFG, treads[-1] * block_m),
        int(140e9), "--device-memory-gb", copies_read=treads[-1])
    prov = PW.PV.provenance_block(
        instrument="a stand-in basis, so the block is the shape a pod's is",
        ridge=160.0, ridge_source="fixed for this test",
        bandwidth=4000.0, bandwidth_source="fixed for this test",
        warmup_ms=skipped.args.warmup, iters=None,
        target_ms=skipped.args.cell_budget_ms)
    return PW.analyse(
        skipped.samples, CFG, block_m=block_m, treads=treads,
        repeats=skipped.args.repeats, alpha=skipped.args.alpha,
        dtype=skipped.args.dtype, b=b, bandwidth_gbps=4000.0,
        bandwidth_source="fixed for this test", ridge=160.0,
        ridge_source="fixed for this test", roof_tflops=640.0,
        roof_source="fixed for this test", reference_mhz=None,
        reference_grade="", reference_source="fixed for this test",
        mem=mem, proof=skipped.proof,
        weight_delta_bytes=skipped.weight_delta,
        high_water_bytes=skipped.high_water, draws=skipped.args.draws,
        seed=skipped.args.seed, header=["PLAN"], card="NVIDIA H200",
        synthetic=False, model_name=PW.DEFAULT_MODEL, pinned={},
        prov=PW._observed_iters(prov, skipped.samples), probe=skipped.probe,
        census=skipped.census, copies_declared=skipped.copies_declared)


def _one_skip_line(log: str) -> str:
    lines = [ln for ln in log.splitlines() if ln.startswith("SWEEP SKIPPED")]
    assert len(lines) == 1, log[-2000:]
    return lines[0]


@pytest.mark.parametrize("world_name,v8", V8_EARLY_EXITS)
def test_the_page_a_v8_early_exit_prints_scores_every_gate_and_is_invalid(
        world_name, v8, monkeypatch, tmp_path, capsys):
    """THE REPORT SHAPE NO WORLD EXERCISED. `--self-test` builds its samples
    from `planted_samples` and never enters `run_sweep`, so the only way to an
    empty `samples` list is the pod's own V8 early return, and nothing in this
    suite called `analyse([], ...)` before this. The defect it guards is a page
    that RAISES instead of printing: `tread_rows`, `ladder_for`,
    `ratio_interval` and `declaration_fit` all group by arm and tread and every
    group is empty here, and an exception there reaches `main` as ERROR (4),
    which `ledger_state` in the session driver maps to RETRY -- so a build that
    can never pass V8 would be re-rented instead of reported. Both early-exit
    worlds are driven, because the page must be the same one whatever the step.
    """
    skipped = _skip_the_sweep(monkeypatch, tmp_path, capsys, world_name)
    assert skipped.samples == []
    assert skipped.proof.parts == {}
    report = _page_after(skipped)
    assert [(g.tag, g.verdict) for g in report.gates] == [
        ("V0", exit_codes.FAIL),
        ("V1", exit_codes.FAIL),
        ("V2", exit_codes.UNKNOWN),
        ("V3", exit_codes.UNKNOWN),
        ("V4", exit_codes.UNKNOWN),
        ("V5", exit_codes.UNKNOWN),
        ("V6", exit_codes.UNKNOWN),
        ("V7", exit_codes.UNKNOWN),
        ("V8", v8),
        ("C1", exit_codes.UNKNOWN),
        ("C2", exit_codes.UNKNOWN),
    ], [(g.tag, g.verdict) for g in report.gates]
    assert exit_codes.classify(g.scored() for g in report.gates) \
        == exit_codes.INVALID
    # V0 counts the grid that was planned and never timed. 162 is THIS
    # command's geometry, stated rather than recomputed from `analyse`'s own
    # `len(treads) x len(ARMS) x repeats`: a test that re-evaluates the
    # expression it is checking follows it into a wrong answer. The line above
    # pins the three defaults the 162 is made of, so moving one fails here and
    # says which.
    assert (len(skipped.treads), len(PW.ARMS), skipped.args.repeats) == (6, 3, 9)
    assert report.gates[0].measured.startswith("0/162 cells, treads 0/0/0")
    # V8 reports the host-bound census on either path, so the page says which
    # machine the probe timed even when the answer is "all of them, the host".
    # The hot count is the WORLD's plant, named here rather than read back out
    # of `probe.host_bound`, which is the call `gate_v8_alignment` itself makes.
    probed = len(skipped.treads) * PW.PROBE_REPEATS
    hot = 0
    assert skipped.probe.host_bound(PW.SHARED)[:2] == (hot, probed)
    assert any(f"called {hot} of {probed} probed cells HOST-BOUND" in ln
               for ln in report.gates[8].lines), report.gates[8].lines
    # And every gate still prints the RESULT line the driver recomputes from.
    assert len(exit_codes.parse_result_lines("\n".join(report.lines))) \
        == len(report.gates)


@pytest.mark.parametrize("world_name,v8", V8_EARLY_EXITS)
def test_the_skipped_pages_report_json_holds_no_nan_token(
        world_name, v8, monkeypatch, tmp_path, capsys):
    """`_main` writes `report.json` with `json.dumps`, which emits a BARE
    `NaN` token for a float nan. `analyse` seeds `interval = (math.nan,
    math.nan)` and only replaces it once a ratio is formed, so on this page
    both ends stay nan; before the patch `payload["ratio_interval"]` was
    `list(interval)` and the file could not be read by any strict JSON parser,
    which is every consumer of these reports except Python's own loader. The
    whole payload is searched, not just that one key: any other field that
    reaches this page as a nan fails here too.
    """
    skipped = _skip_the_sweep(monkeypatch, tmp_path, capsys, world_name)
    report = _page_after(skipped)
    assert report.payload["ratio"] is None
    assert report.payload["ratio_interval"] == [None, None]
    blob = json.dumps(report.payload, indent=2)
    assert "NaN" not in blob and "Infinity" not in blob

    def refuse(token):
        raise AssertionError(f"report.json carries the non-JSON token {token!r}")

    on_disk = tmp_path / "report.json"
    on_disk.write_text(blob)
    assert json.loads(on_disk.read_text(), parse_constant=refuse) \
        == report.payload
    assert report.text().endswith("\n")


@pytest.mark.parametrize("world_name,v8", V8_EARLY_EXITS)
def test_the_skipped_report_json_names_the_v8_verdict_the_page_prints(
        world_name, v8, monkeypatch, tmp_path, capsys):
    """`run_sweep`'s early return stored `{"skipped": "V8 failed on the probe;
    ..."}` on EVERY path, and `analyse` writes `proof.detail` verbatim into
    report.json's `buffer_proof`, so an UNKNOWN page shipped an artefact that
    said FAIL. The stored reason now carries `early.verdict`."""
    skipped = _skip_the_sweep(monkeypatch, tmp_path, capsys, world_name)
    report = _page_after(skipped)
    page_v8 = next(g for g in report.gates if g.tag == "V8").verdict
    assert page_v8 == v8
    stored = report.payload["buffer_proof"]["detail"]["skipped"]
    assert stored.startswith(f"V8 came back {v8} on the probe"), stored
    other = exit_codes.FAIL if v8 == exit_codes.UNKNOWN else exit_codes.UNKNOWN
    assert other not in stored and "failed" not in stored.replace(v8, "")


def test_every_flag_the_script_prints_is_one_its_parser_accepts():
    """V8 once told the operator to "raise --probe-target-ms", a flag argparse
    never had (PROBE_TARGET_MS is a module constant). Every `--flag` in a
    string literal of the script must be an option `build_parser` accepts,
    save the session driver's own flags, which the script cites by name."""
    import ast
    driver_flags = {"--new"}
    accepted = set()
    for action in PW.build_parser()._actions:
        accepted |= set(action.option_strings)
    unknown = {}
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", node.value):
                if flag not in accepted and flag not in driver_flags:
                    unknown.setdefault(flag, node.lineno)
    assert not unknown, unknown


def test_the_memory_refusal_offers_no_remedy_the_census_refuses_first():
    """"take the padding off with --declared-copies" could never be taken:
    padding exists only when the ladder straddles the id bound, and a
    declaration under the padding puts the ratio arms on the crossing, which
    the census refuses before memory is priced. Verified by running."""
    got = run(["--dry-run", "--device-memory-gb", "20"])
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:]
    assert "do not fit this card" in got.stdout, got.stdout[-2000:]
    assert "--declared-copies" not in got.stdout.split("do not fit this card")[1]
    census = run(["--dry-run", "--device-memory-gb", "20",
                  "--declared-copies", "6"])
    assert census.returncode == exit_codes.REFUSED
    assert "do not fit this card" not in census.stdout
    assert "changes alignment kernel at tread" in census.stdout


# --------------------------------------------------------------------------
# 17. the five judgment calls, decided
# --------------------------------------------------------------------------

def _v5_ladders(native_per_tile_ms: float):
    """Native, shared and private ladders over six treads, noise-free, with
    `native_per_tile_ms` of declaration cost per M-tile in NATIVE alone."""
    samples = PW.planted_samples(
        PW.World("x", "x", {}, machinery_ms_per_tile=native_per_tile_ms), CFG,
        block_m=32, treads=[1, 2, 3, 4, 5, 6], repeats=3,
        alpha_shared=PW.ALPHA, ridge=160.0, bandwidth_gbps=4000.0, b=2,
        noise=0.0, seed=0, copies_declared=9)
    return tuple(PW.ladder_for(samples, a) for a in PW.ARMS)


def test_v5_fails_a_declaration_cost_a_tenth_admitted_and_reads_the_far_edge():
    """MACHINERY_BOUND was 0.10, which admits a ratio error larger than
    ALPHA_BAND's entire 0.059 width. At 0.03, a declaration cost of 5% of the
    private slope FAILs (it PASSED before); and a point at 1% whose band
    reaches 4% FAILs too, because the bound is scored on b's FAR edge."""
    assert PW.MACHINERY_BOUND == 0.03
    assert PW.MACHINERY_BOUND < ALPHA_BAND_WIDTH()
    native, shared, private = _v5_ladders(0.0)
    fit = PW.DeclarationFit(None, None, 0.05 * private.slope_ms, 0.0, (), 4)
    band = (0.045 * private.slope_ms, 0.055 * private.slope_ms)
    assert PW.gate_v5_machinery(native, shared, private, fit, band).verdict \
        == exit_codes.FAIL
    fit = PW.DeclarationFit(None, None, 0.01 * private.slope_ms, 0.0, (), 4)
    wide = (-0.04 * private.slope_ms, 0.02 * private.slope_ms)
    gate = PW.gate_v5_machinery(native, shared, private, fit, wide)
    assert gate.verdict == exit_codes.FAIL, gate.lines
    assert gate.measured.startswith("4.00% of the private slope at b's far edge")
    tight = (0.005 * private.slope_ms, 0.015 * private.slope_ms)
    assert PW.gate_v5_machinery(native, shared, private, fit, tight).verdict \
        == exit_codes.PASS


def ALPHA_BAND_WIDTH() -> float:
    return PW.ALPHA_BAND[1] - PW.ALPHA_BAND[0]


def test_c1_passes_only_an_interval_inside_the_band_and_fails_one_starting_at_its_edge():
    """C1 PASSed REFIT-CONFIRMED with an interval reaching into
    ABOVE-THE-REFIT-BAND; now the whole interval must sit inside ALPHA_BAND.
    And the FAIL test was closed at the upper edge while membership is
    half-open, so an interval starting AT 0.588 was UNKNOWN; it misses the
    band and is FAIL."""
    lo_edge, hi_edge = PW.ALPHA_BAND
    assert PW.c1_verdict(PW.ALPHA, (lo_edge + 0.005, hi_edge + 0.007)) \
        == exit_codes.UNKNOWN
    assert PW.c1_verdict(PW.ALPHA, (lo_edge - 0.004, hi_edge - 0.004)) \
        == exit_codes.UNKNOWN
    assert PW.c1_verdict(hi_edge + 0.002, (hi_edge, hi_edge + 0.01)) \
        == exit_codes.FAIL
    assert PW.outcome_for(hi_edge)[0] != PW.outcome_for(PW.ALPHA)[0]
    assert PW.c1_verdict(PW.ALPHA, (lo_edge, hi_edge - 1e-9)) == exit_codes.PASS


def test_the_probe_times_private_ids_as_a_third_series_and_prices_it():
    """PRIVATE's ids are probed at the shared declaration: ~2.5 s more on
    the pod, and the only measurement of the counter asymmetry the plan page
    registers."""
    assert PW.PROBE_LABELS == len(PW.ARMS) == 3
    treads = [1, 2, 3, 4, 5, 6]
    assert PW.probe_seconds(treads, 1) == pytest.approx(2.52)
    probe = PW.planted_probe(PW.WORLDS["refit"], CFG, block_m=32, treads=treads,
                             declared_by_arm={PW.NATIVE: 8, PW.SHARED: 72,
                                              PW.PRIVATE: 72},
                             census=_census(), noise=0.002, seed=0)
    assert probe.labels() == [PW.NATIVE, PW.SHARED, PW.PRIVATE]
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert any(ln.startswith("slope(private ids) - slope(shared ids) = ")
               and "us per tread" in ln for ln in gate.lines), gate.lines


def test_a_step_in_private_ids_alone_is_bounded_and_fails_v8():
    """step_bias assumed the step COMMON to both ratio arms and nothing
    checked it. A 0.5 ms step in PRIVATE's series alone moves the ratio's
    denominator: pre-change V8 read only SHARED and PASSED it."""
    treads = [1, 2, 3, 4, 5, 6]
    probe = _probe_from({PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4),
                         PW.PRIVATE: _series(0.5, 4)})
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.FAIL, gate.lines
    assert any("NOT the same step" in ln for ln in gate.lines), gate.lines
    lev = PW.leverage(treads, 4)
    want = (lev * 0.5) / (0.64 + lev * 0.5)
    assert PW.pair_step_bias(None, (0.5, 4), treads, 0.64) == pytest.approx(want)
    # A COMMON step is bounded no looser than step_bias.
    assert PW.pair_step_bias((0.02, 4), (0.02, 4), treads, 0.64) \
        <= PW.step_bias(0.02, treads, 4, 0.64)


@pytest.mark.parametrize("name,budgets,v8", [
    ("ratio-step-over-budget", 1.5, exit_codes.FAIL),
    ("ratio-step-under-budget", 0.7, exit_codes.PASS)])
def test_planted_worlds_sit_either_side_of_the_v8_budget(name, budgets, v8):
    """No world sat within 2x of V8's budget. These two plant a real common
    ratio step sized IN BUDGETS at the run's own weight stream."""
    world = PW.WORLDS[name]
    assert world.ratio_probe_budgets == budgets and world.expect["V8"] == v8
    got = run(["--self-test", name])
    line = [r for r in exit_codes.parse_result_lines(got.stdout)
            if r.name == "V8"]
    assert [r.verdict for r in line] == [v8], got.stdout[-2000:]
    bias = float(re.search(r"bias <= ([0-9.]+)", got.stdout).group(1))
    assert bias == pytest.approx(budgets * PW.ALIGN_STEP_RATIO_BUDGET, rel=0.05)


def test_a_v8_unknown_runs_the_sweep_and_only_a_fail_skips_it(
        monkeypatch, tmp_path, capsys):
    """THE REVERT, and why it is safe only after NATIVE became V8's positive
    control. A FAIL is a statement about the BUILD (it switches alignment
    kernel inside the ratio arms' ladder) and skipping the sweep saves the
    card's minutes. An UNKNOWN is now a statement about the INSTRUMENT: the
    probe could not see NATIVE's own switch at the census tread. The page
    still latches INVALID on V8, but the ladder and every other gate's number
    are worth having on a rented card, so `run_sweep` goes on to allocate.
    Pre-revert the host-bound-probe world stopped here with SWEEP SKIPPED.
    """
    ran = _skip_the_sweep(monkeypatch, tmp_path, capsys, "host-bound-probe",
                          reach=True)
    assert "SWEEP SKIPPED" not in ran.log
    assert "V8 came back UNKNOWN on the probe: the sweep RUNS" in ran.log
    # And a FAIL still stops before a weight is allocated.
    skipped = _skip_the_sweep(monkeypatch, tmp_path, capsys, "ratio-path-split")
    assert skipped.samples == [] and not skipped.csv_path.exists()
    assert _one_skip_line(skipped.log).startswith("SWEEP SKIPPED: V8 came back FAIL")


def test_the_sweep_skipped_line_names_the_build_and_an_unknown_prints_none(
        monkeypatch, tmp_path, capsys):
    """A FAIL says the build switches alignment kernel inside the ratio arms'
    ladder, which is not fixed by re-running, and it is the only verdict that
    prints SWEEP SKIPPED. Every FAIL world prints the same sentence."""
    for world_name, verdict in V8_EARLY_EXITS:
        assert verdict == exit_codes.FAIL
        line = _one_skip_line(
            _skip_the_sweep(monkeypatch, tmp_path, capsys, world_name).log)
        assert "switches alignment kernel inside the ratio arms' ladder" in line
        assert "UNKNOWN" not in line
        assert line.endswith("Nothing was allocated and nothing was timed.")


def test_the_skipped_page_prints_five_not_run_parts_and_never_its_own_reason(
        monkeypatch, tmp_path, capsys):
    """OPEN FINDING, ASSERTED AS IT IS AND NOT REPAIRED HERE. `run_sweep`'s
    early return builds `BufferProof(parts={}, detail={'skipped': ...})`, but
    `BufferProof.lines` walks `PROOF_PARTS` and prints one line per part, so
    the one key `detail` holds is the one key that is never rendered: V2 says
    NOT RUN five times and never once says why. The stored reason carries
    V8's actual verdict (it used to read 'V8 failed on the probe' on the
    UNKNOWN path too, and report.json contradicted the page).
    """
    skipped = _skip_the_sweep(monkeypatch, tmp_path, capsys, "ratio-path-split")
    assert len(PW.PROOF_PARTS) == 5          # the five this test is named for
    lines = skipped.proof.lines()
    assert len(lines) == len(PW.PROOF_PARTS)
    assert all(ln.startswith("NOT RUN") for ln in lines)
    assert set(skipped.proof.detail) == {"skipped"}
    assert not any("skipped" in ln for ln in lines)
    assert skipped.proof.detail["skipped"].startswith(
        "V8 came back FAIL on the probe")
    report = _page_after(skipped)
    v2 = next(g for g in report.gates if g.tag == "V2")
    assert v2.verdict == exit_codes.UNKNOWN
    assert v2.measured == "0 of 5 parts"
    assert not any("skipped" in ln for ln in v2.lines)


def test_the_skipped_pages_iteration_line_blames_a_world_the_pod_never_ran(
        monkeypatch, tmp_path, capsys):
    """OPEN FINDING, ASSERTED AS IT IS AND NOT REPAIRED HERE. `_iters_line`
    has one empty-case string and it explains the emptiness as a planted
    world's `iters=0` rows, which was the only way to reach it before the V8
    early return existed. The early return produces the same empty list on a
    pod with no planted world anywhere, and `_main` prints that line under a
    report whose `synthetic` field is False, because `_main` sets that field
    from `--self-test` alone.

    THE CONFLATION IS SHOWN, not merely described: the 162 cells of a planted
    world, every one of which ran and carries `iters=0`, and the early
    return's zero cells, none of which ran at all, print the SAME sentence,
    and it names only the first.
    """
    skipped = _skip_the_sweep(monkeypatch, tmp_path, capsys, "ratio-path-split")
    empty = PW._iters_line(skipped.samples)
    assert empty == ("iterations per trial: none recorded (nothing was timed; "
                     "a planted world's cells carry iters=0)")
    world = PW.WORLDS["refit"]
    planted = PW.planted_samples(
        world, CFG, block_m=skipped.block_m, treads=skipped.treads,
        repeats=skipped.args.repeats, alpha_shared=world.alpha, ridge=160.0,
        bandwidth_gbps=4000.0, b=PW.dtype_bytes(skipped.args.dtype), noise=0.0,
        seed=skipped.args.seed, copies_declared=skipped.copies_declared,
        native_switch=skipped.census.switch_tread(PW.NATIVE))
    assert len(planted) == 162 and {s.iters for s in planted} == {0}
    assert PW._iters_line(planted) == empty
