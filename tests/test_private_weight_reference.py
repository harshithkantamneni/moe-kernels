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
from _hermetic import laptop_env  # noqa: E402

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
    (("--self-test", "private-step-alone"), True),
    (("--self-test", "clock-split-elastic"), True),
    (("--self-test", "graph-probe-unresolved"), True),
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
    # Laptop path on every box: a --self-test or --dry-run child must not
    # find a card, and a bare argv must not measure from inside pytest.
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=900,
                          cwd=str(ROOT), env=laptop_env())


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
                        ("--group-m", 16), ("--seed", 3), ("--duty", 0.5),
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
                        ("--alpha", 0.3), ("--replicate-of", "x/report.json")):
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


def _probe_from(series_by_label, spread=0.0, host_bound=False, graph_calls=0,
                note=""):
    """`graph_calls` 0 plants an EAGER probe (the pre-2026-09-22 instrument
    and the capture-refused fallback); N plants the graph-timed one."""
    cells = []
    for label, series in series_by_label.items():
        for rep in range(3):
            for n, numel, ms in series:
                jitter = (rep - 1) * spread
                cells.append(PW.ProbeCell(label, n, numel, 72, rep,
                                          ms * (1.0 + jitter),
                                          host_bound=host_bound,
                                          graph_calls=graph_calls,
                                          replay_ms=(ms * graph_calls
                                                     if graph_calls else None)))
    return PW.AlignProbe(tuple(cells), synthetic=True, note=note)


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
    # The fit alone is UNKNOWN: a PASS is scored on b's FAR EDGE, and with no
    # band the point is all there is. The band from the same noise-free
    # samples PASSES.
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
    # And the false alarms that DO fire must not cluster on the census' own
    # split, which would make them look like a feature of the ladder rather
    # than the search talking. An absolute bound, because `found_rate` is
    # counted inside the `resolved()` branch and `found_rate <= new_rate`
    # therefore cannot fail.
    assert found_rate <= 0.02, found_rate
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
# 13. an EAGER host-bound probe times the HOST, so V8 says UNKNOWN
# --------------------------------------------------------------------------

def _verdict_probe(verdicts_by_label, graph_calls=0):
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
                                          host_bound=hot, host_note=note,
                                          graph_calls=graph_calls))
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
    # THIS FIXTURE plants no PRIVATE cells, so PRIVATE has no verdict here and
    # the note is empty rather than SHARED's. The PRODUCTION probe does emit
    # them -- PROBE_LABELS is len(ARMS) since step 5 -- which is exactly why
    # `host_bound` has to be asked per label rather than over every cell.
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
    assert any(ln.startswith("POSITIVE CONTROL NOT CONFIRMED: the probe's "
                             "ratio cells were host-bound (6 of 15), timed "
                             "eagerly; NATIVE's "
                             "switch is due at tread 4 and the probe resolved "
                             "no step") for ln in gate.lines), gate.lines
    # And a cleared probe says so on the same line rather than staying silent,
    # so "0 of 18" is a positive record that the instrument did look.
    cool = PW.gate_v8_alignment(
        _probe_from({PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4)}),
        treads=treads, census=_census(), weight_stream_ms=0.64)
    assert ("the instrument called 0 of 18 probed cells HOST-BOUND at this "
            "declaration") in cool.lines, cool.lines


# --------------------------------------------------------------------------
# 13b. NATIVE as the EAGER host-bound probe's POSITIVE CONTROL
# --------------------------------------------------------------------------

def test_v8_passes_a_host_bound_flat_series_when_natives_switch_resolves_at_the_census_tread():
    """THE EAGER-FALLBACK CASE (the rented-card case until 2026-09-22, when
    the probe moved under a CUDA graph). On an H200 an EAGER probe is
    host-bound (session 4: 32-36 us of host against a few us of GPU per
    call), and V8 used to return UNKNOWN for every host-bound probe -- which
    skipped the sweep, so a rented pod produced no ladder at all. NATIVE
    declares E=8, under the expert
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
# 13d. the two arms V8 scores: which one may refuse the design, and which
#      cells the instrument judged
# --------------------------------------------------------------------------

def _per_label_probe(series_by_label, hot_by_label=None, graph_calls=0):
    """An `AlignProbe` whose per-label host-bound verdict is set per LABEL.

    `_probe_from` stamps one verdict on every cell of every label, so it
    cannot express the case the gate has to survive: the two ratio id sets
    are timed at one declaration with one host enqueue cost and two GPU
    times, so the instrument's `g > h` test can land differently on them.
    """
    hot_by_label = hot_by_label or {}
    cells = []
    for label, series in series_by_label.items():
        for rep in range(3):
            for n, numel, ms in series:
                cells.append(PW.ProbeCell(label, n, numel, 72, rep, ms,
                                          host_bound=hot_by_label.get(label),
                                          graph_calls=graph_calls))
    return PW.AlignProbe(tuple(cells), synthetic=True)


def test_v8_does_not_fail_on_a_step_its_own_rule_called_noise():
    """THE REGRESSION step 5 introduced. `bias` is taken over BOTH ratio
    series but `real` was `r.real or rp.real`, so an UNRESOLVED noise step in
    one arm could carry the bias over budget while the OTHER arm's resolved
    step supplied the licence to FAIL. The gate's own registered rule -- its
    docstring, its printed criterion and the plan page -- is "FAIL needs it
    over budget AND resolved", and on a pod a FAIL skips the whole sweep
    before a byte is allocated, which is the trade the step-5 revert exists
    to refuse.

    Planted here: SHARED carries a resolved step worth well under the budget,
    PRIVATE carries a larger step its own standard error does not resolve.
    The pair bound clears the budget; the bound over the RESOLVED steps alone
    does not. Pre-fix: FAIL. Now: UNKNOWN, and the page says which number it
    was taken on.
    """
    treads = [1, 2, 3, 4, 5, 6]
    shared = _series(0.02, 4)
    private = _series(0.18, 4, noise=0.25, seed=5)
    probe = _per_label_probe({PW.NATIVE: _series(0.02, 4), PW.SHARED: shared,
                              PW.PRIVATE: private}, {PW.NATIVE: False,
                                                     PW.SHARED: False,
                                                     PW.PRIVATE: False})
    sh = PW.step_fit(probe.series(PW.SHARED))
    pv = PW.step_fit(probe.series(PW.PRIVATE))
    assert sh.resolved() and not pv.resolved(), (sh, pv)
    wide = PW.pair_step_bias((sh.step_ms, sh.split_tread),
                             (pv.step_ms, pv.split_tread), treads, 0.64)
    narrow = PW.pair_step_bias((sh.step_ms, sh.split_tread), None, treads, 0.64)
    assert wide > PW.ALIGN_STEP_RATIO_BUDGET >= narrow, (wide, narrow)
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.UNKNOWN, gate.lines
    assert any("only through a step the rule did NOT resolve" in ln
               for ln in gate.lines), gate.lines
    # And a RESOLVED over-budget step still refuses the design.
    still = _per_label_probe({PW.NATIVE: _series(0.02, 4),
                              PW.SHARED: _series(0.5, 4),
                              PW.PRIVATE: _series(0.0, 4)})
    assert PW.gate_v8_alignment(still, treads=treads, census=_census(),
                                weight_stream_ms=0.64).verdict \
        == exit_codes.FAIL


def test_v8_reads_the_host_bound_verdict_of_every_series_it_scores():
    """PRIVATE's id set has been half the scored bias since step 5, but the
    host-bound census read SHARED's cells alone -- so a PRIVATE series the
    instrument had timed on the HOST was scored with the positive control
    never consulted, and the page printed a count that was true of SHARED and
    false of the declaration.

    The two are not host-bound together. `host_bound_verdict` thresholds the
    GPU's backlog, which is the sign test `g > h`; the host cost `h` is
    identical for the two id sets (one op, one declaration, one call shape,
    ids built outside the timed region) while `g` differs by exactly the
    counter asymmetry this gate prints -- PRIVATE spreads the same increments
    over n times as many counters, so it has LESS contention and crosses
    first. Planted here: PRIVATE hot, SHARED cool.
    """
    treads = [1, 2, 3, 4, 5, 6]
    flat = {PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.0, 4),
            PW.PRIVATE: _series(0.0, 4)}
    hot = {PW.NATIVE: False, PW.SHARED: False, PW.PRIVATE: True}
    gate = PW.gate_v8_alignment(_per_label_probe(flat, hot), treads=treads,
                                census=_census(), weight_stream_ms=0.64)
    # Pre-fix this PASSED: SHARED was cool, so the control was never asked.
    assert gate.verdict == exit_codes.UNKNOWN, gate.lines
    assert any(ln.startswith("POSITIVE CONTROL NOT CONFIRMED")
               for ln in gate.lines), gate.lines
    assert any("0 of 18 of shared's, 18 of 18 of private's probed cells "
               "HOST-BOUND at the ratio arms' declaration" in ln
               for ln in gate.lines), gate.lines
    # With NATIVE's switch resolved where the census puts it, the same
    # asymmetric probe earns PASS through the control.
    controlled = dict(flat, **{PW.NATIVE: _series(0.02, 4)})
    ok = PW.gate_v8_alignment(_per_label_probe(controlled, hot), treads=treads,
                              census=_census(), weight_stream_ms=0.64)
    assert ok.verdict == exit_codes.PASS, ok.lines
    assert any(ln.startswith("POSITIVE CONTROL CONFIRMED") for ln in ok.lines)


def test_a_planted_world_steps_one_ratio_arm_and_not_the_other():
    """WHAT NO WORLD COULD SHOW. `planted_probe` gave both ratio arms the
    same step, so `pair_step_bias` -- added because SHARED's and PRIVATE's
    steps need NOT be one step -- was numerically the common-step bound it
    replaced in every planted world, and no --self-test page exercised the
    asymmetry. `private-step-alone` plants the same 1.5 budgets
    `ratio-step-over-budget` plants in both, in PRIVATE's series alone.
    """
    treads = [1, 2, 3, 4, 5, 6]
    census = _census()
    by_arm = {PW.NATIVE: 8, PW.SHARED: 72, PW.PRIVATE: 72}
    stream = WEIGHTS.weight_stream_ms(CFG, "bf16", 4000.0)
    probe = PW.planted_probe(PW.WORLDS["private-step-alone"], CFG, block_m=32,
                             treads=treads, declared_by_arm=by_arm,
                             census=census, noise=0.0, seed=0,
                             weight_stream_ms=stream)
    shared = PW.step_fit(probe.series(PW.SHARED))
    private = PW.step_fit(probe.series(PW.PRIVATE))
    # SHARED is a straight line; PRIVATE carries the step, at the census tread.
    assert abs(shared.step_ms) < 1e-12, shared
    assert private.split_tread == 4 and private.step_ms > 0
    # And the step is worth the budgets the world registers, through the
    # bound that needed no common-step premise.
    bias = PW.pair_step_bias(None, (private.step_ms, 4), treads, stream)
    assert bias == pytest.approx(1.5 * PW.ALIGN_STEP_RATIO_BUDGET, rel=1e-9)
    gate = PW.gate_v8_alignment(probe, treads=treads, census=census,
                                weight_stream_ms=stream)
    assert gate.verdict == exit_codes.FAIL, gate.lines
    assert any("NOT the same step" in ln for ln in gate.lines), gate.lines
    # THE PUNCHLINE: the common-step bound this replaced reads SHARED's series
    # alone, which is flat, so it scores this same probe at ~0 and PASSES it.
    # This world is the one place in the table where the two bounds disagree.
    old_rule = PW.step_bias(shared.step_ms, treads, shared.split_tread or 4,
                            stream)
    assert old_rule < PW.ALIGN_STEP_RATIO_BUDGET / 100 < bias
    # ITS TWIN plants the same size in BOTH arms and reads the same bias, so
    # the pair differ in one variable: which series carries the step.
    both = PW.planted_probe(PW.WORLDS["ratio-step-over-budget"], CFG,
                            block_m=32, treads=treads, declared_by_arm=by_arm,
                            census=census, noise=0.0, seed=0,
                            weight_stream_ms=stream)
    assert PW.step_fit(both.series(PW.SHARED)).step_ms \
        == pytest.approx(PW.step_fit(both.series(PW.PRIVATE)).step_ms)
    assert PW.gate_v8_alignment(both, treads=treads, census=census,
                                weight_stream_ms=stream).measured \
        == gate.measured


def _page_from(samples, probe, treads, census, copies_declared=9):
    """`analyse` over planted samples and a chosen probe, with every basis
    fixed, so a test can ask what the page says about the tread V5 was
    fitted at."""
    block_m = 32
    b = PW.dtype_bytes("bf16")
    mem = PW.memory_plan(CFG, "bf16", b, copies_declared,
                         PW.SWEEP.tokens_for_rows(CFG, treads[-1] * block_m),
                         int(140e9), "--device-memory-gb",
                         copies_read=treads[-1])
    return PW.analyse(
        samples, CFG, block_m=block_m, treads=treads, repeats=3,
        alpha=PW.ALPHA, dtype="bf16", b=b, bandwidth_gbps=4000.0,
        bandwidth_source="fixed for this test", ridge=160.0,
        ridge_source="fixed for this test", roof_tflops=640.0,
        roof_source="fixed for this test", reference_mhz=None,
        reference_grade="", reference_source="fixed for this test",
        mem=mem, proof=PW.planted_proof(True), weight_delta_bytes=None,
        high_water_bytes=None, draws=200, seed=0, header=["PLAN"],
        card="NVIDIA H200", synthetic=True, model_name=PW.DEFAULT_MODEL,
        pinned={}, prov=None, probe=probe, census=census,
        copies_declared=copies_declared, duty=1.0)


def test_a_host_timed_native_step_does_not_choose_the_tread_v5_is_fitted_at():
    """b IS WHAT V5 SCORES, and the tread the step term sits at moves it.
    `analyse` took the probe's split whenever NATIVE's step was resolved,
    with no regard for what the probe had TIMED -- so on the host-bound
    probe an H200 is expected to give, a step in the host's enqueue cost
    could choose the tread `declaration_fit` puts its step term at, and the
    page said nothing about where the split came from.

    `native_step_is_admissible` is the rule, read by V8's control and by
    this override alike: the probe's split stands in for the census
    hypothesis only when NATIVE's own cells were JUDGED and none came back
    host-bound. Planted here: NATIVE's probe step at tread 3, where the
    census says 4, on cells the instrument called host-bound.
    """
    treads = [1, 2, 3, 4, 5, 6]
    census = _census()
    world = PW.WORLDS["refit"]
    samples = PW.planted_samples(world, CFG, block_m=32, treads=treads,
                                 repeats=3, alpha_shared=world.alpha,
                                 ridge=160.0, bandwidth_gbps=4000.0, b=2,
                                 noise=0.0, seed=0, copies_declared=9,
                                 native_switch=4)
    hot = {PW.NATIVE: True, PW.SHARED: False, PW.PRIVATE: False}
    probe = _per_label_probe({PW.NATIVE: _series(0.02, 3),
                              PW.SHARED: _series(0.0, 4),
                              PW.PRIVATE: _series(0.0, 4)}, hot)
    assert PW.read_probe(probe, PW.NATIVE, census).fit.split_tread == 3
    report = _page_from(samples, probe, treads, census)
    v5 = next(g for g in report.gates if g.tag == "V5")
    # The census tread stands, and the page names the refusal and its reason.
    assert any("the cited hypothesis (tread 4)" in ln for ln in v5.lines), v5.lines
    assert any("REFUSED as the fit's tread because" in ln
               and "HOST-BOUND" in ln for ln in v5.lines), v5.lines
    assert not any("the probe, which resolved native's step at tread 3" in ln
                   for ln in v5.lines), v5.lines
    # And a GPU-timed probe that disagrees IS taken, with its provenance.
    cool = _per_label_probe({PW.NATIVE: _series(0.02, 3),
                             PW.SHARED: _series(0.0, 4),
                             PW.PRIVATE: _series(0.0, 4)},
                            {PW.NATIVE: False, PW.SHARED: False,
                             PW.PRIVATE: False})
    taken = next(g for g in _page_from(samples, cool, treads, census).gates
                 if g.tag == "V5")
    assert any("the probe, which resolved native's step at tread 3" in ln
               and "read in GPU time" in ln for ln in taken.lines), taken.lines


def test_the_plan_page_says_whether_this_ladder_supplies_v8s_positive_control():
    """WHAT V8 CAN SAY IS A PROPERTY OF THE LADDER, and the operator should
    read it in the dry run rather than discover it after the card is paid
    for. NATIVE's own kernel switch is the positive control; a ladder whose
    id counts stay on one side of the 1024-id bound has none, so on the
    host-bound probe this op's size makes likely V8 can only read UNKNOWN --
    which latches the page INVALID after the whole ladder has run, because
    the sweep is skipped on a FAIL alone.

    At the booked tile (32) mixtral's ids run 256..1536 and cross; at 16 they
    run 128..768 and do not. NOT A REFUSAL: a probe the instrument judges and
    clears still PASSES at such a tile, and refusing would forbid a
    configuration that works -- qwen2-57b-a14b, for one, is above the bound
    at every tile the parser accepts.
    """
    ladder = run(["--dry-run", "--device-memory-gb", "140", "--block-m", "16"])
    assert "AND THIS LADDER GIVES V8 NO POSITIVE CONTROL" in ladder.stdout
    assert "128..768" in ladder.stdout and "1024-id bound" in ladder.stdout
    assert "NATIVE switches kernel nowhere in it" in ladder.stdout
    booked = run(["--dry-run", "--device-memory-gb", "140", "--block-m", "32"])
    assert "AND THIS LADDER GIVES V8 NO POSITIVE CONTROL" not in booked.stdout
    assert "native switches at tread 4" in booked.stdout
    # The two tiles' arithmetic, from the census rather than from this file.
    treads = [1, 2, 3, 4, 5, 6]
    assert PW.ids_for_tread(CFG, 6, 16) < PW.ALIGN_SMALL_BATCH_MAX_IDS
    assert PW.ids_for_tread(CFG, 4, 32) >= PW.ALIGN_SMALL_BATCH_MAX_IDS
    for bm, want in ((16, None), (32, 4)):
        decl, _why = PW.declared_copies_for(CFG, treads, bm, 0)
        census = PW.path_census(CFG, treads, bm, {
            a: PW.declared_experts(a, CFG.num_experts, decl) for a in PW.ARMS})
        assert census.switch_tread(PW.NATIVE) == want, bm


def test_the_probe_repeat_count_is_out_of_the_run_id_and_the_page_says_why():
    """`--probe-repeats` changes the probe's cells and can change V8's
    verdict, which makes it look like a key. It is deliberately OUT: the
    probe is re-timed every invocation and never resumed -- nothing in
    cells.csv comes from it -- so two runs differing only in it hold the same
    measured ladder, and keying on it would split their directories and stop
    the second resuming the first's card minutes.

    The classification was unwritten, which is how a knob drifts into a key.
    It is now in `default_run_id`'s OUT list and in the flag's own help, and
    the value stays recoverable from report.json's `align_probe`.
    """
    a = PW.build_parser().parse_args(["--probe-repeats", "3"])
    b = PW.build_parser().parse_args(["--probe-repeats", "9"])
    assert PW.default_run_id(a, "NVIDIA H200") == PW.default_run_id(b, "NVIDIA H200")
    doc = " ".join(PW.default_run_id.__doc__.split())
    assert "--probe-repeats" in doc and "OUT ON PURPOSE" in doc
    assert "never resumed" in doc and "align_probe" in doc
    help_text = " ".join(PW.build_parser().format_help().split())
    assert "NOT in the run id" in help_text
    # And the artefact really carries what the key leaves out.
    probe = _per_label_probe({PW.NATIVE: _series(0.02, 4),
                              PW.SHARED: _series(0.0, 4),
                              PW.PRIVATE: _series(0.0, 4)})
    assert {c["repeat"] for c in probe.as_dict()["cells"]} == {0, 1, 2}


def test_a_refused_bootstrap_on_b_is_not_reported_as_a_refused_fit():
    """`declaration_fit` and `declaration_interval` shared one try, so a
    failed BOOTSTRAP printed the FIT's sentence -- "declaration difference
    NOT FITTED" two lines above the page's own "DECLARATION, native - shared
    per tread", which IS the fit. And since V5 reads UNKNOWN when b has no
    band, that failure decides a verdict, so which of the two refused is the
    difference between a page a reader can follow and one that contradicts
    itself. The ratio's own interval has said it this way since it was
    written ("interval NOT FORMED: ...").

    `--draws 1` reaches it: one draw cannot form a percentile band.
    """
    got = run(["--self-test", "refit", "--draws", "1"])
    assert "b's band NOT FORMED over 1 draws" in got.stdout, got.stdout[-3000:]
    assert "declaration difference NOT FITTED" not in got.stdout
    # The fit itself is on the page, two lines under the refusal that used to
    # claim it had failed.
    assert "DECLARATION, native - shared per tread" in got.stdout
    v5 = [ln for ln in got.stdout.splitlines() if "NO BAND on b" in ln]
    assert len(v5) == 1, got.stdout[-3000:]
    assert "b's band was NOT FORMED" in v5[0], v5[0]
    assert next(r for r in exit_codes.parse_result_lines(got.stdout)
                if r.name == "V5").verdict == exit_codes.UNKNOWN
    # And with the draws the default gives, the band forms and V5 scores it.
    assert next(r for r in exit_codes.parse_result_lines(
        run(["--self-test", "refit"]).stdout)
        if r.name == "V5").verdict == exit_codes.PASS


def test_the_counter_asymmetry_line_carries_the_slope_difference_it_measured():
    """The one MEASUREMENT step 5 added had its value asserted nowhere: the
    tests checked that the line is printed, not that the number on it is the
    difference it claims. Planted here: PRIVATE's series costs a known extra
    per-id slope, so the printed microseconds per tread are arithmetic.
    """
    treads = [1, 2, 3, 4, 5, 6]
    extra_per_id = 2e-6                      # ms per id, PRIVATE's ids only
    shared = _series(0.0, 4)
    private = [(n, numel, ms + extra_per_id * numel)
               for n, numel, ms in shared]
    probe = _per_label_probe({PW.NATIVE: _series(0.02, 4), PW.SHARED: shared,
                              PW.PRIVATE: private})
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    ids_per_tread = 256                      # this ladder's own id step
    want_us = extra_per_id * ids_per_tread * 1e3
    assert want_us == pytest.approx(0.512)
    line = next(ln for ln in gate.lines
                if ln.startswith("slope(private ids) - slope(shared ids)"))
    got = float(re.search(r"= \+?(-?[0-9.]+) us per tread", line).group(1))
    assert got == pytest.approx(want_us, rel=1e-6), line
    # And in the ratio's own units against the weight stream.
    ratio_units = float(re.search(r"([+-][0-9.]+) of the weight stream", line)
                        .group(1))
    assert ratio_units == pytest.approx(extra_per_id * ids_per_tread / 0.64,
                                        rel=1e-6), line


def test_the_control_says_which_case_it_is_in_and_whether_native_was_host_bound():
    """`native_control` opened "POSITIVE CONTROL CONFIRMED: the probe was
    host-bound" on BOTH branches that reach it -- including the one where the
    instrument returned no host-bound verdict at all, one line under a page
    that says exactly that. And it claimed the switch was seen "through the
    host cost present" without ever reading NATIVE's own verdicts, which are
    what would show that.
    """
    treads = [1, 2, 3, 4, 5, 6]
    stepped = {PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4),
               PW.PRIVATE: _series(0.0, 4)}
    blind = PW.gate_v8_alignment(_per_label_probe(stepped), treads=treads,
                                 census=_census(), weight_stream_ms=0.64)
    assert blind.verdict == exit_codes.PASS, blind.lines
    assert any("returned no host-bound verdict for any probed cell" in ln
               for ln in blind.lines), blind.lines
    assert not any("the probe was host-bound" in ln for ln in blind.lines)
    assert any("no host-bound verdict for the ratio cells" in ln
               for ln in blind.lines), blind.lines
    # NATIVE's own verdicts are read and reported: a switch seen in GPU time
    # is not the same demonstration as one seen through host cost.
    gpu_native = PW.gate_v8_alignment(
        _per_label_probe(stepped, {PW.NATIVE: False, PW.SHARED: True,
                                   PW.PRIVATE: True}),
        treads=treads, census=_census(), weight_stream_ms=0.64)
    assert gpu_native.verdict == exit_codes.PASS, gpu_native.lines
    assert any("were NOT called host-bound, so the switch below was seen in "
               "GPU time" in ln for ln in gpu_native.lines), gpu_native.lines
    hot_native = PW.gate_v8_alignment(
        _per_label_probe(stepped, {PW.NATIVE: True, PW.SHARED: True,
                                   PW.PRIVATE: True}),
        treads=treads, census=_census(), weight_stream_ms=0.64)
    assert any("NATIVE's own cells were host-bound in 18 of 18" in ln
               for ln in hot_native.lines), hot_native.lines


# --------------------------------------------------------------------------
# 18. the clock-corrected ratio: printed beside the raw one, scored by nothing
# --------------------------------------------------------------------------

def test_clock_corrected_carries_each_cell_by_its_own_clock_to_the_power_eta():
    """`ms x (f_cell / f_ref) ** eta`, per RATIO-ARM cell; NATIVE untouched."""
    cells = [_sample(PW.PRIVATE, 2, 0, 2.0, load=1444.0),
             _sample(PW.SHARED, 2, 0, 1.0, load=1600.0),
             _sample(PW.NATIVE, 2, 0, 3.0, load=1444.0)]
    out = PW.clock_corrected(cells, 0.5, 1600.0)
    by = {s.arm: s for s in out}
    assert by[PW.PRIVATE].ms_p50 == pytest.approx(2.0 * (1444.0 / 1600.0) ** 0.5)
    assert by[PW.SHARED].ms_p50 == pytest.approx(1.0)
    assert by[PW.NATIVE].ms_p50 == 3.0
    # The originals are not mutated: the raw ratio is still formed from them.
    assert cells[0].ms_p50 == 2.0


def test_the_clock_corrected_ratio_does_not_depend_on_the_reference_clock():
    """Both arms carry `f_ref ** -eta`, which cancels in the ratio; f_ref only
    names the clock the corrected CELLS sit at."""
    samples = _pair_world(private_clock=1425.0, shared_clock=1455.0)
    eta = PW.ClockElasticity(0.7436, 0.7277, 0.7559, "x")
    a = PW.clock_corrected_ratio(samples, eta, f_ref=1455.0, f_ref_source="a",
                                 draws=50, seed=0)
    b = PW.clock_corrected_ratio(samples, eta, f_ref=1000.0, f_ref_source="b",
                                 draws=50, seed=0)
    assert a.ratio == pytest.approx(b.ratio, rel=1e-12)
    assert a.at_unit == pytest.approx(b.at_unit, rel=1e-12)


def test_a_planted_clock_split_with_a_known_eta_corrects_back_to_the_planted_alpha():
    """THE EXACT IDENTITY the new world registers. The world plants a 2%
    private clock deficit AND the law `ms x (1 + skew) ** -eta`; the
    correction inverts that law cell by cell, so corrected / raw is
    (1 + skew) ** -eta to rounding and the corrected ratio IS the refit
    world's raw ratio at the same seed and noise. Pre-change neither the
    world nor `clock_corrected_ratio` exists.
    """
    treads = [1, 2, 3, 4, 5, 6]
    kw = dict(block_m=32, treads=treads, repeats=3, ridge=160.0,
              bandwidth_gbps=4000.0, b=2, noise=0.0, seed=0, copies_declared=9,
              native_switch=4)
    world = PW.WORLDS["clock-split-elastic"]
    assert world.planted_eta == 0.75 and world.private_clock_skew == -0.02
    elastic = PW.planted_samples(world, CFG, alpha_shared=world.alpha, **kw)
    refit = PW.WORLDS["refit"]
    plain = PW.planted_samples(refit, CFG, alpha_shared=refit.alpha, **kw)
    raw = (PW.ladder_for(elastic, PW.SHARED).slope_ms
           / PW.ladder_for(elastic, PW.PRIVATE).slope_ms)
    refit_raw = (PW.ladder_for(plain, PW.SHARED).slope_ms
                 / PW.ladder_for(plain, PW.PRIVATE).slope_ms)
    cc = PW.clock_corrected_ratio(
        elastic, PW.ClockElasticity(0.75, 0.75, 0.75, "PLANTED"),
        f_ref=1500.0, f_ref_source="planted", draws=50, seed=0)
    assert cc.ratio / raw == pytest.approx((1 - 0.02) ** -0.75, abs=1e-12)
    assert cc.ratio == pytest.approx(refit_raw, abs=1e-9)
    # V7 still fails: the correction is printed, the gate is unmoved.
    assert PW.gate_v7_clock_parity(elastic, treads=treads).verdict == exit_codes.FAIL


def test_the_corrected_envelope_covers_the_interval_and_collapses_without_one():
    samples = _pair_world(private_clock=1425.0, shared_clock=1455.0)
    wide = PW.clock_corrected_ratio(samples, PW.ClockElasticity(0.74, 0.70, 0.78, "x"),
                                    f_ref=1455.0, f_ref_source="a", draws=50, seed=0)
    assert wide.interval is not None and wide.envelope is not None
    assert wide.envelope[0] <= wide.interval[0] and wide.interval[1] <= wide.envelope[1]
    fixed = PW.clock_corrected_ratio(samples, PW.ClockElasticity(0.74, 0.74, 0.74, "x"),
                                     f_ref=1455.0, f_ref_source="a", draws=50, seed=0)
    assert fixed.envelope == pytest.approx(fixed.interval)


def test_a_corrected_ratio_is_not_formed_when_a_ratio_arm_cell_has_no_clock():
    samples = _pair_world()
    victim = next(s for s in samples if s.arm == PW.PRIVATE)
    samples[samples.index(victim)] = PW.replace(victim, sm_clock_load_mhz=None)
    with pytest.raises(PW.Unmeasurable, match="1 usable ratio-arm cell"):
        PW.clock_corrected(samples, 0.7, 1455.0)


def test_c1_prints_the_clock_corrected_ratio_and_scores_the_raw_one():
    eta = PW.ClockElasticity(0.7436, 0.7277, 0.7559, "R1 report.json @81f80b7")
    clock = PW.ClockCorrection(eta, 1455.0, "the calibration's reference clock",
                               0.9743, (0.9700, 0.9790), (0.9679, 0.9829),
                               0.9809, 0.9551)
    gate = PW.gate_c1_ratio(0.9551, (0.9544, 0.9695), 2000, corrected=None,
                            clock=clock)
    assert gate.verdict == exit_codes.FAIL          # NO-REUSE, scored RAW
    assert gate.measured.startswith("0.9551 [0.9544, 0.9695]")
    joined = "\n".join(gate.lines)
    assert ("clock-corrected ratio 0.9743 at eta = 0.7436 [0.7277, 0.7559] "
            "(R1 report.json @81f80b7)") in joined
    assert "PRINTED ONLY" in joined and "moved +0.0192" in joined
    assert ("at eta = 1, the first-order figure DD11 argues from, it would "
            "read 0.9809; not a bound") in joined
    assert "ONE estimator's bootstrap carried across eta" in joined


def test_the_elasticity_flag_needs_a_source_and_one_or_three_numbers():
    got = run(["--self-test", "clock-split-elastic", "--clock-elasticity", "0.7"])
    assert got.returncode == exit_codes.REFUSED
    assert "needs a SOURCE" in got.stdout and "unrecognized arguments" not in got.stdout
    got = run(["--self-test", "clock-split-elastic", "--clock-elasticity", "0.7",
               "0.6", "--clock-elasticity-source", "x"])
    assert got.returncode == exit_codes.REFUSED
    assert "takes ETA or ETA LO HI, got 2 number(s)" in got.stdout
    # A world that plants no elasticity refuses one: its registration is about
    # its own plants.
    got = run(["--self-test", "refit", "--clock-elasticity", "0.7",
               "--clock-elasticity-source", "x"])
    assert got.returncode == exit_codes.REFUSED
    assert "a planted world's registration is about its own planted eta" in got.stdout


def test_the_clock_correction_is_out_of_the_run_id():
    """It changes no verdict, which is the only reason a re-analysis knob may
    sit outside the key; the docstring says so where the OUT list lives."""
    base = ["--device-memory-gb", "140"]
    a = PW.build_parser().parse_args(base)
    b = PW.build_parser().parse_args(base + ["--clock-elasticity", "0.7436",
                                             "0.7277", "0.7559",
                                             "--clock-elasticity-source", "x"])
    assert PW.default_run_id(a, "nvidia_h200") == PW.default_run_id(b, "nvidia_h200")
    assert "--clock-elasticity" in " ".join(PW.default_run_id.__doc__.split())


def test_the_elastic_world_prints_recovers_and_stays_invalid_end_to_end():
    """The whole page, through the CLI: --self-test supplies the planted eta
    itself, the payload carries the block with its provenance, World.check
    holds the exact identity, and the exit is 3 INVALID because V7 FAILs."""
    got = run(["--self-test", "clock-split-elastic"])
    assert got.returncode == exit_codes.INVALID, got.stdout[-2000:]
    assert "SELF-TEST OK" in got.stdout
    assert "clock-corrected ratio" in got.stdout and "PLANTED by --self-test" in got.stdout
    results = {r.name: r.verdict for r in exit_codes.parse_result_lines(got.stdout)}
    assert results["V7"] == exit_codes.FAIL and results["C1"] == exit_codes.PASS
    # And a planted eta that the report does not recover is a mismatch.
    world = PW.WORLDS["clock-split-elastic"]
    report = types.SimpleNamespace(
        gates=[types.SimpleNamespace(tag=k, verdict=v) for k, v in world.expect.items()],
        payload={"clock_correction": {"ratio": 1.0, "ratio_raw": 1.0}})
    bad = world.check(report)
    assert any("clock_correction: corrected / raw = 1.0" in b for b in bad), bad


def test_no_standing_prose_calls_the_elasticity_unknown():
    """Session 4 measured it. A description left standing after the behaviour
    changed is this repository's recurring defect."""
    samples = _pair_world(private_clock=1400.0)
    gate = PW.gate_v7_clock_parity(samples, treads=[1, 2, 3])
    assert "unknown elasticity" not in gate.consequence
    assert "unknown elasticity" not in PW.__doc__
    assert "clock_elasticity" in PW.__doc__ or "clock-elasticity" in PW.__doc__



# --------------------------------------------------------------------------
# 13e. the probe under a CUDA graph: GPU time, and NATIVE as a bound
# --------------------------------------------------------------------------

class _Timing:
    def __init__(self, ms_p50, host_bound=False, host_note=""):
        self.ms_p50, self.host_bound, self.host_note = ms_p50, host_bound, host_note


def _fake_timer(ms_p50, log, **fields):
    """A `time_kernel`-shaped timer: records the callable and kw, runs the
    callable ONCE, returns a KernelTiming-shaped object."""
    def timer(fn, **kw):
        log.append((fn, kw))
        fn()
        return _Timing(ms_p50, **fields)
    return timer


def test_a_probe_cell_is_graph_timed_with_n_calls_per_replay_and_its_ms_is_per_call():
    graph_log, eager_log, ran = [], [], []
    cell = PW.time_probe_cell(
        lambda: ran.append(1), arm=PW.SHARED, tread=3, numel=768, declared=72,
        repeat=1, reference_clock=1485.0,
        calls_per_replay=PW.PROBE_CALLS_PER_REPLAY,
        graph_timer=_fake_timer(0.16, graph_log),
        eager_timer=_fake_timer(9.9, eager_log))
    assert len(ran) == PW.PROBE_CALLS_PER_REPLAY        # one replay = N calls
    assert cell.ms == pytest.approx(0.16 / PW.PROBE_CALLS_PER_REPLAY)
    assert cell.graph_calls == PW.PROBE_CALLS_PER_REPLAY
    assert cell.replay_ms == 0.16
    assert cell.host_bound is False
    assert graph_log[0][1] == dict(warmup_ms=PW.PROBE_WARMUP_MS,
                                   target_ms=PW.PROBE_TARGET_MS,
                                   trials=PW.PROBE_TRIALS, l2_flush=False,
                                   reference_clock_mhz=1485.0)
    assert eager_log == []
    # The eager path: one call per timing, no replay fields.
    ran.clear()
    cell = PW.time_probe_cell(
        lambda: ran.append(1), arm=PW.SHARED, tread=3, numel=768, declared=72,
        repeat=1, reference_clock=None, calls_per_replay=0,
        graph_timer=_fake_timer(0.16, graph_log),
        eager_timer=_fake_timer(0.03, eager_log, host_bound=True, host_note="h"))
    assert len(ran) == 1 and cell.ms == 0.03
    assert cell.graph_calls == 0 and cell.replay_ms is None
    assert cell.host_bound is True and cell.host_note == "h"


def test_a_refused_capture_on_any_cell_reruns_the_whole_probe_eagerly_and_the_note_says_so():
    """ONE SERIES, ONE INSTRUMENT. The refusal lands on the FOURTH cell, not
    the first: the three graph cells already collected are discarded and
    every cell is re-timed eagerly, so no series carries two instruments and
    nothing escapes to main as an ERROR."""
    from moe.bench.timing import NotCapturable
    graph_calls_made, eager_log = [], []

    def graph_timer(fn, **kw):
        graph_calls_made.append(fn)
        if len(graph_calls_made) == 4:
            raise NotCapturable("operation not permitted when stream is capturing")
        fn()
        return _Timing(0.16)
    treads = [1, 2]
    probe = PW.probe_cells(
        CFG, block_m=32, treads=treads,
        declared_by_arm={PW.NATIVE: 8, PW.SHARED: 72, PW.PRIVATE: 72},
        copies_declared=9, reference_clock=None, repeats=2,
        calls_per_replay=PW.PROBE_CALLS_PER_REPLAY,
        op=lambda ids, bm, d: None, sync=lambda: None,
        graph_timer=graph_timer, eager_timer=_fake_timer(0.03, eager_log),
        device="cpu")
    assert len(probe.cells) == 2 * 2 * 3
    assert {c.graph_calls for c in probe.cells} == {0}
    assert all(c.replay_ms is None for c in probe.cells)
    assert len(eager_log) == 2 * 2 * 3
    assert len(graph_calls_made) == 4
    assert "capture refused" in probe.note and "not permitted" in probe.note
    assert "EAGERLY" in probe.note
    for label in PW.ARMS:
        assert probe.graph_calls(label) == 0
    # And the graph path, unrefused, stamps every cell with the count.
    ok = PW.probe_cells(
        CFG, block_m=32, treads=treads,
        declared_by_arm={PW.NATIVE: 8, PW.SHARED: 72, PW.PRIVATE: 72},
        copies_declared=9, reference_clock=None, repeats=2,
        calls_per_replay=PW.PROBE_CALLS_PER_REPLAY,
        op=lambda ids, bm, d: None, sync=lambda: None,
        graph_timer=_fake_timer(0.16, []), eager_timer=_fake_timer(0.03, []),
        device="cpu")
    assert {c.graph_calls for c in ok.cells} == {PW.PROBE_CALLS_PER_REPLAY}
    assert ok.note == ""
    assert ok.graph_calls(PW.PRIVATE) == PW.PROBE_CALLS_PER_REPLAY


def test_a_series_timed_on_two_instruments_is_refused_not_fitted():
    series = _series(0.0, 4)
    cells = []
    for rep in range(3):
        for n, numel, ms in series:
            g = 16 if rep < 2 else 0
            cells.append(PW.ProbeCell(PW.SHARED, n, numel, 72, rep, ms,
                                      host_bound=False, graph_calls=g))
            cells.append(PW.ProbeCell(PW.NATIVE, n, numel, 8, rep, ms,
                                      host_bound=False, graph_calls=16))
    probe = PW.AlignProbe(tuple(cells), synthetic=True)
    with pytest.raises(PW.Unmeasurable, match="one series, one instrument"):
        probe.graph_calls(PW.SHARED)
    assert probe.graph_calls(PW.NATIVE) == 16
    assert probe.graph_calls("nobody") == 0
    gate = PW.gate_v8_alignment(probe, treads=[1, 2, 3, 4, 5, 6],
                                census=_census(), weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.UNKNOWN
    assert gate.measured == "the ratio declaration was not probed"
    assert any(ln.startswith("shared: series not fitted") for ln in gate.lines)


def test_v8_passes_a_graph_timed_flat_series_and_bounds_natives_switch_in_gpu_time():
    """THE PAGE SESSION 4 WOULD HAVE PRINTED under the graph: every series
    flat, host_bound False. PASS on the ratio series; NATIVE's unresolved
    step is a BOUND, no ASSUMED line, no graph remedy."""
    treads = [1, 2, 3, 4, 5, 6]
    probe = _probe_from({PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.0, 4),
                         PW.PRIVATE: _series(0.0, 4)}, spread=1e-4,
                        graph_calls=16)
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.PASS, gate.lines
    joined = "\n".join(gate.lines)
    bound = next(ln for ln in gate.lines
                 if ln.startswith("NATIVE's switch NOT RESOLVED IN GPU TIME"))
    assert "us per call" in bound and "a bound on the KERNEL" in bound
    reading = PW.read_probe(probe, PW.NATIVE, _census())
    assert f"threshold {reading.fit.threshold_ms() * 1e3:.2f} us" in bound
    assert "ASSUMED: the host's enqueue cost" not in joined
    assert "time the op under a CUDA graph" not in joined
    assert "under a CUDA graph, 16 calls per replay" in joined
    native_line = next(ln for ln in gate.lines if ln.startswith("native "))
    assert native_line.endswith("[GPU time: 16 calls per graph replay]")
    # The eager instrument says so on its own line.
    eager = PW.gate_v8_alignment(
        _probe_from({PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.0, 4)},
                    spread=1e-4), treads=treads, census=_census(),
        weight_stream_ms=0.64)
    assert any(ln.startswith("the probe timed the op EAGERLY") for ln in eager.lines)
    assert any(ln.endswith("[eager]") for ln in eager.lines)


def test_v8_confirms_the_control_in_gpu_time_when_natives_step_resolves_at_the_census_tread():
    treads = [1, 2, 3, 4, 5, 6]
    probe = _probe_from({PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4),
                         PW.PRIVATE: _series(0.0, 4)}, spread=1e-4,
                        graph_calls=16)
    gate = PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.PASS, gate.lines
    line = next(ln for ln in gate.lines
                if ln.startswith("POSITIVE CONTROL CONFIRMED IN GPU TIME"))
    assert "tread 4" in line
    # The host-cost assumption is moot inside a graph and is not printed (the
    # bias bound's own "ASSUMED ratio <= 1" line is a different assumption).
    assert not any("ASSUMED: the host's enqueue cost" in ln for ln in gate.lines)
    # Resolved elsewhere: informational, still PASS, V5's tread is the probe's.
    other = _probe_from({PW.NATIVE: _series(0.02, 3), PW.SHARED: _series(0.0, 4),
                         PW.PRIVATE: _series(0.0, 4)}, spread=1e-4,
                        graph_calls=16)
    gate = PW.gate_v8_alignment(other, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.PASS
    assert any("resolved at tread 3 in GPU time, not the census tread 4" in ln
               for ln in gate.lines), gate.lines
    tread, why = PW.native_switch_source(other, _census())
    assert tread == 3 and "GPU time by construction" in why


def test_a_graph_timed_probe_called_host_bound_names_the_graph_and_the_right_remedy():
    """Branch (b): the replay's launch outran N calls. UNKNOWN as the eager
    hot case, but the page names the graph, the anomaly and the RIGHT remedy,
    and the ASSUMED line stays (host time IS in these cells)."""
    treads = [1, 2, 3, 4, 5, 6]
    flat = {PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.0, 4),
            PW.PRIVATE: _series(0.0, 4)}
    hot = _probe_from(flat, spread=1e-4, host_bound=True, graph_calls=16)
    gate = PW.gate_v8_alignment(hot, treads=treads, census=_census(),
                                weight_stream_ms=0.64)
    assert gate.verdict == exit_codes.UNKNOWN
    joined = "\n".join(gate.lines)
    nc = next(ln for ln in gate.lines if ln.startswith("POSITIVE CONTROL NOT CONFIRMED"))
    assert "under a CUDA graph with 16 calls per replay" in nc and "anomaly" in nc
    assert "raise PROBE_CALLS_PER_REPLAY" in nc
    assert "time the op under a CUDA graph" not in joined
    assert "ASSUMED: the host's enqueue cost" in joined
    assert "the replay's launch outran 16 calls" in joined
    # The same cells timed eagerly keep the eager remedy, now naming the default.
    eager = PW.gate_v8_alignment(
        _probe_from(flat, spread=1e-4, host_bound=True,
                    note="capture refused (x); the WHOLE probe was re-run EAGERLY"),
        treads=treads, census=_census(), weight_stream_ms=0.64)
    joined = "\n".join(eager.lines)
    assert "time the op under a CUDA graph (the probe's default; this probe ran eagerly" in joined
    assert "timed eagerly" in joined
    # NATIVE's own graph-timed hot cells are not admissible even when they
    # resolve a step, with the anomaly named.
    stepped = _probe_from({PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4)},
                          spread=1e-4, host_bound=True, graph_calls=16)
    reading = PW.read_probe(stepped, PW.NATIVE, _census())
    assert reading.real
    ok, why = PW.native_step_is_admissible(reading, stepped)
    assert ok is False and "UNDER A CUDA GRAPH of 16 calls" in why


def test_the_planted_probe_is_graph_timed_like_the_real_one_except_in_the_eager_worlds():
    treads = [1, 2, 3, 4, 5, 6]
    kw = dict(block_m=32, treads=treads,
              declared_by_arm={PW.NATIVE: 8, PW.SHARED: 72, PW.PRIVATE: 72},
              census=_census(), noise=0.0, seed=0, weight_stream_ms=0.64)
    refit = PW.planted_probe(PW.WORLDS["refit"], CFG, **kw)
    assert {c.graph_calls for c in refit.cells} == {PW.PROBE_CALLS_PER_REPLAY}
    assert all(c.replay_ms == pytest.approx(c.ms * PW.PROBE_CALLS_PER_REPLAY)
               for c in refit.cells)
    for name in ("host-bound-probe", "host-bound-controlled"):
        eager = PW.planted_probe(PW.WORLDS[name], CFG, **kw)
        assert {c.graph_calls for c in eager.cells} == {0}, name
        assert all(c.replay_ms is None for c in eager.cells), name
    first = refit.as_dict()["cells"][0]
    assert "graph_calls" in first and "replay_ms" in first


def test_the_graph_probe_world_differs_from_the_host_bound_world_in_instrument_and_verdict():
    a, b = PW.WORLDS["host-bound-probe"], PW.WORLDS["graph-probe-unresolved"]
    assert a.native_probe_step_ms == b.native_probe_step_ms == 0.0
    assert a.probe_graph_calls == 0 and b.probe_graph_calls == PW.PROBE_CALLS_PER_REPLAY
    assert a.probe_host_bound is True and b.probe_host_bound is False
    assert a.expect["V8"] == exit_codes.UNKNOWN and b.expect["V8"] == exit_codes.PASS
    got = run(["--self-test", "graph-probe-unresolved"])
    assert got.returncode == exit_codes.DONE, got.stdout[-1500:]
    assert "RESULT: VALIDITY V8 PASS" in got.stdout
    assert "NOT RESOLVED IN GPU TIME" in got.stdout
    assert "SELF-TEST OK" in got.stdout


def test_analyse_names_the_gpu_time_bound_when_the_graph_probe_resolves_no_native_step():
    flat = {PW.NATIVE: _series(0.0, 4), PW.SHARED: _series(0.0, 4)}
    tread, text = PW.native_switch_source(
        _probe_from(flat, spread=1e-4, graph_calls=16), _census())
    assert tread == 4
    assert "IN GPU TIME" in text and "threshold" in text and "under that per call" in text
    tread, text = PW.native_switch_source(_probe_from(flat, spread=1e-4), _census())
    assert tread == 4
    assert "the hypothesis stands unconfirmed" in text and "IN GPU TIME" not in text
    stepped = {PW.NATIVE: _series(0.02, 3), PW.SHARED: _series(0.0, 4)}
    tread, text = PW.native_switch_source(
        _probe_from(stepped, spread=1e-4, graph_calls=16), _census())
    assert tread == 3 and "the probe, which resolved native's step at tread 3" in text
    # Resolved AT the census tread but host-timed: the tread is the census's,
    # and the sentence no longer says "no step".
    at4 = {PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4)}
    tread, text = PW.native_switch_source(
        _probe_from(at4, spread=1e-4, host_bound=True), _census())
    assert tread == 4 and "resolved native's step at that tread too" in text
    assert PW.native_switch_source(None, _census()) == (
        4, "the cited hypothesis (tread 4)")



# --------------------------------------------------------------------------
# 19. the cross-run spread: C1 over the envelope of several runs
# --------------------------------------------------------------------------

def _reading(ratio, half=0.002, **kw):
    return PW.RunReading(ratio, (ratio - half, ratio + half), **kw)


def _measured_shaped_report(tmp_path, name, alpha_shared, treads, seed=0):
    """A report.json a MEASURED run would have written, from planted cells at
    a stated alpha: the payload `analyse` hands the writer, with `synthetic`
    cleared. The ratio is moved through the planted samples, never by editing
    the number."""
    kw = dict(block_m=32, treads=treads, repeats=3, ridge=160.0,
              bandwidth_gbps=4000.0, b=2, noise=0.0, seed=seed,
              copies_declared=9, native_switch=4)
    samples = PW.planted_samples(PW.WORLDS["refit"], CFG,
                                 alpha_shared=alpha_shared, **kw)
    # The pinned dict a real run writes, from the parser's own defaults, so
    # the dry run's design comparison sees a replicate and not a stranger.
    defaults = PW.build_parser().parse_args([])
    pinned = dict(PW.SWEEP.FIXED, num_stages=defaults.num_stages,
                  GROUP_SIZE_M=defaults.group_m, BLOCK_SIZE_N=defaults.block_n)
    report = _analyse(samples, treads, draws=50, run_id=name, pinned=pinned,
                      seed=seed)
    payload = dict(report.payload)
    payload["synthetic"] = False
    payload["provenance"] = {"utc": f"2026-09-21T2{seed}:00:00Z",
                             "hostname": "planted", "git_sha": "0" * 7,
                             "git_dirty": False}
    p = tmp_path / name / "report.json"
    p.parent.mkdir()
    p.write_text(json.dumps(payload, indent=2))
    return p, payload


def test_c1_over_two_runs_is_scored_on_the_envelope_and_names_the_spread():
    hi_edge = PW.ALPHA_BAND[1]
    a = _reading(hi_edge - 0.005)
    b = _reading(a.ratio + 0.022)
    assert PW.c1_verdict(a.ratio, a.interval) == exit_codes.PASS
    assert PW.c1_verdict(b.ratio, b.interval) == exit_codes.FAIL
    cross = PW.cross_run([a, b])
    assert cross.spread == pytest.approx(0.022)
    assert cross.envelope == (a.interval[0], b.interval[1])
    assert cross.verdict == exit_codes.UNKNOWN       # one in, one past the edge
    assert cross.disjoint_pairs() == [(0, 1)]
    assert cross.sd is None
    assert PW.cross_run([a]).verdict == PW.c1_verdict(a.ratio, a.interval)
    # Both past the edge: the envelope misses the band and the joint is FAIL.
    c = _reading(hi_edge + 0.01)
    assert PW.cross_run([b, c]).verdict == exit_codes.FAIL
    # Three points give an sd; the lines carry every quantity the page needs.
    three = PW.cross_run([a, _reading(a.ratio + 0.001), _reading(a.ratio - 0.001)])
    assert three.sd is not None and three.verdict == exit_codes.PASS
    joined = "\n".join(three.lines())
    assert "spread of the points" in joined and "envelope" in joined
    assert "does not cover the run-to-run spread" in joined
    d = three.as_dict()
    assert d["n"] == 3 and d["verdict"] == exit_codes.PASS and len(d["runs"]) == 3


def test_a_lone_run_says_it_was_scored_alone():
    """OPT-IN, said out loud: a run with no replicate is scored on its own
    interval, and the page says that is a within-run statement. The sentence
    is absent on the parent."""
    got = run(["--self-test", "refit"])
    assert "scored on this run's within-run interval alone" in got.stdout
    assert "no replicate was named (--replicate-of)" in got.stdout
    assert "replicates  (none: this run is scored ALONE" in got.stdout


def test_the_page_names_the_cross_run_spread_and_the_verdict_it_implies(tmp_path):
    treads = [1, 2, 3, 4, 5, 6]
    pa, payload_a = _measured_shaped_report(tmp_path, "run-a", PW.ALPHA, treads)
    pb, payload_b = _measured_shaped_report(tmp_path, "run-b", 0.605, treads, seed=1)
    assert PW.c1_verdict(payload_a["ratio"], tuple(payload_a["ratio_interval"])) \
        == exit_codes.PASS
    assert PW.c1_verdict(payload_b["ratio"], tuple(payload_b["ratio_interval"])) \
        == exit_codes.FAIL
    design = {k: payload_a[k] for k in PW.DESIGN_KEYS}
    readings = PW.load_replicates([pb], design=design, card_known=True,
                                  this_run_id="run-a")
    assert len(readings) == 1 and readings[0].run_id == "run-b"
    assert readings[0].seed == 1 and readings[0].slopes[PW.SHARED] > 0
    kw = dict(block_m=32, treads=treads, repeats=3, ridge=160.0,
              bandwidth_gbps=4000.0, b=2, noise=0.0, seed=0,
              copies_declared=9, native_switch=4)
    samples = PW.planted_samples(PW.WORLDS["refit"], CFG, alpha_shared=PW.ALPHA, **kw)
    report = _analyse(samples, treads, draws=50, replicates=tuple(readings),
                      run_id="run-a")
    text = report.text()
    this = PW.run_reading(report.payload, None)
    cross = PW.cross_run([this, *readings])
    assert f"spread of the points {cross.spread:.4f}" in text
    assert "within-run intervals that do not overlap: runs 0 and 1" in text
    c1 = next(g for g in report.gates if g.tag == "C1")
    assert c1.verdict == cross.verdict == exit_codes.UNKNOWN
    assert report.payload["c1_verdict_alone"] == exit_codes.PASS
    assert "this run alone would read PASS" in text
    assert "over 2 runs: spread" in c1.measured
    assert "ENVELOPE" in c1.threshold
    assert report.payload["replicates"]["spread"] == pytest.approx(cross.spread)
    assert report.payload["replicates"]["runs"][1]["run_id"] == "run-b"
    assert report.payload["run_id"] == "run-a"
    json.loads(json.dumps(report.payload), parse_constant=_no_json_constants)


def test_a_replicate_of_another_design_or_a_duplicate_is_refused_before_the_card(tmp_path):
    """Every refusal is asserted on its SENTENCE: argparse's unknown-flag exit
    is also 2 == REFUSED, so a return code alone is green on the parent."""
    treads = [1, 2, 3, 4, 5, 6]
    pa, payload_a = _measured_shaped_report(tmp_path, "run-a", PW.ALPHA, treads)
    g16 = dict(payload_a)
    g16["pinned"] = dict(payload_a["pinned"], GROUP_SIZE_M=16)
    pg = tmp_path / "g16.json"
    pg.write_text(json.dumps(g16))
    base = ["--dry-run", "--device-memory-gb", "140", "--repeats", "3"]
    got = run(base + ["--replicate-of", str(pg)])
    assert "REFUSED: --replicate-of" in got.stdout and "differs in pinned" in got.stdout
    assert "unrecognized arguments" not in got.stderr
    got = run(base + ["--replicate-of", str(pa), str(pa)])
    assert "REFUSED" in got.stdout and "named twice" in got.stdout
    planted = dict(payload_a, synthetic=True)
    pp = tmp_path / "planted.json"
    pp.write_text(json.dumps(planted))
    got = run(base + ["--replicate-of", str(pp)])
    assert "REFUSED" in got.stdout and "planted (--self-test) report" in got.stdout
    got = run(base + ["--replicate-of", str(tmp_path / "nope.json")])
    assert "REFUSED" in got.stdout and "no such report" in got.stdout
    # A matching one is ADMITTED on the plan page before the dry run's own
    # refusal, and named with its ratio.
    got = run(base + ["--replicate-of", str(pa)])
    assert got.returncode == exit_codes.REFUSED
    assert "replicates  1 earlier run(s) of this design" in got.stdout
    assert f"ratio {payload_a['ratio']:.4f}" in got.stdout
    assert "REFUSED: --replicate-of" not in got.stdout


def test_a_planted_world_refuses_a_measured_replicate(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    got = run(["--self-test", "refit", "--replicate-of", str(p)])
    assert "REFUSED" in got.stdout
    assert "a planted world has no measured replicate" in got.stdout
    assert exit_codes.parse_result_lines(got.stdout) == []


def test_the_payload_carries_the_run_id_the_seed_and_the_session_tag():
    argv = ["--self-test", "refit", "--draws", "5", "--seed", "3",
            "--session-tag", "t"]
    rc, payload, _ = _payload_for(argv)
    assert payload["seed"] == 3 and payload["session_tag"] == "t"
    expect = PW.default_run_id(PW.build_parser().parse_args(argv),
                               PW.detect_card_slug())
    assert payload["run_id"] == expect and payload["run_id"].startswith("synthetic-")
    assert payload["replicates"] is None
    c1 = next(g for g in payload["gates"] if g["tag"] == "C1")
    assert payload["c1_verdict_alone"] == c1["verdict"]


def test_read_mode_rescores_a_stored_pair_without_measuring_or_writing(tmp_path):
    treads = [1, 2, 3, 4, 5, 6]
    pa, payload_a = _measured_shaped_report(tmp_path, "run-a", PW.ALPHA, treads)
    pb, payload_b = _measured_shaped_report(tmp_path, "run-b", 0.605, treads, seed=1)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    got = run(["--read", str(pa), "--replicate-of", str(pb)])
    assert "READ MODE: nothing measured, nothing written" in got.stdout, got.stdout[-800:]
    assert "experiment  private_weight_reference / run-a" in got.stdout
    lines = exit_codes.parse_result_lines(got.stdout)
    stored = [g["tag"] for g in payload_a["gates"]]
    assert [ln.name for ln in lines] == stored and "C1" in stored
    a = PW.run_reading(payload_a, pa)
    b = PW.run_reading(payload_b, pb)
    cross = PW.cross_run([a, b])
    c1 = next(ln for ln in lines if ln.name == "C1")
    assert c1.verdict == cross.verdict == exit_codes.UNKNOWN
    assert f"spread of the points {cross.spread:.4f}" in got.stdout
    assert exit_codes.classify_text(got.stdout) == got.returncode
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    # Alone there is nothing to read it against.
    got = run(["--read", str(pa)])
    assert "REFUSED" in got.stdout and "nothing to read it against" in got.stdout
    # The two files are the same run twice: refused, not averaged.
    got = run(["--read", str(pa), "--replicate-of", str(pa)])
    assert "REFUSED" in got.stdout and "this run's own report" in got.stdout


def test_a_run_that_formed_no_interval_still_prints_the_replicates_it_was_given(tmp_path):
    """The second C1 construction site: `--draws 1` forms no interval, so this
    run enters no reading; the replicates are read together anyway and C1
    stays UNKNOWN for this run."""
    treads = [1, 2, 3, 4, 5, 6]
    pb, payload_b = _measured_shaped_report(tmp_path, "run-b", PW.ALPHA, treads, seed=1)
    design = {k: payload_b[k] for k in PW.DESIGN_KEYS}
    readings = PW.load_replicates([pb], design=design, card_known=True)
    kw = dict(block_m=32, treads=treads, repeats=3, ridge=160.0,
              bandwidth_gbps=4000.0, b=2, noise=0.0, seed=0,
              copies_declared=9, native_switch=4)
    samples = PW.planted_samples(PW.WORLDS["refit"], CFG, alpha_shared=PW.ALPHA, **kw)
    report = _analyse(samples, treads, draws=1, replicates=tuple(readings))
    c1 = next(g for g in report.gates if g.tag == "C1")
    assert c1.verdict == exit_codes.UNKNOWN
    assert "no interval was formed" in c1.measured
    assert any("this run enters no reading" in ln for ln in c1.lines)
    assert any(ln.startswith("REPLICATES: 1 run(s)") for ln in c1.lines)
    assert report.payload["replicates"]["n"] == 1



# --------------------------------------------------------------------------
# 20. the duty cycle: bursts sized to keep both arms off the power cap (DESIGN DECISION 15)
# --------------------------------------------------------------------------

class _CellTiming:
    """`time_kernel`'s return, with the fields `KernelTiming` carries: the
    power, the host verdict and the clock list included, because a fake
    without them is how "the full-duty timer records no power" stayed green
    while the real one recorded it."""
    def __init__(self, **kw):
        defaults = dict(ms_p50=0.5, ms_min=0.49, ms_std=0.01, iters=40, trials=3,
                        warmup_ms=300.0, instrument="fake/time_kernel",
                        sm_clock_load_mhz=1455.0, clock_level_ok=True,
                        clock_level_side="", clock_drift_ok=True, l2_flush=True,
                        power_w=695.0, host_bound=True,
                        clock_samples_mhz=(1455.0, 1440.0, 1455.0))
        defaults.update(kw)
        self.__dict__.update(defaults)


class _CellDutyTiming(_CellTiming):
    """`clock_elasticity.time_duty`'s return, every diagnostic it measures."""
    def __init__(self, **kw):
        # 3 trials x 5 bursts x (80 calls - the discarded lead): `samples` is
        # the kept calls summed over EVERY trial, as `time_duty` returns it.
        super().__init__(**{**dict(
            samples=3 * 5 * 79, power_w=480.0, host_bound=False,
            clock_note="LEVEL HIGH (recorded)", instrument="fake/time_duty",
            sm_clock_load_mhz=1965.0, duty_achieved=0.47, calls_per_burst=80,
            gap_ms=40.0, head_ms=0.51, tail_ms=0.52, within_burst_ok=True,
            clock_samples_mhz=(1965.0, 1950.0, 1965.0)), **kw})


def test_the_duty_knob_keeps_every_full_duty_run_id_and_moves_the_others():
    base = PW.default_run_id(_args(), "NVIDIA H200")
    assert PW.default_run_id(_args(**{"--duty": 1.0}), "NVIDIA H200") == base
    assert PW.default_run_id(_args(**{"--duty": 0.5}), "NVIDIA H200") != base


def test_a_duty_outside_the_unit_interval_is_refused_before_anything():
    for bad in ("0", "1.5", "-0.5"):
        got = run(["--dry-run", "--device-memory-gb", "140", "--duty", bad])
        assert got.returncode == exit_codes.REFUSED
        assert "REFUSED: --duty" in got.stdout, got.stdout[-400:]
        assert "unrecognized arguments" not in got.stderr


def test_the_plan_page_prices_the_duty_and_says_what_it_buys():
    """The printed formats the chain's helpers parse are held byte-compatible
    (the duty line's head, the kernel estimate, the wall figure); what the
    page PROMISES changed. It said "V7 holds by construction" at duty 0.5,
    and session 4's clock arm says the clock still tracked power there
    (findings 0, 2 and 29): the page now says the duty's adequacy is
    measured, names the evidence, and that V7 checks it."""
    full = run(["--dry-run", "--device-memory-gb", "140"])
    half = run(["--dry-run", "--device-memory-gb", "140", "--duty", "0.5"])
    quarter = run(["--dry-run", "--device-memory-gb", "140", "--duty",
                   str(PW.FLAT_DUTY)])
    assert "duty        1.00: the queue kept full" in full.stdout
    assert f"(--duty {PW.FLAT_DUTY} is the setting" in full.stdout
    assert "WALL CLOCK" not in full.stdout
    assert "duty        0.50: every cell timed as bursts of ~40 ms" in half.stdout
    assert "idle gaps of 40 ms" in half.stdout
    assert f"duty        {PW.FLAT_DUTY:.2f}: every cell timed as bursts" in quarter.stdout
    for page in (full.stdout, half.stdout, quarter.stdout):
        assert "holds by construction" not in page
        assert "boost ceiling" not in page
    assert "Whether this duty is low enough is measured, not assumed" in half.stdout
    assert PW.FLAT_DUTY_EVIDENCE in half.stdout and PW.FLAT_DUTY_EVIDENCE in quarter.stdout
    assert "V7 checks it here and a FAIL names a lower duty" in quarter.stdout
    # At the pod setting the plan carries the owner's reading of that FAIL
    # (XS-1), and above it, where the chain never runs, it does not.
    assert f"(at this duty a FAIL means {PW.V7_FAIL_AT_FLAT_DUTY})" in quarter.stdout
    assert "at this duty a FAIL means" not in half.stdout
    # V7's registration says where "by construction" applies: full duty.
    assert "at FULL duty a card that cannot lock its clock fails this by " \
        "construction" in quarter.stdout
    est = lambda s: int(re.search(r"estimated GPU time (\d+) s", s).group(1))  # noqa: E731
    for page, duty in ((half.stdout, 0.5), (quarter.stdout, PW.FLAT_DUTY)):
        m = re.search(r"the ladder's (\d+) s of kernel time takes about (\d+) s", page)
        assert m, page[-1500:]
        assert int(m.group(2)) == pytest.approx(int(m.group(1)) / duty, abs=2)
        # The kernel estimate itself does not move: the same calls, the same bytes.
        assert est(full.stdout) == est(page)


def test_time_cell_at_a_duty_sizes_the_bursts_off_a_short_reading_and_records_power():
    calls = []

    def timer(fn, **kw):
        calls.append(("timer", kw))
        fn()
        return _CellTiming(ms_p50=0.5)

    def duty_timer(fn, **kw):
        calls.append(("duty", kw))
        fn()
        return _CellDutyTiming(ms_p50=0.52)
    ran = []
    ct = PW.time_cell(lambda: ran.append(1), duty=0.5, warmup_ms=300.0,
                      cell_budget_ms=200.0, trials=3, l2_flush=True,
                      reference_clock_mhz=1455.0, timer=timer,
                      duty_timer=duty_timer)
    assert [c[0] for c in calls] == ["timer", "duty"]
    sizing = calls[0][1]
    assert sizing["target_ms"] == PW.DUTY_SIZING_MS and sizing["trials"] == 1
    assert sizing["warmup_ms"] == PW.DUTY_SIZING_MS      # min(300, 20)
    kw = calls[1][1]
    assert kw["duty"] == 0.5
    assert kw["calls_per_burst"] == round(PW.DUTY_BURST_MS / 0.5)     # 80
    assert kw["bursts"] == round(200.0 / PW.DUTY_BURST_MS)          # 5
    assert kw["per_call_ms"] == 0.5 and kw["trials"] == 3
    assert kw["warm_ms"] == 300.0 and kw["l2_flush"] is True
    assert kw["reference_clock_mhz"] == 1455.0
    assert ct.duty == 0.5 and ct.power_w == 480.0 and ct.ms_p50 == 0.52
    # ITERATIONS PER TRIAL, as at full duty: the kept calls of one trial.
    assert ct.iters == 5 * 79 and ct.trials == 3
    assert ct.instrument == "fake/time_duty"
    assert ct.sm_clock_load_mhz == 1965.0 and ct.host_bound is False
    assert ct.note.startswith("LEVEL HIGH")
    # Everything the duty timer measured is kept (findings 17 and 24).
    assert (ct.duty_achieved, ct.calls_per_burst, ct.gap_ms) == (0.47, 80, 40.0)
    assert (ct.head_ms, ct.tail_ms, ct.within_burst_ok) == (0.51, 0.52, True)
    assert ct.clock_samples_mhz == "1965 1950 1965"
    # Full duty never touches the duty timer, and records the power, host
    # verdict and clock list `time_kernel` read (finding 18: it does read
    # power, and this assertion said None only because the fake had none).
    calls.clear()
    ct = PW.time_cell(lambda: None, duty=1.0, warmup_ms=300.0,
                      cell_budget_ms=200.0, trials=3, l2_flush=True,
                      reference_clock_mhz=1455.0, timer=timer,
                      duty_timer=duty_timer)
    assert [c[0] for c in calls] == ["timer"]
    assert calls[0][1]["target_ms"] == 200.0 and calls[0][1]["warmup_ms"] == 300.0
    assert ct.duty == 1.0 and ct.instrument == "fake/time_kernel"
    assert ct.power_w == _CellTiming().power_w and ct.host_bound is True
    assert ct.clock_samples_mhz == "1455 1440 1455"
    assert (ct.duty_achieved, ct.calls_per_burst, ct.gap_ms, ct.head_ms,
            ct.tail_ms, ct.within_burst_ok) == (None,) * 6
    assert ct.iters == 40 and ct.note == ""


def test_the_duty_and_the_power_travel_through_the_csv(tmp_path):
    path = tmp_path / "cells.csv"
    store = PW.Store(path, PW.CSV_FIELDS)
    store.append(_sample(PW.SHARED, 2, 0, 1.0, load=1965.0))
    a = PW.replace(_sample(PW.PRIVATE, 2, 0, 1.1, load=1960.0), duty=0.5, power_w=480.0)
    store.append(a)
    back = PW.read_samples(path)
    assert [s.duty for s in back] == [1.0, 0.5]
    assert [s.power_w for s in back] == [None, 480.0]
    assert PW.duty_of(back) == 0.5
    # A file written before the column reads back as full duty.
    text = path.read_text().splitlines()
    header = text[0].split(",")
    keep = [i for i, h in enumerate(header) if h not in ("duty", "power_w")]
    (tmp_path / "old.csv").write_text("\n".join(
        ",".join(ln.split(",")[i] for i in keep) for ln in text) + "\n")
    old = PW.read_samples(tmp_path / "old.csv")
    assert [s.duty for s in old] == [1.0, 1.0] and all(s.power_w is None for s in old)


def test_v7_names_a_lower_duty_on_a_split_at_any_duty():
    """At full duty the remedy is the pod setting. BELOW full duty a split is
    a duty not yet low enough, and the page used to print no remedy there, so
    a FAIL at duty 0.5 read as a card fault under a plan that had promised V7
    would hold (findings 0, 2 and 29)."""
    treads = [1, 2, 3]
    split = _pair_world(private_clock=1425.0, shared_clock=1740.0)

    def remedy(samples):
        gate = PW.gate_v7_clock_parity(samples, treads=treads)
        assert gate.verdict == exit_codes.FAIL
        lines = [ln for ln in gate.lines if ln.startswith("the remedy is")]
        assert len(lines) == 1, gate.lines
        return lines[0]
    full = remedy(split)
    assert full.startswith(f"the remedy is --duty {PW.FLAT_DUTY}:")
    assert PW.FLAT_DUTY_EVIDENCE in full
    half = remedy([PW.replace(s, duty=0.5) for s in split])
    assert half.startswith("the remedy is a lower --duty than 0.50:")
    assert f"--duty {PW.FLAT_DUTY} is the pod setting" in half
    assert PW.FLAT_DUTY_EVIDENCE in half
    at_flat = remedy([PW.replace(s, duty=PW.FLAT_DUTY) for s in split])
    assert at_flat.startswith(f"the remedy is a lower --duty than {PW.FLAT_DUTY:.2f}:")
    assert "is the pod setting" not in at_flat
    ok = PW.gate_v7_clock_parity(_pair_world(private_clock=1965.0, shared_clock=1965.0),
                                 treads=treads)
    assert ok.verdict == exit_codes.PASS
    assert not any("remedy" in ln for ln in ok.lines)


def test_a_replicate_at_another_duty_is_refused_and_a_pre_duty_report_is_full_duty(tmp_path):
    treads = [1, 2, 3, 4, 5, 6]
    pa, payload_a = _measured_shaped_report(tmp_path, "run-a", PW.ALPHA, treads)
    assert payload_a["duty"] == 1.0
    design = {k: payload_a.get(k, PW.DESIGN_KEY_DEFAULTS.get(k)) for k in PW.DESIGN_KEYS}
    half = dict(payload_a, duty=0.5)
    ph = tmp_path / "half.json"
    ph.write_text(json.dumps(half))
    with pytest.raises(PW.PrivateWeightRefusal, match="differs in duty"):
        PW.load_replicates([ph], design=design, card_known=True)
    old = dict(payload_a)
    del old["duty"]
    po = tmp_path / "old.json"
    po.write_text(json.dumps(old))
    assert len(PW.load_replicates([po], design=design, card_known=True)) == 1
    assert "duty" in PW.DESIGN_KEYS


# --------------------------------------------------------------------------
# 21. a duty page names the instrument that timed it, and counts per trial
# --------------------------------------------------------------------------

def _provenance_instrument(monkeypatch, tmp_path, duty: str) -> str | None:
    """Drive the real `_main` of a MEASURING run (no --dry-run, no
    --self-test) to the line that builds the provenance block, and hand back
    the instrument it names. The two host checks that refuse a laptop are
    stood in; the block itself raises once it is asked, so nothing past it
    runs and nothing is written."""
    seen: dict = {}

    class _Built(Exception):
        pass

    def block(**kw):
        seen.update(kw)
        raise _Built

    monkeypatch.setattr(PW.SWEEP, "missing_gpu_stack", lambda: "")
    monkeypatch.setattr(PW, "clock_sampler_refusal", lambda: "")
    monkeypatch.setattr(PW.PV, "provenance_block", block)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_Built):
        PW._main(["--duty", duty, "--ridge", "160", "--bandwidth-gbps", "4000",
                  "--device-memory-gb", "140", "--capability", "9.0",
                  "--out", str(tmp_path)])
    assert not any(tmp_path.iterdir()), "the run wrote before its provenance"
    return seen["instrument"]


def test_a_duty_run_stamps_the_duty_timer_as_its_instrument(monkeypatch, tmp_path):
    """Finding 5. The provenance block is what every row's `prov_instrument`
    column and report.json's top-level `instrument` are written from, and at a
    duty below 1 it named the queue-deep `time_kernel` loop the cells were NOT
    timed with, while each row's own `instrument` column named the duty
    timer. It now names the duty timer's own string, the one those rows carry,
    and the duty."""
    import clock_elasticity as CE
    row_instrument = CE.DutyTiming.__dataclass_fields__["instrument"].default
    got = _provenance_instrument(monkeypatch, tmp_path, "0.25")
    assert got != TIMING.TIMING_BASIS
    assert got.startswith(row_instrument) and got.endswith("duty 0.25")
    assert PW.ladder_instrument(0.25) == got
    # Full duty keeps the queue-deep basis, which is what timed its cells.
    assert _provenance_instrument(monkeypatch, tmp_path, "1.0") == TIMING.TIMING_BASIS
    assert PW.ladder_instrument(0.5, synthetic=True) == PW.SYNTHETIC_INSTRUMENT
    # report.json's top-level key is the block's, through `stamp`.
    prov = PW.PV.Provenance(instrument=got, ridge_source="this test",
                            bandwidth_source="this test")
    treads = [1, 2, 3, 4, 5, 6]
    kw = dict(block_m=32, treads=treads, repeats=3, ridge=160.0,
              bandwidth_gbps=4000.0, b=2, noise=0.0, seed=0, copies_declared=9,
              native_switch=4)
    samples = PW.planted_samples(PW.WORLDS["refit"], CFG, alpha_shared=PW.ALPHA, **kw)
    report = _analyse(samples, treads)
    assert prov.stamp(report.payload)["instrument"] == got


def test_iterations_per_trial_mean_one_thing_at_either_duty():
    """Finding 5's second call site. At a duty below 1 `time_cell` stored the
    kept calls summed over EVERY trial, and the page printed that as
    "iterations per trial" and wrote it into provenance's `iters`: three
    times a per-trial count at the default three trials."""
    def timer(fn, **kw):
        return _CellTiming(ms_p50=0.5)

    def duty_timer(fn, **kw):
        return _CellDutyTiming(ms_p50=0.52)
    ct = PW.time_cell(lambda: None, duty=0.25, warmup_ms=300.0,
                      cell_budget_ms=200.0, trials=3, l2_flush=True,
                      reference_clock_mhz=None, timer=timer,
                      duty_timer=duty_timer)
    per_trial = _CellDutyTiming().samples // 3
    assert ct.iters == per_trial
    rows = [PW.replace(_sample(PW.SHARED, n, 0, 1.0), iters=ct.iters,
                       duty=0.25) for n in (1, 2, 3)]
    line = PW._iters_line(rows)
    assert line.startswith(f"iterations per trial: median {per_trial} ")
    assert "At duty 0.25 an iteration is a KEPT call" in line
    assert "first call of every burst discarded" in line
    prov = PW._observed_iters(PW.PV.Provenance(), rows)
    assert prov.iters == per_trial
    # Full duty keeps its sentence.
    full = [_sample(PW.SHARED, n, 0, 1.0) for n in (1, 2, 3)]
    assert PW._iters_line(full).endswith(
        "Sized per cell by the instrument from --cell-budget-ms.")


# --------------------------------------------------------------------------
# 22. the duty timer's own diagnostics, kept per cell and printed beside V7
# --------------------------------------------------------------------------

#: The columns this build adds for the duty timer's diagnostics (findings 17
#: and 24). Named here so the pre-change header below is the one a cells.csv
#: written at 12ec932 carries.
DIAGNOSTIC_COLUMNS = ("duty_achieved", "calls_per_burst", "gap_ms", "head_ms",
                      "tail_ms", "within_burst_ok", "clock_samples_mhz",
                      "host_bound")

#: The top-level keys of session 4's published report.json files (written at
#: 81f80b7, 2026-09-21, on the pod-h200-session4 branch): what a report from
#: before this build, and before the duty, run id and seed keys, carries.
SESSION4_REPORT_KEYS = (
    "experiment", "synthetic", "card", "model", "dtype", "block_m", "pinned",
    "treads", "repeats", "alpha_refit", "alpha_band", "ridge", "ridge_source",
    "bandwidth_gbps", "bandwidth_source", "roof_tflops", "roof_source",
    "reference_clock_mhz", "reference_clock_grade", "reference_clock_source",
    "weight_stream_ms", "memory_plan", "weight_delta_bytes", "high_water_bytes",
    "buffer_proof", "ladders", "treads_table", "copies_declared", "path_census",
    "align_probe", "declaration_fit", "ratio", "ratio_interval",
    "ratio_interval_pct", "ratio_draws", "ratio_corrected", "outcome",
    "outcomes_partition", "gates", "git_sha", "gpu_name", "instrument",
    "provenance")


def test_the_duty_timers_diagnostics_travel_through_the_csv_and_an_old_file_reads_none(
        tmp_path):
    """Findings 17 and 24: `time_duty` measures the achieved duty, the burst
    shape, the in-burst head and tail, every per-burst clock and the host
    verdict, and the arm kept none of them, so a V7 or V0 FAIL at a duty
    could not be read off cells.csv. They are columns now, each read back BY
    NAME, and a file written before them reads None, never 0 or False."""
    assert set(DIAGNOSTIC_COLUMNS) <= set(PW.CSV_FIELDS)
    path = tmp_path / "cells.csv"
    store = PW.Store(path, PW.CSV_FIELDS)
    full = PW.replace(_sample(PW.SHARED, 2, 0, 1.0, load=1455.0), power_w=695.0,
                      clock_samples_mhz="1455 1440", host_bound=False)
    duty = PW.replace(_sample(PW.PRIVATE, 2, 0, 1.1, load=1965.0), duty=0.25,
                      power_w=300.0, duty_achieved=0.231, calls_per_burst=80,
                      gap_ms=120.0, head_ms=0.5, tail_ms=0.495,
                      within_burst_ok=True, clock_samples_mhz="1965 1965 1980",
                      host_bound=True)
    store.append(full)
    store.append(duty)
    back = PW.read_samples(path)
    assert back == [full, duty]
    assert back[1].burst_sag == pytest.approx(-0.01)
    assert back[0].burst_sag is None
    text = path.read_text().splitlines()
    header = text[0].split(",")
    keep = [i for i, h in enumerate(header) if h not in DIAGNOSTIC_COLUMNS]
    (tmp_path / "old.csv").write_text("\n".join(
        ",".join(ln.split(",")[i] for i in keep) for ln in text) + "\n")
    old = PW.read_samples(tmp_path / "old.csv")
    for s in old:
        assert all(getattr(s, c) is None for c in DIAGNOSTIC_COLUMNS), s
        assert s.burst_sag is None
    assert [s.power_w for s in old] == [695.0, 300.0]


def test_a_pre_change_directory_is_refused_while_new_runs_and_old_reports_read(
        tmp_path):
    """The Store refuses a header it did not write, so adding the columns
    REFUSES a resume of a directory written before them. Nothing the chain
    runs is such a directory: its R3 runs are at a duty below 1, whose run id
    is not any full-duty run's, so they start a fresh file under this
    header; and --read / --replicate-of read report.json, whose keys this
    build does not touch, so a pre-change report pair still re-scores."""
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    pre_change = [c for c in PW.CSV_FIELDS if c not in DIAGNOSTIC_COLUMNS]
    with (old_dir / "cells.csv").open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=pre_change + PW.PROVENANCE_COLUMNS).writeheader()
    with pytest.raises(PW.SchemaCollision) as exc:
        PW.Store(old_dir / "cells.csv", PW.CSV_FIELDS + PW.PROVENANCE_COLUMNS)
    assert "clock_samples_mhz" in str(exc.value) and "fresh directory" in str(exc.value)
    # A duty run is a different id, so a different, empty directory.
    full = PW.default_run_id(_args(), "NVIDIA H200")
    quarter = PW.default_run_id(_args(**{"--duty": 0.25}), "NVIDIA H200")
    assert quarter != full
    fresh = tmp_path / quarter / "cells.csv"
    fresh.parent.mkdir()
    PW.Store(fresh, PW.CSV_FIELDS + PW.PROVENANCE_COLUMNS).append(
        PW.replace(_sample(PW.SHARED, 1, 0, 1.0), duty=0.25, calls_per_burst=80))
    assert PW.read_samples(fresh)[0].calls_per_burst == 80
    # A pre-change report pair, session 4's key set, re-read off GPU.
    treads = [1, 2, 3, 4, 5, 6]
    paths = []
    for name, alpha, seed in (("s4-a", PW.ALPHA, 0), ("s4-b", 0.60, 1)):
        p, payload = _measured_shaped_report(tmp_path, name, alpha, treads, seed=seed)
        old = {k: v for k, v in payload.items() if k in SESSION4_REPORT_KEYS}
        old["instrument"] = TIMING.TIMING_BASIS
        p.write_text(json.dumps(old, indent=2))
        paths.append(p)
    got = run(["--read", str(paths[0]), "--replicate-of", str(paths[1])])
    assert "READ MODE: nothing measured, nothing written" in got.stdout, (
        got.stdout[-800:] + got.stderr[-800:])
    assert "duty        1.0" in got.stdout
    assert exit_codes.classify_text(got.stdout) == got.returncode


def test_a_full_duty_command_names_its_old_directory_and_the_store_refuses_it(
        tmp_path):
    """The review's R3-RUNID-RESUME-COMMENT. The duty is out of the run id at
    1.0, so the command that wrote a session-4 directory names that same
    directory today, and the comment beside the key said the directory
    RESUMES. It does not: session 4's cells.csv carries neither the duty
    columns nor the diagnostics, and the Store refuses its header. Its rows
    still read back, with every missing field None. The comment now says what
    happens."""
    import dataclasses
    import inspect
    full = PW.default_run_id(_args(), "NVIDIA H200")
    assert PW.default_run_id(_args(**{"--duty": 1.0}), "NVIDIA H200") == full
    pre_duty = [c for c in PW.CSV_FIELDS
                if c not in DIAGNOSTIC_COLUMNS + ("duty", "power_w")]
    old = tmp_path / full / "cells.csv"
    old.parent.mkdir()
    with old.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=pre_duty + PW.PROVENANCE_COLUMNS,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerow(dataclasses.asdict(_sample(PW.SHARED, 1, 0, 1.0)))
    with pytest.raises(PW.SchemaCollision) as exc:
        PW.Store(old, PW.CSV_FIELDS + PW.PROVENANCE_COLUMNS)
    assert "'duty', 'power_w'" in str(exc.value)
    back = PW.read_samples(old)
    assert len(back) == 1 and back[0].duty == 1.0 and back[0].power_w is None
    src = " ".join(inspect.getsource(PW.default_run_id).replace("#", " ").split())
    assert "a resumed session-4 directory resumes" not in src
    assert "It does NOT resume there" in src and "SchemaCollision" in src


def test_every_column_the_instrument_measured_reaches_the_row():
    """The second call site of findings 17 and 24. `run_sweep` copied the
    timing's fields into a `Sample` one by one, and `host_bound` was on
    `CellTiming` and on no row. `sample_from_timing` copies every field the
    two share BY NAME, and a `CellTiming` field with no column fails here."""
    timing_fields = set(PW.CellTiming.__dataclass_fields__) - {"note"}
    assert timing_fields <= set(PW.Sample.__dataclass_fields__), (
        timing_fields - set(PW.Sample.__dataclass_fields__))
    ct = PW.CellTiming(
        ms_p50=1.25, ms_min=1.2, ms_stdev=0.01, iters=95, trials=2,
        warmup_ms=300.0, instrument="an instrument", sm_clock_load_mhz=1950.0,
        clock_level_ok=False, clock_level_side="high", clock_drift_ok=True,
        l2_flush=True, duty=0.25, power_w=301.0, host_bound=True,
        note="a clock note", clock_samples_mhz="1950 1965",
        duty_achieved=0.24, calls_per_burst=33, gap_ms=121.0, head_ms=1.2,
        tail_ms=1.3, within_burst_ok=False)
    s = PW.sample_from_timing(ct, arm=PW.PRIVATE, repeat=4, block_m=32,
                              tiles=3, rows_per_expert=96, tokens=384,
                              copies=3, experts_declared=72)
    for name in timing_fields:
        assert getattr(s, name) == getattr(ct, name), name
    assert s.detail == "a clock note" and s.status == "ok"
    assert (s.arm, s.repeat, s.tiles) == (PW.PRIVATE, 4, 3)


class _BurstEvents:
    """The event seam `clock_elasticity.time_duty` injects off GPU: the lead
    call of a burst costs 8 ms, the kept ones ramp from 2 ms by 1 us per
    call, so the burst's last quarter runs slower than its first and inside
    `timing.DRIFT_FRACTION` of it."""
    def __init__(self, n):
        rec = types.SimpleNamespace(record=lambda: None)
        self.starts, self.ends = [rec] * n, [rec] * n

    def synchronize(self):
        pass

    def elapsed(self, n):
        return [8.0] + [2.0 + 0.001 * i for i in range(n - 1)]


def test_time_cell_over_the_real_duty_timer_fills_every_diagnostic(tmp_path):
    """No laptop test drove `time_cell` through the REAL `time_duty`, so the
    names it reads off `DutyTiming` were checked only against a fake written
    beside them. Here the arm's own timer runs behind injected events, clock
    reads and sleep, and every diagnostic lands on the row and comes back
    from cells.csv; the row's instrument is the string the page's provenance
    starts with."""
    import functools

    import clock_elasticity as CE
    reads = iter([1965, 1950, 1965, 1980] * 10)

    def clock_read():
        return TIMING.ClockState(next(reads), 60, source=TIMING.CLOCK_SOURCE_NVML,
                                 power_w=300.0)
    duty_timer = functools.partial(CE.time_duty, events=_BurstEvents,
                                   clock_read=clock_read, sleep=lambda s: None)
    ct = PW.time_cell(lambda: None, duty=0.25, warmup_ms=0.0,
                      cell_budget_ms=200.0, trials=2, l2_flush=False,
                      reference_clock_mhz=None,
                      timer=lambda fn, **kw: _CellTiming(ms_p50=2.0),
                      duty_timer=duty_timer)
    calls = round(PW.DUTY_BURST_MS / 2.0)
    bursts = round(200.0 / PW.DUTY_BURST_MS)
    assert ct.calls_per_burst == calls and ct.iters == bursts * (calls - 1)
    assert ct.gap_ms == pytest.approx(calls * 2.0 * (1 / 0.25 - 1))
    assert ct.head_ms < ct.tail_ms and ct.within_burst_ok is True
    assert ct.duty_achieved is not None and ct.power_w == 300.0
    assert ct.host_bound is not None
    assert ct.clock_samples_mhz.split() == ["1965", "1950", "1965", "1980"] * 2 + ["1965", "1950"]
    assert ct.instrument == CE.INSTRUMENT
    assert PW.ladder_instrument(0.25).startswith(ct.instrument)
    row = PW.sample_from_timing(ct, arm=PW.SHARED, repeat=0, block_m=32,
                                tiles=1, rows_per_expert=32, tokens=128,
                                copies=1, experts_declared=72)
    path = tmp_path / "cells.csv"
    PW.Store(path, PW.CSV_FIELDS).append(row)
    back = PW.read_samples(path)[0]
    assert back.burst_sag == pytest.approx(row.burst_sag) and back.burst_sag > 0
    for name in DIAGNOSTIC_COLUMNS:
        got, want = getattr(back, name), getattr(row, name)
        assert got == (pytest.approx(want) if isinstance(want, float) else want), name


def _powered_pair(*, duty=0.25, shared_w=290.0, private_w=320.0,
                  shared_sag=0.0, private_sag=-0.01):
    """`_pair_world` at matched clocks with each arm's power and in-burst
    head and tail planted, so V7 PASSES on the clocks whatever is printed."""
    out = []
    for s in _pair_world(private_clock=1965.0, shared_clock=1965.0):
        watts, sag = {PW.SHARED: (shared_w, shared_sag),
                      PW.PRIVATE: (private_w, private_sag)}.get(s.arm, (None, None))
        out.append(PW.replace(s, duty=duty, power_w=watts,
                              head_ms=None if sag is None else 1.0,
                              tail_ms=None if sag is None else 1.0 + sag))
    return out


def test_v7_prints_each_arms_power_and_sag_beside_its_clocks_and_scores_neither():
    """Findings 2 and 24: power_w was written per cell and never shown, so a
    V7 FAIL could not say whether the arms split (one draws more) or both
    jittered; and the in-burst sag R1 gates as V5 had no view here at all.
    Both are printed per tread now, labelled as what they are, and V7's
    verdict does not move with either."""
    treads = [1, 2, 3]
    gate = PW.gate_v7_clock_parity(_powered_pair(), treads=treads)
    assert gate.verdict == exit_codes.PASS
    joined = "\n".join(gate.lines)
    assert ("power (NVML's ~1 s average, duty-averaged over bursts and idle "
            "gaps): shared 290 W, private 320 W") in joined, joined
    # A sag of exactly zero is a reading, not a missing one.
    assert "in-burst sag (tail - head) / head: shared +0.00%, private -1.00%" in joined
    assert "RECORDS: V7 scores the clocks alone" in joined
    assert joined.count("power (NVML") == len(treads)
    # Wildly different power and sag, same clocks: the verdict is the clocks'.
    loud = PW.gate_v7_clock_parity(
        _powered_pair(private_w=600.0, private_sag=-0.2), treads=treads)
    assert loud.verdict == exit_codes.PASS
    # At full duty the power is NVML's average of a busy card, not averaged
    # with gaps, and a queue-deep cell has no sag to print.
    full = [PW.replace(s, head_ms=None, tail_ms=None)
            for s in _powered_pair(duty=1.0)]
    lines = "\n".join(PW.gate_v7_clock_parity(full, treads=treads).lines)
    assert "power (NVML's ~1 s average): shared 290 W, private 320 W" in lines
    assert "duty-averaged" not in lines and "in-burst sag" not in lines
    # A page with neither prints what it always printed.
    bare = PW.gate_v7_clock_parity(_pair_world(), treads=treads)
    assert not any("power" in ln or "sag" in ln for ln in bare.lines)


# --------------------------------------------------------------------------
# 23. the duty premise and the elasticity, described as the evidence has them
# --------------------------------------------------------------------------

def _comment_above(name: str) -> str:
    """The `#:` block directly above `name = ...` in the script, joined."""
    lines = SCRIPT.read_text().splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.startswith(f"{name} = "))
    block = []
    for ln in reversed(lines[:at]):
        if not ln.startswith("#:"):
            break
        block.append(ln[2:].strip())
    return " ".join(reversed(block))


def _helps() -> dict[str, str]:
    return {flag: " ".join((a.help or "").split())
            for a in PW.build_parser()._actions for flag in a.option_strings}


def test_no_description_promises_v7_by_construction_below_full_duty():
    """THE RECURRING DEFECT, one premise described in eight places (findings
    0, 2 and 29). Design decision 15 said every state at or below duty 0.5
    sat at the boost ceiling and V7 held there by construction; session 4's
    own clock arm says the clock still tracked board power at 0.5 and sat
    flat at 0.25. Every description of the premise now states that evidence,
    and "by construction" stays only where it holds: at full duty."""
    doc = " ".join(PW.__doc__.split())
    assert "V7 then holds by construction" not in doc
    assert "boost ceiling" not in doc
    assert f"SO THE POD SETTING IS `--duty {PW.FLAT_DUTY}`" in doc
    assert f"--duty {PW.FLAT_DUTY}    # the pod setting (V7 checks it)" in PW.__doc__
    assert "both arms off the power cap" not in PW.__doc__, "a fact about neither arm"
    assert "9f91fa91" in doc and "2026-09-21" in doc
    assert "V7 fails by construction there" in doc and "At FULL duty" in doc
    # The default stays 1.0 (the owner's decision D1), and says why.
    assert "THE DEFAULT STAYS 1.0" in doc
    assert PW.DESIGN_KEY_DEFAULTS["duty"] == 1.0
    assert PW.build_parser().parse_args([]).duty == 1.0
    duty_help = _helps()["--duty"]
    assert "by construction" not in duty_help and "boost ceiling" not in duty_help
    assert f"The pod setting is {PW.FLAT_DUTY}" in duty_help
    assert "stays the default" in duty_help
    parity = _comment_above("CLOCK_PARITY")
    assert "at FULL duty on a card that cannot lock its clock" in parity
    v7 = " ".join(PW.gate_v7_clock_parity.__doc__.split())
    assert "`clock_elasticity.time_duty`'s one NVML read per burst" in v7
    assert "AT FULL DUTY" in v7
    v4 = " ".join(PW.gate_v4_memory_bound.__doc__.split())
    assert "every cell here run under one board power cap" not in v4
    assert "Below full duty" in v4 and "conservative for this gate" in v4
    # Finding 18: the power description matches what both instruments do.
    assert "the full-duty timer does not" not in SCRIPT.read_text()


def _two_elasticity_world(clock, *, intercept_eta, tile_eta, a=0.3,
                          b_shared=0.10, b_private=0.20, f0=1965.0):
    """Ratio-arm cells under `ms = a (f0/f) ** intercept_eta + b n (f0/f) **
    tile_eta`, the additive law with a different elasticity on each term, at
    the clock `clock(arm, n)` gives. The planted ratio is b_shared / b_private
    at any one clock. Every number here is PLANTED; none is a card's."""
    out = []
    for arm, b in ((PW.SHARED, b_shared), (PW.PRIVATE, b_private)):
        for n in range(1, 7):
            f = clock(arm, n)
            ms = a * (f0 / f) ** intercept_eta + b * n * (f0 / f) ** tile_eta
            out.extend(_sample(arm, n, rep, ms, load=f) for rep in range(3))
    return out


def _corrected_ratio(samples, eta):
    cells = PW.clock_corrected(samples, eta, 1500.0)
    return (PW.ladder_for(cells, PW.SHARED).slope_ms
            / PW.ladder_for(cells, PW.PRIVATE).slope_ms)


def test_one_clock_per_arm_is_carried_exactly_by_the_per_m_tile_elasticity():
    """WHICH eta `clock_corrected` needs, as arithmetic on planted cells. The
    ratio reads only slopes, so with each arm at one clock the per-call
    factor reaches it through the slope alone: the per-M-tile elasticity
    recovers the planted ratio exactly, and a per-call blend of the
    intercept's elasticity with the tiles' does not, whatever its weights.
    With a clock that moves across treads inside an arm, no single eta is
    exact, which is what the descriptions say."""
    intercept_eta, tile_eta = 0.2, 1.1
    planted = 0.10 / 0.20
    one = _two_elasticity_world(
        lambda arm, n: 1740.0 if arm == PW.SHARED else 1425.0,
        intercept_eta=intercept_eta, tile_eta=tile_eta)
    assert _corrected_ratio(one, tile_eta) == pytest.approx(planted, abs=1e-12)
    for blend in (0.25, 0.5, 0.75):
        per_call = intercept_eta + blend * (tile_eta - intercept_eta)
        assert abs(_corrected_ratio(one, per_call) - planted) > 1e-3, per_call
    moving = _two_elasticity_world(
        lambda arm, n: 1740.0 - 50.0 * (n - 1) if arm == PW.SHARED else 1425.0,
        intercept_eta=intercept_eta, tile_eta=tile_eta)
    assert abs(_corrected_ratio(moving, tile_eta) - planted) > 1e-3
    # One elasticity on both terms is the case any consistent eta carries,
    # which is the clock-split-elastic world's law.
    shared_law = _two_elasticity_world(
        lambda arm, n: 1740.0 - 50.0 * (n - 1) if arm == PW.SHARED else 1425.0,
        intercept_eta=0.75, tile_eta=0.75)
    assert _corrected_ratio(shared_law, 0.75) == pytest.approx(planted, abs=1e-12)


def test_the_eta_1_line_is_a_reference_point_the_correction_can_pass():
    """DD11's figure is first-order: the per-M-tile elasticity the correction
    takes read above 1 on session 4's cells, and at such an eta the corrected
    ratio lies past the eta = 1 one. The page used to call that line a bound
    the correction cannot exceed."""
    one = _two_elasticity_world(
        lambda arm, n: 1740.0 if arm == PW.SHARED else 1425.0,
        intercept_eta=0.2, tile_eta=1.1)
    cc = PW.clock_corrected_ratio(one, PW.ClockElasticity(1.1, 1.1, 1.1, "x"),
                                  f_ref=1500.0, f_ref_source="planted",
                                  draws=50, seed=0)
    assert (cc.ratio - cc.raw) * (cc.ratio - cc.at_unit) > 0
    assert abs(cc.ratio - cc.raw) > abs(cc.at_unit - cc.raw)
    joined = "\n".join(cc.lines())
    assert "cannot exceed" not in joined
    assert "not a bound: an eta above 1 carries the correction past it" in joined
    doc = " ".join(PW.ClockCorrection.__doc__.split())
    assert "cannot exceed" not in doc and "NOT a bound" in doc


def test_the_elasticity_the_correction_takes_is_the_per_m_tile_claim_everywhere():
    """Finding 19 and the review's R3-ETA-PER-CALL. `clock_corrected` scales
    each cell's whole per-call time, and a first reading of that took it to
    want the per-call elasticity; the ratio reads only slopes, so it wants
    the per-M-tile one (the test above plants both and shows it), which is
    also what clock_elasticity.py says its claim is for. Every description of
    the flag names that quantity, where it sits in either vintage of report,
    and that session 4's `value` is the per-call reading instead. The
    numbers quoted beside it carry their run and date."""
    for doc in (PW.ClockElasticity.__doc__, PW.clock_corrected.__doc__):
        joined = " ".join(doc.split())
        assert "PER-M-TILE" in joined and "PER-CALL ELASTICITY" not in joined, doc
    cls = " ".join(PW.ClockElasticity.__doc__.split())
    assert "elasticity.value" in cls and "elasticity.fixed_tread" in cls
    assert "8d4eb78" in cls and "session 4's included" in cls
    helps = _helps()
    assert "per-M-TILE" in helps["--clock-elasticity"]
    assert "gated claim (elasticity.value" in helps["--clock-elasticity"]
    assert "NOT its pooled per-call" in helps["--clock-elasticity"]
    assert "e.g. elasticity.value" in helps["--clock-elasticity-source"]
    parity = _comment_above("CLOCK_PARITY")
    assert "WHICH ELASTICITY `--clock-elasticity` TAKES: THE PER-M-TILE ONE" in parity
    assert "NOT its pooled per-call reading" in parity
    assert "no single eta is exact" in parity
    assert "bounded above by 1" not in parity
    assert "9f91fa91" in parity and "2026-09-21" in parity and "12ec932" in parity
    doc = " ".join(PW.__doc__.split())
    assert "measured PER-M-TILE elasticity" in doc
    assert "PER-CALL elasticity" not in doc


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
    series (one per arm) x 6 treads x repeats x (PROBE_CAPTURE_MS +
    PROBE_WARMUP_MS + PROBE_TRIALS x PROBE_TARGET_MS)` = `3 x 6 x repeats x
    160 ms`, which is 5.76 s at two repeats and 8.64 s at three (5.04 and 7.56
    before the capture was booked on 2026-09-22).

    THE TWO FIGURES ARE ASSERTED AS FIGURES, not re-derived from the same
    constants the function multiplies, because a test that recomputes
    `probe_seconds`' one line and compares cannot notice the budget constants
    moving underneath the prose above. If PROBE_CAPTURE_MS, PROBE_WARMUP_MS,
    PROBE_TRIALS or PROBE_TARGET_MS changes, 5.76 and 8.64 are what has to be
    re-derived and this docstring is what has to be rewritten.
    """
    assert PW.MIN_PROBE_REPEATS == 2
    assert PW.MIN_PROBE_REPEATS < PW.PROBE_REPEATS

    treads = PW.ladder_treads(CFG, PW.DEFAULT_BLOCK_M, PW.DEFAULT_TREADS)
    assert len(treads) == 6, treads
    at_two = PW.probe_seconds(treads, PW.PROBE_LABELS, PW.MIN_PROBE_REPEATS)
    at_default = PW.probe_seconds(treads, PW.PROBE_LABELS, PW.PROBE_REPEATS)
    assert at_two == pytest.approx(5.76), at_two
    assert at_default == pytest.approx(8.64), at_default
    assert PW.PROBE_CAPTURE_MS > 0
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
             draws: int = 10, copies: int = 9, replicates=(), run_id="",
             pinned=None, seed: int = 0, duty: float = 1.0):
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
        high_water_bytes=mem.predicted_peak_bytes, draws=draws, seed=seed,
        header=[], card="no card", synthetic=True,
        model_name=PW.DEFAULT_MODEL, pinned=(pinned or {}), copies_declared=copies,
        run_id=run_id, replicates=tuple(replicates), duty=duty)


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
        census=skipped.census, copies_declared=skipped.copies_declared,
        duty=skipped.args.duty)


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
    # The counts are the WORLD's plant, named here rather than read back out
    # of `probe.host_bound`, which is the call `gate_v8_alignment` itself
    # makes. BOTH ratio series are counted and both are named: PRIVATE's id
    # set is scored too, and the two need not be host-bound together.
    probed = len(skipped.treads) * PW.PROBE_REPEATS
    hot = 0
    for label in (PW.SHARED, PW.PRIVATE):
        assert skipped.probe.host_bound(label)[:2] == (hot, probed)
    assert any(f"called {hot} of {probed} of shared's, {hot} of {probed} of "
               "private's probed cells HOST-BOUND at the ratio arms' "
               "declaration" in ln
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


def test_v5_scores_a_pass_on_the_far_edge_and_a_fail_on_the_near_one():
    """MACHINERY_BOUND was 0.10, which admits a ratio error larger than
    ALPHA_BAND's entire 0.059 width; it is 0.03, and a declaration cost of 5%
    of the private slope FAILs where it once PASSED.

    THE TWO EDGES DECIDE DIFFERENT VERDICTS, and that is the half this test
    was rewritten for. A PASS is a claim about how far the ratio CAN sit from
    the study's own call, so it holds only at b's WORST edge. A FAIL is a
    claim that the declaration's cost IS over the bound, and the measurement
    says that only when b's NEAREST edge is over it too. In between -- a band
    straddling the bound -- the run resolved nothing about the declaration,
    which is UNKNOWN, and a VALIDITY UNKNOWN latches the page INVALID anyway.
    Scoring the FAIL on the far edge printed the registered harm ("the wider
    declaration changes the per-M-tile cost itself") off a measurement whose
    point sat at 1% of a 3% bound.

    Every other gate in this file is built the same way: V8 FAILs on the
    bound over the RESOLVED steps and answers UNKNOWN when only the wide one
    is over; C1 FAILs only when the whole interval misses ALPHA_BAND.
    """
    assert PW.MACHINERY_BOUND == 0.03
    assert PW.MACHINERY_BOUND < ALPHA_BAND_WIDTH()
    native, shared, private = _v5_ladders(0.0)
    # The whole band over the bound: the cost IS there, and V5 refuses.
    fit = PW.DeclarationFit(None, None, 0.05 * private.slope_ms, 0.0, (), 4)
    band = (0.045 * private.slope_ms, 0.055 * private.slope_ms)
    assert PW.gate_v5_machinery(native, shared, private, fit, band).verdict \
        == exit_codes.FAIL
    # A point at 1% whose band reaches 4%: the far edge is over, the near
    # edge is not. Pre-change this read FAIL.
    fit = PW.DeclarationFit(None, None, 0.01 * private.slope_ms, 0.0, (), 4)
    wide = (-0.04 * private.slope_ms, 0.02 * private.slope_ms)
    gate = PW.gate_v5_machinery(native, shared, private, fit, wide)
    assert gate.verdict == exit_codes.UNKNOWN, gate.lines
    assert any("band STRADDLES the bound" in ln for ln in gate.lines), gate.lines
    assert gate.measured.startswith(
        "4.00% of the private slope at b's far edge, 2.00% at its near edge")
    # And a band wholly inside the bound still PASSES.
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
    """PRIVATE's ids are probed at the shared declaration: ~2.9 s more on
    the pod (2.5 before the capture was booked), and the only measurement of
    the counter asymmetry the plan page registers."""
    assert PW.PROBE_LABELS == len(PW.ARMS) == 3
    treads = [1, 2, 3, 4, 5, 6]
    assert PW.probe_seconds(treads, 1) == pytest.approx(2.88)
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


# --------------------------------------------------------------------------
# 24. --probe-check: V8's on-card probe check alone, for the vLLM venv
# --------------------------------------------------------------------------

def _stand_in_probe_check(monkeypatch, *, eager_ms=0.030, replay_ms=None,
                          graph_host_bound=False, graph_note="",
                          refuse_capture=False):
    """Run the REAL `probe_check` under `--probe-check` with its four seams
    filled by fakes: the op, the sync, and the two timers `time_probe_cell`
    already takes. The host check that refuses a laptop is stood in, so the
    mode's own wiring (argparse, the gate it prints, the exit it returns) is
    what runs. Hands back the op's calls."""
    replay = (PW.PROBE_CALLS_PER_REPLAY * eager_ms / 8 if replay_ms is None
              else replay_ms)
    ops: list = []

    def op(ids, block_m, declared):
        ops.append((int(ids.numel()), block_m, declared))

    def graph_timer(fn, **kw):
        if refuse_capture:
            raise TIMING.NotCapturable("operation not permitted when stream is "
                                       "capturing")
        fn()
        return _Timing(replay, host_bound=graph_host_bound, host_note=graph_note)

    def eager_timer(fn, **kw):
        fn()
        return _Timing(eager_ms, host_bound=True, host_note="eager, host-timed")

    real = PW.probe_check

    def check(cfg, **kw):
        return real(cfg, op=op, sync=lambda: None, graph_timer=graph_timer,
                    eager_timer=eager_timer, device="cpu", **kw)
    monkeypatch.setattr(PW, "probe_check_refusal", lambda: "")
    monkeypatch.setattr(PW, "probe_check", check)
    return ops


def _probe_check_log(argv=("--probe-check",)) -> tuple[int, str]:
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rc = PW.main(list(argv))
    return rc, log.getvalue()


#: (world, the fakes' settings, the verdict the one RESULT line must carry).
PROBE_CHECK_WORLDS: tuple[tuple[str, dict, str], ...] = (
    ("gpu-time", {}, exit_codes.PASS),
    ("host-bound-replay", {"graph_host_bound": True,
                           "graph_note": "the queue drained while the host "
                                         "was still enqueueing"},
     exit_codes.FAIL),
    ("not-under-eager", {"replay_ms": PW.PROBE_CALLS_PER_REPLAY * 0.031},
     exit_codes.FAIL),
    ("capture-refused", {"refuse_capture": True}, exit_codes.FAIL),
    ("no-host-bound-verdict", {"graph_host_bound": None}, exit_codes.UNKNOWN),
)


@pytest.mark.parametrize("world,fakes,verdict", PROBE_CHECK_WORLDS,
                         ids=[w for w, _f, _v in PROBE_CHECK_WORLDS])
def test_probe_check_prints_one_result_line_its_exit_code_is_recomputed_from(
        world, fakes, verdict, monkeypatch, tmp_path):
    """THE NEW INTERFACE THE CHAIN CALLS. One RESULT line, `P1`, VALIDITY, in
    the format `Gate.result_line` renders every gate on the page in, and an
    exit through `exit_codes`: 0 on PASS, 3 on FAIL or UNKNOWN, which is what
    `classify_text` recomputes from the log. The fakes plant each way the
    on-card test could fail: a replay the instrument calls host-bound, a
    per-call graph time not under the eager p50, a refused capture, and no
    host-bound verdict at all (UNKNOWN, not PASS)."""
    ops = _stand_in_probe_check(monkeypatch, **fakes)
    root = tmp_path / "results-root"
    root.mkdir()
    monkeypatch.setenv("MOE_RESULTS_DIR", str(root))
    rc, out = _probe_check_log()
    lines = exit_codes.parse_result_lines(out)
    assert len(lines) == 1, out
    assert [ln for ln in out.splitlines() if ln.startswith("RESULT: ")] == \
        [lines[0].render()]
    (line,) = lines
    assert (line.kind, line.name, line.verdict) == (exit_codes.VALIDITY, "P1", verdict)
    assert line.detail.startswith("[VALIDITY] ")
    assert " | measured " in line.detail and " | gate " in line.detail
    assert rc == exit_codes.classify_text(out)
    assert rc == (exit_codes.DONE if verdict == exit_codes.PASS
                  else exit_codes.INVALID)
    # What it times: tread 1 (r = BLOCK_M) of the default model at NATIVE's
    # declaration, the cell the on-card test times.
    tokens = PW.SWEEP.tokens_for_rows(CFG, PW.DEFAULT_BLOCK_M)
    assert set(ops) == {(tokens * CFG.top_k, PW.DEFAULT_BLOCK_M, CFG.num_experts)}
    if not fakes.get("refuse_capture"):
        assert f"{PW.PROBE_CALLS_PER_REPLAY} calls per replay" in line.detail
    # It measures nothing else and writes no results directory.
    assert list(root.rglob("*")) == []
    assert "private_weight_reference/" not in out


def test_probe_check_refuses_with_no_cuda_before_it_measures_anything(
        no_cuda, monkeypatch, tmp_path):
    """No card: REFUSED (2) with a line that says why, before `probe_check`
    (and with it vLLM's op and the timers) is reached. The return code alone
    is argparse's too, so the sentence is asserted."""
    def never(*a, **k):
        raise AssertionError("probe_check ran on a host with no card")
    monkeypatch.setattr(PW, "probe_check", never)
    root = tmp_path / "results-root"
    root.mkdir()
    monkeypatch.setenv("MOE_RESULTS_DIR", str(root))
    rc, out = _probe_check_log()
    assert rc == exit_codes.REFUSED
    refused = [ln for ln in out.splitlines() if ln.startswith("REFUSED: ")]
    assert len(refused) == 1 and "no CUDA device" in refused[0], out
    assert exit_codes.parse_result_lines(out) == []
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(out)
    assert list(root.rglob("*")) == []
    # And as the chain runs it: a child process, the laptop world laid over it.
    got = run(["--probe-check", "--model", PW.DEFAULT_MODEL, "--block-m",
               str(PW.DEFAULT_BLOCK_M)])
    assert got.returncode == exit_codes.REFUSED, got.stderr[-800:]
    assert "REFUSED: " in got.stdout and "no CUDA device" in got.stdout
    assert "unrecognized arguments" not in got.stderr


def test_probe_check_refuses_without_vllm_even_on_a_card(monkeypatch):
    """The base venv on the pod: a card and no vLLM. Refused naming vLLM and
    the venv to run it from, and vLLM is LOOKED FOR, not imported."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm", None)

    def never(*a, **k):
        raise AssertionError("probe_check ran with no vLLM")
    monkeypatch.setattr(PW, "probe_check", never)
    rc, out = _probe_check_log()
    assert rc == exit_codes.REFUSED
    refused = [ln for ln in out.splitlines() if ln.startswith("REFUSED: ")]
    assert len(refused) == 1 and "vLLM" in refused[0] and "venv" in refused[0], out
    assert exit_codes.parse_result_lines(out) == []


def test_probe_check_refuses_a_vllm_whose_op_does_not_import(monkeypatch):
    """vLLM found, its op not where `probe_check` imports it from: an import
    that drifted, REFUSED (2) with the import error named, not a crash (4)."""
    monkeypatch.setattr(PW, "probe_check_refusal", lambda: "")
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.fused_moe.moe_align_block_size",
        None)
    rc, out = _probe_check_log()
    assert rc == exit_codes.REFUSED, out
    refused = [ln for ln in out.splitlines() if ln.startswith("REFUSED: ")]
    assert len(refused) == 1 and "did not import" in refused[0], out
    assert exit_codes.parse_result_lines(out) == []


def test_probe_check_is_a_mode_of_its_own():
    """Beside another mode it would print one of the two and drop the other
    without a word, so the pair is refused."""
    for extra in (["--dry-run"], ["--self-test", "refit"],
                  ["--read", "x.json", "--replicate-of", "y.json"]):
        rc, out = _probe_check_log(["--probe-check", *extra])
        assert rc == exit_codes.REFUSED, extra
        assert "REFUSED: --probe-check is a mode of its own" in out, out
        assert exit_codes.parse_result_lines(out) == []


def test_probe_check_says_what_it_is_for_and_the_on_card_test_calls_it():
    """The --help says why the chain runs it (the vLLM venv has no pytest, and
    the base venv can only skip the on-card test), and the on-card test calls
    the ONE function the mode calls rather than a copy of its logic."""
    got = _helps()["--probe-check"]
    for phrase in ("scripts/alpha_g_chain.sh", "vLLM venv", "no pytest",
                   "test_the_alignment_probe_is_gpu_time_under_the_graph",
                   "can only skip", "writes nothing"):
        assert phrase in got, (phrase, got)
    assert "--probe-check" in PW.__doc__.split("WHY THIS ARM EXISTS")[0]
    tree = ast.parse((ROOT / "tests" / "test_gpu.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "test_the_alignment_probe_is_gpu_time_under_the_graph")
    called = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "PW.probe_check" in called, called
    assert "pytest.importorskip" in called
    assert "PW.time_probe_cell" not in called, "the logic lives in probe_check"


# --------------------------------------------------------------------------
# 25. report.json's duty is the one REQUESTED, whatever was timed
# --------------------------------------------------------------------------

def _measure_through_main(monkeypatch, tmp_path, sweep, duty="0.25"):
    """Drive the real `_main` of a MEASURING run at `--duty duty` to the
    report.json it writes, with `run_sweep` replaced by `sweep` and the two
    host checks that refuse a laptop stood in. Everything between the argv
    and the file (the run id, the plan, `analyse`, the writer) is shipped
    code. Returns `(exit code, the payload on disk, the log)`."""
    monkeypatch.setattr(PW.SWEEP, "missing_gpu_stack", lambda: "")
    monkeypatch.setattr(PW, "clock_sampler_refusal", lambda: "")
    monkeypatch.setattr(PW, "run_sweep", sweep)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rc = PW._main(["--duty", duty, "--ridge", "160", "--bandwidth-gbps",
                       "4000", "--device-memory-gb", "140", "--capability",
                       "9.0", "--out", str(tmp_path)])
    written = list(tmp_path.rglob("report.json"))
    assert len(written) == 1, log.getvalue()[-2000:]
    return rc, json.loads(written[0].read_text()), log.getvalue()


def test_a_v8_fail_page_records_the_requested_duty_and_null_for_the_timed_one(
        no_cuda, monkeypatch, tmp_path):
    """E2E-1. The REAL `run_sweep` returns no samples when the probe reads V8
    FAIL, and `duty` was `duty_of(samples)`, whose default is 1.0: a run
    requested at 0.25, keyed at 0.25 in its run id and naming the duty timer
    as its instrument recorded full duty, the session-4 regime, and the
    chain printed and tabulated it. `duty` is now the requested duty and
    `duty_timed` is null, because nothing was timed."""
    world = PW.WORLDS["ratio-path-split"]
    real_sweep = PW.run_sweep

    def planted(cfg, *, block_m, treads, declared_by_arm, **kw):
        census = PW.path_census(cfg, treads, block_m, declared_by_arm)
        return PW.planted_probe(world, cfg, block_m=block_m, treads=treads,
                                declared_by_arm=declared_by_arm, census=census,
                                noise=0.0, seed=0,
                                weight_stream_ms=WEIGHTS.weight_stream_ms(
                                    cfg, "bf16", 4000.0))

    def never(*a, **k):
        raise AssertionError("build_private_weights ran past the early return")

    _stand_in_for_vllm(monkeypatch)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "unused-cache"))
    monkeypatch.setattr(PW, "probe_alignment", planted)
    monkeypatch.setattr(PW, "build_private_weights", never)
    rc, payload, log = _measure_through_main(monkeypatch, tmp_path, real_sweep)
    assert "SWEEP SKIPPED" in log
    assert rc == exit_codes.INVALID
    assert payload["duty"] == 0.25
    assert payload["duty_timed"] is None
    assert "duty0.25" in payload["run_id"]
    assert payload["instrument"].endswith("duty 0.25")
    v8 = next(g for g in payload["gates"] if g["tag"] == "V8")
    assert v8["verdict"] == exit_codes.FAIL


def test_a_page_whose_every_cell_failed_records_the_requested_duty(
        no_cuda, monkeypatch, tmp_path):
    """The same default, the other empty path: every cell raised, so no row
    is `ok` and `duty_of` had nothing to read. The failed rows carry the
    requested duty (their `duty` column is the REQUESTED one), and the page
    is INVALID on V0 with `duty` 0.25 and `duty_timed` null."""
    treads = list(range(1, PW.DEFAULT_TREADS + 1))

    def sweep(args, cfg, *, block_m, treads, census, stream_ms,
              copies_declared, **kw):
        declared = {a: PW.declared_experts(a, cfg.num_experts, copies_declared)
                    for a in PW.ARMS}
        failed = [PW.Sample(arm=arm, repeat=rep, block_m=block_m, tiles=n,
                            rows_per_expert=n * block_m,
                            tokens=PW.SWEEP.tokens_for_rows(cfg, n * block_m),
                            copies=PW.copies_read(arm, n),
                            experts_declared=declared[arm], ms_p50=0.0,
                            status="failed", detail="RuntimeError: planted",
                            duty=args.duty)
                  for rep in range(args.repeats) for n in treads
                  for arm in PW.ARMS]
        probe = PW.planted_probe(PW.WORLDS["refit"], cfg, block_m=block_m,
                                 treads=treads, declared_by_arm=declared,
                                 census=census, noise=0.0, seed=0,
                                 weight_stream_ms=stream_ms)
        return failed, PW.planted_proof(False), None, None, probe

    rc, payload, _log = _measure_through_main(monkeypatch, tmp_path, sweep)
    assert rc == exit_codes.INVALID
    v0 = next(g for g in payload["gates"] if g["tag"] == "V0")
    assert v0["verdict"] == exit_codes.FAIL
    assert payload["duty"] == 0.25
    assert payload["duty_timed"] is None
    assert len(treads) == PW.DEFAULT_TREADS


def test_duty_timed_is_what_the_timed_rows_carry_and_duty_what_was_asked():
    """With cells timed, `duty_timed` is `duty_of` over them; the requested
    `duty` is recorded beside it and never read off the rows."""
    treads = [1, 2, 3, 4, 5, 6]
    kw = dict(block_m=32, treads=treads, repeats=3, ridge=160.0,
              bandwidth_gbps=4000.0, b=2, noise=0.0, seed=0, copies_declared=9,
              native_switch=4)
    samples = [PW.replace(s, duty=0.25) for s in PW.planted_samples(
        PW.WORLDS["refit"], CFG, alpha_shared=PW.ALPHA, **kw)]
    payload = _analyse(samples, treads, duty=0.25).payload
    assert payload["duty"] == 0.25 and payload["duty_timed"] == PW.duty_of(samples)
    assert PW.duty_of([], default=None) is None and PW.duty_of([]) == 1.0


def test_a_report_written_before_duty_timed_still_loads_and_reads(tmp_path):
    """Old reports carry `duty` and no `duty_timed`: they load as replicates,
    and `--read` prints the requested duty and says the timed one was not
    recorded, rather than inventing one."""
    treads = [1, 2, 3, 4, 5, 6]
    pa, payload_a = _measured_shaped_report(tmp_path, "run-a", PW.ALPHA, treads)
    pb, payload_b = _measured_shaped_report(tmp_path, "run-b", PW.ALPHA, treads,
                                            seed=1)
    assert "duty_timed" in payload_a
    for p, payload in ((pa, payload_a), (pb, payload_b)):
        old = dict(payload)
        del old["duty_timed"]
        p.write_text(json.dumps(old))
    design = {k: payload_a.get(k, PW.DESIGN_KEY_DEFAULTS.get(k))
              for k in PW.DESIGN_KEYS}
    assert len(PW.load_replicates([pb], design=design, card_known=True)) == 1
    got = run(["--read", str(pa), "--replicate-of", str(pb)])
    assert "READ MODE: nothing measured, nothing written" in got.stdout, \
        got.stdout[-800:]
    assert f"duty        {payload_a['duty']}" in got.stdout
    assert "duty timed  unrecorded (a report written before duty_timed)" \
        in got.stdout
    assert exit_codes.classify_text(got.stdout) == got.returncode


def test_a_cell_that_failed_in_the_sweep_carries_the_duty_it_was_asked_at():
    """The second call site of E2E-1. `run_sweep` builds a failed row with
    `Sample(...)` directly, and `duty` defaulted to 1.0 there while every
    timed row carried the requested duty through `time_cell`: a failed cell
    at 0.25 was written to cells.csv as a full-duty row. Every `Sample(` the
    sweep builds now names its duty."""
    tree = ast.parse(SCRIPT.read_text())
    sweep = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "run_sweep")
    built = [n for n in ast.walk(sweep) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "Sample"]
    assert built, "run_sweep builds no Sample directly any more; retarget"
    for call in built:
        kws = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kws.get("duty") == "args.duty", ast.unparse(call)


# --------------------------------------------------------------------------
# 26. a V7 FAIL at the pod duty: one reading, in every place that says it
# --------------------------------------------------------------------------

#: The owner's resolved reading of a V7 FAIL at `--duty 0.25` (finding XS-1),
#: in the words every description carries.
V7_AT_POD_DUTY = ("not yet flat for this arm on this card",
                  "does not re-run at it",
                  "at a G's seed 0",
                  "skips the G's later seeds and prints the follow-up command")


def test_a_v7_fail_at_the_pod_duty_reads_the_same_everywhere_this_file_says_it():
    """XS-1. The page said "a lower duty"; the driver and the runbook said "a
    finding about the card, not a setting to change". The owner's reading:
    that duty is not yet flat for this arm on this card, the page names a
    lower duty, and the chain does not act on it: it skips the G's later
    seeds and prints the hand command for a lower-duty follow-up, a new
    design key with runs of its own. Every place this file says what follows
    a FAIL at the pod duty now says that, and none says the chain re-runs."""
    remedy = PW.v7_remedy(PW.FLAT_DUTY)
    doc = " ".join(PW.__doc__.split())
    v7_doc = " ".join(PW.gate_v7_clock_parity.__doc__.split())
    remedy_doc = " ".join(PW.v7_remedy.__doc__.split())
    for where, text in (("v7_remedy(FLAT_DUTY)", remedy), ("module docstring", doc),
                        ("V7's docstring", v7_doc),
                        ("v7_remedy's docstring", remedy_doc),
                        ("the plan's clause", PW.V7_FAIL_AT_FLAT_DUTY)):
        for phrase in V7_AT_POD_DUTY:
            assert phrase in text, (where, phrase, text[:400])
    # The remedy still NAMES a lower duty, and says what a run there is.
    assert remedy.startswith(f"the remedy is a lower --duty than {PW.FLAT_DUTY:.2f}:")
    assert "new design key" in remedy and "--read" in remedy
    assert "PAIRS.tsv" in remedy
    # Above the pod setting the chain has nothing to say: it never runs there.
    assert "alpha_g_chain" not in PW.v7_remedy(0.5)
    assert "alpha_g_chain" not in PW.v7_remedy(1.0)
    src = " ".join(SCRIPT.read_text().split())
    assert "not a setting to change" not in src
    assert "finding about the card" not in src
