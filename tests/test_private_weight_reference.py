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
    (("--self-test", "clock-split"), True),
    (("--self-test", "alignment-step"), True),
    (("--self-test", "ratio-path-split"), True),
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
                + PW.probe_seconds(kw["treads"], 2))
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


def test_step_bias_is_the_leverage_over_the_private_slope():
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


def _probe_from(series_by_label, spread=0.0):
    cells = []
    for label, series in series_by_label.items():
        for rep in range(3):
            for n, numel, ms in series:
                jitter = (rep - 1) * spread
                cells.append(PW.ProbeCell(label, n, numel, 72, rep,
                                          ms * (1.0 + jitter)))
    return PW.AlignProbe(tuple(cells), synthetic=True)


def _census():
    return PW.path_census(CFG, [1, 2, 3, 4, 5, 6], 32,
                          {PW.NATIVE: 8, PW.SHARED: 72, PW.PRIVATE: 72})


def test_v8_passes_a_flat_ratio_series_fails_a_real_step_and_doubts_a_noisy_one():
    treads = [1, 2, 3, 4, 5, 6]
    flat = _probe_from({PW.NATIVE: _series(0.02, 4), PW.SHARED: _series(0.0, 4)})
    assert PW.gate_v8_alignment(flat, treads=treads, census=_census(),
                                private_slope_ms=0.64).verdict == exit_codes.PASS
    split = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.5, 4)}, spread=1e-4)
    gate = PW.gate_v8_alignment(split, treads=treads, census=_census(),
                                private_slope_ms=0.64)
    assert gate.verdict == exit_codes.FAIL, gate.lines
    # Over budget but the probe's own spread swallows it: not shown either way.
    noisy = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.5, 4)}, spread=5.0)
    assert PW.gate_v8_alignment(noisy, treads=treads, census=_census(),
                                private_slope_ms=0.64).verdict == exit_codes.UNKNOWN
    assert PW.gate_v8_alignment(None, treads=treads, census=_census(),
                                private_slope_ms=0.64).verdict == exit_codes.UNKNOWN


def test_v8_budget_is_a_bias_on_the_ratio_and_not_a_step_in_microseconds():
    """The same step passes at a deep private slope and fails at a shallow one:
    the gate is on what the step does to the RATIO."""
    treads = [1, 2, 3, 4, 5, 6]
    probe = _probe_from({PW.NATIVE: _series(0.02, 4),
                         PW.SHARED: _series(0.02, 4)}, spread=1e-4)
    assert PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                private_slope_ms=0.64).verdict == exit_codes.PASS
    assert PW.gate_v8_alignment(probe, treads=treads, census=_census(),
                                private_slope_ms=0.2).verdict == exit_codes.FAIL


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
    assert PW.gate_v5_machinery(native, shared, private, fit).verdict \
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
