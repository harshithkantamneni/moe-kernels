"""The BLOCK_K diagonal arm: does it separate residency from pipeline depth.

FIVE GROUPS, in the order a reader has to take them.

  - THE ARITHMETIC. The shared-memory bill, the iso-shared-memory pairings and
    the feasibility refusals, checked as RELATIONS against the repo's own
    tables rather than as literals. `SMEM_PER_BLOCK_BYTES` and the register
    file are calibration-shaped facts about a card, so this file asserts
    `smem(stages, BK) == stages x (BM BK + BK BN) x b` and that the grid fits
    under the table's own sm_90 entry -- never that some cell is 48 KiB.

  - THE DESIGN. Rank, and what makes it lose rank. The hazard this arm is
    exposed to is the register file: every limit but shared memory is
    independent of BLOCK_K, so a tight register bound flattens the whole
    residency ladder onto one rung and every coefficient still prints. Both
    sides of that are planted.

  - THE CLOCK RULE, planted five ways. DRIFT excludes, both LEVEL sides are
    kept with the side recorded, and the fifth row is HIGH-AND-DRIFT in one
    tread -- because every earlier world in this repo had its drifting row
    sitting LEVEL, so a counter filtering on LEVEL alone counted that tread
    twice and stayed green. The AST check that there is exactly ONE reader of
    those verdicts is here too: this project's recurring defect is a rule
    applied at one of N call sites.

  - THE EXIT CONTRACT. Every off-GPU mode, `classify_text(log) == rc` or a
    REFUSED with no RESULT line; one RESULT line per gate and nothing else
    with that shape; a crash is ERROR (4) and not CLAIM_FAIL.

  - THE FOUR PLANTED WORLDS, which are the claim that these gates
    DISCRIMINATE. A self test that asserts nothing is a smoke test, so each
    world registers a verdict per gate AND the exit code, and this file checks
    that the four worlds do not all land on the same page.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes, timing  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

SCRIPT = ROOT / "scripts" / "blockk_diagonal.py"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts"
                                                  / filename)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bk():
    return _load("blockk_diagonal", "blockk_diagonal.py")


@pytest.fixture(scope="module")
def limits(bk):
    """The H200's occupancy limits, from the sibling arm's own tables.

    NOT literals. `card_limits` reads `occupancy_vs_swizzle.SMEM_PER_SM_BYTES`
    and refuses a capability that is not in it, which is the behaviour this
    fixture also documents: a guessed per-SM shared memory moves every rung of
    the axis this arm contrasts.
    """
    return bk.OCC.card_limits((9, 0), bk.SWEEP.DEFAULT_SM_COUNT,
                              bk.OCC.HYPOTHESIS_L2_BYTES, "test fixture")


# --------------------------------------------------------------------------
# 1. the arithmetic
# --------------------------------------------------------------------------

def test_shared_memory_is_the_relation_and_not_a_number(bk):
    """`stages x (BM BK + BK BN) x b`, checked against the study's own bill.

    The number is NOT asserted. `SWEEP.tile_resources` is the one statement of
    this arithmetic in the repository -- it is the function whose refusal
    already stopped a BN=256 arm -- so what is pinned here is that this file
    reads it rather than recomputing it, at every cell of the default grid.
    """
    b = dtype_bytes("bf16")
    for cell in bk.parse_cells(bk.DEFAULT_CELLS):
        want = (cell.num_stages
                * (bk.SUBJECT_BLOCK_M * cell.block_k
                   + cell.block_k * bk.PINNED_BLOCK_N) * b)
        assert cell.smem_bytes(b) == want, cell.key
        assert cell.smem_bytes(b) == bk.SWEEP.tile_resources(
            cell.pinned(), bk.SUBJECT_BLOCK_M, b, (9, 0)).smem_bytes


def test_every_registered_pair_has_equal_shared_memory(bk):
    """The design's whole claim to hold residency fixed while moving depth."""
    b = dtype_bytes("bf16")
    by_key = {c.key: c for c in bk.parse_cells(bk.DEFAULT_CELLS)}
    assert len(bk.ISO_SMEM_PAIRS) == 2, bk.ISO_SMEM_PAIRS
    for low_key, high_key in bk.ISO_SMEM_PAIRS:
        low, high = by_key[low_key], by_key[high_key]
        assert low.smem_bytes(b) == high.smem_bytes(b), (low_key, high_key)
        assert low.num_stages != high.num_stages, "a pair that is not a contrast"
        # And the depths differ by a factor of two, which is the unit every
        # coefficient on the page is quoted in.
        ratio = max(low.num_stages, high.num_stages) / min(low.num_stages,
                                                           high.num_stages)
        assert ratio == 2.0, (low_key, high_key, ratio)


def test_the_iso_depth_ladder_holds_the_depth_and_moves_the_footprint(bk):
    b = dtype_bytes("bf16")
    by_key = {c.key: c for c in bk.parse_cells(bk.DEFAULT_CELLS)}
    cells = [by_key[k] for k in bk.ISO_DEPTH_LADDER]
    assert len({c.num_stages for c in cells}) == 1, "the depth moved"
    assert len({c.smem_bytes(b) for c in cells}) == len(cells)


def test_the_whole_default_grid_fits_the_cards_own_per_block_ceiling(bk):
    """Read off `SWEEP.SMEM_PER_BLOCK_BYTES`, never asserted as a figure.

    Both entries, because a grid that fits an H200 and not an A100 is a grid
    that silently becomes a different experiment on the other card.
    """
    b = dtype_bytes("bf16")
    for cap in ((9, 0), (8, 0)):
        ceiling = bk.SWEEP.SMEM_PER_BLOCK_BYTES[cap]
        for cell in bk.parse_cells(bk.DEFAULT_CELLS):
            assert cell.smem_bytes(b) <= ceiling, (cap, cell.key)
            assert not bk.SWEEP.tile_resources(
                cell.pinned(), bk.SUBJECT_BLOCK_M, b, cap).refusal, cell.key


def test_an_infeasible_cell_is_refused_at_plan_time_with_the_arithmetic(bk,
                                                                        capsys):
    """A cell that cannot compile is refused where it is chosen.

    `scripts/dtype_tile_confound.py` is the precedent and the reason: a setting
    that overflows shared memory does not produce a slow cell, it produces a
    launch failure halfway through a metered run, and one that overflows the
    register file produces a SPILLED kernel whose time still fits a straight
    line.
    """
    rc = bk.main(["--dry-run", "--capability", "9.0",
                  "--cells", "3x64,64x128"])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert "REFUSED: a cell cannot physically run as pinned" in out
    assert "64x128" in out and "shared memory" in out
    assert "227 KiB" in out or "232448" in out or "per-block ceiling" in out


def test_a_block_k_that_does_not_divide_k_is_refused_with_the_division(bk,
                                                                      capsys):
    """EVEN_K is a constexpr, so an uneven K is a DIFFERENT kernel.

    vLLM's fused_moe takes a masked load path when K is not a multiple of
    BLOCK_SIZE_K. Its time is not comparable with the even path's, and the
    difference would land in `bK`, the coefficient this arm registers a
    prediction of zero for.
    """
    cfg = MODEL_CONFIGS["deepseek-v2-lite"]
    # intermediate_size 1408 = 11 x 128, so 256 leaves a remainder.
    why = bk.even_k_refusal(cfg, bk.Cell(2, 256))
    assert "intermediate_size" in why and "EVEN_K=False" in why
    assert bk.even_k_refusal(cfg, bk.Cell(2, 64)) == ""
    # Through the CLI, on the model whose rows quantum lets a ladder form at
    # all: mixtral's intermediate_size is 14336 = 2^11 x 7, so BLOCK_K=4096
    # divides its hidden_size and not its intermediate_size.
    assert MODEL_CONFIGS["mixtral-8x7b"].intermediate_size % 4096 != 0
    rc = bk.main(["--dry-run", "--capability", "9.0",
                  "--cells", "1x4096,3x64"])
    assert rc == exit_codes.REFUSED
    assert "EVEN_K=False" in capsys.readouterr().out


def test_a_model_whose_routing_cannot_form_a_full_stack_is_refused(bk):
    """REFUSED, never nudged. deepseek-v2-lite's rows quantum is 3 and
    BLOCK_M=64 stacks are 64, 128, 192 ... so only every third tread is a legal
    token count. A nudged row is a partly-filled tile and a slope fitted over
    padding is a slope of the padding."""
    cfg = MODEL_CONFIGS["deepseek-v2-lite"]
    assert bk.SWEEP.rows_quantum(cfg) == 3
    with pytest.raises(SystemExit) as caught:
        bk.ladder_rows(cfg, 8)
    assert caught.value.code == exit_codes.REFUSED
    # mixtral's quantum is 1, so the same ladder forms.
    rows = bk.ladder_rows(MODEL_CONFIGS["mixtral-8x7b"], 8)
    assert rows == [64 * n for n in range(1, 9)]


def test_a_grid_too_shallow_for_v4_is_refused_before_it_is_measured(bk):
    """A grid that cannot satisfy its own validity gate is refused at plan
    time, not measured and then voided. This is the 2026-09-09 cap_test arm."""
    assert bk.main(["--dry-run", "--capability", "9.0",
                    "--treads", str(bk.MIN_FIT_TREADS - 1)]) == \
        exit_codes.REFUSED


# --------------------------------------------------------------------------
# 2. the design, and the register file that can flatten it
# --------------------------------------------------------------------------

def test_the_default_grid_is_identified_at_the_residency_bound(bk, limits):
    cells = bk.parse_cells(bk.DEFAULT_CELLS)
    res = {c.key: bk.cell_residency(c, 2, limits) for c in cells}
    design = bk.build_design([c.key for c in cells], res,
                             {c.key: c for c in cells})
    assert design.rank == 4 and design.identified
    assert design.se_multiplier is not None
    assert all(v > 0 for v in design.se_multiplier)


def test_a_register_bound_that_flattens_the_ladder_loses_rank(bk, limits):
    """THE HAZARD, planted. by_smem is the only one of the four limits that
    depends on BLOCK_K, so a register bound below every smem rung puts every
    cell on one rung, `log2 resident` becomes a constant column, and the design
    is a rank-3 matrix that still produces four coefficients."""
    cells = bk.parse_cells(bk.DEFAULT_CELLS)
    file_size = bk.registers_per_sm((9, 0))
    threads = 32 * bk.PINNED_WARPS
    # The n_regs at which by_regs is 1, derived rather than typed.
    flattening = file_size // threads
    flat = {c.key: bk.cell_residency(c, 2, limits, registers=flattening)
            for c in cells}
    assert len({r.resident_blocks for r in flat.values()}) == 1
    design = bk.build_design([c.key for c in cells], flat,
                             {c.key: c for c in cells})
    assert design.rank == 3 and not design.identified
    assert design.se_multiplier is None
    assert "SINGULAR" in " ".join(
        bk.register_sensitivity(cells, 2, limits, bk.PINNED_WARPS))


def test_halving_the_warps_doubles_how_many_ctas_the_register_file_holds(
        bk, limits):
    """The escape `--num-warps` exists for, asserted as the relation."""
    cell = bk.Cell(3, 64)
    for n_regs in (64, 128):
        wide = bk.cell_residency(cell, 2, limits, warps=8, registers=n_regs)
        half = bk.cell_residency(cell, 2, limits, warps=4, registers=n_regs)
        assert half.by_regs == 2 * wide.by_regs, n_regs


def test_the_residency_ladder_is_an_upper_bound_until_n_regs_is_read(bk, limits):
    """occupancy_vs_swizzle says its residency is an upper bound because the
    register limit is not modelled. Here residency IS the contrast, so the
    bound is a state a gate refuses to score rather than a footnote."""
    cell = bk.Cell(3, 64)
    bound = bk.cell_residency(cell, 2, limits)
    assert bound.bound and bound.by_regs is None
    assert "UPPER BOUND" in bound.line()
    measured = bk.cell_residency(cell, 2, limits, registers=64)
    assert not measured.bound
    gate = bk.gate_residency({cell.key: bound}, [])
    assert gate.passed is None and "UPPER BOUND" in gate.observed
    assert gate.kind == bk.VALIDITY, "an unread x axis must void the page"


def test_the_registered_pairs_share_a_rung_at_every_identified_register_count(
        bk, limits):
    """The pairing is an arithmetic identity, not a coincidence of one card:
    equal shared memory gives equal `by_smem`, and the other three limits do
    not depend on BLOCK_K at all, so the two cells of a pair sit on one rung
    whichever limit binds."""
    by_key = {c.key: c for c in bk.parse_cells(bk.DEFAULT_CELLS)}
    for n_regs in (None, 32, 64, 96, 128, 168, 255):
        for low_key, high_key in bk.ISO_SMEM_PAIRS:
            low = bk.cell_residency(by_key[low_key], 2, limits,
                                    registers=n_regs)
            high = bk.cell_residency(by_key[high_key], 2, limits,
                                     registers=n_regs)
            assert low.resident_blocks == high.resident_blocks, (
                low_key, high_key, n_regs)


def test_the_census_refusal_names_the_flag_and_exits_refused(bk):
    assert issubclass(bk.CensusRefusal, bk.RefusedBeforeMeasuring)
    assert issubclass(bk.CensusRefusal, SystemExit)
    text = SCRIPT.read_text()
    assert "--num-warps" in text
    # The refusal that fires when the realised design is singular says what to
    # do about it, at the one place it is raised.
    tree = ast.parse(text)
    raises = [ast.unparse(node) for node in ast.walk(tree)
              if isinstance(node, ast.Raise)
              and "CensusRefusal" in ast.unparse(node)]
    assert len(raises) == 2, raises
    assert any("--num-warps" in r and "singular" in r for r in raises)


# --------------------------------------------------------------------------
# 3. the clock rule
# --------------------------------------------------------------------------

#: The three clocks the planted rows sit at, against a reference of 1485 MHz,
#: which is the operating point the calibration GEMM holds on this card. A HIGH
#: row is a memory-shaped tile boosting; a LOW row is a hungry tile sagging
#: under the power cap. NEITHER is an exclusion.
PLANTED_REFERENCE_MHZ = 1485.0
H200_MEMORY_LOAD_MHZ = 1980.0
SAGGED_MHZ = 1395.0


def _row(bk, cell: str, tiles: int, rep: int, ms: float, *, level, drift, side):
    return bk.Sample(
        cell=cell, num_stages=int(cell.split("x")[0]),
        block_k=int(cell.split("x")[1]), tiles=tiles,
        rows_per_expert=tiles * bk.SUBJECT_BLOCK_M, tokens=tiles * 256, rep=rep,
        ms_p50=ms, ms_min=ms, ms_stdev=0.0, iters=100,
        instrument=bk.SWEEP.SYNTHETIC_INSTRUMENT, warmup_ms=300.0, trials=3,
        l2_flush=True,
        sm_clock_load_mhz=(H200_MEMORY_LOAD_MHZ if side == timing.LEVEL_HIGH
                           else SAGGED_MHZ if side == timing.LEVEL_LOW
                           else PLANTED_REFERENCE_MHZ),
        sm_clock_start_mhz=PLANTED_REFERENCE_MHZ,
        sm_clock_end_mhz=(SAGGED_MHZ if drift is False
                          else PLANTED_REFERENCE_MHZ),
        clock_samples_mhz="1485 1480 1485", power_w=690.0,
        clock_level_ok=level, clock_drift_ok=drift, host_bound=False,
        clock_level_side=side, compiled_smem=1024, compiled_regs=64,
        compiled_spills=0)


def _five_clock_rows(bk):
    """Level, HIGH, LOW, DRIFT, and HIGH-AND-DRIFT in one tread.

    FIVE and not four. Every earlier world in this repo had its drifting row
    sitting LEVEL, so a counter filtering on LEVEL alone counted that tread in
    `level` AND in `drift` and the two counts summed past the number of treads.
    """
    return [
        _row(bk, "3x64", 1, 1, 1.00, level=True, drift=True, side=""),
        _row(bk, "3x64", 1, 2, 1.00, level=False, drift=True,
             side=timing.LEVEL_HIGH),
        _row(bk, "3x64", 1, 3, 1.00, level=False, drift=True,
             side=timing.LEVEL_LOW),
        _row(bk, "3x64", 1, 4, 9.99, level=True, drift=False, side=""),
        _row(bk, "3x64", 1, 5, 9.99, level=False, drift=False,
             side=timing.LEVEL_HIGH),
    ]


def test_drift_excludes_and_both_level_sides_are_kept(bk):
    rows = _five_clock_rows(bk)
    points, _ = bk.collapse(rows, "3x64")
    assert points == [(1, 1.00)], (
        "the two DRIFT rows carry 9.99 ms; a median of 1.00 means both were "
        "excluded and both LEVEL sides were kept")
    assert bk.clock_excluded(True, "", False) is True
    assert bk.clock_excluded(False, timing.LEVEL_HIGH, True) is False
    assert bk.clock_excluded(False, timing.LEVEL_LOW, True) is False
    # None is NOT DETERMINED and an exclusion has to be positively established.
    assert bk.clock_excluded(None, "", None) is False


def test_the_five_clock_counts_partition_the_timed_treads(bk):
    state = bk.clock_state(_five_clock_rows(bk))
    assert state["timed"] == 5
    assert (state["level"] + state["low"] + state["high"] + state["drift"]
            + state["unknown"]) == state["timed"]
    assert state["drift"] == 2 and state["excluded"] == 2
    assert state["level"] == 1 and state["high"] == 1 and state["low"] == 1
    # The HIGH-and-DRIFT tread is counted ONCE, among the drifted, and its side
    # is still recorded there.
    assert state["drift_high"] == 1 and state["drift_level"] == 1
    assert "DRIFT excludes" in state["rule"]
    assert "kept" in " ".join(bk.clock_state_lines(state))


def test_exactly_one_reader_of_the_clock_verdicts(bk):
    """THE RECURRING DEFECT, pinned by AST rather than by a docstring's count.

    A rule applied at one of N call sites is this project's standing failure and
    a clock rule is the exact shape of it. Docstrings are stripped before the
    search, so a function that DESCRIBES the rule is not counted as applying it.
    """
    found = bk._clock_verdict_readers(SCRIPT)
    assert found == set(bk.CLOCK_VERDICT_READERS), found
    assert "clock_excluded" in found, "the exclusion"
    assert "clock_state" in found, "the counter, which excludes nothing"


def test_this_file_reads_the_level_side_and_is_not_on_any_ledger():
    """The repo-wide tripwire, asserted here too so this arm carries its own
    reason: a reader that drops on `clock_level_ok is False` alone would take a
    memory-shaped tread boosted to 1980 MHz for one that ran cold."""
    text = SCRIPT.read_text()
    assert "clock_level_ok" in text and "clock_level_side" in text


# --------------------------------------------------------------------------
# 4. the CSV contract
# --------------------------------------------------------------------------

def test_a_row_travels_through_the_csv_and_comes_back_the_same(bk, tmp_path):
    path = tmp_path / "cells.csv"
    for row in _five_clock_rows(bk):
        bk.append_sample(path, row)
    done, back = bk.read_samples(path)
    assert len(back) == 5
    assert done == {("3x64", 1, rep) for rep in range(1, 6)}
    assert [s.clock_drift_ok for s in back] == [True, True, True, False, False]
    assert [s.clock_level_side for s in back] == [
        "", timing.LEVEL_HIGH, timing.LEVEL_LOW, "", timing.LEVEL_HIGH]
    # And the rule still reads the same way off the file as off the objects.
    assert bk.collapse(back, "3x64")[0] == [(1, 1.00)]


def test_appending_under_a_narrower_header_refuses(bk, tmp_path):
    """`DictWriter` writes the fieldnames it was given and never looks at the
    file, so a wider row under a narrower header shifts every field past the
    first difference: `clock_drift_ok` would read the LEVEL side and a FAILED
    drift would come back None, which every gate keeps."""
    path = tmp_path / "cells.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell", "num_stages", "block_k"])
        writer.writerow(["3x64", 3, 64])
    with pytest.raises(SystemExit) as caught:
        bk.append_sample(path, _five_clock_rows(bk)[0])
    assert caught.value.code == exit_codes.REFUSED


def test_an_optional_column_missing_from_the_file_reads_back_absent(bk,
                                                                    tmp_path):
    """Never as 0 or False. A resumed directory whose older half says nothing
    about how it was timed must be visibly missing that."""
    path = tmp_path / "cells.csv"
    bk.append_sample(path, _five_clock_rows(bk)[0])
    text = path.read_text().replace("True", "").replace("False", "")
    path.write_text(text)
    _, back = bk.read_samples(path)
    assert back[0].clock_drift_ok is None and back[0].clock_level_ok is None


# --------------------------------------------------------------------------
# 5. w, and why it is not the EXA ratio
# --------------------------------------------------------------------------

def test_w_scales_one_to_one_in_the_rate_it_names(bk):
    """The confound the statistic exists to name. `moe.bench.weights` owns the
    arithmetic and this file has NO local fallback: a fallback is how a second
    estimator ran beside the first for a day."""
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    rows = [_row(bk, "3x64", n, rep, 0.12 + 0.60 * n, level=True, drift=True,
                 side="")
            for n in range(1, 9) for rep in (1, 2, 3)]
    one = bk.fit_cell(cfg, "3x64", bk.Cell(3, 64), rows, dtype="bf16",
                      bandwidth_gbps=4374.5, bandwidth_source="test")
    two = bk.fit_cell(cfg, "3x64", bk.Cell(3, 64), rows, dtype="bf16",
                      bandwidth_gbps=8749.0, bandwidth_source="test")
    assert one is not None and two is not None
    assert two.streams == pytest.approx(2.0 * one.streams, rel=1e-9)
    assert one.w.bandwidth_source == "test"
    # The slope itself is what the ladder measured and does not move with it.
    assert one.slope_ms == pytest.approx(two.slope_ms, rel=1e-12)
    assert one.slope_ms == pytest.approx(0.60, rel=1e-9)


def test_the_marginal_arithmetic_intensity_is_measured_not_assumed(bk):
    """V4's input. A cell at or above the ridge has a COMPUTE slope, and its w
    is not a fraction of a stream of anything. The intensity is
    `useful flops per extra M-tile / (w x weight bytes)`, so it comes from this
    run's own w and not from the byte model."""
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    rows = [_row(bk, "3x64", n, 1, 0.12 + 0.60 * n, level=True, drift=True,
                 side="") for n in range(1, 9)]
    fit = bk.fit_cell(cfg, "3x64", bk.Cell(3, 64), rows, dtype="bf16",
                      bandwidth_gbps=4374.5, bandwidth_source="test")
    flops = bk.SWEEP.useful_flops(cfg, bk.SUBJECT_BLOCK_M * cfg.num_experts)
    want = flops / (fit.streams * fit.w.weight_bytes)
    assert fit.marginal_ai == pytest.approx(want, rel=1e-9)
    # And a cell twice as fast per tile is twice as compute-intense.
    faster = [_row(bk, "3x64", n, 1, 0.12 + 0.30 * n, level=True, drift=True,
                   side="") for n in range(1, 9)]
    quick = bk.fit_cell(cfg, "3x64", bk.Cell(3, 64), faster, dtype="bf16",
                        bandwidth_gbps=4374.5, bandwidth_source="test")
    assert quick.marginal_ai == pytest.approx(2.0 * fit.marginal_ai, rel=1e-9)


# --------------------------------------------------------------------------
# 6. identity
# --------------------------------------------------------------------------

#: Parser destinations that must NOT be in the run id, with the reason. Each
#: re-analyses one set of timings rather than changing one, or names where the
#: output goes, or is a mode rather than a sweep. Two analyses of one sweep
#: belong in one directory.
ID_EXEMPT_DESTS = {
    "ridge", "ridge_band", "bandwidth_gbps", "draws", "smem_tolerance",
    "out", "run_id", "dry_run", "self_test", "new", "fail_on_gate",
    "card", "capability", "sm_count", "l2_bytes", "help",
}


def test_every_knob_that_changes_the_measurement_is_in_the_run_id(bk):
    """The bug that overwrote a whole arm once already: an id that omitted
    GROUP_SIZE_M, so the second run resumed into the first's directory, found
    every timing present, skipped all of them, and printed the first run's
    numbers under the second's heading.

    Enumerated from the PARSER rather than from a list kept here, so a knob
    added later is caught rather than silently exempt.
    """
    parser = bk.build_parser()
    base = parser.parse_args(["--capability", "9.0"])
    base_id = bk.default_run_id(base, "nvidia_h200")
    moved = {
        "model": "deepseek-v2-lite", "dtype": "fp8_e4m3",
        "cells": "3x64,6x32", "treads": 6, "reps": 3, "warmup": 100.0,
        "trials": 1, "cell_budget_ms": 50.0, "no_l2_flush": True,
        "seed": 7, "plant_noise": 0.01, "num_warps": 8,
    }
    seen = set()
    for action in parser._actions:                      # noqa: SLF001
        dest = action.dest
        if dest in ID_EXEMPT_DESTS:
            continue
        seen.add(dest)
        assert dest in moved, (
            f"{dest} is neither exempt nor exercised here. Decide whether it "
            "changes the measured milliseconds; if it does it belongs in the "
            "run id, and if it does not it belongs in ID_EXEMPT_DESTS with a "
            "reason")
        args = parser.parse_args(["--capability", "9.0"])
        setattr(args, dest, moved[dest])
        assert bk.default_run_id(args, "nvidia_h200") != base_id, dest
    assert seen, "the parser exposed nothing"
    # And the exempt ones really do leave the id alone.
    for dest, value in (("ridge", 145.8), ("bandwidth_gbps", 1799.4),
                        ("draws", 11), ("smem_tolerance", 0.5)):
        args = parser.parse_args(["--capability", "9.0"])
        setattr(args, dest, value)
        assert bk.default_run_id(args, "nvidia_h200") == base_id, dest


def test_the_card_is_in_the_id_and_an_absent_one_may_not_contradict_a_present(
        bk):
    parser = bk.build_parser()
    args = parser.parse_args(["--capability", "9.0"])
    assert (bk.default_run_id(args, "nvidia_h200")
            != bk.default_run_id(args, "nvidia_a100_sxm4_80gb"))
    assert bk.default_run_id(args, "nvidia_h200").startswith("nvidia_h200-")


def test_two_cells_at_one_block_m_do_not_share_a_triton_cache(bk, tmp_path):
    """`block_m_crossing_sweep.arm_triton_cache` keys on BLOCK_M alone, which is
    right for a sweep whose only variable is BLOCK_M and wrong here: the second
    cell would find the directory warm, compile nothing, and be scored by V1 as
    a grid that ran one kernel."""
    a = bk.arm_cache(tmp_path, bk.Cell(3, 64))
    b = bk.arm_cache(tmp_path, bk.Cell(6, 32))
    assert a != b and a.exists() and b.exists()


# --------------------------------------------------------------------------
# 7. the exit contract
# --------------------------------------------------------------------------

OFF_GPU_MODES = (
    (["--dry-run", "--capability", "9.0"], exit_codes.REFUSED),
    (["--dry-run", "--capability", "9.0", "--num-warps", "8"],
     exit_codes.REFUSED),
    (["--dry-run", "--capability", "9.0", "--cells", "3x64,64x128"],
     exit_codes.REFUSED),
    (["--self-test", "depth", "--capability", "9.0"], exit_codes.DONE),
    (["--self-test", "residency", "--capability", "9.0"],
     exit_codes.CLAIM_FAIL),
    (["--self-test", "blockk", "--capability", "9.0"], exit_codes.CLAIM_FAIL),
    (["--self-test", "null", "--capability", "9.0"], exit_codes.CLAIM_FAIL),
)


@pytest.mark.parametrize("argv,code", OFF_GPU_MODES)
def test_the_log_recomputes_the_code_the_process_returned(bk, capsys, argv,
                                                          code):
    rc = bk.main(list(argv))
    out = capsys.readouterr().out
    assert rc == code, out[-2000:]
    lines = exit_codes.parse_result_lines(out)
    if lines:
        assert exit_codes.classify_text(out) == rc
    else:
        assert rc == exit_codes.REFUSED, (
            "a mode that scored no gate printed no RESULT line, which is what "
            "a REFUSED log looks like from exit_codes")


def test_every_gate_prints_exactly_one_result_line_and_nothing_else_does(bk,
                                                                        capsys):
    bk.main(["--self-test", "depth", "--capability", "9.0"])
    out = capsys.readouterr().out
    parsed = exit_codes.parse_result_lines(out)
    raw = [line for line in out.splitlines() if line.startswith("RESULT: ")]
    assert len(raw) == len(parsed), "a RESULT line the parser cannot read back"
    assert [r.name for r in parsed] == ["V0", "V1", "V2", "V3", "V4", "V5",
                                        "C1", "C2", "C3"]
    # The tally is prose and must not be parseable as a verdict.
    assert "PASS, " in out and not any("PASS," in line for line in raw)


def test_a_crash_is_error_and_not_claim_fail(bk, monkeypatch, capsys):
    """ERROR (4) is outside FINISHED_CODES precisely so "the apparatus broke"
    can be told from "the claim did not hold"."""
    def boom(_argv=None):
        raise RuntimeError("the pod lost its device")

    monkeypatch.setattr(bk, "_main", boom)
    rc = bk.main([])
    capsys.readouterr()
    assert rc == exit_codes.ERROR
    assert rc != exit_codes.CLAIM_FAIL
    assert rc not in exit_codes.FINISHED_CODES
    assert exit_codes.ledger_state(rc) == "RETRY"


def test_a_string_system_exit_becomes_refused_and_not_claim_fail(bk,
                                                                 monkeypatch):
    """`raise SystemExit("some sentence")` exits ONE, which is CLAIM_FAIL in
    the table this repo reads, so a run that refused before measuring anything
    would be filed as a measured refutation."""
    def refuse(_argv=None):
        raise SystemExit("no calibration for this device")

    monkeypatch.setattr(bk, "_main", refuse)
    assert bk.main([]) == exit_codes.REFUSED


def test_the_dry_run_scores_no_gate_and_says_so(bk, capsys):
    rc = bk.main(["--dry-run", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert "RESULT: " not in out
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(out)
    # And the plan is PRINTED before the refusal, which is what the session
    # driver's `printed_a_plan` distinguishes PLANNED from PLAN_REFUSED by.
    assert out.index("## The plan") < out.index("REFUSED. Nothing was measured")


def test_the_plan_prices_the_run_the_pod_makes(bk, capsys):
    """The cost model is the INSTRUMENT'S: `time_kernel` warms for a duration
    and then runs `trials` trials each sized to hold `cell-budget-ms` of kernel
    time, so one timing costs `warmup + trials x budget` whatever the kernel's
    own duration is. The old formula multiplied a per-call time by a call count
    and read a warmup duration as a count of calls."""
    bk.main(["--dry-run", "--capability", "9.0"])
    out = capsys.readouterr().out
    cells = len(bk.parse_cells(bk.DEFAULT_CELLS))
    parser = bk.build_parser()
    args = parser.parse_args([])
    want = cells * args.treads * args.reps * (
        args.warmup + args.trials * args.cell_budget_ms) / 1e3
    assert f"estimate     {want:.0f} s of GPU" in out
    assert "excluding compiles and allocation" in out
    assert f"timings      {cells * args.treads * args.reps} " in out


# --------------------------------------------------------------------------
# 8. the four planted worlds
# --------------------------------------------------------------------------

def test_the_planted_worlds_register_a_verdict_per_gate_and_an_exit_code(bk):
    """A self test that asserts nothing is a smoke test."""
    assert set(bk.SELF_TEST_WORLDS) == {"depth", "residency", "blockk", "null"}
    for world in bk.SELF_TEST_WORLDS.values():
        assert world.expect, world.name
        assert set(world.expect) <= {"C1", "C2", "C3"}
        assert world.exit_code in exit_codes.MEASURED_CODES


@pytest.mark.parametrize("world,expected", [
    ("depth", {"C1": "PASS", "C2": "PASS", "C3": "PASS"}),
    ("residency", {"C1": "FAIL", "C2": "FAIL", "C3": "PASS"}),
    ("blockk", {"C1": "UNKNOWN", "C2": "PASS", "C3": "FAIL"}),
    ("null", {"C1": "UNKNOWN", "C2": "PASS", "C3": "PASS"}),
])
def test_each_world_lands_on_the_verdicts_it_registered(bk, capsys, world,
                                                        expected):
    bk.main(["--self-test", world, "--capability", "9.0"])
    out = capsys.readouterr().out
    assert "REGISTRATION MISMATCH" not in out, out[-3000:]
    got = {r.name: r.verdict for r in exit_codes.parse_result_lines(out)}
    for token, want in expected.items():
        assert got[token] == want, (world, token, got)
    # Every VALIDITY gate passes in a planted world: the worlds are about the
    # CLAIMS, and a page that is INVALID cannot discriminate anything.
    for token in ("V0", "V1", "V2", "V3", "V4", "V5"):
        assert got[token] == "PASS", (world, token)


def test_the_worlds_do_not_all_land_on_one_page(bk, capsys):
    """The discrimination itself. Three distinct exit codes and four distinct
    claim triples over four worlds; a gate that cannot separate them is a gate
    that would have passed on any of them."""
    seen = {}
    for world in sorted(bk.SELF_TEST_WORLDS):
        rc = bk.main(["--self-test", world, "--capability", "9.0"])
        out = capsys.readouterr().out
        got = {r.name: r.verdict for r in exit_codes.parse_result_lines(out)}
        seen[world] = (rc, got["C1"], got["C2"], got["C3"])
    assert len(set(seen.values())) == 4, seen
    assert len({v[0] for v in seen.values()}) == 2, (
        "DONE for the registered reading, CLAIM_FAIL for the other three")


def test_the_planted_drift_rows_would_break_the_fit_if_they_were_kept(bk):
    """The DRIFT rows carry a time 30% off the world's own ladder ON PURPOSE.

    A self test whose excluded rows sat on the ladder would pass whether or not
    the exclusion ran. Here, keeping them moves every planted coefficient.
    """
    text = SCRIPT.read_text()
    assert "ms *= 1.30" in text
    assert len(bk.PLANTED_CLOCKS) == 5
    drifting = [row for row in bk.PLANTED_CLOCKS if row[2] is False]
    assert len(drifting) == 2
    assert any(row[3] == "LEVEL_HIGH" for row in drifting), (
        "one drifting row must also be off LEVEL, or a counter filtering on "
        "LEVEL alone counts that tread twice and stays green")


def test_the_verdict_words_cover_the_null_and_the_unreadable_case(bk):
    assert bk.verdict_of(None) == bk.VERDICT_NEITHER
    flat = bk.Coefficients(0.0, 0.001, 0.001, 0.001,
                           {"depth": 0.01, "block_k": 0.01, "residency": 0.01},
                           0.0, 4)
    assert bk.verdict_of(flat) == bk.VERDICT_NEITHER
    both = bk.Coefficients(0.0, 0.5, 0.0, 0.5,
                           {"depth": 0.01, "block_k": 0.01, "residency": 0.01},
                           0.0, 4)
    assert bk.verdict_of(both) == bk.VERDICT_BOTH
    # A coefficient with no spread is UNKNOWN, never "did not move".
    unknown = bk.Coefficients(0.0, 0.5, 0.0, 0.5,
                              {"depth": None, "block_k": None,
                               "residency": None}, 0.0, 4)
    assert unknown.moves("depth") is None
    assert bk.gate_residency_null(unknown, None, "").passed is None


# --------------------------------------------------------------------------
# 9. the plan page carries every design decision
# --------------------------------------------------------------------------

def test_the_plan_registers_both_readings_before_the_run(bk, capsys):
    bk.main(["--dry-run", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert "PREDICTIONS, registered before the run" in out
    for phrase in ("FOLLOWS DEPTH", "FOLLOWS RESIDENCY", "NEITHER",
                   "SCORED ON w, NOT ON B/(A+B)",
                   "REGISTER SENSITIVITY",
                   "MINIMUM DETECTABLE EFFECT",
                   "THE TWO REGISTERED ISO-SHARED-MEMORY PAIRS",
                   "AND THE ISO-DEPTH RESIDENCY LADDER",
                   "THE THIRD MECHANISM",
                   "NOT A READOUT", "NOT A PRODUCTION CLAIM", "NOT RUN"):
        assert phrase in out, phrase


def test_the_plan_says_the_residency_column_is_a_bound_off_gpu(bk, capsys):
    bk.main(["--dry-run", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert "UPPER BOUND: no n_regs read" in out
    assert "COMPUTED AT THE RESIDENCY BOUND" in out


def test_the_ridge_is_labelled_a_hypothesis_off_gpu_and_is_used_only_by_v4(
        bk, capsys):
    """A calibration-derived quantity is never an asserted literal. Off GPU the
    ridge is the module's HYPOTHESIS band and the page says so; no claim is
    scored against it."""
    bk.main(["--dry-run", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert "HYPOTHESIS" in out
    assert "Used ONLY by V4" in out
    assert "No claim here is scored against a roof" in out


# --------------------------------------------------------------------------
# 10. the two places a kernel is built, counted once
# --------------------------------------------------------------------------

def test_v1_reads_the_compiles_from_both_places_a_kernel_is_built(bk):
    """THE DEFECT THE CENSUS ALMOST INTRODUCED, pinned.

    The compile census builds each cell's first specialisation before the
    metered loop. The loop then finds the Triton directory WARM and counts zero
    new entries for the tile it just proved distinct, so a V1 that read the
    loop alone would score every cell as having compiled nothing -- the exact
    shape of this project's recurring defect, a check reading one of two call
    sites. `_main` sums the two, and `main` is where the sum is formed.
    """
    text = SCRIPT.read_text()
    tree = ast.parse(text)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    body = ast.unparse(main)
    assert "census_compiles.get(c.key, 0)" in body
    assert "loop_compiles.get(c.key, 0)" in body
    # And both producers really do count: one assignment each, in the two
    # functions that own a compile.
    for fn in ("compile_census", "measure"):
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        assert "SWEEP.count_new" in ast.unparse(node), fn


def test_a_cell_that_compiled_nothing_fails_v1(bk):
    """A grid that ran ONE kernel has swept nothing: w is identical across the
    cells by construction, all three coefficients are exactly zero and the
    residual is exactly noise -- a tidy, false NEITHER."""
    executed = {"3x64": 120, "6x32": 120}
    silent = bk.gate_geometry({"3x64": 4, "6x32": 0}, executed, {})
    assert silent.passed is False and "6x32" in silent.observed
    assert silent.kind == bk.VALIDITY
    # Compiled, but nothing read back: UNKNOWN, never PASS on the free half.
    unread = bk.gate_geometry({"3x64": 4, "6x32": 4}, executed, {})
    assert unread.passed is None
    # Compiled, and two distinct shared-memory sizes seen: PASS.
    read = bk.gate_geometry({"3x64": 4, "6x32": 4}, executed,
                            {"3x64": {"shared": 49152},
                             "6x32": {"shared": 24576}})
    assert read.passed is True


#: Everything a measuring run needs before it reaches a device, so a laptop can
#: drive `_main` past the ridge, bandwidth and card resolvers and reach the
#: refusals that come after them. Every value here is the OPERATOR'S assertion,
#: which is the first branch of each resolver and the one that puts the number
#: in the run's own argv; none of them is read back as a calibration.
ASSERTED_CARD = ["--capability", "9.0", "--card", "nvidia_h200",
                 "--ridge", "162.8", "--bandwidth-gbps", "4374.5",
                 "--sm-count", "132", "--l2-bytes", "50000000"]


def test_resuming_with_new_refuses_instead_of_appending_a_second_copy(
        bk, tmp_path, capsys):
    """Both copies would enter the same per-tread median, and the file would
    still read as one run.

    Driven past the ridge and bandwidth resolvers with asserted values, so what
    fires is THIS refusal and not one of theirs: a test satisfied by any
    REFUSED is a test that would pass with the check deleted.
    """
    out = tmp_path / "out"
    parser = bk.build_parser()
    args = parser.parse_args(ASSERTED_CARD)
    run_id = bk.default_run_id(args, "nvidia_h200")
    csv_path = out / "blockk_diagonal" / run_id / "cells.csv"
    csv_path.parent.mkdir(parents=True)
    bk.append_sample(csv_path, _five_clock_rows(bk)[0])
    rc = bk.main([*ASSERTED_CARD, "--out", str(out), "--new"])
    out_text = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert "--new was given" in out_text and str(csv_path) in out_text
    # Without --new the same command resumes and gets as far as the missing
    # GPU stack, which is a DIFFERENT refusal and proves the first one was the
    # --new check rather than anything upstream of it.
    rc = bk.main([*ASSERTED_CARD, "--out", str(out)])
    resumed = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert "--new was given" not in resumed


def test_a_resume_across_two_cards_refuses(bk, tmp_path):
    """Fitting one ladder across two machines. The CARD file beside cells.csv
    is what says which."""
    tree = ast.parse(SCRIPT.read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    body = ast.unparse(main)
    assert "card_path.exists()" in body
    assert "one ladder across two machines" in body


def test_the_census_refuses_a_cell_that_could_not_compile(bk):
    """Every cell is a rung of the design matrix, so dropping one changes its
    rank and the predictions were registered against the whole grid."""
    tree = ast.parse(SCRIPT.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "compile_census")
    body = ast.unparse(node)
    assert "if failed:" in body
    assert "Nothing was timed" in body


def test_an_absent_card_may_be_named_and_a_present_one_may_not_be_contradicted(
        bk, capsys):
    """`detect_card_slug` returns the STRING `nocard` and never None, so a
    reader that takes its return as "a card was detected" makes every `--card`
    on a laptop contradict a device that is not there. The refusal then says
    the opposite of what it means, and it fires on exactly the machine the flag
    exists for."""
    assert bk.SWEEP.detect_card_slug() in (bk.SWEEP.NO_CARD_SLUG,), (
        "this test box has a CUDA device; the laptop path is what is asserted "
        "here")
    rc = bk.main(["--dry-run", "--capability", "9.0", "--card", "nvidia_h200"])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, "a dry run refuses AFTER printing its plan"
    assert "may never contradict" not in out
    assert "run id      nvidia_h200-" in out
