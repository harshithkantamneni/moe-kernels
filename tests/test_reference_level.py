"""The compute reference has to be the RIGHT SIZE, not merely the right shape.

WHAT WENT WRONG, AND WHY NOTHING CAUGHT IT. `compute_reference` qualified a
candidate ladder by fitting `t = C n` through the origin and rejecting it when
the residual was large. That is a test of PROPORTIONALITY and nothing else, and
a line 44x too steep is perfectly proportional. In the BN=256 arm the
BLOCK_M=256 reference took 249.765 ms for one tile on the A100 against 5.724 ms
for the identical setting in its BN=64 twin, and qualified at 0.2% mean error.
Every tread in that arm was then classified against a compute branch two orders
of magnitude too steep, no tread could stand above it, and all 8 cells across
two cards printed as blanks in SURFACE.txt under a caption blaming the tread
count. The corruption read as a boring null.

THREE THINGS ARE PINNED HERE.

  - the LEVEL checks refuse the corrupt reference and clear every sound one, on
    the committed reports themselves rather than on a synthetic stand-in;
  - a refused reference is DISTINGUISHABLE from a sweep that lacked treads, at
    the row level, because that single conflation is what hid the defect;
  - the tile setting that produced the corrupt timing is refused BEFORE it is
    timed, from arithmetic that needs no GPU: a 256x256 fp32 accumulator needs
    256 registers per thread against a hardware maximum of 255.

Non-vacuity is asserted throughout: several of these tests would pass trivially
against an empty scan, so each one counts what it examined.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BM = _load_script()

from moe.spec import MODEL_CONFIGS  # noqa: E402

PUBLISHED = ROOT / "results" / "published"

#: The two cards' own contemporaneous triad ceilings, from
#: `moe/bench/hardware/measured_<device>.yaml`. Used as the ruler for the
#: replayed reports rather than the 160.3 those reports carry, because 160.3 is
#: the stale H200 band that belongs to neither card and the level checks are
#: about level.
CARDS = {
    "a100": {"bandwidth": 1799.3642842463203, "peak_tflops": 262.3712016979615},
    "h200": {"bandwidth": 4374.763038177949, "peak_tflops": 712.2591737976092},
}

#: The BN=256 arm, on both cards. These are the 8 cells this work withdraws.
CORRUPT_REPORT = "qwen2-57b-a14b-bf16-r1024-g1-n256-23a131.report.json"
CORRUPT_ARMS = (
    "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3",
    "2026-09-01-nvidia_h200-cross-card-s3",
)


def card_of(path: Path) -> str:
    return "a100" if "a100" in str(path) else "h200"


def cells_from_report(report: dict, cfg, block_n: int, sm_count: int = 132):
    """Rebuild the sweep's exactly-full cells from a committed report's ladder.

    The reports carry `(tiles, ms)` per block size, which is exactly what the
    ladder fit reads, so the qualification can be re-run on published numbers
    instead of on a re-creation of them.
    """
    cells = []
    for bm, lad in report["ladder"].items():
        for n, ms in lad["points"]:
            cells.append(BM.make_cell(cfg, n * int(bm), int(bm), ms,
                                      sm_count=sm_count, block_n=block_n))
    return cells


def qualify(path: Path):
    """Re-run today's qualification over one committed report."""
    report = json.loads(path.read_text())
    cfg = MODEL_CONFIGS[report["model"]]
    card = CARDS[card_of(path)]
    ridge = card["peak_tflops"] * 1e12 / (card["bandwidth"] * 1e9)
    fixed = report["fixed"]
    block_sizes = tuple(sorted(int(k) for k in report["ladder"]))
    cells = cells_from_report(report, cfg, fixed["BLOCK_SIZE_N"])
    ref = BM.compute_reference(
        cells, block_sizes, cfg=cfg, ridge=ridge,
        bandwidth_gbps=card["bandwidth"], b=report["dtype_bytes"],
        pinned=fixed, capability=(8, 0) if card_of(path) == "a100" else (9, 0))
    return report, ref


def all_published_reports():
    return sorted(PUBLISHED.glob("2026-09-0*/*.report.json"))


# --------------------------------------------------------------------------
# The corrupt reference, replayed from what was published.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", CORRUPT_ARMS)
def test_the_bn256_reference_is_refused_on_both_cards(arm):
    """249.765 ms on the A100 and 10.617 ms on the H200 for ONE tile at
    BLOCK_M=256, against 5.724 and 2.712 in the BN=64 twins. Both qualified on
    shape at 0.2% and 1.0%; both have to be refused on level."""
    path = PUBLISHED / arm / CORRUPT_REPORT
    report, ref = qualify(path)
    assert report["compute_reference"]["block_m"] == 256, (
        "the committed report no longer carries the corrupt reference; this "
        "test has stopped testing what it was written for")
    assert ref.block_m is None
    assert ref.refused
    assert ref.refused_block_m == 256
    assert "REFUSED" in ref.note


def test_the_corrupt_reference_still_passes_the_old_shape_test():
    """The point of the whole item. Proportionality is scale free, so the shape
    test cannot see a 44x error and reported 0.2%. If this ever fails, the shape
    test started catching it and the level checks are no longer load bearing."""
    report = json.loads((PUBLISHED / CORRUPT_ARMS[0] / CORRUPT_REPORT).read_text())
    assert report["compute_reference"]["mean_rel_err"] < 0.005
    cfg = MODEL_CONFIGS[report["model"]]
    cells = cells_from_report(report, cfg, 256)
    pts = BM.ladder_points(cells, 256)
    xs = [float(n) for n, _ in pts]
    ys = [ms for _, ms in pts]
    c = BM._through_origin(xs, ys)
    err = sum(abs(c * x - y) / y for x, y in zip(xs, ys, strict=True)) / len(xs)
    assert err < 0.005, "the corrupt ladder is no longer straight"
    assert c > 240.0, "the corrupt slope is no longer 44x its twin's"


def test_the_bn64_twin_qualifies_so_the_check_is_not_just_rejecting_everything():
    """The same model, the same card, the same session, the same BLOCK_M -- only
    BLOCK_SIZE_N differs. It has to pass, or the level checks are a blanket
    refusal rather than a discrimination."""
    twin = PUBLISHED / CORRUPT_ARMS[0] / (
        "qwen2-57b-a14b-bf16-r1024-g1-n64-eca45c.report.json")
    _, ref = qualify(twin)
    assert ref.block_m == 256
    assert not ref.refused
    assert ref.refusals == ()


def test_every_sound_published_reference_survives_and_only_bn256_does_not():
    """A refusal rule is only worth having if it separates. Over every published
    2026-09 report: the two BN=256 arms are refused and nothing else is."""
    refused, kept = [], []
    for path in all_published_reports():
        report, ref = qualify(path)
        if report["compute_reference"]["block_m"] is None:
            continue                    # already declined on shape, not ours
        (refused if ref.refused else kept).append(path)
    assert len(kept) + len(refused) >= 20, (
        f"only {len(kept) + len(refused)} reports were examined; a scan that "
        "examined nothing also reports zero failures")
    assert len(refused) == 2, [p.name for p in refused]
    assert all(CORRUPT_REPORT == p.name for p in refused)


def test_the_three_level_numbers_separate_the_corrupt_arm_from_the_sound_ones():
    """Not just the verdict: the numbers behind it have to be far apart, or the
    thresholds are sitting on noise."""
    corrupt, sound = [], []
    for path in all_published_reports():
        report, ref = qualify(path)
        if report["compute_reference"]["block_m"] is None:
            continue
        (corrupt if ref.refused else sound).append(ref)
    assert len(sound) >= 20 and len(corrupt) == 2
    # Non-vacuity: at or above 1.0 no tread anywhere could be memory bound.
    assert max(r.vacuity_ratio for r in sound) < 0.6
    assert min(r.vacuity_ratio for r in corrupt) > 1.5
    # Against the sweep's own smaller ladders, which needs no calibration.
    assert max(r.level_ratio for r in sound) < BM.REFERENCE_LEVEL_TOLERANCE
    assert min(r.level_ratio for r in corrupt) > 5.0
    # Nothing sound claims to beat its own roof.
    assert max(r.roof_fraction for r in sound) <= BM.REFERENCE_ROOF_CEILING


# --------------------------------------------------------------------------
# The checks, one at a time, on planted ladders.
# --------------------------------------------------------------------------

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
TILES = (32, 64, 128, 256)
RIDGE = 160.3
BANDWIDTH = 4374.5
PINNED = dict(BM.FIXED, BLOCK_SIZE_N=64, num_stages=3)


def planted(alpha: float, r_max: int = 1024, tiles=TILES):
    grid = BM.build_grid(MIXTRAL, tiles, r_max, 32, 6)
    return BM.synthetic_cells(MIXTRAL, grid, tiles, alpha=alpha, ridge=RIDGE,
                              bandwidth_gbps=BANDWIDTH, b=2, sm_count=132)


def reference(cells, tiles=TILES, *, pinned=None, capability=(9, 0)):
    return BM.compute_reference(
        cells, tiles, cfg=MIXTRAL, ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2,
        pinned=pinned or PINNED, capability=capability)


@pytest.mark.parametrize("alpha", [0.558, 0.10])
def test_a_healthy_sweep_still_qualifies_in_both_worlds(alpha):
    """The level checks must not decide the experiment. Cells generated at
    either alpha come from the same compute roof, so the reference qualifies in
    both and the discrimination is left to the gates where it belongs."""
    ref = reference(planted(alpha))
    assert ref.block_m == 256
    assert not ref.refused
    assert ref.level_comparisons >= 3, (
        "the cross-ladder check compared nothing, so it is not a pass")


@pytest.mark.parametrize("factor", [2.0, 10.0, 44.0])
def test_a_perfectly_proportional_ladder_at_the_wrong_level_is_refused(factor):
    """Multiply the reference ladder by a constant. The through-origin residual
    does not move at all -- that is what scale free means -- and the level
    checks have to be what catches it."""
    cells = [c for c in planted(0.558) if c.block_m != 256]
    slow = [BM.make_cell(MIXTRAL, n * 256, 256, ms * factor, sm_count=132,
                         block_n=64)
            for n, ms in BM.ladder_points(planted(0.558), 256)]
    ref = reference(cells + slow)
    assert ref.refused, f"a reference {factor}x too steep was accepted"
    assert ref.refused_block_m == 256
    assert any("slower than the best smaller block size" in w
               for w in ref.refusals)


def test_a_reference_faster_than_the_compute_roof_is_refused():
    """The other side of the band. Nothing runs faster than `ridge x bandwidth`,
    so a reference that appears to means the ruler belongs to another machine --
    which is exactly the state the A100 reports shipped in, quoting an H200
    ridge."""
    cells = [c for c in planted(0.558) if c.block_m != 256]
    fast = [BM.make_cell(MIXTRAL, n * 256, 256, ms * 0.4, sm_count=132,
                         block_n=64)
            for n, ms in BM.ladder_points(planted(0.558), 256)]
    ref = reference(cells + fast)
    assert ref.refused
    assert ref.roof_fraction > BM.REFERENCE_ROOF_CEILING
    assert any("cannot beat the compute roof" in w for w in ref.refusals)


def test_the_non_vacuity_check_fires_before_any_tread_can_be_misclassified():
    """The failure it names is not "slightly wrong", it is "no answer was
    possible". When the scaled compute branch reaches one full weight read, no
    memory branch can stand above it at any block size, so every blank in the
    report is a property of the reference."""
    full_read_ms = (1e3 * MIXTRAL.num_experts
                    * BM.weight_bytes_per_expert(MIXTRAL, 2)
                    / (BANDWIDTH * 1e9))
    # A reference whose scaled slope at BLOCK_M=32 is exactly one full read.
    per_tile = full_read_ms * 256 / 32
    cells = [c for c in planted(0.558) if c.block_m != 256]
    flat = [BM.make_cell(MIXTRAL, n * 256, 256, per_tile * n, sm_count=132,
                         block_n=64) for n in (1, 2, 3, 4)]
    ref = reference(cells + flat)
    assert ref.refused
    assert ref.vacuity_ratio >= 1.0
    assert any("no tread at any block size" in w for w in ref.refusals)


def test_the_cross_ladder_check_reports_that_it_examined_nothing():
    """A check with nothing to compare against must not read as a pass. When the
    reference is the smallest ladder swept there is no smaller one, and the
    report has to say NOT CHECKED rather than print a ratio of 1.0."""
    cells = [c for c in planted(0.558) if c.block_m == 32]
    ref = reference(cells, tiles=(32,))
    assert ref.level_comparisons == 0
    assert any("NOT CHECKED" in line for line in ref.render())


def test_a_refusal_does_not_fall_through_to_the_next_largest_ladder():
    """Taking the runner-up would swap a loud refusal for a quiet, differently
    wrong reference measured under the same broken pinning on the same card."""
    cells = [c for c in planted(0.558) if c.block_m != 256]
    slow = [BM.make_cell(MIXTRAL, n * 256, 256, ms * 44.0, sm_count=132,
                         block_n=64)
            for n, ms in BM.ladder_points(planted(0.558), 256)]
    ref = reference(cells + slow)
    assert ref.block_m is None
    assert ref.slope_per_tile is None
    assert ref.slope_for(64) is None


# --------------------------------------------------------------------------
# A refused reference must not look like a sweep that lacked treads.
# --------------------------------------------------------------------------

def analyse_with(cells, tiles=TILES, alpha: float = 0.558):
    return BM.analyse(
        cells, MIXTRAL, block_sizes=tiles, alpha=alpha, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, model_name=MIXTRAL.name, dtype="bf16",
        compiles={bm: 1 for bm in tiles}, executed={bm: 1 for bm in tiles},
        sm_count=132, sm_source="test", pinned=PINNED, capability=(9, 0))


def refused_report():
    cells = [c for c in planted(0.558) if c.block_m != 256]
    slow = [BM.make_cell(MIXTRAL, n * 256, 256, ms * 44.0, sm_count=132,
                         block_n=64)
            for n, ms in BM.ladder_points(planted(0.558), 256)]
    return analyse_with(cells + slow)


def test_a_blank_from_a_refused_reference_carries_a_different_reason():
    """THE CONFLATION THAT HID THE DEFECT. Eight cells printed the same blank as
    a tread-poor sweep. Every ladder under a refused reference now says so."""
    report = refused_report()
    ladder = report.payload["ladder"]
    assert ladder, "no ladders were analysed; this test examined nothing"
    reasons = {bm: row["unidentifiable_reason"] for bm, row in ladder.items()}
    blanks = [bm for bm, row in ladder.items() if not row["identifiable"]]
    assert blanks, "nothing came out unidentifiable, so the reason is untested"
    for bm in blanks:
        assert reasons[bm] == BM.NOT_IDENTIFIABLE_REFERENCE_REFUSED, (bm, reasons)


def test_a_genuinely_tread_poor_ladder_says_tread_poor_instead():
    """The other half of the discrimination. With a sound reference, a ladder
    that simply has too few memory-bound treads must NOT be labelled as a
    reference refusal."""
    report = analyse_with(planted(0.558))
    assert not report.payload["compute_reference"]["refusals"]
    reasons = {bm: row["unidentifiable_reason"]
               for bm, row in report.payload["ladder"].items()
               if not row["identifiable"]}
    assert reasons, "every ladder was identifiable; this test examined nothing"
    assert BM.NOT_IDENTIFIABLE_REFERENCE_REFUSED not in reasons.values()
    assert set(reasons.values()) <= {BM.NOT_IDENTIFIABLE_TOO_FEW_TREADS,
                                     BM.NOT_IDENTIFIABLE_IS_REFERENCE}


def test_the_report_says_withdraw_this_arm_and_names_the_refused_block_size():
    """A reader who meets the blanks first reaches for the tread count, which is
    what happened across two cards and eight published cells. The refusal is
    printed above the table, inside it, and in the closing verdict."""
    text = refused_report().text()
    assert "LEVEL roof fraction" in text
    assert "LEVEL non-vacuity" in text
    assert "EVERY BLANK BELOW IS CAUSED BY THE REFUSED REFERENCE" in text
    assert "BLOCK_M=256" in text
    assert "WITHDRAW THIS ARM" in text


def test_the_fit_basis_names_the_refusal_rather_than_a_missing_reference():
    """`basis` is the field a reader checks to learn how membership was decided.
    Under a refusal it has to say refusal, not "no usable compute reference"."""
    fits = refused_report().payload["ladder"]
    assert any("REFUSED at BLOCK_M=256" in row["basis"] for row in fits.values())


# --------------------------------------------------------------------------
# The setting should never have been timed.
# --------------------------------------------------------------------------

def test_the_bn256_tile_needs_more_registers_than_a_thread_has():
    """WHY BLOCK_M=256 AT BLOCK_SIZE_N=256 IS SLOW ON BOTH CARDS, computed
    off-GPU. `tl.dot` accumulates in fp32, so one CTA holds a 256x256 fp32
    accumulator: 256 registers per thread at num_warps=8, against a hardware
    maximum of 255. The accumulator alone does not fit, so the kernel spills to
    local memory. This is card independent, which is why the setting is slow on
    the H200 too."""
    pinned = dict(BM.FIXED, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, num_stages=3,
                  num_warps=8)
    res = BM.tile_resources(pinned, 256, 2, (9, 0))
    assert res.acc_registers_per_thread == 256
    assert res.acc_registers_per_thread > BM.MAX_REGISTERS_PER_THREAD
    assert not res.registers_fit
    assert "registers per thread" in res.refusal


def test_the_bn256_tile_also_blows_the_a100_shared_memory_ceiling_but_not_the_h200():
    """AND WHY THE A100 IS 43.6x SLOW WHERE THE H200 IS ONLY 3.92x. Triton
    multi-buffers the K loop, so one CTA holds `num_stages` copies of the A and
    B tiles: 3 x (256x64 + 64x256) x 2 B = 192 KiB. That is over the A100's 163
    KiB per-block ceiling and under the H200's 227 KiB."""
    pinned = dict(BM.FIXED, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, num_stages=3,
                  num_warps=8)
    a100 = BM.tile_resources(pinned, 256, 2, (8, 0))
    h200 = BM.tile_resources(pinned, 256, 2, (9, 0))
    assert a100.smem_bytes == h200.smem_bytes == 196608     # 192 KiB
    assert a100.smem_fits is False
    assert h200.smem_fits is True
    assert "shared memory" in a100.refusal
    assert "shared memory" not in h200.refusal


def test_four_stages_puts_the_same_tile_over_both_cards():
    """The committed corroboration. At num_stages=4 the same tile needs 256 KiB,
    over BOTH ceilings -- and in the two published num_stages=4 BN=256 arms the
    BLOCK_M=256 ladder is simply ABSENT. The setting is on a resource cliff and
    the sweep walked off it silently."""
    pinned = dict(BM.FIXED, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, num_stages=4,
                  num_warps=8)
    res = BM.tile_resources(pinned, 256, 2, (9, 0))
    assert res.smem_bytes == 262144                         # 256 KiB
    assert res.smem_fits is False
    for arm in ("2026-09-01-nvidia_h200-alpha-surface-s4",):
        for name in ("qwen2-57b-a14b-bf16-r1024-g1-n256-eb7046.report.json",
                     "mixtral-8x7b-bf16-r1024-g1-n256-16cc16.report.json"):
            report = json.loads((PUBLISHED / arm / name).read_text())
            assert report["fixed"]["num_stages"] == 4
            assert report["fixed"]["BLOCK_SIZE_N"] == 256
            assert "256" not in report["ladder"], (
                f"{name} has a BLOCK_M=256 ladder; the cliff this test rests "
                "on is not where it was")


def test_the_settings_this_sweep_actually_pins_are_all_runnable():
    """The refusal must not fire on the experiment's own grid, or it is a
    blanket ban rather than a guard."""
    for stages in (3, 4):
        pinned = dict(BM.FIXED, BLOCK_SIZE_N=64, num_stages=stages)
        plan, refusals = BM.tile_resource_plan(pinned, TILES, 2, (8, 0))
        assert not refusals, refusals
        assert len(plan) == len(TILES)


def test_an_unknown_device_is_unknown_and_not_a_pass():
    """A missing entry in the shared-memory table must never read as "fits".
    The register check needs no device and still fires, which is what makes the
    laptop dry run worth running."""
    pinned = dict(BM.FIXED, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, num_stages=3,
                  num_warps=8)
    res = BM.tile_resources(pinned, 256, 2, None)
    assert res.smem_fits is None
    assert res.refusal, "the register ceiling should still refuse this setting"
    assert "shared memory" not in res.refusal
    assert BM.tile_resources(pinned, 256, 2, (99, 9)).smem_fits is None


def test_the_dry_run_prints_the_resource_plan_and_refuses_off_gpu(capsys):
    """`--dry-run` has to print the full plan and cost on a laptop, and the
    refusal has to be visible there -- that is the run that would have stopped
    the BN=256 arm before a minute of pod time was spent on it."""
    rc = BM.main(["--dry-run", "--model", "qwen2-57b-a14b", "--block-n", "256",
                  "--num-stages", "3", "--capability", "8.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TILE RESOURCE PLAN" in out
    assert "REFUSED BLOCK_M=256" in out
    assert "registers per thread" in out
    assert "shared memory" in out
    assert "sweeping [32, 64, 128] only" in out
    assert "estimated GPU time" in out


def test_the_dry_run_of_the_experiment_as_pinned_refuses_nothing(capsys):
    """Non-vacuity for the test above: the same code path on the sweep's own
    pinned constants must print a clean plan."""
    rc = BM.main(["--dry-run", "--model", "qwen2-57b-a14b", "--capability", "8.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TILE RESOURCE PLAN" in out
    assert "REFUSED" not in out


# --------------------------------------------------------------------------
# SURFACE.txt: the table where eight withdrawn cells looked like eight nulls.
# --------------------------------------------------------------------------

def _surface(tmp_path, reports_by_name):
    """Lay reports out the way the pod does -- one `report.json` per run dir --
    and run the real table generator over them."""
    import subprocess
    for name, payload in reports_by_name.items():
        run = tmp_path / name
        run.mkdir(parents=True)
        (run / "report.json").write_text(json.dumps(payload))
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "alpha_surface.py"),
         str(tmp_path)], capture_output=True, text=True, check=True)
    return out.stdout


def test_the_surface_marks_a_withdrawn_arm_differently_from_a_quiet_one(tmp_path):
    """THE ROW-LEVEL HALF OF THE DEFECT. In the published SURFACE.txt the four
    BN=256 rows printed `--` exactly like the BLOCK_M=128 rows that genuinely
    had too few treads, under a caption blaming tread count. They have to look
    different, and the withdrawal has to be counted."""
    sound = json.loads(
        (PUBLISHED / CORRUPT_ARMS[0]
         / "qwen2-57b-a14b-bf16-r1024-g1-n64-eca45c.report.json").read_text())
    corrupt = json.loads(
        (PUBLISHED / CORRUPT_ARMS[0] / CORRUPT_REPORT).read_text())
    # Re-qualify the corrupt arm under today's checks and stamp the verdict in,
    # which is what a re-run of the sweep over its cells.csv would write.
    _, ref = qualify(PUBLISHED / CORRUPT_ARMS[0] / CORRUPT_REPORT)
    corrupt["compute_reference"] = {
        "block_m": ref.block_m, "refused_block_m": ref.refused_block_m,
        "refusals": list(ref.refusals)}
    sound["compute_reference"] = dict(sound["compute_reference"], refusals=[],
                                      refused_block_m=None)

    text = _surface(tmp_path, {"bn256": corrupt, "bn64": sound})
    assert "REF!" in text
    assert "4 cell(s) WITHDRAWN" in text
    assert "reference refused at BLOCK_M=256" in text
    # The sound arm is untouched: its identifiable fits still print numbers and
    # its own tread-poor rows still print the ordinary blank.
    assert "0.651" in text            # BLOCK_M=32 alpha in the BN=64 twin
    assert "identifiable fit(s)" in text
    table = [ln.split() for ln in text.splitlines()
             if ln.startswith("  qwen2-57b-a14b") and len(ln.split()) == 9]
    assert len(table) == 8, table
    # Column 2 is BLOCK_SIZE_N. Exactly the four BN=256 rows carry the marker,
    # and the four BN=64 rows of the very same model on the very same card do
    # not -- which is the distinction the published table could not make.
    assert {row[2] for row in table} == {"64", "256"}
    for row in table:
        assert ("REF!" in row) == (row[2] == "256"), row


def test_the_surface_says_when_a_report_was_never_level_checked(tmp_path):
    """Non-vacuity for the marker itself. A corpus written before these checks
    existed shows zero withdrawals for the same reason an empty scan does, and
    the table has to say so rather than look clean."""
    stale = json.loads(
        (PUBLISHED / CORRUPT_ARMS[0] / CORRUPT_REPORT).read_text())
    assert "refusals" not in stale["compute_reference"]
    text = _surface(tmp_path, {"stale": stale})
    assert "NO level check recorded" in text
    assert "cell(s) WITHDRAWN" not in text
    assert "REF!" not in [ln.split()[-1] for ln in text.splitlines()
                          if ln.startswith("  qwen2-57b-a14b")]
