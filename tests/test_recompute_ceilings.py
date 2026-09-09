"""Re-deriving the ceiling columns without re-running a three-hour sweep.

The calibration feeds `recompute.CEILING_COLUMNS` into a row and nothing
else: `achieved_bw_gbps`, `bw_ceiling_pattern`, `achieved_peak_tflops` with
`pct_of_achieved_tflops`, `implied_traffic_ratio`, and since schema v6 the roof
at the cell's own clock (`roof_at_cell_clock_tflops`, `pct_of_roof_at_cell_clock`,
`roof_note`). Everything else, `ms_p50`, `tflops`, `compulsory_gbps`,
`arith_intensity_compulsory`, `sm_clock_load_mhz`, comes from the timing, the
byte model and the clock poller and never touches it (`driver._apply_cost`).

So when `calibrate.py` was found to settle under the wrong workload and to name a
tree reduction as its read ceiling, the fix does NOT require re-measuring 17,640
cells. It requires re-deriving those columns. The timings were never wrong.

THE v6 COLUMNS WERE THE SECOND CALL SITE. The per-row roof was added to the
driver and not to this mirror, so a recompute under a new peak rewrote the
fixed roof and left the per-row roof at the old peak's value beside it. The
last section of this file plants v6 rows (HIGH side kept, LOW side excluded,
an fp8 row against the fp8 GEMM's clock) and checks the mirror on them.

The test that this is faithful is an identity: recomputing with the SAME
calibration a row was measured against must reproduce that row exactly. If it
does not, the recompute does not model what the driver did and cannot be trusted
with a different calibration either.

AND THE SCRIPT AROUND IT, from 2026-09-02. The arithmetic was always tested and
the CLI never was, which is how it came to write a new published arm that
recorded no commit of its own and to restamp H200 rows against an A100 ruler
without a word. The second half of this file runs `recompute_ceilings.main` and
checks what it leaves on disk.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from moe.bench import provenance as PV
from moe.bench.published import DERIVED_MARKER
from moe.bench.recompute import (
    CEILING_COLUMNS,
    ceiling_columns,
    load_calibration_hardware,
)


def _load_script():
    """The CLI, loaded by path. `scripts/` is not a package and never has been."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "recompute_ceilings", root / "scripts" / "recompute_ceilings.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RC = _load_script()
HARDWARE = Path(__file__).resolve().parents[1] / "moe" / "bench" / "hardware"

ARM = (Path(__file__).resolve().parents[1] / "results" / "published"
       / "2026-08-26-nvidia_h200-full-three-way")


def rows(limit: int = 2000) -> list[dict]:
    p = ARM / "merged.csv"
    if not p.exists():
        pytest.skip(f"no published sweep at {p}")
    with p.open(newline="") as fh:
        return [r for _, r in zip(range(limit), csv.DictReader(fh), strict=False)]


@pytest.fixture
def hardware():
    cal = ARM / "measured.yaml"
    if not cal.exists():
        pytest.skip("no calibration beside the published arm")
    return load_calibration_hardware(cal)


def test_recomputing_with_the_same_calibration_changes_nothing(hardware):
    """The identity check. Anything else means the recompute is not modelling
    what driver.py did."""
    checked = 0
    for row in rows():
        if not row.get("ms_p50") or float(row["ms_p50"]) <= 0:
            continue
        got = ceiling_columns(row, hardware)
        for col, value in got.items():
            stored = row.get(col, "")
            if col == "bw_ceiling_pattern":
                assert value == stored, (col, value, stored)
                continue
            if stored in ("", None):
                continue
            assert float(value) == pytest.approx(float(stored), rel=1e-9), (
                f"{col} on {row['impl']}/{row['model']}/T{row['num_tokens']}: "
                f"recomputed {value}, stored {stored}")
        checked += 1
    assert checked > 500, f"only {checked} rows exercised"


def test_a_faster_ceiling_lowers_the_implied_traffic_ratio(hardware):
    """implied_traffic_ratio is ceiling x time / compulsory bytes, so raising
    the ceiling raises it. The C4 correction raises the ceiling ~2%, which makes
    every published efficiency figure PESSIMISTIC, and this fixes the sign."""
    import dataclasses
    faster = dataclasses.replace(
        hardware, bandwidth_bytes_s=hardware.bandwidth_bytes_s * 1.02)
    moved = 0
    for row in rows(400):
        if not row.get("ms_p50") or float(row["ms_p50"]) <= 0:
            continue
        a = ceiling_columns(row, hardware).get("implied_traffic_ratio")
        b = ceiling_columns(row, faster).get("implied_traffic_ratio")
        if a in (None, "", 0.0) or b in (None, "", 0.0):
            continue
        assert float(b) == pytest.approx(float(a) * 1.02, rel=1e-9)
        moved += 1
    assert moved > 20, f"only {moved} rows carried a traffic ratio"


def test_the_ratio_is_omitted_for_compute_bound_rows(hardware):
    """driver.py only writes it when the cell is memory bound, because the
    bound it expresses is unsound otherwise. The recompute must agree."""
    seen_compute = False
    for row in rows():
        if not row.get("ms_p50") or float(row["ms_p50"]) <= 0:
            continue
        ai = float(row.get("arith_intensity_compulsory") or 0)
        if ai < hardware.ridge_point(row["dtype"]):
            continue
        seen_compute = True
        assert "implied_traffic_ratio" not in ceiling_columns(row, hardware)
    assert seen_compute, "no compute-bound rows in the sample; test proved nothing"


def test_an_unreadable_calibration_is_refused(tmp_path):
    bad = tmp_path / "nope.yaml"
    bad.write_text("name: something\n")
    with pytest.raises((KeyError, ValueError)):
        load_calibration_hardware(bad)


# --------------------------------------------------------------------------
# The rewrite has to be attributable, and it has to refuse the two ways it was
# found to go wrong when it was run twice.
# --------------------------------------------------------------------------

def _tiny_arm(root: Path, card: str = "NVIDIA H200") -> Path:
    """A one-row arm, so these tests do not depend on a published directory."""
    arm = root / "tiny-arm"
    arm.mkdir()
    row = {"impl": "base", "model": "mixtral-8x7b", "dtype": "bf16",
           "num_tokens": "512", "ms_p50": "1.0", "gpu_name": card,
           "git_sha": "0" * 40, "compulsory_bytes": "1000000",
           "flops": "2000000", "arith_intensity_compulsory": "2.0",
           # Every ceiling column is a FIELD from the start: `rewrite_csv`
           # writes into the header it was given, so an arm missing one of them
           # would fail for a reason that has nothing to do with these tests.
           **{column: "" for column in CEILING_COLUMNS}}
    with (arm / "run_a.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return arm


def test_the_rewrite_records_the_commit_that_did_it(tmp_path):
    """The rows keep the git_sha of the commit that MEASURED them, which is
    right and is also why nothing named the commit that REWROTE them:
    `check_published_shas` validated a commit that did not produce the numbers
    now in the file."""
    arm = _tiny_arm(tmp_path)
    out = tmp_path / "derived"
    cal = HARDWARE / "measured_nvidia_h200.yaml"
    assert RC.main(["--arm", str(arm), "--calibration", str(cal),
                    "--out", str(out)]) == 0

    doc = json.loads((out / RC.RECOMPUTE_JSON).read_text())
    for key in PV.TOP_LEVEL_KEYS:
        assert key in doc, key
    assert doc["provenance"]["git_sha"], "the commit that rewrote the rows"
    assert doc["provenance"]["utc"]
    assert doc["instrument"] == RC.INSTRUMENT
    # The ceiling every re-derived column comes from, and where it came from.
    assert doc["provenance"]["bandwidth"] > 0
    assert "measured_nvidia_h200.yaml" in doc["bandwidth_source"]
    assert doc["calibration_sha256"] == RC.calibration_sha(cal)
    assert doc["derived_from"] == arm.name and doc["row_cards"] == ["NVIDIA H200"]
    assert doc["csvs"]["run_a.csv"]["rows"] == 1
    # ...and the marker still leads with the source arm, which is what
    # `published.derived_from` reads.
    assert (out / DERIVED_MARKER).read_text().splitlines()[0] == arm.name


def test_another_cards_ruler_is_refused_rather_than_stamped(tmp_path):
    """`rewrite_csv` restamps every row from whatever calibration it is handed.
    Run against the H200 yaml then the A100 yaml, both derived
    `<arm>-recalibrated`, and H200 rows came back with achieved_bw_gbps
    4377.2 -> 1799.4 and implied_traffic_ratio 3.716 -> 1.527, with no refusal.
    """
    arm = _tiny_arm(tmp_path, card="NVIDIA H200")
    with pytest.raises(SystemExit, match="REFUSING"):
        RC.main(["--arm", str(arm), "--out", str(tmp_path / "x"),
                 "--calibration",
                 str(HARDWARE / "measured_nvidia_a100_sxm4_80gb.yaml")])
    # ...and the matching card is not refused, or the gate would be a wall.
    assert RC.main(["--arm", str(arm), "--out", str(tmp_path / "y"),
                    "--calibration",
                    str(HARDWARE / "measured_nvidia_h200.yaml")]) == 0


def test_rows_with_no_card_at_all_are_refused(tmp_path):
    """A check that examined nothing reports no failures."""
    arm = _tiny_arm(tmp_path, card="")
    with pytest.raises(SystemExit, match="no row"):
        RC.main(["--arm", str(arm), "--out", str(tmp_path / "z"),
                 "--calibration", str(HARDWARE / "measured_nvidia_h200.yaml")])


def test_a_second_ruler_for_the_same_card_may_not_overwrite_the_first(tmp_path):
    """The H200 was recalibrated six times to six distinct ceilings, and two
    calibrations of ONE card derive ONE default destination. The second run
    overwrote the first's CSVs, measured.yaml, README and marker in silence."""
    arm = _tiny_arm(tmp_path)
    out = tmp_path / "derived"
    first = HARDWARE / "measured_nvidia_h200.yaml"
    assert RC.main(["--arm", str(arm), "--calibration", str(first),
                    "--out", str(out)]) == 0

    # A DIFFERENT FILE FOR THE SAME CARD: same name, different contents, so a
    # path comparison would call these one ruler and a content hash does not.
    second = tmp_path / "measured_again.yaml"
    shutil.copy2(first, second)
    second.write_text(second.read_text() + "\n# recalibrated later that day\n")
    assert RC.calibration_sha(second) != RC.calibration_sha(first)
    with pytest.raises(SystemExit, match="REFUSING to overwrite"):
        RC.main(["--arm", str(arm), "--calibration", str(second),
                 "--out", str(out)])

    # Re-running with the SAME ruler is idempotent and must NOT be refused.
    assert RC.main(["--arm", str(arm), "--calibration", str(first),
                    "--out", str(out)]) == 0


# --------------------------------------------------------------------------
# v6: the roof at the cell's own clock is a calibration-derived column too,
# and it was the second call site the per-row roof fix was never applied at.
# --------------------------------------------------------------------------

import dataclasses  # noqa: E402

from moe.bench import driver as D  # noqa: E402
from moe.bench import roofline as RF  # noqa: E402
from moe.bench import schema as SC  # noqa: E402
from moe.bench import timing as T  # noqa: E402
from moe.bench.bytes_model import PipelineCost  # noqa: E402
from moe.bench.recompute import (  # noqa: E402
    CELL_CLOCK_ROOF_COLUMNS,
    carries_cell_clock_roof,
    rewrite_csv,
)

CARD = "NVIDIA H200"
PROFILE = "NVIDIA H200 (measured)"
BF16_PEAK, FP8_PEAK = 700.0, 1447.7
BF16_CLOCK, FP8_CLOCK = 1515, 1905


def _yaml(tmp_path: Path, *, under_load: bool = True, fp8: bool = True,
          bf16_peak: float = BF16_PEAK) -> Path:
    """A calibration in the shape `calibrate_hardware.py` publishes, with the
    under-load `gemm_clock` / `fp8_gemm_clock` blocks a recalibration records
    (or, `under_load=False`, only the idle scalars the committed files carry).
    """
    import yaml

    detail: dict = {"gpu_name": CARD, "ceiling_pattern": "triad",
                    "gemm_clock_mhz": BF16_CLOCK,
                    "fp8_gemm_clock_mhz": FP8_CLOCK if fp8 else 0}
    if under_load:
        detail["gemm_clock"] = {"median_mhz": BF16_CLOCK, "source": "nvml"}
        if fp8:
            detail["fp8_gemm_clock"] = {"median_mhz": FP8_CLOCK, "source": "nvml"}
    peaks = {"bf16": bf16_peak, "fp16": bf16_peak}
    if fp8:
        peaks["fp8_e4m3"] = FP8_PEAK
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{RF.measured_slug(CARD)}.yaml"
    path.write_text(yaml.safe_dump({
        "name": PROFILE, "verified": True, "source": "planted",
        "memory": {"bandwidth_tb_s": 4.37},
        "compute_dense_tflops": peaks, "detail": detail}))
    return path


def _v6_row(tmp_path: Path, hw, *, dtype: str, load_mhz: float,
            drift_ok: bool = True) -> dict:
    """One row EXACTLY as the driver writes it: `_apply_kernel_timing` then
    `_apply_cost`, off a config whose reference clocks were resolved from the
    same file `hw` was, then through the CSV so every value is a string the
    way `rewrite_csv` will read it."""
    cfg = D.RunConfig(
        out_dir=tmp_path, device="cpu", hardware=hw,
        reference_clock_resolver=lambda: RF.reference_clock(
            CARD, directory=tmp_path),
        reference_clock_resolvers={RF.FP8_FAMILY: lambda: RF.reference_clock(
            CARD, directory=tmp_path, family=RF.FP8_FAMILY)})
    ref = cfg.reference_for(dtype)
    level, drift = T.clock_flags(load_mhz, load_mhz,
                                 load_mhz if drift_ok else load_mhz * 0.9, ref.mhz)
    kt = T.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=4, trials=1,
        warmup_ms=300.0, l2_flush=True, sm_clock_load_mhz=load_mhz,
        sm_clock_start_mhz=load_mhz,
        sm_clock_end_mhz=load_mhz if drift_ok else load_mhz * 0.9,
        clock_level_ok=level, clock_drift_ok=drift, samples=4, warmup_calls=9,
        flush_mb=8, clock_samples=4, clock_source="injected", clock_poll_ms=0.01,
        host_bound=False, host_enqueue_ms=0.1,
        clock_level_side=T.level_side(load_mhz, ref.mhz) or "")
    row = SC.Row(impl="base", model="mixtral-8x7b", dtype=dtype, num_tokens=64,
                 gpu_name=CARD)
    D._apply_kernel_timing(row, kt, cfg)
    # 2e11 FLOP in 1 ms = 200 TFLOP/s; 1e9 bytes -> intensity 200, compute
    # bound against a 4.37 TB/s ruler at either peak.
    D._apply_cost(row, PipelineCost(flops=2e11, bytes_total=10**9), 1.0, cfg)
    with SC.CsvWriter(tmp_path / f"v6_{dtype}_{int(load_mhz)}.csv") as w:
        w.write(row)
    return SC.read_csv(tmp_path / f"v6_{dtype}_{int(load_mhz)}.csv")[0]


def test_the_ceiling_columns_include_the_per_row_roof():
    """The set the schema names as calibration-derived, and the set this module
    rewrites, are one set. `recompute.json` and the derived arm's README list
    `CEILING_COLUMNS`, so a reader of either sees the v6 columns named."""
    for col in CELL_CLOCK_ROOF_COLUMNS:
        assert col in CEILING_COLUMNS, col
        assert col in SC.COLUMNS_ADDED_IN[6], col
    assert len(CEILING_COLUMNS) == 8


def test_recomputing_a_v6_row_with_the_same_calibration_changes_nothing(tmp_path):
    """The identity, on a row that carries the per-row roof. The old mirror
    passed this on the published v5 arm and could not see the v6 hole because
    its fixture predated the column."""
    hw = load_calibration_hardware(_yaml(tmp_path))
    for dtype, load in (("bf16", 1980.0), ("bf16", 1400.0), ("fp8_e4m3", 1905.0)):
        row = _v6_row(tmp_path, hw, dtype=dtype, load_mhz=load)
        assert SC.has_cell_clock_roof(row), (dtype, load, row["roof_note"])
        got = ceiling_columns(row, hw)
        for col in CEILING_COLUMNS:
            if col not in got:
                continue
            if col in ("bw_ceiling_pattern", "roof_note"):
                assert got[col] == row[col], (col, got[col], row[col])
            else:
                assert float(got[col]) == pytest.approx(float(row[col]), rel=1e-12), (
                    dtype, load, col, got[col], row[col])


def test_a_new_peak_moves_the_per_row_roof_with_the_fixed_one(tmp_path):
    """THE PROOF THE VERDICT ASKED FOR. A v6 row measured at 1980 MHz against
    an under-load 1515 reference, recomputed under a peak of 800 TFLOP/s: the
    roof at the cell's clock is 800 x 1980 / 1515 = 1045.5, not the old peak's
    914.9 that the pre-fix mirror left standing beside a rewritten
    `achieved_peak_tflops` of 800."""
    hw = load_calibration_hardware(_yaml(tmp_path))
    row = _v6_row(tmp_path, hw, dtype="bf16", load_mhz=1980.0)
    assert float(row["roof_at_cell_clock_tflops"]) == pytest.approx(
        BF16_PEAK * 1980 / 1515)
    assert row["clock_level_side"] == T.LEVEL_HIGH        # the boosted cell
    assert SC.row_bool(row, "throttled") is False          # and it is KEPT

    newer = load_calibration_hardware(_yaml(tmp_path, bf16_peak=800.0))
    got = ceiling_columns(row, newer)
    assert got["achieved_peak_tflops"] == pytest.approx(800.0)
    assert got["roof_at_cell_clock_tflops"] == pytest.approx(800 * 1980 / 1515)
    assert got["roof_at_cell_clock_tflops"] == pytest.approx(1045.5, abs=0.05)
    assert got["pct_of_roof_at_cell_clock"] == pytest.approx(
        100.0 * float(row["tflops"]) / (800 * 1980 / 1515))
    assert got["roof_note"] == ""
    # The fixed-roof fraction moved by 700/800 and the per-row one by the
    # same factor: the two columns are derived from ONE peak again.
    assert (got["pct_of_roof_at_cell_clock"] / float(row["pct_of_roof_at_cell_clock"])
            == pytest.approx(700 / 800))


@pytest.mark.parametrize("load,side,excluded", [
    (1980.0, T.LEVEL_HIGH, False),   # boosted: kept, roof rescaled UP
    (1400.0, T.LEVEL_LOW, True),     # throttled: excluded, roof still at its clock
])
def test_the_roof_is_written_on_both_sides_and_only_low_is_excluded(
        tmp_path, load, side, excluded):
    """The rule every consumer has to carry: LOW excludes, HIGH does not. The
    recompute is not a gate, so it writes the roof at the cell's clock on both
    sides (a LOW cell's roof is lower, and correct for it) and never touches
    `throttled`, which is the driver's verdict and is what a consumer
    branches on."""
    hw = load_calibration_hardware(_yaml(tmp_path))
    row = _v6_row(tmp_path, hw, dtype="bf16", load_mhz=load)
    assert row["clock_level_side"] == side
    assert SC.row_bool(row, "throttled") is excluded
    got = ceiling_columns(row, hw)
    assert got["roof_at_cell_clock_tflops"] == pytest.approx(BF16_PEAK * load / 1515)
    assert "throttled" not in got and "clock_level_side" not in got


def test_an_fp8_row_is_roofed_at_the_fp8_gemm_s_clock_not_the_bf16_one(tmp_path):
    """TWO ROOFS, TWO CLOCKS. The fp8 GEMM ran at 1905 MHz, the bf16 one at
    1515. An fp8 cell at 1905 is AT its own roof's clock: LEVEL passes, the
    per-row roof is the measured fp8 peak unchanged. Levelled against the bf16
    number it read LEVEL-failed HIGH and `roof_at_clock(1447.7, 1515, 1905)`
    = 1820, 26% above what the calibration measured at that clock."""
    hw = load_calibration_hardware(_yaml(tmp_path))
    row = _v6_row(tmp_path, hw, dtype="fp8_e4m3", load_mhz=1905.0)
    assert SC.timing_verdict(row, "clock_level_ok") == SC.VERDICT_OK
    assert row["clock_level_side"] == ""
    assert float(row["reference_clock_mhz"]) == FP8_CLOCK
    assert "fp8" in row["reference_clock_source"]
    got = ceiling_columns(row, hw)
    assert got["roof_at_cell_clock_tflops"] == pytest.approx(FP8_PEAK)
    assert got["roof_at_cell_clock_tflops"] != pytest.approx(FP8_PEAK * 1905 / 1515)
    # THE FAIL BRANCH: a calibration with no fp8 clock refuses the fp8 row's
    # per-row roof BY NAME and leaves the bf16 row's alone.
    no_fp8 = load_calibration_hardware(_yaml(tmp_path / "nofp8", fp8=False))
    no_fp8 = dataclasses.replace(
        no_fp8, peak_flops={**no_fp8.peak_flops, "fp8_e4m3": FP8_PEAK * 1e12})
    refused = ceiling_columns(row, no_fp8)
    assert refused["roof_at_cell_clock_tflops"] == 0.0
    assert "dtype family" in refused["roof_note"]
    bf16 = _v6_row(tmp_path, hw, dtype="bf16", load_mhz=1500.0)
    assert ceiling_columns(bf16, no_fp8)["roof_note"] == ""


def test_an_idle_scalar_reference_refuses_the_per_row_roof_by_name(tmp_path):
    """Both committed calibrations carry only the idle scalar. The driver
    refuses the per-row roof against it and so must the mirror: a recompute
    that scaled a roof by a number the calibration disowns would move every
    fraction by up to the scalar's 30% spread under the name of a correction.
    """
    under_load = load_calibration_hardware(_yaml(tmp_path))
    row = _v6_row(tmp_path, under_load, dtype="bf16", load_mhz=1980.0)
    idle = load_calibration_hardware(_yaml(tmp_path / "idle", under_load=False))
    assert idle.reference_clocks[RF.BF16_FAMILY].grade == RF.REFERENCE_IDLE_SCALAR
    got = ceiling_columns(row, idle)
    assert got["roof_at_cell_clock_tflops"] == 0.0
    assert got["pct_of_roof_at_cell_clock"] == 0.0
    assert "idle-scalar" in got["roof_note"]
    assert not SC.has_cell_clock_roof({**row, **{k: str(v) for k, v in got.items()}})


def test_a_v5_arm_is_not_given_columns_it_never_carried(tmp_path):
    """The header is the row's schema. A CSV written before v6 has no
    `roof_at_cell_clock_tflops` column, `ceiling_columns` emits none for it,
    and `rewrite_csv` writes the header it was given. Adding the column
    would make `has_cell_clock_roof` True on rows whose under-load clock was
    never sampled against any reference."""
    hw = load_calibration_hardware(_yaml(tmp_path))
    v5 = {"impl": "base", "model": "mixtral-8x7b", "dtype": "bf16",
          "num_tokens": "64", "ms_p50": "1.0", "tflops": "200.0",
          "compulsory_bytes": "1000000000", "arith_intensity_compulsory": "200.0",
          "sm_clock_load_mhz": "1980.0", "gpu_name": CARD,
          "achieved_bw_gbps": "", "bw_ceiling_pattern": "",
          "achieved_peak_tflops": "", "pct_of_achieved_tflops": "",
          "implied_traffic_ratio": ""}
    assert not carries_cell_clock_roof(v5)
    assert not carries_cell_clock_roof({**v5, "roof_at_cell_clock_tflops": SC.UNRECORDED})
    assert set(ceiling_columns(v5, hw)).isdisjoint(CELL_CLOCK_ROOF_COLUMNS)
    src = tmp_path / "v5.csv"
    with src.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(v5))
        w.writeheader()
        w.writerow(v5)
    info = rewrite_csv(src, tmp_path / "out" / "v5.csv", hw)
    with (tmp_path / "out" / "v5.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(v5)
        assert float(next(reader)["achieved_peak_tflops"]) == BF16_PEAK
    assert all(info["changed"][c] == 0 for c in CELL_CLOCK_ROOF_COLUMNS)


def test_the_cli_rewrites_the_per_row_roof_of_a_v6_arm(tmp_path):
    """Through `recompute_ceilings.main`, the path the runbook takes after a
    rental: the derived arm's rows carry the roof under the NEW peak, and the
    change is counted so the console line names the column that moved."""
    arm = tmp_path / "v6-arm"
    arm.mkdir()
    old = load_calibration_hardware(_yaml(tmp_path / "old"))
    row = _v6_row(tmp_path / "old", old, dtype="bf16", load_mhz=1980.0)
    shutil.copy2(tmp_path / "old" / "v6_bf16_1980.csv", arm / "run_v6.csv")
    new_yaml = _yaml(tmp_path / "new", bf16_peak=800.0)
    out = tmp_path / "derived"
    assert RC.main(["--arm", str(arm), "--calibration", str(new_yaml),
                    "--out", str(out)]) == 0
    got = SC.read_csv(out / "run_v6.csv")[0]
    assert float(got["achieved_peak_tflops"]) == pytest.approx(800.0)
    assert float(got["roof_at_cell_clock_tflops"]) == pytest.approx(800 * 1980 / 1515)
    assert got["roof_note"] == ""
    assert float(row["roof_at_cell_clock_tflops"]) != pytest.approx(800 * 1980 / 1515)
    doc = json.loads((out / RC.RECOMPUTE_JSON).read_text())
    assert "roof_at_cell_clock_tflops" in doc["ceiling_columns"]
    assert doc["csvs"]["run_v6.csv"]["changed"]["roof_at_cell_clock_tflops"] == 1
