#!/usr/bin/env python
"""Measure this machine's achievable ceilings. Run once per pod type.

Nsight Compute needs GPU performance counters, which need a host-level module
flag a container tenant cannot set; on a rented pod `ncu` fails with
ERR_NVGPUCTRPERM. So DRAM traffic cannot be read directly, and the roofline
would otherwise rest entirely on a datasheet peak.

This measures the ceilings with ordinary kernels and a clock instead. Efficiency
can then be quoted against what the machine actually delivers, which is both
fairer to your kernel and more defensible in public.

    python scripts/calibrate_hardware.py --dry-run   # the plan and its MDE, free
    python scripts/calibrate_hardware.py             # -> results/.../measured_<device>.yaml
    python scripts/calibrate_hardware.py --publish   # ALSO into the tracked tree

WHY THE DEFAULT NO LONGER WRITES INTO THE TREE
----------------------------------------------
It used to write `moe/bench/hardware/measured_<device>.yaml` directly, and that
file is TRACKED. So the first metered step of every session modified the
checkout, `pod_session.sh` P1 (clean working tree) went from PASS to FAIL, and
`driver.py` stamped `git_dirty=True` on every row measured afterwards: all
3,696 rows of the 2026-09-01 alpha-0558 arm, 44,872 of the 100,144 published
rows in total. Rows that name a commit they were not measured at are rows
nobody can reproduce, and the cause was a side effect of the calibration rather
than anything the operator chose.

The measurement now lands on an untracked session path under the results root
(`results/` is gitignored), which costs nothing and dirties nothing. Copying it
into the tree is a separate decision spelled `--publish`, because the copy is
what makes the ruler visible to `roofline.load_measured()` and therefore to the
sweep, and a session that needs it should say so out loud. `--publish` prints
exactly which file it dirtied and what that does to the rows measured next.

WHAT INSTRUMENT THIS IS
-----------------------
The ceilings are timed by `moe.bench.calibrate`, not by
`moe.bench.timing.time_kernel`, and this script does not pretend otherwise: the
`instrument` field it records names the function that actually ran. Migrating
`calibrate` onto the shared queue-deep loop is that module's own phase; until it
lands, an arm timed with `time_kernel` and a roof timed here are two
instruments, and the only defensible thing to do is write down which is which.

EXIT CODES are `moe.bench.exit_codes`. `--dry-run` exits REFUSED, because a plan
measured nothing and scored no gate, and `classify([])` raises rather than
calling an empty gate list DONE for exactly that reason. (Note the divergence:
`scripts/block_m_crossing_sweep.py --dry-run` exits 0 today. One of the two
should move; this one is the reading the table supports.) A calibration whose
clock could not be established is INVALID rather than DONE, because `sustained_peak_tflops` and
`gemm_efficiency_pct` are normalised by that clock and the module refuses to
quote them; the yaml is still written, so the numbers that do not depend on the
clock survive for a reader who wants them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from moe.bench import calibrate as calibrate_mod  # noqa: E402
from moe.bench import exit_codes as EX  # noqa: E402
from moe.bench.calibrate import DEFAULT_CEILING, calibrate  # noqa: E402
from moe.bench.provenance import provenance_block, run_id  # noqa: E402
from moe.bench.roofline import (  # noqa: E402
    HARDWARE_DIR,
    ambiguous_for_device,
    current_gpu_name,
    for_device,
    load_hardware,
    measured_slug,
    power_limit_w,
)
from moe.bench.schema import git_provenance  # noqa: E402

#: Two-sided 5%, 80% power. The multiplier a minimum-detectable-effect is
#: `z(1-a/2) + z(power)` = 1.96 + 0.84; written out so the line an arm prints
#: can be checked rather than believed.
MDE_Z = 2.80


def instrument_name() -> str:
    """What actually timed these ceilings, named rather than assumed.

    Reads a `TIMING_BASIS` from `moe.bench.calibrate` when that module grows
    one (its migration onto `timing.time_kernel` is a separate phase), and
    otherwise says plainly which loop ran. A calibration that claimed
    `timing.TIMING_BASIS` while running `time_eager` would put the roof and the
    ladders under one label when they are two instruments, which is the
    confusion the audit measured at 12-16% in alpha.
    """
    basis = getattr(calibrate_mod, "TIMING_BASIS", None)
    if basis:
        return str(basis)
    return ("moe.bench.calibrate via timing.time_eager (queue-deep, pre-primed "
            "events, L2 flush between iterations); NOT timing.TIMING_BASIS")


def mde_line(sigma_pct: float | None, trials: int, why: str) -> str:
    """One line: the smallest effect this run could resolve, and from what.

    `sigma_pct` is a stated noise level in percent of the measured value, `why`
    says where it came from. Returns a REFUSAL line when no noise level is
    available, because an MDE derived from a number nobody measured is a prior
    dressed as a bound, which is exactly what B14 found across the arms.
    """
    if sigma_pct is None or sigma_pct <= 0 or trials < 1:
        return (f"MDE: REFUSED. No noise level to derive one from ({why}). "
                "State one before quoting a difference as real.")
    mde = MDE_Z * sigma_pct / math.sqrt(trials)
    return (f"MDE: {mde:.2f}% of the measured value, at alpha 0.05 and 80% "
            f"power, from sigma {sigma_pct:.2f}% over {trials} trial(s) "
            f"({why}). A difference smaller than this is not resolvable here.")


def published_sigma_pct(published: Path) -> tuple[float | None, int, str]:
    """`(median relative within-cell sd in %, rows used, provenance sentence)`.

    Derived from the published rows themselves rather than asserted, so the MDE
    moves when the apparatus does. It is a LOWER BOUND on the real noise: the
    column is within-cell `ms_std / ms_p50`, and the between-run spread nobody
    has measured yet is larger. Said in the sentence, so the number is never
    quoted as the whole story.
    """
    rel: list[float] = []
    for path in sorted(published.glob("*/run_*.csv")):
        try:
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        p50 = float(row.get("ms_p50") or 0)
                        sd = float(row.get("ms_std") or 0)
                    except (TypeError, ValueError):
                        continue
                    if p50 > 0 and sd > 0:
                        rel.append(sd / p50)
        except OSError:
            continue
    if not rel:
        return None, 0, f"no timed rows with an ms_std under {published}"
    return (100.0 * statistics.median(rel), len(rel),
            f"median within-cell ms_std/ms_p50 over {len(rel)} published rows; "
            "a LOWER bound, since between-run spread is not in that column")


def _device_facts() -> dict:
    """Device facts that settle what a percentage is a percentage OF.

    The memory clock is the important one. HBM is DDR, so the pin rate is
    clk x 2 x bus_width / 8, and the doubling is mandatory: omitting it gives
    exactly half, the same class of silent 2x error as the read-for-ownership
    question. On this H200, 3201 MHz gives 4916.7 GB/s, which is 2.4% above
    NVIDIA's published 4.8 TB/s -- so the datasheet figure is already derated
    and is not a theoretical maximum.
    """
    import torch

    from moe.bench.timing import _nvidia_smi

    out: dict = {}
    props = torch.cuda.get_device_properties(0)
    out["l2_bytes"] = getattr(props, "L2_cache_size", 0)
    out["sm_count"] = props.multi_processor_count

    vals = _nvidia_smi("clocks.max.memory,clocks.current.memory,"
                       "clocks_throttle_reasons.active")
    if vals:
        parts = [v.strip() for v in vals[0].split(",")]
        try:
            out["clocks_max_memory_mhz"] = float(parts[0].split()[0])
            out["clocks_current_memory_mhz"] = float(parts[1].split()[0])
            out["throttle_reasons"] = parts[2] if len(parts) > 2 else ""
        except (ValueError, IndexError):
            pass
    clk = out.get("clocks_max_memory_mhz")
    bits = _memory_bus_bits(props.name)
    if clk and bits:
        # DDR doubles the transfer rate, hence the factor of two.
        out["memory_bus_bits"] = bits
        out["pin_rate_gbps"] = round(clk * 2 * bits / 8 / 1000, 1)
    elif clk:
        # No width, no pin rate. A wrong one is worse than none: it is the
        # only hard physical bound in this file, and every "nothing exceeded
        # the pin rate" argument rests on it.
        out["memory_bus_bits_unknown_for"] = props.name
    return out


#: Memory bus width in bits, by substring of the device name. nvidia-smi has no
#: query for this and torch does not expose cudaDeviceProp.memoryBusWidth, so it
#: is a table. Each entry is checked by reproducing the vendor bandwidth figure:
#:   H200  3201 MHz x 2 x 6144 / 8 = 4916.7 GB/s  (spec 4.8 TB/s, derated)
#:   A100  1593 MHz x 2 x 5120 / 8 = 2038.8 GB/s  (spec 2039 GB/s)
#:   H100  2619 MHz x 2 x 5120 / 8 = 3352.3 GB/s  (spec 3.35 TB/s)
#: An earlier version hardcoded 6144 for every device, which reported an A100's
#: pin rate as 2446.8 instead of 2038.8: 20% high, on the one number that is
#: supposed to be a hard physical bound.
_MEMORY_BUS_BITS = {
    "H200": 6144,
    "H100": 5120,
    "A100": 5120,
    "B200": 8192,
}


def _memory_bus_bits(gpu_name: str) -> int | None:
    for key, bits in _MEMORY_BUS_BITS.items():
        if key in gpu_name:
            return bits
    return None


def swept(args) -> dict:
    """The knobs that change the numbers, in the order `run_id` will sort them.

    Everything here is a knob the operator can vary between two runs on the
    same card; `--out`, `--publish` and `--compare-to` are deliberately absent
    because they re-file or re-describe a measurement rather than change it.
    """
    return {"buffer_gb": float(args.buffer_gb), "gemm_n": int(args.gemm_n),
            "ceiling": str(args.ceiling), "settle": bool(args.settle),
            "settle_s": float(args.settle_seconds)}


def session_out(args, card: str) -> Path:
    """Where a calibration lands by default: untracked, card-named, id-named.

    Under the results root, which `.gitignore` excludes, so a calibration never
    dirties the checkout. The run id carries every swept knob, so two ceilings
    measured with different buffers or a different settle cannot land on each
    other; the card is at the front of it for the reason `provenance.run_id`
    documents.
    """
    root = Path(args.results_root)
    return (root / "calibration" / run_id(card=card, **swept(args))
            / f"{measured_slug(card)}.yaml")


def write_cells(path: Path, cal, prov) -> Path:
    """One row per bandwidth pattern, beside the yaml, with its provenance.

    `find results/published -name cells.csv` returned zero across every arm, so
    no published number could be traced to the samples under it. A ceiling has
    the same duty as a cell: the row says what was timed, by what, and how
    little of the timing state is known. The KernelTiming columns this
    instrument does not report (`warmup_ms`, `iters`, `trials`, the clock
    verdicts) are written EMPTY rather than filled with a plausible default,
    and `instrument` says which loop ran, so a reader can see the gap instead
    of inheriting a guess.
    """
    columns = ["pattern", "bytes_moved", "ms_p50", "ms_min", "gbps",
               "gbps_peak_min", "sm_clock_start_mhz", "sm_clock_end_mhz",
               "instrument", "warmup_ms", "iters", "trials", "l2_flush",
               "sm_clock_load_mhz", "clock_level_ok", "clock_drift_ok", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns + list(prov.as_columns()))
        writer.writeheader()
        for pat in cal.bandwidth_patterns:
            row = dict.fromkeys(columns, "")
            row.update(pattern=pat.pattern, bytes_moved=pat.bytes_moved,
                       ms_p50=pat.ms_p50, ms_min=pat.ms_min, gbps=pat.gbps,
                       gbps_peak_min=pat.gbps_peak_min,
                       sm_clock_start_mhz=pat.sm_clock_start_mhz,
                       sm_clock_end_mhz=pat.sm_clock_end_mhz,
                       instrument=prov.instrument, l2_flush=True, note=pat.note)
            row.update(prov.as_columns())
            writer.writerow(row)
    return path


def score(cal, pin_rate_gbps: float | None) -> list[tuple[str, str, str, str]]:
    """`(kind, name, verdict, detail)` for every gate this calibration scores.

    SIX gates, and `tests/test_calibrate_hardware.py` plants both branches of
    every one of them against a synthetic `Calibration`: clock_established,
    ceiling_pattern_measured, no_pattern_exceeds_the_pin_rate,
    write_rate_is_a_store_rate, clock_steady_across_patterns, not_throttled.
    The docstring said five and the tests planted five, and the one left out was
    `write_rate_is_a_store_rate`, whose FAIL branch had therefore never run:
    the sound fixture puts write below the pin rate and the over-the-pin fixture
    drops the write pattern entirely, so the gate was not even emitted there.

    Two of the six are CONDITIONAL, and a gate that is absent is not a gate that
    passed: `write_rate_is_a_store_rate` is scored only when a write pattern was
    measured, and neither pin-rate gate can be scored without a bus width, which
    is what the UNKNOWN below is for.

    UNKNOWN is used where the run could not decide rather than where it decided
    "fine": no settle means no clock plateau to check against, and a card
    outside the memory-bus table has no pin rate, so the claim that nothing
    exceeded it was not tested. Both count against their gate, which is what
    makes an untested claim visible in the exit code.
    """
    gates: list[tuple[str, str, str, str]] = []

    established = cal.clock_established
    gates.append((EX.VALIDITY, "clock_established",
                  EX.PASS if established is True else
                  EX.FAIL if established is False else EX.UNKNOWN,
                  "the samples agree with each other and with the settle plateau"
                  if established is True else
                  "the samples disagree with each other or with the settle "
                  "plateau; nothing normalised by the clock may be quoted"
                  if established is False else
                  "no settle to check the samples against (--no-settle)"))

    ceiling = cal.pattern(cal.ceiling_pattern)
    disowned = bool(ceiling and calibrate_mod.DISOWNED in (ceiling.note or ""))
    gates.append((EX.VALIDITY, "ceiling_pattern_measured",
                  EX.PASS if (ceiling and not disowned) else EX.FAIL,
                  f"{cal.ceiling_pattern} = {ceiling.gbps:.1f} GB/s" if
                  (ceiling and not disowned) else
                  f"{cal.ceiling_pattern} was disowned by the measurement: "
                  f"{(ceiling.note if ceiling else 'pattern absent')}"))

    write = cal.pattern("write")
    if pin_rate_gbps is None:
        gates.append((EX.CLAIM, "no_pattern_exceeds_the_pin_rate", EX.UNKNOWN,
                      "no memory-bus width for this device, so the one hard "
                      "physical bound in the file could not be derived"))
    else:
        worst = max(cal.bandwidth_patterns, key=lambda p: p.gbps)
        over = worst.gbps > pin_rate_gbps
        gates.append((EX.CLAIM, "no_pattern_exceeds_the_pin_rate",
                      EX.FAIL if over else EX.PASS,
                      f"fastest pattern {worst.pattern} {worst.gbps:.1f} GB/s "
                      f"against a derived pin rate of {pin_rate_gbps:.1f} GB/s"
                      + ("; a figure above the pin rate means the byte "
                         "accounting is wrong, not that the hardware exceeded "
                         "its specification" if over else "")))
        if write is not None:
            gates.append((EX.CLAIM, "write_rate_is_a_store_rate",
                          EX.PASS if write.gbps <= pin_rate_gbps else EX.FAIL,
                          f"write {write.gbps:.1f} GB/s is "
                          f"{100 * write.gbps / pin_rate_gbps:.1f}% of the pin "
                          "rate; a read-for-ownership would make it 2N and "
                          "impossible"))

    gates.append((EX.CLAIM, "clock_steady_across_patterns",
                  EX.FAIL if cal.clock_ramped else EX.PASS,
                  "SM clock differed by >5% ACROSS the patterns, so they were "
                  "measured in different states and are not comparable"
                  if cal.clock_ramped else
                  "every pattern was measured in the same clock state"))

    throttled = bool((cal.clocks or {}).get("throttled"))
    gates.append((EX.CLAIM, "not_throttled",
                  EX.FAIL if throttled else EX.PASS,
                  f"clocks {(cal.clocks or {}).get('sm_start_mhz')} -> "
                  f"{(cal.clocks or {}).get('sm_end_mhz')} MHz"
                  + (", THROTTLED: ceilings measured on a throttling GPU are "
                     "low" if throttled else "")))
    return gates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the yaml exactly here. Default: an untracked "
                         "path under --results-root named by the card and the "
                         "run id, so a calibration never dirties the checkout")
    ap.add_argument("--results-root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results",
                    help="root of the untracked results tree (gitignored)")
    ap.add_argument("--publish", action="store_true",
                    help="ALSO copy the yaml into moe/bench/hardware/, which is "
                         "where roofline.load_measured() looks and therefore "
                         "what the sweep reads. This modifies a TRACKED file: "
                         "every row measured before you commit it carries "
                         "git_dirty=True")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the paths and the MDE, measure "
                         "nothing, write nothing")
    ap.add_argument("--card", default=None,
                    help="name the card for --dry-run planning on a machine "
                         "that has none; ignored when a GPU is present")
    ap.add_argument("--buffer-gb", type=float, default=8.0,
                    help="STREAM buffer size; must dwarf L2 and run long enough "
                         "that launch and clock ramp do not matter")
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--ceiling", default=DEFAULT_CEILING,
                    choices=("read_stream", "read_reduce", "copy", "triad",
                             "write"),
                    help="which pattern defines achieved bandwidth. triad is the "
                         "canonical STREAM metric and the default; read_stream "
                         "is the ruler this study's ~98.5%%-read traffic "
                         "actually matches, and read_reduce is the reduction's "
                         "lower bound on it. Changing this re-baselines every "
                         "efficiency column; price it with "
                         "scripts/ruler_rebaseline.py first")
    ap.add_argument("--no-settle", dest="settle", action="store_false",
                    help="skip the clock settle. Only for a quick smoke check: "
                         "measuring from idle walks the clock ramp across the "
                         "patterns and the ceilings are not comparable")
    ap.add_argument("--settle-seconds", type=float, default=30.0)
    ap.add_argument("--compare-to", default=None,
                    help="datasheet profile to compare against; auto-detected "
                         "when the device name is unambiguous")
    args = ap.parse_args(argv)

    card = current_gpu_name() or (args.card or "")
    sigma_pct, _n, why = published_sigma_pct(
        Path(__file__).resolve().parents[1] / "results" / "published")

    if args.dry_run:
        # THE PLAN, AND NOTHING ELSE. A dry run writes no file, tracked or
        # otherwise, so a laptop rehearsal of a session cannot leave the tree
        # dirty for the rows that follow it.
        print("[calibrate] dry run: nothing is measured and nothing is written")
        print(f"  patterns          bandwidth ladder, ceiling = {args.ceiling}")
        print(f"  buffers           {args.buffer_gb:g} GiB, L2 flushed between "
              "iterations")
        print(f"  gemm              {args.gemm_n}^3 dense, bf16 and fp8 where "
              "the silicon has it")
        settle = (f"up to {args.settle_seconds:.0f}s under load" if args.settle
                  else "SKIPPED (--no-settle)")
        print(f"  settle            {settle}")
        print(f"  instrument        {instrument_name()}")
        if card:
            print(f"  card              {card}")
            print(f"  would write       {session_out(args, card)}")
        else:
            print("  card              NOT VISIBLE from here, so the run id "
                  "cannot be formed (provenance.NoCard); pass --card to plan "
                  "for a named part")
        publish = (f"yes, into {HARDWARE_DIR} (dirties a TRACKED file)"
                   if args.publish else
                   "no; --publish copies it into the tree")
        print(f"  publish           {publish}")
        # B14: an arm that states no MDE lets any difference be read as real.
        print(f"  {mde_line(sigma_pct, 3, why)}")
        return EX.REFUSED

    if not torch.cuda.is_available():
        print("REFUSED: calibration needs a GPU; there is no ceiling to measure "
              "here. --dry-run prints the plan without one.", file=sys.stderr)
        return EX.REFUSED

    print("[calibrate] measuring achievable bandwidth and dense BF16 ...")
    print(f"[calibrate] buffers {args.buffer_gb:g} GiB, L2 flushed between iterations")
    if args.settle:
        print(f"[calibrate] settling clocks under load (up to "
              f"{args.settle_seconds:g}s) before measuring anything")
    cal = calibrate(int(args.buffer_gb * (1 << 30)), args.gemm_n, args.ceiling,
                    settle=args.settle, settle_seconds=args.settle_seconds)

    props = torch.cuda.get_device_properties(0)
    print(f"\n  device            {cal.gpu_name}")
    l2_mib = getattr(props, "L2_cache_size", 0) / 2**20
    ratio = args.buffer_gb * 1024 / max(l2_mib, 1)
    print(f"  L2                {l2_mib:.0f} MiB   (buffers are {ratio:.0f}x larger)")

    # The board power limit is what actually tells an H200 SXM (700 W) from an
    # H200 NVL (600 W); torch reports both as "NVIDIA H200".
    tdp = power_limit_w()
    observed = _device_facts()
    if tdp:
        print(f"  power limit       {tdp:.0f} W"
              + ("   (SXM)" if tdp > 650 else "   (NVL)"))
    pin = observed.get("pin_rate_gbps")
    if pin:
        print(f"  memory clock      {observed['clocks_max_memory_mhz']:.0f} MHz"
              f"  -> pin rate {pin:.1f} GB/s")
        print("                    (clk x 2 for DDR x 6144 bits / 8; a "
              "datasheet figure below this is already derated)")

    spec_bw = spec_tf = None
    profile = args.compare_to
    if profile is None:
        profile = for_device(cal.gpu_name, tdp_w=tdp)
        if profile is None:
            tied = ambiguous_for_device(cal.gpu_name)
            if tied:
                print(f"\n  device name is AMBIGUOUS between {tied}; pass "
                      "--compare-to to pick one. Measured values below stand "
                      "on their own regardless.")
    if profile:
        try:
            hw = load_hardware(profile)
            spec_bw, spec_tf = hw.bandwidth_bytes_s / 1e9, hw.peak("bf16") / 1e12
        except Exception as e:  # noqa: BLE001
            print(f"  (no spec comparison: {e})")

    # Never print a bare "% of peak": the same figure is 97.0% of NVIDIA's
    # published 4.8 TB/s and 94.7% of the real 4916.7 GB/s pin rate, and a
    # reader cannot tell which unless the denominator is in the string.
    print(f"\n  {'pattern':<8}{'GB/s':>10}{'of spec':>9}{'of pin':>9}"
          f"{'p50 ms':>10}   note")
    for pat in cal.bandwidth_patterns:
        of_spec = f"{100 * pat.gbps / spec_bw:>7.1f}%" if spec_bw else f"{'':>8}"
        of_pin = f"{100 * pat.gbps / pin:>7.1f}%" if pin else f"{'':>8}"
        mark = " <-- ceiling" if pat.pattern == cal.ceiling_pattern else ""
        print(f"  {pat.pattern:<8}{pat.gbps:>10.1f}{of_spec}{of_pin}"
              f"{pat.ms_p50:>10.3f}   {pat.note}{mark}")
    if spec_bw and pin:
        print(f"  {'':8}{'':10}{'^ vs':>9}{'^ vs':>9}")
        print(f"  {'':8}{'':10}{spec_bw:>8.0f} {pin:>8.0f}  GB/s "
              "(published, derived)")

    print(f"\n  achieved BW       {cal.achieved_bandwidth_gbps:8.1f} GB/s "
          f"(pattern: {cal.ceiling_pattern})")
    print(f"  achieved BF16     {cal.achieved_bf16_tflops:8.1f} TFLOP/s "
          f"(GEMM {args.gemm_n}^3)")
    if spec_tf:
        print(f"                    {100 * cal.achieved_bf16_tflops / spec_tf:8.1f}% "
              f"of the {spec_tf:.1f} datasheet figure")
    sustained = cal.sustained_peak_tflops
    if sustained:
        print(f"  measured at       {cal.gemm_clock_mhz} MHz, where the silicon "
              f"can do {sustained:.1f} TFLOP/s")
        print(f"                    {cal.gemm_efficiency_pct:8.1f}% of THAT "
              "<-- efficiency at the measured clock")
        if spec_tf:
            implied = spec_tf * 1e12 / (sustained * 1e12 / max(cal.gemm_clock_mhz, 1))
            print(f"                    the datasheet assumes ~{implied:.0f} MHz, "
                  "a boost clock this part does not hold under dense tensor load")

    # WHERE THAT CLOCK CAME FROM. It used to be one nvidia-smi sample taken
    # after the GEMM had already synchronised, and across the eleven committed
    # H200 calibrations it moved 30% (1485-1935 MHz) while the achieved rate
    # moved 12%. The line below is the whole reason this run can be believed:
    # samples taken under load, their spread, and the idle sample beside them.
    for label, rec in (("bf16 GEMM", cal.gemm_clock),
                       ("fp8 GEMM", cal.fp8_gemm_clock)):
        if not rec:
            continue
        print(f"  clock, {label:<10}{rec['median_mhz']:>5} MHz median of "
              f"{rec['samples']}, spread {rec['spread_pct']:.1f}%")
        print(f"  {'':<17}post-hoc idle sample {rec['after_idle_mhz']} MHz "
              f"(what this field used to be)"
              + (f", {rec['power_w']:.0f} W" if rec.get("power_w") else ""))
    established = cal.clock_established
    if established is False:
        print("  CLOCK NOT ESTABLISHED: the samples disagree with each other or "
              "with the settle plateau.")
        print("                    Nothing normalised against a clock -- "
              "sustained_peak_tflops, gemm_efficiency_pct -- may be quoted.")
    elif established is None:
        print("  clock state       not established (no settle to check against)")

    print("\n  ridge point by choice of denominator (FLOP/byte):")
    for pat in cal.bandwidth_patterns:
        mark = " <-- used" if pat.pattern == cal.ceiling_pattern else ""
        if pat.pattern == (cal.matched_pattern().pattern
                           if cal.matched_pattern() else None):
            mark += " <-- matched to this study's traffic"
        print(f"    {pat.pattern:<12}{cal.ridge_point(pat.gbps):>8.0f}{mark}")
    if spec_bw and spec_tf:
        print(f"    {'datasheet':<12}{spec_tf * 1e12 / (spec_bw * 1e9):>8.0f}")
    band = cal.ridge_band()
    if band:
        print(f"    band          {band[0]:.1f} to {band[1]:.1f}  "
              f"({100 * (band[1] / band[0] - 1):.1f}% wide) -- quote this, not "
              "one end, for any crossing inside it")

    for line in cal.refusals:
        print(f"  REFUSED           {line}")

    # BOTH settles, each labelled with what it governs. The memory settle has
    # been measured since 2026-08-27 and was never printed, so this report read
    # "settle reached at 1500 MHz" followed by "clocks 1500 -> 1980 MHz" -- the
    # signature of an unsettled ramp -- and docs/STUDY.md still listed the work
    # as not done five days after it was done, on the strength of this output.
    print()
    for line in cal.settle_lines():
        print(f"  {line}")
    for info in (cal.settle, (cal.settle or {}).get("bandwidth_settle") or {}):
        if info and not info.get("skipped") and not info.get("settled"):
            print("                    clocks still moving when the budget ran "
                  "out; raise --settle-seconds")
            break

    if cal.warmup_pass:
        print(f"\n  warm-up pass      discarded; largest per-pattern change "
              f"{cal.warmup_drift_pct:.1f}%")
        if cal.warmup_drift_pct > 3.0:
            print("                    >3% means the settle did not reach the "
                  "state the measurement induces; raise --settle-seconds")

    if cal.clock_ramped:
        print("\n  WARNING: SM clock differed by >5% ACROSS the patterns, so "
              "they were measured in different states and are NOT comparable:")
        for pat in cal.bandwidth_patterns:
            print(f"    {pat.pattern:<8}{pat.sm_clock_start_mhz:>6} -> "
                  f"{pat.sm_clock_end_mhz:<6} MHz")

    c = cal.clocks
    print(f"\n  clocks            {c['sm_start_mhz']} -> {c['sm_end_mhz']} MHz, "
          f"{c['temp_start_c']} -> {c['temp_end_c']} C"
          + ("  THROTTLED" if c["throttled"] else ""))
    if c["throttled"]:
        print("                    ceilings measured on a throttling GPU are low; "
              "let it cool and re-run")

    # A write figure at or above datasheet peak means the byte accounting is
    # wrong (a read-for-ownership would make real traffic 2N), not that the
    # hardware exceeded its specification.
    if pin:
        w = cal.pattern("write")
        if w and w.gbps > pin * 0.93:
            print(f"\n  NOTE: write is {100 * w.gbps / pin:.1f}% of the derived "
                  "pin rate. That is a REAL store rate, not an accounting error: "
                  "a read-for-ownership would make it 2N, which would exceed the "
                  "pin rate by ~94% and is impossible. Write is not the ceiling "
                  "because it is the least representative pattern for a "
                  "read-dominated workload.")

    sha, dirty = git_provenance()
    # ONE PROVENANCE BLOCK, the shared one, so this yaml answers the same
    # questions a cells.csv and a report.json answer: which commit, which card,
    # which driver, which ruler, and WHICH INSTRUMENT. The 26 published reports
    # carried none of it and the H200 s4 filenames could not be tied to any
    # commit at all.
    prov = provenance_block(
        instrument=instrument_name(),
        bandwidth=cal.achieved_bandwidth_gbps,
        bandwidth_source=f"measured here, '{cal.ceiling_pattern}' pattern",
        ridge=round(cal.ridge_point(), 1),
        ridge_source="derived from this run's own ceilings",
        target_ms=None,
    )
    payload = {
        "name": f"{cal.gpu_name} (measured)",
        "verified": True,
        "source": "measured on this machine by scripts/calibrate_hardware.py",
        "source_note": (
            "Achievable ceilings, not datasheet peaks. Nsight Compute is "
            "unavailable on a rented pod (ERR_NVGPUCTRPERM), so DRAM traffic "
            "cannot be measured directly; these ceilings are what make the "
            f"efficiency columns meaningful without counters. Bandwidth is the "
            f"'{cal.ceiling_pattern}' pattern; every pattern measured is under "
            "`detail` so a different denominator can be applied without "
            "re-running."),
        "checked_by": "scripts/calibrate_hardware.py",
        "checked_on": time.strftime("%Y-%m-%d"),
        "measured_commit": sha,
        "measured_dirty": dirty,
        "memory": {"bandwidth_tb_s": cal.achieved_bandwidth_gbps / 1000.0},
        # fp8 keys appear ONLY when the card measured one. Absent is the right
        # answer on Ampere, which has no fp8 tensor cores; a key there would
        # give `roofline` a verified peak for a format the silicon cannot run.
        "compute_dense_tflops": {
            "bf16": cal.achieved_bf16_tflops,
            "fp16": cal.achieved_bf16_tflops,
            **({"fp8_e4m3": cal.achieved_fp8_tflops,
                "fp8_e5m2": cal.achieved_fp8_tflops}
               if cal.achieved_fp8_tflops else {}),
        },
        "detail": cal.as_dict(),
        "observed": {**observed, "power_limit_w": tdp},
        "spec_comparison": {"profile": profile, "bandwidth_gbps": spec_bw,
                            "bf16_tflops": spec_tf},
        "provenance": prov.as_dict(),
    }
    # One calibration file per device, on an UNTRACKED path. A single shared
    # measured.yaml meant calibrating a second GPU overwrote the first, and a
    # later re-plot of an earlier sweep then scored it against the wrong roof;
    # writing into the tracked tree meant every row measured afterwards carried
    # git_dirty=True. The run id keeps two settings of the same card apart.
    out = args.out if args.out is not None else session_out(args, cal.gpu_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    cells = write_cells(out.parent / "cells.csv", cal, prov)
    print(f"\n[calibrate] wrote {out}")
    print(f"[calibrate] wrote {cells}")
    print(f"  {mde_line(sigma_pct, 3, why)}")

    published_to = None
    if args.publish:
        published_to = HARDWARE_DIR / f"{measured_slug(cal.gpu_name)}.yaml"
        published_to.parent.mkdir(parents=True, exist_ok=True)
        published_to.write_text(out.read_text())
        print(f"[calibrate] PUBLISHED to {published_to}")
        print("  That file is TRACKED. roofline.load_measured() reads it, so the "
              "sweep now has ceilings; until you commit it, every row measured "
              "carries git_dirty=True and cannot be reproduced from the commit "
              "it names.")
    else:
        print("  Not in the tree: roofline.load_measured() will not see it and "
              "the sweep's efficiency columns stay empty. Re-run with --publish "
              "when you want the sweep to use it, and commit the result.")

    band = cal.ridge_band()
    print(json.dumps({"achieved_bw_gbps": cal.achieved_bandwidth_gbps,
                      "ceiling_pattern": cal.ceiling_pattern,
                      "achieved_bf16_tflops": cal.achieved_bf16_tflops,
                      "ridge": round(cal.ridge_point(), 1),
                      # The band, not just the point, so a consumer that greps
                      # this line cannot quote one end without seeing the other.
                      "ridge_band": [round(band[0], 1), round(band[1], 1)]
                      if band else None,
                      "gemm_clock_mhz": cal.gemm_clock_mhz,
                      "clock_established": cal.clock_established,
                      "out": str(out),
                      "published_to": str(published_to) if published_to else None,
                      "instrument": prov.instrument}))

    # THE ONLY LINES THE DRIVER MAY GREP. Everything above is prose, including
    # the words PASS and REFUSED where they appear in it.
    print()
    gates = score(cal, pin)
    for kind, name, verdict, detail in gates:
        print(EX.result_line(kind, name, verdict, detail))
    return EX.classify([(k, v) for k, _n, v, _d in gates])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:                                   # noqa: BLE001
        # ERROR is the table's word for an exception nobody planned; the
        # traceback is the reason and it goes to stderr, not into an exit code
        # that would read as a result.
        import traceback

        traceback.print_exc()
        raise SystemExit(EX.ERROR) from None
