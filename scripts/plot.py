#!/usr/bin/env python
"""Turn a results directory into figures. Reads CSVs only; never touches a GPU.

Three views, each answering a question the project actually asks:
  roofline     is this kernel memory bound or compute bound, and how close to
               the roof does it get at the intensity its tiling implies
  scaling      latency against token count, which is where the decode wall shows
  imbalance    latency against measured load skew at fixed token count, which is
               the load-imbalance story stated as a curve
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# THE SHIM EVERY OTHER SCRIPT HAS. Without it `.venv/bin/python scripts/plot.py`
# fails at the `moe` import unless the repo happens to be installed or
# PYTHONPATH happens to be set, and `docs/STUDY.md` lists this script as
# runnable "anywhere" while naming neither (audit R19/B11).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # must precede pyplot; no display on a headless pod

import matplotlib.pyplot as plt  # noqa: E402

from moe.bench import roofline as RL  # noqa: E402
from moe.bench.schema import passed, read_csv, row_bool  # noqa: E402
from moe.bench.schema import row_float as read_float  # noqa: E402
from moe.bench.schema import series_label as facet  # noqa: E402


def _f(row, key, default=0.0):
    return read_float(row, key, default)


def load_rows(results: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results.glob("*.csv")):
        if path.name == "merged.csv":
            continue
        try:
            rows.extend(read_csv(path))
        except (ValueError, OSError) as e:
            print(f"[plot] skipping {path.name}: {e}")
    merged = results / "merged.csv"
    if merged.exists() and not rows:
        rows = read_csv(merged)
    return [r for r in rows if passed(r)]


def plot_scaling(rows, out_dir: Path, dtype: str = "bf16"):
    """Latency against token count, one line per implementation, per model."""
    by_model = defaultdict(list)
    for r in rows:
        if r["dtype"] == dtype and r["scope"] == "span" and _f(r, "ms_p50") > 0:
            by_model[r["model"]].append(r)

    written = []
    for model, mrows in sorted(by_model.items()):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        by_impl = defaultdict(list)
        for r in mrows:
            by_impl[f"{facet(r)} ({r['routing_kind']})"].append(r)
        for label, series in sorted(by_impl.items()):
            series.sort(key=lambda r: int(float(r["num_tokens"])))
            xs = [int(float(r["num_tokens"])) for r in series]
            ys = [_f(r, "ms_p50") for r in series]
            ax.loglog(xs, ys, "o-", ms=4, lw=1.3, alpha=0.85, label=label)
        ax.set_xlabel("tokens entering the layer (1 = decode step)")
        ax.set_ylabel("p50 latency (ms)")
        ax.set_title(f"{model}: grouped-GEMM latency vs token count, {dtype}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / f"scaling_{model}_{dtype}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def plot_imbalance(rows, out_dir: Path, dtype: str = "bf16"):
    """Latency against measured load skew, at each token count."""
    pts = [r for r in rows
           if r["dtype"] == dtype and r["scope"] == "span"
           and _f(r, "ms_p50") > 0 and _f(r, "load_max_over_mean") > 0]
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(7.5, 5))
    by_impl = defaultdict(list)
    for r in pts:
        by_impl[facet(r)].append(r)
    for impl, series in sorted(by_impl.items()):
        xs = [_f(r, "load_max_over_mean") for r in series]
        ys = [_f(r, "ms_p50") for r in series]
        ax.loglog(xs, ys, "o", ms=5, alpha=0.8, label=impl)
    ax.set_xlabel("expert load imbalance (max rows / mean rows)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title(f"Sensitivity to routing skew, {dtype}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / f"imbalance_{dtype}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def plot_l2_absorption(rows, out_dir: Path, dtype: str = "bf16"):
    """Traffic the cache absorbed, inferred from the flush axis.

    Nsight Compute is unavailable on a rented pod, so there is no cache hit-rate
    counter to read. But every cell is already timed twice, once with L2 flushed
    and once warm, and the time difference times achievable bandwidth estimates
    the bytes L2 served instead of DRAM. Free, from an axis already swept.
    """
    from moe.bench.calibrate import l2_absorbed_bytes

    paired = {}
    for r in rows:
        if r.get("dtype") != dtype or r.get("scope") != "span":
            continue
        if row_bool(r, "cuda_graph"):
            continue
        key = (r["impl"], r["model"], r["num_tokens"], r["routing_kind"],
               r["routing_param"], r["seed"])
        paired.setdefault(key, {})[row_bool(r, "l2_flush")] = r

    pts = []
    for key, both in paired.items():
        cold, warm = both.get(True), both.get(False)
        if not cold or not warm:
            continue
        bw = _f(cold, "achieved_bw_gbps") * 1e9
        if bw <= 0:
            continue
        absorbed = l2_absorbed_bytes(_f(cold, "ms_p50"), _f(warm, "ms_p50"), bw)
        pts.append((key[0], int(float(key[2])), absorbed,
                    _f(cold, "compulsory_bytes")))
    if not pts:
        return []

    fig, ax = plt.subplots(figsize=(7.5, 5))
    by_impl = defaultdict(list)
    for impl, tokens, absorbed, compulsory in pts:
        by_impl[impl].append((tokens, absorbed / max(compulsory, 1)))
    for impl, series in sorted(by_impl.items()):
        series.sort()
        ax.semilogx([t for t, _ in series], [v for _, v in series], "o-",
                    ms=4, lw=1.3, label=impl)
    ax.set_xlabel("tokens entering the layer")
    ax.set_ylabel("L2-absorbed traffic / compulsory traffic")
    ax.set_title(f"Cache absorption inferred from the flush axis, {dtype}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / f"l2_absorption_{dtype}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def dtypes_present(rows, requested: str | None = None) -> list[str]:
    """Which dtypes to draw figures for.

    `requested` wins when given. Otherwise every dtype actually in the rows,
    sorted for stable filenames.

    The default used to be a bare "bf16", and run_all.sh never overrode it, so
    an all-fp8 sweep published eight `*_bf16.png` figures drawn from whatever
    bf16 rows were left in the results directory. Nothing crashed and the names
    were honest; the reader just found charts about a different format sitting
    inside an fp8 result set.
    """
    have = sorted({r["dtype"] for r in rows if r.get("dtype")})
    if requested is None:
        return have
    return [requested] if requested in have else []


#: The calibration `publish_results.sh` copies in beside an arm's CSVs.
ARM_CALIBRATION = "measured.yaml"


def arm_profile(results: Path) -> str | None:
    """The profile NAME for the calibration sitting beside these CSVs, or None.

    THE BUG THIS FIXES. `main` used to call `RL.load_measured()` with no
    argument, which reads the calibration of the GPU ATTACHED TO THIS PROCESS.
    On a laptop that returns None and the script fell through to
    `RL.for_device`, which on "NVIDIA H200" is ambiguous between h200_sxm and
    h200_nvl and so refused: `plot.py --results results/published/<arm>` did not
    run as documented, and the only thing that worked was an undocumented
    `--hardware measured_nvidia_h200`. On a pod it is worse than a refusal: the
    attached card's ceilings would silently roof another arm's rows. Meanwhile
    the arm's OWN `measured.yaml` was sitting next to the CSVs being read, and
    for the whole-layer and fp8-refixed arms it differs from the committed
    per-device file by 7.6% in bf16 compute (audit R14/B11).

    HOW THE NAME REACHES THE ROOF, because it looks like a coincidence and is
    not. `roofline.load_hardware` builds its path as
    `(directory or HARDWARE_DIR) / f"{name}.yaml"`, and `roofline.plot` offers
    no `directory`. Joining an ABSOLUTE right-hand side onto any path yields the
    absolute path, by pathlib's documented rule, so an absolute path stem passed
    as the profile name resolves to exactly that file. It is the only seam
    `roofline.plot` gives a calibration that does not live in the hardware
    directory, and the resolution is pinned by
    `tests/test_analysis_tools.py::test_arm_profile_resolves_the_arms_own_yaml`.
    """
    cal = (results / ARM_CALIBRATION).resolve()
    return str(cal.with_suffix("")) if cal.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    ap.add_argument("--dtype", default=None,
                    help="one dtype; default draws every dtype in the rows")
    ap.add_argument("--hardware", default=None,
                    help="hardware profile; OVERRIDES the arm's own "
                         f"{ARM_CALIBRATION}. Without it the calibration "
                         "published beside these CSVs is used, and a results "
                         "directory carrying none is REFUSED rather than "
                         "roofed against this machine's card")
    ap.add_argument("--allow-unverified-roof", action="store_true")
    args = ap.parse_args()

    if args.hardware is None:
        # THE ARM'S OWN RULER, NOT THIS MACHINE'S. The efficiency columns in
        # these CSVs were computed against the yaml published beside them, so
        # that is the only file whose roof the plotted points agree with.
        args.hardware = arm_profile(args.results)
        if args.hardware is None:
            print(f"[plot] REFUSING: {args.results} carries no "
                  f"{ARM_CALIBRATION}, so these rows have no published roof to "
                  "be drawn against.")
            print("[plot] The ceilings of whatever GPU happens to be attached "
                  "to this process are")
            print("[plot] not this arm's, and every efficiency number drawn "
                  "from them would be")
            print("[plot] wrong by the ratio of two calibrations. Pass "
                  "--hardware to assert one")
            print(f"[plot] yourself; available: {RL.available_profiles()}.")
            return 1
        print(f"[plot] roof: {args.results / ARM_CALIBRATION}, the calibration "
              "published with these rows")

    rows = load_rows(args.results)
    if rows:
        measured_on = {r.get("gpu_name", "") for r in rows} - {""}
        try:
            hw = RL.hardware_for_rows(args.hardware, rows, allow_unverified=True)
            mismatched = [g for g in measured_on if not RL.device_matches(hw, g)]
            if mismatched:
                print(f"[plot] REFUSING: rows were measured on {mismatched} but "
                      f"--hardware is {hw.name!r}. Plotting against the wrong "
                      f"part misstates every efficiency number.")
                print("[plot] run scripts/calibrate_hardware.py on the box, or "
                      "pass --hardware explicitly.")
                return 1
        except (RL.UnverifiedHardware, FileNotFoundError, ValueError) as e:
            # This guard exists to REFUSE a mismatched roof, so a failure to
            # resolve one at all must not pass silently: that is how a missing
            # calibration got past here and killed the script forty lines later,
            # after a three-hour sweep.
            print(f"[plot] could not resolve {args.hardware!r} for the "
                  f"device-match check: {e}")
    if not rows:
        print(f"[plot] no correctness-passing rows in {args.results}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    written = []

    draw = dtypes_present(rows, args.dtype)
    if not draw:
        print(f"[plot] no rows in dtype {args.dtype!r}; nothing to draw")
        return 1
    if args.dtype is None and len(draw) > 1:
        print(f"[plot] drawing every dtype present: {', '.join(draw)}")

    for dtype in draw:
        try:
            written.append(RL.plot(rows, args.out / f"roofline_{dtype}.png",
                                   hardware=args.hardware, dtype=dtype,
                                   allow_unverified=args.allow_unverified_roof))
        except (RL.UnverifiedHardware, ValueError) as e:
            print(f"[plot] roofline skipped for {dtype}: {e}")

        written += plot_scaling(rows, args.out, dtype)
        written += plot_imbalance(rows, args.out, dtype)
        written += plot_l2_absorption(rows, args.out, dtype)

    for p in written:
        print(f"[plot] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
