#!/usr/bin/env python3
"""Where does a sweep actually cross its ridge, and does C2 predict it?

Reads one or more result CSVs, aggregates to one median per (model, dtype,
token count), recovers the crossing from measured TIME via moe.bench.crossing,
and prints it beside the `2R/b` prediction.

The measured side never consults the byte model, so the prediction can be wrong.

    python scripts/crossing_report.py /workspace/results/run_h200fp8b_vllm.csv \
        --ridge 160.3 --impl vllm_fused_experts

`--uncertainty` adds a 90% band, propagated from the replicate spread each token
count already carries. Off by default so existing output is unchanged, but every
crossing this study has quoted was quoted bare, and on a flat curve the
interpolation multiplies a 6% timing wobble into a 35% move in the answer.

EVERY crossing is printed, not the first. 8 of the 16 canonical uniform cells
cross 0.5 going up more than once, because the curve is a staircase in M-tiles
per expert rather than one flat-to-linear transition, and a cell that crosses
twice is labelled a staircase here rather than reduced to whichever step the
token grid sampled first. The M-tile count sits beside every token count so the
steps are visible in the table itself; `--block-m` names the tile it is counted
at, since the published arms predate the column that records the tile actually
run.

`--max-affine` adds the OTHER estimator beside this one: TEMPO's global
`t = max(a + b G, c + beta N)`, which has one inflection by construction and
therefore cannot have the multiple-crossing failure the slope threshold has. Off
by default, and the default output is byte for byte what it was without it. Read
`moe/bench/maxaffine.py` before quoting the number it prints: on the five-stage
cells it fits to a mean relative error of 24% and lands three times below the
ridge, so it removes the ambiguity without resolving it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench.crossing import (  # noqa: E402
    DEFAULT_DRAWS,
    ROUTING_COLUMN,
    STORED_TILE_EFF,
    crossing_interval,
    local_slopes,
    m_tiles_for_row,
    routing_domain,
    timed_rows,
    upcrossings,
)
from moe.bench.maxaffine import (  # noqa: E402
    PADDED_ROWS,
    RELATIVE,
    TOKENS,
    WEIGHTINGS,
    Comparison,
    MaxAffineFit,
    fit_rows,
)
from moe.bench.published import (  # noqa: E402
    filter_superseded,
    superseded_impls,
    superseded_reason,
)
from moe.bench.ridge import (  # noqa: E402
    crossing_batch,
    ridge_for_dtype,
    rows_per_expert,
    saturation_batch,
)
from moe.bench.schema import TileConfigUnrecorded  # noqa: E402
from moe.routing.imbalance import TileEfficiencyUndetermined  # noqa: E402


def tile_cell(counted: list[float] | None, previous: float | None) -> str:
    """One table cell: the median M-tile count, its step, and its disagreement.

    Medianed across replicates the same way `ms_p50` is, because the two have to
    describe the SAME cell: a tile count taken off one row beside a time taken
    off eight would put a step where the timed curve has none. Uniform routing
    is sampled per replicate, so the counts genuinely differ -- mixtral at
    T=1024 draws 19 tiles on two rows and 21 on four -- and `~` says so rather
    than letting a rounded median look exact.

    `--` when nothing determined it, never a blank: an empty cell in a column of
    numbers reads as zero steps, which is the claim this column exists to test.
    """
    if not counted:
        return "--"
    median = statistics.median(counted)
    spread = "~" if len({round(v) for v in counted}) > 1 else ""
    step = ""
    if previous is not None and round(median) != round(previous):
        step = f" ({round(median) - round(previous):+d})"
    return f"{round(median)}{spread}{step}"


def print_staircase(found: list, predicted: float,
                    cell_tiles: dict[int, list[float]]) -> None:
    """Every crossing, and what a cell with more than one of them is.

    Silent for a cell that crosses once, so the output of a single-crossing run
    keeps its shape. Loud for the rest, because the alternative -- printing one
    number off a curve that supplies several -- is how a tile step got quoted
    as a ridge crossing for the length of this study.

    Names no winner. The last crossing has the better claim on the ridge (rows
    per expert at the last: mean 175.8, CV 21.2% against the measured ridge band
    160.3-176.2; at the first: 123.4 and CV 40.0%), and choosing on that here
    would bake a preference into a report whose job is to show that the sweep
    does not resolve it. A dense token grid resolves it.
    """
    if len(found) < 2:
        return
    print(f"  STAIRCASE: the slope crosses 0.5 upward at {len(found)} token "
          "counts, not one.")
    for i, u in enumerate(found, 1):
        lo = tile_cell(cell_tiles.get(int(u.step_lo)), None)
        hi = tile_cell(cell_tiles.get(int(u.step_hi)), None)
        print(f"    {i} of {len(found)}: {u.tokens:8.0f} tokens   "
              f"{u.tokens / predicted:.2f}x predicted   "
              f"on the step T {u.step_lo:.0f} -> {u.step_hi:.0f}, "
              f"M-tiles {lo} -> {hi}")
    print(f"    last over first: {found[-1].tokens / found[0].tokens:.2f}x. "
          "M-tiles per expert is a step")
    print("    function of T, and each extra tile is another pass over that "
          "expert's weights,")
    print("    so the slope spikes above 0.5 at every step and sags below it "
          "on every tread.")
    print("    A first crossing can be a tile step rather than a roofline "
          "transition; which")
    print("    of these is the ridge is not decided by this grid.")


#: The only columns `moe.bench.maxaffine` reads. Projected rather than keeping
#: the whole row for the same reason the routing census is: a published row is
#: 94 columns and there are up to 70k of them, and this script already refuses
#: to hold them. Nothing is retained at all unless `--max-affine` is passed.
AFFINE_COLUMNS = ("num_tokens", "ms_p50", "load_active_experts",
                  "load_total_rows", "load_max_rows",
                  "load_tile_eff_bm64", "load_tile_eff_bm128")


def print_max_affine(fit: MaxAffineFit | None, comparison: Comparison,
                     model: str, predicted: float) -> dict | None:
    """The second estimator's answer for this cell, with its residuals attached.

    Prints the fit quality on the SAME lines as the inflection, never below a
    fold and never optional. A max-affine inflection is only worth more than the
    slope detector's several if the model describes the curve, and on the
    five-stage cells here it misses by a mean of 24% and a p95 of up to 263%. A
    reader who copies one line out of this report has to copy the residual with
    it, which is the lesson of every retraction in docs/FINDINGS.md.

    Returns the row the head-to-head table is built from, or None when there was
    no fit, so the summary is assembled from the same numbers that were printed
    rather than recomputed beside them.
    """
    if fit is None:
        print("  MAX-AFFINE: not fitted. Fewer than four token counts, or a "
              "padded regressor")
        print("    this row set cannot determine at the named BLOCK_M.")
        return None
    mem, comp = fit.memory, fit.compute
    n = len(fit.observations)
    b = ("unidentifiable, G never moved here" if mem.degenerate
         else f"{mem.slope:.6f} ms per active expert")
    beta = ("unidentifiable, N never moved here" if comp.degenerate
            else f"{comp.slope:.8f} ms per unit N")
    print("  MAX-AFFINE (TEMPO-style global fit): t = max(a + b G, c + beta N)")
    print(f"    G = active experts (observed column); {fit.compute_side.label}")
    print(f"    residuals weighted {fit.weighting}, split searched over all "
          f"{n - 3} contiguous partitions")
    print(f"    memory  branch:  a = {mem.intercept:9.4f} ms   b    = {b}"
          f"   [{fit.n_memory} of {n} points]")
    print(f"    compute branch:  c = {comp.intercept:9.4f} ms   beta = {beta}"
          f"   [{fit.n_compute} of {n} points]")
    inflection = fit.inflection
    rpe = None
    if inflection is None:
        print(f"    inflection:      none on this grid "
              f"({len(fit.inflections)} upward crossings)")
    else:
        rpe = rows_per_expert(model, inflection)
        print(f"    inflection:      {inflection:8.0f} tokens   "
              f"{inflection / predicted:.2f}x predicted   "
              f"{rpe:.1f} rows per expert")
    print(f"    fit quality:     mean |rel err| {fit.mean_rel_err:.1%}   "
          f"p95 {fit.p95_rel_err:.1%}   max {fit.max_rel_err:.1%}")
    if not fit.single_crossing:
        print(f"    NOT SINGLE-CROSSING: the fitted max also crosses back "
              f"{fit.reversals} time(s) at")
        print("      small T, where G rises steeply and N has barely moved. "
              "One inflection is a")
        print("      property of two planes, not of the path a sweep walks "
              "through them.")
    found = comparison.slope_crossings
    if found:
        listed = ", ".join(f"{t:.0f}" for t in found)
        print(f"    against the slope detector: {len(found)} crossing(s) "
              f"({listed})")
        first, last = comparison.ratio_to(comparison.first), \
            comparison.ratio_to(comparison.last)
        if first is not None:
            print(f"      max-affine is {first:.2f}x the first and "
                  f"{last:.2f}x the last")
    return {"first": comparison.first, "last": comparison.last,
            "n_crossings": len(found), "inflection": inflection, "rpe": rpe,
            "ratio_first": comparison.ratio_to(comparison.first),
            "ratio_last": comparison.ratio_to(comparison.last),
            "mean": fit.mean_rel_err, "p95": fit.p95_rel_err,
            "single": fit.single_crossing}


def print_head_to_head(summary: list[tuple[str, dict]]) -> None:
    """Both estimators for every cell in one table, then what they disagree by.

    The per-cell blocks above are 40 lines apart, and a reader comparing two
    estimators across sixteen cells cannot hold them. This is the same numbers
    in one place -- assembled from the dicts those blocks returned, so the table
    and the blocks cannot drift.

    Reports the disagreement as a distribution rather than a mean, because the
    interesting part is its width: the two estimators agree within 20% on the
    smooth one-stage curves and differ by up to 5x on the stepped five-stage
    ones, and a single average over both would hide exactly that.
    """
    print("=== ESTIMATOR HEAD TO HEAD: slope threshold against max-affine ===")
    print("  The slope detector's crossings (all of them, first and last shown) "
          "against the")
    print("  max-affine inflection, on the SAME medians. R is rows per expert "
          "at the")
    print("  inflection, T k / E. `1x` is whether the fitted max crosses this "
          "grid once.")
    print(f"  {'cell':44} {'first':>7} {'last':>7} {'n':>2} "
          f"{'affine':>7} {'x1st':>6} {'xlast':>6} {'R':>7} "
          f"{'mean%':>6} {'p95%':>7}  1x")
    for label, row in summary:
        def num(value, fmt):
            return "--" if value is None else format(value, fmt)
        print(f"  {label:44} {num(row['first'], '7.0f'):>7} "
              f"{num(row['last'], '7.0f'):>7} {row['n_crossings']:>2} "
              f"{num(row['inflection'], '7.0f'):>7} "
              f"{num(row['ratio_first'], '6.2f'):>6} "
              f"{num(row['ratio_last'], '6.2f'):>6} "
              f"{num(row['rpe'], '7.1f'):>7} "
              f"{row['mean'] * 100:6.1f} {row['p95'] * 100:7.1f}  "
              f"{'yes' if row['single'] else 'no'}")
    rows = [r for _, r in summary]
    ambiguous = [r for r in rows if r["n_crossings"] > 1]
    resolved = [r for r in ambiguous if r["inflection"] is not None]
    print(f"\n  {len(ambiguous)} of {len(rows)} cells give the slope detector "
          f"more than one crossing.")
    print(f"  Max-affine returns exactly one inflection on {len(resolved)} of "
          f"those, and is")
    print(f"  single-crossing along the grid on "
          f"{sum(1 for r in rows if r['single'])} of {len(rows)} cells overall.")
    for name, key in (("first", "ratio_first"), ("last ", "ratio_last")):
        vals = sorted(r[key] for r in rows if r[key] is not None)
        if vals:
            print(f"  max-affine over the {name} crossing: median "
                  f"{statistics.median(vals):.2f}, range {vals[0]:.2f} to "
                  f"{vals[-1]:.2f} over {len(vals)} cells")
    means = sorted(r["mean"] for r in rows)
    p95s = sorted(r["p95"] for r in rows)
    print(f"  fit quality across cells: mean |rel err| median "
          f"{statistics.median(means):.1%} (range {means[0]:.1%} to "
          f"{means[-1]:.1%}),")
    print(f"  p95 median {statistics.median(p95s):.1%} (range {p95s[0]:.1%} to "
          f"{p95s[-1]:.1%}). A fit that misses by that much is not locating a "
          "ridge.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--ridge", type=float, required=True,
                    help="measured bf16 FLOP/byte from the calibration; other "
                         "dtypes are scaled by their FLOP rate, since a format "
                         "changes the peak as well as the bytes")
    ap.add_argument("--impl", default=None, help="restrict to one implementation")
    ap.add_argument("--routing", default=None, help="restrict to one routing kind")
    ap.add_argument("--include-throttled", action="store_true")
    ap.add_argument("--l2-flush", choices=["true", "false"], default=None,
                    help="restrict to one L2 mode; default mixes both")
    ap.add_argument("--cuda-graph", choices=["true", "false"], default=None,
                    help="restrict to one capture mode; default mixes both")
    ap.add_argument("--block-m", type=int, default=128,
                    help="BLOCK_M the M-tile column is counted at (default "
                         "128). The published arms predate the column that "
                         "records the tile actually run, so this is a stated "
                         "assumption rather than a measurement; 64 and 128 read "
                         "a stored efficiency, any other value is "
                         "reconstructed and refuses once an expert spans tiles")
    ap.add_argument("--max-affine", action="store_true",
                    help="also fit TEMPO's global t = max(a + b G, c + beta N) "
                         "per cell and print both estimators side by side. Off "
                         "by default and the default output is unchanged; this "
                         "estimator has one inflection by construction, which "
                         "is the reason to try it and the reason to check its "
                         "residuals before quoting it")
    ap.add_argument("--max-affine-n", choices=[TOKENS, PADDED_ROWS],
                    default=TOKENS,
                    help="the compute-side regressor. `tokens` is TEMPO's own "
                         "and is a recorded column; `padded_rows` rounds up to "
                         "whole M-tiles at --block-m, fits better, and is "
                         "DERIVED -- below saturation it equals BLOCK_M x "
                         "active experts exactly, so the two branches go "
                         "collinear and the inflection stops meaning anything")
    ap.add_argument("--max-affine-weighting", choices=list(WEIGHTINGS),
                    default=RELATIVE,
                    help="`relative` weights each residual by 1/ms^2 so the "
                         "objective is relative error; `absolute` is textbook "
                         "least squares, which on a curve spanning 0.13 to "
                         "27.9 ms is a fit to the largest token count alone")
    ap.add_argument("--uncertainty", action="store_true",
                    help="add a 90%% band to each measured crossing, Monte "
                         "Carlo'd from the replicate spread of each token "
                         "count. Off by default so existing output is byte for "
                         "byte what it was")
    args = ap.parse_args()

    # A superseded arm holds the SAME measurements as the one that replaced it,
    # so reading both weights every one of its rows twice. Announced rather than
    # silent: a dropped input nobody sees is the same class of error.
    csvs, dropped = filter_superseded(args.csvs)
    for d in dropped:
        print(f"[skip] {d.parent.name}: {superseded_reason(d).splitlines()[0]}")
    partial = {}
    for c in csvs:
        names = superseded_impls(c)
        if names:
            partial[c] = names
            print(f"[skip] {c.parent.name}: {', '.join(sorted(names))} "
                  f"({superseded_reason(c).splitlines()[0]})")
    if dropped or partial:
        print()
    if not csvs:
        print("every input was superseded; nothing to report")
        return 1

    cells: dict[tuple[str, str], dict[int, list[float]]] = {}
    # One tiny mapping per kept row, so `routing_domain` sees rows and the
    # census is of what actually reached a median rather than of what the file
    # holds. Only the routing column is kept: the full rows are 200 columns
    # wide and there are up to 70k of them across the published arms.
    routing_rows: dict[tuple[str, str, str], list[dict]] = {}
    # M-tiles per row, reduced to a number at ingest for the same reason: the
    # count is one float and the row it came from is 200 columns.
    tiles: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    # Projected rows for the max-affine fit, and only when it was asked for:
    # with the flag off nothing here is populated and the report allocates
    # exactly what it did before.
    affine: dict[tuple[str, str, str], dict[int, list[dict]]] = {}
    modes: collections.Counter = collections.Counter()
    kept = skipped = untimed = tileless = 0
    for path in csvs:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
            timed = timed_rows(rows)
            untimed += len(rows) - len(timed)
            skip_impls = partial.get(path, set())
            for r in timed:
                if r["impl"] in skip_impls:
                    skipped += 1
                    continue
                if args.impl and r["impl"] != args.impl:
                    continue
                if args.routing and r["routing_kind"] != args.routing:
                    continue
                if r.get("correctness_passed") not in ("True", "true", "1", ""):
                    skipped += 1
                    continue
                if not args.include_throttled and r.get("throttled") in ("True", "true", "1"):
                    skipped += 1
                    continue
                if args.l2_flush is not None and \
                        str(r.get("l2_flush", "")).lower() != args.l2_flush:
                    continue
                if args.cuda_graph is not None and \
                        str(r.get("cuda_graph", "")).lower() != args.cuda_graph:
                    continue
                modes[(str(r.get("l2_flush")), str(r.get("cuda_graph")))] += 1
                try:
                    t, ms = int(r["num_tokens"]), float(r["ms_p50"])
                except (ValueError, KeyError):
                    continue
                # `impl` is in the key, not just an optional filter. Rows from
                # different implementations measure different SCOPES -- one
                # stage against five against a whole layer, 16.7x apart on the
                # published sweep -- so a median across them describes nothing.
                key = (r["model"], r["dtype"], r["impl"])
                cells.setdefault(key, {}).setdefault(t, []).append(ms)
                routing_rows.setdefault(key, []).append(
                    {ROUTING_COLUMN: r.get(ROUTING_COLUMN, "")})
                if args.max_affine:
                    affine.setdefault(key, {}).setdefault(t, []).append(
                        {c: r.get(c, "") for c in AFFINE_COLUMNS})
                try:
                    tiles.setdefault(key, {}).setdefault(t, []).append(
                        m_tiles_for_row(r, args.block_m))
                except (TileEfficiencyUndetermined, TileConfigUnrecorded,
                        KeyError, ValueError):
                    # A row with no load columns, or one whose experts span
                    # more tiles than the stored efficiencies pin down. Counted
                    # and announced: an empty column with no explanation reads
                    # as "no steps here", which is the opposite of the truth.
                    tileless += 1
                kept += 1

    print(f"kept {kept} rows, skipped {skipped} (throttled or failed), "
          f"{untimed} never timed (skipped graph mode: ms_p50 is 0.0, "
          f"which is not a measurement)")
    if len(modes) > 1:
        print("  timing modes mixed into each median (l2_flush, cuda_graph): "
              + ", ".join(f"{k}x{v}" for k, v in sorted(modes.items())))
        print("  pass --l2-flush/--cuda-graph to isolate one; the crossing is a "
              "slope, so mixing adds spread rather than bias")
    # The staircase is the reason a crossing can be a tile step, so the column
    # that shows it gets explained once, above the cells, rather than being an
    # unlabelled number a reader has to reverse-engineer.
    stored = " (a stored column)" if args.block_m in STORED_TILE_EFF else \
             " (reconstructed)"
    print(f"  M-tiles at BLOCK_M {args.block_m}{stored}: "
          "ceil(rows_per_expert / BLOCK_M) summed")
    print("  over active experts. Each extra tile is another pass over that "
          "expert's weights,")
    print("  so time steps where the count steps and flatlines where it holds. "
          "(+n) is the")
    print("  step from the row above; ~ marks replicates that drew different "
          "tile counts.")
    if tileless:
        print(f"  {tileless} rows carry no usable M-tile count and are absent "
              "from that column only:")
        print("  a row records no routing load, or an expert spans more tiles "
              "than the stored")
        print("  efficiencies pin down, and a reconstructed count there would "
              "be a guess.")
    if args.uncertainty:
        print(f"  bands are {DEFAULT_DRAWS} draws per cell, each token count "
              f"shaken by its own replicate spread; the crossing interpolates "
              f"between")
        print("  two slopes with leverage 1/(s1-s0), so it is widest exactly "
              "where the curve is flattest")
    # Louder than the timing-mode note above, and deliberately not a change of
    # default: pooled rows do not make the crossing noisier, they make it a
    # crossing of nothing. Silently switching to `--routing uniform` here would
    # make this report answer differently than it did yesterday with no flag
    # changed, which is its own failure mode -- so it warns and names the flag.
    routing = routing_domain(itertools.chain.from_iterable(routing_rows.values()))
    banner = routing.warning_lines()
    if banner:
        print()
        for line in banner:
            print(line)
    print()
    if not cells:
        print("nothing to report")
        return 1

    summary: list[tuple[str, dict]] = []
    for key_, by_t in sorted(cells.items()):
        model, dtype, impl = key_
        # The replicate lists survive to here rather than being medianed away at
        # the top of the loop: the band needs each token count's own scatter,
        # and a median has thrown that away.
        replicates = sorted(by_t.items())
        points = [(t, statistics.median(v)) for t, v in replicates]
        print(f"=== {model} / {dtype} / {impl} ===")
        slopes = dict(local_slopes(points))
        cell_tiles = tiles.get(key_, {})
        print(f"  {'T':>6} {'ms_p50':>9} {'slope':>7} {'M-tiles':>12}   regime")
        prev = None
        prev_tiles = None
        for t, ms in points:
            s = next((v for k, v in slopes.items() if prev and prev < k < t), None)
            tag = "" if s is None else ("weight-bound" if s < 0.5 else "compute-bound")
            counted = cell_tiles.get(t)
            # Padded rather than empty on the first row: an unpadded blank used
            # to be invisible because the regime column was blank too, and now
            # it would shift the M-tile column left and hide the step.
            print(f"  {t:>6} {ms:>9.4f} {'' if s is None else f'{s:.3f}':>7} "
                  f"{tile_cell(counted, prev_tiles):>12}   {tag}")
            prev = t
            if counted:
                prev_tiles = statistics.median(counted)

        # Below E/k a batch misses experts, so weight traffic grows with the
        # batch and the slope crosses for a reason unrelated to the ridge.
        sat = saturation_batch(model)
        # Every upcrossing, not the first. The first is what the line below
        # still prints, because published figures were read off it, but a cell
        # that crosses twice gets said so in as many words underneath.
        found = upcrossings(points, min_tokens=sat)
        measured = found[0].tokens if found else None
        dtype_ridge = ridge_for_dtype(args.ridge, dtype)
        predicted = crossing_batch(model, dtype_ridge, dtype)
        print(f"\n  saturation (E/k, floor):             {sat:8.0f} tokens")
        print(f"  ridge for {dtype:<9}                  {dtype_ridge:8.1f} FLOP/byte")
        print(f"  predicted (2R/b at that ridge):      {predicted:8.0f} tokens")
        if measured is None:
            print("  measured:                            not bracketed by this "
                  "token grid")
            print("  -> add token counts on both sides of the prediction")
        else:
            ratio = measured / predicted
            # `[1 of n]` on the line itself, because this line is the one that
            # gets copied out of a report and into a table, and a reader who
            # copies it has to carry the ambiguity with it.
            of_n = f"   [1 of {len(found)}]" if len(found) > 1 else ""
            print(f"  measured (slope crosses 0.5):        {measured:8.0f} tokens"
                  f"   {ratio:.2f}x predicted{of_n}")
            # Beside the number, not only in the header. A banner forty lines
            # up does not stop this figure being quoted on its own, and every
            # crossing this study has retracted was quoted on its own.
            for line in routing_domain(routing_rows[key_]).crossing_note():
                print(f"  {line}")
            if args.uncertainty:
                band = crossing_interval(replicates, min_tokens=sat)
                if band is None:
                    # The medians bracket the crossing and no perturbed draw
                    # does, so the number above is an artefact of the spread it
                    # was quoted without.
                    print("  90% band (replicate noise):          no draw kept "
                          "the bracket; the crossing does not survive its own "
                          "noise")
                else:
                    _, lo, hi = band
                    print(f"  90% band (replicate noise):          {lo:8.0f} - "
                          f"{hi:.0f} tokens"
                          f"   {lo / predicted:.2f}-{hi / predicted:.2f}x "
                          f"predicted")
            print_staircase(found, predicted, tiles.get(key_, {}))
        # Outside the else: a cell whose grid does not bracket a slope crossing
        # is exactly the cell where a second estimator is worth having, and
        # printing it only where the first one succeeded would hide that.
        if args.max_affine:
            print()
            fit = fit_rows(affine.get(key_, {}), args.max_affine_n,
                           args.block_m, args.max_affine_weighting)
            comparison = Comparison(tuple(u.tokens for u in found), fit)
            row = print_max_affine(fit, comparison, model, predicted)
            if row is not None:
                summary.append((f"{model} / {dtype} / {impl}", row))
        print()
    if args.max_affine and summary:
        print_head_to_head(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
