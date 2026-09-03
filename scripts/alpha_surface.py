"""Read every block_m_crossing report under a root and print the alpha surface.

WHY THIS EXISTS. The 2026-09-01 session produced alpha from four routes and
reconciled the wrong two: it scored the PUBLISHED 0.558 against an ablation whose
own gates had failed, and never looked at the sweep's per-BLOCK_M fits -- the only
alpha values measured on that card. This reads the reports themselves, so the
comparison is over what was measured rather than over what was quoted.

The surface is alpha against the two things that set REUSE DISTANCE: the swizzle
width, which decides how many M-tiles share one weight read, and the per-expert
footprint, which decides how much else evicts it in between. Neither is "does the
expert fit in L2": qwen2's 37 MB expert fits in 60 MiB and still pays alpha 0.71.

TWO DEFECTS FIXED 2026-09-02, both of which made the tables unquotable rather
than wrong-by-a-little (audit S38). The glob could not read the layout the
published arms actually use, so this script printed "no report.json under ..."
on every one of them and the committed SURFACE.txt files were unregenerable.
And the lever levels were sorted as TEXT, so the paired-change line named
1 -> 8 as the swizzle's extremes when the sweep runs 1 -> 64. The committed
summaries were regenerated from the reports at the same time, and the pooled
versions they replace are kept beside them as SURFACE.pooled.txt.

Every paired comparison prints an MDE first, because a lever effect quoted
without one cannot be told from a lever effect this design could not have found
(B14). The sigma behind it comes from `replicate_noise_floor.sizing_sigma`,
which prefers a MEASURED between-replicate floor and falls back to the declared
s3/s4 proxy with the word ASSUMED. Both words are printed. Until 2026-09-03
this file read the proxy field out of the JSON itself, so a measured floor
published by the noise-floor arm would have changed nothing here and every MDE
below would have stayed sized against the assumption.

REGENERATE THE THREE COMMITTED SUMMARIES WHENEVER THE FLOOR CHANGES. That fix
made this script's output a function of `results/published/NOISE_FLOOR.json`,
and the three `SURFACE.txt` beside it are this script's output, committed. The
day part (a) publishes a measured floor they stop matching, and worse, they go
on carrying the sentence "part (a) has not run on a card yet" as PUBLISHED
EVIDENCE about a repository that by then holds the measurement, under a
detection limit 2.5x too loose wearing the word ASSUMED. Nothing regenerates
them automatically, so publishing the floor is a two-step act:

    .venv/bin/python scripts/replicate_noise_floor.py ... --publish
    for a in 2026-09-01-nvidia_h200-alpha-surface-s4 \
             2026-09-01-nvidia_h200-cross-card-s3 \
             2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3; do
      .venv/bin/python scripts/alpha_surface.py "results/published/$a" \
        > "results/published/$a/SURFACE.txt"
    done

`test_analysis_tools.test_the_committed_surface_files_regenerate_byte_for_byte`
is what fails if the second step is skipped. It is the right alarm and it is
not a false one: a published summary a stranger cannot rebuild from this
repository is not published evidence, which is the property that test holds.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# THE SHIM EVERY OTHER SCRIPT HAS. Without it `python scripts/alpha_surface.py`
# cannot import `moe` unless the repo happens to be installed or PYTHONPATH
# happens to be set, and no document says to do either (audit R19/B11). It is
# the first thing here because the imports below depend on it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from moe.bench.published import Z_SUM_90_80, paired_mde  # noqa: E402


def _load_noise_floor():
    """Load the study's ONE noise-floor accessor BY PATH. `scripts/` is not a
    package, and `scripts/bn_decomposition.py` loads the same module the same
    way for the same reason.

    THE EXPORT CHECK IS THE POINT. This file used to read `prior_sd` straight
    out of the JSON, which is the s3/s4 proxy and stays the proxy after a pod
    run publishes a MEASURED replicate floor into the same file. If
    `sizing_sigma` is ever renamed away, a re-inlined read is what comes back,
    so the rename fails loudly here instead.
    """
    spec = importlib.util.spec_from_file_location(
        "replicate_noise_floor", ROOT / "scripts" / "replicate_noise_floor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    missing = [n for n in ("sizing_sigma", "NoiseFloorUnmeasured",
                           "NOISE_FLOOR_JSON")
               if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            "scripts/replicate_noise_floor.py no longer exports "
            f"{', '.join(missing)}. Every MDE this file prints is sized against "
            "that module's sigma; re-point the accessor rather than reading "
            "prior_sd out of the JSON again.")
    return module


NOISE = _load_noise_floor()

#: The noise this surface's paired comparisons are read against, and the file
#: that carries it. ONE definition, taken from the module that writes the file,
#: so this script and the floor's own publisher can never point at two paths.
NOISE_FLOOR = NOISE.NOISE_FLOOR_JSON

#: Below this many memory-bound treads the fit is not a measurement. The sweep
#: itself refuses a verdict under 3 and says so; this refuses to TABLE it, so a
#: blank cell means "not identifiable here" rather than a number nobody checked.
MIN_TREADS = 3

#: WHY THERE ARE TWO BLANK MARKERS NOW. The BN=256 arm's compute reference was
#: 44x too steep, so no tread in it could stand above the compute branch and all
#: 8 of its cells came out unidentifiable -- and this table printed them as `--`
#: under a caption blaming the tread count. The corruption read as a boring null
#: across two cards. A refused reference is a WITHDRAWN arm, not a quiet one, so
#: it gets its own marker and its own count.
REFUSED = "REF!"


def report_paths(root: Path) -> list[Path]:
    """Every block_m report under `root`, in BOTH committed layouts.

    THE BUG THIS FIXES. The published arms hold `<name>.report.json` -- one file
    per (model, GROUP_SIZE_M, BLOCK_SIZE_N) cell, named after the cell -- and
    this script globbed for the bare `report.json` a single-cell run writes. It
    therefore printed "no report.json under ..." on all three published surface
    arms, so the committed SURFACE.txt files could not be regenerated from the
    repository at all, and the only version that COULD run was the pooled one
    they were produced by (audit S38).

    A set, then sorted: a directory holding both spellings must not yield the
    same file twice, because every median and every paired count below is over
    this list.
    """
    return sorted({*root.rglob("report.json"), *root.rglob("*.report.json")})


def level_sort_key(value):
    """Sort a lever's levels NUMERICALLY where they are numbers.

    THE BUG THIS FIXES. `sorted(..., key=str)` put GROUP_SIZE_M in the order
    1, 16, 64, 8, so `vals[0]` and `vals[-1]` were 1 and 8 and the paired-change
    line reported the swizzle sweeping 1 -> 8 when it sweeps 1 -> 64. The number
    printed was a real paired median; it was a paired median of the wrong pair,
    and the lever's actual extremes never appeared (audit S38).

    Numbers first and in numeric order, then everything else lexically, then
    None last. A lever whose levels are model names still sorts as before.
    """
    if value is None:
        return (2, 0.0, "")
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def reports(root: Path):
    for p in report_paths(root):
        try:
            yield p, json.loads(p.read_text())
        except Exception as exc:                      # noqa: BLE001
            print(f"  UNREADABLE {p}: {type(exc).__name__}", file=sys.stderr)


def _checkout_independent(text: str) -> str:
    """This repo's own path, never the path this checkout happens to live at.

    `sizing_sigma` names the floor file by ABSOLUTE path, and these MDE lines
    are written into the committed `SURFACE.txt` summaries that
    `test_analysis_tools` regenerates byte for byte. A published summary
    carrying /home/whoever/ in it is one a stranger cannot rebuild, which is
    the exact property that test exists to hold.
    """
    return text.replace(str(ROOT) + "/", "")


def prior_sd(path: Path | None = None) -> tuple[float | None, str, str]:
    """`(sd of one alpha, MEASURED / ASSUMED / NONE, where it came from)`.

    THE BUG THIS FIXES, and it is the second call site of a fix that already
    landed. This function read `payload["prior_sd"]` out of the JSON directly.
    That field is the s3-vs-s4 proxy: the paired sd over 11 matched cells
    divided by sqrt(2), an UPPER bound because those two arms differ in
    `num_stages` as well as in nothing. A measured between-replicate floor is
    published into `replicate_floor` of the SAME file and leaves `prior_sd`
    untouched, so after the 120-minute part (a) arm this file would have gone on
    sizing every MDE it prints against the assumption, and every "below the
    detection limit" verdict below would have been scored against the wrong
    denominator. `replicate_noise_floor.sizing_sigma` is the one accessor that
    prefers the measurement and falls back to the proxy, and it is now the only
    way this file gets a sigma.

    THE BASIS WORD TRAVELS WITH THE NUMBER because a sigma printed without it
    reads as a measurement, and for most of this study's life it will not be
    one. `NONE` is the third state: the file is absent, unreadable or carries
    neither a floor nor a usable prior, and the MDE line then says the surface
    has no noise model, which is a statement about the study and not a reason
    to invent one.

    The field is `sizing_sigma`'s own default, `alpha_corrected`, which is the
    column the floor is pooled over. It is not the raw `alpha` this table's
    paired medians are taken in; the two differ by the reference's fixed cost
    and by 0.001-0.015 on the published arms, and the MDE is the looser of the
    two readings.
    """
    try:
        sd, basis, source = NOISE.sizing_sigma(path or NOISE_FLOOR)
    except NOISE.NoiseFloorUnmeasured as exc:
        return None, "NONE", _checkout_independent(" ".join(str(exc).split()))
    except (OSError, ValueError) as exc:
        # `sizing_sigma` promises `NoiseFloorUnmeasured` but `noise_floor`
        # parses the JSON before its own guards, so a TRUNCATED file arrives
        # here as a decode error. A surface that dies on a corrupt floor prints
        # no table at all; one that says NONE prints the table with no MDE.
        return None, "NONE", (f"{NOISE_FLOOR.name} unreadable "
                              f"({type(exc).__name__})")
    return sd, basis, _checkout_independent(source)


def print_mde(n_pairs: int) -> None:
    """The smallest paired lever effect this many matched cells could find.

    Printed BEFORE the paired medians, not after, because a reader who sees the
    medians first has already formed a view of them. Audit B14: not one arm in
    this study states an MDE, so every "the levers move it" and every "it does
    not move" below has been read without the one number that says which of them
    the design could have detected.
    """
    sd, basis, source = prior_sd()
    if sd is None:
        print(f"  MDE: NOT STATED -- {source}. Every paired change below is "
              "therefore")
        print("  uncompared with the smallest change this design could have "
              "found.")
        return
    sd_diff = sd * (2 ** 0.5)
    if n_pairs < 2:
        print(f"  MDE: only {n_pairs} matched cell(s); a paired MDE needs two, "
              "so no")
        print("  paired change here is comparable with anything.")
        return
    mde = paired_mde(sd_diff, n_pairs)
    print(f"  MDE {mde:.3f} alpha over {n_pairs} matched cells (paired, 90% "
          f"two-sided, 80% power,")
    print(f"  z-sum {Z_SUM_90_80:.4f}, {basis} sd of one alpha {sd:.4f} so sd "
          f"of a paired difference {sd_diff:.4f}).")
    # The word is on the sigma AND on its own line: an operator reading only
    # the numeric line must still be able to tell a measured floor from the
    # assumed proxy without opening the JSON.
    if basis == "MEASURED":
        print(f"  MEASURED, a between-replicate floor: {source}.")
    else:
        print(f"  {basis}, and it is an upper bound: {source}.")
    print("  A paired median below this is an unmeasured change, not a small "
          "one.")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = []
    for path, r in reports(root):
        fixed = r.get("fixed", {})
        cref = r.get("compute_reference") or {}
        # A report written before the level checks existed carries no
        # `refusals` key. Absent is treated as "not refused" but is NOT the
        # same as checked-and-clean, and the legend says which reports were
        # readable at all, so a stale corpus cannot pass itself off as a clean
        # one.
        refused = bool(cref.get("refusals"))
        for bm, lad in sorted(r.get("ladder", {}).items(), key=lambda kv: int(kv[0])):
            rows.append({
                "model": r.get("model", "?"),
                "g": fixed.get("GROUP_SIZE_M"),
                "bn": fixed.get("BLOCK_SIZE_N"),
                "bm": int(bm),
                "treads": lad.get("memory_points") or 0,
                "alpha": lad.get("alpha"),
                "corr": lad.get("alpha_corrected"),
                "hi": lad.get("alpha_upper"),
                "err": lad.get("mean_rel_err"),
                "refused": refused,
                "refused_bm": cref.get("refused_block_m"),
                "checked": "refusals" in cref,
                "path": path,
            })
    if not rows:
        print(f"no report.json under {root}")
        return 1

    print("ALPHA SURFACE -- every fit found under", root)
    print()
    print("alpha is the fraction of a weight re-read that MISSES L2. 1.0 means an")
    print("extra M-tile costs a full re-read; 0.0 means L2 absorbs it entirely.")
    print("A blank alpha means the fit was not identifiable at that BLOCK_M --")
    print(f"fewer than {MIN_TREADS} memory-bound treads -- not that it is zero.")
    print(f"{REFUSED} is a DIFFERENT state and not a null: that arm's compute")
    print("reference was refused on its level, so nothing in it could be")
    print("classified at all. Those rows are WITHDRAWN, not uninformative.")
    print()
    print(f"  {'model':<18} {'G':>4} {'BN':>4} {'BM':>4} {'treads':>7} "
          f"{'alpha':>7} {'corrected':>10} {'upper':>7} {'fit err':>8}")
    print(f"  {'-'*18} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*7} {'-'*10} "
          f"{'-'*7} {'-'*8}")
    for x in rows:
        # A refused reference outranks every other reason a cell is blank: it
        # is not that this BLOCK_M was uninformative, it is that nothing in the
        # arm was classified against a real compute branch.
        blank = REFUSED if x["refused"] else "--"
        ident = (not x["refused"] and (x["treads"] or 0) >= MIN_TREADS
                 and x["alpha"] is not None)
        a = f"{x['alpha']:.3f}" if ident else blank
        c = f"{x['corr']:.3f}" if ident and x["corr"] is not None else blank
        h = f"{x['hi']:.3f}" if ident and x["hi"] is not None else blank
        e = f"{x['err']*100:.2f}%" if ident and x["err"] is not None else blank
        print(f"  {x['model']:<18} {str(x['g']):>4} {str(x['bn']):>4} "
              f"{x['bm']:>4} {x['treads']:>7} {a:>7} {c:>10} {h:>7} {e:>8}")

    ident = [x for x in rows if not x["refused"]
             and (x["treads"] or 0) >= MIN_TREADS and x["alpha"] is not None]
    withdrawn = [x for x in rows if x["refused"]]
    unchecked = [x for x in rows if not x["checked"]]
    print()
    print(f"{len(ident)} identifiable fit(s) of {len(rows)} block sizes swept.")
    if withdrawn:
        arms = sorted({(x["model"], x["bn"], x["g"], x["refused_bm"])
                       for x in withdrawn})
        print(f"{len(withdrawn)} cell(s) WITHDRAWN: their compute reference was "
              "refused on its level, so they are not nulls and must not be "
              "read as one.")
        for model, bn, g, bm in arms:
            print(f"  {model} G={g} BN={bn}: reference refused at BLOCK_M={bm}")
    if unchecked:
        # Non-vacuity for the marker: a corpus of reports written before the
        # level checks existed would show zero withdrawals for the same reason
        # an empty scan does.
        print(f"{len(unchecked)} cell(s) come from reports with NO level check "
              "recorded, so their reference was never tested for level. Absence "
              "of a withdrawal above is not evidence of a sound reference in "
              "those; re-run the analysis over their cells.csv to find out.")
    if not ident:
        return 0

    # The two levers, each collapsed over the other, so a trend is visible even
    # when the grid is ragged. Medians rather than means: a single unidentified
    # BLOCK_M dropping out should not move the summary.
    def med(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    # PAIRED AT MATCHED LEVELS, never pooled. Averaging one lever over the others
    # is only valid on a balanced grid and this one is ragged: the 2026-09-01
    # H200 run had ten identifiable mixtral fits against seven for qwen2, spread
    # differently over GROUP_SIZE_M, and the pooled medians came out 0.731 and
    # 0.713 -- a footprint effect of 0.018, which is nothing. At matched
    # (G=1, BN=64) the same data give 0.98 and 0.72. The pooling did not measure
    # a small effect, it averaged a large one away, and it would have been read
    # as evidence AGAINST the footprint mechanism.
    others = {"g": ("model", "bn", "bm"), "model": ("g", "bn", "bm"),
              "bn": ("model", "g", "bm")}
    for lever, label in (("g", "GROUP_SIZE_M, the swizzle width"),
                         ("model", "model, i.e. per-expert footprint"),
                         ("bn", "BLOCK_SIZE_N, the activation-confound control")):
        vals = sorted({x[lever] for x in ident}, key=level_sort_key)
        if len(vals) < 2:
            continue
        keys = others[lever]
        cells: dict = {}
        for x in ident:
            cells.setdefault(tuple(x[k] for k in keys), {})[x[lever]] = x
        # A cell only counts when it holds EVERY level of the lever, so the
        # comparison is within one shape at one tile rather than across shapes.
        full = [c for c in cells.values() if set(c) == set(vals)]
        print()
        print(f"alpha against {label}:")
        print_mde(len(full))
        if not full:
            print(f"  no {'/'.join(keys)} cell holds every level of this lever, "
                  "so it cannot be compared without pooling. Not reported.")
            for v in vals:
                sel = [x["alpha"] for x in ident if x[lever] == v]
                print(f"    (unpaired: {v} n={len(sel)} median "
                      f"{med(sel):.3f} -- NOT a comparison)")
            continue
        print(f"  {len(full)} cell(s) of ({', '.join(keys)}) hold every level")
        for v in vals:
            sel = [c[v]["alpha"] for c in full]
            cor = [c[v]["corr"] for c in full if c[v]["corr"] is not None]
            print(f"  {str(v):<20} n={len(sel):<3} median alpha {med(sel):.3f}"
                  + (f"   corrected {med(cor):.3f}" if cor else ""))
        lo, hi = vals[0], vals[-1]
        deltas = [c[hi]["alpha"] - c[lo]["alpha"] for c in full]
        print(f"  paired change {lo} -> {hi}: median {med(deltas):+.3f}"
              f"   (every cell moves the same way: "
              f"{'yes' if len({d > 0 for d in deltas}) == 1 else 'NO'})")

    print()
    print("READ IT AGAINST THE THREE CANDIDATES:")
    for name, cand in (("this repo, retracted", 0.100),
                       ("TEMPO arXiv:2608.13057", 0.330),
                       ("the 2026-09-01 pooled refit", 0.558)):
        inside = [x for x in ident if abs(x["alpha"] - cand) < 0.05]
        print(f"  alpha = {cand:.3f} ({name}): "
              f"{len(inside)} of {len(ident)} fits within 0.05")
    print()
    print("A pooled scalar cannot describe this surface if the levers move it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
