#!/usr/bin/env python
"""Re-derive a published sweep's ceiling columns against a new calibration.

    python scripts/recompute_ceilings.py \
        --arm results/published/2026-08-26-nvidia_h200-full-three-way \
        --calibration moe/bench/hardware/measured_nvidia_h200.yaml

WHY, INSTEAD OF RE-SWEEPING. The calibration determines four columns and nothing
else (driver.py:262-289). `ms_p50`, `tflops`, `compulsory_gbps` and
`arith_intensity_compulsory` come from the measured time and the byte model. So
when calibrate.py was found to settle under a matmul and then measure bandwidth,
and to name a tree reduction as its read ceiling, the correction is four columns
wide, not three hours long. The timings were never wrong.

Writes a NEW arm rather than editing in place. The original is the record of
what was measured against the ruler of the day, and overwriting it would erase
the evidence that the ruler moved.

AND IT DECLARES ITSELF. The arm this produces is the one legitimate case of a
published result whose calibration comes from a later session than its rows --
that is the whole point of it -- and `publish_results.sh` now refuses exactly
that shape, because an undeclared instance of it cost claim C5 its target. So a
`DERIVED_FROM` marker goes in beside the rows, the same way `SUPERSEDED` does,
and `moe.bench.published.derived_from` reads it. The README keeps saying so too:
prose for a human, a marker for the check, and the marker is authoritative
because a README gets hand-edited and this one already has been.

TWO THINGS THIS DID NOT DO UNTIL 2026-09-02, both found by running it twice.

  * IT RECORDED NO PROVENANCE OF ITS OWN. The rows keep the `git_sha` of the
    commit that MEASURED them, which is right, and nothing anywhere named the
    commit of the `moe/bench/recompute.py` that REWROTE them, the time it was
    done, or the calibration it was done against as anything but a path in a
    sentence. `check_published_shas` then validated a commit that did not
    produce the numbers now in the file. `recompute.json` beside the rows is
    that record, with the full `moe.bench.provenance` block in it.
  * IT RESTAMPED ROWS AGAINST ANOTHER CARD'S CALIBRATION WITHOUT A WORD. Run
    against `measured_nvidia_h200.yaml` and then `measured_nvidia_a100_sxm4_80gb.yaml`,
    both runs derive `<arm>-recalibrated`, the second overwrote the first, and
    H200 rows came back with `achieved_bw_gbps` 4377.2 -> 1799.4 and
    `implied_traffic_ratio` 3.716 -> 1.527. `publish_results.sh` would block
    that arm, which is what kept it off the blocking list, but the wrong numbers
    were on disk for anyone reading the arm directly. Both halves are refusals
    now: a calibration for a different card than the rows were measured on, and
    a destination already derived from a DIFFERENT calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import provenance as PV  # noqa: E402
from moe.bench.published import DERIVED_MARKER  # noqa: E402
from moe.bench.recompute import (  # noqa: E402
    CEILING_COLUMNS,
    load_calibration_hardware,
    rewrite_csv,
)
from moe.bench.roofline import device_matches  # noqa: E402

#: What this script did, named as an instrument so a reader of `recompute.json`
#: never mistakes it for a measurement. Nothing here times anything: four
#: columns are re-derived from timings another commit measured.
INSTRUMENT = "recompute/four-ceiling-columns/no-kernel-timed"

#: Where the record of the rewrite lands, beside the rows it rewrote.
RECOMPUTE_JSON = "recompute.json"

#: The line `DERIVED_FROM` carries the calibration's content hash on. The first
#: line stays the source arm name, which is what `published.derived_from` reads.
SHA_PREFIX = "calibration sha256: "


def calibration_sha(path: Path) -> str:
    """The calibration's CONTENT hash, not its path.

    The path is what the marker used to record, and the H200 was recalibrated
    six times to six distinct ceilings: two different files at one path are two
    different rulers, and a path cannot tell them apart.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_cards(csvs: list[Path]) -> set[str]:
    """Every distinct `gpu_name` in the rows about to be rewritten.

    Read from the CSVs rather than from the directory name: the name is a label
    a human chose and the column is what the driver stamped on the row.
    """
    cards: set[str] = set()
    for src in csvs:
        with src.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("gpu_name") or "").strip()
                if name:
                    cards.add(name)
    return cards


def build_parser() -> argparse.ArgumentParser:
    """Split from `main` so a test can build an argv without a shell."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=Path, required=True,
                    help="published results directory to re-derive")
    ap.add_argument("--calibration", type=Path, required=True,
                    help="the NEW calibration yaml")
    ap.add_argument("--out", type=Path, default=None,
                    help="destination arm (default: <arm>-recalibrated)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.arm.is_dir():
        raise SystemExit(f"no such arm: {args.arm}")
    hw = load_calibration_hardware(args.calibration)
    out = args.out or args.arm.parent / (args.arm.name + "-recalibrated")

    csvs = sorted(args.arm.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs under {args.arm}")

    # REFUSAL 1: ANOTHER CARD'S RULER. `rewrite_csv` will restamp every row from
    # whatever calibration it is handed, and an A100 calibration over H200 rows
    # moved achieved_bw_gbps 4377.2 -> 1799.4 and implied_traffic_ratio
    # 3.716 -> 1.527 without a word. `publish_results.sh` would block the arm
    # afterwards, but "blocked at publish time" is not the same as "never
    # written", and the wrong numbers were on disk in between.
    #
    # The ROWS are asked which machine produced them, not the directory name,
    # which is a label a human chose; and the comparison is `device_matches`
    # rather than a slug, because the yaml calls the card "NVIDIA H200
    # (measured)" and the rows call it "NVIDIA H200". That loose rule is
    # imported rather than written a second time here.
    cards = row_cards(csvs)
    if not cards:
        raise SystemExit(
            f"REFUSING: no row under {args.arm} carries a gpu_name, so the "
            f"calibration cannot be checked against the machine that measured "
            f"them. A check that examined nothing reports no failures.")
    wrong = sorted(c for c in cards if not device_matches(hw, c))
    if wrong:
        raise SystemExit(
            f"REFUSING: {args.arm.name} was measured on {', '.join(wrong)} and "
            f"{args.calibration} describes {hw.name}. Re-deriving the ceiling "
            f"columns against another card's ruler produces rows that are wrong "
            f"rather than stale, and nothing in the output would say so.")

    # REFUSAL 2: A DESTINATION ALREADY DERIVED FROM A DIFFERENT RULER. Two
    # calibrations of ONE card derive one default destination -- the H200 was
    # recalibrated six times to six distinct ceilings -- and the second run
    # overwrote the first's CSVs, measured.yaml, README and marker in silence.
    sha = calibration_sha(args.calibration)
    marker = out / DERIVED_MARKER
    if marker.exists():
        previous = [ln.split(SHA_PREFIX, 1)[1].strip()
                    for ln in marker.read_text().splitlines() if SHA_PREFIX in ln]
        if previous and previous[0] != sha:
            raise SystemExit(
                f"REFUSING to overwrite {out}: it was derived from a "
                f"calibration whose sha256 is {previous[0][:12]} and this one is "
                f"{sha[:12]}. Two rulers, two arms; pass --out to name the "
                f"second one.")

    out.mkdir(parents=True, exist_ok=True)

    print(f"calibration : {hw.name}")
    print(f"  bandwidth : {hw.bandwidth_bytes_s / 1e9:.1f} GB/s "
          f"(pattern {hw.ceiling_pattern or 'unnamed'})")
    for dt, fl in sorted(hw.peak_flops.items()):
        print(f"  peak {dt:9}: {fl / 1e12:.1f} TFLOP/s")
    print(f"source arm  : {args.arm}")
    print(f"destination : {out}\n")

    rewritten: dict[str, dict] = {}
    for src in csvs:
        info = rewrite_csv(src, out / src.name, hw)
        rewritten[src.name] = {"rows": info["rows"],
                               "changed": {k: v for k, v in info["changed"].items()}}
        moved = {k: v for k, v in info["changed"].items() if v}
        print(f"  {src.name:44} {info['rows']:6d} rows  "
              + (", ".join(f"{k}:{v}" for k, v in moved.items()) or "unchanged"))

    # Only the manifests travel. FINDINGS.md is a human analysis of the arm it
    # was written for and quotes ratios against the ruler of that day; copying
    # it here puts prose that disagrees with the rows right beside them.
    # SUMMARY.md is generated and would be equally stale.
    for extra in args.arm.iterdir():
        if extra.is_file() and extra.name.endswith(".manifest.jsonl"):
            shutil.copy2(extra, out / extra.name)
    shutil.copy2(args.calibration, out / "measured.yaml")

    # The declaration the publish gate reads. First line is the source arm, so
    # `derived_from` can name it without parsing prose.
    (out / DERIVED_MARKER).write_text(
        f"{args.arm.name}\n"
        f"recomputed by scripts/recompute_ceilings.py against "
        f"{args.calibration}\n"
        f"{SHA_PREFIX}{sha}\n"
        f"Its measured.yaml is deliberately from a different session than its\n"
        f"rows: only the ceiling columns were re-derived, and the timings are\n"
        f"the source arm's untouched.\n")

    # THE RECORD OF THE REWRITE ITSELF. The rows keep the git_sha of the commit
    # that MEASURED them, which is right and is also why nothing in this arm
    # named the commit that REWROTE them until now. `bandwidth` is the ceiling
    # every re-derived column is computed from and its source is the calibration
    # file by name, which is what makes this arm's numbers checkable rather than
    # merely declared.
    prov = PV.provenance_block(
        instrument=INSTRUMENT,
        bandwidth=hw.bandwidth_bytes_s,
        bandwidth_source=f"{Path(args.calibration).name} sha256:{sha[:12]} "
                         f"({hw.name}, pattern {hw.ceiling_pattern or 'unnamed'})")
    (out / RECOMPUTE_JSON).write_text(json.dumps(prov.stamp({
        "derived_from": args.arm.name,
        "source_arm": str(args.arm),
        "calibration": str(args.calibration),
        "calibration_sha256": sha,
        "calibration_name": hw.name,
        "ceiling_columns": list(CEILING_COLUMNS),
        "row_cards": sorted(cards),
        "csvs": rewritten,
    }), indent=2) + "\n")

    (out / "README.md").write_text(
        f"# {out.name}\n\n"
        f"Derived from `{args.arm.name}` by `scripts/recompute_ceilings.py`.\n"
        f"The measurements are identical: `ms_p50`, `tflops`, `compulsory_gbps`\n"
        f"and `arith_intensity_compulsory` come from the timing and the byte\n"
        f"model and were never affected by the calibration. Only the four\n"
        f"calibration-derived columns differ:\n\n"
        + "".join(f"  - `{c}`\n" for c in CEILING_COLUMNS) +
        f"\nRecomputed against **{hw.name}**, {hw.bandwidth_bytes_s / 1e9:.1f} "
        f"GB/s (pattern `{hw.ceiling_pattern or 'unnamed'}`).\n\n"
        f"No FINDINGS.md here on purpose. The analysis in `{args.arm.name}` was\n"
        f"written against the ruler of that day, and the two arms together are\n"
        f"the evidence that the ruler moved. Read that one, and treat these rows\n"
        f"as the corrected numbers.\n")
    print(f"\nwrote {out}")
    print("The original arm is untouched: it is what was measured against the")
    print("ruler of the day, and the pair is the evidence that the ruler moved.")
    print(f"Dropped a {DERIVED_MARKER} marker, so the publish gate knows this")
    print("arm's calibration is from a later session on purpose, and a")
    print(f"{RECOMPUTE_JSON} naming the commit, the clock and the ruler that")
    print("did the rewriting, which the rows themselves cannot carry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
