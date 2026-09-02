# This arm quoted a ridge from another calibration of its own card, and has been rescored

## Which arm this is

`results/published/2026-09-01-nvidia_h200-cross-card-s3`: the H200 leg of the
2026-09-01 cross-card pair, session `20260901T214218Z/cross_card/nvidia_h200-s3`,
as its `ARMS.tsv` records. 7 `*.report.json` files, 2 mixtral cells and 5 qwen2.
Every number in this file was read out of those 7 and out of this arm's
`ARMS.tsv`. Its A100 twin,
`2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3`, has a NOTE of its own with
different numbers, and the two must not be read across.

## What was wrong

Every one of the seven reports was written with

    ridge = 160.3      ridge_band = [160.3, 176.2]

This is an H200 arm, so unlike its A100 twin the card was right and the
CALIBRATION was not. `160.3` is 701.6 TFLOP/s over 4377.2 GB/s, calibration md5
`4d84542b` in `docs/INSTRUMENTATION.md`'s six-calibration table, which ships
beside the 2026-08-28 `-h200-v2lite` and `-h200-fp8-three-kernel` arms. It was
passed to this sweep on the command line and nothing checked it against the
device. The band was worse than the point: `[160.3, 176.2]` is that figure
paired with 770.9 over 4374.5 from a DIFFERENT session's calibration of the same
card, so its width is the H200's compute ceiling failing to reproduce between
sessions rather than any measurement's uncertainty about a ridge.

The H200's committed calibration, `moe/bench/hardware/measured_nvidia_h200.yaml`,
gives

    712.2592 TFLOP/s bf16 / 4374.763 GB/s = 162.8 FLOP/byte

and a band `[152.1, 165.6]` from that same card's own bandwidth patterns carried
as a ratio against its triad ceiling, which is what a band should be made of.
162.8 sits inside it; 176.2 does not.

## How big the change is, stated before it is quoted

1.6%, and the rescoring run's own MDE line prices the noise these reports carry
at 2.7% relative (median `timing_spread_median` 0.77% across the 26 published
reports, two independent cells, 90% two-sided, 80% power). The shift is INSIDE
that. So for this arm the rescoring is a provenance repair and not a changed
conclusion: what moved is that the ceiling now names a card and a file. The A100
twin's shift was 9.9% and is the case where the number itself was wrong.

## What was rescored, and what was not

Rescored by `scripts/rescore_published_reports.py` in all seven reports, which
carry one alpha (0.558) and therefore one prediction block:

| field | was | is |
|---|---|---|
| `ridge` | 160.3 | 162.8 |
| `ridge_band` | [160.3, 176.2] | [152.1, 165.6] |
| `predictions[128].crossing_rows_ridge_lo` | 249.75 | 236.97 |
| `predictions[128].crossing_rows_ridge_hi` | 372.84 | 350.41 |
| `predictions[256].crossing_rows_ridge_lo` | 160.30 | 152.10 |
| `predictions[256].crossing_rows_ridge_hi` | 176.20 | 165.60 |
| `bracketing.horizon_rows` | 416.78 | 423.28 |

plus `ridge_source`, `ridge_band_source`, `bandwidth_source`, `rescored_utc`,
and `rescored_from` carrying the withdrawn numbers so nothing is quietly erased.

`predictions[32]` and `predictions[64]` were null before and are null after.
That null is a PREDICTION, not a gap: their `ai_cap` (57.3 and 114.7 FLOP/byte)
is below the ridge at either figure, so those tiles never cross at any batch
size, and moving the ridge by 1.6% does not reach that verdict.

**Not rescored, because the ridge does not enter them.** Checked field by field
rather than argued: `scripts/rescore_published_reports.py`'s `fields_confined`
gate re-serialises the untouched half of every report and compares it with the
original.

- **Every gate verdict.** Gate 4 scores BLOCK_M=64's peak against this run's own
  BLOCK_M=256 plateau (`peak roof fraction 0.661` on the mixtral G=1 cell, whose
  `plateau_tflops` is 357.27), gate 3 is a ratio of two crossings computed at one
  ridge so the ridge cancels (`1.916x` on the same cell), and gates 0-2 are
  about the kernel and the ladder.
- **`ai_cap`**, which is `2 BM / (alpha b)` and contains no ridge.
- **Every `ladder` fit, `plateau_tflops`, `compute_reference` and
  `timing_spread_median`.** These are slopes and times.

## Why the reports were not simply regenerated

This arm published no `cells.csv` and no `run_*.csv`: its directory holds the
seven reports, `ARMS.tsv`, a PTX tarball and the two SURFACE files. Neither did
any of the other 25 published reports (`find results/published -name cells.csv`
returns none, audit B6). The ladders, the plateau and the gate inputs therefore
cannot be recomputed from this repository. The prediction block can, because a
prediction is arithmetic over `(BLOCK_M, alpha, ridge, dtype bytes)` and all
four are in the file, so the recomputable half was recomputed and stamped with
its source.

## How to check this

    python scripts/rescore_published_reports.py --dry-run  # plan; writes nothing
    python scripts/rescore_published_reports.py --write    # idempotent

A second `--write` rewrites nothing and leaves the tree clean; the run scores
four gates and prints one `RESULT:` line for each. `--self-test` plants the
failure branch of every gate.

## What is still open

Whether 162.8 is THIS arm's own ridge. It is this card's, which is what was
rescored, and no more than that can be shown from here. This arm's `ARMS.tsv`
records `calibrate PASS 23s` as its second arm, so a contemporaneous H200
calibration did exist on 2026-09-01; it was not published beside these reports,
and `scripts/calibrate_hardware.py` writes the tracked
`moe/bench/hardware/measured_nvidia_h200.yaml` in place, so the file now in the
tree is a LATER one: `checked_on: '2026-09-02'` at `measured_commit 63de5b9f`,
the day after this session. `results/published/CALIBRATION_PROVENANCE.md` lists
this arm `unknown` and entitles it to no ridge, because it published no timed
rows carrying the `achieved_peak_tflops` that would settle it, and that verdict
has not changed. The claim this NOTE supports is the narrow one: the reports now
quote an H200 calibration that names its file, instead of a constant from a
command line.
