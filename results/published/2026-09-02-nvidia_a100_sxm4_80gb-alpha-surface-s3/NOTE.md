# This arm was published against the wrong card's ridge, and has been rescored

## Which arm this is

`results/published/2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3`: the
A100-SXM4-80GB leg of the cross-card pair, session
`20260902T012557Z/cross_card/nvidia_a100_sxm4_80gb-s3`, as its `ARMS.tsv`
records. 7 `*.report.json` files, 2 mixtral cells and 5 qwen2. Every number in
this file was read out of those 7 and out of this arm's `ARMS.tsv`. Its H200
twin, `2026-09-01-nvidia_h200-cross-card-s3`, ran a day earlier in a different
session and has a NOTE of its own with different numbers; the two must not be
read across, and the fact that the run ids in the two directories are the same
seven hashes (the ids omit the card, audit A5) is exactly why that is easy to
get wrong.

## What was wrong

Every one of the 7 `*.report.json` files here was written with

    ridge = 160.3      ridge_band = [160.3, 176.2]

That is an **H200** figure, and dating it matters because two H200 ridges are in
circulation. It is not the 2026-08-26 one: the arms of that date quote 160.4 and
162.8. `160.3` is 701.6 TFLOP/s over 4377.2 GB/s, calibration md5 `4d84542b` in
`docs/INSTRUMENTATION.md`'s six-calibration table, which ships beside the
2026-08-28 `-h200-v2lite` and `-h200-fp8-three-kernel` arms and is the ridge the
2026-08-28 `-h200-whole-layer` arm's own rows were computed against. It reached
this A100 arm because the driver passed it on the command line, and nothing in
the sweep at the time checked the number against the attached device. The A100's
own calibration, committed as
`moe/bench/hardware/measured_nvidia_a100_sxm4_80gb.yaml`, gives

    262.3712 TFLOP/s bf16 / 1799.364 GB/s = 145.8 FLOP/byte

so the published ceiling was 9.9% too high, and it belonged to no device
attached to the machine that produced these rows. The band was worse than the
point. Both ends of `[160.3, 176.2]` are H200 figures, from two calibrations of
that one card: 701.6 TFLOP/s over 4377.2 GB/s and 770.9 over 4374.5
(`docs/INSTRUMENTATION.md`'s six-calibration table, where bandwidth reproduces
to 0.06% and the compute term does not). Its width is therefore the H200's
compute ceiling failing to reproduce, carried onto an A100 arm as if it were
this card's uncertainty about its own ridge.

`results/published/CALIBRATION_PROVENANCE.md` lists this arm as unknown,
`results/published/ANCHOR_RESCORE.txt` and `docs/COUNTERS.md` disclose the
substitution, and until this file nothing inside the arm directory did. A
reader who opened only the arm saw a measurement-shaped number that was a
constant from another machine.

## What was rescored, and what was not

Rescored, by `scripts/rescore_published_reports.py`, against this card's own
calibration:

| field | was | is |
|---|---|---|
| `ridge` | 160.3 | 145.8 |
| `ridge_band` | [160.3, 176.2] | [139.6, 149.3] |
| `predictions[BM].crossing_rows_ridge_lo` | solved at 160.3 | solved at 139.6 |
| `predictions[BM].crossing_rows_ridge_hi` | solved at 176.2 | solved at 149.3 |
| `predictions[BM].first_compute_tread` | at 160.3 | at 139.6 |
| `bracketing.horizon_rows` | 416.78 | 349.92 |

and four provenance fields were added: `ridge_source`, `ridge_band_source`,
`bandwidth_source`, `rescored_utc`, plus `rescored_from` carrying the withdrawn
numbers so nothing is quietly erased.

**Not rescored, because the ridge does not reach them.** This was established by
the audit's refuters, not assumed:

- **Every gate verdict.** Gate 4 scores BLOCK_M=64's peak throughput against
  this run's own BLOCK_M=256 plateau (100.02/165.80 = 0.603 on the mixtral G=1
  cell); gate 3 is a ratio of two crossings computed at one ridge, so the ridge
  cancels; gates 0-2 are about the kernel and the ladder fits. Changing the
  ridge moves none of them.
- **`ai_cap`**, which is `2 BM / (alpha b)` and contains no ridge.
- **Every `ladder` fit, `plateau_tflops`, `compute_reference` and
  `timing_spread_median`.** These are slopes and times.

## Why the reports were not simply regenerated

This arm published no `cells.csv` and no `run_*.csv` at all: its directory holds
the seven reports, an `ARMS.tsv` ledger, a PTX tarball and the ISA census, and
nothing else. Nor did any of the other 25 published reports get a `cells.csv`
(`find results/published -name cells.csv` returns none, audit B6). The ladders,
the plateau and the gate inputs
therefore cannot be recomputed from this repository at all. What *can* be
recomputed is the prediction block, because a prediction is arithmetic over
`(BLOCK_M, alpha, ridge, dtype bytes)` and all four are in the file. So the
recomputable half was recomputed and stamped with its source, which is strictly
more than the alternative of leaving seven files quoting a ceiling that belongs
to no attached device.

## How to check this

    python scripts/rescore_published_reports.py --dry-run  # plan; writes nothing
    python scripts/rescore_published_reports.py --write    # idempotent

A second `--write` rewrites nothing and leaves the tree clean; the run scores
four gates and prints one `RESULT:` line for each. `--self-test` plants the
failure branch of every gate.

## What is still open

Whether the calibration is contemporaneous with these reports. It is the right
CARD, which is what was rescored, and the evidence for "same session" stops
short of establishing it. In favour: this arm's own `ARMS.tsv` records
`calibrate PASS 38s` as the first arm of session `20260902T012557Z`, and the
committed yaml carries `checked_on: '2026-09-02'` at `measured_commit
63de5b9f`. Against: `checked_on` has day resolution, no `measured.yaml` was
published beside these reports, and the arm published no timed rows carrying
their own `achieved_peak_tflops`, which is the column
`moe.bench.published.provenance_report` uses to settle the question. It
therefore lists this arm `unknown` and entitles it to no ridge, and that verdict
has not changed. The claim this NOTE supports is the narrow one: the reports now
quote the A100's committed calibration rather than an H200's.
