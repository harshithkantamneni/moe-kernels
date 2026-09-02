# This arm was published against the wrong card's ridge, and has been rescored

## What was wrong

Every one of the seven `*.report.json` files here was written with

    ridge = 160.3      ridge_band = [160.3, 176.2]

That is the **H200's** 2026-08-26 figure. It reached this A100 arm because the
driver passed it on the command line, and nothing in the sweep at the time
checked the number against the attached device. The A100's own calibration,
measured in this same session and committed as
`moe/bench/hardware/measured_nvidia_a100_sxm4_80gb.yaml`, gives

    262.3712 TFLOP/s bf16 / 1799.364 GB/s = 145.8 FLOP/byte

so the published ceiling was 9.9% too high, and it belonged to no device
attached to the machine that produced these rows. The band was worse than the
point: `[160.3, 176.2]` is one card's triad ridge beside another card's read
ridge, a band drawn across two machines.

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

No `cells.csv` was published beside any report in this arm, or in the other two
arms of the same session. The ladders, the plateau and the gate inputs
therefore cannot be recomputed from this repository at all. What *can* be
recomputed is the prediction block, because a prediction is arithmetic over
`(BLOCK_M, alpha, ridge, dtype bytes)` and all four are in the file. So the
recomputable half was recomputed and stamped with its source, which is strictly
more than the alternative of leaving seven files quoting a ceiling that belongs
to no attached device.

## How to check this

    python scripts/rescore_published_reports.py            # plan; writes nothing
    python scripts/rescore_published_reports.py --write    # idempotent

A second `--write` rewrites nothing and leaves the tree clean; the run scores
four gates and prints one `RESULT:` line for each. `--self-test` plants the
failure branch of every gate.

## What is still open

The ridge above is this card's, from this card's yaml, and the arm's rows were
not published with a `measured.yaml` of their own, so "same session" cannot be
established from inside the directory. The claim this NOTE supports is narrow:
the reports now quote the A100's committed calibration rather than an H200's.
Whether that calibration is contemporaneous with these rows is a separate
question and `CALIBRATION_PROVENANCE.md` still answers it "unknown".
