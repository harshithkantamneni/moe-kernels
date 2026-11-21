# Index of the code-docstring registrations scored on the 2026-09-27 GH200 pages (ledger rows 1 to 9)

**THIS FILE IS AN INDEX, NOT A REGISTRATION.** It was written on 2026-10-05, after the
2026-09-27 pages existed (they were published at c7fd3db, 2026-09-27 10:08:03 UTC). It
registers nothing and changes no prediction. It points at the registrations, which are
the module docstrings of two scripts as committed before the run, and says which numbers
were literal there. By the owner's decision of 2026-10-05 (gap 6.10), rows 1 to 9 count as
registered with this disclosure: they were registered in code docstrings, not in this
directory, and part of their numbers (group b below) were never printed before the pages.

Read each registration as committed, not at HEAD: at HEAD the P5 lines carry a FALSIFIED
note added after the pages.

    git show fe73508:scripts/r3_timing_model.py     # P1 to P8
    git show f0a831b:scripts/r3_timing_model.py     # P1's third clock as 1605 MHz
    git show bb979d4:scripts/wave_split_bytes.py    # rows 7 and 8

Every quote, line number and ancestry fact below is checked by
`scripts/scoring/session0927/score.py` (outputs `score.txt`, `score.json` beside it), which
also recomputes every group-(b) value from the pre-run code and scores the nine rows.

## The two groups

- **(a) REGISTERED, literal**: the number is in the registering commit's module docstring.
- **(b) REPRODUCED FROM PRE-RUN CODE**: the docstring registers the procedure that
  computes the number, not the number. It is recomputed by running the registering
  commit's own tool on the 2026-09-25 GH200 pages (published at a49a2a4; the page directory
  is one git tree, 187db863, at a49a2a4, fe73508, bb979d4, f0a831b and 161f9ec). No
  docstring printed it before the pages. Several were committed before the run as test
  pins in `tests/` (named per row): those are pre-run commits, but not the docstring.

## The M3 facts (the same for every row)

| commit | committer time (UTC) | what | ancestor of 161f9ec (run) | ancestor of c7fd3db (pages) |
|---|---|---|---|---|
| a49a2a4 | 2026-09-25 22:16:53 | 2026-09-25 GH200 pages published (the predictions' inputs) | yes | yes |
| fe73508 | 2026-09-26 17:55:37 | `scripts/r3_timing_model.py` docstring "THE REGISTERED PREDICTIONS", P1 to P8 | yes | yes |
| bb979d4 | 2026-09-26 17:55:37 | `scripts/wave_split_bytes.py` docstring "WHAT WOULD FALSIFY IT" | yes | yes |
| f0a831b | 2026-09-26 21:26:59 | P1's third clock re-registered 1600 -> 1605 MHz (docstring line 144) | yes | yes |
| 161f9ec | 2026-09-27 05:14:58 | the run commit (`git_sha` of every 2026-09-27 page) | - | yes |
| c7fd3db | 2026-09-27 10:08:03 | the 2026-09-27 pages and session README published | - | - |

a49a2a4 is an ancestor of fe73508, bb979d4 and f0a831b. fe73508 and bb979d4 share a
timestamp and neither is an ancestor of the other; both are ancestors of f0a831b. The
2026-09-27 page directory does not exist at 161f9ec. The instance launched 2026-09-27 05:46
UTC (session README). M3 (`grading/checks/m3_order.py`) reads only `docs/registered/*.json`,
`*.txt` and `README.md` sections, so it does not see these registrations and does not see
this file.

## Per row

Line numbers are at the registering commit (fe73508 and f0a831b have the same docstring
line numbers; only line 144 differs between them).

| row | test | commit, file:line | quoted text (exact) | group |
|---|---|---|---|---|
| 1 | P1 floor clock exponent | f0a831b `scripts/r3_timing_model.py`:144-147 (fe73508 :144 read 1600) | "P1  a held lock at 1410, 1500 and 1605 MHz: the G=4 SHARED slope 2-6, and / the G=2 increments at 1410, at eta 1 (a fixed cycle count) and at eta / 0.35 (what the GH200's unlocked pages read). The eta the G=4 slope gives / and the eta the G=2 low steps give must agree." | (a) the clocks, the two etas and the agreement rule. (b) the G=4 slopes 0.6654 0.6259 0.5854 (eta 1) and 0.5878 0.5754 0.5621 (eta 0.35), the G=2 increments 0.550 0.731 0.600 0.733 0.598 and 0.393 0.703 0.473 0.704 0.471. Test pins before the run: fe73508 and f0a831b `tests/test_r3_timing_model.py`:323 (slopes), :326, :327 (increments); 0.5854 and 0.5621 also in f0a831b's `P1_CLOCKS` comment, `scripts/r3_timing_model.py`:342-343 |
| 1 | P1 falsifier | fe73508 `scripts/r3_timing_model.py`:23-24 | "P1's two eta readings (from the G=4 slope and from the G=2 low steps) / disagreeing;" | (a), no tolerance stated |
| 2 | P2 G=8, 32 SHARED steps | fe73508 `scripts/r3_timing_model.py`:148-156 | "P2  G=8 and G=32 at the fit clock. The band holds the per-tread increments / from n=2 to n=6 and their slope 2-6. The n=1 to 2 step is a separate / point prediction and is not held to the band: on the GH200 it is 0.315 / ms against a band of [0.538, 0.556]. ... [0.98, 1.01] x the model's slope, rounded outward to 3 / decimals ..." | (a) the band [0.538, 0.556], the n=1 to 2 step 0.315 ms, the band rule. (b) the model slope 0.5499 the band is built on (test pin fe73508 `tests/test_r3_timing_model.py`:332) |
| 3 | P3 G=2 odd-n excess | fe73508 `scripts/r3_timing_model.py`:160-161 and :20-21 | "P3  G=2 at n=7 to 10, and its odd-n excess over G=4: at least 0.10 ms and / flat in n." | (a) the 0.10 ms floor and "flat in n". (b) the excesses 0.143 0.145 0.144 0.151 ms at n = 3, 5, 7, 9, literal nowhere before the pages (the G=2 times at n=7 to 10 are pinned at `tests/test_r3_timing_model.py`:344) |
| 4 | P4 G=4 n=8 to 9 | fe73508 `scripts/r3_timing_model.py`:162 | "P4  G=4 n=8 to 9, where q_w2 drops (1.031 to 1.003 at 132 SMs)." | (a) q_w2 1.031 to 1.003. (b) the n=8 to 9 increment 0.504 ms (test pin `tests/test_r3_timing_model.py`:346) and the other increments' median 0.550 |
| 5 | P5 G=3 flat | fe73508 `scripts/r3_timing_model.py`:163-164 | "P5  G=3: flat when rho*_w1 exceeds the largest slab-fetching share a full / G=3 group has (2/3), a period-3 ripple when it is below." | (a) the 2/3 threshold and the flat/ripple rule. (b) rho*_w1 0.690 (test pin :310) and the ladder 0.342 0.523 0.550 0.567 0.533 0.563 0.566 (test pin :350) |
| 6 | P6 floor cycles | fe73508 `scripts/r3_timing_model.py`:165-166, :190-192, :25-26 | "P6  the floor in cycles per CTA k-step, beside the census PTX's / shared-memory cycles when the session carries its PTX." and "353.8 cycles per CTA k-step / at the 1710 MHz lock on both the GH200 and the H100 matches the 352 cycles" and "- P6 showing achieved occupancy near the occupancy limit AND cycles per CTA / k-step far from 354;" | (a) 353.8 cycles, 352 PTX cycles, the falsifier. No tolerance for "near" or "far"; the tool's printed line (not docstring, :1681) says "or" where the docstring says AND |
| 7 | WSC w2 SHARED G=8 n=2, 4 | bb979d4 `scripts/wave_split_bytes.py`:171-173 | "WHAT WOULD FALSIFY IT (any one): / - GH200 w2 SHARED at G=8, n = 2 or 4, above 1.12 (it predicts 1.082 and / 1.118; the LRU rival predicts 1.139 and 1.159);" | (a) all of it: the 1.12 ceiling, 1.082, 1.118, 1.139, 1.159 |
| 8 | WSC n=5, 8 within 5% | bb979d4 `scripts/wave_split_bytes.py`:174 | "- n=5 or n=8 at G = 2, 4 or 16 outside +-5% of its registered predictions;" | (a) the rule. (b) every per-cell prediction (the tool's "PREDICTED" table, `scripts/scoring/session0927/prerun/wave_split_bytes.bb979d4.txt`); one is a test pin, w2 SHARED G=2 n=5 2.690 (bb979d4 `tests/test_wave_split_bytes.py`:696) |
| 9 | P7, P8 | fe73508 `scripts/r3_timing_model.py`:167-179 | "P7  on a LOCKED fit only, every unlocked page of the same card ..." and "P8  where T3 trips: ..." | (a) the 09-25 readings 1782 and 1787 MHz, 1.32 to 4.84 W_w2. Not answered on 2026-09-27 |

## Scored

By `scripts/scoring/session0927/score.py` (written 2026-10-05, after the pages), whose verdicts
are the ledger's by the owner's decision (the rental-2 precedent: a post-page scorer makes the
printed verdict follow the registration's text): rows 1, 3, 4 and 7 HELD; rows 2, 5 and 8
FALSIFIED; row 6 UNDECIDED; row 9 not answered. The 2026-09-27 hand scoring in
`results/published/2026-09-27-nvidia_gh200_480gb-session/session/README.md` ("What it answered",
left as published) read rows 2 and 6 HELD. It differed on P2 because it compared G=32's n=4 to 5
step (0.53754 ms) to the band [0.538, 0.556] after rounding to 3 decimals, which nothing
registered says; and on P6 because it read the cycles alone (and left out w1's 365.0 at G=2,
n=3), while the falsifier also names achieved occupancy (0.88 to 0.99 of the limit), gives no
number for "far" or "near", and reads AND in the docstring but OR in the tool's printout. The
correction is in `docs/FINDINGS.md`, the 2026-09-27 section (CORRECTED 2026-10-05).
