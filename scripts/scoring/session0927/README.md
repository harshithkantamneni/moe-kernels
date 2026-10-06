# Ledger rows 1 to 9: the 2026-09-27 GH200 session, scored by the registered procedure (written 2026-10-05)

The scorer for OUTLINE section 3.1, rows 1 to 9: the timing model's P1 to P8
(`scripts/r3_timing_model.py`, docstring "THE REGISTERED PREDICTIONS" at fe73508, P1's
third clock 1605 MHz at f0a831b) and the byte model's two tests
(`scripts/wave_split_bytes.py`, "WHAT WOULD FALSIFY IT" at bb979d4), on the pages of
`results/published/2026-09-27-nvidia_gh200_480gb-session` (published at c7fd3db).
**It was written on 2026-10-05, after those pages.** The registrations fixed the
procedure and some numbers before the run; this scorer only applies them, and where the
registered words carry no number it says so and labels its own reading.

    python scripts/scoring/session0927/score.py [--repo GIT_REPO] [--out DIR]

`--repo` must hold the history (default `$MOE_HISTORY_REPO`, else this checkout); it is
read only. About 16 s on a laptop. Outputs, committed here:

| file | what |
|---|---|
| `score.txt` | the M3 facts, every group-(a) quote at its file:line, every group-(b) value with where it was literal before the run, the nine verdicts with their rules, and the comparison with the hand scoring |
| `score.json` | the same, with every cell |
| `prerun/r3_timing_model.f0a831b.txt` | f0a831b's `r3_timing_model.py` on the 2026-09-25 GH200 pages (the five VALID 1710 MHz pages and their counter pages), captured: the registered printout as it would have read before the run |
| `prerun/r3_timing_model.fe73508.txt` | the same at fe73508 (P1's third clock 1600 MHz; nothing else differs) |
| `prerun/wave_split_bytes.bb979d4.txt` | bb979d4's `wave_split_bytes.py` on the 2026-09-25 GH200 counter pages |

`tests/test_scoring_session0927.py` reruns the scorer and holds every output to the
committed one, and pins the verdicts.

## How it works

1. `git archive` extracts f0a831b, fe73508 and bb979d4 (scripts, moe, the 2026-09-25 GH200
   session) and c7fd3db's 2026-09-27 session into a temporary directory. The 2026-09-25 page
   directory is one git tree (187db863) at a49a2a4, at all three registering commits and at
   161f9ec; the scorer checks it.
2. `prerun.py` (glue, no model code) runs each commit's own tool with `python -I` from that
   commit's tree, so `import moe` is that commit's. The 2026-09-27 timed pages are read by
   f0a831b's `discover_timed` (every page VALID by its rule) and the byte pages by
   bb979d4's `load_card`.
3. Group (a), the numbers literal in a registering docstring, are checked at file:line.
   Group (b), the numbers the procedure computes, are recomputed and searched for in the
   registering commit's tool and test files.
4. Rows 1 to 9 are scored. Rules marked REGISTERED are the registration's; rules marked
   READING are this scorer's (the number used is printed). Reading R0 is P2's registered
   band ratio, [0.98, 1.01] x the reference, the one tolerance the timing registration
   states.
5. The hand scoring in the session README is typed into `HAND` for the comparison only.

## What it found (2026-10-05)

| row | procedure | hand | why they part |
|---|---|---|---|
| 1 P1 | HELD, eta 1 | HELD, eta 1 | same verdict. The registered "slope 2-6" is an OLS slope; the README quotes (T6 - T2) / 4 (0.6598, 0.5812 against OLS 0.6593, 0.5809). Inverted eta 0.95, 0.95, 0.88 (G=4) and 0.97 (G=2 low steps) |
| 2 P2 | **FALSIFIED** | HELD | G=32 SHARED n=4 to 5 reads 0.53754 ms, under the band's 0.538 by 0.46 us; it is inside only if rounded to 3 decimals first, which no registered text says |
| 3 P3 | HELD | HELD | same numbers |
| 4 P4 | HELD | HELD | same numbers |
| 5 P5 | FALSIFIED | FALSIFIED | same numbers; "ripple" is a READING (phase-mean range 0.074 ms against the registered ladder's own 0.038) |
| 6 P6 | **UNDECIDED** | HELD | the registration says "far from 354" and "near the limit" with no number, and states the falsifier two ways (AND in the docstring list, OR in the printed line). Achieved occupancy is 0.975 to 0.992 of the limit on w1 and 0.883 to 0.961 on w2, which the hand scoring did not read; cycles run 350.3 to 365.0 (w1; the README's 350 to 352 leaves out r3f-g2 n=3) and 354.1 to 367.4 (w2), at most 3.8% from 353.8. Loose reading: AND HELD, OR FALSIFIED; R0: both FALSIFIED |
| 7 WSC G=8 | HELD | HELD | same numbers |
| 8 WSC n=5, 8 | FALSIFIED | FALSIFIED | same verdict. The registered bar is relative to the prediction; the README's percentages divide by the measured q (G=16 +8.1, +19.4% against its 7.5, 16.3%) and include n=7, which the falsifier does not name. The README's "every w1 and PRIVATE cell within 1.5%" does not hold over the predicted cells: PRIVATE w2 G=16 n=8 is +3.1% |
| 9 P7, P8 | not answered | not answered | no unlocked timed page on the board; T3 did not trip |

Group (b) values committed before the run outside the docstring, as test pins: P1's slopes
at 1410 and 1500 (fe73508 `tests/test_r3_timing_model.py:323`) and at 1605 (f0a831b, the same
line, and the `P1_CLOCKS` comment at `scripts/r3_timing_model.py:342`), P1's G=2 increments
(:326, :327), P2's slope 0.5499 (:332), P3's G=2 times (:344), P4's 0.504 (:346), P5's ladder
(:350) and rho* 0.690 (:310), and WSC's w2 SHARED G=2 n=5 2.690 (bb979d4
`tests/test_wave_split_bytes.py:696`). P3's excesses 0.143 0.145 0.144 0.151 and every other
WSC cell at n=5 and 8 were literal nowhere before the pages.

By the owner's decision of 2026-10-05 (the rental-2 precedent) the ledger carries this
scorer's verdicts: 4 HELD (rows 1, 3, 4, 7), 3 FALSIFIED (2, 5, 8), 1 UNDECIDED (6), row 9 not
answered. Rows 2 and 6 note that the 2026-09-27 hand scoring read HELD; the session README
stays as published and is corrected by `docs/FINDINGS.md` (the 2026-09-27 section).
