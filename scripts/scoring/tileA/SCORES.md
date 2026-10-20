# Tile rental A scores (2026-10-20)

The six rental-A scorers in this directory, run on
`results/published/2026-10-20-nvidia_gh200_480gb-tileA-session` (625 run files, each matching the
run branch's SHA256SUMS; the 21 compiled `cuda_utils*.so` files are left out), in the order their
inputs need (floor reads census.score.json; dead reads floor and rep; e1cal reads all four):

    for p in census floor bytes rep dead e1cal; do
      PYTHONHASHSEED=0 PYTHONPATH=.:scripts python scripts/scoring/tileA/score_$p.py . \
        results/published/2026-10-20-nvidia_gh200_480gb-tileA-session/results scripts/scoring/tileA
    done

Outputs: `census`, `floor`, `bytes`, `rep`, `dead`, `e1cal` `.score.{json,txt}` beside the scorers.
No scorer crashed or refused once `PYTHONPATH` is set (see "Open defects": without it `score_floor.py`
and `score_bytes.py` do not start). **No post-page fix**: no scorer, registration or rule was changed
after the pages existed. The first scoring (2026-10-20, on the fetched run tree) and this one agree:
the `.score.txt` files and the bytes, rep, dead and e1cal JSON are byte-identical; census and floor
JSON are equal as data and differ only in the order of one dict's keys and in the census file path
(the tree's location). `PYTHONHASHSEED=0` fixes that order for the committed files.

## The verdicts

| registration | test | registered verdict | numbers |
|---|---|---|---|
| tileA-census | VALIDITY (MMA path) and occupancy | **VALID (INCOMPLETE)** | 72 configs, 0 path mismatches; 6 errors, all deepseek-v2-lite; occupancy 2 / 3 / 3 / 4 / 3 / 4 (128x128 w8, 64x128 w8, 64x128 w4, 32x128 w4, 16x64x128 w4, 32x64x128 w4); 32x128x64 w4 differs from the design's 5 |
| tileA-floor | CAL-A: c_128, F_wg,w1, F_wg,w2 (joint fit) | **NOT SCORED** (under 6 floor-bound cells or rank deficient) | f128 pages: 6 of 27 (OLMoE) and 7 of 27 (Mixtral) cells in the clock band, all PRIVATE; the fit reads NATIVE and SHARED only |
| | 64x128 test, selection, F form, w4, mma.sync | not reached (the scorer returns at the calibration) | |
| tileA-bytes | the A key (NAMED when 2x nearer and every row within 5%, else CLAMP) | **NOT NAMED (nearest CLAMP); v4 takes CLAMP** | 19 of 56 registered rows scored: CLAMP 5.025%, v3 5.025% (1.00002x), ROWKEY30 12.29%, ROWKEY60 12.31%; worst CLAMP row +7.47% |
| tileA-e1cal | E1-CAL; the stop rule (tileB-stop) | **E1-CAL NOT SCORED; STOP: rental B does not run** | the only clause printed: tA-floor has no calibration |
| tileA-rep | sigma_page at 128x128 | **MEASURED 0.1275%** | 24 cells valid on both pages; mean shift -0.154% |
| tileA-dead | D against SLOT at occupancy 2 (EXCLUDED when \|z\| > 3) | **D, D2, SLOT, d 1.06 EXCLUDED; D against SLOT NOT SEPARATED by this page; kappa rows NOT SCORED** | delta +20.01 us, se 0.61 (2 cells, sigma 0.1275%); D 10.02 us (z 16.3), D2 10.10 (16.2), SLOT 24.69 (-7.7), d 1.06 11.41 (14.1); laws at the nominal c_128 900 |

## floor: why there is no calibration

`score_floor.pages_of` keeps a counter cell only when `r6common.clock_rule` puts it in
[1640, 1710] MHz (`BAND_MHZ`, rental 6's counter clock rule). `tcommon.joint_cF` then needs 6
floor-bound NATIVE or SHARED cells at n 4 to 9 (`floor_rows`). Per arm, from the same
`clock_rule` (DESCRIPTIVE):

| page | NATIVE in band | SHARED in band | PRIVATE in band | NATIVE / SHARED MHz |
|---|---:|---:|---:|---|
| olmoe f128 | 0 of 9 | 0 of 9 | 6 of 9 | 1601.1 to 1639.4 |
| mixtral f128 | 0 of 9 | 0 of 9 | 7 of 9 | 1599.3 to 1639.9 |
| olmoe f64w8 | 6 of 9 | 6 of 9 | 9 of 9 | 1610.1 to 1663.3 |
| mixtral f64w8 | 2 of 9 | 2 of 9 | 9 of 9 | 1614.3 to 1657.2 |
| olmoe f64w4 | 6 of 9 | 6 of 9 | 9 of 9 | 1607.9 to 1667.5 |
| olmoe f32w4 | 8 of 9 | 8 of 9 | 9 of 9 | 1636.6 to 1682.6 |

Every page passes `lock_gate.check_page`; the band is what removes the cells. The floor registration
text does not restate the band (the bytes registration's `gates.clock` does); the scorer applies it
to every counter page, as written and tested before the pages.

## bytes: what the 19 rows are

| page | registered scored rows | in the clock band | why the rest dropped |
|---|---:|---:|---|
| olmoe-1b-7b b128g1 | 24 | 0 | clock (every cell 1601.6 to 1620.7 MHz) |
| mixtral-8x7b b64g1 | 16 | 16 | |
| qwen2-57b-a14b b128g1 | 16 | 3 | clock (13 cells) |
| mixtral-8x7b b128g16 | printed only (G 16) | 8 of 24 | clock (16 cells) |

On the 19 rows CLAMP and v3 are the same numbers except on the four Mixtral n 1 uniform and balanced
rows (CLAMP's floor at the active weights moves them by 0.04 points). The skewed rows read v3 / CLAMP
+2.5 to +7.5% and the row keys -0.4 to -27.9% (Mixtral b64g1 n 2 PT, -27.6 and -27.9%). w2
(printed): v3 rms 54.8% over 27 rows, worst +115%.

## e1cal: the clauses not printed

`tcommon.stop_rule` was not reached past the calibration, so the scorer prints one clause. Rebuilt
under r7-median-run3 (`r7common.rebuild`, the function `tcommon`'s timed-page loader calls;
DESCRIPTIVE, not the registered verdict's printed reason), two of the five E1-CAL pages fail G1:

| page | repeat rows | r7 non-clean | lock slips | run-guarded | lost cells | rebuilt G1 |
|---|---:|---:|---:|---:|---:|---|
| olmoe-1b-7b e128g1 | 216 | 1 (0.5%) | 0 | 1 | 0 | PASS |
| mixtral-8x7b e128g16 | 216 | 73 (33.8%) | 56 | 31 | 7 | **FAIL** |
| mixtral-8x7b e64g1 | 144 | 0 | 0 | 0 | 0 | PASS |
| qwen2-57b-a14b e128g1 | 144 | 39 (27.1%) | 33 | 12 | 4 | **FAIL** |
| mixtral-8x7b e128g1 | 72 | 6 (8.3%) | 4 | 3 | 0 | PASS |
| olmoe-1b-7b e128g1r (rep) | 216 | 7 (3.2%) | 1 | 7 | 0 | PASS |
| olmoe-1b-7b d9 (dead) | 36 | 0 | 0 | 0 | 0 | PASS |
| olmoe-1b-7b d15 (dead) | 36 | 0 | 0 | 0 | 0 | PASS |

`stop_rule` STOPs on any E1-CAL page whose state is not "counted", and the timed-page loader gives a
G1-failed page "counted in ALL only" (ALL) or "out of CLEAN" (CLEAN). So, read off the code, the stop
rule would also have STOPPED on these two pages had the floor calibrated. That reading is inferred
from the code; the scorer did not print it.

## Scorer check (no post-page fix)

- **census**: `tile-census.json` carries `path_mismatch` [] and 72 entries, each with
  `occupancy.ctas_per_sm`; the scorer reads both. Its rebuild of the registrations at the census
  occupancy (`register.py --check --census-score census.score.json`) leaves every registration file
  the same except tileA-floor's DESCRIPTIVE `mma_sync_design` block (32x128x64 w4 occupancy 5 to 4,
  c_counter_w1 442.14 to 443.20). tileA-bytes is the same, so the mechanical step the census asks for
  changes nothing scored. The committed registration files are not rewritten.
- **counter pages**: `design.block_m/n/k`, `num_warps`, `num_stages` equal the registered tile on all
  ten; G5 (histogram sha256, shuffle seed, counts sha256) passes on the four histogram pages.
- **timed pages**: `slip_policy` "cell", `min_clean_repeats` 0, the registered histogram file sha256
  and shuffle seed 20261410, `bincount_ok` on every row, and `cells.csv` row counts equal to the
  reports' (`r7common.rebuild` refuses otherwise) on all eight. The rep and dead scorers loaded four
  of them; the other four (E1-CAL) were checked here because score_e1cal.py returns before reading
  pages.
- `finalise.py . scripts/scoring/tileA <out>` exits 2 ("REFUSED: the stop rule reads STOP") and
  writes nothing.

## Open defects (recorded, not fixed: the scorers are registered files)

- **Import order.** `score_floor.py` imports `r3_timing_model_v4`, and `score_bytes.py` imports
  `r3_timing_model`, before `tcommon` puts `scripts/` on `sys.path`, so from the command line both
  stop with `ModuleNotFoundError` unless `PYTHONPATH` holds the repo and `scripts/`. The tests import
  them after `tcommon` and do not see it.
- **Key order.** `score_census.py` builds `occupancy` by iterating a set of tile names, so its key
  order (and floor's `census_occupancy`, copied from it) follows the hash seed. The data are the same
  on every run.
- **Census tool.** deepseek-v2-lite (E 64, k 6) raises "N rows per expert is not an integer token
  count" on all six tiles (N 16 to 128: N x 64 / 6 is not an integer). The census reads INCOMPLETE;
  rental B's deepseek-v2-lite cells would have no census occupancy.
