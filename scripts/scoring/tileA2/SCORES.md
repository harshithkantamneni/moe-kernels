# Tile rental A2 scores (2026-10-21)

The nine A2 scorers in this directory, run on
`results/published/2026-10-21-nvidia_gh200_480gb-tileA2-session` (936 run files, each matching the
run branch's SHA256SUMS; the 31 compiled `cuda_utils*.so` files are left out), in the order their
inputs need (finalise and the decode gate read session, census, floor and e1cal; bytes, rep, dead and
e1cal read the band floor recorded):

    for p in session probe census floor bytes rep dead e1cal convl; do
      PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/tileA2/score_${p}_a2.py . \
        results/published/2026-10-21-nvidia_gh200_480gb-tileA2-session/results scripts/scoring/tileA2
    done

Outputs: `session`, `probe`, `census`, `floor`, `bytes`, `rep`, `dead`, `e1cal`, `convl`
`.score.{json,txt}` beside the scorers, and the merged census the census scorer writes,
`census-merged/2026-10-20-nvidia_gh200_480gb-tile-census-merged-ta2m/tile-census.json`. Every scorer
exited 0. **No post-page fix**: no scorer, registration or rule was changed after the pages existed;
`register_a2.py --check` reads `same` on all 22 files, and rental A's `register.py --check` on its own.
The first scoring (2026-10-20, on the fetched run tree, into a scratch directory) and this one agree:
all nine `.score.txt` files, the printed outputs, the merged census and six of the nine JSON files are
byte-identical; census, probe and session JSON differ only in path strings (the run tree's location
against this directory's, and the output directory in census's merged-file path). A third scoring of
the fetched tree in this checkout reproduces the first byte for byte except census's output path.

## The verdicts

| registration | test | registered verdict | numbers |
|---|---|---|---|
| tileA2-gates (session) | the lock and the power limit rental B2 would run at | **NOT READ: the two power readings disagree (900.0 and 700.0 W)** | lock 1410 MHz; `clocks-at-start.txt` power.limit 900 W, `power-set.txt` GPU Current Power Limit 700 W |
| tileA2-gates (probe) | the lock-hold probe, ladder 1710, 1410, 1305 (DESCRIPTIVE reading of the gate records) | **1710 FAILS; 1410 PASSES; A2's lock 1410** | 1710: Mixtral lp1710m 4 LOST cells, 33 of 144 rows non-clean (22.9%); 1410: Mixtral 0 of 144, Qwen2-57B 0 of 72; OLMoE lp1410f P5 PASS in the fallback band |
| tileA2-census | VALIDITY and occupancy; the merge | **VALID (COMPLETE); MERGED** | 84 entries, 0 errors; equality EQUAL over 72 rental-A entries; OLMoE n 1 / n 3 SAME, deepseek-v2-lite admitted at tread 3; occupancy 2 / 3 / 3 / 4 / 3 / 4 as rental A |
| tileA2-floor | CAL-A at 1410 | **CAL-A** in the fallback band [1352.19, 1410] | c_128 691.65 +- 9.87, F_w1 2898 +- 582, F_w2 3346 +- 323; 34 cells, 6 series, rel rms 0.55% |
| | 64x128 test (DESCRIPTIVE), selection, F form | **B unreachable, W HOLDS, T FAILS; SELECTED W; F-step** | c_64 fit 418.3 +- 9.4; W within 2.4% on all four rows, T -14.5% to -18.4% |
| tileA2-bytes | the A key | **NOT NAMED (nearest v3); v4 takes CLAMP** | 45 rows: v3 8.70%, ROWKEY60 9.78% (1.12x), CLAMP 10.20%, ROWKEY30 14.38%; worst v3 row 20.4% |
| tileA2-rep | sigma_page at 128x128, 1410 | **MEASURED 0.1199%** | 24 cells, mean shift -0.094% |
| tileA2-e1cal | E1-CAL; tileB2-stop | **E1-CAL FAILS; STOP: rental B does not run** | 42 NATIVE cells: rms 4.23%, worst -8.52%, mean -2.31% |
| tileA2-dead | SEEN-REPLICATE (DESCRIPTIVE) | **SLOT NOT EXCLUDED; D, D2, d 1.06 EXCLUDED; kappa NOT SCORED** | delta 25.35 us, se 0.62; SLOT 24.69 (z 1.08), D 10.73 (23.7), D2 10.76 (23.7), d 1.06 12.11 (21.5), at the measured c_128 |
| tileA2-convl | CONV-L, 1410 against 1710 on R3's tile | **HOLDS** | G1 7 cells rms 0.61% (mean +0.38%, worst +1.00%); G8 8 cells rms 0.16% (mean -0.13%) |
| decode-tilesource | the decode gate | **GO** (`decode_gate_a2.py`, exit 0) | the floor calibration is scored and `finalise.selections` reads the directory; A2's stop rule printed beside: STOP |

`finalise_a2.py . scripts/scoring/tileA2 <out>` exits 2 ("REFUSED: no READ session.score.json ...")
and writes nothing. It would also refuse on the STOP (it requires tileB2-stop to read GO).

## e1cal: where the 42 cells miss

e = T_v4 / T_meas - 1 per NATIVE cell, v4 at 1410 from A2's selections (c_128 691.65, F_wg 2898 /
3346, W, F-step, CLAMP, sigma_page 0.1199%):

| page | cells | rms | worst | mean |
|---|---:|---:|---:|---:|
| olmoe-1b-7b e128g1 | 10 | 4.85% | -6.60% | -4.66% |
| mixtral-8x7b e128g16 | 12 | 1.69% | -2.92% | -1.38% |
| mixtral-8x7b e64g1 | 8 | 3.94% | +5.29% | +3.57% |
| qwen2-57b-a14b e128g1 | 8 | 6.22% | -8.52% | -6.06% |
| mixtral-8x7b e128g1 | 4 | 3.49% | -3.85% | -3.48% |

OLMoE's four n 1 uniform and balanced cells are dropped by G2 (host-bound by rule). The class means
are the stop rule's: 64g1 +3.574% (Mixtral only), 128g1 -4.955% (Mixtral -3.48, OLMoE -4.66,
Qwen2-57B -6.06), 128g16 -1.376%; the 128 G 1 models' spread is 2.581%. SHARED (printed): rms 5.16%.

G3 (uniform minus balanced, rental 7's TOST) reads "FAIL in strata ['1'] (heterogeneous: only those
strata's histogram cells void)": stratum 1 (each page's second tread) is NOT EQUIVALENT (mean +0.52%,
ci -0.81% to +1.85%; Mixtral e64g1's cluster mean +2.22%), strata 0 and 2 EQUIVALENT. `score_e1cal.py`
voids cells only when every stratum fails (`void_strata == "ALL"`) and records `g3_void` false, so the
42 cells include stratum 1's. That is the registered scorer's behaviour and is recorded, not changed;
how the bar would read without stratum 1's cells is not computed here.

## bytes: what the 45 rows are

| page | rows scored | dropped by the band |
|---|---:|---:|
| olmoe-1b-7b b128g1 | 22 | 2 (uniform n 4, both arms) |
| mixtral-8x7b b64g1 | 16 | 0 |
| qwen2-57b-a14b b128g1 | 7 | 9 (n 1 and n 2) |
| mixtral-8x7b b128g16 | printed only (G 16) | 4 (uniform and balanced n 3) |

By label (rms): PT v3 / CLAMP 11.2%, ROWKEY60 13.0%; PW v3 / CLAMP 10.5%, ROWKEY60 9.3%; balanced v3
3.6%, CLAMP 11.4%, ROWKEY60 6.7%; uniform v3 3.7%, CLAMP 5.1%, ROWKEY60 7.4%. CLAMP and v3 differ only
on the uniform and balanced rows. w2 (printed): v3 rms 43.1% over 65 cells, worst +113%.

## The timed pages under r7-median-run3

Rebuilt from `cells.csv` by `r7common.rebuild` at each page's own lock, every timed page but one passes
G1 with no non-clean row: the five E1-CAL pages, the replicate, the dead-CTA pair, the six CONV-L pages
(three at 1410, three at 1710) and the two 1410 probe pages. The exception is the 1710 probe page,
Mixtral lp1710m: 33 of 144 rows non-clean (25 lock slips, 15 run-guarded), 4 cells lost.

## Scorer check (no post-page fix)

- **census**: A2's `tile-census.json` carries `path_mismatch` [] and the equality gate compares its 72
  entries with rental A's (no drift). The merge writes 84 entries (rental A's 72 and deepseek-v2-lite's
  12 at tread 3).
- **counter pages**: all eleven record lock 1410 and pass `lock_gate.check_page`; the floor scorer
  recorded the band in force as the fallback ("the primary band left joint_cF None"), and bytes, rep,
  dead and e1cal read it from `floor.score.json`.
- **timed pages**: every page records the registered histogram file and lock; `r7common.rebuild`
  accepts every `cells.csv` (row counts equal the reports').
- **decode gate**: `decode_gate_a2.py scripts/scoring/tileA2` prints GO and score_cal's command
  (`score_cal.py <tree> <out> --tile-scores scripts/scoring/tileA2`).

## Open defects (recorded, not fixed)

- **The session record.** `score_session_a2.py` requires `clocks-at-start.txt` power.limit and
  `power-set.txt`'s first Current Power Limit to agree within 1 W. On this run the first (NVML
  `--query-gpu=power.limit`) reads 900 W and the second (`nvidia-smi -q -d POWER`, GPU scope) 700 W.
  The disagreement is not confined to the read-back just after `-pl`: every `nvidia-smi -q` read of
  the session reads 700 W (six files and 1,612 per-second samples during the probe, 22:47Z to 23:14Z),
  and every `--query-gpu` read 900 W. So the record cannot be read on any timing of the read-back, and
  which limit the board enforced is not established by the run files. The registered refusal stands;
  rental B2 cannot be finalised from this directory whatever its stop rule reads.
- **Ledger labels.** The driver's `step_what` prints "at the 1710 lock" / "at 1710" for bytes and
  timed units from `LOCK_TIMED` before the unit reads `a2-lock.env`; the pages record 1410.
- **Key order.** `score_census.py` builds `occupancy` from a set of tile names (rental A's open
  defect 2); `PYTHONHASHSEED=0` fixes the order for these files.
