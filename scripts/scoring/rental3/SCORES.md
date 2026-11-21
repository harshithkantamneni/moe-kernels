# Rental 3 scores (2026-10-06)

The six scorers in this directory, run at 49f2198 on
`results/published/2026-10-06-nvidia_gh200_480gb-rental3-session` (908 run files, each matching
the run branch's SHA256SUMS; the 34 compiled `cuda_utils*.so` files are left out):

    MOE_HISTORY_REPO=<a clone with history> \
    python scripts/scoring/rental3/score_<part>.py . results/published/2026-10-06-nvidia_gh200_480gb-rental3-session scripts/scoring/rental3

Outputs: `e2e`, `floorlaw`, `zform`, `stages`, `flush`, `replicate` `.score.{json,txt}` beside the
scorers. Each JSON's `addendum.verdicts` lists every verdict path with its ALL, CLEAN and registered
reading (rental 2's addendum rule 3: the common verdict when the two agree, INCONCLUSIVE when they
differ, NOT SCORED with ALL's printed when CLEAN has no data). No scorer crashed or refused. One
post-page fix, to reading code in `score_e2e.py`, changed one printed verdict (E2, end of this file).
Five of the six outputs are byte-identical before and after it.

## Pages each view drops

| page | failed validity gates | ALL | CLEAN |
|---|---|---|---|
| e2e timed G = 32 (`f037a06b`, tag -p2-) | V5 UNKNOWN (b 3.09% of the private slope, band 2.42 to 3.64% against 3%) | counted | counted (V5 never gates timed pages, `gates.timed`) |
| e2e timed G = 8 (`3908df9d`), G = 3 (`5989ba0c`), deep G = 4, 2; rep8 G = 8 (`ef2ea3e8`) | none (C1 UNKNOWN, a claim gate) | counted | counted |
| e2e byte pages `r3c-g{3,8,32}` | V5 FAIL (PRIVATE w1 q_P 0.957 n, under the 0.97 n floor) | counted | excluded (every byte V-gate gates CLEAN for E3, E3s, E6) |
| every floor capture (base, 1710, 1410, 1005) | none | counted | counted (lock in force: null kernel -1.18 to -1.78% of the lock, L2 1697.0 to 1697.8 MHz at 1710, nvidia-smi on the lock before and after) |
| stage pages `r3c-g1` (l2base and five knobs) | none | counted | counted |

Granite-1B's 1710 capture has fewer than 3 cells of 0.5 ms or more, so its L2 leg is printed, not
gated, as registered. No floor cell is above its lock. The launch-floor directories carry no gate.

## Part E, end-to-end blind time on qwen2-57b-a14b-tp8 (`2026-10-05-rental3-e2e-gh200`)

Scope (registered): an unseen shape inside the 64-expert, top-8 family on which every
post-2026-09-28 term was chosen; not a test of transfer across expert counts.

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| E1 time from PREDICTED bytes (primary), 31 core cells | FALSIFIED | FALSIFIED | **FALSIFIED** (resolved) | rms 3.50%, worst -6.08% (PRIVATE G = 3 n = 2); every cell measured under M, by 9.7 to 29.3 us; by G rms 3.17 / 3.67 / 3.77% (G = 3 / 8 / 32), SHARED 2.91%, PRIVATE 3.92%; resolution 0.31% (sigma_page 0.18%, sigma_board 0.26%) |
| E1 deep pages (printed), 30 cells | FALSIFIED | FALSIFIED | printed | rms 2.83%, worst -5.94% (PRIVATE G = 4 n = 2) |
| E2 time from COUNTED bytes, 31 cells | FALSIFIED | FALSIFIED | **FALSIFIED** (post-page fix) | rms 2.52%, worst +4.72% (predicted / measured - 1), none beyond 5%; 29 of 31 over-predicted; before the fix NOT SCORED (22 of 31 cells, under 24) |
| E3 PRIVATE q, 54 cells | FALSIFIED | NOT SCORED: no byte page | **NOT SCORED** (CLEAN has no data); ALL: FALSIFIED | rms 3.34%, 14.8% beyond 5%; w1 -3.8 to -6.0% (median -4.4%, the V5 shortfall), w2 within 0.2% |
| E3s SHARED q (secondary), 54 cells | FALSIFIED | NOT SCORED | **NOT SCORED** (CLEAN has no data); ALL: FALSIFIED | rms 17.5%, 44% beyond 5%; w1 -4 to -40% (median -22%), w2 -2.5 to +0.1% |
| E4 w2 per-CTA unit (S 5, F 36% of u) on the 1005 DUR | M HOLDS (F0 FALSIFIED) | same | **M HOLDS (F0 FALSIFIED)** | b 0.984 (se 0.0002) in [0.98, 1.02], 11 cells; F0 band [0.597, 0.677]; theta 0.063 +- 0.096; w1 q-slope printed 0.997 |
| E5 M against M+Z (printed, not a test) | printed | printed | printed | rms M 3.50%, M+Z 1.89% (worst 3.52%), sigma_rms 0.26% |
| E6 dead-CTA term, w2 SHARED - NATIVE in-kernel | INCONCLUSIVE | NOT SCORED | **NOT SCORED** (CLEAN has no data); ALL: INCONCLUSIVE | median 29.37 us over 24 cells; DEAD band [29.62, 44.43] (predicted 37.0), FIXED [17.23, 25.85] (parent's 21.5); w1 0.14 us (predicted 0) |

The miss is a uniform over-prediction of 10 to 29 us a call, largest at small n. Counted bytes take
the rms from 3.50 to 2.52%: part of E1's miss is PRIVATE w1's 4.4% byte shortfall, the rest is the
time model. M+Z's -9.6 us shift would meet E1's bar; it is printed only, as registered (a board
offset reproduces it), and is not a result.

## Part R and RK, the noise models (`2026-10-05-rental3-replicate-gh200`)

| quantity | ALL | CLEAN | registered reading | key numbers |
|---|---|---|---|---|
| R sigma_page (rep8 against e2e G = 8, 9 cells) | RECORD | RECORD | 0.18% | -0.36 to +0.22% per cell; resolution with sigma_board 0.31% |
| RK per-capture sigma, tp4 1710 floor against floorrep | RECORD | RECORD | printed | w1 L 3,888, D_imb 4,307 cycles; w2 L 1,456, D_imb 471 cycles (11 and 10 cells) |
| K1 on rental 3's tp4 1710 pool | UNRESOLVED | UNRESOLVED | **UNRESOLVED** (as measured HOLDS) | 27 of 42 cells within 2 sigma of 5,000 cycles; extremes HOLDS / FALSIFIED |
| K4 on rental 3's tp4 1710 pool | UNRESOLVED | UNRESOLVED | **UNRESOLVED** (as measured HOLDS) | extremes HOLDS / FALSIFIED (H_IMB) |
| K1 on rental 2's pool (SEEN) | UNRESOLVED | UNRESOLVED | **UNRESOLVED**; rental 2's HOLDS stands | 38 of 60 cells within 2 sigma |
| K4 on rental 2's pool (SEEN) | UNRESOLVED | UNRESOLVED | **UNRESOLVED**; rental 2's HOLDS stands | extremes HOLDS / FALSIFIED (H_IMB) |

## Part B, the floor law (`2026-10-05-rental3-floorlaw-gh200`)

A GEMM capture enters a test only when se(theta) on that ruler is 0.15 or less; most 1710 DUR
and X fits are wider (se 0.16 to 1.48), so B0 and B4 count one GEMM each (tp4 floor w2).

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| B0 theta(X / u) in [-0.15, 0.15] at 1710 | FALSIFIED | FALSIFIED | **FALSIFIED** | tp4 floor w2 +0.21 (se 0.10), the only GEMM within the se rule |
| B1 theta_ACT.max in [0.70, 1.05] on 80% of lock captures | HOLDS | HOLDS | **HOLDS** | 14 of 16 (87.5%); out: qwen2-tp8 w2 1005 0.09, tp8 w2 1005 0.69 |
| B2 theta_DUR at 1005 in [0.70, 1.05] | FAILS on 1 | same | **FAILS on 1 of 8** | qwen2-tp8 w2 0.06 (fluid); others 0.72 to 0.97; granite-1B w1, w2 out on se |
| B3 dtheta(1005 - 1710), counted pairs | RECORD | RECORD | tp4 w2 only: 0.15 | theta 0.786 / 0.634; CLOCK needs 0.25; qwen2-tp8 w2 printed 0.06 / -0.91 |
| B4 theta_DUR at 1710 | RECORD | RECORD | CLOCK clause false, CEIL clause false | tp4 w2 only: theta_DUR 0.634, theta_ACT.max 0.763 |
| B5 X at 1005 (printed) | printed | printed | printed | 306 to 353 cycles |
| B6 wall-time floor outside ncu (GR) | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** | theta_wall 0.58 (se 0.14), corr(n, W) 0.22; E240 printed 0.57 (0.11) |
| B7 part 4 on tp8 w1's 1005 DUR | H_EST HOLDS | H_EST HOLDS | **H_EST HOLDS** | theta 0.964, delta -1.15% (printed), residual rms 0.12% |
| CLOCK | neither | neither | not held, not falsified | B0 fails, B3 and B4 clauses fail; B2 fails on 1 (2 falsify) |
| CEIL | FALSIFIED | FALSIFIED | **FALSIFIED** | B4's CEIL clause fails (tp4 w2 0.63) |
| FLUID | FALSIFIED | FALSIFIED | **FALSIFIED** | B1 holds |
| part B | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** | B6 inconclusive: the claim stays "ncu-measured" |
| X clock law (printed) | none | none | printed | tp4 X at 1710 / 1410 / 1005: w1 25,031 / 7,951 / 327, w2 24,006 / 7,918 / 328 cycles; LIN centre 14,519 / 13,930 |

## Part A, the per-GEMM constant's form (`2026-10-05-rental3-zform-gh200`)

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| qwen2-tp8 w2 (scored) | NOT SCORED | NOT SCORED | **NOT SCORED** (curvature) | 1710 half-fit intercepts 8,380 / 7,021 against limit 1,106 |
| granite-1B w2 A1 | PROP out; AFF, MIX in | same | as ALL | Z 3,540 at 1,673 MHz; PROP band [111, 2,398], AFF [2,483, 4,770], MIX [2,934, 5,222] |
| granite-1B w2 A2 base / 1710 | AFF, PROP FAIL; MIX passes | same | as ALL | r 0.842 +- 0.063 (2 sigma 0.125), r_MIX 0.858 |
| granite-1B w1 A1 (PROP, MIX scored) | PROP out; MIX in | same | as ALL | Z 3,719; PROP [881, 3,169], MIX [3,366, 5,653] |
| granite-1B w1 A2 | all pass | same | as ALL | r 0.951 +- 0.067, r_MIX 0.874 |
| qwen2-tp8 w1 (printed) | printed | printed | printed | Z 6,281 at 1,666 MHz; r 0.932 +- 0.061, r_MIX 0.922 |
| PROP | FALSIFIED | FALSIFIED | **FALSIFIED** | fails A1 on both scored GEMMs |
| AFF | not falsified | not falsified | fails on 1 of 2 | granite-1B w2 A2 |
| MIX | no failure | no failure | not selected | AFF is not FALSIFIED |
| MID | none | none | not read on the primary lever | 1005 lever (printed): granite-1B w2 alone reads MID (r' 0.811 +- 0.060, MIX 0.673), w1 0.913 (MIX 0.711), qwen2-tp8 w1 0.933 (MIX 0.819); MID needs 2 GEMMs |
| form | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** | K3's tp4 r printed: w1 0.924, w2 0.936 (AFF 1.0, MIX 0.829 / 0.821) |

## Part C, what lifts G = 1 survival (`2026-10-05-rental3-stages-gh200`)

Board check passes (l2base w2 s within 0.05 of rental 2's at 6 of 6 n). Every page passes every
validity gate and F/Mn. ds = s_knob - s_base, w2, n = 4..9.

| page | recorded (expected) CTAs/SM | stages, BLOCK_K | ds | class |
|---|---|---|---|---|
| k32s8 | 5 (4) | 8, 32 | -0.023 to +0.007 | base |
| k128s4 | 3 (2) | 4, 128 | +0.083 to +0.140 | unclassified (3 of 6 in the lifted band) |
| k128s3 | 4 (3): re-keyed | 3, 128 | +0.038 to +0.068 | base |
| k64s7 | 3 (2) | 7, 64 | -0.001 to +0.033 | base |
| k64s3 (control) | 4 (4) | 3, 64 | +0.011 to +0.040 | base |

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| DEPTH | FALSIFIED | FALSIFIED | **FALSIFIED** | k32s8, k64s7 base at 8 and 7 stages |
| WIDTH | FALSIFIED | FALSIFIED | **FALSIFIED** | k128s3 base at BLOCK_K 128 |
| U-OCC | FALSIFIED | FALSIFIED | **FALSIFIED** | k128s3 |
| U-DEPTH | FALSIFIED | FALSIFIED | **FALSIFIED** | k32s8, k128s3, k64s7 |
| OCC | consistent | consistent | survives | no page recorded 2 or fewer CTAs/SM, so OCC predicts base everywhere |
| NULL | consistent | consistent | survives | every classified page base |
| pairs | OCC / NULL and WIDTH / U-OCC: NO SEPARATION | same | as ALL | |
| part C | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** (survivors OCC, NULL) | far share printed 0.24 to 0.27 on every page (base pages >= 0.23) |

## Part D, the flush ladder (`2026-10-05-rental3-flush-gh200`)

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| P0 (printed) | HELD | HELD | printed | no host drift |
| qualifying cells | 3 | 3 | n = 1 of each arm | host-bound at E240, E360 and E480 |
| OVERLAP (Phi rises 33.8 us a 120 MiB step, band [21.77, 45.77]) | SELECTED | SELECTED | **OVERLAP SELECTED** | d1 +33.8, +34.9, +41.6 us; d2 +30.7, +27.2, +27.8 us (NATIVE, SHARED, PRIVATE); slopes 0.95, 0.92, 1.03 |
| SAT | FALSIFIED | FALSIFIED | **FALSIFIED** | 3 of 3 outside [-10, 10] |

## Session facts the scores do not carry

- Unit 13 (the tp4 wall-time check) exited 3 with every one of its 66 timed cells `ok`: the one
  failure is a host probe, not a cell. `hostprobe.csv`, pre phase, GR NATIVE n = 13: `refused`,
  "call: an iteration took 11.7 x the median: something in the body synchronised". B6 reads the
  timed GR cells only, so it has all 11 treads.
- The driver's recovery step retook nothing. It asked for a G = 32 timed retake (the V5 page) and
  passed `--model rental3`, the plan's name, where R3 needs a model; R3 exited 2 REFUSED before its
  first cell (`session/locked-r3/gh200-20261006T065851Z-t32b/`). An operations defect: nothing was
  measured and no page is affected. The registration counts the V5 page in both views, so the
  retake was not needed for any verdict.

## The post-page fix

`score_e2e.py`, `default_e2` (`git diff` of this directory). E2 is registered as "the E1 population
restricted to G with a byte page (3, 8, 32): every core cell", and `gates.timed` says V5 never gates
a timed page. The scorer's view kept the G = 32 page (V5 alone), but `cross_model_score.score`
builds the target through `r3_timing_model.build`, whose `admit` keeps VALID pages only, so the
page was dropped inside the call and E2 printed `NOT SCORED: too few population cells` (22 of 31
present, under 24). The fix admits the target's pages as the view chose them, and builds the source
fit (8x7B's CAL pages) first under the unchanged `admit`, passing it in through `score(source=)`.
No rule, band, cell set or estimator changes; `cross_model_score.py` and `r3_timing_model.py` are
untouched (their registered sha256 pins still pass). The printed verdict changes from NOT SCORED to
FALSIFIED (rms 2.52%), which is what the registration's text gives on its cell set. Test:
`tests/test_scoring_rental3.py::test_e2_default_admits_the_pages_the_view_counted`.

```diff
-    d = XS.score(CP.source_pages(), CP.C27, C3.TARGET, page_dirs, counters_dir)
-    return {"cells": {(c["arm"], c["G"], c["n"]): c["resid"] for c in d["cells"]}}
+    src = XS._build(CP.source_pages(), CP.C27)
+    admit = TM.admit
+
+    def admit_as_viewed(pages):
+        return admit([p if p.label == TM.PTF.VALID else
+                      dataclasses.replace(p, label=TM.PTF.VALID, failed=()) for p in pages])
+    TM.admit = admit_as_viewed
+    try:
+        d = XS.score(None, None, C3.TARGET, page_dirs, counters_dir, source=src)
+    finally:
+        TM.admit = admit
+    return {"cells": {(c["arm"], c["G"], c["n"]): c["resid"] for c in d["cells"]},
+            "target_pages": d["target_pages"]}
```

## A reading left to the owner (no scorer change)

Part C's registration says that when k128s3's recorded occupancy is not 3, "its column is re-keyed
and the pairs it alone separates (OCC / U-OCC) read NO SEPARATION". k128s3 recorded 4 CTAs/SM. The
scorer re-keys the column and computes the pair from it: at 4 CTAs/SM OCC predicts base and U-OCC
(through WIDTH) lifted, so it prints OCC / U-OCC SEPARATED and U-OCC FALSIFIED by k128s3. Reading
the text literally would print the pair NO SEPARATION; whether U-OCC's falsification by the same
page then stands is a judgment, so the scorer was not changed. The part's verdict is INCONCLUSIVE
either way (OCC and NULL survive).
