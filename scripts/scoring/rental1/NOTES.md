# Rental 1 scorer notes

Readings written by hand into `atile.score.txt` and `l2.score.txt` after the scorers ran
(commit 2ac9512, 2026-10-01). A rerun of either scorer does not print them, so they were moved
here on 2026-10-05 and each `.score.txt` is now exactly what its scorer writes (rerun with the
arguments in `README.md`; `score_l2.py` takes `<results> <registration json> <out>`). The text
below is unchanged; every number in it is read off the committed `.score.json` beside it.

## atile.score.txt (was its first 17 lines)

The "Neither pass nor fail" reading of T1 on Qwen2-57B is the scorer's `NOT HELD (inside
neither pass nor falsifier: within-2% pass fails, nearer R1 everywhere)` in
`atile.score.json` (`T1.qwen2-57b-a14b.verdict`). The registration
(`docs/registered/2026-10-01-atile-ksteps-gh200.json`, T1) defines only a pass and a
falsifier, no INCONCLUSIVE, so the verdict to quote is NOT HELD. The `diag.txt` the first line
names is not committed.

```
VERDICTS (registered rules as written; numbers below and in score.json, diagnosis in diag.txt)
T1 Qwen2-57B PRIVATE w1 G=128: NOT FALSIFIED (nearer R1 than R0 at every n; R0 off -8.46 to -11.55% at n>=2), but the
   registered pass (within 2% of R1 at every n>=2) is MISSED at n = 6, 7, 9 (+2.18, +2.11, +2.64%). Neither pass nor fail.
T1 OLMoE PRIVATE w1 G=128: HELD (worst +0.52% from R1 at n>=2; R0 off -5.41 to -7.32%).
T2 rho42 1.199, rho84 1.184: outside every candidate's band, both between the R1/R0 bands (upper 1.087/1.104, 1.061/1.062)
   and the R2 bands (lower 1.640/1.475, 1.338/1.267): by the registered rule EVERY candidate is refuted (R0, R1, R2 primary
   and secondary each FALSIFIED: both rho outside its band); the k-step rise exists but not in R2's form.
T3 G=1 credit: tp8 PASS (q-1 -0.00438 vs R1 -0.00446, R0 band [-0.001, 0.001]); tp4 PASS (-0.00138 vs R1 -0.00193, band
   [-0.00293, -0.00093]; R0 band [-0.00089, 0.00111]); tp2 RECORD (-0.00044 vs R1 -0.00039, R0 +0.00025).
T4 (270 PRIVATE cells, G=1 aside): R0 FALSIFIED (rms 3.68%, 24 cells G>=32 n>=6 beyond 4%), R1-cal FALSIFIED (2.87%, 15),
   R2-cal FALSIFIED (rms 1.55% passes, 10 cells beyond 4%: 8x7B G32 w2 n7-9, tp2 G64 w2 n7-9, Qwen2 G128 w2 n6-9),
   R1-all8 FALSIFIED (2.73%, 14), R2-all8 FALSIFIED (1.77%, 10).
Pages: every PRIVATE cell scored. Exit 3 pages fail V10 (clock; does not gate, registered), V7 (NATIVE vs SHARED bytes),
   V6 (tp8 G=8, NATIVE vs SHARED L2 sectors at n=2 w2, 1.3031% vs 1.3%): none is a PRIVATE gate. Exit 1 pages (sameboard
   G=8, Qwen2 G=128) fail only CLAIM gates (both: C1/C2 REFUSE on SHARED's bracket; Qwen2 also C6 FAIL, PRIVATE's
   excess over n, the quantity scored here): no validity failure, scored as is.
```

## l2.score.txt (was its last 12 lines)

Summaries of `far_share`, `secondary_G2` and the page gates in `l2.score.json`.

```
far-share RECORD (cells with s >= 0.1; band [0.15, 0.45]):
  tp2 w1: 2 cells, range 0.17..0.23, outside []
  tp2 w2: 8 cells, range 0.28..0.37, outside []
  tp4 w1: 8 cells, range 0.19..0.25, outside []
  tp4 w2: 8 cells, range 0.14..0.39, outside [(8, 0.137), (9, 0.137)]
  tp8 w1: 8 cells, range 0.15..0.39, outside []
  tp8 w2: 8 cells, range 0.06..0.39, outside [(5, 0.115), (6, 0.097), (7, 0.081), (8, 0.062), (9, 0.062)]
  sameboard 8x7B w2: 8 cells, range 0.31..0.44, outside []
  total 50 cells, 7 outside the band
secondary G=2 shared: 54 cells, 31 beyond 5%, worst -58.8%
secondary G=2 private: 54 cells, 0 beyond 5%, worst +0.4%
V6 passes on every scored page: no s flagged. V10 fails on all six tp pages (does not gate survival); C1 CLAIM fails on tp4 and tp8 G=2 (the group_reads claim, not scored here). Same-board anchor page passes every gate.
```
