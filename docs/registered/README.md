# Registered predictions

A prediction is scored against a page only if it was committed here before
the page existed. Each file names the fit it came from; nothing in it is
fitted on what it predicts.

## 2026-09-27: Mixtral 8x22B on a GH200, from Mixtral 8x7B's fit

`2026-09-27-mixtral-8x22b-gh200.json` (every cell) and `.txt` (the table),
written by `scripts/cross_model_predict.py` from the 2026-09-27 GH200 session
(`results/published/2026-09-27-nvidia_gh200_480gb-session`, board 9b6d01):
its five lock-1710 timed pages (G = 2, 3, 4, 8, 32) fitted by
`scripts/r3_timing_model.py` at the knee p = 14 (T0 0.0540 ms, c 206.69 ns =
353.4 cycles per CTA k-step at 1710 MHz, bw 3601 GB/s), and its eight lock-1710
counter pages fitted by `scripts/wave_split_bytes.py` in the registered mix
view. Only the shapes change (w1 K 6144 x 512 N-tiles, 3.221 GB; w2 K 16384 x
96 N-tiles, 1.611 GB, from `moe/spec.py`'s `mixtral-8x22b`, read off the
model's config.json). The 8x22B session runs on whatever GH200 board Lambda
gives; the two GH200 boards measured so far agree within about 1% on every
shared cell, and a different board is part of what this tests.

What falsifies what, stated before any 8x22B page:

- **The floor is the tile's, not the model's.** The floor capture reads the
  same cycles per CTA k-step as 8x7B's (350 to 352 on w1, 354 to 367 on w2):
  falsified by w1 outside 340 to 365. At G >= 8 the SHARED slope over treads 2
  to 6 is 0.9270 ms per tread (1.714 x 8x7B's, the ratio of N-tiles x k-steps):
  falsified outside 0.908 to 0.946 (2%).
- **Time.** Every timed SHARED and PRIVATE cell at G = 2, 3, 4, 8 and 32,
  treads 1 to 9: the rms of (predicted / measured - 1) over them at or under
  2%, and no cell beyond 5%. The source fit's own leave-one-G-out rms was 0.30%;
  the margin is the untested shape change, not a fitted tolerance.
- **Bytes.** Every w1 cell and every PRIVATE cell within 5% of the prediction;
  w2 SHARED at G <= 16, n <= 4 within 5%. w2 SHARED at n >= 5 and every
  OUT-OF-DOMAIN cell are printed and scored as open questions only: the mix
  view missed those by 5 to 12% on 8x7B itself.
- **The G=2 zig-zag and G=3's period-3 ripple** keep their phase: at G=2 the
  steps n = 2 to 3, 4 to 5 and 6 to 7 exceed n = 3 to 4, 5 to 6 and 7 to 8, and
  the G=3 steps repeat with period 3 as the table's do.

**Scored 2026-09-28** on `results/published/2026-09-28-nvidia_gh200_480gb-8x22b-session`
(board 435984; its session README has every cell): the floor HELD (w1 346 to
347 cycles per CTA k-step), the G >= 8 slope HELD (0.9155, 0.9141), time
HELD at the 1710 lock (40 cells at G=3, 8, 32, rms 1.71%, worst -4.73%; G=2
and 4 ran at 1605 on the module's power cap and are not scored), the G=2
zig-zag and G=3 ripple HELD; bytes FALSIFIED on 5 of the 240 named cells
(PRIVATE w1 G=64 n=9, PRIVATE w2 G=64 n=5, w2 SHARED G=3 n=3, G=4 n=4, G=8
n=4, 5.3 to 6.7%). The files above are unchanged; changes made after these
pages (the timing model's partial-wave rule, the byte model's stage 3) are
scored in docs/FINDINGS.md as post-registration.

## 2026-09-28: Qwen2-57B-A14B (64 experts, top 8) on a GH200, from Mixtral 8x7B's fit

`2026-09-28-qwen2-57b-a14b-gh200.json` (every cell) and `.txt` (the table),
written by `scripts/cross_model_predict.py` at ae62402 from the same 2026-09-27
GH200 session (board 9b6d01): its five lock-1710 timed pages fitted by
`scripts/r3_timing_model.py` (p = 14, the partial-last-wave rule; T0 0.0529
ms, c 206.58 ns = 353.3 cycles per CTA k-step at 1710 MHz, bw 3598 GB/s) and
its eight lock-1710 counter pages fitted by `scripts/wave_split_bytes.py` in
the MIX view (LATER_MISS and stage 3 in force). Everything fitted is 8x7B's.
What changes is the shapes (w1 K 3584 x 80 N-tiles, 2.349 GB; w2 K 2560 x 56
N-tiles, 1.174 GB, from `moe/spec.py`'s verified `qwen2-57b-a14b`) and the
design: 64 experts, 9 copies (576 slots for SHARED and PRIVATE, 64 for
NATIVE), NATIVE on vLLM's block-scan alignment kernel at every tread. The
session runs `scripts/gh200_model_session.sh --model qwen2-57b-a14b`
(`--declared-copies 9` on every timed page) on whatever GH200 board Lambda
gives.

What falsifies what, stated before any Qwen2-57B page:

- **The floor is the tile's.** The floor capture reads 340 to 365 cycles per
  CTA k-step on w1 (8x7B 350 to 352, 8x22B 346 to 348) though a CTA runs 56
  k-steps, not 64 or 96. The G >= 8 SHARED slope over treads 2 to 6 is 0.6837
  ms per tread at G=8 and at G=32: falsified outside 0.670 to 0.697 (2%).
- **Time.** Every timed SHARED and PRIVATE cell at the 1710 lock, treads 1 to
  9: rms of (predicted / measured - 1) at or under 2%, no cell beyond 5%.
  Cells measured at another lock (the module's power cap slipped 8x22B's deep
  ladder) are printed and not scored.
- **Bytes.** Every w1 cell and every PRIVATE cell within 5%; w2 SHARED at
  G <= 16, n <= 4 within 5%. The 64-expert signature, scored with them: w1
  SHARED at G=1 reads 1.16, 1.21, 1.22, 1.23 weight sets at n = 2 to 5 (a
  co-residency window now spans several experts' M-tiles; on 8x7B it climbs
  toward n), and w2 SHARED at G=1 stays at 1.00 to 1.08. The 8 ILL-POSED and
  36 OUT-OF-DOMAIN w2 cells are printed and scored as open questions only.
- **G=2's zig-zag** keeps its phase: the SHARED steps n = 2 to 3, 4 to 5, 6 to
  7 and 8 to 9 (0.795 0.788 0.810 0.804 ms) exceed n = 3 to 4, 5 to 6, 7 to 8
  (0.581 0.578 0.575).

The stated risk, before the pages: the byte model's first-window and
activation parameters are keyed by GEMM name, and at E = 64 w1's co-residency
window is 4.1 M-rows (8x7B's w1 0.7, its w2 4.1). If w1 misses and w2 holds,
the parameters encode the window regime, not the GEMM, and the model must be
re-keyed on the window; that would be a finding, not a refit. The G=3 ripple
is predicted weak here (steps 0.426 0.634 0.714 | 0.655 0.681 0.702 | 0.688)
and is not registered as a falsifier.

**Scored 2026-09-29** on `results/published/2026-09-28-nvidia_gh200_480gb-qwen2-57b-session`
(board 50e61f; its session README has every cell): the floor HELD (w1 353.3 to
353.7), the G >= 8 slope HELD (0.6962, 0.6953, at the band's upper edge), time
FALSIFIED (58 cells at 1710, rms 2.71% against 2%, worst -5.00%: a uniform
2.3 to 2.7% bias), G=2's zig-zag HELD in phase (amplitude a fifth), bytes
FALSIFIED (w1 14.3% rms; the G=1 w1 signature read 1.23 to 1.91 against 1.16
to 1.23; w2 SHARED at G = 2 to 4 up to +97%; PRIVATE w2 1.89%). The stated risk
(w1 in Mixtral w2's window regime) is among the causes; the files above are
unchanged.

## 2026-09-29: OLMoE-1B-7B (64 experts, top 8, small K) on a GH200, from Mixtral 8x7B's fit

`2026-09-29-olmoe-1b-7b-gh200.json` and `.txt`, written by
`scripts/cross_model_predict.py` from the 2026-09-27 GH200 session (board
9b6d01), the same pages and fit as the Qwen2-57B registration, with the model
as it stands after the Qwen2-57B diagnosis: the dead-CTA term (ea2c77c, 1.333 ns
per dead CTA, measured on 8x7B's counters) and the capacity rule for later
re-reads (15a9533). Every fitted number is 8x7B's (T0 0.0441 ms, c 206.58 ns =
353.3 cycles per CTA k-step, bw 3598 GB/s). Shapes from `moe/spec.py`'s
`olmoe-1b-7b` (config.json read 2026-09-29): w1 K 2048 x 32 N-tiles, 0.537 GB;
w2 K 1024 x 32 N-tiles, 0.268 GB: a CTA runs 32 and 16 k-steps (8x7B 64 and
224), and w1's co-residency window is 10.3 M-rows (8x7B's 0.7). Design: 64
experts x 9 copies, NATIVE on block-scan at every tread.

**This is a test of the timing model, not of the byte model.** Every OLMoE
re-read sits at 0.07 to 1.0 x the L2 in LRU reuse distance, most of them below
0.3, where no page has measured survival and where a single card-level law was
refuted on 8x7B and Qwen2-57B together (2026-09-29, docs/FINDINGS.md). The
byte predictions are registered and printed; they are not a falsifier except
PRIVATE's.

What falsifies what, stated before any OLMoE page:

- **Time, the primary test.** `scripts/cross_model_score.py` at this commit,
  8x7B's fit from the 2026-09-27 pages held, every VALID lock-1710 timed cell
  of OLMoE priced from its own counter pages' counted bytes: SHARED and
  PRIVATE rms at or under 2%, no cell beyond 5%. The same scorer reads
  Qwen2-57B at 1.52% (its diagnosis data). Cells at another lock are printed,
  not scored.
- **The floor is the tile's.** The floor capture reads 340 to 365 cycles per
  CTA k-step on w1 though a CTA runs 32 k-steps; a per-CTA fixed cost the
  model does not have would show here first.
- **The G >= 8 SHARED slope** over treads 2 to 6 is 0.1597 ms per tread at
  G=8 and G=32 (predicted bytes): falsified outside 0.1565 to 0.1629 (2%).
- **PRIVATE bytes**, w1 and w2, every G and tread, within 5% of the JSON's q.
- **Printed, not scored:** time from predicted bytes (the JSON's T); w1 and w2
  SHARED/NATIVE bytes, the first measurements of L2 survival below 0.3 x L2.

## 2026-09-29: the same OLMoE test on the H100 SXM5, from that card's own 8x7B fit

`2026-09-29-olmoe-1b-7b-h100.json` and `.txt`: the OLMoE registration above,
made the second card's way, so the test runs on whichever of the two cards
Lambda has first (both, if both come). The fit is the 2026-09-25 H100 80GB HBM3
session's (board b533dd): its five lock-1710 timed pages (G = 1, 2, 4, 16, 64;
T0 0.0242 ms, c 205.76 ns = 351.8 cycles per CTA k-step, bw 2971 GB/s) and its
base-clock counter pages (the only ones that card has). That fit reproduces
its own pages to 1.32% rms, against the GH200's 0.26%, so the 2% bar has less
room on this card. Before any page, the two cards' fits agree on OLMoE: the G >=
8 SHARED slope 0.1588 (G=8) and 0.1583 (G=32) ms per tread here, 0.1597 on the
GH200.

The falsifiers are the GH200 registration's, with this card's numbers: time
from OLMoE's own counted bytes (`scripts/cross_model_score.py`, source = the
2026-09-25 H100 pages above) at rms at or under 2% and no cell beyond 5%; the
floor on w1 inside 340 to 365 cycles per CTA k-step; the G >= 8 SHARED slope
inside 0.1556 to 0.1620 (G=8) and 0.1551 to 0.1615 (G=32); PRIVATE bytes within
5%; SHARED/NATIVE bytes and time from predicted bytes printed only.

**Scored 2026-09-29** on `results/published/2026-09-29-nvidia_gh200_480gb-olmoe-session`
(board d663f7, the GH200 registration; the H100 registration was not run, the
GH200 being available first): time from its own bytes FALSIFIED (76 cells, rms
3.53%, 12 beyond 5%), the G >= 8 slope FALSIFIED (0.1705 and 0.1707 against the
band 0.1565 to 0.1629), w1's floor HELD (360 cycles), PRIVATE bytes HELD (w1
1.97%, w2 0.30%, none beyond 5%); printed: time from predicted bytes 3.83%,
SHARED bytes w1 9.8%, w2 2.9%. The files above are unchanged.

## 2026-09-29: four held-out models on a GH200, from Mixtral 8x7B's fit with the per-CTA fixed cost

`2026-09-29-{qwen1.5-moe-a2.7b,granite-3.0-3b-a800m,phi-3.5-moe,jetmoe-8b}-gh200.json`
and `.txt`, written by `scripts/cross_model_predict.py` at 0f77622 from the
2026-09-27 GH200 session (board 9b6d01), the model as it stands after all four
measured models were used as calibration or diagnosis data: the dead-CTA term,
the capacity rule for later re-reads, and the per-CTA fixed cost F (0f77622:
measured on the four models' counters, c 344.1 cycles, F 520 cycles on w1 and
979 on w2). The timing fit is 8x7B's alone (T0 0.0442 ms, c 202.55 ns, bw 3598
GB/s). None of these four models has a page; their configs were read from
their config.json on 2026-09-29 (`moe/spec.py`). Each runs as its own session
of `scripts/gh200_model_session.sh --model M` on whatever GH200 board Lambda
gives, 9 copies declared on every page.

| model | E, top-k | k-steps w1 / w2 | registered floor per CTA k-step, w1 / w2 | G >= 8 SHARED slope (ms per tread) |
|---|---|---|---|---|
| qwen1.5-moe-a2.7b | 60, 4 | 32 / 22 | 360.4 / 388.6 | 0.2159 |
| granite-3.0-3b-a800m | 40, 8 | 24 / 8 | 365.8 / 466.5 | 0.0440 |
| phi-3.5-moe | 16, 2 | 64 / 100 | 352.2 / 353.9 | 0.4898 (G=8), 0.4896 (G=32) |
| jetmoe-8b | 8, 2 | 32 / 88 | 360.4 / 355.2 | 0.1076 |

What falsifies what, per model, stated before any of its pages:

- **Time, the primary test.** `scripts/cross_model_score.py` at this commit,
  8x7B's fit held, every VALID lock-1710 timed cell priced from the model's
  own counted bytes: SHARED and PRIVATE rms at or under 2%, no cell beyond 5%.
- **The floor.** Each GEMM's cycles per CTA k-step on the floor capture (the
  slope of `sm__cycles_elapsed.avg` over CTA k-steps per SM) within 2% of the
  table: the per-CTA term's direct test, sharpest on Granite's 8-k-step w2.
- **The G >= 8 SHARED slope** over treads 2 to 6 within 2% of the table.
- **PRIVATE bytes**, w1 and w2, every G and tread, within 5% of the JSON's q.
- **Printed, not scored:** time from predicted bytes (the JSON's T); SHARED and
  NATIVE bytes (an expert's weights fit in the L2 on all four, the region where
  no survival law is calibrated).

All four models were chosen for the ways they differ from what the model has
seen: an expert count between 8 and 64 (Phi 16, Granite 40, Qwen1.5 60), top-k
4 (Qwen1.5), a CTA shorter than any measured (Granite w2, 8 k-steps), and
Mixtral's own design at another shape (JetMoE).

**Scored 2026-09-30, Qwen1.5-MoE-A2.7B** on
`results/published/2026-09-29-nvidia_gh200_480gb-qwen1.5-session` (board
d67185, five VALID lock-1710 pages): time from its own bytes HELD (76 cells,
rms 1.87%, worst +4.81%, none beyond 5%), the floor HELD (w1 361.0, w2 385.8
cycles per CTA k-step at base, +0.17% and -0.73%; every cell at 7 or more
waves), the G >= 8 slope HELD (0.2156 and 0.2154, -0.15% and -0.21%), PRIVATE
bytes HELD (w1 1.31%, w2 0.53%, none beyond 5%); printed: time from predicted
bytes 2.12%, SHARED bytes w1 7.4%, w2 3.9%.

**Scored 2026-09-30, Phi-3.5-MoE** on
`results/published/2026-09-29-nvidia_gh200_480gb-phi3.5-session` (board
d67185, five VALID pages): time HELD (76 cells, rms 0.57%, worst -1.27%), the
floor HELD (w1 349.6, -0.73%; w2 355.4, +0.42%, the lock capture 354.4), the
slope HELD (0.4839 and 0.4828, -1.21% and -1.39%), PRIVATE bytes w1 HELD
(0.36%) and w2 FALSIFIED (1.81% rms, but 3 cells beyond 5%: G=64 n = 7, 8, 9 at
-5.1, -8.0, -6.8%); printed: time from predicted bytes 0.65%, SHARED bytes w1
10.8%, w2 18.5%.

**Scored 2026-09-30, JetMoE-8B** on
`results/published/2026-09-29-nvidia_gh200_480gb-jetmoe-session` (board
d67185, four VALID lock-1710 pages; G=2 slipped at 1710 twice and held at
1605, not scored): time FALSIFIED (58 cells, rms 7.17%, 12 beyond 5%: every
SHARED and PRIVATE cell at n = 1, -15 to -20%, and SHARED at n = 3, +10 to
+11%; n >= 4 at 1% or under), w1's floor HELD (360.3, -0.02%), w2's floor
FALSIFIED (368.0 against 355.2, +3.6%, on cells of 0.97 to 2.91 waves; the
estimator bias corrected below, for later pages only), the slope FALSIFIED
(0.1135 and 0.1105, +5.4% and +2.6%), PRIVATE bytes HELD (w1 0.19%, w2 0.59%);
printed: time from predicted bytes 7.16%, SHARED bytes w1 20.1%, w2 11.0%.

**Scored 2026-09-30, Granite-3.0-3B-A800M** on
`results/published/2026-09-30-nvidia_gh200_480gb-granite-session` (board
1310e2, the GH200 registration with the corrected floor estimator below; the
H100 registration was not run): time NOT SCORABLE and the slope NOT SCORABLE,
every one of the ten lock-1710 timed pages INVALID on V5 (eight UNKNOWN, two
FAIL), nine of them also failing the claim C2 (PRIVATE's apparent stream 4169
to 7005 GB/s against the 3726 GB/s ruler); SHARED and NATIVE calls read 0.238
to 0.291 ms at every n from 1 to 5, and the printed SHARED slopes are 71 to 84%
under 0.0440. w2's floor HELD on n = 3, 4, 6 (459.8 cycles per CTA k-step on the
base-clock capture, -1.43%; the lock capture, FL1 FAIL, 456.1, -2.22%); w1's
floor NOT SCORABLE (all-cell 374.2, printed); PRIVATE bytes HELD (w1 3.02%,
worst +4.60%; w2 0.17%); printed: SHARED bytes w1 19.3%, w2 3.9%. The files
above are unchanged.

## 2026-09-29: Granite-3.0-3B-A800M on the H100 SXM5, from that card's own 8x7B fit

`2026-09-29-granite-3.0-3b-a800m-h100.json` and `.txt`: Granite's test (above)
registered a second way, so it can run on an H100 SXM5 in parallel with the
GH200 queue; if no H100 is rented before the GH200 queue reaches Granite, it
runs on the GH200 under the GH200 registration instead, and this one is not
run. The fit is the 2026-09-25 H100 session's (board b533dd; five lock-1710
timed pages, base-clock counter pages), refit at this commit with the dead-CTA
and per-CTA terms: T0 0.0241 ms, c 201.72 ns = 344.9 cycles per CTA k-step, bw
2970 GB/s. The per-CTA fixed cost (CTA_FIXED_KSTEPS, measured on the GH200's
counters) is carried to this card as a property of the tile: the H100's
2026-09-25 pages carry no cycle counters to measure it. Before any page, the
two cards' registrations agree on Granite: the G >= 8 SHARED slope 0.0442 here,
0.0440 on the GH200.

Falsifiers, this card's numbers: time from Granite's own counted bytes
(`scripts/cross_model_score.py`, source = the 2026-09-25 H100 pages) at rms at
or under 2%, no cell beyond 5%; the floor per CTA k-step within 2% of 366.7
(w1) and 467.7 (w2), a test of carrying F across cards; the G >= 8 SHARED
slope within 2% of 0.0442; PRIVATE bytes within 5%; the rest printed.

**Not run.** No H100 SXM5 was rented before the GH200 queue reached Granite;
it ran on the GH200 under that registration (scored above, 2026-09-30). The
files above are unchanged.

## 2026-09-29, before any Granite page: the floor falsifier's estimator, corrected

The floor falsifier above (the slope of `sm__cycles_elapsed.avg` over CTA
k-steps per SM across the floor capture's n = 2, 3, 4, 6) is biased where a
GEMM's grid runs few waves: each cell's last wave is partly filled, by a
different fraction per n, and a line through four such cells tilts. JetMoE-8B's
w2 (0.97 to 2.91 waves) read 368.0 against 355.2 on it, while every GEMM with
at least four waves per cell reads within 1.3% of its prediction on the seven
measured models. JetMoE's registered result stands as registered (FALSIFIED on
the estimator as written); this corrects the estimator for pages not yet taken.

For Granite-3.0-3B-A800M (both registrations), the floor is scored on the cells
of the floor capture whose live CTAs fill at least 4 waves (live CTAs / (132 x
the GEMM's occupancy)): w2 on n = 3, 4, 6 (5.5, 7.3, 10.9 waves), its slope
within 2% of 466.5 cycles per CTA k-step (GH200) and 467.7 (H100). w1 has one
such cell (n = 6) and is NOT SCORABLE; its all-cell slope is printed. The other
falsifiers are unchanged.

## 2026-09-30: the floor at 4+ waves, JetMoE-8B and Mixtral 8x7B (floor-only sessions)

Before any page: two floor-only sessions (`gh200_model_session.sh --steps
prelude,floor`, `vm_run.sh start --steps`) take the floor capture at the treads
`dram_counter_route.r3_floor_treads` now gives each model so that every GEMM has
at least 3 cells of at least 4 waves: JetMoE-8B at n = 2, 3, 4, 6, 9, 10, 11 and
Mixtral 8x7B at n = 2, 3, 4, 6, 7, 8 (their earlier floor captures, at n = 2, 3,
4, 6, had no such w2 cells: JetMoE's w2 read 368.0 on the biased all-cell
estimator). The predictions are the per-CTA floor, 344.1 + F / S cycles per CTA
k-step (c and F from the four calibration models' counters, 0f77622), scored
as the slope over each GEMM's cells of at least 4 waves, within 2%:

| model | GEMM | S | scored n | registered cycles per CTA k-step |
|---|---|---|---|---|
| jetmoe-8b | w1 | 32 | 2, 3, 4, 6, 9, 10, 11 | 360.4 |
| jetmoe-8b | w2 | 88 | 9, 10, 11 | 355.2 |
| mixtral-8x7b | w1 | 64 | 2, 3, 4, 6, 7, 8 | 352.2 |
| mixtral-8x7b | w2 | 224 | 6, 7, 8 | 348.5 |

Mixtral 8x7B's floor counters are calibration data for c and F; its w2 cells
at 6 to 8 are new. JetMoE's floor data at n = 2 to 6 is published; n = 9 to 11 is
not yet measured.

**Scored 2026-09-30** on `results/published/2026-09-30-nvidia_gh200_480gb-{jetmoe,mixtral8x7b}-floor-session`
(board 594c0f; each README has the cells), the base-clock captures: JetMoE-8B w1
360.8 (+0.12%) and w2 on n = 9 to 11 361.7 (+1.84%), HELD; Mixtral 8x7B w1 349.9
(-0.66%) and w2 on n = 6 to 8 354.2 (+1.65%; unlocked -0.33%, lock +0.06%),
HELD. The lock captures fail FL1 alone (JetMoE's w2 reads +2.61% there). The
files above are unchanged.

## 2026-09-30, before any page: the partial wave's co-residency law, Mixtral 8x7B at TP=8 (floor capture)

The timing model's partial-wave rule (TAIL, a partial wave's CTAs at their
full-occupancy lifetime S'c occ) is replaced by a co-residency law (CORES in
`scripts/r3_timing_model.py`): the partial wave is dealt round-robin, k_SM =
ceil(k / 132) CTAs per SM, its floor lifetime S'c g(k_SM) with g(k) = k (620
floor-bound GEMM cells of the four calibration models: g free reads 0.76 to
1.15 x k, the old lifetime's rms is 1.86% against 0.35%), and its bytes at
min(bw, rho x k), rho = 11.48 GB/s per resident CTA from the calibration
counters (8x22B's n = 1 tail). Whether rho is pooled over the wave or held per
slab-fetching CTA the calibration cannot tell; the pooled form was chosen on
JetMoE-8B's published n = 3 page (diagnosis). This registration tests both.

Model: `mixtral-8x7b-tp8` (moe/spec.py, a TP=8 shard of the verified
Mixtral 8x7B; no R3 page of it exists). Capture: NATIVE floor captures at
G = 64 and 8, `--floor-treads 1 2 3 4 5 6 8 10`, base clock and the 1710 lock
(primary: the base-clock captures; every lock capture of a short-kernel model
has so far failed FL1 on its clock fit alone, and a lock capture that passes
FL1 is scored beside it). Run as `gh200_model_session.sh --steps prelude,floor
--floor-groups 64,8 --floor-treads 1,2,3,4,5,6,8,10`. Predictions:
`2026-09-30-mixtral-8x7b-tp8-floor-gh200.{json,txt}` (per GEMM
`sm__cycles_elapsed.avg`, every cell, under the law and its three rivals; one
unit S'c = 22,690 cycles on w1, 10,655 on w2), from
`scripts/cores_heldout_predict.py`, 8x7B's 2026-09-27 fit.

Coverage. The one cell that separates the laws is w1 at n = 2: 896 live CTAs,
one full wave of 660 and a partial wave of 236 at 2 per SM (104 SMs) and 1 (28),
118 of them fetching a slab, floor-bound. w1 n = 1 (a lone wave of 448, 4 and 3
per SM below occupancy 5, every CTA fetching) is bandwidth-bound and w2 n = 1, 2
run k_SM = 4 = occ: controls, every law prints the same. n = 3 to 10 run two or
more waves (partial waves of 24 to 520 CTAs, 1 to 4 per SM) and are throughput
under every law: the rule's domain boundary.

| w1, n = 2 (both G) | cycles | units | D = cycles(n=4) - cycles(n=2) |
|---|---:|---:|---:|
| co-residency law (registered) | 162,474 | 7.16 | 155,193 (6.84 u) |
| old lifetime (`--no-cores`) | 230,546 | 10.16 | 87,121 (3.84 u) |
| per fetching CTA (`--cores-per-lead`) | 196,496 | 8.66 | 121,171 (5.34 u) |
| throughput (`--no-tail`) | 158,962 | 7.01 | 158,705 (6.99 u) |

What this separates, stated before the pages: the law from the old lifetime
(`--no-cores`) and from rho per fetching CTA (`--cores-per-lead`), each
outside F1 and F2 at w1 n = 2; NOT from plain throughput pricing (`--no-tail`,
158,962 and D 158,705, inside both), which the law reduces to wherever a partial
wave is floor-bound.

Falsifiers, per G, on the base-clock captures: F1 D within 0.5 unit (11,345 cycles)
of 155,193 (D cancels the per-GEMM constant the model lacks: every calibration
series reads 0.3 to 1.1 units above it); F2 w1 n = 2 within -3% to +8% of
162,474; F3 every other cell within 8% of its prediction (a control that misses
says the miss is not this law); F4 the per-CTA floor over the cells of >= 4
waves (w1 n = 6, 8, 10; w2 n = 5, 6, 8, 10) within 2% of 352.2 (w1) and 379.1
(w2) cycles per CTA k-step. The score reprices every cell with the capture's
own `dram__bytes_read.sum`; the registered numbers use the launch order's own
reads (sigma = 1).

**Scored 2026-10-01** on `results/published/2026-10-01-nvidia_gh200_480gb-rental1-session`
(board 7269a7, unit 2; `results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters/r3f-g{64,8}.json`,
the base-clock captures; every cell repriced with its own `dram__bytes_read.sum`, sigma 0.58
to 1.06, which moves no co-residency number by more than 0.1%). F1 HELD: D = 156,941 (G=64) and
155,820 (G=8) cycles against 155,193, +0.08 and +0.03 unit; the old lifetime (87,121) and
the per-fetching-CTA form (121,171; repriced 116,675 and 121,423) are outside, plain
throughput (158,705) is inside, as registered. F2 HELD: w1 n = 2 170,316 (+4.8%) and
170,367 (+4.9%) against 162,474 (old lifetime -26%, per lead -13 to -15%, throughput +7.0
to +7.2%, inside). F3 FALSIFIED: G=64 w1 n = 1 +13.4%, w2 n = 1 +8.6%, n = 2 +10.4%, n = 3
+8.3%; G=8 w1 n = 1 +14.9%, w2 n = 2 +11.1%; every other cell within 8%. F4: w1 on n = 6,
8, 10 339.6 and 342.5 against 352.2 (-3.6%, -2.8%), FALSIFIED; w2 on n = 5, 6, 8, 10 374.3
and 374.2 against 379.1 (-1.3%), HELD. The lock captures fail FL1 alone (w1 fitted 1671
and 1662 MHz, w2 1655) and read the same verdicts (D 155,988 and 155,914; w1 n = 2 +3.1%,
+5.4%; F4 w1 -3.6%, -3.3%). The JSON's `primary` field names the lock captures "(FL1
PASS)"; with FL1 failing, both texts give the base captures. The files above are unchanged.

## 2026-10-01, before any page: rental 1, three registrations on one GH200

One `gpu_1x_gh200` runs `scripts/plans/rental1-2026-10.plan` through
`gh200_model_session.sh --plan` (`vm_run.sh start --plan`): units in the file's
order, each with its own model, overrides and page directory, each pushed as it
lands. The three sections below are committed in ONE commit with the
scaffolding, before the rental (the census licenses pages by commit), and none
is revised after another's pages land: B's G = 2 secondary and C's R0 are the
same byte model.

| # | unit | pages | est. min |
|---|---|---|---:|
| 1 | prelude | | 3 |
| 2 | (i) mixtral-8x7b-tp8 floor, EXACTLY the 2026-09-30 design (`--floor-groups 64,8 --floor-treads 1,2,3,4,5,6,8,10`), pushed before any other tp8 page | r3f-g64, g8, base and lock | 8 |
| 3-5 | (ii) C: tp8 G = 64 32 16 8; tp4 and tp2 G = 64 32 16 | 10 byte pages | 28 |
| 6 | (ii) C: same-board 8x7B G = 16 32 8 1 (the rho denominators; G = 1 is B's anchor and a board check) | 4 | 26 |
| 7-8 | (ii) C: Qwen2-57B and OLMoE at G = 128 | 2 | 13 |
| 9-11 | (iii) B: tp2, tp4, tp8 at G = 1 and 2 | 6 | 19 |
| 12-14 | (iv) A: launch floor, Granite-3B, JetMoE-8B, Granite-1B (ncu-free) | | 51 |

148 min estimated (byte pages 55 s + 108 s per GB of routed weights, from the
published ledgers; the 09-30 floor-only step 5 min; launch_floor's own dry-run
estimate), against the deadline of launch + 400 min and the guardian's 410-min
hard cap. When time runs short the driver drops the LAST unit first, up the
file: A, then B, then C; the floor capture goes last of all.

## 2026-10-01, before any page: the launch floor (Granite-3.0-3B, JetMoE-8B, Granite-3.0-1B-A400M)

R3's flat 0.24 to 0.29 ms at small n (Granite-3.0-3B SHARED/NATIVE n = 1..5,
PRIVATE n = 1..4; JetMoE n = 1) is read here as the eager timer running
host-paced, not a GPU-side floor. The published pages already carry the
timer's own verdict: 90/90 host-bound on every such (arm, n), host enqueue
0.341 to 0.362 ms per call, and the fit-free rule host-bound iff C_reg + F < H
(F = 251.7 MB / 3726 GB/s = 0.0676 ms, the flush's read) matches 439 of 441
non-edge published cells. That is post hoc; this registration is the test.

Tool: `scripts/launch_floor.py`, R3's call imported (`build_private_weights`,
`arm_inputs`, `arm_call`, `pinned_config`, `counter_declaration`; R3's page
path untouched), 1710 lock, duty 0.25, G = 4, BLOCK_M 32, 9 copies, 3
repeats. Modes per (arm, n): E240 (R3 as published: `time_cell`'s duty
sizing mirrored line for line, a 240 MiB flush), E0 (no flush), E480 (480
MiB), GR (one captured `fused_experts` call replayed as a CUDA graph under
E240's flush). Kineto traces (no counter, no ncu) TR-E and TR-G at E240 at
the trace treads, and at E0 and E480 at n = 1, 2. Run by the driver's
`launchfloor` step: Granite-3B treads 1..9 all modes, JetMoE 1..4 all modes,
Granite-1B 1..9 E240 and GR (treads 1..9, not the design's 1..12: the 9-copy
declaration and cross_model_predict's ladder). Predictions:
`2026-10-01-launch-floor-gh200.{json,txt}`, C_reg from the Granite and
JetMoE registrations of 2026-09-29 (G = 4) and, for Granite-1B, from
`scripts/cross_model_predict.py` at this commit on 8x7B's 2026-09-27 pages.

Fitted on what: C_reg on Mixtral 8x7B only; F from bytes and the ruler; H is
each cell's own measured host enqueue (recorded on every cell, not only
host-bound ones); Phi = 0.0998 ms and P8's plateau band are FITTED ON GRANITE
(SEEN); the composition model MP has no fitted constant. Edge cells, fixed
now: host-bound fraction over the 3 repeats in [0.2, 0.8], out of P4 and P5.

| # | prediction (cells scored) | fails if |
|---|---|---|
| P1 | GR follows C_reg within [0.80, 1.10] x C_reg on the 15 Granite (arm, n <= 5) cells; n = 1 NATIVE 0.0758-0.1042, SHARED and PRIVATE 0.0877-0.1206 ms; I_GR(5) - I_GR(1) >= 0.10 ms (SHARED, PRIVATE) | more than 1 of 15 outside, or an increment under 0.10; rival line: any GR n <= 3 >= 0.20 ms (a GPU floor predicts >= 0.23) |
| P2 | GR = E240 on GPU-bound cells (Granite n >= 7, JetMoE n >= 3) | any off by more than max(2%, 6 us) |
| P3 | E0 - E240 = F_trace - h_flush_trace - dD_tail (the start event fires after the flush, so its host dispatch h_flush is not in the E240 interval; first order +0.053 to +0.063 ms), inputs from TR-E/TR-G at E0 and E240, Granite n = 1, 2 (6 cells) | outside +/-12 us on more than 1 of 6; a cell without both traces is scored only through MP (P6) |
| P4 | E480 - E240 = -(F480 - F240) +/- 12 us where still host-bound (about -0.068 ms); crossover moves to C* = H - F480 = 0.219 ms: host-bound NATIVE n <= 4, SHARED n <= 3, PRIVATE n <= 2 (E240: 5, 5, 4) | the moved verdict wrong on more than 2 non-edge cells, or a shift outside the band |
| P5 | C_reg + F < H, each cell's own H, gives the E240 and E480 verdicts on all three models | under 95% of non-edge cells (a low bar; P4 is the test) |
| P6 | MP (max-plus, each kernel at max(previous end, launch + lambda), 50 iterations) predicts I_eager on every E240/E0/E480 cell at the trace treads, JetMoE n = 1 included | any outside max(5%, 12 us) |
| P7 | Granite n = 1: TR-G kernel sum <= 0.13 ms, non-GEMM <= 0.03 ms, eager host span >= 0.25 ms with launch-API time under 40% | classifies only |
| P8 | Granite-1B (unseen): host-bound wherever C_reg + 0.0676 < H (at H = 0.3545: every NATIVE and SHARED n <= 9, PRIVATE n <= 7); plateau H - Phi in [0.23, 0.27] ms (Granite-fitted, seen); GR within P1's band on all 27 cells | P1's rule (more than 1 of 27) or P5's on this model |

What it separates: P1, P3 and P4 move a host-paced interval and leave a
GPU-side fixed floor where it is (GR near 0.25 ms, no flush lever). If P1 to
P6 hold, R3's eager pages get the regime boundary C + F < H, host-bound cells
leave the time test and C is scored on GR cells; no T_floor is added to
call_ms. No claim about fused_experts inside a forward pass, where small-n
production is graph-captured.

## 2026-10-01, before any page: what sets L2 survival, the TP-shard distance block (rental 1 of the L2 design)

Published G=1 slab survival is not a function of the sequential reuse distance
x = (P - 1) SLAB / L2 alone: Phi-3.5 w1 and 8x7B w2 sit at one x and differ 5x
in s. The leading candidate is d = P / W_c, the re-reader's lag behind its
leader in co-residency windows. Rental 1 takes only the block that needs no new
knob: Mixtral 8x7B's TP shards at G = 1 and 2, default kernel, default metric
list. The TP shards keep E 8, top 2, H 4096, the tile and W_c and move x and d
apart: w1 keeps 64 k-steps and its 512 KiB slab, w2 keeps P = 64 and d = 0.121.

Measure, per page and GEMM, G = 1, n = 2..9: s = (B_PRIVATE - B_SHARED) /
((n - 1) W_g), B = `per_gemm[g].dram_bytes_read` (PRIVATE reads n weight
sets, SHARED 1 + (1 - s)(n - 1); the activation traffic is the same tiles in
the same order and cancels to first order). Law (`design-l2/work/register.py`,
calibration pages ONLY: 8x7B 09-27, 8x22B, Qwen2-57B, OLMoE): s = s_cap(x) + (1 -
s_cap) s_sync(d, n), s_sync from 8x7B w2 and 8x22B w2, s_cap 1 at x <= 0.286 and
solved at 0.576 from Qwen2 w1. Between 0.576 and 1.84 L2 s_cap is not identified:
H0 (one 60 MiB LRU) holds it to x = 1.0, H1 (two partitions with replication,
C_eff 0.5 to 0.67 L2) ramps it to 0 by 0.8. That d leads was suggested by Phi-3.5
and JetMoE (seen): hypothesis generation, not a test. Predictions:
`2026-10-01-l2-survival-tp-gh200.{json,txt}`.

| cell | x (L2) | d | s, n = 2..9, H1 + H2a | H0 + H2a (where different) |
|---|---:|---:|---|---|
| tp8 w2 | 0.23 | 0.121 | 1.00 at every n | same |
| tp4 w2 | 0.47 | 0.121 | 0.85 0.84 0.85 0.86 0.87 0.88 0.88 0.89 | same |
| tp2 w2 | 0.93 | 0.121 | 0.70 0.54 0.52 0.52 0.48 0.48 0.46 0.45 (= 8x7B w2) | 0.76 0.74 0.76 0.77 0.79 0.80 0.81 0.82 |
| tp8 w1 | 0.47 | 0.085 | 1.00 (extrapolated in d) | same |
| tp4 w1 | 0.93 | 0.170 | 0.35 0.22 0.20 0.17 0.15 0.15 0.13 0.12 | 0.49 0.56 0.60 0.60 0.66 0.67 0.69 0.71 |
| tp2 w1 | 1.87 | 0.339 | 0.03 0.01 0.01 0.00 0.00 0.00 0.00 0.00 | same |

Pages that score it: the plan's units 9 to 11, `<date>-<card>-mixtral-8x7b-tpT-l2-r3-counters/lock1710/r3c-g{1,2}.json`
for T = 2, 4, 8, and the same-board anchor, unit 6's `...-mixtral-8x7b-sameboard-r3-counters/lock1710/r3c-g1.json`
(8x7B w2 at G = 1, the curve the tp2 w2 rule compares against; the +/-0.05 band
in s is the 09-25 to 09-27 board-to-board reproduction, 0.04, and holds only
against a same-board baseline).

Decision rules, fixed now. H1 over H0: tp2 w2 within 0.08 of the same-board 8x7B
w2 curve AND below 0.65 at every n = 4..9; H0: >= 0.74 at every n = 4..9; else
INCONCLUSIVE. tp4 w1: H1 if under 0.40 at every n = 6..9 and within 0.08 of H1's
numbers, H0 if over 0.40 and within 0.08 of H0's. An x-only law (s a function of
x alone) is falsified if tp4 w1 and tp2 w2, one x and two d, differ by more than
0.10 at three or more of n = 4..9. Controls (both hypotheses): tp8 w2 and tp8 w1
>= 0.95; tp4 w2 within 0.05 of its row; tp2 w1 <= 0.08; a control outside says
the law's shape failed, not H0 or H1. PRIVATE F / Mn in [0.50, 0.56] is a CONTROL
(it holds on 1202 published cells and every hypothesis predicts it). The far
share of SHARED's extra hits, from the recorded tex/fabric sectors, in [0.15,
0.45] where s >= 0.1 is a RECORD. V10 does not gate survival scoring (bytes are
clock-free; tp8 and tp4 pages will likely read 1641 to 1688 MHz like every
short-kernel lock page); a page failing V6 has its s flagged, since V6 checks
the activation-cancels premise (JetMoE G = 1 already fails it, 0.70%).

Secondary, G = 2: every SHARED and PRIVATE cell against the registered byte
model R0 (8x7B 09-27 MIX through `cross_model_predict.target_geometry`, nothing
fitted on the shards) with its own 5% bar; R0's numbers are in the JSON. This R0
is C's R0 below: both registrations are committed together, before rental 1,
and neither is revised after the other's pages land.

Not in rental 1: the R0 anchors with partition metrics, num_stages, BLOCK_K,
the 1005 MHz clock and slot padding (rental 2: `--block-k`, `--slot-pad-rows`,
the partition metrics and per-knob directories).

**Scored 2026-10-01** on `results/published/2026-10-01-nvidia_gh200_480gb-rental1-session`
(board 7269a7, units 6 and 9 to 11; every tp page fails V10 alone, the anchor passes every
gate, V6 passes everywhere so no s is flagged). Measured s, n = 2..9: tp2 w2 0.82 0.59 0.57
0.58 0.54 0.56 0.55 0.54; same-board 8x7B w2 0.72 0.54 0.52 0.53 0.50 0.51 0.49 0.48; tp4
w1 0.65 0.54 0.47 0.45 0.42 0.38 0.38 0.36. H1 over H0 on tp2 w2: H1 (within 0.066 of the
same-board curve at n = 4..9, 0.54 to 0.58, under 0.65; H0 needed >= 0.74). tp4 w1:
INCONCLUSIVE (0.42 0.38 0.38 0.36 at n = 6..9: +0.23 to +0.27 over H1, 0.24 to 0.35 under
H0, 0.42 over 0.40 at n = 6). The x-only law: FALSIFIED (tp2 w2 minus tp4 w1 0.10 to 0.19
at all six of n = 4..9). Controls: tp8 w2 HELD (>= 0.973); tp8 w1 FAILED (0.82 to 0.91);
tp4 w2 FAILED at n = 2 only (+0.087, n = 3..9 within 0.025); tp2 w1 FAILED at n = 2, 3
(0.16, 0.10; then 0.03 to 0.06). By the rule above the law's shape failed at x <= 0.93 L2,
so the H1 reading of tp2 w2 is not a pass of the law. PRIVATE F / Mn HELD (0.504 to 0.521,
126 cells). Far share RECORD: 43 of 50 cells with s >= 0.1 inside [0.15, 0.45] (outside:
tp4 w2 n = 8, 9 at 0.137; tp8 w2 n = 5..9, 0.06 to 0.115). G = 2 secondary against R0:
PRIVATE HELD (54 of 54 within 5%, worst +0.4%); SHARED FALSIFIED (31 of 54 beyond 5%,
worst tp4 w1 n = 8 -58.8%, R0 low on tp2 w2 and tp4 w1 from n = 4). The files above are
unchanged.

## 2026-10-01, before any page: PRIVATE's A-tile reads at large G and short K (rental 1)

The registered byte model (8x7B's MIX fit, carried by `cross_model_predict`)
misses PRIVATE's A-tile reads two ways on the eight published GH200 models: w1
is over-predicted on every top_k = 8 model at every G, G = 1 included (Qwen2
G=64 +4.7 to +6.8%, Granite +2.7% at G=1), and w2 is under-predicted at G >= 32
on the long-K models (Phi -8.0%, Qwen2 -5.8%). Two candidate fixes, both
scaffolding of the existing model and both OFF by default
(`scripts/wave_split_bytes.py`: `Model(geom, content_a=True)`,
`Params.k0_A/a_A`; every registered number is unchanged with them off, pinned
in `tests/test_atile_pooled_fit.py` from the code at 6d2595d):

- **R1, content-keyed w1 A.** R3's routing (`SWEEP.balanced_ids` ->
  `realize_counts`, a heap greedy with ties by expert id) puts the experts in
  E/k classes of k with identical token sets, so k M-tiles read one w1 A tile.
  w1's A is keyed by tile content (first read compulsory, the credit (U - E n)
  ATILE in S_c), one A survival law for both GEMMs (w1 fill, w2 ws).
  THIS IS A HARNESS PROPERTY: real top-k routing has no such classes, so the
  term models R3's pages, not the hardware or a served model.
  `wave_split_bytes.DESIGN_KEYS` now carries `routing`.
- **R2, R1 times a duration factor** s(ks) = (1 + ks / k0_A)^(-a_A) on every A
  distance, ks = K / BLOCK_K: long CTAs stay wave-coherent, so their A
  re-reads sit closer in time than the byte distance says. The fit runs to its
  exponential limit, s = exp(-c ks).

Fits (`scripts/atile_pooled_fit.py`, PRIVATE cells, every G and tread, both
GEMMs). PRIMARY, calibration models only (8x7B 09-27, 8x22B, Qwen2-57B,
OLMoE): R1 C_A 159.61 MiB, beta_A 1.053, theta1_A 0 (fit rms 1.21%, worst
-9.51%); R2 C_A 63.68 MiB, beta_A 1.338, c = 0.00344 per k-step (0.87%, -7.24%).
SECONDARY, all eight models, the four diagnosis models SEEN: R1 148.13 /
1.072 (1.00%, -9.23%); R2 73.37 / 1.316, c = 0.00293 (0.75%, -7.95%). R0 = the
registered 8x7B MIX fit, nothing refitted. The knee form with its knee fixed at
40 k-steps is NOT registered: the 40 was read off JetMoE and Phi.

Pages (one GH200, rental 1, lock 1710, treads 1..9, all three arms; each page
group in its own directory): `mixtral-8x7b-tp8` G = 8, 16, 32, 64 (taken only
after the registered tp8 floor capture is pushed); `-tp4` G = 16, 32, 64;
`-tp2` G = 16, 32, 64; Qwen2-57B and OLMoE at G = 128; Mixtral 8x7B G = 8, 16,
32 on the SAME board (the rho denominators); the L2-survival registration's
tp2/tp4/tp8 G = 1 pages for T3. Predictions:
`2026-10-01-atile-ksteps-gh200.{json,txt}`, every PRIVATE cell, w1 and w2, n =
1..9, R0 and the four fits.

| cell (q, n = 8) | R0 | R1 | R2 | R2/R1 |
|---|---:|---:|---:|---:|
| tp8 w2 G=64 | 8.305 | 8.388 | 8.629 | +2.9% |
| tp4 w2 G=64 | 8.690 | 8.757 | 9.246 | +5.6% |
| tp2 w2 G=64 | 9.435 | 9.389 | 10.018 | +6.7% |
| Qwen2 w2 G=128 | 9.135 | 9.162 | 10.076 | +10.0% |
| Qwen2 w1 G=128 | 9.235 | 8.100 | 8.141 | R0/R1 +14.0% |
| OLMoE w1 G=128 | 8.589 | 7.968 | 7.987 | R0/R1 +7.8% |

Tests, what each separates, falsifiers fixed now:

- **T1, content-keyed w1 (R1/R2 against R0).** PRIVATE w1 at G = 128 on Qwen2
  and OLMoE: pass within 2% of R1 at every n >= 2; falsified if the measured q
  sits nearer R0 than R1 at any n >= 4.
- **T2, duration (R2 against every byte law).** w2 n = 9, f = (q - n) x 128 /
  (63 n): rho42 = f(tp4 G=64) / f(8x7B G=16), rho84 = f(tp2 G=64) / f(8x7B
  G=32), both pages from ONE board. Bands = prediction +- 2 sigma, sigma from
  sigma_q = 0.036 (90th percentile of the per-call standard error of PRIVATE w2
  q, G >= 8, n >= 6, 8x7B 09-27; median 0.009), propagated to each f in
  quadrature. rho42: R1 0.961 [0.835, 1.087], R2 1.869 [1.640, 2.098] (R0
  0.964 [0.824, 1.104]; secondary R2 1.689 [1.475, 1.902]). rho84: R1 0.989
  [0.918, 1.061], R2 1.426 [1.338, 1.515] (R0 0.992; secondary R2 1.354 [1.267,
  1.441]). A candidate is falsified when BOTH rho fall outside its band; both
  between the R1 and R2 bands refutes every candidate; one in, one out is
  inconclusive. The board-to-board difference (09-25 against 09-27, PRIVATE w2
  up to 2.8% at G=64 n=6) is why the denominators are re-measured. Confound:
  along each row L (8 -> 64) and W_c / L (66 -> 8 columns in flight) move with
  the k-steps, so a byte law with a columns-in-flight term also rises; the
  separator (8x7B G=32 at BLOCK_K 32 / 8 stages against 128 / 2) needs
  `--block-k` and is rental 2. The 21 MiB row is supporting evidence only.
- **T3, the G = 1 credit (sharing against none, at G = 1).** PRIVATE w1 G=1
  n=1 on the tp shards' G=1 pages: q - 1 within +-0.0010 of R1 (the worst
  |measured - R1| over the eight published G=1 n=1 cells is 0.00098): tp8 R1
  -0.00446 (R0 0.00000), tp4 -0.00193 (R0 +0.00011); pass inside R1's band and
  outside R0's. tp2 (R1 -0.00039, R0 +0.00025) is a record: its share distance,
  W_w1 / E = 117 MB, is past the L2. The design's "within 5% of q" is replaced
  (5% of q = 1 is ten times every credit).
- **T4, the score.** Each candidate on every PRIVATE cell of the scoring pages
  (G = 1 aside): pass at rms <= 2% and no G >= 32, n >= 6 cell beyond 4%.

V10 does not gate any score here: the tp8 and tp4 lock pages will likely fail it
(short kernels: the JetMoE and OLMoE lock pages read 1641-1688 MHz against 1710),
and DRAM bytes are clock-free (base-clock against lock bytes agree to 0.03% w1,
0.8% w2 on 8x7B 09-27).

**Scored 2026-10-01** on `results/published/2026-10-01-nvidia_gh200_480gb-rental1-session`
(board 7269a7, units 3 to 8 and the G = 1 pages of 9 to 11; every PRIVATE cell scored: the
failing gates are V10, V7 and V6 on NATIVE/SHARED and claim gates C1, C2, C6, none on
PRIVATE's bytes; R1 and R2 recomputed on the pages' geometry match the JSON to 0.0005%).
T1: OLMoE HELD (worst +0.52% from R1 at n >= 2; R0 -5.4 to -7.3%); Qwen2 INCONCLUSIVE
(nearer R1 than R0 at every n, R0 -8.5 to -11.6%, but +2.18, +2.11, +2.64% from R1 at n =
6, 7, 9, outside the 2% pass). T2: rho42 1.199, rho84 1.184 (f: tp4 G=64 0.2963, 8x7B
G=16 0.2472, tp2 G=64 0.5802, 8x7B G=32 0.4900), both between the R1/R0 bands and the R2
bands: every candidate refuted (R0, R1, R2, primary and secondary). T3: tp8 PASS (q - 1
-0.00438, R1 -0.00446), tp4 PASS (-0.00138, R1 -0.00193, outside R0's band [-0.00089,
0.00111]); tp2 record -0.00044 (R1 -0.00039, R0 +0.00025). T4, 270 PRIVATE cells:
FALSIFIED for every candidate: R0 rms 3.68% (24 cells of G >= 32, n >= 6 beyond 4%), R1
2.87% (15), R2 1.55% (10: 8x7B G=32 w2 n = 7-9, tp2 G=64 w2 n = 7-9, Qwen2 G=128 w2 n =
6-9), secondary R1 2.73% (14), R2 1.77% (10). The files above are unchanged.

## 2026-10-01, before any page: rental 2, four registrations on one GH200

One `gpu_1x_gh200` runs `scripts/plans/rental2-2026-10.plan` through
`gh200_model_session.sh --plan` (`vm_run.sh start --plan`), as rental 1 did. The four
registrations below, their scorers (`scripts/scoring/rental2/`, written and tested on
synthetic pages before any page) and the scaffolding they need are committed in ONE
commit before the rental. Every number in the four JSON files is written by
`scripts/scoring/rental2/register.py` from committed files (`--check` recomputes them);
the `.txt` beside each is rendered from its JSON. Design: the scratchpad's
`design-r2/DESIGN.md` with every fix of `design-r2-review/REVIEW.md`.

| block | units | est. min |
|---|---|---:|
| prelude (also locks 1005 MHz once, read back, reset) | 1 | 3 |
| floors: tp8 floor2, tp4, tp2 (base and 1710, shape metrics, null kernel), tp4 at 1005 | 4 | 21 |
| B lag: tp2 l2base, l2s8 (partition metrics); tp4 l2base, l2s8; tp8 l2s8 (the negative control); tp2 l2s6 | 6 | 24 |
| C T5: tp2 G = 64 at BLOCK_K 64/s4, 32/s4, 128/s2 | 3 | 12 |
| launch floor: tp8, Granite-3B, each a timed and a trace process | 2 | 56 |
| tail: tp2 l2bk32, l2bk128, l2pad7, ats8; 8x7B l2base + l2s8 (one drop-group); JetMoE, Granite-1B | 8 | 60 |

176 min of units, about 178 min with setup: 2.97 h, about $6.8 at $2.29/h; the guardian's
hard cap of 410 min is $15.6. The core (through the launch floor) is 116 min. The driver
drops the last unit first; the 8x7B base/s8 pair goes together (`drop-group`).

**Part 1, B's knobs and C's T5** (`2026-10-01-rental2-knobs-gh200.{json,txt}`). L2r: past
the edge (x >= 0.92 L2) s_knob = s_base exp(-dd / lambda(n)), below it s_cap + (s_base -
s_cap) exp(-dd / lambda(n)), dd the d of the knob page less the same-board base page's,
each page's W_c read off its own occupancy. lambda(n) is SEEN (rental 1's tp2 w2 / tp4 w1
pair) with a per-n band from +-0.03 on each s in quadrature (42% at n = 4, 33% at n = 5,
24 to 34% at n = 6..9; n = 2, 3 printed only). Rivals NL (s_base +- 0.05) and ST (the
registered 2026-10-01 lambda(n), CAL). tp2 w2 s8 is the primary test and the only L2r
against ST call; tp4 w1 s8 is registered as NL against LAG (L2r and ST together), since ST's
band lies inside L2r's there; tp2 s6 is secondary and needs all four of n = 6..9. A
hypothesis is SELECTED on 4 of n = 4..9 in its band and no other, FALSIFIED on 4 of 6
outside. Controls: PRIVATE F / Mn in [0.50, 0.56] demotes that page's SELECTED to
INCONCLUSIVE, and tp2 w1 under its own L2r upper edge (replacing rental 1's fixed 0.08)
demotes that page's L2r or LAG selection (an NL or ST one stands); the tp8 w2
s8 negative control (s >= 0.95 at n = 4..9) demotes every part-1 SELECTED; a V6 failure
flags the page and its verdicts read INCONCLUSIVE. T5: rho_BK = f(BK128) / f(BK32) at n = 9,
f = (q - 9) 128 / (63 x 9) from `dram_counter_route.r3_q`, each candidate re-priced on each
page's own geometry with its registered band half-width kept fixed: R0 and R1 1.00 [0.93,
1.07], R2 1.67 [1.57, 1.77] (CAL), R2h 1.29 [1.23, 1.35] (SEEN, m = 0.567 from rental 1's
rho84); a failed BK64 board check (f within 2.8% of 0.5802) turns R2h's verdict to
INCONCLUSIVE.

**Part 2, the launch-floor rerun** (`2026-10-01-rental2-launch2-gh200.{json,txt}`). The
timed process has the profiler guarded off (`torch.profiler.profile` raises, and
`_profiler_enabled()` is asserted False before every cell) and measures each (arm, n,
mode)'s host time H_pre before any cell and again after the last; the traces run in a
second process. Every prediction is priced on H_pre. tp8 is BLIND on every host-side
prediction; its P1 is C_reg CAL with GEMM durations SEEN under ncu. P0 is an instrument
gate (the profiler, a post/pre drift over 5%, H_cell / H_pre outside [0.93, 1.10] on over
5% of cells, over 5% refused probes); when it fails P2, P4, P5 and P8 are printed marked
"host drift".

**Part 3, the per-GEMM constant** (`2026-10-01-rental2-const-gh200.{json,txt}`). Z is the
intercept of SM-active cycles on q per capture and GEMM, the slope free. Re-derived that
way on the calibration models, Z / u at the 1710 lock reads 0.23 to 0.51 (w1) and 0.26 to
0.68 (w2), and the base-to-lock exponent reads -0.61 to 3.02 over 10 series, so it is not
used as a band. K1: L = elapsed - active.max <= 5000 cycles on 80% of the pool. K2: the
pooled w2 Z / u of tp4 and tp2 in [0.25, 0.69] (the full CAL w2 range, no model excluded).
K3: Z(1005) / Z(1710) on tp4 against the two forms, f_1005 / f_1710 (nanoseconds) and 1.0
(SM cycles), each within 2 sigma of the registered intercept noise (sigma_r / r 0.10 on
w1, 0.13 on w2); inside both or neither is INCONCLUSIVE. K4: D_imb in [-0.2 u, 0.2 u] on
80%, H_IMB on D_imb >= 0.3 u on over 20%; the CTA-count clause is a RECORD only.

**Part 4, tp8's w1 floor** (`2026-10-01-rental2-w1floor-gh200.{json,txt}`). The tread sets
are the review's unseen ones (tp8 w1 (13, 14, 15) -11.0%, (11, 15, 16) -0.4%, (11, 12, 13)
+3.8%; tp4 w1 (14, 15, 16) -3.6%, (11, 13, 15) +0.1%, (10, 12, 13) +3.8%; tp2 w1 (10, 11,
12), (12, 13, 14), delta only; w2 an offset test at +3.8%). FLUID_LOW, a set-independent
0.964 x asymptote, is the rival; H_NPN is a labelled secondary. Bands come from a
registered noise model: sigma_cell 2.7k cycles (CAL) propagated exactly through the sets'
weights (sigma_theta 0.28 tp8, 0.20 tp4). The theta call is at 0.5. Co-primary: per GEMM
on the unseen cells, the rms about ceil(q) u + a against q u + a, a won by more than
sigma_cell. The primary capture is the base clock. tp8 w1 (6, 8, 10), 339.5, is printed as
a SEEN replication only.

**The review's open scorer items, decided before any page** (each is in the named JSON):

- knobs: s(n) is `scripts/l2_survival.py`'s formula, written out in `measure`; V6-flagged
  cells are excluded from every count; W_c = sm_count x the least of the four occupancy
  limits; P = npn; the s_cap knots per n and the linear interpolation are in the JSON; r()
  below the edge is exp(-dd / lambda) on s_base - s_cap; a failed control demotes SELECTED to
  INCONCLUSIVE; the partition cross-check (direct fabric hits against F - (M - Mn) within 5%)
  is an instrument check whose failure on over 10% of a page's cells only switches that
  page's far-share RECORD to the direct count; T5 is decided on n = 9 alone with f from
  r3_q's PRIVATE w2 q; a failed BK64 board check makes R2h INCONCLUSIVE.
- launch2: H_cell = host_enqueue_ms / calls_per_burst (median over repeats); the host-bound
  fraction is over the cell's 3 repeat rows; P0's drift is the median over (arm, n, mode) of
  |H_post / H_pre - 1|; the edge rule's F is the cell's own mode's; P3 needs 8 qualifying
  cells and fails on more than 1 outside; P6's function is rental 1's text, copied verbatim
  into `launch_mp.py` (a test pins the copy).
- const: the floor-bound share is at sigma 1, never re-priced; Z by intercept; K1's pool is
  every floor-bound cell of the three 1710 captures, both GEMMs; K2 pools tp4 and tp2 w2 (the
  median of the two); K3 by the intercept ratio against the two forms.
- w1floor: the primary capture is the base clock; theta and delta by unweighted OLS over
  the sets, their noise propagated exactly through the shared cells; FLUID and FLUID_LOW
  have their own falsification rules; when every family fails the verdict is NEITHER; the x
  grid is the page's own launch grid, asserted equal to the registered one.
