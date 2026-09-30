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
