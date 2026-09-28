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
