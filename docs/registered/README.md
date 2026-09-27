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
