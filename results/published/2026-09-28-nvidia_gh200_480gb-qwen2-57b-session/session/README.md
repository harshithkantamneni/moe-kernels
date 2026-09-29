# The 64-expert test: Qwen2-57B-A14B on one NVIDIA GH200 480GB (Lambda Cloud), run unattended (docs/LAMBDA.md section 3c with `--model qwen2-57b-a14b`); every measurement here is this card's, never averaged with another card's

This session scored the predictions registered in
`docs/registered/2026-09-28-qwen2-57b-a14b-gh200.json` (commit b0312ea) before
any page here existed: Qwen2-57B-A14B (64 experts, top 8) predicted from
Mixtral 8x7B's fit on the 2026-09-27 GH200, nothing fitted on Qwen2-57B. The
R3 design is 64 experts x 9 copies (576 slots for SHARED and PRIVATE, 64 for
NATIVE, `--declared-copies 9` on every timed page). A fourth GH200 board:
`board 50e61f` in `PREFLIGHT.txt` and on every page's CARD line.

## Where it came from

Two instances, both gpu_1x_gh200 in us-east-3 at $2.29/h (ids and times from
the laptop's `~/.lambda/rent-ledger.log`, not from a file here):

- 72d2daa123884d94a245dbabe8ad3315, active 2026-09-28 07:52 UTC, **lost**
  about 08:50 UTC, 55 minutes into the byte step: ssh stopped answering and
  Lambda stopped listing it, with no terminate from the laptop. The step pushed
  only at its end, so none of its pages reached the branch. Every byte page is
  now pushed as it lands (commit 684a9e9). Its branch
  `run-gh200-qwen57-2026-09-28` holds only setup and the prelude.
- dc582b37864e4e98b024852640ad12c8, active 21:27:38 UTC, terminated 2026-09-29
  00:29:09 UTC (3.0 h, about $6.9) by the laptop's guard once every pushed file
  had been verified (`vm_run.sh verify`: all 525 files under vm/ match
  SHA256SUMS). This directory is its run.

Driver 570.148.08 upgraded to 580.105.08 and rebooted before setup
(`driver580.log`, `setup_vm.log`). `scripts/gh200_model_session.sh --setup
--model qwen2-57b-a14b` ran `setup_vm.sh` at commit
684a9e9880f158b02ef3305c67bc5787aa9485d9 (`commit.txt`), the registration's
tree plus the per-page push. This directory is the branch
`run-gh200-qwen57-20260928t2127` at ff01a0b, curated as 2026-09-27's:
`results/` is `~/moe/results` less its calibration run directory (in
`../calibration/` with the ruler); this directory is `~/moe/session` plus
`setup_vm.log`, `driver580.log` and `gh200-driver.out`; `SHA256SUMS.run-branch`
and `PUSHES.run-branch.txt` are the branch's manifest and push log.
Preflight PF1 to PF7 PASS (`PREFLIGHT.txt`).

## What ran (`gh200-driver/status`)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 21:30 | 0 | 0 | persistence, 700 W, the registered locks |
| bytes | 21:30 | 60 | 0 | eight byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1 to 9, 442 to 444 s each, each pushed as it landed; G=3 INVALID on V7 (one cell, NATIVE and SHARED w1 at n=9 1.15% apart against 1%), G=64 on V4 (q(1) = 1.030 on all three arms) |
| calibrate | 22:30 | 0 | 0 | the ruler stands: `../calibration/` |
| timed | 22:30 | 36 | 0 | locked R3 at 1710: G=8 VALID, G=3 VALID, G=32 INVALID on V0 (139 of 162 cells) |
| floor | 23:06 | 3 | 0 | floor counters G=64 at base and at the 1710 lock; the lock held (fit 1713 and 1717 MHz, FL1 PASS) |
| deep | 23:09 | 32 | 0 | locked R3 at 1710 to tread 9: G=4 VALID, G=2 INVALID on V0 (206 of 243 cells); the lock held |
| recovery | 23:42 | 11 | 0 | G=32 timed again at 1710: VALID (`recovery/retake_qwen.sh`) |
| recovery 2 | 23:55 | 16 | 0 | deep G=2 again at 1710: INVALID on V0 again (197 of 243) (`recovery/retake2_qwen.sh`) |

Both V0 failures are NATIVE's: cells whose clock drifted across their own
trials are excluded, and NATIVE lost 14 to 19 cells a page (SHARED 3 to 9,
PRIVATE 6 to 9), leaving it 1 or 2 repeats at one tread against a floor of 3.
The module's software power cap (0x4) is in 16 to 23% of every ladder's
`smi.csv` rows, samples down to 1470 to 1485 MHz, as on 2026-09-28's 8x22B
session; every page's own per-cell clock readings held 1695 to 1710 MHz.

## What it answered (registered at b0312ea, scored on these pages)

- **The floor is the tile's.** Cycles per CTA k-step (the slope of
  `sm__cycles_elapsed.avg` over CTA k-steps per SM across n = 2, 3, 4, 6), w1
  353.3 at base and 353.7 at the lock, though a CTA runs 56 k-steps (8x7B's 64,
  8x22B's 96): registered 340 to 365. HELD. w2 369.0 to 369.6.
- **The G >= 8 SHARED slope** over treads 2 to 6: 0.6962 (G=8) and 0.6953
  (G=32) ms per tread against 0.6837 registered, band 0.670 to 0.697. HELD, at
  the upper edge.
- **Time**, the VALID pages at 1710 (G=3 `-p5`, G=8 `-p2`, G=32 `-p2b`, G=4
  `-deep`): 58 SHARED and PRIVATE cells, rms 2.71%, worst -5.00% (SHARED G=32
  n=2), none beyond 5% (registered: rms at or under 2%). FALSIFIED on the rms.
  The miss is a uniform bias: the prediction is 2.3 to 2.7% fast at every G
  (tread 1 included), SHARED -2.5 to -4.7% by tread, PRIVATE -1.8 to -2.6%. The
  INVALID deep G=2 pages' SHARED and PRIVATE cells (their arms kept 7 repeats)
  read 2.39% and 2.38% rms, printed and not scored.
- **G=2's zig-zag** keeps its phase on both deep G=2 pages: steps 0.727 0.661
  0.711 0.689 0.720 0.686 0.724 (the n = 2 to 3, 4 to 5, 6 to 7 and 8 to 9
  steps above the ones after them). HELD in phase, on pages that fail V0; the
  amplitude is about 0.04 ms against the registered 0.21.
- **Bytes** (the eight lock-1710 counter pages, MIX view): FALSIFIED.
  - w1, SHARED and PRIVATE (140 cells): 14.3% rms, 39 cells beyond 5%.
  - The 64-expert signature: w1 SHARED at G=1 read 1.227 1.517 1.712 1.911
    weight sets at n = 2 to 5 against 1.160 1.209 1.223 1.225 registered.
  - PRIVATE w2 (72 cells): 1.89% rms, 7 beyond 5%, all at G=64 (-5.0 to -5.8%).
  - w2 SHARED at G <= 16, n <= 4 (24 cells): 33% rms, worst +97% (G=2 n=4):
    at G = 2 to 4 the model prints about n/2 weight sets and the card read about
    one. w2 SHARED at G=1 read 1.00 to 1.02 (registered 1.00 to 1.08).
  - Open questions, printed not scored: w2 SHARED at G <= 16, n >= 5 up to
    +382% (G=2 n=9); at G >= 32 (OUT-OF-DOMAIN) 7.8% rms, worst +15.9%.

What the misses point at is in docs/FINDINGS.md (2026-09-28, the 64-expert
test); none of it is fitted here.

## Files

- `gh200-driver/status`: the driver's ledger, with the RECOVERY and RECOVERY2
  lines; `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger, `smi.csv` and summary
- `lock-plan.txt`, `power-*.txt`, `clocks-*.txt`: the card's state around each
  step
- `census-qwen2-57b-a14b.json` and its profiles: the model's census
- `recovery/`: the two retakes as they ran

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
