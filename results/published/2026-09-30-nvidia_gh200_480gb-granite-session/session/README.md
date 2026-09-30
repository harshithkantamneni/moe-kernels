# The held-out test, fourth of four: Granite-3.0-3B-A800M on one NVIDIA GH200 480GB (Lambda Cloud), run unattended (docs/LAMBDA.md section 3c with `--model granite-3.0-3b-a800m`); every measurement here is this card's, never averaged with another card's

This session scored the predictions registered in
`docs/registered/2026-09-29-granite-3.0-3b-a800m-gh200.json` (commit 0f77622,
pushed at a1b6118), with the floor falsifier's estimator as corrected before
any Granite page (docs/registered/README.md, a156392): Granite-3.0-3B-A800M (40
experts, top 8; a CTA runs 24 k-steps on w1 and 8 on w2, the shortest CTA the
study has registered) predicted from Mixtral 8x7B's fit on the 2026-09-27
GH200, with the dead-CTA term, the capacity rule for later re-reads and the
per-CTA fixed cost, nothing fitted on Granite. The registration named time
from the model's own counted bytes as the primary test. A seventh GH200 board:
`board 1310e2` in `PREFLIGHT.txt`. The same test was also registered from the
H100 SXM5's fit (`2026-09-29-granite-3.0-3b-a800m-h100.json`, 7f824d4); no H100
was rented and that registration was not run.

## Where it came from

The first attempt did not run. Lambda Cloud instance
bafb8d5cb87842c8bc4f4a926c594847 (gpu_1x_gh200, us-east-3, active 2026-09-29
19:59:12 UTC, terminated 20:14:43 UTC, about 16 minutes) came up at
192.222.51.218, the address the Qwen1.5 instance had just released, and the
`known_hosts` key pinned for that address did not match the new instance's,
so the laptop's prepare step could not reach it by ssh within its 900 s limit
and refused. No driver ran, nothing was pushed and no run branch exists for it
(the planned `run-gh200-granite303ba800m-20260929t1959`). The Phi-3.5-MoE and
JetMoE-8B sessions ran next; this session is the second attempt.

Lambda Cloud instance 0546c5de99fa460b901ac244c3d27c72, gpu_1x_gh200,
us-east-3, active 2026-09-30 00:41:27 UTC, terminated 03:18:26 UTC (2.6 h);
both instances' ids and times are from the laptop's
`~/.lambda/rent-ledger.log`. Driver 570.148.08 upgraded to 580.105.08 and
rebooted before setup (`driver580.log`, `setup_vm.log`).
`scripts/gh200_model_session.sh --setup --model granite-3.0-3b-a800m` ran at
commit a156392 (`commit.txt`), which carries both Granite registrations and
the corrected estimator. This directory is the branch
`run-gh200-granite303ba800m-20260930t0041` at d89eba8, curated as the earlier
sessions were; `SHA256SUMS.run-branch` (all 599 files under vm/) and
`PUSHES.run-branch.txt` are its manifest and push log. Preflight PF1 to PF7
PASS (`PREFLIGHT.txt`).

## What ran (`gh200-driver/status`)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 00:45 | 0 | 0 | persistence, 700 W, the registered locks |
| bytes | 00:45 | 10 | 0 | eight byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1 to 9, 69 to 71 s each, each pushed as it landed; every page INVALID on V10 (below) |
| calibrate | 00:55 | 0 | 0 | the ruler stands (triad 3726.1 GB/s): `../calibration/` |
| timed | 00:56 | 36 | 0 | locked R3 at 1710: G=8, G=32 (treads 6), G=3 (treads 8), every page INVALID (below), worst cell 1710.0 MHz |
| floor | 01:33 | 1 | 3 | floor counters G=64 at base (`r3f-g64.json`) and at the 1710 lock (`r3f-g64-lock1710.json`, FL1 FAIL) |
| deep | 01:34 | 32 | 0 | locked R3 at 1710 to tread 9: G=4 and G=2 INVALID |
| recovery | 02:08 | 67 | 0 | `recovery/retake_auto.sh`: every timed and deep G taken once more at 1710 (tags `-t3b`, `-t8b`, `-t32b`, `-deepb`), every page INVALID again |

**Why every timed page is INVALID.** All ten lock-1710 timed pages (five and
their five retakes, `../results/gaps-nvidia_gh200_480gb/private_weight_reference/*/report.json`)
fail validity on V5, the bound on the declaration's own per-M-tile cost: eight
read UNKNOWN (b's far edge 3.8 to 9.1% of the private slope against 3%), the
two G=32 pages FAIL. Nine of the ten also fail the claim C2: PRIVATE's slope
delivers an apparent 4169 to 7005 GB/s against the calibrated 3726.1 GB/s
(the gate allows 4098.7); the tenth (G=4, `-deepb`, 5302d6ff) passes C2 at
4086.7 GB/s and still fails V5 (UNKNOWN). Both gates divide by PRIVATE's slope,
and the slope is small because the calls do not grow with the tread at first:
SHARED at G=2 reads 0.2698, 0.2696, 0.2849, 0.2910, 0.2886, 0.3033 ms at n = 1
to 6, then 0.3459, 0.3906, 0.4365 (0.043 to 0.046 ms a tread) (the deep lines
of `gh200-driver/status`), where the registered SHARED call grows from 0.110 ms
at n = 1 to 0.444 at n = 9 and comes within 3% of the measured one only at
n = 8 and 9 (0.4007, 0.4443). A call-time floor near 0.25 ms that Granite's GEMMs are under until
about n = 6. The pages carry no host timeline, so what sets that floor
(launch rate, the non-GEMM kernels) is not measured here. The pages at treads
to 9 read C2 closest to the ruler (4086.7 to 4464.3 GB/s), those to 6 farthest
(5568.6 to 7005.0), as a floor would make them. No timed ladder saw the power
cap (0x4 in none of their `smi.csv` rows).

**V10 and FL1.** Every byte page fails V10 on w1 (fitted 1623 to 1628 MHz,
with a negative fixed offset, -1.3 to -1.6 us) and the floor's lock capture
fails FL1 (w1 1619, w2 1657). Byte and cycle counts do not depend on the clock.

## What it answered (registered at 0f77622 and a156392, scored on these pages)

Sign convention: predicted / measured - 1.

- **Time, the primary test**: NOT SCORABLE. `scripts/cross_model_score.py`
  (8x7B's fit held, Granite's own counted bytes) refuses: no VALID R3 page
  among the ten.
- **The G >= 8 SHARED slope** over treads 2 to 6 (registered 0.0440 ms per
  tread): NOT SCORABLE, no VALID G=8 or G=32 page. Printed from the INVALID
  pages: 0.0071 to 0.0126 ms per tread on all ten (-71 to -84%), the plateau
  above.
- **The floor, w2** (the slope of `sm__cycles_elapsed.avg` over CTA k-steps
  per SM on the cells of at least 4 waves, live CTAs / (132 x occupancy 4):
  n = 3, 4, 6 at 5.45, 7.27, 10.91 waves): 459.8 cycles per CTA k-step on the
  base-clock capture (`r3f-g64.json`, registered 466.5: -1.43%). HELD. The lock
  capture (`r3f-g64-lock1710.json`, FL1 FAIL) reads 456.1 (-2.22%), just
  outside the 2%. The all-cell slope, the estimator as first registered,
  prints 449.5 and 443.1 (-3.63%, -5.03%). The registered 466.5 is 344.1 +
  979 / 8, the per-CTA fixed cost spread over 8 k-steps; without that term
  the model would print 344.1, 25% under what the base capture reads.
- **The floor, w1**: NOT SCORABLE as registered (one cell of at least 4 waves,
  n = 6 at 5.82). All cells print 374.2 at base and 373.5 on the lock capture
  (registered 365.8: +2.29%, +2.09%), on cells of 1.94 to 5.82 waves.
- **PRIVATE bytes** (72 cells a GEMM on the eight byte pages): w1 3.02% rms,
  worst +4.60%; w2 0.17%, worst +0.54%; no cell beyond 5%. HELD.
- **Printed, not scored:** SHARED bytes w1 19.3% rms (worst +35.2%), w2 3.9%
  (worst +7.6%). Time from predicted bytes has no VALID page.

What the plateau means for the model is in docs/FINDINGS.md (the 2026-09-29/30
held-out queue); none of it is fitted here.

## Files

- `gh200-driver/status`: the driver's ledger, with the RECOVERY lines;
  `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger, `smi.csv` and summary
  (`-t3b`, `-t8b`, `-t32b` and `-deepb` are the recovery's)
- `census-granite-3.0-3b-a800m.json` and its profiles: the model's census
- `recovery/retake_auto.sh`: the retake step as it ran (it decides from the
  driver's ledger which timed pages to take again)

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
