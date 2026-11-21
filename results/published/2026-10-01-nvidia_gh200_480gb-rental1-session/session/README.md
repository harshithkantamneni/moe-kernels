# Rental 1: one NVIDIA GH200 480GB, 14 units of four registrations (docs/registered/README.md, 2026-09-30 and 2026-10-01)

One rental ran `scripts/plans/rental1-2026-10.plan` unattended: the Mixtral 8x7B TP=8 floor capture registered on 2026-09-30, then the byte pages of the A-tile k-steps registration (C) and the L2-survival TP-shard registration (B), then the launch floor (A), each registered before any page at 436e41c. Every verdict below is scored against the registered files as committed, with each design's own scorer; the full scoring notes are in `docs/registered/README.md` ("**Scored 2026-10-01**", one per registration) and `docs/FINDINGS.md` (rental 1).

## Where it came from

Lambda Cloud instance b77ae7afc5bf4a9682e79d853c7a8d6e, gpu_1x_gh200, us-east-3, launched
2026-10-01 20:15:30 UTC, active 20:17:48 UTC, terminated 22:20:53 UTC by the laptop's guard
once every pushed file had been verified (976 files match SHA256SUMS), gone 22:21:54 UTC; ids
and times from the laptop's `~/.lambda/rent-ledger.log`. Board `7269a7` (`PREFLIGHT.txt`, a
board no earlier session used), driver 570.148.08 upgraded to 580.105.08 before setup
(`driver580.log`). `scripts/gh200_model_session.sh --plan scripts/plans/rental1-2026-10.plan`
ran at commit 436e41c (`commit.txt`), which carries the three 2026-10-01 registrations; the
plan's sha256 655fdf59... is in `gh200-driver/status`. This directory is the branch
`run-gh200-rental1-20261001t2017` at 77c0a45; `SHA256SUMS.run-branch` and
`PUSHES.run-branch.txt` are its manifest and push log (40 pushes, one per unit). Power limit
700 W, persistence on (`power-set.txt`); the 1710 MHz lock was held for every byte page and
the launch floor (`lock-plan.txt`, `clocks-after-*.txt`).

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, design | start to end (UTC) | min | exit | pages (each page's exit) |
|---|---|---|---|---:|---:|---|
| 1 | prelude | - | 20:21:24 | 0 | 0 | |
| 2 | floor | mixtral-8x7b-tp8, G = 64, 8, n = 1..10 | 20:21:25 to 20:23:53 | 2 | 3 | r3f-g64, r3f-g8 (0); r3f-g64-lock1710, r3f-g8-lock1710 (3, FL1) |
| 3 | bytes | mixtral-8x7b-tp8 atile, G = 64 32 16 8 | 20:23:56 to 20:30:00 | 6 | 0 | 4 pages (3) |
| 4 | bytes | mixtral-8x7b-tp4 atile, G = 64 32 16 | 20:30:02 to 20:36:53 | 7 | 0 | 3 pages (3) |
| 5 | bytes | mixtral-8x7b-tp2 atile, G = 64 32 16 | 20:36:56 to 20:47:47 | 11 | 0 | 3 pages (3) |
| 6 | bytes | mixtral-8x7b sameboard, G = 16 32 8 1 | 20:47:50 to 21:12:29 | 25 | 0 | G=16, 32 (3); G=8 (1); G=1 (0) |
| 7 | bytes | qwen2-57b-a14b g128, G = 128 | 21:12:33 to 21:20:30 | 8 | 0 | G=128 (1) |
| 8 | bytes | olmoe-1b-7b g128, G = 128 | 21:20:33 to 21:23:23 | 3 | 0 | G=128 (3) |
| 9 | bytes | mixtral-8x7b-tp2 l2, G = 1 2 | 21:23:27 to 21:30:35 | 7 | 0 | 2 pages (3) |
| 10 | bytes | mixtral-8x7b-tp4 l2, G = 1 2 | 21:30:39 to 21:35:09 | 5 | 0 | 2 pages (3) |
| 11 | bytes | mixtral-8x7b-tp8 l2, G = 1 2 | 21:35:13 to 21:38:23 | 3 | 0 | 2 pages (3) |
| 12 | launchfloor | granite-3.0-3b-a800m, n = 1..9, E240 E0 E480 GR | 21:38:27 to 21:58:58 | 21 | 0 | |
| 13 | launchfloor | jetmoe-8b, n = 1..4, E240 E0 E480 GR | 21:59:02 to 22:08:20 | 9 | 0 | |
| 14 | launchfloor | granite-3.0-1b-a400m, n = 1..9, E240 GR | 22:08:25 to 22:18:45 | 10 | 0 | |

117 minutes of units against the plan's 148-minute estimate; no unit was dropped. The driver
ends with exit 3 because unit 2's lock captures fail FL1; the recovery step found nothing to
retake (`RECOVERY start: timed G to retake [], deep G to retake []`; `recovery/retake_auto.sh`
is the script as it ran). A byte page's exit 3 is a VALIDITY gate: V10 on every tp-shard page but tp2 atile G=32
and on OLMoE's (short-kernel lock pages read off 1710 by their own counters, GEMM fits 1658
to 1703 MHz, tp2's n = 1 w2 cells up to 1783 MHz), V7 (NATIVE against SHARED bytes) on
sameboard G=16 and 32 and five atile pages, V6 on tp8 atile G=8; exit 1 a CLAIM gate (sameboard G=8, Qwen2 G=128: C1/C2 on SHARED, C6 on Qwen2's
PRIVATE excess). The registrations say V10 does not gate a byte or survival score; no failing
gate is on a quantity scored below except as stated (each page's `gates` list).

## What it answered

**tp8 floor (2026-09-30 registration)**, base captures
`results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters/r3f-g{64,8}.json`,
each cell repriced with its own `dram__bytes_read.sum`:

| falsifier | registered | G = 64 | G = 8 | verdict |
|---|---|---|---|---|
| F1 D = c(n=4) - c(n=2), w1 | 155,193 +/- 11,345 | 156,941 | 155,820 | HELD |
| F2 w1 n = 2 | 162,474, -3% to +8% | 170,316 (+4.8%) | 170,367 (+4.9%) | HELD |
| F3 every other cell within 8% | | w1 n=1 +13.4%, w2 n=1 +8.6%, n=2 +10.4%, n=3 +8.3% | w1 n=1 +14.9%, w2 n=2 +11.1% | FALSIFIED |
| F4 w1, n = 6, 8, 10 | 352.2 +/- 2% | 339.6 (-3.6%) | 342.5 (-2.8%) | FALSIFIED |
| F4 w2, n = 5, 6, 8, 10 | 379.1 +/- 2% | 374.3 (-1.3%) | 374.2 (-1.3%) | HELD |

The rivals: old lifetime D 87,121 and per-fetching-CTA 121,171 are outside F1 (refuted);
plain throughput (D 158,705, w1 n = 2 +7.0%) is inside both, as the registration said it
would be. The lock captures (`r3f-g{64,8}-lock1710.json`, FL1 FAIL: w1 fitted 1671 and 1662
MHz, w2 1655) give the same verdicts.

**A: the launch floor** (`results/2026-10-01-nvidia_gh200_480gb-launch-floor-rental1/*/cells.csv`,
`manifest.json`, `traces/`): P1 HELD (graph replay GR within [0.80, 1.10] x C_reg on 15 of 15
Granite-3B cells, n = 1 at 0.084 to 0.090 ms; increments 0.163 and 0.233 ms). P2 FALSIFIED
(1 of 15: Granite NATIVE n = 7, GR 8.9 us under E240, a cell that was still host-bound on this
host). P3 FALSIFIED (6 of 6 outside +/-12 us; measured E0 - E240 is +45 to +65 us). P4
FALSIFIED (the moved verdicts all right on each cell's own H, 5 wrong on the printed list
priced at H = 0.3545 ms; the shift -54 to -73 us against -67.4, PRIVATE n = 1 outside +/-12 us). P5 HELD (98.1%, 95.5%, 100%). P6 FALSIFIED (55
of 57; Granite SHARED and PRIVATE n = 2 at E480 miss by -14 and -17 us). P7 classifies: TR-G
kernels 79 to 85 us at n = 1, non-GEMM 11 us, launch API 11% of the traced call. P8 FALSIFIED
(GR outside P1's band on 3 of 27 Granite-1B cells, P5's rule 27 of 27; the plateau 0.328 to
0.371 ms, a record outside the seen [0.23, 0.27] band). In each of the three processes the
cells timed before the first kineto trace enqueue a call in 0.30 to 0.35 ms (Granite-3B
NATIVE n = 1 E240 reads 0.2463 ms, the published page 0.2456) and every cell after it in
0.38 to 0.46 ms (`cells.csv`, `host_enqueue_ms / calls_per_burst`); the host-bound eager
cells read 14 to 33% above the published Granite G = 4 cells, the GPU-bound ones within 0.6%.
No launch-floor scorer was registered (`launch_floor.py` records, it does not score): P1 to P8
were scored by a script written after the pages, committed as
`scripts/scoring/rental1/score_launch.py`, and P6's max-plus model is that script's reading of
the registered text.

**B: L2 survival on TP shards** (`results/*-mixtral-8x7b-tp{2,4,8}-l2-r3-counters/lock1710/r3c-g{1,2}.json`,
anchor `results/*-mixtral-8x7b-sameboard-r3-counters/lock1710/r3c-g1.json`), s at n = 2..9:
tp2 w2 0.82 to 0.54, against the same-board 8x7B w2 0.72 to 0.48: H1 (within 0.066, under
0.65 at n = 4..9). tp4 w1 0.65 to 0.36: INCONCLUSIVE. The x-only law FALSIFIED (tp2 w2 and tp4
w1 differ by 0.10 to 0.19 at all six of n = 4..9). Controls: tp8 w2 HELD (>= 0.973); tp8 w1
(0.82 to 0.91), tp4 w2 (n = 2 only) and tp2 w1 (n = 2, 3) FAILED: by the registration the law's
shape failed. PRIVATE F / Mn HELD (0.504 to 0.521). G = 2 against R0: PRIVATE HELD, SHARED
FALSIFIED (31 of 54 beyond 5%).

**C: A-tile reads** (`results/*-{mixtral-8x7b-tp{8,4,2}-atile,mixtral-8x7b-sameboard,qwen2-57b-a14b-g128,olmoe-1b-7b-g128}-r3-counters/lock1710/r3c-g*.json`):
T1 HELD on OLMoE (+0.52% worst from R1), INCONCLUSIVE on Qwen2 (nearer R1 everywhere, R0 -8.5
to -11.6%, but +2.1 to +2.6% from R1 at n = 6, 7, 9). T2: rho42 1.199, rho84 1.184, between the
byte-law bands and R2's: every candidate refuted. T3 PASS on tp8 (q - 1 -0.00438, R1 -0.00446)
and tp4 (-0.00138, R1 -0.00193); tp2 a record. T4 FALSIFIED for every candidate (R2 primary
rms 1.55% but 10 cells of G >= 32, n >= 6 beyond 4%).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and consoles
- `results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters/`: the floor captures (unit 2)
- `results/2026-10-01-nvidia_gh200_480gb-*-r3-counters/lock1710/`: byte pages, one directory per unit (3 to 11), each with `summary.json`
- `results/2026-10-01-nvidia_gh200_480gb-launch-floor-rental1/<model>/`: the launch floor's cells, manifest and kineto traces (12 to 14)
- `census-*.json`, `census-*.profiles/`: each model's census, written before its first page
- `recovery/retake_auto.sh`: the retake step as it ran (nothing to retake)

The `card:` lines and every page's CARD block carry the card's UUID, as every
published session does.
