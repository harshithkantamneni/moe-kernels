# Settle rental S1: one NVIDIA GH200 480GB, 14 units of one descriptive registration (docs/registered/README.md, 2026-10-21)

One rental ran `scripts/plans/rentalS1-settle-2026-10.plan` unattended through the settle rental's driver
(`scripts/gh200_settleS1_session.sh`, tile rental A2's pinned driver plus lines marked S1). It set the GPU power
limit, read it back, ran a sustained GEMM to see which limit held (M0), and then the five measurements
DIAGNOSIS-A2's section Q7 could not settle: kernel against application replay on three E1-CAL cells (M1), the
OLMoE f128 counter page at 1410, 1305 and 1005 MHz (M2), Mixtral 8x7B 128x128 G 1 byte pages (M3), PRIVATE-arm
ladder pages on Mixtral at 128x128 G 16 and 64x128 G 1 (M4) and three stall and pipe metrics on 128x128 G 1 byte
cells of Qwen2-57B and OLMoE (M5). The registration (`2026-10-21-settleS1-gh200`), its reader, the SYNTHETIC
histograms and the plan were committed before any page (52ed2cac, which is also the run commit).

S1 is DESCRIPTIVE: it certifies nothing and no model verdict is scored on its pages, and every page is SEEN for
model v5. Every reading below is the registered reader's (`scripts/scoring/settleS1/read_s1.py`, pinned by the
registration), run on this directory with no post-page fix; its output is
`scripts/scoring/settleS1/settleS1.read.{json,txt}`, and the notes are in `docs/registered/README.md` ("**Read
2026-10-21**") and `docs/FINDINGS.md` (settle rental S1).

## Where it came from

Lambda Cloud instance 702c75b6 (one GH200, us-east-3), launched 07:04:51Z and terminated 07:35:00Z on
2026-10-21: about 30 min, about $1.15 at the registration's $2.29/h (an estimate; the instance, the region
and the two times are the rental's records, not in the run files, and the bill is in neither). The VM's setup ran 07:06:36Z to 07:08:07Z and the driver
07:08:08Z to 07:33:36Z. Board `9da3c0` (`PREFLIGHT.txt`), which no earlier published session names.
`scripts/gh200_settleS1_session.sh --plan scripts/plans/rentalS1-settle-2026-10.plan` ran at commit 52ed2cac
(`commit.txt`); the plan's sha256 79891a29... is in `gh200-driver/status`. This directory is the branch
`run-gh200-tile1-20261021t0704`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and push
log. Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was set.

## The power limit

The registration asks for 900 W and refuses unless two read-backs agree: `--query-gpu=enforced.power.limit` and
the GPU scope's Current Power Limit in `nvidia-smi -q -d POWER`. Before the limit was set (`power-at-start.txt`)
the GPU scope read Current and Requested 700 W, Default 900 W, Max 900 W; the module scope 1000 W. The prelude
ran `nvidia-smi -pl 900 -sc 0`, which printed "Power limit for GPU 00000000:DD:00.0 was set to 900.00 W from
700.00 W", and polled for 30 s; at no poll did both readings give 900 W, and the last read gives
enforced.power.limit 700 W, the -q GPU scope Current and Requested 700 W, and power.limit 900 W
(`logs/driver-prelude-mixtral-8x7b.log`, `power-set-unsettled-900.txt`, `gh200-driver/s1-power.env`
`S1_POWER_900=refused`). By the owner's pre-authorised fallback the prelude then ran `-pl 700 -sc 0` and read
back 700 W on all three: enforced.power.limit, the -q GPU scope Current (Requested 700 W) and power.limit
(`power-set.txt`, `S1_POWER_W=700`). Every unit ran at 700 W: the driver's `clocks:` line after every unit
reads power.limit 700 W, all 48 power.limit records in the 12 counter pages read 700.00 W, and every one of the
power proof's 896 samples reads enforced.power.limit and power.limit 700 W.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | measurement | start to end (UTC) | min | exit | page exit |
|---|---|---|---|---|---:|---:|---|
| 1 | prelude | 900 W refused, 700 W read back; locks 1305, 1005, 1410 held | power | 07:08:08 to 07:08:46 | 1 | 0 | |
| 2 | powerproof | pp: bf16 matmul 8192^3, 90 s, no lock | M0 | 07:08:47 to 07:10:20 | 2 | 0 | GEMM exit 0, 896 samples |
| 3 | bytes | olmoe-1b-7b kr (kernel replay, G 1) | M1 | 07:10:21 to 07:11:13 | 1 | 0 | 0 |
| 4 | bytes | olmoe-1b-7b ar (application replay, G 1) | M1 | 07:11:15 to 07:12:19 | 1 | 0 | 0 |
| 5 | bytes | qwen2-57b-a14b kr | M1 | 07:12:21 to 07:13:24 | 1 | 0 | 0 |
| 6 | bytes | qwen2-57b-a14b ar | M1 | 07:13:26 to 07:14:38 | 1 | 0 | 0 |
| 7 | bytes | olmoe-1b-7b f1305 (G 64 at 1305 MHz) | M2 | 07:14:40 to 07:17:44 | 3 | 0 | 3 |
| 8 | bytes | olmoe-1b-7b f1005 (G 64 at 1005 MHz) | M2 | 07:17:47 to 07:21:17 | 4 | 0 | 3 |
| 9 | bytes | olmoe-1b-7b f1410 (G 64 at 1410 MHz, the same-board reference) | M2 | 07:21:20 to 07:24:16 | 3 | 0 | 3 |
| 10 | bytes | mixtral-8x7b b128g1 (G 1) | M3 | 07:24:19 to 07:25:39 | 1 | 0 | 0 |
| 11 | bytes | mixtral-8x7b p128g16 (G 16, treads 3, 4, 8) | M4 | 07:25:42 to 07:28:13 | 3 | 0 | 3 |
| 12 | bytes | mixtral-8x7b p64g1 (64x128 w8, G 1, tread 2) | M4 | 07:28:16 to 07:29:24 | 1 | 0 | 3 |
| 13 | bytes | qwen2-57b-a14b st (G 1, stall metrics) | M5 | 07:29:27 to 07:32:05 | 3 | 0 | 0 |
| 14 | bytes | olmoe-1b-7b st (G 1, stall metrics) | M5 | 07:32:08 to 07:33:33 | 1 | 0 | 0 |

14 of 14 units exit 0 in about 25 min of units against 47 estimated; none was dropped; the driver ends with exit
0. A page exit 3 is the route's own page gates (the three G 64 pages: V4, V5, C1 and C6 FAIL, C2 REFUSE, V10
PASS; the two ladder pages: V1 FAIL), which S1 does not read: no S1 reading uses a page verdict. Each unit's
`analyse` step exits 3 as well (`summary.json`), for the same reason.

## What the reader reads (`scripts/scoring/settleS1/settleS1.read.txt`)

| measurement | reading | numbers |
|---|---|---|
| power | 900 W refused, 700 W read back | limit 700 W |
| M0 | CAP-BELOW-680 | 884 of 896 samples busy (power.draw over 200 W); largest 1 s mean 662.96 W; 0 s of 1 s means over 720 W; SW power cap (0x4) on 884 of 884 busy samples and no other reason; busy clock median 1410 MHz against clocks.max.sm 1980; enforced.power.limit 700 W |
| M1 OLMoE | INSTRUMENT-INDEPENDENT | d = off_app - off_kernel: n 1 PT -0.44 us, n 4 PT +3.47 us, median 1.52 us |
| M1 Qwen2-57B | INSTRUMENT-INDEPENDENT | n 2 PT +9.65 us (off_kernel 100.77 us, bound 30.19 us) |
| M2 | 1305 PRESENT, 1005 GONE | A (tau) by lock: 1410 47.32 us (450 us), 1305 37.74 us (575 us), 1005 0.70 us (75 us); 54 cells each; reference this board's 1410 page |
| M3 | W1-EXACT | mean e, v3 and CLAMP alike: w1 +0.10%, w2 +21.3% (4 cells) |
| M4 p128g16 | w1 A-DOMINATED, w2 A-DOMINATED | e_P / (q_S - 1): w1 0.56, 1.0, 1.0; w2 0.29, 0.71, 1.0 at treads 3, 4, 8 |
| M4 p64g1 | w1 B-DOMINATED, w2 MIXED | w1 e_P 0.0027 weight sets; w2 e_P 0.0211, share 0.207 |
| M5 Qwen2-57B | LATENCY | Spearman rho with x_s1 over 16 GEMMs: long scoreboard 0.83, barrier 0.98, tensor active -0.73 |
| M5 OLMoE | NONE-SEEN | over 24 GEMMs: long scoreboard 0.64, barrier 0.82, tensor active -0.47 |

## The clock

The prelude locked, read back and reset 1305, 1005 and 1410 MHz once each; each held. On the counter pages the
per-GEMM effective clock (sm__cycles_elapsed.avg over gpu__time_duration.sum) ran under the lock at 1410 and
1305 and at it at 1005: on the M2 pages the median is 1333.1 MHz at 1410, 1255.7 at 1305 and 1004.1 at 1005;
on the M1 pages 1278 to 1352 MHz at 1410 in both replay modes. The power proof, with no lock and the SW power
cap on throughout, ran at a median 1410 MHz (1395 to 1905) and peaked at 75 C; the GEMM reads a median 642.45
TFLOP/s (`gemm.json`).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/s1-power.env`: the power outcome; `gh200-driver/locks.env`: the unit locks the prelude probed
- `power-at-start.txt`, `power-set-unsettled-900.txt`, `power-set.txt`: the -q -d POWER snapshots before `-pl`,
  at the end of the 900 W poll and after the 700 W read-back
- `results/2026-10-21-nvidia_gh200_480gb-powerproof-pp/`: M0's `smi.csv` (100 ms samples), `gemm.json` and the
  `power-before.txt` and `power-after.txt` snapshots
- `results/2026-10-21-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock<L>/`: the twelve counter pages
  (M1 kr and ar, M2 f1410, f1305 and f1005, M3 b128g1, M4 p128g16 and p64g1, M5 st), each `r3c-g<G>.json` with
  `summary.json`, profiles and Triton caches
- `census*.json`, `census*.profiles/`: the per-model GEMM censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console; `logs/`
  holds every per-unit log, named by model and label

**Histograms.** Every histogram page (the seven G 1 pages of M1, M3 and M5) reads one of the registered
SYNTHETIC files under `docs/registered/2026-10-21-settleS1-hist/`, with its sha256 on the page. No
trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session does (the
files are the run branch's, byte for byte); prose here names the board by its hash. The Triton caches'
compiled `cuda_utils*.so` files (15) are left out, as in every earlier session. Every other file of the run
branch is here, byte for byte (455 files against SHA256SUMS, which lists 470).
