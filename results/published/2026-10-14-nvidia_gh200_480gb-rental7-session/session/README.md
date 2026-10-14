# Rental 7: one NVIDIA GH200 480GB, 15 units of four registrations (docs/registered/README.md, 2026-10-13)

One rental ran `scripts/plans/rental7-2026-10.plan` unattended. It covered model v3's skew pages on
fresh synthetic draws (Mixtral 8x7B and Phi-3.5-MoE, pages A, B and C each, timed under the
1710 MHz lock with `--slip-policy cell --min-clean-repeats 0`), all six Q1 gap pages (Qwen1.5-MoE-A2.7B,
Phi-3.5-MoE and JetMoE-8B at 9 and 15 declared copies), and one OLMoE w1 byte page (`l2open`). The
registrations, their scorers, the synthetic histograms and the plan were committed before any page
(ae010ff, which is also the run commit). Every verdict below is the scorers'
(`scripts/scoring/rental7/`, written and tested on synthetic pages before any page), run on this
directory with no post-page fix. The full tables are in `scripts/scoring/rental7/SCORES.md`, and the
scoring notes in `docs/registered/README.md` ("**Scored 2026-10-14**") and `docs/FINDINGS.md`
(rental 7).

## Where it came from

Lambda Cloud instance 6c3c7e28 (one GH200), 2026-10-14 00:43Z to about 02:18Z: about 1 h 35 min,
about $3.65. Board `02e86e` (`PREFLIGHT.txt`), a board no earlier session ran on.
`scripts/gh200_model_session.sh --plan scripts/plans/rental7-2026-10.plan` ran at commit ae010ff
(`commit.txt`); the plan's sha256 8d7e07a6... is in `gh200-driver/status`. This directory is the
branch `run-gh200-rental7-20261014t0043`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Power limit 700 W, persistence on (`power-set.txt`). The ruler (unit 2)
reads triad 3725.6 GB/s.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, design | start to end (UTC) | min | exit | R3's own stored G1 (not read by the scorers) |
|---|---|---|---|---:|---:|---|
| 1 | prelude | clocks, persistence, 700 W | 00:47:43 to 00:47:43 | 0 | 0 | |
| 2 | calibrate | the ruler (triad 3725.6 GB/s) | 00:47:45 to 00:48:09 | 0 | 0 | |
| 3 | timed | mixtral-8x7b ska (PT, PW, uniform, balanced; n 3, 6, 16) | 00:48:11 to 01:01:49 | 14 | 0 | FAIL (R3 exit 3): drift flag on 3 repeats |
| 4 | timed | mixtral-8x7b skb (DW, uniform, balanced) | 01:01:51 to 01:12:20 | 10 | 0 | FAIL: 4 repeats |
| 5 | timed | mixtral-8x7b skc (PW2, uniform, balanced) | 01:12:22 to 01:22:50 | 10 | 0 | FAIL: 7 repeats |
| 6 | timed | phi-3.5-moe ska (n 2, 4, 12) | 01:22:53 to 01:36:41 | 14 | 0 | FAIL: 13 repeats |
| 7 | timed | phi-3.5-moe skb | 01:36:43 to 01:47:12 | 10 | 0 | FAIL: 14 repeats |
| 8 | timed | phi-3.5-moe skc | 01:47:15 to 01:57:43 | 10 | 0 | FAIL: 10 repeats |
| 9 | timed | qwen1.5-moe-a2.7b q9 (balanced, n 1, 2) | 01:57:46 to 02:00:24 | 3 | 0 | PASS |
| 10 | timed | qwen1.5-moe-a2.7b q15 | 02:00:26 to 02:03:04 | 3 | 0 | PASS |
| 11 | timed | phi-3.5-moe q9 (n 1, 2) | 02:03:07 to 02:05:45 | 3 | 0 | FAIL: 1 repeat |
| 12 | timed | phi-3.5-moe q15 | 02:05:47 to 02:08:25 | 3 | 0 | PASS |
| 13 | timed | jetmoe-8b q9 (n 1, 2, 4) | 02:08:28 to 02:12:16 | 4 | 0 | PASS |
| 14 | timed | jetmoe-8b q15 | 02:12:19 to 02:16:07 | 4 | 0 | PASS |
| 15 | bytes | olmoe-1b-7b l2open, G = 8 counter page | 02:16:10 to 02:18:21 | 2 | 0 | (counter page; analyse exit 3 on V1, as rental 6's byte pages) |

91 minutes from the prelude to the last unit against the plan's 99 minutes of estimates. No unit
was dropped and the driver ends with exit 0. Units 13 and 14 got the tags `-p2-11` and `-p2-12`
(`gh200-driver/tags-used`): the `fresh_tag` fix after rental 6 held. Every timed page reads every
cell's median clock at 1710.0 MHz. No repeat slipped the lock.

**R3's stored G1 against the scorers' rule.** R3 is byte-identical to rental 6's pin, so its
histogram G1 still fails a page for one drift-flagged repeat (the first and last of a repeat's 15
per-burst NVML reads more than 5% apart). That stored G1 fails the six Mixtral and Phi skew pages
and Phi q9, as expected. The registered rental-7 rule (`docs/registered/2026-10-13-rental7-g1rule-gh200`)
is applied by the scorers to each page's `cells.csv`; they never read the stored G1 or the stored
medians. A repeat is non-clean only if its median clock slipped a 15 MHz step under the lock, if 3
or more consecutive reads dipped that far, or if its status is not ok. Rebuilt that way, all twelve
timed pages pass G1 with 0 non-clean repeat rows of 1,332, and every cell keeps 9 of 9 repeats. The
dips are real and isolated: 668 repeats carry at least one read of 1694 MHz or lower (down to 1515
MHz), and no repeat has a run longer than 2. The owner's P20 view (DESCRIPTIVE) also passes every
page; it drops the 52 drift-flagged repeats and loses one cell (Phi skb DW SHARED n 2).

## What it answered

Each verdict is the registered reading (rental 2's rule 3 over ALL and CLEAN). CLEAN equals ALL:
every timed page passes the rebuilt G1, and the byte page passes the counter clock rule (20 of 20
cells in [1640, 1710] MHz).

**skew** (`results/gaps-*-{mixtral-8x7b,phi-3.5-moe}-sk{a,b,c}/`): **E1-SKEW HOLDS** with both
shapes CERTIFIED in CLEAN. 84 cells: rms 0.61%, worst -1.61%, mean -0.24%. Mixtral 42 cells rms
0.51%; Phi 42 cells rms 0.69%. E1-UNI, SKEW-RATIO (v3) and G3 (CI [-0.013, +0.015]%) hold. LT3 is
EXCLUDED as an E1 rival (1.83% against 0.68% on 62 lever cells); v2, H8, v3_all and v3_noca are
UNDECIDED by construction; U is EXCLUDED on the ratios.

**q1** (`results/gaps-*-{qwen1.5-moe-a2.7b,phi-3.5-moe,jetmoe-8b}-q{9,15}/` and the skew pages'
control pairs): **Q1-H FAILS**. The level clause holds (24 pairs, 5 clusters, mean g +0.42 us, se
0.52), but the copies clause fails (c15 minus c9 +2.24 us: Phi n 2 +3.44, Qwen1.5 n 2 +3.05, JetMoE
n 4 +0.22). Rivals v2 and H8 are EXCLUDED; v3_all is NOT EXCLUDED (identical to v3 on these pairs
by construction). **Q1-C SELECTED CALL** (D +0.58 us against DEAD's -5.47). Q1-P is DESCRIPTIVE:
Mixtral n 3 mean g -10.56 us, against v3_all's -10.61. Q1-SK is printed.

**l2open** (`results/2026-10-14-nvidia_gh200_480gb-olmoe-1b-7b-l2open-r3-counters/lock1710/`):
**NAMED ROWKEY60** (OPEN). On the 10 skewed w1 rows: row-key at 60 MiB rms 1.99% (worst -3.21%),
row-key at 39.5 MiB 8.10%, v3 15.33% (v3 +7.7% at n 4 to +25.8% at n 16).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `gh200-driver/tags-used`, `logs/driver-*.log`:
  the ledger, the consoles and the session tags the driver handed out
- `results/gaps-nvidia_gh200_480gb-<model>-sk{a,b,c}/private_weight_reference/`: the timed
  histogram pages (units 3 to 8), each with `report.json`, `cells.csv` (one row per timed repeat,
  the rebuilt rule's input) and `DECLARED_COPIES.txt`; `locked-r3/*-p2*/` are their lock ledgers
- `results/gaps-nvidia_gh200_480gb-<model>-q{9,15}/private_weight_reference/`: the six Q1 pages
  (units 9 to 14)
- `results/2026-10-14-nvidia_gh200_480gb-olmoe-1b-7b-l2open-r3-counters/lock1710/`: the byte page
  (`r3c-g8.json`, unit 15) with `summary.json`, profiles and Triton caches
- `census*.json`, `census*.profiles/`: the censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console;
  `logs/` holds every per-unit log, named by model and label

**Histograms.** Every histogram page reads one of the registered SYNTHETIC files under
`docs/registered/2026-10-13-rental7-skew-hist/` (skew pages, byte page) or
`docs/registered/2026-10-09-rental6-skew-hist/` (Q1 pages), with sha256 on every page and G5 PASS.
No trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session
does (the files are the run branch's, byte for byte); prose here names the board by
`r3_timing_model.board()`. The Triton caches' compiled `cuda_utils*.so` files (15) are left out, as
in rentals 1 to 6. Every other file of the run branch is here, byte for byte (449 files against
SHA256SUMS).
