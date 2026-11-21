# Rental 6: one NVIDIA GH200 480GB, 21 units of four registrations (docs/registered/README.md, 2026-10-09)

One rental ran `scripts/plans/rental6-2026-10.plan` unattended: model v3's skew pages (Mixtral
8x7B, OLMoE and Phi-3.5-MoE, pages A, B and C each, timed under the 1710 MHz lock with
`--slip-policy cell`), the Q1 gap pages (Qwen1.5-MoE-A2.7B, Phi-3.5-MoE and JetMoE-8B at 9 and 15
declared copies), the OLMoE and Phi byte legs, and the second-K counter pages on
mixtral-8x7b-tp4 (BLOCK_K 64 and 32, s4). The registrations, their scorers, the synthetic
histograms and the plan were committed before any page (650d295, which is also the run commit).
Every verdict below is the scorers' (`scripts/scoring/rental6/`, written and tested on synthetic
pages before any page), run on this directory with no post-page fix. The full tables are in
`scripts/scoring/rental6/SCORES.md`, the scoring notes in `docs/registered/README.md` ("**Scored
2026-10-13**") and `docs/FINDINGS.md` (rental 6).

## Where it came from

Lambda Cloud instance 19aa9f52, gpu_1x_gh200, 2026-10-13 10:50Z to 12:57Z: 2 h 8 min, about
$4.90. Board `ff9942` (`PREFLIGHT.txt`), a board no earlier session ran on.
`scripts/gh200_model_session.sh --plan scripts/plans/rental6-2026-10.plan` ran at commit 650d295
(`commit.txt`); the plan's sha256 57beae34... is in `gh200-driver/status`. This directory is the
branch `run-gh200-rental6-20261013t1050`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Power limit 700 W, persistence on (`power-set.txt`). The ruler (unit 2)
reads triad 3725.7 GB/s.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, design | start to end (UTC) | min | exit | pages (gates not PASS) |
|---|---|---|---|---:|---:|---|
| 1 | prelude | clocks, persistence, 700 W | 10:53:33 to 10:53:34 | 0 | 0 | |
| 2 | calibrate | the ruler (triad 3725.7 GB/s) | 10:53:35 to 10:54:00 | 0 | 0 | |
| 3 | timed | mixtral-8x7b ska (PT, PW, uniform, balanced; n 3, 6, 16) | 10:54:02 to 11:07:50 | 14 | 0 | G1_lock_thermal FAIL (R3 exit 3): drift flag on 8 cells; 1 slipped repeat of 216 (1620 MHz), out of its cell median |
| 4 | timed | mixtral-8x7b skb (DW, uniform, balanced) | 11:07:52 to 11:18:21 | 10 | 0 | G1 FAIL: drift flag on 4 cells |
| 5 | timed | mixtral-8x7b skc (PW2, uniform, balanced) | 11:18:23 to 11:28:52 | 10 | 0 | G1 FAIL: drift flag on 5 cells |
| 6 | timed | olmoe-1b-7b ska (n 3, 6, 12) | 11:28:54 to 11:43:03 | 14 | 0 | none |
| 7 | timed | olmoe-1b-7b skb | 11:43:05 to 11:53:54 | 11 | 0 | none |
| 8 | timed | olmoe-1b-7b skc | 11:53:56 to 12:04:44 | 11 | 0 | none |
| 9 | timed | phi-3.5-moe ska (n 2, 4, 12) | 12:04:47 to 12:18:55 | 14 | 0 | G1 FAIL: drift flag on 5 cells; 1 cell lost (under 6 clean repeats) |
| 10 | timed | phi-3.5-moe skb | 12:18:58 to 12:29:36 | 11 | 0 | G1 FAIL: drift flag on 5 cells; 1 cell lost |
| 11 | timed | phi-3.5-moe skc | 12:29:39 to 12:40:17 | 11 | 0 | G1 FAIL: drift flag on 5 cells; 2 cells lost |
| 12 | timed | qwen1.5-moe-a2.7b q9 (balanced, n 1, 2) | 12:40:20 to 12:42:58 | 3 | 0 | none |
| 13 | timed | qwen1.5-moe-a2.7b q15 | 12:43:01 to 12:43:07 | 0 | 3 | no page: the dry run was refused (`--session-tag -lock1710`) |
| 14, 15 | timed | phi-3.5-moe q9, q15 | 12:43:09 to 12:43:23 | 0 | 3 | no page (the same) |
| 16 | bytes | olmoe-1b-7b skbytes, G = 8 counter page | 12:43:26 to 12:45:37 | 2 | 0 | the histogram byte page records no gate |
| 17 | bytes | phi-3.5-moe skbytes, G = 8 counter page | 12:45:40 to 12:50:12 | 5 | 0 | the histogram byte page records no gate |
| 18, 19 | timed | jetmoe-8b q9, q15 | 12:50:15 to 12:50:30 | 0 | 3 | no page (the same) |
| 20 | bytes | mixtral-8x7b-tp4 G = 64 k64s4 | 12:50:33 to 12:53:06 | 3 | 0 | V7 FAIL; page exit 3 on CLAIM C2 / C6 (byte findings) |
| 21 | bytes | mixtral-8x7b-tp4 G = 64 k32s4 | 12:53:09 to 12:55:27 | 2 | 0 | V7 FAIL; page exit 3 on CLAIM C2 / C6 |

122 minutes from the prelude to the last unit against the plan's 145 minutes of estimates. No
unit was dropped. The driver ends with exit 3 (the five lost timed units). Every timed page reads
every cell's median clock at 1710.0 MHz.

**The lost units (a driver defect).** `fresh_tag` in `scripts/gh200_model_session.sh` had ten
suffixes ("", b to j). Units 3 to 12 took `$B-p2` to `$B-p2j` (`gh200-driver/tags-used`); unit 13,
the eleventh timed unit, got an empty tag, its R3 dry run read `--session-tag -lock1710`, and
argparse refused it ("expected one argument", `logs/r3-g8-t2-dry-q15.log`), so the driver
recorded "no design passed its dry run" and moved on; so did units 14, 15, 18 and 19. Their
directories hold only `DECLARED_COPIES.txt`. Fixed after the session: the tag counts on past ten
(`$B-p2-11`, ...), and every caller refuses its step on a failed tag.

**The six G1 failures (a gate defect).** R3's histogram G1 fails a page if any repeat of any cell
is drift-excluded. The Mixtral and Phi pages carry 6, 9, 10, 11, 14 and 17 flagged repeats (3.7
to 10.6% of their repeat rows, nearly all at the smallest tread), each already out of its cell's
median; every cell reads 1710.0 MHz. The rental-6 G1 is "R3's G1 PASS and slipped rows at most
10%", so ALL counts these pages and CLEAN drops them. The drift flag compares the first and last
of a repeat's 15 per-burst clock reads; see `scripts/scoring/rental6/SCORES.md`.

**The dry-run logs.** They are named by G, tread and unit label and not by model, so a later
unit with the same three overwrote an earlier one's: Phi's `r3-g8-t12-dry-sk{a,b,c}.log` replaced
OLMoE's, and Phi's `r3-g8-t2-dry-q9.log` replaced Qwen1.5's. The units' own outcomes are in
`gh200-driver/status` and the `locked_r3-*.log` files, which are per tag.

## What it answered

Each verdict is the registered reading (rental 2's rule 3 over ALL and CLEAN). CLEAN drops the
six Mixtral and Phi timed pages (G1); every byte and counter page is in both views.

**skew** (`results/gaps-*-{mixtral-8x7b,olmoe-1b-7b,phi-3.5-moe}-sk{a,b,c}/`,
`results/*-skbytes-r3-counters/`): G3 PASSES (ALL 51 cells, 9 pages, CI [-0.140, -0.003]%;
CLEAN 18 cells, OLMoE's 3 pages, CI [-0.227, -0.208]%). **E1-SKEW HOLDS**: model v3 from
predicted bytes, ALL 122 cells on three models rms 0.55%, worst -1.55%, mean -0.00%; CLEAN 42
OLMoE cells rms 0.54%, worst -1.47%, mean +0.16%. E1-UNI, SKEW-RATIO (v3), E2-SKEW and B-SKEW
HOLD; U is EXCLUDED, v2 EXCLUDED as an E1 rival; LT3, H8 and v3_all INCONCLUSIVE. OPEN-L2SKEW:
v3 FAILS OLMoE's skewed w1 bytes (+4.8% at n 3 to +20.6% at n 12) and the 39.5 MiB row-key rival
is NOT BETTER; B-CA: v3 FAILS (OLMoE w1 controls +3.6%), v3_noca EXCLUDED.

**q1** (`results/gaps-*-qwen1.5-moe-a2.7b-q9/` and the skew pages' control pairs): Q1-H
UNDECIDED (no c15 page for the copies clause; ALL 10 pairs mean g -0.98 us); Q1-C NOT SCORED;
Q1-P DESCRIPTIVE (Mixtral n 3 -6.38 us); Q1-SK printed.

**secondk** (`results/2026-10-13-nvidia_gh200_480gb-mixtral-8x7b-tp4-k{64,32}s4-r3-counters/`):
UNDECIDED: w1 reads H_ITER (+0.78%; H_CTA +17.8%), w2 NEITHER (H_ITER +3.17%, 0.17 point past
the 3% tolerance; H_CTA +26.3%).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `gh200-driver/tags-used`, `logs/driver-*.log`:
  the ledger, the consoles and the session tags the driver handed out
- `results/gaps-nvidia_gh200_480gb-<model>-sk{a,b,c}/private_weight_reference/`: the timed
  histogram pages (units 3 to 11), each with `DECLARED_COPIES.txt`; `locked-r3/*-p2*/` their lock
  ledgers
- `results/gaps-nvidia_gh200_480gb-qwen1.5-moe-a2.7b-q9/`: the one Q1 page (unit 12); the five
  `-q9` / `-q15` directories of units 13 to 15, 18 and 19 hold `DECLARED_COPIES.txt` only
- `results/2026-10-13-nvidia_gh200_480gb-{olmoe-1b-7b,phi-3.5-moe}-skbytes-r3-counters/lock1710/`:
  the byte-leg counter pages (`r3c-g8.json`, units 16 and 17) with `summary.json`, profiles and
  Triton caches
- `results/2026-10-13-nvidia_gh200_480gb-mixtral-8x7b-tp4-k{64,32}s4-r3-counters/lock1710/`: the
  second-K pages (`r3c-g64.json`, units 20 and 21)
- `census*.json`, `census*.profiles/`: the censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console

**Histograms.** Every histogram page reads one of the registered SYNTHETIC files under
`docs/registered/2026-10-09-rental6-skew-hist/` (sha256 on every page, G5 PASS); no
trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session
does; prose here names the board by `r3_timing_model.board()`. The Triton caches' compiled
`cuda_utils*.so` files (18) are left out, as in rentals 1 to 5; every other file of the run branch
is here, byte for byte (511 files against SHA256SUMS).
