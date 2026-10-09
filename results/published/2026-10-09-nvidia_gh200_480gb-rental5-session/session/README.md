# Rental 5: one NVIDIA GH200 480GB, 22 units of six registrations (docs/registered/README.md, 2026-10-07)

One rental ran `scripts/plans/rental5-2026-10.plan` unattended: the stamps regcheck (rc), the
synthetic-skew histogram pages of Mixtral 8x7B, OLMoE and Qwen1.5-MoE-A2.7B (skew: pages A and B
timed, page C the byte leg), the OLMoE 15-copy SHARED page under skew (c15), the perturbation gate
(pt) and the five stamps units it gates (stamps2), the second-K counter pages on
qwen2-57b-a14b-tp8 (secondk) and the OLMoE G = 64 BLOCK_K 64 / 128 timed pages (bk128). The
registrations, their scorers, the synthetic histograms and the plan were committed before any page
(d1a81d7); the run commit is d4d1767. Every verdict below is the scorers'
(`scripts/scoring/rental5/`, written and tested on synthetic pages before any page), run on this
directory with one flagged post-page fix to reading code (`score_skew.py`: a page whose G1
lock / thermal gate failed is ruled by the registered ALL / CLEAN views instead of being refused in
both; it changes G3 from FAIL to INCONCLUSIVE). The full tables are in
`scripts/scoring/rental5/SCORES.md`, the scoring notes in `docs/registered/README.md` ("**Scored
2026-10-09**") and `docs/FINDINGS.md` (rental 5).

## Where it came from

Lambda Cloud instance 0e169708, gpu_1x_gh200, 2026-10-08 23:26Z to 2026-10-09 01:21Z: 1 h 55 min,
about $4.40. Board `1cd741` (`PREFLIGHT.txt`), a board no earlier session ran on.
`scripts/gh200_model_session.sh --plan scripts/plans/rental5-2026-10.plan` ran at commit d4d1767
(`commit.txt`); the plan's sha256 b23536c5... is in `gh200-driver/status`. This directory is the
branch `run-gh200-rental5-20261008t2326`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Power limit 700 W, persistence on (`power-set.txt`). The ruler (unit 2)
reads triad 3727.6 GB/s.

**Two earlier launches** were terminated at start, before any unit, by `scripts/vm_run.sh`
refusing its own plan print: instance e7f6783d (2026-10-07 17:49Z, run commit d1a81d7: the
pre-setup dry run could not read the plan's histogram pages, fixed in c384cd7) and instance
b44558da (2026-10-08 07:22Z, run commit c384cd7: ssh read the copy loop's stdin, so only the first
histogram page arrived, fixed in d4d1767). Each lasted a few minutes, about $0.20 for both; neither
wrote a page, and their run branches carry only the first commit.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, design | start to end (UTC) | min | exit | pages (gates not PASS) |
|---|---|---|---|---:|---:|---|
| 1 | prelude | clocks, persistence, 700 W | 23:30:40 to 23:30:40 | 0 | 0 | |
| 2 | calibrate | the ruler (triad 3727.6 GB/s) | 23:30:42 to 23:31:07 | 0 | 0 | |
| 3 | regcheck | compile only: every stamps variant's CTAs/SM against the plain kernel's | 23:31:09 to 23:31:36 | 0 | 1 | FAIL on 9 of 11 variants (5 CTAs/SM against the plain 4); PASS on std2's v2 and v3 |
| 4 | timed | mixtral-8x7b ska (PT, PW, PW-hotfirst, PW-rand1, uniform, balanced; n 2, 8, 32) | 23:31:40 to 23:43:50 | 12 | 1 | stopped at the first slip: a cell read 1635 MHz (n 2, repeat 7 of 9); no report.json |
| 5 | timed | mixtral-8x7b skb (DW, uniform, balanced) | 23:43:54 to 23:53:23 | 9 | 1 | stopped at the first slip: 1680 MHz (n 2, repeat 9 of 9); no report.json |
| 6 | bytes | mixtral-8x7b skc, G = 8 counter page | 23:53:26 to 23:57:14 | 4 | 0 | the histogram byte page records no gates |
| 7 | timed | olmoe-1b-7b ska (n 2, 4, 16) | 23:57:18 to 00:14:47 | 17 | 0 | none |
| 8 | timed | olmoe-1b-7b skb | 00:14:50 to 00:25:29 | 11 | 0 | none |
| 9 | bytes | olmoe-1b-7b skc, G = 8 counter page | 00:25:33 to 00:27:45 | 2 | 0 | the histogram byte page records no gates |
| 10 | timed | qwen1.5-moe-a2.7b ska (n 1, 2, 8) | 00:27:49 to 00:45:08 | 17 | 0 | none |
| 11 | timed | qwen1.5-moe-a2.7b skb | 00:45:12 to 00:55:41 | 10 | 0 | G1_lock_thermal FAIL (R3 exit 3): drift on DW / native, balanced / native and balanced / shared at n 2; every cell 1710 MHz, no throttle |
| 12 | timed | olmoe-1b-7b sk15: SHARED at 15 copies (PT, PW, DW, uniform), NATIVE uniform, n 2, 4 | 00:55:45 to 01:01:53 | 6 | 0 | none |
| 13 | perturb | the gate of units 14 to 18 | 01:01:57 to 01:02:54 | 1 | 1 | FAIL on 11 of 11 variants |
| 14 to 18 | stamps | stf, stt8x22, stt, std4, std2 | 01:02:59 to 01:03:18 | 0 | 2 | REFUSED: no listed variant reads PASS on both regcheck and the gate; nothing launched |
| 19 | bytes | qwen2-57b-a14b-tp8 G = 64 k64s4 | 01:03:23 to 01:05:36 | 2 | 0 | V5 FAIL (PRIVATE w1 q_P 0.957 n, as on rental 3's qwen2-tp8 pages); page exit 3 |
| 20 | bytes | qwen2-57b-a14b-tp8 G = 64 k32s4 | 01:05:41 to 01:07:40 | 2 | 0 | V5 FAIL, as unit 19; page exit 3 |
| 21 | timed | olmoe-1b-7b bk64, G = 64 NATIVE, n 1..9, s4 | 01:07:45 to 01:13:13 | 5 | 0 | none |
| 22 | timed | olmoe-1b-7b bk128, BLOCK_K 128 s4 | 01:13:19 to 01:18:48 | 5 | 0 | none |

108 minutes from the prelude to the last unit against the plan's 152 minutes of estimates. No unit
was dropped and there is no automatic retake (a plan rental). The driver ends with exit 3 (units 3,
4, 5 and 13 exit 1, the five refusals exit 2). Every timed page that wrote a report reads its worst
cell at 1710.0 MHz.

**The two Mixtral pages.** `scripts/locked_r3.py` stops R3 at the first timed cell more than one
15 MHz step under the lock (a slip), by design, so neither Mixtral page wrote a `report.json`:
only `cells.csv` (186 and 146 cell rows, the first six and eight of nine repeats) and the lock
ledgers (`locked-r3/*-p2/` and `*-p2b/`). The registration's G4 would have excluded the one sub-1705
MHz cell and kept the page; with no page, its `missing` rule applies and every Mixtral timed cell is
NOT SCORED. The skb page also carries four 1695 MHz cells (inside the watchdog's step, under G4's
1705 cut).

**The perturbation gate (unit 13) and regcheck (unit 3).** regcheck read the stamped copy at 5
CTAs/SM against the plain kernel's 4 on the w2 config at BLOCK_K 64 s4 for every variant of stf,
stt8x22, stt and std4 (w2 registers 55 against the copy's 48; w1 48 against 44, 5 CTAs/SM on both;
the same in every variant, v3 included), and passed only std2 (s8, shared-memory bound at 2 CTAs/SM
on both).
The gate then failed every variant: median |plain / copy - 1| 2.2 to 6.6%, worst 2.7 to 14.7%
(`results/*-perturb-pt/gate.env`): stf 2.2 to 2.3% / 2.7 to 3.0%, stt8x22 6.0 to 6.6% / 11.8 to
12.4%, stt 2.2 to 2.3% / 14.3 to 14.7%, std4 2.3 to 2.4% / 3.3 to 3.6%, std2 2.8 to 3.2% / 5.9 to
6.6%. v3, every stamp site at 1 in 17, reads 2.2 to 2.8% median. Units 14 to 18 were refused in 0
minutes, as designed: their pages were never written.

## What it answered

Each verdict is the registered reading (rental 2's rule 3 over ALL and CLEAN). The only page a view
drops is Qwen1.5 skb (G1), counted in ALL and excluded from CLEAN; the qwen2-tp8 byte pages are
excluded from CLEAN on V5 for secondk, which gates on V5.

**skew** (`results/gaps-*-{olmoe-1b-7b,qwen1.5-moe-a2.7b}-sk{a,b}/`, `results/*-skc-r3-counters/`):
G3, the token-order control, is INCONCLUSIVE: ALL (4 pages, 24 cells) is EQUIVALENT, 90% cluster-t
CI [-0.274, +0.015]% inside +-0.36%; CLEAN (3 pages, 18 cells) is not, CI [-0.373, +0.044]%, which
voids every histogram cell in CLEAN. So SKEW-RATIO and E1-SKEW are NOT SCORED (CLEAN has no data);
in ALL, S HOLDS (36 ratios, rms 1.05%, worst -2.09%, mean -0.85%), U and PW are EXCLUDED (rms 12.95%
and 17.81% on their lever cells), LT is UNDECIDED (5 lever cells), and E1-SKEW HOLDS (72 cells, rms
1.26%, worst +4.38%, mean +0.40%). B-SKEW and E2-SKEW are NOT SCORED in both views: the registered
G4 cut (each byte cell's sm__cycles_elapsed.avg / gpu_time_ns at 1705 MHz or above) drops every
cell (1656 to 1698 MHz under the held lock). E1-UNI and E1-UNI-EXT HOLD (balanced cells, rms 1.80%
and 0.26%).

**c15** (`results/gaps-*-olmoe-1b-7b-sk15/`): D, d 0.995 ns, HOLDS (median Delta 25.39 us against
23.69, z +1.32); M (1.333 ns) EXCLUDED (z -4.93); FIXED EXCLUDED; the skewed and uniform increments
agree (-0.46 us, SKEW-DEAD not shown).

**secondk** (`results/*-qwen2-57b-a14b-tp8-k{64,32}s4-r3-counters/`): NOT SCORED (CLEAN has no
data: V5 FAIL on both pages); ALL reads UNDECIDED, w1 H_ITER (+1.5%), w2 NEITHER (H_ITER -5.0%,
H_CTA -32.8%).

**bk128** (`results/gaps-*-olmoe-1b-7b-bk{64,128}/`): H_C HOLDS (q 0.9946 to 0.9958, rms 0.13%);
SYNC EXCLUDED (+71 to +75% on 6 of 6).

**stamps2**: every unit NOT SCORED: NOT RUN (no variant passes regcheck and the gate).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and consoles;
  `gh200-driver/instr-variants.json`, `instr-regcheck.env` and `instr-gate.env`: the variants and
  their regcheck and gate verdicts
- `results/gaps-nvidia_gh200_480gb-<model>-sk{a,b}/private_weight_reference/`: the timed histogram
  pages (units 4, 5, 7, 8, 10, 11; Mixtral's carry `cells.csv` only), each with
  `DECLARED_COPIES.txt`; `-sk15/` the 15-copy page (NOT JOINABLE TO COUNTER BYTES); `locked-r3/`
  their lock ledgers
- `results/<date>-nvidia_gh200_480gb-<model>-skc-r3-counters/lock1710/`: the byte-leg counter pages
  (`r3c-g8.json`, units 6 and 9) with `summary.json`, profiles and Triton caches
- `results/2026-10-09-nvidia_gh200_480gb-qwen2-57b-a14b-tp8-k{64,32}s4-r3-counters/lock1710/`: the
  second-K pages (`r3c-g64.json`, units 19 and 20)
- `results/gaps-nvidia_gh200_480gb-olmoe-1b-7b-bk{64,128}/`: the BK 128 timed pair (units 21, 22)
- `results/2026-10-08-nvidia_gh200_480gb-regcheck-rc/`: `regcheck.json` (registers, shared memory and
  CTAs/SM per variant and config) and `regcheck.env`
- `results/2026-10-09-nvidia_gh200_480gb-perturb-pt/`: `perturb.json`, `gate.env`, the compared SASS
  and the unit's own Triton cache
- `census*.json`, `census*.profiles/`: the censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console

**Histograms.** Every histogram page reads one of the registered SYNTHETIC files under
`docs/registered/2026-10-07-rental5-skew-hist/` (sha256 on every page, G5 PASS), fitted to public
routing statistics and published with the registration; no trace-derived count is on the VM or in
this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session
does; prose here names the board by `r3_timing_model.board()`. The Triton caches' compiled
`cuda_utils*.so` files (18) are left out, as in rentals 1 to 4; every other file of the run branch
is here, byte for byte (1301 files against SHA256SUMS).
