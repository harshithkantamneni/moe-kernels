# Calibration rental C3: one NVIDIA GH200 480GB, 8 units of one calibration registration (docs/registered/README.md, 2026-10-26)

One rental ran `scripts/plans/rentalC3-cal-2026-10.plan` unattended through the calibration rental's driver
(`scripts/gh200_cal3_session.sh`, C2's pinned driver plus lines marked C3). It set the GPU power limit to 700 W
and read it back, probed the 1410 MHz session lock, measured the sag estimator's two f128 pages on this board and
gated them with A2's G-BAND, checked the instrumented copy compile-only (regcheck), timed the plain kernel against
the copy on every stamped cell (the perturbation gate), then took per-CTA start and end stamps (the copy at its
"ends" level, `moe/instrumented/fused_moe_instr.py` unchanged) on Mixtral-8x7B (C3-S7) and Qwen2-57B (C3-SQ) at
vLLM's 128x128x64 w8 s3 G 1 tile on A2's E1-CAL cells, NATIVE. The registration (`2026-10-26-cal3-gh200`), its
priced table, the reader, the probe, the driver and the plan were committed before any page (e10fda1b, which is
also the run commit).

C3 is a CALIBRATION rental and DESCRIPTIVE: it certifies nothing, scores no verdict and moves no registered
verdict, and every C3 page is SEEN once read. The perturbation gate FAILED on its timing legs alone, so under the
owner's OD-1 every stamp page is labelled PERTURBED and every reading below is a PERTURBED reading. Every reading
is the registered reader's (`scripts/scoring/cal3/read_c3.py`, pinned by the registration), run on this directory
with no post-page fix; its output is `scripts/scoring/cal3/cal3.read.{json,txt}`, and the notes are in
`docs/registered/README.md` ("**Read 2026-10-27**") and `docs/FINDINGS.md` (calibration rental C3).

## Where it came from

Lambda Cloud instance f3128422 (one GH200, us-east-3), launched 04:10:28Z and terminated 04:26:34Z on
2026-10-27: about 16 min, about $0.62 at the registration's $2.29/h (an estimate; the instance, the region and
the two times are the rental's records, not in the run files, and the bill is in neither). The VM's setup ran
04:12:16Z to 04:14:02Z (7 preflight checks passed) and the driver 04:14:03Z to 04:25:52Z. Board `02e86e`
(`PREFLIGHT.txt`), the board of rental 7 (2026-10-14); no other published session names it.
`scripts/gh200_cal3_session.sh --plan scripts/plans/rentalC3-cal-2026-10.plan` ran at commit e10fda1b
(`commit.txt`); the plan's sha256 51faa798... is in `gh200-driver/status` and equals the registration's. This
directory is the branch `run-gh200-tile3-20261027t0410`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was
set.

## The power limit and the lock

The registration asks for 700 W with no fallback and stops the session unless two read-backs agree:
`--query-gpu=enforced.power.limit` and the GPU scope's Current Power Limit in `nvidia-smi -q -d POWER`. Before
the limit was set (`power-at-start.txt`) the GPU scope read Current and Requested 700 W. After `-pl 700 -sc 0`
the first poll (0 s) read enforced.power.limit 700 W, the -q GPU scope Current 700 W (Requested 700 W) and
power.limit 700 W (`power-set.txt`, `gh200-driver/c2-power.env` `C2_POWER_W=700`); the module scope read 1000 W.
No `power-set-unsettled-700.txt` was written. Every unit ran at 700 W: the driver's `clocks:` line after every
unit reads power.limit 700 W, and the 4 power.limit records on the 2 counter pages read 700.00 W.

The prelude locked 1410 MHz, read clocks.sm back and recorded `LOCK_PROBE_1410=held` (`gh200-driver/locks.env`).
G-BAND (`gh200-driver/gate-band-1410.json`) then PASSED on the registered fallback band [1352.19, 1410] MHz: in
the primary band [1374.75, 1410] the joint fit was None (0 of 27 OLMoE cells, so OLMoE NOT SCORED, and 4 of 27
Mixtral-8x7B cells in band); in the fallback 18 of 27 and 16 of 27. A2 and C2 also passed on the fallback. The
perturbation unit and both stamp units record 1410 MHz on `nvidia-smi` before and after (`perturb.json`,
`stamps.json` "smi").

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | registered as | start to end (UTC) | min | exit | page exit |
|---|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W read back; lock 1410 held | C3-P | 04:14:03 to 04:14:06 | 0 | 0 | |
| 2 | bytes | olmoe-1b-7b f128 (G 64, treads 1 to 9) | C3-F | 04:14:07 to 04:17:18 | 3 | 0 | 3 |
| 3 | bytes | mixtral-8x7b f128 (G 64, treads 1 to 9) | C3-F | 04:17:20 to 04:24:16 | 7 | 0 | 3 |
| 4 | gate | gband at 1410 | C3-F | 04:24:18 to 04:24:20 | 0 | 0 | PASS (fallback band) |
| 5 | regcheck | rc3, compile only | C3-G | 04:24:22 to 04:24:38 | 0 | 0 | `REGCHECK_u7_st7=PASS`, `REGCHECK_u8_stq=PASS` |
| 6 | perturb | pt3, plain against the copy at 1410 | C3-G | 04:24:40 to 04:25:11 | 1 | 1 | timing FAIL, hard legs PASS (both variants) |
| 7 | stamps | mixtral-8x7b st7 (4 cells, n 2, w1 and w2) | C3-S7 | 04:25:14 to 04:25:28 | 0 | 0 | PERTURBED, 24 launches |
| 8 | stamps | qwen2-57b-a14b stq (8 cells, n 1 and 2) | C3-SQ | 04:25:31 to 04:25:47 | 0 | 0 | PERTURBED, 48 launches |

8 of 8 units ran in about 12 min of units against 30 estimated; none was dropped; the driver ends with exit 0.
The perturb unit's exit 1 is its gate: `GATE_u7_st7=FAIL median=0.0203 worst=0.0242`,
`GATE_u8_stq=FAIL median=0.0146 worst=0.0301` (median and worst |plain / copy - 1| over the variant's (cell,
GEMM) ratios, against 1% and 2%), with `HARD_u7_st7=PASS` and `HARD_u8_stq=PASS` (CTAs per SM identical, the
all-off copy's SASS equal to the plain kernel's, the upstream file, C3's SASS legs) in `perturb-pt3/gate.env`.
By owner OD-1 the stamps units ran and labelled their pages PERTURBED (`stamps.json` "page_label"). Each counter
page's exit 3 is the route's own page gates on the two G 64 f128 pages (OLMoE V4, V5, C1 and C6 FAIL, C2
REFUSE; Mixtral-8x7B V5, V7, C2 and C6 FAIL, C1 REFUSE), the same gates C2's f128 pages failed; no C3 reading
uses a page verdict. Each byte unit's `analyse` step exits 3 as well (`summary.json`).

The copy compiled at 113 and 114 registers against the plain kernel's 116, 98,304 B shared and 2 CTAs per SM on
both (regcheck); every stamped config is on the wgmma path, with a clock read before the first global load and
none inside the HGMMA loop (`stamps.json` "sass").

## What the reader reads (`scripts/scoring/cal3/cal3.read.txt`)

Every reading is on PERTURBED pages; the w2 overhead beside C3-R1 and C3-R2 is the median over Mixtral-8x7B's
four w2 cells of |median copy / median plain - 1|, 1.38%, and the worst burst pair 2.97%, inside the NOT-READ
bound (10% median, 20% worst).

| reading | outcome | numbers |
|---|---|---|
| board | valid | C2_POWER_W=700, LOCK_PROBE_1410=held, G-BAND PASS (fallback); the commit check passed on 33 files at e10fda1b |
| C3-R4 validity | valid | Mixtral-8x7B and Qwen2-57B: every launch passes; gate hard legs and regcheck PASS |
| C3-R1 | OUT, MEASURED-PERTURBED | Mixtral-8x7B w2 theta_L 0.3513 over 12 launches (theta (1 + o) 0.3562); per cell PT 0.262, PW 0.350, balanced 0.353, uniform 0.353; registered IN [0.25, 0.30], NEAR [0.20, 0.35] |
| C3-R2 | LAUNCH-TOGETHER | 0.901 of 1872 later-wave, schedule-same-wave re-read pairs launch within 0.5 of M4's staggered gap; all pairs 0.877 of 3840; schedule same-wave share 0.925 |
| C3-R3 | NOT REPRODUCED, DESCRIPTIVE ONLY | Mixtral-8x7B w1 e_plain +0.94% under K1 (N0 +1.70%), under the registered +2%; LIFE +0.034, PACKING -0.030, total +0.003 (overhead-corrected), life share not read (total under 0.005); w1 overhead median 2.30% |
| descriptive theta | | Mixtral-8x7B w1 0.494; Qwen2-57B w1 0.742, w2 0.477 |
| descriptive e_plain under K1 | | Mixtral-8x7B w2 -5.27%; Qwen2-57B w1 -4.59%, w2 -4.29% |
| the instrument's cost | | stamped over plain per cell, mean: Mixtral-8x7B w1 +2.88%, w2 +2.49%; Qwen2-57B w1 +3.05%, w2 +1.44% |

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/c2-power.env`: the power outcome; `gh200-driver/locks.env`: the lock probe;
  `gh200-driver/gate-band-1410.json`: G-BAND; `gh200-driver/instr-gate.env`, `gh200-driver/instr-regcheck.env`,
  `gh200-driver/instr-variants.json`: the copy's gate, its regcheck and the two stamp variants
- `power-at-start.txt`, `power-set.txt`: the -q -d POWER snapshots before `-pl` and after the 700 W read-back
- `results/2026-10-27-nvidia_gh200_480gb-<model>-f128-r3-counters/lock1410/`: the two counter pages (C3-F f128
  on OLMoE and Mixtral-8x7B), each `r3c-g64.json` with `summary.json`, profiles and Triton caches
- `results/2026-10-27-nvidia_gh200_480gb-regcheck-rc3/`: regcheck (`regcheck.env`, `regcheck.json`, SASS)
- `results/2026-10-27-nvidia_gh200_480gb-perturb-pt3/`: the perturbation gate (`perturb.json`, `gate.env`, SASS);
  its plain medians are C3's uninstrumented reference
- `results/2026-10-27-nvidia_gh200_480gb-instr-<model>-<label>/`: the two stamp pages, `stamps.json` with
  `stamps/<cell>-native-call<k>-<gemm>.npy` (the int64 rows) and `-experts.npy` (the expert_ids vLLM handed the
  launch), SASS and Triton caches
- `census*.json`, `census*.profiles/`: the per-model GEMM censuses written before the first page;
  `unit-dir-*.txt`, `counters-dir-*.txt`: each unit's output directory
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console; `logs/`
  holds every per-unit log, named by model and label

**Histograms.** The Mixtral-8x7B stamp page reads `docs/registered/2026-10-14-tileA-hist/mixtral-8x7b-128g1.json`
and the Qwen2-57B page `docs/registered/2026-10-14-tileA-hist/qwen2-57b-a14b-128g1.json`, each with its sha256 on
the page and equal to the registration's. No trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session does (the
files are the run branch's, byte for byte); prose here names the board by its hash. The Triton caches'
compiled `cuda_utils*.so` files (8) are left out, as in every earlier session. Every other file of the run
branch is here, byte for byte (548 files against SHA256SUMS, which lists 556).
