# H200 session 6: the alpha(G) chain at G=2, and a same-card duty sweep

Pod w226zpjmj8p1d3, restarted on its pinned host: the card of session 5,
GPU-0ffa33b8-eeac-8768-2331-dc7e2f8d6490 (the `DEVICE` file of both chain
sessions here, and of session 5's). Container hostname `1cdbd6a71870` (session
5's was `421a523e82cb`). 1x H200, CUDA 13.0 host, driver 580.159.04, 700 W,
1980 MHz maximum SM clock. Up 2026-09-24 04:16 to 05:24 UTC (68 min, about $5.20
at the $4.59/h the dry run prices at). Tree c09b353 (r3-align); venvs torch
2.13.0 with Triton 3.7.1 (base) and vLLM 0.27.1 (vllm). Driven by
`scripts/alpha_g_chain.sh` with `G_LADDER=2 SEEDS=0`, then by `duty_sweep_s6.sh`
by hand.

## What is here

- `alpha_g-nvidia_h200-20260924T042015Z/`: the first chain session, whole. It
  was REFUSED at the preconditions (below). `CHAIN.tsv` is its ledger and
  `ARMS.tsv` the driver's; `PAIRS.tsv` has no row. `logs/` holds thermal and
  calibrate only. `chain.lock` is kept: it is not empty, it holds
  `1636 1cdbd6a71870` (a PID and the pod's hostname).
- `alpha_g-nvidia_h200-20260924T042526Z/`: the second chain session, whole, laid
  out as session 5's: `CHAIN.tsv`, `ARMS.tsv`, `DEVICE`, `R3_DUTY`, the PAIRS
  tables and `PAIRS-README.txt`, `chain-logs/` (one log per step, plus each
  arm's `.price.log`), `logs/` (thermal, calibrate, pin_probe-n64-g1, and
  `counter_plan.log`, a printed DRAM counter plan: `chain-logs/preconditions.log`
  reads `SKIP counter_plan` and `ARMS.tsv` has no row for it) and
  `pin-probe-n64-g1/`. Beside session 5's layout it carries the counter probe's
  page `COUNTERS`, its payload `COUNTERS.json` and its log
  `chain-logs/counter-probe.log`. The duty sweep wrote into it after the chain:
  `DUTY-SWEEP.tsv` (each run's start, end and exit code) and
  `chain-logs/duty-sweep-g{1,16}-d{0.5,1.0}.log`. `chain.lock` holds
  `2778 1cdbd6a71870`.
- `alpha_g_chain_s6.out`: the chain's console, both sessions appended in order.
- `alpha_g_dry_s6.out`: the console of the dry run that priced the chain. Its own
  session directory (`alpha_g-nvidia_h200-20260924T041829Z`) was not exfiltrated.
- `duty_sweep_s6.sh`: the script the duty sweep ran, as it was on the pod.
- `../results/gaps-nvidia_h200/`: all 6 run directories in the tarball, 5 in
  `private_weight_reference/` (R3: the chain's r3-g2-s0 and the four sweep
  pages) and 1 in `clock_elasticity/` (R1: r1-g2). thermal's run directory under
  the pod's `thermal_acceptance/` is not in the tarball.
- `../calibration/`: the ruler the second session's calibrate measured and its
  run directory; NOT adopted, see its README.

Paths inside the ledgers and the consoles are the pod's: `/workspace/session/<dir>`
is `./<dir>` here, `/workspace/results/gaps-nvidia_h200` is
`../results/gaps-nvidia_h200`, and `/workspace/moe-kernels/results/calibration/<dir>`
is `../calibration/<dir>`. Times below are UTC, from the ledgers, from
`DUTY-SWEEP.tsv` and from the file mtimes the tarball carries.

## What ran, in order

1. **Dry run** (`alpha_g_dry_s6.out`, 04:18 to 04:20): every arm's own
   `--dry-run`. Ladder G in {2}, seeds {0}, R3 at duty 0.25, elasticity states
   1.0 0.5 0.25; priced at 2787 s = 47 min, about $3.55 at $4.59/h, "Book 2 h".
2. **Session `042015Z`** (console header `NEW`), 04:20 to 04:23:
   - preflight-r1, preflight-r3: DONE.
   - preconditions: REFUSED (rc 2, 198 s). The work tree had 0 dirty files
     before anything ran.
     - thermal DONE (155 s): 1455 MHz median (min 1440, max 1485) over 55
       samples, first and last third 1455 MHz, 0.0% apart, board power median
       691 W of a 700 W limit.
     - calibrate INVALID (exit 3, 29 s) on `VALIDITY clock_established FAIL`,
       "the samples disagree with each other or with the settle plateau": bf16
       GEMM clock samples 1455, 1455, 1485, 1545, 1455 MHz (median 1455, spread
       5.8%), compute settle reached at 1455 MHz. Its other five RESULT lines
       read PASS. It printed `PUBLISHED to
       /workspace/moe-kernels/moe/bench/hardware/measured_nvidia_h200.yaml`, and
       the work tree went from 0 to 1 dirty file.
     - The driver stopped: "REFUSED: arm 0 did not stand behind a ruler for
       nvidia_h200." pin_probe-n64-g1 did not run; no counter probe, no GPU
       tests, no R1 or R3 step ran.
   - calibrate's run directory (`...-2ea5b191`) was overwritten at 04:29:01 by
     the second session's calibrate, which carries the same run id. This
     calibrate's numbers survive in `logs/calibrate.log` only; see
     `../calibration/README.md`.
3. **Session `042526Z`** (console header `NEW`, a second session beside the
   first), 04:25 to 05:02:
   - preflight-r1, preflight-r3: DONE.
   - preconditions: DONE (231 s). The work tree had 1 dirty file before anything
     ran; the `dirty` column of every `CHAIN.tsv` row reads 1.
     - thermal DONE (155 s): 1455 MHz median (min 1425, max 1485) over 55
       samples, first and last third 1455 MHz, 0.0% apart, 691 W.
     - calibrate DONE (28 s): `clock_established PASS`, bf16 GEMM clock samples
       1440, 1455, 1440, 1470, 1455 MHz (median 1455, spread 2.0%); triad 4377.0
       GB/s, ridge 148.5.
     - pin_probe-n64-g1 DONE (30 s): F1 and F2 PASS.
   - counter-probe: INFO (rc 1, 10 s; informational, it gates nothing). BLOCKED:
     `ERR_NVGPUCTRPERM`. ncu is 2025.1.1.0 at `/usr/local/cuda/bin/ncu`, NOT on
     PATH (found under `/usr/local/cuda*/bin/`; the probe ran with
     `/usr/local/cuda/bin` first on PATH). The probe kernel launched and the
     counter read was refused. The effective capability mask read
     `00000000a80425fb`: CAP_SYS_ADMIN (bit 21) and CAP_PERFMON (bit 38) clear.
     The host module flag was not read ("the parameter is not listed by this
     driver"). No nsys on PATH.
   - gpu-tests (`tests/test_gpu.py` on the card): 37 passed, 1 skipped, 2
     warnings in 42.09 s.
   - probe-check: P1 PASS, the alignment probe under a CUDA graph at 5.29 us per
     call against an eager p50 of 24.19 us.
   - r3-g2-s0 (`ff6f9ed1`, 04:31 to 04:43, 727 s): CLAIM_FAIL.
   - r1-g2 (`305890de`, 04:43 to 05:02, 1164 s): INVALID on V5.
   - suite: SKIPPED, `END_SUITE=skip`.
4. **Duty sweep, by hand** (`duty_sweep_s6.sh`, 05:03:32 to 05:22:35). The
   script's header reads "Session 6 same-card duty sweep (owner-approved plan,
   2026-09-24): R3 at G=1 and G=16, duty 0.5 and 1.0, seed 0" and "Duty 1.0
   pages are controls (V7 expected to fail at full duty); they are not claims."
   Each run is `scripts/private_weight_reference.py --model mixtral-8x7b
   --block-m 32 --treads 6 --repeats 9 --group-m G --duty D --seed 0
   --session-tag alpha_g-nvidia_h200-20260924T042526Z` from the vllm venv under
   `timeout 1800`, in the order duty 0.5 (G=1, G=16), then duty 1.0 (G=1,
   G=16). All four exited 3 (INVALID).

## The readings

R3 (`private_weight_reference`), mixtral-8x7b, BLOCK_M=32, seed 0. The ratio is
slope(shared) / slope(private) over treads 2 to 6, with its 90% percentile
bootstrap interval; C1's band is ALPHA_BAND [0.529, 0.588). C2 PASSES on every
page. An INVALID page's exit line reads "nothing quotable"; its ratio is printed
here as its page prints it.

| G | duty | directory | UTC | exit | gates that FAIL | ratio [90%] | C1 world |
|---|---|---|---|---|---|---|---|
| 2 | 0.25 | `ff6f9ed1` | 04:31 to 04:43 | CLAIM_FAIL | C1 (V0 to V8 PASS) | 0.7516 [0.7507, 0.7528] | ABOVE-THE-REFIT-BAND |
| 1 | 0.5 | `d1ce0c83` | 05:03 to 05:09 | INVALID | V0 (drift 32.7%), V7 (worst 1.56%, tread 4), C1 | 0.9265 [0.9258, 0.9270] | NO-REUSE |
| 16 | 0.5 | `1659b013` | 05:09 to 05:15 | INVALID | V7 (worst 3.09%, treads 4 5 6), C1 | 0.7004 [0.6979, 0.7051] | ABOVE-THE-REFIT-BAND |
| 1 | 1.0 | `5a721df8` | 05:15 to 05:19 | INVALID | V7 (worst 2.06%, treads 2 to 6), C1 | 0.9794 [0.9734, 0.9814] | NO-REUSE |
| 16 | 1.0 | `a61d99f5` | 05:19 to 05:22 | INVALID | V0 (drift 32.7%), V7 (worst 17.75%, treads 2 to 6), C1 | 0.7515 [0.7470, 0.7535] | ABOVE-THE-REFIT-BAND |

V0's gate includes DRIFT under 20% of the timed cells at every tread; V7's is
shared and private under-load clocks within 1% at every tread. The two duty 1.0
directories carry no `duty` in their names. The other drift figures V0 printed:
16.7% (G=2), 14.8% (G=16, duty 0.5), 13.6% (G=1, duty 1.0). V7 at G=2 read
0.39%.

Session 5 on this card, R3 at duty 0.25, 3 seeds per G, all VALID (its
`PAIRS-by-G.tsv`): G=1 mean 0.9150, G=16 mean 0.6803.

R1 (`clock_elasticity`), r1-g2, `305890de`, states 1.0 0.5 0.25, 8 treads x 13
repeats: INVALID on V5 alone, 1 of 265 kept rows moved more than 0.05 from its
burst's first quarter to its last. eta of the per-M-tile cost over treads 2 and
deeper 1.1014 [1.0678, 1.1497] (95%). V1 PASS (narrowest tread spans 1.1864x
against 1.060x); V4 PASS (47 of 312 rows excluded, 15.1%, all drift); V7 PASS
(the interval intersects the admissible [-0.075, 1.075]); C1 PASS
(CLOCK-CARRIES); C2 FAIL. `PAIRS.tsv` reads its band `withheld:INVALID` and G=2
as `unresolved`.

`PAIRS-by-G.tsv`, G=2: the bytes-rate bound puts alpha <= 0.8477 at the
read_stream 4613.2 GB/s and <= 0.8934 at the pin rate 4814.3 GB/s (the shared
arm's tread 6 at 3.2007 ms over a 2.8186 GB expert set).

## Ruler and tree state

Every one of the 6 `report.json` files carries `git_sha` c09b353,
`git_dirty: true` and `git_dirty_files: 1`, and so does `COUNTERS.json`. Every R3
report cites the second session's calibrate: ridge 148.49, bandwidth 4376.99
GB/s, roof 649.92 TFLOP/s, reference clock 1455 MHz. The R1 report records no
ridge and no bandwidth.

## Provenance of these files

The exfil tarball `exfil-session6.tar.gz`, sha256
`b8769ac6a161153d0207b8c0a340a70934d1b99c057a56bc69d45d2fa9eedab1`, holds 179
files. Every one was copied here and compared by sha256, tarball against
destination: 179 identical, 0 different. The tarball's top level maps as
`alpha_g-*`, `alpha_g_*_s6.out` and `duty_sweep_s6.sh` to `./`,
`gaps-nvidia_h200/` to `../results/gaps-nvidia_h200/`,
`moe/bench/hardware/measured_nvidia_h200.yaml` to
`../calibration/measured_nvidia_h200.yaml`, and `results/calibration/<dir>` to
`../calibration/<dir>`. 6 of the 179 are not in git: Triton's
`cuda_utils.cpython-312-x86_64-linux-gnu.so`, one per run directory under
`triton-cache/MIH4X24CEAJDWYGXCZ4EKW6DTWSUG2ZBSBPV7NGI62HAUKJ7F7BQ/`, which the
repository's `.gitignore` drops by its `*.so` rule (its `results/published/`
block re-includes `.ptx`, `.cubin`, `.ncu-rep`, `.nsys-rep` and `.qdrep`, not
`.so`). The other 173 are tracked, beside this README, `../calibration/README.md`
and `../KIND`.
