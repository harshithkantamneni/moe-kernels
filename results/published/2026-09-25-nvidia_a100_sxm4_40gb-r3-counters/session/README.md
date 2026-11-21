# One of the study's four cards, the non-Hopper control: R3 arms under DRAM counters on one NVIDIA A100-SXM4-40GB (Lambda Cloud); every measurement here is this card's, never averaged with another card's

Every number in this directory is the A100-SXM4-40GB's. None of it is an H200
number, and no H200 number is set beside it here.

Lambda Cloud instance accf77cf075842ae95701a12a5d58b68, us-west-2, $1.99/h,
1x NVIDIA A100-SXM4-40GB, GPU-5b67366f-3d8c-12eb-32df-400838f594e1, sm_80, 108
SMs, 40 MiB L2, 400 W power limit. Launched at about 00:05 UTC on 2026-09-25
(the evening of 2026-09-24 in US Central time) and terminated at 02:22 UTC
(about 2 h 17 min; at $1.99/h, about $4.54). The instance id, the region, the
price and the two instance times are from the owner's session; no file here
records them. The card's name, UUID, capability, SM count and L2 are from
`PREFLIGHT.json`; the power limit from `setup_vm.log`.

Host: Ubuntu 22.04, 216 GB RAM, Lambda stack 0.1.17 (its system torch 2.7.0 is
recorded and never used). Venvs torch 2.13.0+cu130 and triton 3.7.1 (base),
plus vLLM 0.27.1 (vllm), Python 3.12.14, CUDA 13.0. Nsight Compute 2025.3.1.0
(build 36398880) at `/usr/local/cuda-13.0/bin/ncu`, installed by `setup_vm.sh`
S5 as `cuda-nsight-compute-13-0`. Counter door `sudo`: uid 1000 had CapEff 0,
the probe ran through `sudo` with CapEff 0x1ffffffffff (CAP_SYS_ADMIN and
CAP_PERFMON).

Driver: the instance shipped 570.148.08. By owner decision it was upgraded to
`nvidia-driver-580-server-open` 580.105.08 from Lambda's repository and
rebooted, before `setup_vm.sh` first ran. The upgrade is not logged here: every
S0 DETECT block in `setup_vm.log` and `preflight2.log` already reads
`580.105.08 (CUDA 13.0 per nvidia-smi)`, and the S5 apt output in
`setup_vm.log` lists `nvidia-firmware-570-server-570.148.08` as "automatically
installed and is no longer required".

## What is here

Paths inside the logs are the VM's: `/home/ubuntu/moe/session/<x>` is `./<x>`
here, `/home/ubuntu/moe/results/<x>` is `../results/<x>`, and
`/home/ubuntu/moe/repo` was the checkout.

- `../KIND`: `session`. See "Why KIND is session" below.
- `../results/2026-09-25-nvidia_a100_sxm4_40gb-r3-counters/`: the run
  directory `run_counters.sh` wrote. `r3c-g{1,2,4,16,64}.json` is one page per
  G; `r3c-g<G>.profiles/` holds that G's `g<G>.ncu-rep`, the raw `g<G>.csv`,
  `g<G>.capture.json` (ncu's argv, version and return code), `g<G>.plan.json`,
  `g<G>.manifest.json`, `g<G>.ncu.log` and the Triton cache; `summary.json`
  is the `--analyse` output over the five pages.
- The VM's session root, `/home/ubuntu/moe/session`, whole:
  - `PREFLIGHT.txt`, `PREFLIGHT.json`: the second preflight (00:37:21Z, 60db8ba,
    READY). The first preflight wrote the same two paths and was overwritten;
    its text survives in `setup_vm.log`.
  - `probe.json` and `logs/pf5-probe.log`: PF5's route probe.
  - `census.json`, `census.profiles/` and `logs/pf6-census.log`: PF6's launch
    census.
  - `ncu.txt` (the ncu binary, its version and `--list-chips`),
    `ncu-query-metrics.txt`, `ncu_raw_sample.csv`.
  - `logs/setup_runpod.log` (the venv build `setup_vm.sh` S4 ran),
    `logs/r3c-g<G>.log` (one per G), `logs/analyse.log`.
- The tarball's top-level files:
  - `setup_vm.log`: the first `setup_vm.sh` run, at 3dfc3b5, ending NOT READY.
  - `preflight2.log`: the second run, at 60db8ba, ending READY.
  - `run_counters.sh`: the driver of the counter run.
  - `run_counters.status`: its ledger, one start and one end line per G.
  - `run_counters.out`: its console, 0 bytes (every step redirects its own
    output into `logs/`).

`ncu_raw_sample.csv` is byte-identical (sha256 1b2f1bd6...) to
`tests/fixtures/ncu/raw_wide_a100_ncu2025.3.1.csv`, which 60db8ba committed.

There is no `calibration/` directory. `calibrate` did not run on this box and
the tarball holds no measured yaml; `setup_vm.log` prints `roofline profile NO
MEASURED PROFILE, and no datasheet entry either`. Every page and `summary.json`
record `ridge_source` and `bandwidth_source` as null, and each `logs/r3c-g<G>.log`
prints "none uses a bandwidth, a ridge, an intercept or a calibration" under
its alpha line. No ruler is adopted by this publication and none is changed.

## What ran, in order

Times are UTC on 2026-09-25, from the logs, the ledger and the pages.

1. **`setup_vm.sh` at 3dfc3b5** (`setup_vm.log`). S1 checked the repository out
   of a git bundle at 3dfc3b5; S4 built the base and vllm venvs from
   `requirements/resolved-*.txt`; S5 installed ncu; S6 chose the `sudo` counter
   door; S7 wrote `env.sh`. S8, the preflight, at 00:32:51Z:
   - PF1 to PF4 PASS; PF7 PASS.
   - PF5 FAIL: "verdict REFUSE, exit 2: the probe kernel ran and ncu returned no
     readable dram__bytes_read.sum: gpu__time_duration.sum came back in unit
     'ns', which this parser has never been shown".
   - PF6 SKIP: "PF5 did not read a counter, so a census under ncu would test
     nothing".
   - verdict NOT READY (exit 1).
2. **60db8ba, on the laptop** (authored and committed 00:36:04 UTC). The
   parser reads the duration unit as ncu 2025.3.1 prints it, `ns`, and this
   box's raw page is committed as a test fixture.
3. **The checkout at 60db8ba** (`preflight2.log`: `setup_vm.sh` stage 1 from
   `/home/ubuntu/moe/repo/scripts/setup_vm.sh`, commit 60db8ba). S0, then S8 at
   00:37:21Z: PF1 to PF7 PASS, verdict READY (exit 0).
   - PF5: route OPEN; all 5 STRICT metrics of the r3-arms family read back as
     numbers on a profiled launch of the probe kernel (dram__bytes_read.sum
     4.19904e+06); 11 of 11 cross-check and recorded metrics proven; the CSV
     was WIDE.
   - PF6: CEN1 to CEN3 PASS; one `fused_experts` call makes exactly 2
     `fused_moe_kernel` launches (8 over 4 calls), every grid is the grid the
     child derived, and the child ran on this card.
   - PF7: "the save spills to the host; 228 GB available >= 1.5 x 25.8 GB".
4. **`run_counters.sh`**: `scripts/dram_counter_route.py --run --family r3-arms
   --group-m G --census /home/ubuntu/moe/session/census.json` (`./census.json`
   here), one G at a time, in the order 64, 1, 4, 2, 16, then `--analyse`
   over the five pages. Every page names commit 60db8ba with `git_dirty:
   false`, every `capture.json` names 60db8ba, and every ncu return code is 0.
   From `run_counters.status`:

   | G | start | end | wall | exit |
   |---|---|---|---|---|
   | 64 | 00:37:33 | 00:58:14 | 20 min 41 s | 3 |
   | 1 | 00:58:14 | 01:19:00 | 20 min 46 s | 3 |
   | 4 | 01:19:00 | 01:39:42 | 20 min 42 s | 1 |
   | 2 | 01:39:42 | 02:00:25 | 20 min 43 s | 1 |
   | 16 | 02:00:25 | 02:21:06 | 20 min 41 s | 3 |
   | `--analyse` | | 02:21:08 | | 3 |

   Every `g<G>.ncu.log` carries, on its third line, inside the first profiled
   launch,
   `==WARNING== Backing up device memory in system memory. Kernel replay might
   be slow. Consider using "--replay-mode application" to avoid memory
   save-and-restore.`; `census.ncu.log` does not.

### How each page was captured

From `g<G>.capture.json` and the page: ncu's argv includes `--replay-mode kernel
--cache-control all --clock-control base --nvtx -k regex:^fused_moe_kernel$
--launch-skip 60 --launch-count 90` over 16 metrics, the child being
`scripts/private_weight_reference.py --counter-child g<G>.plan.json` in the
vllm venv. mixtral-8x7b, bf16, BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=8,
num_stages=4; arms native, shared and private at treads 1, 2, 3, 4 and 6 (15
cells); 3 measured calls per cell after 2 warmups; 90 profiled launches; 9
copies declared.

## The readings

The alpha(G) table as `logs/analyse.log` prints it (lines 1 to 10). These alphas
are THIS card's, measured by DRAM counters on the A100-SXM4-40GB; they are not
the study's H200 alphas.

```
CARD NVIDIA A100-SXM4-40GB (nvidia_a100_sxm4_40gb, UUID 5b67366f-3d8c-12eb-32df-400838f594e1, sm_80, 108 SMs, 40 MiB L2): every number here is THIS card's; the study's timing pages are nvidia_h200.
ALPHA(G) OVER 5 PAGES, one card, one commit, one vLLM, one design
  alpha lo/hi: the least and greatest OLS slope over SHARED's weight-only q in [q_S - e, q_S],
  e PRIVATE's excess over n; slope: the upper edge's, every activation re-read counted as a weight re-read
     G  alpha lo  alpha hi    slope       w1       w2    resid    ratio     diff  exit
     1    0.9724    0.9760   0.9752   1.0002   0.9251    7.58%   0.9734   0.9732  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
     2    0.4177    0.4244   0.4237   0.4222   0.4267   19.85%   0.4234   0.4184  1 CLAIM_FAIL: measured; VALIDITY passed; a CLAIM gate did not (a result, not a retry)
     4    0.1849    0.2025   0.2010   0.1812   0.2406   41.52%   0.2007   0.1865  1 CLAIM_FAIL: measured; VALIDITY passed; a CLAIM gate did not (a result, not a retry)
    16   -0.0241    0.0946   0.0840   0.0436   0.1649    9.12%   0.0793  -0.0135  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
    64   -0.1428    0.1455   0.1323   0.0062   0.3844    3.93%   0.1072  -0.1296  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
```

Three of the five pages are INVALID (G = 1, 16, 64), so `--analyse` exited 3.
The gates, as `logs/analyse.log` prints them. V0, V1, V2, V4, V5, V6, V8 and V9
PASS on every page.

| G | exit | VALIDITY gates that FAIL | CLAIM gates |
|---|---|---|---|
| 1 | 3 INVALID | V3: worst (max - min) / median 2.0422% against 1% (native/3, native/6); V7: n=2 w2 +1.5238% against 1.0896%, n=4 w2 -1.4324% against 1.0000% | C1 PASS, C2 PASS, C3 PASS, C6 PASS |
| 2 | 1 CLAIM_FAIL | none | C1 PASS, C2 FAIL, C6 PASS |
| 4 | 1 CLAIM_FAIL | none | C1 PASS, C2 FAIL, C6 PASS |
| 16 | 3 INVALID | V3: 1.0173% against 1% (native/2) | C1 UNKNOWN (its GATE line prints REFUSE), C2 FAIL, C6 FAIL |
| 64 | 3 INVALID | V3: 1.6431% against 1% (native/6, shared/2, shared/6); V7: n=2 w2 +3.7683% against 3.3258% | C1 UNKNOWN (its GATE line prints REFUSE), C2 FAIL, C6 FAIL |

The per-(G, arm, GEMM) q tables and each gate's measured values and reasons
follow the alpha table in `logs/analyse.log`; the same are in each
`logs/r3c-g<G>.log`, which also prints the activation working set of one column
pass per tread against this card's 40 MiB L2.

Each `logs/r3c-g<G>.log` and `logs/analyse.log` ends with `git check-ignore
exited 128; path UNVERIFIED`: the results root `/home/ubuntu/moe/results` is
outside the checkout `/home/ubuntu/moe/repo`.

## Why KIND is session

The repository's KIND vocabulary has one word, `session`
(`moe/bench/published.py`, `SESSION` and `KIND_MARKER`). No word names a
counter run, and none is added here. The docstring of `SESSION` describes a raw
session directory that holds the driver's ledger, its per-arm logs and the run
directories the arms wrote, kept so each verdict printed can be re-derived,
with no `measured.yaml` and no `merged.csv` of its own. This directory is that:
`run_counters.status` is the ledger, `logs/` the per-G logs, `../results/` the
run directory, and there is no `measured.yaml` or `merged.csv`.

With any other word, `calibration_provenance` reads the directory as an arm
without a calibration: verdict `unknown`, blocking with "no measured.yaml beside
the rows, so the efficiency columns cannot be interpreted", and
`tests/test_docs.py` counts it as a published arm. With `session`, the
regenerated `results/published/CALIBRATION_PROVENANCE.md` lists it as
`session | n/a | n/a | n/a | **refused**` with the line "rows carry 0 dtypes [],
so there is no single ridge; pass one", and on this branch (r3-align at
60db8ba plus this directory) its count reads 13 of 17.

## Provenance of these files

The exfil tarball `exfil-lambda-a100.tar.gz`, sha256
`89d1fb22c9ad4d62c00a9b6d67175f905066cd7febd4d938ff0345cc40204ebf`, holds 164
files. It was extracted fresh, and every one of the 164 was copied here and
compared by sha256 against that extraction: 164 identical, 0 different, 0
missing, no extra file. 6 of them are not in git: Triton's
`cuda_utils.cpython-312-x86_64-linux-gnu.so`, one per Triton cache (the five
`r3c-g<G>.profiles/g<G>.triton-cache/` and `census.profiles/census.triton-cache/`,
each under `MIH4X24CEAJDWYGXCZ4EKW6DTWSUG2ZBSBPV7NGI62HAUKJ7F7BQ/`), which the
repository's `.gitignore` drops by its `*.so` rule (its `results/published/`
block re-includes `.ptx`, `.cubin`, `.ncu-rep`, `.nsys-rep` and `.qdrep`, not
`.so`). The 6 `.ncu-rep` files (five pages and the census) are tracked through
`!results/published/**/*.ncu-rep`. The other 158 files are tracked, beside
this README and `../KIND`.
