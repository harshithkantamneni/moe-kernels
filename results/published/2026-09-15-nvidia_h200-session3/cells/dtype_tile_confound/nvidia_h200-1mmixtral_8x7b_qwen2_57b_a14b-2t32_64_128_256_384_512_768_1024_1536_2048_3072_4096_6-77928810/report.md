# Is the fp8/bf16 crossing shift the DTYPE or the TILE?

run id nvidia_h200-1mmixtral_8x7b_qwen2_57b_a14b-2t32_64_128_256_384_512_768_1024_1536_2048_3072_4096_6-77928810
config lookup device `NVIDIA H200`   dtypes bf16,fp8_e4m3   routing uniform (histogram realised exactly balanced)   seed 0
card `NVIDIA H200`   instrument queue-deep/l2-flush/settled-warmup/clock-under-load/v4
3 repeats per (arm, dtype), round-robin with dtype innermost; each repeat is one time_kernel call of 3 queue-deep trials sized to 200 ms, after 300 ms of warmup, L2 flushed
clock reference per dtype family: bf16 1470 MHz; fp8 1350 MHz

EVERYTHING IS SAVED TO  /workspace/results/gaps-nvidia_h200/dtype_tile_confound/nvidia_h200-1mmixtral_8x7b_qwen2_57b_a14b-2t32_64_128_256_384_512_768_1024_1536_2048_3072_4096_6-77928810
  git     outside the repo, so git has no opinion about it
  rows    /workspace/results/gaps-nvidia_h200/dtype_tile_confound/nvidia_h200-1mmixtral_8x7b_qwen2_57b_a14b-2t32_64_128_256_384_512_768_1024_1536_2048_3072_4096_6-77928810/timings.csv
  report  /workspace/results/gaps-nvidia_h200/dtype_tile_confound/nvidia_h200-1mmixtral_8x7b_qwen2_57b_a14b-2t32_64_128_256_384_512_768_1024_1536_2048_3072_4096_6-77928810/report.md
  plan    /workspace/results/gaps-nvidia_h200/dtype_tile_confound/nvidia_h200-1mmixtral_8x7b_qwen2_57b_a14b-2t32_64_128_256_384_512_768_1024_1536_2048_3072_4096_6-77928810/plan.json
Re-run the same command to resume; completed (cell, arm, dtype) triples are skipped.

COST  28 cells x 3 arms x 2 dtypes; 32 distinct Triton specialisations;
      454 s of timed kernel: 3 repeats x (300 ms warmup + 3 trials x max(200 ms budget, one call)) per (cell, arm, dtype), excluding compiles and allocation.

## The matched arms, and the shared-memory arithmetic behind them

Triton asks for `num_stages x (BM*BK + BK*BN) x element bytes` of pipeline shared memory; the sm_90 limit is 232448 B. Every number below is derived, and the 2026-09-09 H200 run's own `Required:` bytes reproduce it exactly.

  cfg_bf16: pins BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages; pairs at 28 of 28 cells
  tile_fp8: pins BLOCK_SIZE_M, GROUP_SIZE_M; pairs at 28 of 28 cells

| model | T | full fp8 config | at bf16 | tile_fp8 at bf16 | at fp8 | cfg_bf16 at fp8 |
|---|---:|---|---:|---:|---:|---:|
| mixtral-8x7b | 32 | `M64  N256 K128 G32 w4 s5` | 409600 INFEASIBLE | 147456 fits | 204800 fits | 55296 fits |
| mixtral-8x7b | 64 | `M64  N128 K128 G1  w4 s3` | 147456 fits | 147456 fits | 73728 fits | 61440 fits |
| mixtral-8x7b | 128 | `M64  N128 K128 G1  w4 s3` | 147456 fits | 196608 fits | 73728 fits | 98304 fits |
| mixtral-8x7b | 256 | `M128 N256 K128 G32 w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| mixtral-8x7b | 384 | `M128 N256 K128 G32 w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| mixtral-8x7b | 512 | `M128 N256 K128 G32 w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| mixtral-8x7b | 768 | `M128 N256 K128 G32 w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| mixtral-8x7b | 1024 | `M128 N256 K128 G16 w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| mixtral-8x7b | 1536 | `M128 N256 K128 G16 w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| mixtral-8x7b | 2048 | `M128 N256 K128 G32 w8 s3` | 294912 INFEASIBLE | 196608 fits | 147456 fits | 98304 fits |
| mixtral-8x7b | 3072 | `M128 N256 K128 G32 w8 s3` | 294912 INFEASIBLE | 196608 fits | 147456 fits | 98304 fits |
| mixtral-8x7b | 4096 | `M128 N256 K128 G16 w8 s3` | 294912 INFEASIBLE | 196608 fits | 147456 fits | 98304 fits |
| mixtral-8x7b | 6144 | `M128 N256 K128 G16 w8 s3` | 294912 INFEASIBLE | 196608 fits | 147456 fits | 98304 fits |
| mixtral-8x7b | 8192 | `M128 N256 K128 G16 w8 s3` | 294912 INFEASIBLE | 196608 fits | 147456 fits | 98304 fits |
| qwen2-57b-a14b | 32 | `M64  N256 K128 G1  w4 s5` | 409600 INFEASIBLE | 163840 fits | 204800 fits | 51200 fits |
| qwen2-57b-a14b | 64 | `M64  N128 K128 G1  w4 s3` | 147456 fits | 147456 fits | 73728 fits | 55296 fits |
| qwen2-57b-a14b | 128 | `M64  N128 K128 G1  w4 s3` | 147456 fits | 147456 fits | 73728 fits | 61440 fits |
| qwen2-57b-a14b | 256 | `M64  N128 K128 G1  w4 s3` | 147456 fits | 49152 fits | 73728 fits | 24576 fits |
| qwen2-57b-a14b | 384 | `M64  N128 K128 G1  w4 s3` | 147456 fits | 49152 fits | 73728 fits | 24576 fits |
| qwen2-57b-a14b | 512 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 768 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 1024 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 1536 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 2048 | `M128 N256 K128 G16 w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 3072 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 4096 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 6144 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |
| qwen2-57b-a14b | 8192 | `M128 N256 K128 G1  w8 s4` | 393216 INFEASIBLE | 196608 fits | 196608 fits | 98304 fits |

The full fp8 transplant is INFEASIBLE at 22 of 28 cells at bf16's 2 bytes per element and is NOT an arm of this run. It was one on 2026-09-09: it raised OutOfResources inside vLLM's `override_config`, which has no try/finally, and the leaked fp8 config then took 41 further arms.

THE SIGN, stated once so it cannot be misread. Every ratio below is

    r = time(fp8_e4m3) / time(bf16)          at MATCHED model, tokens and arm
    shift = rm / rc                          memory-region r over compute-region r

r < 1 means fp8 is FASTER, which it must be: it moves half the weight bytes and
runs tensor cores at 2.03x. shift > 1 means the crossing moves LATER in fp8,
which is the direction the published 1.149 reports and the direction the
corrected theory says should not happen. Every shift printed here is followed by
the words LATER or EARLIER.

## The confound, DERIVED from vLLM's shipped configs before any run

vLLM v0.27.1. `resolve_tile` reproduces `get_config_file_name` +
the nearest-key lookup + `get_default_config`; nothing below was observed on a GPU.

| model | T | rows/E | bf16 config | fp8_e4m3 config | BLOCK_M | differs in |
|---|---:|---:|---|---|---|---|
| mixtral-8x7b | 32 | 8.0 | `M16  N128 K128 G16 w8 s3` | `M64  N256 K128 G32 w4 s5` | DIFFERS | BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, num_warps, num_stages |
| mixtral-8x7b | 64 | 16.0 | `M32  N128 K128 G64 w4 s3` | `M64  N128 K128 G1  w4 s3` | DIFFERS | BLOCK_SIZE_M, GROUP_SIZE_M |
| mixtral-8x7b | 128 | 32.0 | `M64  N128 K128 G1  w8 s4` | `M64  N128 K128 G1  w4 s3` | SAME | num_warps, num_stages |
| mixtral-8x7b * | 256 | 64.0 | `M128 N256 K64  G1  w8 s4` | `M128 N256 K128 G32 w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| mixtral-8x7b * | 384 | 96.0 | `M128 N256 K64  G1  w8 s4` | `M128 N256 K128 G32 w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| mixtral-8x7b * | 512 | 128.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G32 w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| mixtral-8x7b | 768 | 192.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G32 w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| mixtral-8x7b | 1024 | 256.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G16 w8 s4` | SAME | BLOCK_SIZE_K |
| mixtral-8x7b | 1536 | 384.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G16 w8 s4` | SAME | BLOCK_SIZE_K |
| mixtral-8x7b | 2048 | 512.0 | `M128 N256 K64  G32 w8 s4` | `M128 N256 K128 G32 w8 s3` | SAME | BLOCK_SIZE_K, num_stages |
| mixtral-8x7b | 3072 | 768.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G32 w8 s3` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M, num_stages |
| mixtral-8x7b | 4096 | 1024.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G16 w8 s3` | SAME | BLOCK_SIZE_K, num_stages |
| mixtral-8x7b | 6144 | 1536.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G16 w8 s3` | SAME | BLOCK_SIZE_K, num_stages |
| mixtral-8x7b | 8192 | 2048.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G16 w8 s3` | SAME | BLOCK_SIZE_K, num_stages |
| qwen2-57b-a14b | 32 | 4.0 | `M16  N64  K128 G1  w4 s5` | `M64  N256 K128 G1  w4 s5` | DIFFERS | BLOCK_SIZE_M, BLOCK_SIZE_N |
| qwen2-57b-a14b | 64 | 8.0 | `M16  N128 K128 G1  w8 s3` | `M64  N128 K128 G1  w4 s3` | DIFFERS | BLOCK_SIZE_M, num_warps |
| qwen2-57b-a14b | 128 | 16.0 | `M32  N128 K128 G1  w4 s3` | `M64  N128 K128 G1  w4 s3` | DIFFERS | BLOCK_SIZE_M |
| qwen2-57b-a14b | 256 | 32.0 | `M64  N64  K64  G1  w4 s3` | `M64  N128 K128 G1  w4 s3` | SAME | BLOCK_SIZE_N, BLOCK_SIZE_K |
| qwen2-57b-a14b * | 384 | 48.0 | `M64  N64  K64  G1  w4 s3` | `M64  N128 K128 G1  w4 s3` | SAME | BLOCK_SIZE_N, BLOCK_SIZE_K |
| qwen2-57b-a14b * | 512 | 64.0 | `M128 N256 K64  G1  w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K |
| qwen2-57b-a14b * | 768 | 96.0 | `M128 N256 K64  G1  w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K |
| qwen2-57b-a14b * | 1024 | 128.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| qwen2-57b-a14b | 1536 | 192.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| qwen2-57b-a14b | 2048 | 256.0 | `M128 N256 K64  G1  w8 s4` | `M128 N256 K128 G16 w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| qwen2-57b-a14b | 3072 | 384.0 | `M128 N256 K64  G1  w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K |
| qwen2-57b-a14b | 4096 | 512.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| qwen2-57b-a14b | 6144 | 768.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |
| qwen2-57b-a14b | 8192 | 1024.0 | `M128 N256 K64  G16 w8 s4` | `M128 N256 K128 G1  w8 s4` | SAME | BLOCK_SIZE_K, GROUP_SIZE_M |

`*` marks a cell within a factor of 2.0 of that model's published bf16 crossing -- the only cells C1 and C2 read.
  mixtral-8x7b: BLOCK_SIZE_M agrees in 3/3 bracketing cells and 12/14 over the whole grid; the configs differ somewhere in 14/14
  qwen2-57b-a14b: BLOCK_SIZE_M agrees in 4/4 bracketing cells and 11/14 over the whole grid; the configs differ somewhere in 14/14

## Predictions, registered before the run

Ceilings read from /workspace/moe-kernels/moe/bench/hardware/measured_nvidia_h200.yaml
  measured 2026-09-14 -- measured on this machine by scripts/calibrate_hardware.py
  bandwidth 4374.7 GB/s
  bf16 668.9 TFLOP/s  -> ridge 152.9 FLOP/byte
  fp8  1442.5 TFLOP/s  -> ridge 329.7 FLOP/byte
  achieved fp8/bf16 = 2.157, so the compute branch scales by rc = 0.464
     (the DATASHEET relationship is 2.000 exactly; this card measures 2.157, and the difference is 7.8% of the prediction)

PREDICTED TILT -- the median fp8/bf16 time ratio over the lowest third of the
token grid divided by the median over the highest third, produced by running the
model's own timings through the SAME reduction the measurement uses. `branches`
is what the classifier finds on the predicted bf16 curve; `none` there means
the model says this arm never reaches a compute branch, which is the
`2*BLOCK_M/(alpha*b) < ridge` ceiling and not a defect of the grid.

| model | arm | tilt, activations bf16 | tilt, activations quantised | branches (mem/comp) | model shift |
|---|---|---:|---:|---|---:|
| mixtral-8x7b | native | 1.070 | 1.078 | 6/8 | 1.039 |
| mixtral-8x7b | cfg_bf16 | 1.070 | 1.078 | 6/8 | 1.036 |
| mixtral-8x7b | tile_fp8 | 1.070 | 1.078 | 6/8 | 1.039 |
| qwen2-57b-a14b | native | 0.841 | 0.898 | 7/7 | 0.886 |
| qwen2-57b-a14b | cfg_bf16 | 0.953 | 1.026 | 7/7 | 0.955 |
| qwen2-57b-a14b | tile_fp8 | 0.938 | 1.000 | 9/5 | 0.955 |

Read the two tilt columns as a BAND. `moe/spec.py` keeps activations at bf16 in an
fp8 cell because vLLM's `fused_experts` asserts it, but vLLM quantises them
internally with `a1_scale=None`, and no timing can separate the two. The band is
what the model can honestly say.

QUOTED, not measured here: the published confounded ratio is 1.131 +/- 0.095
over eight measurements from two kernels (docs/FINDINGS.md, from the arms
2026-08-28-nvidia_h200-h200-fp8-three-kernel and -fp8-refixed). Per model, vLLM:
  mixtral-8x7b 1.25,  qwen2-57b-a14b 1.10

## What this design can see (MDE)

Effect under test: the confounded tilt sits 0.131 above 1.0 (1.131 +/- 0.095, uniform-only).
Noise assumption: per-cell timing spread, MEASURED over the 26 published reports; median 0.77%, worst 1.82%.
Design: 3 round-robin repeats per (arm, dtype), compared as a ratio.

  at the median spread 0.77%:  MDE 0.018   resolves the effect
  at the worst  spread 1.82%:  MDE 0.042   resolves the effect

An MDE above the effect does not make a PASS wrong; it makes a FAIL uninformative,
and it is the number to raise --reps against before spending pod minutes.

## The measurement: the fp8/bf16 ratio, its tilt, and its branches

| model | arm | r (low third) | r (high third) | TILT | reads as | cells | branches mem/comp | rm | rc | branch shift |
|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| mixtral-8x7b | native | -- | -- | -- | REFUSED, see below | -- | -- | -- | -- | -- |
| mixtral-8x7b | cfg_bf16 | 0.5835 | 0.6070 | 0.9611 | fp8 crosses 4.0% EARLIER than bf16 | 6 | 3/3 | 0.5835 | 0.6070 | 0.9611 |
| mixtral-8x7b | tile_fp8 | 0.5645 | 0.5532 | 1.0204 | fp8 crosses 2.0% LATER than bf16 | 6 | 0/0 | -- | -- | -- |
| qwen2-57b-a14b | native | -- | -- | -- | REFUSED, see below | -- | -- | -- | -- | -- |
| qwen2-57b-a14b | cfg_bf16 | -- | -- | -- | REFUSED, see below | -- | -- | -- | -- | -- |
| qwen2-57b-a14b | tile_fp8 | -- | -- | -- | REFUSED, see below | -- | -- | -- | -- | -- |
  no branch shift for mixtral-8x7b tile_fp8: no flat stretch (pooled bf16 slope never <= 0.4); no steep stretch (pooled bf16 slope never >= 0.6), which is what 2*BLOCK_M/(alpha*b) < ridge predicts

REFUSED (no number produced, and no substitute):
  - RegimeNotResolved: mixtral-8x7b arm native: 2 paired cells, below the 6 a tilt needs (3 in each third). Widen --tokens, or look at the failed arms below. Token counts with only one dtype: [64, 384, 1536, 3072, 6144].
  - RegimeNotResolved: qwen2-57b-a14b arm native: 0 paired cells, below the 6 a tilt needs (3 in each third). Widen --tokens, or look at the failed arms below. Token counts with only one dtype: [32, 384].
  - RegimeNotResolved: qwen2-57b-a14b arm cfg_bf16: 0 paired cells, below the 6 a tilt needs (3 in each third). Widen --tokens, or look at the failed arms below. Token counts with only one dtype: [32, 64, 128, 256, 384, 2048, 4096, 8192].
  - RegimeNotResolved: qwen2-57b-a14b arm tile_fp8: 2 paired cells, below the 6 a tilt needs (3 in each third). Widen --tokens, or look at the failed arms below. Token counts with only one dtype: [128, 384, 1536, 4096, 6144].

## The ratio curve, cell by cell

  mixtral-8x7b     cfg_bf16  dlog(r)/dlog(T) = +0.0133
    32:0.563L  64:0.583L  768:0.660L  1536:0.643H  3072:0.607H  6144:0.586H
    UNPAIRED (one dtype only, excluded): [1024, 4096, 8192]
  mixtral-8x7b     tile_fp8  dlog(r)/dlog(T) = -0.0102
    768:0.559L  1536:0.567L  2048:0.564L  4096:0.553H  6144:0.547H  8192:0.554H
    UNPAIRED (one dtype only, excluded): [32, 64, 128, 512, 3072]

`L` marks the lowest third of the grid and `H` the highest; the tilt is the
median over `L` divided by the median over `H`. `dlog(r)/dlog(T)` is the same
tendency as a single fitted number over every cell, and a tilt that disagrees
in sign with it is one cell doing the work.

## Decomposition: how much of the confounded tilt is the config

| model | tilt(native) | tilt(cfg_bf16) | tilt(tile_fp8) | config effect | config share |
|---|---:|---:|---:|---:|---:|
| mixtral-8x7b | -- | 0.9611 | 1.0204 | -- | -- |
| qwen2-57b-a14b | -- | -- | -- | -- | -- |

`config effect` is tilt(native) / median(tilt of the two matched arms): what
letting each dtype pick its own config is worth, at fixed dtype physics.
`config share` is (tilt(native) - tilt(matched)) / (tilt(native) - 1): the
fraction of the confounded arm's EXCESS that pinning the config removes. It is
undefined, and printed as --, when the confounded arm shows no excess to split.

## Corroboration: the crossing detector, matched by upcrossing index

| model | arm | bf16 upcrossings | fp8 upcrossings | per-index ratio |
|---|---|---|---|---|
| mixtral-8x7b | native | ['287'] | ['614'] | 2.138 |
| mixtral-8x7b | cfg_bf16 | ['1041'] | ['667'] | 0.640 |
| mixtral-8x7b | tile_fp8 | [] | ['585'] | UNPAIRABLE: 0 vs 1 upcrossings |
| qwen2-57b-a14b | native | -- | -- | not both measured |
| qwen2-57b-a14b | cfg_bf16 | -- | [149.2410225165585, 1457.4449150719272] | not both measured |
| qwen2-57b-a14b | tile_fp8 | ['816'] | ['820'] | 1.006 |

## Headline

NONE. Validity gates V5 noise floor failed, so nothing on this page is a measurement of what it claims to measure.

## Gates

```
RESULT: VALIDITY V0 PASS bf16 668.9 TFLOP/s, fp8_e4m3 1442.5 TFLOP/s, bandwidth 4374.7 GB/s, measured 2026-09-14
[PASS   ] VALIDITY V0 ceilings  one calibration on THIS machine carries both dtypes' ceilings
                    gate: /workspace/moe-kernels/moe/bench/hardware/measured_nvidia_h200.yaml has a measured peak for both ('bf16', 'fp8_e4m3')
                    saw:  bf16 668.9 TFLOP/s, fp8_e4m3 1442.5 TFLOP/s, bandwidth 4374.7 GB/s, measured 2026-09-14
RESULT: VALIDITY V1 PASS fp8 weights arrived as ['torch.float8_e4m3fn'], quant config ['FusedMoEQuantConfig']; fp8 preflight OK: silicon has fp8 tensor cores, torch has torch.float8_e4m
[PASS   ] VALIDITY V1 fp8 is fp8  the fp8 arms handed the kernel fp8 weights AND a quant config
                    gate: every fp8 arm's weight dtype is a torch float8 type and its quant config is not None
                    saw:  fp8 weights arrived as ['torch.float8_e4m3fn'], quant config ['FusedMoEQuantConfig']; fp8 preflight OK: silicon has fp8 tensor cores, torch has torch.float8_e4m3fn, vLLM has fp8_w8a8_moe_quant_config
RESULT: VALIDITY V2 PASS 4 checks, worst rel err 5.436e-02, 0 over budget
[PASS   ] VALIDITY V2 correctness  each dtype computes the layer the fp32 oracle computes
                    gate: relative error against golden_forward within moe.bench.tolerance's budget
                    saw:  4 checks, worst rel err 5.436e-02, 0 over budget
RESULT: VALIDITY V3 PASS 0 of 112 forced arms ran a config they were not given; 21 distinct observed configs
[PASS   ] VALIDITY V3 override  override_config forces what it is given, and the arms are not one kernel
                    gate: zero arms running a config they were not given, and >= 2 distinct observed configs across the run
                    saw:  0 of 112 forced arms ran a config they were not given; 21 distinct observed configs
RESULT: VALIDITY V4 PASS 0 mismatches over 56 observed native arms
[PASS   ] VALIDITY V4 derivation  vLLM v0.27.1 loads the config tile_resolve DERIVES
                    gate: zero native cells where the observed config differs from the derived one
                    saw:  0 mismatches over 56 observed native arms
RESULT: VALIDITY V5 FAIL p90 placebo deviation 0.23% over 9 pairs; p90 per-timing spread 0.87%; under-load clock: 108 of 168 determined arms DRIFTED and are excluded from every ratio on
[FAIL   ] VALIDITY V5 noise floor  the box can resolve an effect the size of the one being measured
                    gate: p90 |placebo - 1| < 3%, p90 timing spread < 3%, and no timed arm carries a False under-load DRIFT flag from time_kernel (LEVEL is recorded on BOTH sides and excludes on neither: on this card the under-load clock is set per tile by the kernel's own power draw)
                    saw:  p90 placebo deviation 0.23% over 9 pairs; p90 per-timing spread 0.87%; under-load clock: 108 of 168 determined arms DRIFTED and are excluded from every ratio on this page; the steady arms are kept and their side recorded per row in clock_level_side (1 below the band, 40 above it, both scored); between-load clock movement +27.3% worst cell, context only; worst qwen2-57b-a14b T=32 fp8_e4m3: tile_fp8/native = 1.0023
                    a FAIL here: the model's predicted band over the matched arms is 0.938 to 1.078, 14.1 points wide against a box of 0.87%, the larger of the p90 placebo deviation and the p90 per-timing spread, so the band is WIDER than the box and C3 can be read where the arms land inside it
RESULT: CLAIM C1 PASS 7 of 7 cells within 2.0x of the published bf16 crossing have differing configs
[PASS   ] CLAIM    C1 confound exists  the two dtypes resolve DIFFERENT configs at the crossing
                    gate: the configs differ in at least one key at every crossing-bracketing cell
                    saw:  7 of 7 cells within 2.0x of the published bf16 crossing have differing configs
RESULT: CLAIM C2 PASS BLOCK_SIZE_M agrees in 7 of 7 bracketing cells (100%); this is DERIVED from vLLM's shipped configs and is decided off GPU
[PASS   ] CLAIM    C2 not the M tile  BLOCK_SIZE_M AGREES across the dtypes where the crossing lives
                    gate: agreement fraction >= 90% over cells within 2.0x of the published bf16 crossing
                    saw:  BLOCK_SIZE_M agrees in 7 of 7 bracketing cells (100%); this is DERIVED from vLLM's shipped configs and is decided off GPU
RESULT: CLAIM C3 PASS median matched tilt 0.991 over 2 (model, arm) pairs, 90% interval [0.961, 1.020]; the published confounded figure is 1.131 +/- 0.095
[PASS   ] CLAIM    C3 pure dtype  at a MATCHED tile the format barely tilts the fp8/bf16 ratio
                    gate: median matched tilt in [0.9, 1.08] over the matched arms as re-scoped on 2026-09-09 (cfg_bf16 pins BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages; tile_fp8 pins BLOCK_SIZE_M, GROUP_SIZE_M); the model at the MEASURED alpha predicts [0.938, 1.078] over the same cells
                    saw:  median matched tilt 0.991 over 2 (model, arm) pairs, 90% interval [0.961, 1.020]; the published confounded figure is 1.131 +/- 0.095
RESULT: CLAIM C4 UNKNOWN the native arm showed no excess over 1.0 to apportion, so there is nothing for the config to explain
[UNKNOWN] CLAIM    C4 config share  the config carries the majority of the confounded arm's excess
                    gate: median (tilt_native - tilt_matched) / (tilt_native - 1) >= 50% over the matched arms as re-scoped on 2026-09-09
                    saw:  the native arm showed no excess over 1.0 to apportion, so there is nothing for the config to explain

8 PASS, 1 FAIL, 1 UNKNOWN
VALIDITY GATES FAILED: ['V5 noise floor']. No number on this page may be quoted.
```
