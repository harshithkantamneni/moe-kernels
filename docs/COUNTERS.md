# Is there a route to a DRAM counter?

**Short answer: yes, three of them, and the one this project already had was
abandoned for the wrong reason.** The 2026-09-01 H200 session concluded that
`nsys` cannot sample DRAM on that device. Its own recorded data does not support
that: every attempt that launched, *including the control that requested no GPU
metrics at all*, died at the same place, and that place is an install defect, not
a permission wall. Separately, and independent of any counter, two inequalities
over already-published data bound `alpha` tightly enough to **exclude four of the
twelve published A100 alphas as physically impossible**.

This page covers four things, in the order they are worth acting on:

| | question | status |
|---|---|---|
| [1](#1-the-importer) | can the missing `nsys` importer be obtained | **YES, verified**: exact version, public URL, byte-for-byte inspected |
| [2](#2-which-providers-grant-counter-access) | which providers grant counter access | **partly**: the mechanism is verified from NVIDIA, most provider docs are NOT verified here |
| [3](#3-counter-free-bounds-on-l) | can `L` be anchored without a counter | **partly, and it already produces a result**: a bracket that excludes four published fits, plus one experiment worth building |
| [4](#4-the-counter-run-written-as-a-plan) | what exactly would be measured | **written and runnable**: `scripts/dram_counter_route.py --dry-run` |

Everything below is re-verifiable from this repository or from a public URL. Where
something could not be verified in this session it says so in those words.

---

## 0. What is actually known, re-checked

### 0.1 `ncu` is blocked on RunPod. That part stands.

`ERR_NVGPUCTRPERM`. Hardware performance counters are gated behind the NVIDIA
kernel module parameter `NVreg_RestrictProfilingToAdminUsers`, which a container
tenant cannot set. Recorded in `docs/RUNPOD.md:246` and
`scripts/profile_open_questions.sh`.

**But the mechanism has two doors, not one.** NVIDIA's own
[ERR_NVGPUCTRPERM page](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
says, verbatim:

> "When profiling within a container, access must be enabled on the host, or the
> container must be started with the appropriate permissions by passing
> `--cap-add=SYS_ADMIN` as an admin user."

So the ask to a provider is either *change a host module parameter and reboot*
(expensive, fleet-wide) **or** *add one Linux capability to my container* (cheap,
per-instance). Every note in this repository asks for the first. Asking for the
second is a materially easier request and has never been made.

### 0.2 The `nsys` verdict is wrong, and the raw data in this repo says so

`results/published/2026-09-01-nvidia_h200-alpha-0558/session/nsys_probe.json`
records six attempts. Read the `stderr_head` field of each:

```bash
.venv/bin/python - <<'EOF'
import json
p = "results/published/2026-09-01-nvidia_h200-alpha-0558/session/nsys_probe.json"
for a in json.load(open(p))["attempts"]:
    print(f'{a["returncode"]}  {a["seconds"]:5.2f}s  {a["label"][:52]:<52}  '
          f'{a["stderr_head"].splitlines()[0][:60]}')
EOF
```

| rc | label | first line of stderr |
|---|---|---|
| 0 | `--gpu-metrics-device=all` + `--gpu-metrics-set=gh100` @ 200 kHz | `Importer error status: The importer binary and its dependencies were not found.` |
| 0 | `--gpu-metrics-device=all` @ 200 kHz | same |
| 0 | `--gpu-metrics-device=0` | same |
| 0 | `--gpu-metrics-device=all` @ 10 kHz | same |
| 1 | `--gpu-metrics-devices=all` (wrong spelling) | `unrecognised option` |
| 0 | **no gpu metrics at all: the CONTROL** | same importer error |

The control requested **no counters**. It failed identically. A ladder in which
the negative control fails the same way as every treatment discriminates nothing
about the treatment. The recorded verdict --

> "NO. No invocation of this nsys sampled a DRAM metric on this device. The open
> path in docs/FINDINGS.md is now a closed one for this pod"

-- is not supported by the run that produced it. What the run actually shows is
that **no attempt ever reached the point where a counter permission could
matter**: all five that launched collected a `.qdstrm` and then failed to convert
it. Whether the H200's DRAM sampler is available to a RunPod tenant remains
**untested**, not closed.

`scripts/alpha_surface.sh` gets this right in its comments (lines 75-93) and even
ran an `nsys_importer_hunt` arm on the 2026-09-01 session, which
`results/published/2026-09-01-nvidia_h200-alpha-surface-s4/ARMS.tsv` records as
`PASS` in 1 second. **Its log was never published**, so what the hunt found is
lost. That is a publishing gap to close on the next session, not a result.

### 0.3 The existing `.qdstrm` captures are gone

They were written to `/workspace/session/20260901T214218Z/nsys/probe-*.qdstrm` on
a pod that has been terminated, and were never copied off.
`results/nsys_dram_probe/` on the laptop holds seven empty directories:

```bash
find results/nsys_dram_probe -type f | wc -l      # 0
find . -name '*.qdstrm' -o -name '*.nsys-rep' | wc -l   # 0
```

So the "convert the capture elsewhere" route is **dead for the captures already
taken**. It is alive for the next one, which is why section 1 still matters.

---

## 1. The importer

### 1.1 Where it ships, verified by unpacking the package

The pod reported `NVIDIA Nsight Systems version 2022.4.2.50-32196742v0`
(`nsys_probe.json` -> `discovery.version_raw`). The matching package is in
NVIDIA's **public** CUDA apt tree, no login, no EULA click-through:

```
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
  nsight-systems-2022.4.2_2022.4.2.50-1_amd64.deb
```

Verified in this session, by download and inspection:

| check | result |
|---|---|
| HTTP status / size | `200`, `286,008,550` bytes |
| `control` -> `Version:` | `2022.4.2.50-32196742v0` -- **exactly the pod's version string** |
| contains the importer | `./opt/nvidia/nsight-systems/2022.4.2/host-linux-x64/QdstrmImporter` |
| importer file | 263,432 bytes, `ELF 64-bit LSB executable, x86-64, stripped` |
| `host-linux-x64` total | 498.5 MB |
| `target-linux-x64` total | 201.5 MB (this is the half the pod had: `nsys`, `libcupti.so.11.7`, the reports) |
| importer `RUNPATH` | `$ORIGIN` |
| importer `DT_NEEDED` | 53 entries; every private one (`libAnalysis.so`, `libHostCommon.so`, `libStreamSections.so`, ...) sits in `host-linux-x64` |

Two consequences.

* **The importer is not missing from the world, it is missing from the image.**
  The pod has a target-only Nsight Systems. `apt-cache search nsight` returning
  nothing was an apt-repository problem, and the package was reachable by plain
  `curl` the whole time.
* **No root, no `Depends`, no Qt.** The `Depends:` line lists X libraries for the
  GUI; `QdstrmImporter` needs none of them. `dpkg -x` unpacks without root and
  ignores `Depends` entirely, and `$ORIGIN` makes the unpacked directory
  self-contained apart from glibc and libstdc++.

```bash
cd /workspace
curl -fSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2022.4.2_2022.4.2.50-1_amd64.deb
dpkg -x nsight-systems-2022.4.2_2022.4.2.50-1_amd64.deb /workspace/nsight
NS=/workspace/nsight/opt/nvidia/nsight-systems/2022.4.2
$NS/bin/nsys --version                      # should print the same version string
$NS/host-linux-x64/QdstrmImporter --help    # read the flags before using them
```

### 1.2 Converting a `.qdstrm` on a different machine

The Nsight Systems User Guide states the rule that governs this:

> "Use the same `nsys` version that generated the `.qdstrm` file to convert it."

and describes `nsys import` as the command that turns a `.qdstrm` into a
`.nsys-rep`. So a capture taken on a pod **can** be converted on a laptop or in a
different container, provided the converting install is the same version. For the
2022.4.2.50 captures that means precisely the package above.

Two things this page does **not** claim, because they were not verified here:

* the exact `QdstrmImporter` command line (`-i` versus `--input`) in the 2022.4.2
  build. Run `--help` first, and prefer `nsys import` from the unpacked `bin/nsys`
  if that build offers it.
* that a macOS or Windows host package of the same version can convert a Linux
  `.qdstrm`. The importer verified above is `linux-x64`. Any Linux container on
  the laptop runs it; a native macOS run does not.

### 1.3 The better move: do not repair 2022.4.2 at all

The same public tree carries current builds beside the old one --
`nsight-systems-2025.6.3`, `nsight-systems-2026.1.3`, and the `nsight-compute-*`
packages that provide `ncu`. A modern `nsys` writes `.nsys-rep` directly and never
enters the importer path, so the whole failure disappears rather than being
worked around.

This also removes a risk the repair route keeps: **2022.4.2 predates the H200.**
Its metric-set list does offer `gh100`
(`nsys_probe.json` -> `discovery.metric_sets`), but a 2022 build recognising a
2024 board's chip ID for GPU-metrics sampling is an assumption, not a finding.
Repairing the importer would only turn "no report" into "a report that may say
the sampler is unavailable", and the two look identical from the outside. Install
a current build and the ambiguity is gone.

---

## 2. Which providers grant counter access

### 2.1 The mechanism, verified

Two independent doors, from NVIDIA's own page quoted in 0.1:

1. **Host side.** `NVreg_RestrictProfilingToAdminUsers=0` in the module options,
   then a module reload or a reboot. Requires ownership of the guest kernel.
2. **Container side.** `--cap-add=SYS_ADMIN` on the container, set by whoever
   starts it.

A tenant can *observe* both without paying for a long session:

```bash
grep RestrictProfilingToAdminUsers /proc/driver/nvidia/params   # the host's setting
grep CapEff /proc/self/status                                   # bit 21 is CAP_SYS_ADMIN
```

`scripts/dram_counter_route.py --probe` reads both, runs one sub-second `ncu`
attach against `/bin/true`, and returns `OPEN` / `BLOCKED` / `REFUSE` with the
specific next action. It distinguishes four failures that are indistinguishable
in a log: no `ncu`, blocked by the host flag, blocked but fixable with a
capability, and `nsys` present without its importer. Run it in the first minute
of any new pod.

### 2.2 The structural answer, which needs no provider documentation

**If you own the guest kernel, you can turn counters on yourself.** That covers
every ordinary IaaS GPU VM (AWS `p4d`/`p5`, GCP `a2`/`a3`, Azure `ND`, Oracle
`BM.GPU`, and any bare-metal rental) because you have root on the instance:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia   # or reboot
grep RestrictProfilingToAdminUsers /proc/driver/nvidia/params                    # confirm 0
```

**If you rent a container on someone else's kernel, you cannot**, and the only
lever is whether that provider will start your container with `SYS_ADMIN`. That
is the entire distinction, and it is a property of the *product shape*
(bare metal / VM versus managed container), not of the brand.

### 2.3 What I verified about specific providers, and what I did not

| provider | offering | finding | verified? |
|---|---|---|---|
| RunPod | container | `ncu` -> `ERR_NVGPUCTRPERM`; containers not privileged | **yes**, measured in this repo (`docs/RUNPOD.md:246`, `profile_open_questions.sh`) |
| Vast.ai | container | its docs page for instance Docker options lists exactly three settable options: environment variables, hostname, ports. No capabilities, no `--privileged`. | **yes**, [docs.vast.ai/instances/docker-execution-environment](https://docs.vast.ai/instances/docker-execution-environment), read in this session. Caveat: one page only; another surface may expose more |
| AWS / GCP / Azure / Oracle GPU VMs | VM, root in guest | the module-parameter route in 2.2 applies because you own the guest kernel | **mechanism yes** (NVIDIA's page), **per-provider docs no** |
| Lambda, CoreWeave, Crusoe, Nebius, Together, Modal, Paperspace | mixed VM and container | **NOT VERIFIED.** I could not read their documentation in this session: the web-search budget was exhausted and I had no reliable direct URLs. Do not quote a claim about these from this page | **no** |

The honest summary: **the cheapest unexplored option is not a different provider,
it is asking the current one for `--cap-add=SYS_ADMIN`.** It is one flag, it is
NVIDIA's own documented remedy, and nothing in this project has asked for it.

---

## 3. Counter-free bounds on `L`

### 3.1 The bracket, which is a result and not a plan

`alpha = B / L`, and `L` is the unidentified quantity. Two inequalities pin it
from opposite sides, and **neither uses the fitted intercept**:

```
LOWER   alpha >= B / t(1)
        At n=1 there is exactly one M-tile per expert, so the kernel reads the
        weight set exactly once and t(1) is that read PLUS non-negative extras
        (launch, occupancy, tail). Extras are never negative, so L <= t(1).

UPPER   alpha <= B * peak / (W + a)
        Traffic at one tile is at least the compulsory weight read plus one tile
        of activations, and no traffic moves faster than the pin rate, so
        L >= (W + a) / peak.
```

Both bound the **uncorrected** `alpha` (the reports' `alpha` column), because `B`
is the raw fitted slope and the activation share sits inside it. Subtracting that
share needs a bandwidth constant, and refusing to need one is the point.

```bash
.venv/bin/python scripts/dram_counter_route.py --bracket
```

On the two published alpha surfaces, 28 identifiable fits (the withdrawn
`BLOCK_SIZE_N=256` arm excluded on both cards):

```
  card  model            G   BM    pub     t1   n>=3     lo     hi  pub    n>=3      %pk
  A100  mixtral-8x7b     1   32  1.018  0.872  1.046  0.872  1.204  in     in       72.4
  A100  mixtral-8x7b    16   32  0.647  0.452  0.705  0.452  0.631  ABOVE  ABOVE    71.6
  A100  qwen2-57b-a14b  16   32  0.739  0.460  0.832  0.460  0.649  ABOVE  ABOVE    70.9
  A100  qwen2-57b-a14b  64   32  0.679  0.430  0.771  0.430  0.639  ABOVE  ABOVE    67.2
  A100  qwen2-57b-a14b   8   32  0.706  0.471  0.770  0.471  0.659  ABOVE  ABOVE    71.5
  ... 23 more rows in
GATE B1  CLAIM  FAIL   4 of 28 ABOVE their own pin-rate bound
GATE B2  CLAIM  FAIL   4 of 28 ABOVE their own pin-rate bound
GATE B3  CLAIM  FAIL   median width 0.289 alpha, gate <= 0.10
```

**Four published A100 alphas are not uncertain, they are impossible.** Each
asserts a memory branch that moves the compulsory weight bytes faster than the
A100's 2039 GB/s pin rate. The same four `n >= 3` refits are impossible by a
wider margin (11.6% over the bound on the mixtral cell, against 2.5% for the
published anchor). Every one of the four is `BLOCK_M=32` **with a swizzle**; not
one of the `G=1` fits is among them, which is exactly the shape the swizzle
explanation predicts and is a non-trivial thing for the arithmetic to have found
on its own.

Two things the bracket does not do.

* **It does not pin a number.** Median width is 0.289 in `alpha`. The width is
  `alpha_lo * (peak/achieved(1) - 1)`, so a cell running at 70% of peak carries a
  43% relative bracket no matter how good the timing is. That is the argument for
  the counter, stated as a number.
* **Its lower bound is not a physics statement in the other direction.** Two H200
  fits sit ~3% *below* their lower bound, which means the fitted branch sits above
  the measured `t(1)` -- a fit-and-noise question, not an impossibility. The
  script marks the two cases differently (`ABOVE` versus `below`) and gates only
  on `ABOVE`.

### 3.2 The l2_flush axis cannot do this job, and here is the arithmetic

`docs/RUNPOD.md:288` offers the flush axis as the counter-free stand-in for a hit
rate. For *this* question it is too small by an order of magnitude:

| | A100 | H200 |
|---|---|---|
| L2 (`moe/bench/hardware/measured_*.yaml` -> `observed.l2_bytes`) | 41,943,040 B (40 MiB) | 62,914,560 B (60 MiB) |
| as a fraction of `W = 2,818,572,288 B` for mixtral | **1.49%** | **2.23%** |

The whole flush axis can move at most 1.5-2.2% of `R(1)`. The three anchors
differ by 17.2% at `n=2` and 41.7% at `n=8`. An instrument with a 2% dynamic
range cannot separate signals that differ by 17-42%. The `l2_absorption` plot
stays useful for what it was built for and is not a substitute here.

### 3.3 Two other rejected substitutes, with the reason

* **`implied_traffic_ratio`** (`docs/RUNPOD.md:277`) is `time x bandwidth` over
  modelled bytes. It is an upper bound that also absorbs occupancy and latency
  stalls. It is *the same inequality* as the bracket's upper bound, already used
  above, and it cannot be tightened into a measurement.
* **`nvidia-smi` memory utilisation** reports the fraction of *time* the memory
  controller was busy, not bytes moved. No arithmetic converts a duty cycle into
  a byte count without assuming the bandwidth achieved while busy, which is the
  unknown. (Asserted from the field's semantics; not verified in this session.)

### 3.4 The one counter-free experiment worth building

**Vary `W` at fixed `n=1` and fit time against known traffic.** The reason `L` is
unidentified is that it is an extrapolation *along `n`*. It does not have to be:
`L = W / bw_eff`, and `bw_eff` at the `n=1` operating point is directly
measurable if `W` can be changed by a known factor while the kernel, the tile
geometry and the launch shape stay fixed.

The knob that does that is **the number of active experts**. Route rows to a
subset of experts; vLLM's `moe_align_block_size` gives the unrouted experts zero
tiles, so their weights are never read, and `W` scales exactly with the count
while every expert that *is* active keeps its single M-tile.

```
t(1; E_active) = D + E_active * (3 F H b) / bw_eff
```

OLS over `E_active` gives `bw_eff` and the fixed cost `D` as separate measured
numbers, and `L = W / bw_eff` stops being an extrapolation.

* **Occupancy holds even at the bottom of the sweep.** At mixtral with
  `BLOCK_SIZE_N=64`, one M-tile is `ceil(2F/BN) = 448` CTAs for the up GEMM, so
  `E_active=1` still launches 448 CTAs against 108 SMs. The measurement does not
  fall off a cliff at small `W`.
* **Predicted discrimination.** For the mixtral `G=16, BM=32` A100 cell: the
  `t(1)` anchor corresponds to `bw_eff = 1447 GB/s`, the published `A+B` anchor to
  2072 GB/s and the `n>=3` refit to 2243 GB/s. The latter two exceed the 2039 GB/s
  pin rate, so this experiment can only confirm or refute the `t(1)` end -- which
  is precisely the end the bracket leaves open.
* **Not built here.** `block_m_crossing_sweep.py` routes with `balanced_ids`,
  uniform over all `E`, and has no `--active-experts` knob. Adding one is a change
  to a file this workflow does not own, so it is **described** rather than made.

A second, cheaper control worth noting: any per-tile slope measured on a problem
whose *entire* weight set fits in L2 is by construction not DRAM re-read, so it
bounds the non-traffic part of `B`. None of the three swept models reaches that at
full `E` (the smallest, deepseek-v2-lite, is 1.107 GB against 40-60 MB of L2), so
it needs a shrunken synthetic config rather than an existing one.

---

## 4. The counter run, written as a plan

```bash
.venv/bin/python scripts/dram_counter_route.py --dry-run      # off GPU, prints all of this
```

### 4.1 The cell

Not a new one. It is the cell the alpha surface already measured, so the counter
answers the ladder rather than a different question. It is also the cell the
adversarial evaluation named: the widest anchor disagreement anywhere on the
surface.

| | |
|---|---|
| card | A100-SXM4-80GB. Ridge **145.81** FLOP/byte from `measured_nvidia_a100_sxm4_80gb.yaml` (262.371 TFLOP/s over 1.79936 TB/s), **never the 160.3 the reports carry** |
| model | `mixtral-8x7b`, bf16, `E=8 k=2 H=4096 F=14336` |
| pinned | `GROUP_SIZE_M=16`, `BLOCK_SIZE_N=64`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_K=64`, `num_warps=8`, `num_stages=3` |
| swept | tiles per expert `n = 1, 2, 3, 4, 6, 8` (rows per expert 32..256), and `--cache-control` in `{all, none}` |
| `W` | 2,818,572,288 B = **2.81857 GB** (`E x 3 F H x 2`) |
| `a` | 26,214,400 B = **26.2144 MB** per M-tile (`E x BM x (2H+3F) x 2`) |
| control cell | the H200 twin of the same arm, where the three anchors agree within 0.036 (0.649 / 0.629 / 0.613). A counter that disagrees *there* is a broken counter, not a result |

Run the `BLOCK_M=64` twin as well. `BM=32`'s roof threshold is `32/145.81 = 0.219`
and every estimator already sits above it, so `BM=32` alone cannot test the
tile-cap claim; `BM=64`'s threshold is `64/145.81 = 0.439`, which is the number
the surviving result actually rests on.

### 4.2 The metric

```
dram__bytes_read.sum                 the one the evaluation named
dram__bytes_write.sum                a large write share means the byte model is
                                     missing a term, not that alpha is high
lts__t_sector_op_read_hit_rate.pct   "alpha is the fraction of a re-read that
                                     MISSES L2" is the published definition; this
                                     is the only direct measurement of it
gpu__time_duration.sum               bytes and time from the SAME launch
```

**Do not filter by kernel name at capture time.** All of `W` moves inside the two
`fused_moe_kernel` launches, but the byte model charges the whole layer, so the
measured level is only comparable to it if the auxiliary launches are counted
too. Record bytes *per kernel name* and let the analysis report both totals. A
filter applied at capture cannot be undone afterwards.

### 4.3 The registered predictions

`R(n) = W (1 + alpha (n-1)) + a n`. GB of DRAM read, per anchor:

| n | rows/expert | t1 = 0.452 | published = 0.647 | n>=3 = 0.705 | spread |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 2.8448 | 2.8448 | 2.8448 | **0.0%** |
| 2 | 64 | 4.1457 | 4.6956 | 4.8574 | 17.2% |
| 3 | 96 | 5.4466 | 6.5464 | 6.8699 | 26.1% |
| 4 | 128 | 6.7475 | 8.3971 | 8.8825 | 31.6% |
| 6 | 192 | 9.3493 | 12.0987 | 12.9077 | 38.1% |
| 8 | 256 | 11.9511 | 15.8003 | 16.9328 | 41.7% |

**The `n=1` row is the trap.** All three anchors predict the same traffic there,
because they differ only in slope. Anyone who profiles one launch and reports an
`alpha` has measured the byte model's intercept and called it the answer. `n=1` is
a *validity* check; the slope is the claim.

The estimator is

```
alpha = (dR/dn - a) / W
```

with **no bandwidth constant, no extrapolation to zero tiles and no timing
model**. That is why this experiment is decisive and the ladder is not: the
quantity the ladder can only infer is the quantity a counter reads directly.

The spread column is the accuracy the experiment needs: 17% at `n=2`, 42% at
`n=8`. A hardware byte counter separates those with a large margin. (I did not
verify a published accuracy figure for `dram__bytes_read.sum` in this session;
the point stands on the coarseness of the requirement, not on a quoted error bar.)

### 4.4 The gates

Validity gates say whether the run may be read at all. Claim gates say what it
decided. Each states what its own failure invalidates.

| gate | kind | claim | threshold | a FAIL invalidates |
|---|---|---|---|---|
| V1 | validity | four or more tile counts actually profiled | `>= 4` | everything below; a scorer that examined nothing reports no failures |
| V2 | validity | `R(1)` is one compulsory weight read | `\|R(1)/(W+a) - 1\| <= 0.10` | **the units of `alpha` itself**. If `R(1)` is not one weight read then `B/L` is not a re-read fraction and every published alpha on every card is uninterpretable rather than merely uncertain. This gate has never been run |
| V3 | validity | read traffic increases with tile count | strictly increasing | the affine model; a non-monotone ladder means the launches are not all the same kernel |
| V4 | validity | the traffic ladder is affine in `n` | max residual `<= 3%` | the single-slope reading; a curved ladder means no scalar `alpha` describes it |
| C1 | claim | exactly one anchor matches | 1 anchor within 0.05 | zero survivors means all three anchors are wrong and the branch model needs replacing, not re-anchoring; more than one means the cell was badly chosen |
| C2 | claim | the counter lands inside the counter-free bracket | `0.4522 <= alpha <= 0.6313` | one of the two measurements; the bracket uses only measured time and the pin rate, so a counter outside it means timing or counter is wrong and the run cannot say which |
| C3 | claim | this `BLOCK_M` still cannot reach the compute roof | `alpha > BM/ridge` | **the one result the 2026-09 evaluation did not kill.** A counter alpha below the threshold means the tile height *can* reach the roof and the cap claim must be withdrawn |

Note what C2 already says before any run: the published anchor for this cell,
0.6473, is **already above** the bracket's upper bound of 0.6313. The plan is
registered against a number the arithmetic has already excluded.

### 4.5 Cache control is a swept parameter, not a default

`ncu`'s default is Flush All: "all GPU caches are flushed before each kernel
replay iteration during profiling" (Nsight Compute documentation). That is a
different cache state from the timed sweep, which never flushes. The **level** it
can move is bounded by L2, at most 2.1% of `R(1)` (section 3.2). The **slope**,
which is the claim, differences that term away entirely. Run both modes anyway:
two agreeing numbers close the question and two disagreeing ones are themselves
the result.

### 4.6 The command

```bash
# 0. is the route open at all
python scripts/dram_counter_route.py --probe

# 1. the timed ladder, unprofiled, so time and traffic come from the same cell
python scripts/block_m_crossing_sweep.py --model mixtral-8x7b --dtype bf16 \
    --tiles 32 --group-m 16 --block-n 64 \
    --r-max 256 --row-step 32 --step-probes 0

# 2. the same cell under the counter, one tile count per invocation.
#    warmup 0 and iters 1: every profiled launch then belongs to ONE
#    fused_experts call, so the per-launch bytes can simply be summed.
for cc in all none; do
  for n in 1 2 3 4 6 8; do
    ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_op_read_hit_rate.pct,gpu__time_duration.sum \
        --replay-mode kernel --cache-control $cc \
        --csv --page raw --target-processes all \
        --log-file counters-n$n-cc$cc.csv \
        python scripts/block_m_crossing_sweep.py --model mixtral-8x7b \
          --tiles 32 --group-m 16 --block-n 64 \
          --r-max $(( n * 32 )) --row-step $(( n * 32 )) --step-probes 0 \
          --warmup 0 --iters 1
  done
done

# 3. score it against the gates above
python scripts/dram_counter_route.py --analyse counters.json
```

`--dry-run` prints the JSON schema `--analyse` requires. Every key is mandatory;
a missing one raises rather than defaulting, because a zero that was never
measured is the failure this whole file exists to avoid.

### 4.7 Cost

12 profiled invocations (6 tile counts x 2 cache modes), about 60 profiled
launches, per `BLOCK_M`. The largest cell is 256 rows per expert, which the timed
sweep measures in single-digit milliseconds; `ncu` replay and its save/restore of
the 2.8 GB weight buffers dominate. Budget 15 minutes of GPU time and one
pod-hour end to end, which is the smallest unit any of these providers bills.

**The cost of this experiment has never been the money.** It has been one Linux
capability.

---

## 5. What this changes

### 5.1 Nothing published has to be withdrawn *because of this page*

Section 3.1's four impossible A100 alphas are a real refutation, but the same
finding is already in the adversarial evaluation's hands ("Four BM=32 fits with
G>1 imply 102-112% of the A100's theoretical 2039 GB/s, which is impossible").
This page reproduces it independently, from the published JSON, with a runnable
check and a test that asserts it.

### 5.2 Changes needed in files this workflow does not own

* **`scripts/nsys_dram_probe.py`** prints and stores the verdict quoted in 0.2.
  Its own ladder cannot support it: the control failed identically. The verdict
  string should say "this nsys cannot write a report at all, so the sampler was
  never tested", and the probe should check for `host-linux-x64/QdstrmImporter`
  before running the ladder and refuse early if it is absent.
* **`docs/RUNPOD.md:246-296`** should record (a) that `--cap-add=SYS_ADMIN` is
  NVIDIA's documented container-side remedy and is a weaker ask than "a provider
  that grants privileged containers", and (b) that the image's `nsys` is a
  target-only install, so "worth one test at the start of your first session"
  should be preceded by the importer check.
* **`scripts/alpha_surface.sh`** runs `nsys_importer_hunt` but its log is not
  published. Add it to the publish set; the 2026-09-01 answer is lost.
* **`scripts/block_m_crossing_sweep.py`** would need an `--active-experts` knob
  for the experiment in 3.4. Described, not made.

### 5.3 Files this page owns

* `docs/COUNTERS.md` (this file)
* `scripts/dram_counter_route.py` -- `--dry-run`, `--bracket`, `--probe`,
  `--self-test`, `--analyse`
* `tests/test_dram_counter_route.py` -- 38 tests, all off GPU

```bash
.venv/bin/python -m pytest tests/test_dram_counter_route.py -q
.venv/bin/ruff check scripts/dram_counter_route.py tests/test_dram_counter_route.py
```
