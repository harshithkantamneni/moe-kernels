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
| [6](#6-the-r3-arms-under-the-counter) | what R3's three arms read from DRAM at each GROUP_SIZE_M | **written and runnable off GPU**: `scripts/dram_counter_route.py --dry-run --family r3-arms`; never run on a box whose counters were read |

Everything below is re-verifiable from this repository or from a public URL. Where
something could not be verified in this session it says so in those words.

---

## 0. What is actually known, re-checked

### 0.1 `ncu` was refused on both RunPod pods that asked. That part stands, for two pods.

`ERR_NVGPUCTRPERM`, on a rented H200 on 2026-08-25 (`profiles/q2_kernel_names.txt`,
ncu over the harness's own CLI) and on another on 2026-09-15 (section 2.1).
Hardware performance counters are gated behind the NVIDIA kernel module
parameter `NVreg_RestrictProfilingToAdminUsers`, which a container tenant cannot
set. Two pods are not the platform: the 2026-09-09 and 2026-09-10 pods were
never asked, and `scripts/dram_counter_route.py --probe` answers for each pod
(`docs/RUNPOD.md`, its counters section).

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
grep CapEff /proc/self/status         # bit 21 CAP_SYS_ADMIN, bit 38 CAP_PERFMON
```

Both bits matter: from Linux 5.8 and driver R450 on, `CAP_PERFMON` opens the
counter gate as well as `CAP_SYS_ADMIN`, and it is the narrower ask, so it is
the one to make of a provider first. Read the bits off the RAW `/proc` field,
which Linux prints in sixteen zero-padded hex digits. The published
2026-09-09 and 2026-09-10 payloads record `cap_eff` as `0xa80425fb` because the
probe stored `hex(mask)` and `hex` strips leading zeros; those eight digits are
Python's formatting, not the mask's width, and an earlier version of this
paragraph read a 32-bit mask off them. What the value does establish is that
`0xa80425fb` is below 2**38, so bit 38 was clear on both pods.

`scripts/dram_counter_route.py --probe` reads both, launches ONE real CUDA
kernel under `ncu` (`moe/bench/counter_probe_kernel.py`, a 4 MiB in-place add),
and returns `OPEN` / `BLOCKED` / `REFUSE`. It reports `OPEN` only when `ncu`
hands back a NUMBER for `dram__bytes_read.sum`, parsed by the same
`parse_ncu_csv` the measuring path uses. Run it in the first minute of any new
pod. It costs about fifteen seconds, which is the child's `torch` import.

**THIS PROBE RAN `ncu --metrics dram__bytes_read.sum /bin/true` UNTIL
2026-09-15, AND EVERY `OPEN` IT EVER RETURNED WAS A FALSE POSITIVE.**
`/bin/true` launches no CUDA kernel. On this driver the permission check
happens at the first counter collection, which is the first kernel launch, so
`ncu` attached, found nothing to profile, printed `==WARNING== No kernels were
profiled.` and exited 0 without ever attempting a read. The probe reported
"attached with no permission error" and the 2026-09-09 and 2026-09-10 sessions
recorded `OPEN` on that basis; both published payloads carry `ncu`'s own
warning in the `output_head` field the probe captured and never consulted. On
2026-09-15 a rented H200 booked two 120-minute counter arms on that word and
both died in 35 seconds with `ERR_NVGPUCTRPERM`. What is known today: on both
boxes where a counter read was ever attempted it was REFUSED (a rented H200 on
2026-08-25, `profiles/q2_kernel_names.txt`, and this one), and the 2026-09-09
and 2026-09-10 pods were never asked.

The four failures it still distinguishes, because they are indistinguishable in
a log: no `ncu`; a counter read REFUSED by the box (`ERR_NVGPUCTRPERM`, a fact
about the pod and not a broken instrument); a counter read that SUCCEEDED; and
no CUDA device to launch on at all, where nothing has been established either
way and the verdict is `REFUSE` rather than `BLOCKED`.

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
| RunPod | container | `ncu` -> `ERR_NVGPUCTRPERM` on both rented H200s that attempted a read (2026-08-25; 2026-09-15, a pod holding neither `CAP_SYS_ADMIN` nor `CAP_PERFMON`); the 2026-09-09 and 2026-09-10 pods were never asked, so this is two pods and not the platform | **yes, for those two pods**, measured in this repo (`profiles/q2_kernel_names.txt`, docs/FINDINGS.md's 2026-09-15 retraction); `scripts/dram_counter_route.py --probe` answers for each pod |
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

THE REGISTERED CARD IS THE H200, and this section carried the A100 for a day
after the script stopped defaulting to it. `dram_counter_route.py` resolves
`--card` from the attached device and the session driver passes what
`counter_route_card` resolved, so the plan an operator reads on the box prints
the H200's own ridge of **151.43** (662.951 TFLOP/s over 4.37803 TB/s on the
2026-09-21 calibration; 155.93 on the 2026-09-10 one) and not
the A100's 145.81. The A100 row is kept below and labelled RETIRED because it
is the registration the 2026-09-02 analysis was written against and
`tests/test_dram_counter_route.py` still asserts its figures; it is not the
cell any session books.

**The registered cell**, printed by
`dram_counter_route.py --dry-run --card nvidia_h200 --block-m 64 --block-n 32`:

| | |
|---|---|
| card | nvidia_h200. Ridge **151.43** FLOP/byte from `measured_nvidia_h200.yaml` (662.951 TFLOP/s over 4.37803 TB/s, 2026-09-21) |
| model | `mixtral-8x7b`, bf16, `E=8 k=2 H=4096 F=14336` |
| pinned | `GROUP_SIZE_M=16`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N` **32 and 128**, one arm each, `BLOCK_SIZE_K=64`, `num_warps=8`, `num_stages=4` |
| swept | tiles per expert `n = 1, 2, 3, 4, 6, 8` (rows per expert 64..512), and `--cache-control` in `{all, none}` |
| `W` | 2,818,572,288 B = **2.81857 GB** (`E x 3 F H x 2`) |
| `a` | **52.429 MB** per M-tile at `BM=64` (`E x BM x (2H+3F) x 2`) |
| the reading | the RATIO across the two `BLOCK_N`, scored by `--contrast`. One `BLOCK_N` settles nothing about a `BLOCK_N` dependence |

**RETIRED, the A100 registration** (2026-09-02). Kept for provenance and because
the tests assert its arithmetic; not bookable, and not what the script defaults
to.

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
the surviving result actually rests on. (Retracted 2026-09-02, in part: the
threshold `alpha > BM/ridge` is the identity `cap = 2BM/(alpha b)`, which for a
LADDER alpha is high by `(1 + phi + delta)`, `moe/bench/ai_model.py` (EXA). A
COUNTER alpha is a byte ratio, not a ladder fit, so the identity is exact for it
and the `0.219` / `0.439` thresholds stand for what C3 scores; what does not
stand is comparing them with a ladder alpha, which is what every estimator
"already sitting above" 0.219 is. Score the ladder side through
`ai_model.cap_from_fitted` at a stated `alpha_a`, and print the `alpha_a`.)

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
| C1 | claim | **the question the cell can answer, decided from the anchors before the run.** `c1_registration` prints SEPARATING or NOT DISCRIMINATING; on a SEPARATING cell the claim is that exactly one anchor matches, on a NOT DISCRIMINATING one it is that the counter lands INSIDE the cluster | SEPARATING: 1 anchor within 0.05. NOT DISCRIMINATING: at least one | SEPARATING: zero survivors means all three anchors are wrong and the branch model needs replacing, not re-anchoring; more than one means the cell was badly chosen. NOT DISCRIMINATING: a FAIL is every registered anchor refuted at once, and no published alpha on that card survives it |

**The registered H200 cell is NOT DISCRIMINATING and the row above used to say the opposite.** Its anchors are 0.6202 / 0.6583 / 0.6595, a closest pair of 0.0011 against a window of 0.05, so a measurement agreeing with one agrees with its neighbour. Under the retired row a reader would score a successful clustered run, which is the outcome this cell can produce, as a refutation. The A100 cell IS separating (closest pair 0.0574) and it is where the retired wording came from. The claim the H200 cell does carry is the `BLOCK_N` contrast, whose two rivals are 87% apart.
| C2 | claim | the counter lands inside the counter-free bracket | `0.4522 <= alpha <= 0.6313` | one of the two measurements; the bracket uses only measured time and the pin rate, so a counter outside it means timing or counter is wrong and the run cannot say which |
| C3 | claim | this `BLOCK_M` still cannot reach the compute roof | `alpha > BM/ridge`, exact for a COUNTER alpha; a ladder alpha must go through `ai_model.cap_from_fitted` first (retracted 2026-09-02: the bare identity is high by `(1 + phi + delta)` on a fit) | **the one result the 2026-09 evaluation did not kill.** A counter alpha below the threshold means the tile height *can* reach the roof and the cap claim must be withdrawn |

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
#    THE INSTRUMENT WILL NOT RUN ONE LAUNCH PER CELL. Until 2026-09-02 this
#    recipe passed `--warmup 0 --iters 1` so that every profiled launch
#    belonged to one fused_experts call and the per-launch bytes could be
#    summed. Retracted: `--warmup` is now MILLISECONDS of sustained load and
#    moe/bench/timing.warm_until raises TimingRefused at 0, and `--iters` is
#    retired as a timing knob (time_kernel sizes iterations from the warmup's
#    own per-call time). Every profiled cell therefore holds
#    warmup_calls + iters x trials launches of the fused kernel, and the
#    KernelTiming record beside it carries all three counts. Divide the
#    summed per-kernel-name bytes by the launch count the record names, or
#    read one launch's bytes off the ncu CSV by kernel id, before comparing
#    with the table above. Do NOT pass a smaller warmup than the default to
#    make the arithmetic easier: a cold governor is a different cell.
#    scripts/dram_counter_route.py still prints the retracted flags in its
#    own --dry-run recipe; that script owns that fix, and until it lands the
#    recipe it prints cannot run as printed.
for cc in all none; do
  for n in 1 2 3 4 6 8; do
    ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_op_read_hit_rate.pct,gpu__time_duration.sum \
        --replay-mode kernel --cache-control $cc \
        --csv --page raw --target-processes all \
        --log-file counters-n$n-cc$cc.csv \
        python scripts/block_m_crossing_sweep.py --model mixtral-8x7b \
          --tiles 32 --group-m 16 --block-n 64 \
          --r-max $(( n * 32 )) --row-step $(( n * 32 )) --step-probes 0 \
          --warmup-ms 300
  done
done

# 3. score it against the gates above
python scripts/dram_counter_route.py --analyse counters.json
```

`--dry-run` prints the JSON schema `--analyse` requires. Every key is mandatory;
a missing one raises rather than defaulting, because a zero that was never
measured is the failure this whole file exists to avoid.

### 4.7 Cost

Read the plan page, not this paragraph: `--dry-run` prices what it will run and
this section says what shape the figure is. As of 2026-09-10 the plan is five
cells and prints `5 cells x 6 tile counts x 2 cache modes = 60 profiled
invocations`, about 3300 profiled kernel launches, and `at 5 minutes per
profiled invocation that is 5.0 GPU-hours for the whole extended plan, against
1.0 for the single cell the plan used to hold`. It also prints the escape:
`DROP TO 36 INVOCATIONS (3.0 GPU-hours) by running contrast A alone`.

This section said `Budget 15 minutes of GPU time and one pod-hour end to end`
until 2026-09-10, which is 20x under the plan it documents. The fifteen minutes
was a one-launch recipe; the profiled launch count is warmup + iters x trials,
never one.

One `--run` is ONE cell at ONE cache mode, so it is 6 profiled invocations and
half a GPU-hour. The largest cell is 512 rows per expert, which the timed sweep
measures in single-digit milliseconds; `ncu` replay and its save/restore of the
2.8 GB weight buffers dominate, at seconds per launch.

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
  `--self-test`, `--run`, `--analyse`, `--contrast`. `--run` is the ncu loop
  itself and `--contrast` scores the RATIO across two payloads it wrote; both
  landed on 2026-09-10 and this list named neither, so the page described a
  planner where the file also measures and compares. Exactly one mode per
  invocation: the parser refuses two, because interleaving a plan with a result
  is what this study has been burned by
* `tests/test_dram_counter_route.py`, all off GPU. It held 108 tests when this
  line last counted them and holds more since the r3-arms family (section 6)
  landed on 2026-09-24. The count is NOT checked by `tests/test_docs.py`, so
  this line no longer states one: it read 46 while the file collected 108,
  which is what an unchecked count looks like. Re-derive it with the command
  below
* `--family r3-arms`, section 6: the same file's second registered cell
  family, which adds a census and a per-G page to `--run` and a several-page
  summary to `--analyse`, and whose child is
  `scripts/private_weight_reference.py --counter-child`

```bash
.venv/bin/python -m pytest tests/test_dram_counter_route.py -q
.venv/bin/ruff check scripts/dram_counter_route.py tests/test_dram_counter_route.py
```

---

## 6. The R3 arms under the counter

`scripts/dram_counter_route.py --family r3-arms`. Written 2026-09-24. Nothing
in this section has run on a box whose counters were read: every number below
is either arithmetic this repository computes (the dry run prints it) or a
prediction registered before the run.

### 6.1 What it measures, and why in this file

alpha(G) is the fraction of an expert's weight set each extra M-tile re-reads
from DRAM, G being Triton's GROUP_SIZE_M swizzle in vLLM 0.27.1's fused MoE
kernel, on mixtral-8x7b bf16 at BLOCK_M 32 (BLOCK_N 64, BLOCK_K 64, as R3
pins them). R3 (`scripts/private_weight_reference.py`) builds three arms over
one tread ladder of n M-tiles per expert: NATIVE (vLLM as shipped, 8
experts declared), SHARED (72 declared slots all over one copy, reuse
possible) and PRIVATE (9 copies, one per tile, no reuse possible). Session 5's
timing on the H200 could not identify alpha at G >= 4, because an on-chip
floor hides the traffic, and it brackets alpha(1) between R3's timed ratio and
1.0. A DRAM counter reads the traffic itself.

The family lives in `dram_counter_route.py` because that file already owns
the permission probe, the CSV parser and its unit tables, the gates, the
exit codes and the provenance stamp. A second script would fork the parser
and the probe. R3 is imported lazily, inside the family's functions, so the
ladder family's import graph does not change, and `--family ladder` (the
default) behaves exactly as before.

The calls are R3's own. `run_sweep`'s two closures became the module-level
`arm_inputs` and `arm_call`, and `_main`'s pinned dict became
`pinned_config`: the timed ladder and the counter child build every arm
through the same three functions.

### 6.2 The cells

Three arms x G in {64, 1, 4, 2} (in that order; 16 optional) x n in
{1, 2, 3, 4, 6}. n=1 is the identity tread, where SHARED and PRIVATE are one
call. n=3 is needed for an OLS residual and is where G=2 first pays a second
group. n=4 is where the tiles align with the groups at G=4 and G=16 and the
group model predicts a drop, which no per-tile scalar alpha can produce. n=5
is omitted: its prediction equals n=6's at G=2 and G=4.

The declaration is R3's (`declared_copies_for` over R3's whole ladder, 9
copies and 72 slots), never recomputed from the subset, because it sizes the
launch grid. The child refuses a plan whose treads are not a subset of R3's
ladder or whose declaration is not R3's. Its memory plan is R3's
`memory_plan` with no flush buffer, checked against the attached card: the
dry run prints the predicted peak and the free memory it needs.

### 6.3 The metrics

| class | metrics | rule |
|---|---|---|
| STRICT | `dram__bytes_read.sum`, `dram__bytes_write.sum`, `lts__t_sectors_srcunit_tex_op_read.sum`, `launch__grid_size`, `gpu__time_duration.sum` | the run refuses without them |
| CROSS-CHECK | `lts__d_sectors_fill_device.sum`, `lts__t_sectors_op_read_lookup_miss.sum`, `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`, `lts__t_sector_op_read_hit_rate.pct` | gated only when the probe proved them readable |
| RECORDED | the four `launch__occupancy_limit_*`, `launch__registers_per_thread`, `launch__waves_per_multiprocessor`, `lts__t_sectors_srcunit_ltcfabric.sum` | never gated, parsed soft |

`--probe --family r3-arms` asks `ncu --query-metrics` which of these the chip
offers, refuses when a STRICT one is absent, drops the others it lacks, and
asks the probe kernel for everything left. OPEN means every STRICT metric came
back as a number. The `launch__*` names are not held to the list, so one this
ncu does not know refuses the whole ask on a box whose counters work; when the
whole ask reads no counter for any reason other than `ERR_NVGPUCTRPERM`, the
probe asks again, first STRICT plus the metrics the list verified, then STRICT
alone, and records every ask (`attempts`). A metric the last ask left out is
unproven, so no page gates on it. The unit tables gained a sector table and a separate table
for the `launch__*` metrics, the only place an empty unit is accepted.

The parser now reads both CSV layouts ncu may print for `--csv --page raw`:
LONG (one row per launch and metric, which the parser was written against)
and WIDE (one row per launch, a units row under the header). No live ncu CSV
has been captured in this repository, and the recollection this design rests
on is that the raw page is wide; a parser that read only the long layout
would turn a box whose counters work into a probe that reads REFUSE. The
probe records the layout it parsed and the header's first line.

### 6.4 Profiling only the arm GEMMs, and the per-call trap

The ladder family's per-call trap exists because the timed instrument chooses
its own call count. The r3-arms family removes the instrument: the child
(`private_weight_reference.py --counter-child PLAN`) times nothing and makes
an exact, planned number of calls (`counter_schedule`): U = 2 warmups for
every cell first, which compile, then K = 3 measured calls per cell in
manifest order, each inside an NVTX range and followed by a synchronize. ncu
runs with `-k regex:^fused_moe_kernel$ --kernel-name-base function`, skips
GEMMS_PER_CALL x U x cells launches and keeps GEMMS_PER_CALL x K x cells,
exports a `.ncu-rep`, and the CSV is reduced from it afterwards with
`ncu --import ... --csv --page raw --print-units base`, so a parser defect is a
laptop fix and not a re-rent: `--run --family r3-arms --reduce-only` rebuilds a
page from the kept profiles (plan, manifest, the capture's own record, the
`.ncu-rep` or its CSV) with no card, no probe and no child, and the page names
the capture's card, stack and commit. R3's five-part buffer proof runs last,
after the profiled window has closed.

GEMMS_PER_CALL = 2 is cited from vLLM 0.27.1's `fused_experts_impl` and
measured by the census (`--run --family r3-arms --census-only`): NATIVE at
n in {1, 6}, one warmup and one call, no skip and no cap, must profile
exactly 2 x 4 launches at the grids the child derived from vLLM's own
`moe_align_block_size`. Every page needs a census from the same card UUID,
commit and vLLM version. A tree git cannot name writes neither: root running
git in a checkout its login user owns (the sudo counter door) gets "detected
dubious ownership" and no sha, so the census, the page and `--reduce-only`
refuse and name the `safe.directory` remedy, and a census that names no
commit licenses no page.

Attribution (`attribute_launches`) is exact: the CSV must hold exactly the
planned launch count, launch i is the i-th (cell, call, GEMM) of the
manifest, w1 then w2 within a call, and every launch's `launch__grid_size`
must equal the manifest's grid for its arm, tread and GEMM. Anything else
exits INVALID.

### 6.5 L2 state

`--cache-control all` flushes every cache before each replay pass of each
profiled launch, so every GEMM starts cold; the unprofiled kernels between
them run normally. The timed apparatus flushed once per call, not between the
two GEMMs, which changes only activation traffic, and the operand model
charges activations as cold reads for that reason. `--cache-control none` is
not run: under kernel replay ncu's first-pass save of the ~26 GB footprint
streams through L2 right before the kernel. `--clock-control base` is passed
and recorded, because the documented default has moved between versions.

### 6.6 The byte model and the estimators

W = 2,818,572,288 B, split by GEMM as W_w1 = 1,879,048,192 B (the gate+up slab)
and W_w2 = 939,524,096 B (`moe.bench.weights.routed_expert_weight_bytes_by_gemm`).
One more tread makes the w1 GEMM read E x BM x H x 2 = 2,097,152 B of A operand
and the w2 GEMM E x BM x F x 2 = 7,340,032 B
(`gemm_operand_read_bytes_per_row`), 0.33% of W together.

q(n) = (R(n) - n x operand) / W, per GEMM and in total, R being
`dram__bytes_read.sum` per `fused_experts` call. The primary product is
q_SHARED(n) at each tread beside the group model. alpha(G) is the OLS slope of
q_SHARED over n, printed with its residual as a scalar summary that means
something only where the ladder is affine. Also printed: the byte ratio
slope(R_S) / slope(R_P), the analogue of R3's timed ratio; 1 - (slope_P -
slope_S) / W, which cancels any activation term the arms share; and per-GEMM
alphas. None uses a bandwidth, a ridge, an intercept or a calibration.

### 6.7 The registered prediction

`group_reads(E, n, G)`: vLLM's pid mapping puts expert e's tiles at
[e n, e n + n) in every arm, and if L2 serves every re-read inside a
GROUP_SIZE_M group and none across groups, each weight slab is read once per
group its expert's tiles fall in. Card-free. `--self-test` and the test suite
hold the closed form to a brute-force walk of the pid mapping.

| G | n=1 | n=2 | n=3 | n=4 | n=6 | slope |
|---|---|---|---|---|---|---|
| 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 2.0000 | 3.0000 | 4.0000 | 6.0000 | 1.0000 |
| 4 | 1.0000 | 1.0000 | 1.5000 | 1.0000 | 2.0000 | 0.1824 |
| 2 | 1.0000 | 1.0000 | 2.0000 | 2.0000 | 3.0000 | 0.4189 |
| 16 | 1.0000 | 1.0000 | 1.1250 | 1.0000 | 1.2500 | 0.0456 |

PRIVATE reads q(n) = n at every G by construction. The model binds w1 at every
G >= 2 (448 N-tiles per M-row, far more than the CTAs in flight). w2 has 64
N-tiles per M-row and may read less where the co-residency window (SMs x CTAs
per SM, from the page's own occupancy) exceeds G x 64. At G=1 two readings are
registered: co-residency (w1 re-read whole, w2 shared, so alpha(1) can sit
well below the timed bracket) and the timed edge (both arms at one rate, so
alpha(1) sits inside it).

### 6.8 The gates

VALIDITY, any failure exits INVALID and no alpha may be quoted: V0 a live card
block; V1 exact count, every grid, a census that measured GEMMS_PER_CALL; V2
every STRICT metric a number; V3 each cell's K calls within 1% of each other;
V4 at n=1 SHARED and PRIVATE agree and every arm's q(1) is in [0.97, 1.03];
V5 PRIVATE reads between 0.97 n and 1.5 n at every tread and GEMM; V6 the three
arms request the same L2 sectors within 0.5%; V7 NATIVE reads what SHARED reads
within 1%; V8 DRAM bytes agree with 32 x the L2 fill sectors within 2% (asked
only if proven); V9 R3's five-part buffer proof.

The ladder family's monotone and affine gates are NOT applied to SHARED: the
group model predicts non-monotone, non-affine shared ladders at G = 2, 4 and
16, and those gates would void a correct page. A test holds that a planted G=4
staircase with its n=4 drop is VALID.

CLAIM, a failure is a result: C1 w1 within 5% of the group model at every n
for G >= 2, and at G=1 w1 re-read whole where the co-residency window is below
448 (not asked when the occupancy was not proven); C2 w2 never above 1.05 x the
model; C3 at G=1 alpha_w2 < alpha_w1; C6 PRIVATE no more than 1.03 n, whose
failure localises a bytes share of the private arm's timed G-cost by GEMM;
C5 only with `--timed-reference`, labelled cross-card, comparing alpha(1) with
the timed bracket and the G >= 4 byte ratios with the timed ratios, every timed
number read from R3's report.json files and none typed.

### 6.9 The card

Every page's first line reads `CARD <name> (<slug>, UUID <uuid>, sm_<cc>, <SMs>
SMs, <L2> MiB L2): every number here is THIS card's; the study's timing pages
are nvidia_h200.` The run id carries the live slug, `--analyse` refuses to join
pages from two UUIDs, two commits, two vLLM versions or two designs, or pages
that name no commit, and a page without a card block fails V0. The target is 1x H100 SXM5 (132 SMs, 50 MB L2);
an A100 40 GB shake-out is possible first. Neither is the study's H200 and no
alpha either prints is the H200's.

### 6.10 The commands and the cost

```bash
python scripts/dram_counter_route.py --dry-run --family r3-arms          # off GPU: the plan and its price
python scripts/dram_counter_route.py --probe --family r3-arms --out $S/probe.json
$PY_VLLM scripts/dram_counter_route.py --run --family r3-arms --census-only --out $S/census.json
$PY_VLLM scripts/dram_counter_route.py --run --family r3-arms --group-m 64 --census $S/census.json --out $R/r3c-g64.json
#   then --group-m 1, 4, 2, and optionally 16
python scripts/dram_counter_route.py --analyse $R/r3c-g64.json $R/r3c-g1.json $R/r3c-g4.json $R/r3c-g2.json
```

The dry run prices each G from registered constants (child start, weight
build, compiles, 90 profiled launches at a per-launch overhead, proof and
reduction), about 4 minutes per G on 1x H100 SXM5 and about 20 minutes for all
five, plus a few minutes of preflight; it also prints the pessimistic figure
and the recommended VM booking. On an A100 40 GB, ncu's first-pass save goes to
host memory and costs about a second per launch.

### 6.11 What the counter cannot settle

It cannot split weight re-reads from activation re-reads inside one GEMM's
DRAM total; the group model and the private control bound it. It measures bytes
at a locked clock with a cold L2 per GEMM, not the timed apparatus's per-call
flush, which leaves weights unaffected and moves activations slightly. The
group model is derived from vLLM's pid mapping, not measured: a C1 failure is
a result, and the brute-force test guards the arithmetic, not the hardware.
