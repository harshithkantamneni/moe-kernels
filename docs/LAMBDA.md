# Lambda runbook: the R3 arms under a DRAM counter

A Lambda Cloud instance is a plain Ubuntu VM with root, Lambda Stack and a
driver nobody knows until it boots. Its local disk survives a reboot and is
**lost at termination** (Lambda's docs: terminating permanently removes the
instance), and Lambda instances can only be launched, restarted or terminated
(there is no stop). Lambda also offers filesystems, networked persistent
storage attached only when an instance is created (`file_system_names` in the
launch call, in the instance's region). **This runbook launches without
one**, so the local disk is all there is and section 4's exfiltration is the
only copy of the results (section 3c's unattended run instead pushes its
results to its own branch of this repo after every step); attaching one at
launch would keep results past
termination, and a filesystem left behind is billed until it is deleted. The
one command that turns a fresh instance into a box that can take the
measurement is `scripts/setup_vm.sh`; this page is everything around it.

**Each Lambda card is one of the study's four cards, and keeps its own
numbers.** Lambda has no H200. The primary card is the GH200 480GB
(`gpu_1x_gh200`: the GH100 die with 132 SMs, sm_90 and the H200's 60 MiB of
L2, its memory bus 6144 bits by NVML, on a Grace arm64 host; every pin in
`requirements/resolved-{base,vllm}.txt` has a manylinux aarch64 wheel). The
H100 SXM5 (`gpu_1x_h100_sxm5`: 80 GB HBM3, 132 SMs, 50 MiB L2, sm_90) is the
second Hopper, with less L2. The A100 SXM4 40 GB (`gpu_1x_a100_sxm4`: 108
SMs, 40 MiB L2, sm_80) is the non-Hopper control, for bytes only. The fourth
card, `nvidia_h200` (60 MiB L2, HBM3e), is timed on RunPod, where every pod
asked for counters so far refused them and `nvidia-smi -lgc` is refused (a
record of the pods tried, not a fact about the platform; the chain's probe
asks on every pod). An alpha measured on any card is that card's and
is never averaged with another card's, and every page the preflight and the
counter run write opens with a `CARD` line naming the card it measured.

Counters are readable on a Lambda VM through the admin door: the preflight's
PF5 read them on the A100, the GH200 and the H100 (all on 2026-09-25 UTC),
each after its driver moved to r580. PF5 still decides it on every new box,
by reading a counter, and nothing here passes a box on which no counter came
back.

## 0. On the laptop, before renting

The counter run needs `dram_counter_route.py --family r3-arms` (the ncu-route
work) merged with this setup. Check both are on the branch you bundle, then:

```bash
.venv/bin/python -m pytest -q tests/test_setup_vm.py      # the setup, off GPU
python scripts/dram_counter_route.py --dry-run --family r3-arms   # the plan and its prices
bash scripts/setup_vm.sh --commit "$(git rev-parse r3-align)" \
  --repo "$(git remote get-url origin)" --dry-run          # REFUSES here: no GPU. Read the plan.
git bundle create moe.bundle r3-align                      # the repo, no credentials needed on the VM
SHA="$(git rev-parse r3-align)"                            # the full sha setup_vm.sh checks out
```

A bundle needs no GitHub credentials on a rented VM. `--repo <url>` works
instead when the VM can clone.

## 1. Launch

The API key is created in the Lambda Cloud console. The SSH key registered
there for this laptop is `harshith-macbook`.

```bash
export LAMBDA_API_KEY=...                                  # never commit it
API=https://cloud.lambda.ai/api/v1
AUTH=(-H "Authorization: Bearer $LAMBDA_API_KEY")

# what is in stock, where, and at what price
curl -s "${AUTH[@]}" $API/instance-types | jq -r '.data | to_entries[]
  | select(.key == "gpu_1x_h100_sxm5" or .key == "gpu_1x_a100_sxm4" or (.key | test("gh200")))
  | "\(.key)  \(.value.instance_type.price_cents_per_hour / 100) USD/h  "
    + ([.value.regions_with_capacity_available[].name] | join(","))'

# launch one in a region the listing named
curl -s "${AUTH[@]}" -H 'Content-Type: application/json' $API/instance-operations/launch \
  -d '{"region_name": "<region>", "instance_type_name": "gpu_1x_h100_sxm5",
       "ssh_key_names": ["harshith-macbook"], "quantity": 1, "name": "moe-r3-counters"}'
# -> {"data": {"instance_ids": ["<id>"]}}

# poll until active, and read the address
curl -s "${AUTH[@]}" $API/instances/<id> | jq -r '.data | "\(.status) \(.ip)"'
IP=<ip>
```

For the shake-out, the same call with `"instance_type_name": "gpu_1x_a100_sxm4"`.
No `file_system_names` is passed, by choice (the page's opening paragraph):
add `"file_system_names": ["<name>"]` to the launch body, for a filesystem
already created in the same region, to keep a copy past termination.
Write the instance id down: the terminate call in section 5 needs it.

## 2. Setup

```bash
scp moe.bundle scripts/setup_vm.sh ubuntu@$IP:
ssh ubuntu@$IP
tmux new -s setup          # the venv build takes ~25 minutes; a dropped ssh must not kill it
bash setup_vm.sh --bundle moe.bundle --commit <SHA> 2>&1 | tee setup_vm.log
```

What it does, stage by stage, is in the script's header. In short: it detects
the box (S0), refuses a box whose nvidia-smi cannot reach the driver (a
`Driver/library version mismatch` after an unattended userspace upgrade; a
reboot loads the matching module) and a driver below r580 (see below),
installs git/curl/gcc if missing (S3), clones the bundle at exactly `<SHA>`
into `~/moe/repo` (S1), then `exec`s that checkout's own `setup_vm.sh`, which
picks the torch wheel index from the driver (S2: r580+ cu130), builds the base
and vLLM venvs through `scripts/setup_runpod.sh` on Python 3.12 (S4), finds or
installs ncu (S5), picks the counter door (S6), writes `~/moe/env.sh` (S7),
and runs the preflight (S8). Nothing is placed under `/workspace`.

Exit codes are the repo's table: **0** READY (a counter was read), **1** the
preflight did not pass (`~/moe/session/PREFLIGHT.txt` names each check), **2**
refused before anything was spent, **4** a step crashed. It is idempotent: a
re-run skips the clone, the venvs (on setup_runpod.sh's stamps) and ncu, and
always re-runs the preflight. `--check` verifies without installing, and
`--preflight-only` re-asks the box (after a restart, say).

Three things it will refuse rather than do:

- **Move the driver.** ncu is installed from the CUDA apt repo only after
  `apt-get -s` shows the transaction installs, removes or purges no package
  of the driver's families: any `nvidia-*`, `libnvidia-*` or `libcuda*`,
  `cuda-drivers`, `cuda-compat`, the `linux-{modules,objects,signatures}-nvidia`
  kernel packages or `xserver-xorg-video-nvidia` (only the Nsight Compute
  packages themselves are exempt). A userspace package such as
  `nvidia-utils-580` moving under the loaded module is how a VM ends at
  `Driver/library version mismatch`. A driver below
  r580 cannot run the vLLM venv's cu13 torch, and the fix is a driver upgrade
  this script never performs, so such a box is refused at S2 before anything
  is built: rent another instance, or upgrade the driver by hand first. Every
  Lambda instance on record (the A100, the GH200 and the H100, 2026-09-25)
  shipped 570.148.08, and each was upgraded, by the owner's decision, to
  `nvidia-driver-580-server-open` 580.105.08 from Lambda's own repository,
  then rebooted, before this script ran. `--torch-index cu128` on r570-r579 builds
  anyway, by request, and there PF2 fails for the vLLM venv, because a cu128
  vLLM path is untested.
- **Measure from a moved or dirty checkout.** `~/moe/repo` at another sha, or
  with local changes, is refused. `--fresh` re-resolves the requirement sets
  and so dirties the checkout on purpose; copy the new `resolved-*.txt` back
  and commit them.
- **Open the counters by reloading the driver unasked.** The default door is
  `sudo` (the counter steps run as root through `$MOE_COUNTER_LAUNCHER`, and
  the results are chowned back). `--counter-door module` writes
  `/etc/modprobe.d/moe-ncu-profiling.conf` and reloads the nvidia modules, and
  is refused while any process holds the GPU. With
  `RestrictProfilingToAdminUsers=0` already, the door is `open`.

**The preflight** (`scripts/vm_preflight.py`, never cached) writes
`PREFLIGHT.txt` and `PREFLIGHT.json` and exits 0 only when all seven pass:

| check | passes when |
|---|---|
| PF1 card | nvidia-smi and torch in both venvs name one card (name, UUID); records SMs, L2, capability |
| PF2 wheel | each venv's `torch.version.cuda` major <= the driver's CUDA major, and a kernel launches |
| PF3 stack | torch, triton and vllm are the versions `requirements/resolved-*.txt` pin; `fused_experts` imports |
| PF4 metrics | `ncu --query-metrics` lists every STRICT metric of the r3-arms family; absent recorded ones are listed as dropped |
| PF5 probe | `dram_counter_route.py --probe --family r3-arms` under the door reads OPEN, exit 0, and its per-metric record holds every STRICT metric as a number: a counter was READ, and so was every metric the run refuses without |
| PF6 census | `--run --family r3-arms --census-only` passes (two GEMMs per call, grids, memory plan) |
| PF7 RAM | when the card cannot hold ncu's first-pass save beside R3's allocation, host RAM holds 1.5x it |

PF7 matters on the A100 40 GB: R3's allocation (about 25.8 GB, R3's own
`memory_plan` with no flush buffer) leaves less free on the card than ncu's
save of it needs, so the save goes to host memory and each launch costs about a
second more.

## 3. Run

Only after `READY`. Every counter step goes through `moe_counter`, which
`env.sh` defines: the launcher the door chose, then the results handed back to
the login user. Order and flags are dram_counter_route.py's r3-arms family's;
its `--dry-run --family r3-arms` prints them with their prices.

```bash
. ~/moe/env.sh && cd "$REPO"
S=$SESSION_ROOT
R=$RESULTS_ROOT/$(date -u +%F)-$MOE_CARD-r3-counters && mkdir -p "$R"
for G in 64 1 4 2; do                                      # then 16 if the budget holds
  moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms \
    --group-m $G --census "$S/census.json" --out "$R/r3c-g$G.json" 2>&1 | tee "$S/logs/r3c-g$G.log"
done
python3 scripts/dram_counter_route.py --analyse "$R"/r3c-g{1,2,4,64}.json --out "$R/summary.json"
moe_counter ncu --clock-control reset                      # clear a clock lock a killed ncu left
nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem --format=csv | tee "$S/clocks-after-counters.txt"
```

The last two lines are for whatever runs on this card next. ncu locks the GPC
and memory clocks while it profiles (the run passes `--clock-control base`),
and so do the preflight's PF5 probe and PF6 census; a normal exit restores
them, but an ncu killed at a cap can leave them locked, and ncu's own remedy is
`--clock-control reset`. Nothing measured after it would say so: a timed page
run under a left-over lock reads that clock as the card's.

Each page keeps its `.ncu-rep` beside it (`<out>.profiles/`), so a parser
defect found later is a laptop fix, not a re-rent. The first G's measured
per-launch time re-prices the rest.

**Optional, after the pages: the floor counters** (COUNTERS.md 6.12). Two
captures at the base clock, then one G=64 capture under an nvidia-smi lock at
F. Run it in the same shell as the block above (it uses `$S`, `$R` and
`moe_counter`), inside tmux, and only while nothing else uses the GPU.

- **Census.** The floor needs a bundle whose commit carries `--floor` and a
  census at that commit. `C` is the pages' census when the pages ran at that
  commit. Otherwise write a new one to its own path (`--census-only --out
  "$S/census-floor.json"`), so the pages' census stays.
- **F.** A lock this card holds at every G: 1710 MHz held in R3's timed runs on
  both the GH200 and the H100 (2026-09-25, published on branches lambda-gh200
  and lambda-h100). Another card needs its own hold test first, and an F above
  the card's maximum clock (the A100's is 1410 MHz) stops the block before the
  lock.
- **Stops.** The block is one subshell and stops at the first capture that does
  not exit 0: 2 is refused, 3 is INVALID, any other code is the shell or `tee`
  (read the screen). After a stop, fix the cause and rerun only the step that
  stopped: each capture deletes its own `--out` first, so rerunning the whole
  block would delete the files that passed.
- **Locks.** Its trap clears both kinds of lock however it ends, Ctrl-C
  included, and ignores a second Ctrl-C while it does: ncu's own
  (`--clock-control reset`, for clocks a killed ncu can leave locked) and
  nvidia-smi's (`-rgc`). The last lines record the clocks a few seconds after
  the reset, for whatever runs next.

```bash
F=1710                    # a lock this card holds at every G (GH200 and H100: 1710)
C=$S/census.json          # the census at this commit (see Census above)
( set -o pipefail
  trap 'trap "" INT TERM HUP; moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null' EXIT
  trap 'exit 130' INT TERM HUP                             # stop the block; EXIT still resets
  for G in 64 2; do
    moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --group-m $G \
      --census "$C" --floor --out "$R/r3f-g$G.json" 2>&1 | tee "$S/logs/r3f-g$G.log" \
      || exit                                              # 2 refused, 3 INVALID: stop, read it
  done
  moe_counter ncu --clock-control reset                    # a clock a killed ncu left, before the lock
  MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
  [ "$F" -le "$MAX" ] || { echo "F=$F MHz is above this card's maximum, $MAX MHz"; exit 2; }
  sudo -n nvidia-smi -lgc "$F,$F" || exit 2
  moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --group-m 64 \
    --census "$C" --floor --floor-clock none --floor-lock-mhz "$F" \
    --out "$R/r3f-g64-lock$F.json" 2>&1 | tee "$S/logs/r3f-g64-lock$F.log" )
echo "floor block exit $?"; sleep 5
nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-floor.txt"
```

**Optional: the byte pages again, at the timed lock** (COUNTERS.md 6.13). The
pages above count bytes at ncu's base clock; this card's timed R3 pages ran
under an nvidia-smi lock (the GH200's at 1710 MHz). This block takes a census
and the four pages again under that lock (`--page-clock none --page-lock-mhz
F`), so bytes and time are compared at one clock, and gate V10 makes a page
INVALID when any of its cells ran off F. Same shell as the blocks above,
inside tmux, and only while nothing else uses the GPU.

- **Commit.** The block needs a bundle whose commit carries `--page-clock`.
  Its census is its own (`$S/census-lock$F.json`) and its pages go to their
  own directory (`$R/lock$F`): each capture deletes its own `--out` first, so
  nothing the base-clock pages wrote is touched.
- **F.** The lock the timed pages ran at, and one this card holds at every
  G: 1710 MHz on the GH200 and the H100 (2026-09-25). An F above the card's
  maximum clock stops the block before the lock.
- **Stops.** One subshell that stops at the first capture that does not exit
  0: 2 is refused, 3 is INVALID (a V10 failure is INVALID: the page is
  written and names its off cells), any other code is the shell or `tee`.
  After a stop, fix the cause and rerun the block with only the steps that
  did not pass (the census line out once it has passed): each capture
  deletes its own `--out` first, and the trap has cleared the lock, which
  the block sets again.
- **Locks.** Its trap clears both kinds however it ends, Ctrl-C included, and
  ignores a second Ctrl-C while it does: ncu's own (`--clock-control reset`)
  and nvidia-smi's (`-rgc`). Inside the lock no ncu sets a clock: each
  capture, and the probe each capture runs first, passes `--clock-control
  none`. The last lines record the clocks after the reset, then score the
  pages.
- **TIMED.** The timed reports C5 scores the pages against: this card's
  VALID R3 report.json files timed at F, space-separated, set on the block's
  second line before it is pasted. The GH200's of 2026-09-25 are the five
  under `results/published/2026-09-25-nvidia_gh200_480gb-session/results/gaps-nvidia_gh200_480gb/private_weight_reference/`
  whose `session_tag` ends `-lock1710`. C5 refuses another card's report, an
  INVALID one, and one named twice. Left empty, the pages are still scored
  and C5 is not asked.

```bash
F=1710                    # the lock this card's timed pages ran at (GH200 and H100: 1710)
TIMED=""                  # this card's VALID R3 report.json files timed at F (see TIMED above)
L=$R/lock$F && mkdir -p "$L"
( set -o pipefail
  trap 'trap "" INT TERM HUP; moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null' EXIT
  trap 'exit 130' INT TERM HUP                             # stop the block; EXIT still resets
  moe_counter ncu --clock-control reset                    # a clock a killed ncu left, before the lock
  MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
  [ "$F" -le "$MAX" ] || { echo "F=$F MHz is above this card's maximum, $MAX MHz"; exit 2; }
  sudo -n nvidia-smi -lgc "$F,$F" || exit 2
  moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --census-only \
    --page-clock none --page-lock-mhz "$F" --out "$S/census-lock$F.json" 2>&1 \
    | tee "$S/logs/census-lock$F.log" || exit              # 2 refused, 3 INVALID: stop, read it
  for G in 64 1 4 2; do
    moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --group-m $G \
      --census "$S/census-lock$F.json" --page-clock none --page-lock-mhz "$F" \
      --out "$L/r3c-g$G.json" 2>&1 | tee "$S/logs/r3c-g$G-lock$F.log" || exit
  done )
echo "lock block exit $?"; sleep 5
nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-lock-pages.txt"
python3 scripts/dram_counter_route.py --analyse "$L"/r3c-g{1,2,4,64}.json \
  ${TIMED:+--timed-reference $TIMED} --out "$L/summary.json"   # bash splits $TIMED
```

Do not mix doors on one box: ncu's lock file under `/tmp` is created by the
first user that runs it, and a root-owned one refuses a later unprivileged ncu.
The reverse can refuse too: with Ubuntu's `fs.protected_regular`, root's
open-for-create of a user-owned file in sticky `/tmp` is denied. So
`setup_vm.sh` runs no ncu before S6 has chosen the door, and runs its own
(`--version` and `--list-chips`, into `~/moe/session/ncu.txt`) through the
door's launcher, like every ncu after it. When ncu's output names its lock
file, PF4 fails with that line as the cause: remove the file it names and run
every ncu through `moe_counter`.

**When PF4 says ncu listed no metric for this card**, the ncu found first
(PATH is searched before the CUDA directories, as the alpha(G) chain searches)
may predate the card or the driver. `~/moe/session/ncu.txt` records which one
was used and its version. Install the package S2 named (`sudo apt-get install
--no-install-recommends cuda-nsight-compute-13-0` on r580+, after the same
`apt-get -s` check), put its directory first in `~/moe/env.sh`'s `PATH` line,
and re-run `bash ~/moe/repo/scripts/setup_vm.sh --commit <SHA> --preflight-only`.

## 3b. The timing chain on the same card

The counter pages give each arm's bytes; `scripts/alpha_g_chain.sh` times the
same arms (thermal gate, calibrate, the probe check, R3 at duty 0.25, R1 at
duty 1.0, 0.5 and 0.25). Run it on the same instance, AFTER section 3: its
calibrate writes `moe/bench/hardware/measured_<card>.yaml` into the checkout,
every page measured after that is stamped `git_dirty`, and `setup_vm.sh
--check` and `--preflight-only` refuse the checkout from then on.

**No locked clock may be in force when it starts.** The chain moves the SM
clock by duty cycle against the card's power limit and never locks it
(`scripts/clock_elasticity.py`, WHY DUTY CYCLE), and nothing in it notices a
lock: the thermal gate passes a locked card, calibrate takes the locked clock
as its reference, V7 passes when both ratio arms share one locked clock, and
only R1's V1 fails. Two things set a lock here. ncu locks the GPC and memory
clocks while it profiles (section 3's counter run passes `--clock-control
base`), and a killed ncu can leave them locked; ncu's own remedy is
`--clock-control reset`. `nvidia-smi -lgc` sets one on purpose (R3 run by hand
under a lock, below). Section 3's block ends with the first reset; clear both
again here, through the door as every ncu here, then record the state:

```bash
. ~/moe/env.sh && cd "$REPO"
moe_counter ncu --clock-control reset    # a lock a killed ncu left: GPC and memory clocks
sudo nvidia-smi -rgc                     # a lock nvidia-smi -lgc set; harmless when none is set
nvidia-smi -q -d PERFORMANCE,POWER,CLOCK > "$SESSION_ROOT/clocks-before-chain.txt"
nvidia-smi --query-gpu=name,persistence_mode,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.sm,clocks.max.sm --format=csv
```

The chain's own counter probe resets the same way after it runs, unless it
read BLOCKED (refused the counters, it could not profile and so locked
nothing): its ledger note ends with what the reset did and the clocks after
it, or with "no clock reset" and why, and `chain-logs/clock-reset.log` keeps
every pass's. A note that says `CLOCKS NOT RESET` means a lock the probe may
have left still stands: stop the chain, run the two reset lines above, and
`--resume` (the probe asks again on every pass).

**Whether R1 can read on this card is unmeasured.** R1's three duty states
separate in clock only if duty 1.0 holds the card on its power limit: the
H200 drew 694-695 W of its 700 W there. Nothing shows that an H100 or a GH200
reaches its limit on this kernel; the H100's HBM3 is slower than the H200's
HBM3e and may draw less. The thermal gate cannot settle it. Its load is a
dense bf16 GEMM, which draws more than this memory-bound kernel, so a GEMM
well under the limit shows R1 will fail, and a GEMM at the limit shows
nothing. Decide on the first `r1-g1` page, which comes after every G's
seed-0 R3 page (about an hour into the arms at five G): its duty-1.0 median
board power against `power.limit`, and its V1. V1 INVALID with the board well
under its limit means every G's R1 reads `withheld:INVALID`; R3 is not
affected, and R1 gates nothing. Lowering the limit (`sudo nvidia-smi -pl <W>`,
inside `power.min_limit` to `power.max_limit`) changes the card calibrate
measured, so it starts a new session: `--new`, which calibrates again. On a
GH200 the limit has two scopes, the GPU's (`-pl <W> -sc 0`) and the whole
module's, shared with the Grace CPU (`-pl <W> -sc 1`), and the lower of the
two binds: read both in `nvidia-smi -q -d POWER` before setting either. On a
VM there is also R1's lock mode, built for this case: `scripts/clock_elasticity.py
--lock-clocks F1 F2 F3` (root only), run by hand outside the chain, makes each
state a locked SM clock, so R1 no longer needs duty 1.0 to reach the power
limit. Choose locks the card holds at every G (1710 MHz held on both Lambda
Hoppers; lower locks were not tried): V3 needs every planned lock, and a lock
whose kept rows read more than one 15 MHz step off it is named on the page.
The chain itself stays in duty mode.

Then the plan, and the run, detached and appending, with the counter run's five
G so every timed G has its bytes. The session records neither `G_LADDER` nor
`SEEDS`: give the same values to every `--resume`.

```bash
G_LADDER="1 2 4 16 64" RATE_USD_H=<the listing's rate> \
  bash scripts/alpha_g_chain.sh --dry-run | tee "$WORKSPACE/alpha_g_dry.out"
G_LADDER="1 2 4 16 64" nohup setsid bash scripts/alpha_g_chain.sh \
  >> "$WORKSPACE/alpha_g_chain.out" 2>&1 < /dev/null &
tail -f "$WORKSPACE/alpha_g_chain.out"      # and $SESSION_ROOT/alpha_g-<card>-<stamp>/CHAIN.tsv
```

**Ignore the dry run's RunPod lines.** Its footer ends "on the pod, first:"
with `git -C /workspace/moe-kernels checkout -- ...measured_nvidia_h200.yaml`
and `checkout -B r3-align origin/r3-align`, then a launch from `/workspace`.
None of them applies on a VM: the checkout is the bundle's, at `--commit`, and
`setup_vm.sh --check` refuses it once moved. The launch above replaces theirs.

`env.sh` sets no `MOE_RESULTS_DIR`, and unsets one an older `env.sh` exported,
because the chain and its driver refuse one that does not name the card: the
runs land in `$RESULTS_ROOT/gaps-<card>` and the session in `$SESSION_ROOT`.
An arm run by hand, outside the chain, takes `export
MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD` first, so its pages land beside
the chain's. The chain's counter probe runs through `$MOE_COUNTER_LAUNCHER`, as
every ncu here does. The chain's last line prints a `tar czf
"$WORKSPACE/exfil-alpha_g-<card>.tar.gz" ...` holding the session, the runs,
the new ruler yaml, calibrate's run directory and the console: run it before
section 4.

**The record: a Lambda GH200, 2026-09-25.** The chain ran on
`nvidia_gh200_480gb` at `G_LADDER="1 2 4 16 64" SEEDS=0`, from f49a213, whose
`env.sh` still exported `MOE_RESULTS_DIR`: it ran only after an `unset
MOE_RESULTS_DIR` typed after sourcing. It ran at the card's default 900 W GPU
limit (the chain's header, `thermal.log`, `calibrate.log`); the module's limit,
shared with the Grace CPU, read its default 1000 W at 07:50. The thermal gate's
dense GEMM was power-capped (`SwPowerCap`) at a median 652 W and 1440 MHz;
calibrate's bf16 GEMM held 1455 MHz. The chain's preflight, the preconditions
(thermal, calibrate), tests/test_gpu.py and the probe check were DONE. The
counter probe read BLOCKED (`ERR_NVGPUCTRPERM`, ncu 2025.3.1 at
`/usr/local/cuda-13.0/bin/ncu`): at f49a213 it ran as the login user, not
through the door, which is why it now runs through `$MOE_COUNTER_LAUNCHER`. R3
at duty 0.25 then read INVALID at G = 1, 2 and 4. V7 failed at each: the shared
and private arms' clocks were 1.55%, 3.15% and 1.96% apart at their worst
tread. V0 failed too at G=2 and G=4: at G=2 24.7% of the cells drifted, over
the gate's 20%; at G=4 17.3% did; and at both the drifting cells, dropped, left
one arm only 2 repeats at a tread against a floor of 3. This card's power
management does not hold the clock still under the duty design. The chain was
stopped at 07:50, as `r3-g16-s0` began (its log is cut short and it has no
row). R3 was then run by hand at every G (`scripts/private_weight_reference.py`
with the chain's arguments, `--duty 0.25 --seed 0`, session tag
`alpha_g-nvidia_gh200_480gb-20260925T071107Z-lock1710`,
`MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD`) with the SM clock locked by
`sudo nvidia-smi -lgc 1710,1710` and reset with `-rgc` after it. By then the
GPU limit read 700 W (requested; `session/locked-r3/clocks-locked.txt` at
07:50:35), so the 1965 MHz lock (it read 1830 MHz in the second it was set,
and its cells ran at 1785 to 1830; the ledger blames the power cap, but no
file settles the cause) and the 1710 MHz pages all ran at 700 W. Who set that
limit, and when between calibrate and 07:50, is not recorded: whichever of the
chain's three R3 pages ran after it ran at a limit calibrate never measured.
The 1710 MHz lock held: V7 read 0.00% at all five G, and V0's drift was 1.9 to
6.8%. A lock is not one of R3's design keys, so only the session tag says a
page was locked; the chain has no locked-clock mode, and those pages are
outside its ledger and PAIRS.tsv. This card's ruler,
`measured_nvidia_gh200_480gb.yaml`, carries `memory_bus_bits_table: 6016`
beside NVML's 6144: calibrate's name match read "GH200" as an H200, the defect
its whole-word match now fixes.

**And a Lambda H100 80GB HBM3, 2026-09-25.** The chain ran on
`nvidia_h100_80gb_hbm3` at G=1 only, from f49a213: thermal DONE at a median
699 W of its 700 W limit and 1380 MHz, calibrate, the pin probe, the GPU tests
and the probe check DONE, and the counter probe BLOCKED again as the login
user (`ERR_NVGPUCTRPERM`). Its duty-0.25 `r3-g1-s0` page held where the
GH200's did not: every cell at 1980 MHz, V0 to V8 PASS (C1 CLAIM_FAIL at
0.9516, NO-REUSE). The chain was stopped as R1 began, and R3 then ran by hand
under SM clock locks, of which 1710 MHz held at every G. Both cards' sessions
are published on branches lambda-gh200 and lambda-h100.

**Locked R3 is `scripts/locked_r3.py` (2026-09-26).** Run it with the VM's
`python3` after the chain has stopped, never beside it: nothing in the chain
notices a lock. Before each lock it waits up to `--idle-cap-s` for
`nvidia-smi --query-compute-apps` to list nothing, and it never locks a card
another process holds. For each lock in `--locks`, top down, it runs
`-lgc F,F` and then R3 at every G under the tag `<session>-lock<F>`, with the
chain's R3 arguments unless others follow `--`. A timed cell more than 15 MHz under F
stops that lock, and the next lock down starts again at the first G. It takes
each attempt's run directory only from that attempt's own R3 plan (its
`session` and `WRITES TO` lines), never by modification time. Finding it by
time is what left the H100's 1890 MHz lock untested
(`results/published/2026-09-25-nvidia_h100_80gb_hbm3-session/session/README.md`,
"The 1890 attempt measured nothing"). Every lock must be a supported graphics
clock. `-rgc` runs on every exit a process can catch, so stop it early with
`pkill -TERM -f '[l]ocked_r3.py'` (R3 stops, then the clock resets); after a
`kill -9`, run `sudo -n nvidia-smi -rgc` by hand. The ledger (`status`, one
UTC line per event) and `summary.json` are in
`$SESSION_ROOT/locked-r3/<session>/`. Exit 0 means one lock held at every G, 1
every lock slipped, 3 the ladder stopped short after measuring, 2 nothing was
measured (a busy card or an R3 refusal before the first timed cell included),
and 4 a crash or a card left locked.

```bash
. ~/moe/env.sh && cd "$REPO" && export MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD
T=$(basename "$(ls -td "$SESSION_ROOT"/alpha_g-"$MOE_CARD"-*/ | head -1)")   # the chain's session
python3 scripts/locked_r3.py --session-tag "$T" --locks 1980 1890 1800 1710 --dry-run
nohup setsid python3 scripts/locked_r3.py --session-tag "$T" --locks 1980 1890 1800 1710 \
  >> "$WORKSPACE/locked_r3.out" 2>&1 < /dev/null &
```

## 3c. The GH200 model-test session, unattended

This session tests the two NOT-FINAL models of docs/COUNTERS.md 6.7 on a
GH200 (`gpu_1x_gh200`): the wave-split byte model (`scripts/wave_split_bytes.py`)
and the timing model LOWM-1/2 (`scripts/r3_timing_model.py`, predictions P1 to
P8). It runs in place of 3 and 3b on this card. The duty chain cannot hold the
GH200's clock still (3b, the record of 2026-09-25: V7 and V0 failed under
duty), so every timed page here runs under an nvidia-smi lock through
`scripts/locked_r3.py`. The owner decided four things on 2026-09-26: byte
pages count every tread from 1 to 9; calibrate runs on the VM after the byte
pages; the GPU power limit is set to 700 W at the start (`-sc 0`), the limit
the 2026-09-25 lock pages ran at; and a VM that ships r570 gets the same
upgrade as on 2026-09-25 (section 2's driver note) before `setup_vm.sh`.

**Nobody pastes anything on the VM.** `scripts/gh200_model_session.sh` runs
the session unattended, and after every step `scripts/vm_results_push.sh`
pushes the VM's results and session to the run's own branch of this repo, so
the laptop reads what was measured with `git fetch`, minutes after it was
measured, and section 4's copy at the end is not needed. From the laptop,
after launching one GH200 (section 1):

```bash
RUN=gh200-$(date -u +%Y%m%dT%H%MZ); BR=run-gh200-$(date -u +%F)
DEADLINE=$(( $(date +%s) + 400 * 60 ))       # launch + 400 min: the driver ends by then, for a 7 h stop
bash scripts/vm_run.sh prepare --ip "$IP" --run-id "$RUN" --branch "$BR"
bash scripts/vm_run.sh start   --ip "$IP" --run-id "$RUN" --deadline "$DEADLINE"
bash scripts/vm_run.sh watch   --run-id "$RUN"    # every ~10 min: exit 0 once DRIVER-DONE is pushed
bash scripts/vm_run.sh verify  --run-id "$RUN"    # every pushed file against the VM's SHA256SUMS
# terminate (section 5), then:
bash scripts/vm_run.sh forget  --run-id "$RUN"    # delete the run's deploy key
```

`prepare` must run from a clean checkout at a commit GitHub has, since the VM
clones that commit. It waits for ssh, stops apt's timers (nothing may upgrade
a package under the loaded module during the run), upgrades a driver below
r580 to `nvidia-driver-580-server-open` 580.105.08-0lambda0.22.04.1 (the
newest 580 build from Lambda's repository if that one is gone) and reboots,
copies the three scripts and `setup_vm.sh` to the VM, and has the VM make its
own deploy key. Only the public half leaves the VM; `gh` adds it to this repo
with write access, and the VM pushes the branch's first commit to prove the
key. `start` records the driver's plan (`--dry-run`) and starts it detached:
`setup_vm.sh --repo https://github.com/harshithkantamneni/moe-kernels
--commit <sha>`, then the checkout's own copy of the driver. The laptop's
record of the run (ip, branch, commit, deploy key id, plan, log) is under
`~/moe-kernels-exfil/runs/<run id>/`.

**The results branch** is an orphan, `run-gh200-<date>`, never main or a
model or integrate branch, committed in `~/moe/push`, a repository that shares
nothing with the measured checkout `~/moe/repo`, so a push cannot make a page
`git_dirty`. It holds `vm/results/` and `vm/session/` (copies of `~/moe/results`
and `~/moe/session`), `vm/setup_vm.log`, `vm/driver580.log` and
`vm/gh200-driver.out`, and beside them `SHA256SUMS` (the manifest of every file
under `vm/` as committed), `HELD-BACK.txt` (a file over 95 MB is committed
xz-compressed when that fits, else listed with its size and sha256 and left on
the VM, to be copied by hand before the terminate), `PUSHES.txt` (one line per
push) and, with the last push, `DRIVER-DONE` (the driver's exit). Each push is
the whole snapshot, so a push that fails is carried by the next. The driver
pushes only between steps, after both kinds of lock are reset, never while a
page is measured. GitHub's host key is pinned in `vm_results_push.sh`, not
learned on first contact. The publish step curates from this branch.

The steps run in the owner's priority order, so a run cut short has run the
earlier ones. Minutes are estimates from the 2026-09-25 logs and the tools'
own plans; each step also has a cap, past which the driver stops it:

| step | what | min | cap | answers |
|---|---|---|---|---|
| prelude (3c.1) | supported clocks, persistence, 700 W | 3 | 15 | the locks the later steps may use |
| bytes (3c.2) | byte pages at the 1710 MHz lock, G = 1 2 4 16 8 32 3 64, treads 1-9, and one base-clock control | 55 | 120 | the byte predictions (w2 at G=8 and 32, n=5, n=7, n=8); counted bytes for every timed G |
| calibrate (3c.3) | the card's ruler, with no lock in force | 8 | 30 | the ruler R3 needs |
| timed (3c.3) | timed R3 at 1710: G=8 and 32 (treads 6), G=3 (treads 8) | 37 | 90 | P2, P5 |
| eta (3c.4) | timed R3 at held locks 1410 (G=4, 2), 1500 and 1605 (G=4) | 50 | 115 | P1 |
| floor (3c.5) | floor counters: G=64 and G=2 at ncu's base clock, G=64 unlocked (a record), G=64 under the 1710 lock | 15 | 45 | P6 |
| deep (3c.6) | timed R3 at 1710 to tread 9, G=4 then G=2 | 33 | 75 | P3, P4 |
| r1lock (3c.7) | R1 in lock mode at 1710 1500 1410, G=4 then G=1 | 62 | 130 | the floor's clock exponent (eta) |

About 263 minutes of steps, plus about 30 for `setup_vm.sh` and 15 for a
driver upgrade: book 6 hours (about $13.74 at $2.29/h) and stop by 7.
`--deadline` is the epoch second by which the driver must have ended, final
push included. Before each step the driver adds the estimates of every step
still to run; while that overruns the deadline less 8 minutes it drops, in
this order, R1 at G=1, then R1 at G=4, then the 1605 lock, then the G=64 byte
page and the base-clock control, and it skips a step whose own estimate no
longer fits. No step's cap runs past the deadline.

**The rules.** Each step runs in its own process under its cap, with its own
EXIT trap resetting both kinds of lock (ncu's `--clock-control reset` and
`nvidia-smi -rgc`), and between steps the driver resets both again, waits for
an idle card and records the clocks. The rules below are the Stops of the
blocks this section was drafted as, read by the driver instead of a person. A
stop ends its own step: the driver pushes what the step wrote and goes on,
since every step checks the card for itself before it locks. Three things end
the session instead, because nothing after them could be trusted: the prelude
refusing; an `nvidia-smi -rgc` that exits non-zero (the card may still be
locked); and a card still busy five minutes after a step ended.

- **prelude.** Refused (the session ends, exit 2) when `git status
  --porcelain` prints anything, `sudo -n` needs a password, a compute process
  holds the card, 1710 MHz is not a supported graphics clock, or persistence
  (`-pm 1`) or the power limit (`-pl 700 -sc 0`) does not read back. Both
  power scopes are recorded before and after (`power-at-start.txt`,
  `power-set.txt`). 1605 falls back to 1590, whose P1 numbers were registered
  with 1605's (`scripts/r3_timing_model.py`, `P1_CLOCKS`); 1410 or 1500 off
  the card's list take the nearest supported clock one 15 MHz step away,
  labelled SUBSTITUTED (P1 at it is computed after registration), or are
  dropped. The plan is `lock-plan.txt` and the driver's `locks.env`.
- **bytes.** One lock block over G = 1 2 4 16 8 32 3 64, each page at
  `--page-clock none --page-lock-mhz 1710`, treads 1 to 9 (a test holds this to
  `COUNTER_MAX_TREADS`). It goes on past exit 0 and 1, and past exit 3 whenever
  the page was written, whatever its gates read: each page keeps its
  `.ncu-rep` and its gates re-score on the laptop (`--analyse`). The draft
  stopped on any INVALID but V7 alone at G >= 32, which on 2026-09-27 stopped
  the loop at G=1 on V10, a gate defect since fixed. Only a page NOT written,
  a refusal or a crash stops the block. If the G=1 page took over 7 minutes, the
  G=64 page and the base-clock control are dropped. The base-clock control
  (G=2 at ncu's base clock, same board, commit and treads) and the analysis of
  the pages written run however the block ended; the analysis scores C5
  against the five GH200 pages of 2026-09-25 timed at 1710 (another board,
  which its SAME-CARD line says).
- **calibrate.** Runs with no lock in force, because it takes the clock it
  sees as its reference, and writes `moe/bench/hardware/measured_nvidia_gh200_480gb.yaml`
  into the checkout, so every page after it is stamped `git_dirty` and
  `setup_vm.sh --check` refuses the checkout from here on. Exit 0 keeps the
  ruler. Exit 1 with `not_throttled` FAILED, and any exit but 0 or 1, deletes
  it: every later R3 then takes `--ridge 177.93 --bandwidth-gbps 3725.1`, the
  2026-09-25 ruler's (a test holds these to that ruler's file). Exit 1 on
  another gate keeps it, and the ledger names the gates that failed.
- **timed, eta, deep.** Each design runs R3's own `--dry-run` first and is
  locked only when the plan prints no `REFUSED:` line, `n_decl = 9 against
  n_max`, a retracted tread of at least the treads planned, and a ridge that is
  not R3's H200 HYPOTHESIS (without a ruler R3 plans on it and refuses the
  timed run on the card; at treads 9 it also retracts at tread 8). A design
  that fails is left out and the rest run. In `timed`, a locked_r3 exit of 4
  (a crash, or the card left locked) ends the step; otherwise P5 runs whatever
  P2 read. In `eta`, 1410 runs first (it alone answers the falsifier), each lock
  is its own run with its own tag, and an exit 1 (the lock slipped) this far
  under 1710 more likely means a lock left over than the power cap: the
  driver resets, records the clocks, and tries once more under a new tag
  (`...b`). The deep reader prints P3 and P4 off the two pages only when both
  show every VALIDITY gate passing and the 9-copy declaration.
- **floor.** Captures G=64 then G=2 at ncu's base clock, then G=64 unlocked
  with an NVML sampler beside it (a RECORD that scores nothing: P7 is
  registered for the duty-0.25 timed regime, which a kernel-replay capture does
  not reproduce), then G=64 under the 1710 lock. Exit 3 (INVALID; the file is
  written) goes on to the next capture; exit 2 (refused: census, commit or
  door) or worse ends the step. More than 50 replay passes a launch on the
  first capture drops the G=2 capture.
- **r1lock.** Takes 1710 and the eta locks that held (exit 0 in `eta`); its
  lowest lock must be at or under 1459 MHz for the claim's span, or R1 is not
  run. Exit 0 or 1 goes on to G=1; exit 1 is expected either way: the answer
  is report.json's elasticity value and interval.

**Watching it.** `vm_run.sh watch` prints the branch's last push and the tail
of the driver's ledger (`vm/session/gh200-driver/status`: one UTC line per
event, with each step's START and END, every drop and skip, and the lock
ledger lines that name no card). The ledger never copies a `card:` line;
locked_r3.py's own logs and ledgers, and every page, carry the card's UUID,
so read those through a grep, never whole into a chat. To stop the driver on
the VM: `pkill -TERM -f '[g]h200_model_session.sh'` (each layer resets the
clock, then the driver pushes and exits 143). To resume at a step:
`bash ~/moe/repo/scripts/gh200_model_session.sh --from <step> --deadline <epoch s>`;
a rerun takes a fresh session tag (`...b`), since locked_r3.py refuses one
already used.

**Before the terminate**, `vm_run.sh verify` must print VERIFIED, and every
file `HELD-BACK.txt` lists as LEFT ON THE VM must be copied off by hand.
Then terminate (section 5) and `vm_run.sh forget`: the deploy key opens this
repository for writing, and it must not outlive the VM.

**Another model: `--model`.** `gh200_model_session.sh --model mixtral-8x22b`
runs the same steps on another model's shapes, a cross-model test of the two
per-card models against predictions committed first under `docs/registered/`
(`scripts/cross_model_predict.py`). R3, locked_r3 and every counter page take
the model; the byte and floor pages read a census taken for it at ncu's base
clock (`census-<model>.json`), never the preflight's 8x7B census; the analysis
gets no 2026-09-25 8x7B timed references; the estimates and caps scale by the
model's weight bytes (x1.7 for 8x22B, about 4 h of steps), locked_r3 caps one
R3 run at an hour, and eta and R1 are left out unless `--steps` names them. Its
peak is about 44 GB (R3's memory plan: 9 copies of 4.83 GB), inside the GH200's.

### What this session does not answer

P7 as registered: the in-kernel clock of the duty-0.25 unlocked timed pages
against NVML. A kernel-replay capture runs another regime, so the floor step's
unlocked capture is a record, not a test. P8: an H100 measurement. The
cross-card checks (H100 against GH200 at the new cells) and the dead-CTA
declaration sweep need other captures. T(10) at G=2 is not taken: tread 9
keeps the 9-copy, 72-slot declaration of every other page; tread 10 would not.
The timing model scores counted bytes only at treads 1, 2, 3, 4 and 6 until
its `LADDER` is widened after this session, so the counted n = 5, 7, 8 and 9
cells wait for that change.

## 4. Exfiltrate, before anything else

The disk dies with the instance. Section 3c's run does not need this block:
its VM pushed everything to the run's branch, and `scripts/vm_run.sh verify`
checks the branch against the VM's manifest; only a file its `HELD-BACK.txt`
lists as LEFT ON THE VM is copied off by hand. Every other run, from the
laptop:

```bash
DEST=~/moe-kernels-exfil/lambda-$(date -u +%F)
mkdir -p "$DEST"
rsync -avz --partial ubuntu@$IP:moe/results ubuntu@$IP:moe/session ubuntu@$IP:setup_vm.log "$DEST"/
rsync -avz --partial "ubuntu@$IP:moe/exfil-alpha_g-*.tar.gz" "ubuntu@$IP:moe/alpha_g_*.out" "$DEST"/   # section 3b
ssh ubuntu@$IP 'cd moe && find results session -type f | sort | xargs sha256sum' > "$DEST/SHA256SUMS.vm"
ssh ubuntu@$IP 'cd moe && sha256sum exfil-alpha_g-*.tar.gz' >> "$DEST/SHA256SUMS.vm"
(cd "$DEST" && sha256sum -c SHA256SUMS.vm | grep -v ': OK$')   # prints nothing when every file arrived
```

`setup_vm.sh` prints the same rsync line at S7.

## 5. TERMINATE

There is no stop. A shutdown from inside the VM does not end billing (Lambda's
own docs: the instance goes to Alert status and billing continues). Terminate
through the API, then confirm:

```bash
curl -s "${AUTH[@]}" -H 'Content-Type: application/json' $API/instance-operations/terminate \
  -d '{"instance_ids": ["<id>"]}'
curl -s "${AUTH[@]}" $API/instances | jq -r '.data[] | "\(.id) \(.status) \(.name)"'   # <id> gone or terminated
```

After section 3c's run, once the instance is gone, delete the deploy key the
run added, since it opens this repository for writing: `bash scripts/vm_run.sh
forget --run-id <run id>`.

## 6. Cost

The rate is the listing's `price_cents_per_hour` for the type, read at launch
(section 1); it is not typed here because it changes. What the time goes to,
priced by the design and re-priced by the first G's measured per-launch time:

| on | step | time |
|---|---|---|
| H100 | venv downloads and build (S4) | ~25 min |
| H100 | preflight: metric query, probe, census | ~3 min |
| H100 | four G pages, ~4 min each (child start, Triton compiles, 90 profiled launches at ~1.5 s, reduce) | ~16 min |
| H100 | G=16 | +4 min |
| H100 | the same at a pessimistic 5 s per launch | ~40 min |
| H100 | **book** (the above, one parser or door debug loop, exfil) | **1.5 h** |
| A100 40 GB | shake-out, G in {1, 64}; ncu's save goes to host memory, ~1 s per launch more | ~12 min measured, **~45 min** of VM |
| GH200 | section 3c unattended: ~263 min of steps, ~30 min of `setup_vm.sh`, ~15 min for an r580 upgrade and its reboot; the driver's deadline is launch + 400 min | ~5.2 h, **book 6 h, stop by 7 h** |
| any | section 3b's chain at G in {1, 2, 4, 16, 64}, 3 seeds: 15 R3 runs at ~12 min, 5 R1 runs at ~19 min, ~6 min of preconditions and checks (H200 session 5's ledger; the dry run re-prices it on this card); `SEEDS=0`, as the GH200 ran it, is 5 R3 and 5 R1 runs, ~2.7 h | **~4.7 h** |

Cost = hours booked x `price_cents_per_hour` / 100. Treat the meter as running
from the launch call to the terminate call in section 5, which is why every
laptop step in section 0 happens before section 1.
