# Lambda runbook: the R3 arms under a DRAM counter

A Lambda Cloud instance is a plain Ubuntu VM with root, Lambda Stack and a
driver nobody knows until it boots. Its local disk survives a reboot and is
**lost at termination** (Lambda's docs: terminating permanently removes the
instance), and Lambda instances can only be launched, restarted or terminated
(there is no stop). Lambda also offers filesystems, networked persistent
storage attached only when an instance is created (`file_system_names` in the
launch call, in the instance's region). **This runbook launches without
one**, so the local disk is all there is and section 4's exfiltration is the
only copy of the results; attaching one at launch would keep results past
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
two binds: read both in `nvidia-smi -q -d POWER` before setting either.

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

## 4. Exfiltrate, before anything else

The disk dies with the instance. From the laptop:

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
| any | section 3b's chain at G in {1, 2, 4, 16, 64}, 3 seeds: 15 R3 runs at ~12 min, 5 R1 runs at ~19 min, ~6 min of preconditions and checks (H200 session 5's ledger; the dry run re-prices it on this card); `SEEDS=0`, as the GH200 ran it, is 5 R3 and 5 R1 runs, ~2.7 h | **~4.7 h** |

Cost = hours booked x `price_cents_per_hour` / 100. Treat the meter as running
from the launch call to the terminate call in section 5, which is why every
laptop step in section 0 happens before section 1.
