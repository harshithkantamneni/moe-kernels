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

**The card is not the study's.** Lambda has no H200. The target is
`gpu_1x_h100_sxm5` (H100 SXM5: 80 GB HBM3, 132 SMs, 50 MB L2, sm_90), and a
cheaper shake-out is `gpu_1x_a100_sxm4` (A100 SXM4 40 GB: 108 SMs, 40 MB L2,
sm_80). Every timed page of this study was measured on `nvidia_h200` (60 MiB
L2, HBM3e). An alpha measured on either Lambda card is that card's, and every
page the preflight and the counter run write opens with a `CARD` line naming
the card it measured.

Whether counters are readable on a Lambda VM is **not verified in this repo**.
The report that ncu works there is second-hand (GPU MODE), and
`RestrictProfilingToAdminUsers` on Lambda's driver is unread. The preflight's
PF5 decides it on the first box, by reading a counter, and nothing else here
passes a box on which no counter came back.

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
  | select(.key == "gpu_1x_h100_sxm5" or .key == "gpu_1x_a100_sxm4")
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
  is built: rent another instance. `--torch-index cu128` on r570-r579 builds
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
```

Each page keeps its `.ncu-rep` beside it (`<out>.profiles/`), so a parser
defect found later is a laptop fix, not a re-rent. The first G's measured
per-launch time re-prices the rest.

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

## 4. Exfiltrate, before anything else

The disk dies with the instance. From the laptop:

```bash
DEST=~/moe-kernels-exfil/lambda-$(date -u +%F)
mkdir -p "$DEST"
rsync -avz --partial ubuntu@$IP:moe/results ubuntu@$IP:moe/session ubuntu@$IP:setup_vm.log "$DEST"/
ssh ubuntu@$IP 'cd moe && find results session -type f | sort | xargs sha256sum' > "$DEST/SHA256SUMS.vm"
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

Cost = hours booked x `price_cents_per_hour` / 100. Treat the meter as running
from the launch call to the terminate call in section 5, which is why every
laptop step in section 0 happens before section 1.
