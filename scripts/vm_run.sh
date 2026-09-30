#!/usr/bin/env bash
# THE LAPTOP'S SIDE OF AN UNATTENDED VM RUN (docs/LAMBDA.md section 3c).
#
#   bash scripts/vm_run.sh prepare --ip <ip> --run-id <id> --branch run-gh200-<date>
#   bash scripts/vm_run.sh start   --ip <ip> --run-id <id> --deadline <epoch s> [--model M] [--steps a,b]
#   bash scripts/vm_run.sh watch   --run-id <id>     # exit 0 when DRIVER-DONE is on the branch, 3 before
#   bash scripts/vm_run.sh verify  --run-id <id>     # every pushed file against SHA256SUMS
#   bash scripts/vm_run.sh forget  --run-id <id>     # delete the run's deploy key
#
# Launching and terminating the instance are the Lambda helper's
# (~/.lambda/status launch|terminate), never this script's: nothing here holds
# or reads a cloud credential. This script talks to the VM over ssh as
# `ubuntu`, with the laptop key Lambda registered at launch, and to GitHub
# through `gh`, which keeps its own login.
#
# prepare   waits for ssh; records the driver; stops apt's timers so nothing
#           upgrades a package under the loaded module during the run; if the
#           driver is below r580, installs nvidia-driver-580-server-open (the
#           owner's decision, 2026-09-26: 580.105.08, as on 2026-09-25; the
#           newest 580 from Lambda's repository if that build is gone) and
#           reboots, then checks the driver the box came back with; copies
#           setup_vm.sh, vm_results_push.sh and gh200_model_session.sh from THIS
#           checkout, which must be clean and at a commit GitHub has (the VM
#           clones that commit); has the VM make its deploy key and prints
#           nothing of it but the public half, which it adds to the repo with
#           write access; and has the VM push the branch's first commit.
# start     runs the driver's plan (--dry-run) on the VM as a record, then
#           starts it detached: setup_vm.sh at this checkout's commit, then the
#           session, pushing after every step. Nothing more goes over ssh.
# watch     fetches the branch and prints the driver's ledger tail and the
#           last push. The laptop polls this and terminates when it exits 0.
# verify    extracts the branch into a scratch directory and checks every file
#           against the VM's SHA256SUMS; lists HELD-BACK.txt.
# forget    deletes the deploy key the run added, and confirms it is gone.
#
# The run's record on the laptop: $MOE_VM_STATE_ROOT/<run-id>/ (default
# ~/moe-kernels-exfil/runs), holding the ip, branch, commit, the deploy key's
# id and public half, the driver's plan and this script's own log.
#
# EXIT CODES (moe/bench/exit_codes.py): 0 done; 2 refused (a dirty checkout, a
# commit GitHub lacks, a bad argument, no ssh); 3 `watch` before the driver is
# done, or `verify` found a file that does not match; 4 a step crashed.
#
# TEST SEAMS: MOE_VM_SSH, MOE_VM_SCP, MOE_VM_GH, MOE_VM_STATE_ROOT,
# MOE_VM_REPO_SLUG, MOE_VM_REMOTE (the git remote watch and verify fetch from),
# MOE_VM_WAIT_S, MOE_VM_POLL_S.
set -euo pipefail

EXIT_DONE=0
EXIT_REFUSED=2
EXIT_INVALID=3
EXIT_ERROR=4

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SSH="${MOE_VM_SSH:-ssh}"
SCP="${MOE_VM_SCP:-scp}"
GH="${MOE_VM_GH:-gh}"
STATE_ROOT="${MOE_VM_STATE_ROOT:-$HOME/moe-kernels-exfil/runs}"
SLUG="${MOE_VM_REPO_SLUG:-harshithkantamneni/moe-kernels}"
REMOTE="${MOE_VM_REMOTE:-origin}"
WAIT_S="${MOE_VM_WAIT_S:-900}"
POLL_S="${MOE_VM_POLL_S:-10}"
#: The owner's driver (2026-09-26): the 2026-09-25 upgrade's exact build.
DRIVER_PKG="nvidia-driver-580-server-open"
DRIVER_BUILD="580.105.08-0lambda0.22.04.1"
MIN_DRIVER=580

say()    { printf '[vm_run] %s\n' "$*"; [[ -n "${LOG:-}" ]] && printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; return 0; }
refuse() { say "REFUSED: $*"; exit "$EXIT_REFUSED"; }
usage()  { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }

SSH_OPTS=()
ssh_opts() {
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$STATE_ROOT/known_hosts"
            -o ConnectTimeout=10 -o ServerAliveInterval=30 -o BatchMode=yes)
}
vm()  { "$SSH" "${SSH_OPTS[@]}" "ubuntu@$IP" "$@"; }

wait_ssh() {   # up to WAIT_S for `true` to run on the VM
  local t=0
  until vm true 2>/dev/null; do
    (( t >= WAIT_S )) && refuse "no ssh to ubuntu@$IP after $WAIT_S s"
    sleep "$POLL_S"; t=$(( t + POLL_S ))
  done
}

driver_version() { vm nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r'; }

run_dir() {
  [[ "${RUN_ID:-}" =~ ^[A-Za-z0-9._-]+$ ]] || refuse "--run-id is letters, digits, '.', '_' and '-' only"
  RUN="$STATE_ROOT/$RUN_ID"
  mkdir -p "$RUN"
  LOG="$RUN/vm_run.log"
}

need_ip() { [[ "${IP:-}" =~ ^[0-9A-Za-z.:-]+$ ]] || refuse "--ip <address> is required"; }

#: The commit the VM measures (SHA): this checkout's, clean, and on GitHub.
check_commit() {
  [[ -z "$(git -C "$ROOT" status --porcelain)" ]] \
    || refuse "this checkout ($ROOT) has local changes: the VM clones a commit, so commit or stash first"
  SHA="$(git -C "$ROOT" rev-parse HEAD)"
  git -C "$ROOT" fetch -q "$REMOTE" 2>/dev/null || true
  [[ -n "$(git -C "$ROOT" branch -r --contains "$SHA" 2>/dev/null)" ]] \
    || refuse "commit $SHA is on no branch of $REMOTE: push it first (the VM clones it from GitHub)"
}

boot_id() { vm cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d ' \r' || true; }

#: Reboot, and wait for a NEW boot: an ssh that answers before the reboot has
#: happened would read the old driver.
reboot_vm() {
  local before t=0; before="$(boot_id)"
  vm 'sudo -n systemctl reboot' >/dev/null 2>&1 || true
  while :; do
    sleep "$POLL_S"; t=$(( t + POLL_S ))
    local now; now="$(boot_id)"
    [[ -n "$now" && "$now" != "$before" ]] && return 0
    (( t >= WAIT_S )) && refuse "the VM did not come back from its reboot within $WAIT_S s"
  done
}

cmd_prepare() {
  need_ip; run_dir
  [[ "${BRANCH:-}" =~ ^run-[a-z0-9][a-z0-9.-]*$ ]] || refuse "--branch run-<name> is required (never main or a model/integrate branch)"
  local sha; check_commit; sha="$SHA"
  ssh_opts
  printf 'ip=%s\nbranch=%s\ncommit=%s\n' "$IP" "$BRANCH" "$sha" > "$RUN/run.env"
  say "prepare $RUN_ID: ubuntu@$IP, branch $BRANCH, commit $sha"
  # A fresh instance has a fresh host key, and Lambda reuses IPs (2026-09-29: three
  # GH200s in a row at one address): drop any key pinned for this IP by an earlier
  # instance, so accept-new pins this one instead of refusing it as a changed host.
  if [[ -f "$STATE_ROOT/known_hosts" ]] && ssh-keygen -F "$IP" -f "$STATE_ROOT/known_hosts" >/dev/null 2>&1; then
    ssh-keygen -R "$IP" -f "$STATE_ROOT/known_hosts" >/dev/null 2>&1 \
      && say "dropped the host key an earlier instance left for $IP"
  fi
  wait_ssh
  # Nothing upgrades a package under the loaded module during the run (the
  # Driver/library mismatch of docs/LAMBDA.md section 2).
  vm 'sudo -n systemctl stop apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service 2>/dev/null; sudo -n systemctl disable apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service 2>/dev/null; true'
  local have; have="$(driver_version)"
  [[ "$have" =~ ^([0-9]+)\. ]] || refuse "nvidia-smi read no driver version on the VM ('$have')"
  say "driver $have"
  if (( BASH_REMATCH[1] < MIN_DRIVER )); then
    (( BASH_REMATCH[1] >= 570 )) || refuse "driver $have is below r570: rent another instance"
    say "upgrading to $DRIVER_PKG $DRIVER_BUILD (the owner's decision, 2026-09-26), then a reboot"
    local rc=0
    vm "sudo -n apt-get -o DPkg::Lock::Timeout=600 update -q > driver580.log 2>&1; sudo -n DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y $DRIVER_PKG=$DRIVER_BUILD >> driver580.log 2>&1; rc=\$?; echo EXIT \$rc >> driver580.log; exit \$rc" || rc=$?
    if (( rc )); then
      say "the $DRIVER_BUILD build did not install (exit $rc): the newest $DRIVER_PKG from Lambda's repository instead"
      rc=0
      vm "sudo -n DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y $DRIVER_PKG >> driver580.log 2>&1; rc=\$?; echo EXIT \$rc >> driver580.log; exit \$rc" || rc=$?
      (( rc == 0 )) || refuse "$DRIVER_PKG did not install (exit $rc): read driver580.log on the VM"
    fi
    reboot_vm
    have="$(driver_version)"
    [[ "$have" =~ ^([0-9]+)\. ]] && (( BASH_REMATCH[1] >= MIN_DRIVER )) \
      || refuse "after the reboot the driver reads '$have', not r$MIN_DRIVER+"
    say "driver $have after the reboot"
  fi
  printf 'driver=%s\n' "$have" >> "$RUN/run.env"
  "$SCP" "${SSH_OPTS[@]}" "$ROOT/scripts/setup_vm.sh" "$ROOT/scripts/vm_results_push.sh" \
    "$ROOT/scripts/gh200_model_session.sh" "ubuntu@$IP:" >/dev/null
  local name email out
  name="$(git -C "$ROOT" config user.name)"; email="$(git -C "$ROOT" config user.email)"
  out="$(vm bash vm_results_push.sh init --branch "$(printf %q "$BRANCH")" --run-id "$(printf %q "$RUN_ID")" \
           --name "$(printf %q "$name")" --email "$(printf %q "$email")")"
  # Only the public half ever leaves the VM, on the PUBLIC-KEY line.
  printf '%s\n' "$out" | grep '^PUBLIC-KEY ssh-ed25519 ' | head -1 | cut -d' ' -f2- > "$RUN/deploy-key.pub"
  [[ -s "$RUN/deploy-key.pub" ]] || refuse "the VM printed no public key: $(printf '%s' "$out" | grep -v '^PUBLIC-KEY' | tail -2)"
  local id
  id="$("$GH" api -X POST "repos/$SLUG/keys" -f "title=moe VM results $RUN_ID" \
          -f "key=$(cat "$RUN/deploy-key.pub")" -F read_only=false --jq .id)"
  [[ "$id" =~ ^[0-9]+$ ]] || refuse "gh did not add the deploy key (got '$id')"
  printf '%s\n' "$id" > "$RUN/deploy-key.id"
  say "deploy key $id added to $SLUG with write access (delete it with: bash scripts/vm_run.sh forget --run-id $RUN_ID)"
  vm bash vm_results_push.sh check || refuse "the VM could not push with its key (read the lines above)"
  git -C "$ROOT" ls-remote --exit-code "$REMOTE" "refs/heads/$BRANCH" >/dev/null \
    || refuse "branch $BRANCH is not on $REMOTE after the VM's first push"
  say "READY: the VM pushed $BRANCH's first commit; next: bash scripts/vm_run.sh start --ip $IP --run-id $RUN_ID --deadline <epoch s>"
}

cmd_start() {
  need_ip; run_dir
  [[ "${DEADLINE:-}" =~ ^[0-9]+$ ]] || refuse "--deadline <epoch seconds> is required"
  [[ -f "$RUN/run.env" ]] || refuse "no $RUN/run.env: run prepare first"
  local sha; sha="$(sed -n 's/^commit=//p' "$RUN/run.env")"
  [[ "$sha" == "$(git -C "$ROOT" rev-parse HEAD)" ]] \
    || refuse "this checkout moved since prepare ($sha): the scripts on the VM are that commit's"
  ssh_opts
  local model=()
  [[ -z "${MODEL:-}" ]] || { [[ "$MODEL" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || refuse "--model $MODEL: not a model name"; model=(--model "$MODEL"); }
  # --steps: the driver's own step list (e.g. `floor` for a floor-only session), passed as given
  [[ -z "${STEPS_ARG:-}" ]] || { [[ "$STEPS_ARG" =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] || refuse "--steps $STEPS_ARG: a comma-separated list of step names"; model+=(--steps "$STEPS_ARG"); }
  vm bash gh200_model_session.sh --dry-run --deadline "$DEADLINE" ${model[@]+"${model[@]}"} \
    > "$RUN/driver-plan.txt" 2>&1 || true
  grep -q '^THE GH200 MODEL-TEST SESSION' "$RUN/driver-plan.txt" \
    || refuse "the driver's plan did not print on the VM: $RUN/driver-plan.txt"
  printf 'deadline=%s\nmodel=%s\n' "$DEADLINE" "${MODEL:-mixtral-8x7b}" >> "$RUN/run.env"
  vm "nohup setsid bash gh200_model_session.sh --setup --commit $sha --repo https://github.com/$SLUG --deadline $DEADLINE ${model[*]+${model[*]}} >> gh200-driver.out 2>&1 < /dev/null & sleep 2; pgrep -f '[g]h200_model_session.sh --setup' >/dev/null && echo STARTED" \
    | grep -q STARTED || refuse "the driver did not start on the VM (read ~/gh200-driver.out there)"
  say "STARTED: setup_vm.sh at $sha, then the session; deadline $(date -u -r "$DEADLINE" +%FT%TZ 2>/dev/null || date -u -d "@$DEADLINE" +%FT%TZ)"
}

fetch_branch() {
  [[ -f "$RUN/run.env" ]] || refuse "no $RUN/run.env"
  BRANCH="$(sed -n 's/^branch=//p' "$RUN/run.env")"
  git -C "$ROOT" fetch -q "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH" \
    || refuse "could not fetch $BRANCH from $REMOTE"
  REF="refs/remotes/$REMOTE/$BRANCH"
}

cmd_watch() {
  run_dir; fetch_branch
  say "$BRANCH at $(git -C "$ROOT" rev-parse --short "$REF"): $(git -C "$ROOT" show "$REF:PUSHES.txt" 2>/dev/null | tail -1)"
  git -C "$ROOT" show "$REF:vm/session/gh200-driver/status" 2>/dev/null | tail -n "${MOE_VM_TAIL:-12}"
  if git -C "$ROOT" cat-file -e "$REF:DRIVER-DONE" 2>/dev/null; then
    say "DONE: $(git -C "$ROOT" show "$REF:DRIVER-DONE")"
    return "$EXIT_DONE"
  fi
  return "$EXIT_INVALID"
}

cmd_verify() {
  run_dir; fetch_branch
  local out="$RUN/tree"
  rm -rf "$out"; mkdir -p "$out"
  git -C "$ROOT" archive "$REF" | tar -x -C "$out"
  local bad
  if command -v sha256sum >/dev/null 2>&1; then bad="$(cd "$out" && sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$' || true)"
  else bad="$(cd "$out" && shasum -a 256 -c SHA256SUMS 2>&1 | grep -v ': OK$' || true)"; fi
  local n; n="$(wc -l < "$out/SHA256SUMS" | tr -d ' ')"
  if [[ -s "$out/HELD-BACK.txt" ]]; then
    say "HELD-BACK.txt lists $(wc -l < "$out/HELD-BACK.txt" | tr -d ' ') file(s):"
    cut -f1,2,4 "$out/HELD-BACK.txt"
  fi
  if [[ -n "$bad" ]]; then
    printf '%s\n' "$bad" | head -20
    say "VERIFY FAILED: $(printf '%s\n' "$bad" | wc -l | tr -d ' ') of $n files do not match SHA256SUMS"
    return "$EXIT_INVALID"
  fi
  say "VERIFIED: all $n files under vm/ match SHA256SUMS ($out)"
}

cmd_forget() {
  run_dir
  [[ -s "$RUN/deploy-key.id" ]] || refuse "no deploy key on record for $RUN_ID ($RUN/deploy-key.id)"
  local id; id="$(cat "$RUN/deploy-key.id")"
  "$GH" api -X DELETE "repos/$SLUG/keys/$id" >/dev/null || say "gh could not delete key $id (already gone?)"
  if "$GH" api "repos/$SLUG/keys" --jq '.[].id' | grep -qx "$id"; then
    say "deploy key $id is STILL on $SLUG"; return "$EXIT_ERROR"
  fi
  mv "$RUN/deploy-key.id" "$RUN/deploy-key.id.deleted"
  say "deploy key $id deleted from $SLUG"
}

sub="${1:-}"; [[ -n "$sub" ]] && shift
IP=""; RUN_ID=""; BRANCH=""; DEADLINE=""; MODEL=""; STEPS_ARG=""
while (( $# )); do
  case "$1" in
    --ip) IP="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --branch) BRANCH="${2:-}"; shift 2 ;;
    --deadline) DEADLINE="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --steps) STEPS_ARG="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) refuse "unknown argument $1" ;;
  esac
done
case "$sub" in
  prepare) cmd_prepare ;;
  start)   cmd_start ;;
  watch)   cmd_watch ;;
  verify)  cmd_verify ;;
  forget)  cmd_forget ;;
  -h|--help) usage ;;
  *) usage; exit "$EXIT_REFUSED" ;;
esac
