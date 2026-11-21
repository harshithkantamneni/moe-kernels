#!/usr/bin/env bash
# A RENTED VM PUSHES ITS OWN RESULTS INTO THIS REPO, ON THE RUN'S OWN BRANCH.
#
#   bash vm_results_push.sh init --branch run-gh200-2026-09-27 --run-id <id> \
#        --name <git author name> --email <git author email>
#   bash vm_results_push.sh check            # GitHub takes the key: push the branch's first commit
#   bash vm_results_push.sh push --message "3c.2 byte pages: exit 0" [--final EXIT]
#
# WHY A PUSH FROM THE VM, AND NOT A COPY PULLED FROM THE LAPTOP. A Lambda
# instance has no stop state: its disk dies with it, so until 2026-09-26 the
# results were pulled off by rsync from the laptop (docs/LAMBDA.md section 4)
# before anything was terminated. That copy happened once, at the end, and a
# run cut short (a dropped laptop, a budget reached) could lose everything. The
# VM now pushes after every step of its driver, so what it measured is on
# GitHub within minutes of being measured, and the laptop reads it with `git
# fetch`. The owner's design, 2026-09-26: results go into this repo itself, on
# the run's own branch, never main or a model/integrate branch; the publish
# step curates from there.
#
# THE KEY IS THE VM'S OWN. `init` makes an ed25519 key on the VM
# ($MOE_PUSH_KEY, default ~/.ssh/moe_results_ed25519, no passphrase) and prints
# only its PUBLIC half. The laptop adds that half to the repo as a deploy key
# with write access (`gh repo deploy-key add --allow-write`, scripts/vm_run.sh)
# and deletes it right after the run. No token and no private key ever moves
# between machines, and a deploy key opens this one repository and nothing
# else. GitHub's host key is pinned here (GITHUB_HOST_KEY, the ed25519 key
# https://api.github.com/meta lists, fingerprint
# SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU), so the first connection
# trusts nothing it was not told.
#
# THE BRANCH IS AN ORPHAN, IN A CLONE OF ITS OWN. The results are committed in
# $MOE_PUSH_HOME/push (default ~/moe/push), a repository that shares nothing
# with the measured checkout ~/moe/repo: its commits, its index and its files
# are elsewhere, so a push can never make `git status --porcelain` in the
# checkout print a line, and no page is stamped git_dirty by the act of
# publishing another. The branch holds only this run:
#
#   README.md        what the branch is, which commit was measured, how to verify
#   vm/results/      ~/moe/results        (every page, every run directory)
#   vm/session/      ~/moe/session        (logs, ledgers, the driver's status)
#   vm/<log>         ~/setup_vm.log, ~/driver580.log, ~/gh200-driver.out
#   SHA256SUMS       sha256 of every file under vm/, as committed (the manifest)
#   HELD-BACK.txt    files over the size limit: compressed, or left on the VM
#   PUSHES.txt       one UTC line per push, the step it closed
#   DRIVER-DONE      written by the last push (--final): the driver's exit
#
# Each push rebuilds vm/ from the sources, so the branch is always a snapshot
# of the VM, and git stores only what changed. A file over $MOE_PUSH_MAX_BYTES
# (95 MB: GitHub refuses 100 MB) is committed xz-compressed when that fits, and
# otherwise listed in HELD-BACK.txt with its size and sha256 and left on the
# VM, to be copied by hand before the terminate.
#
# PUSH ONLY BETWEEN MEASUREMENTS. The driver (scripts/gh200_model_session.sh)
# calls `push` after a step has ended and its locks are reset, never while a
# page is being measured, so the copy, the hash and the upload share the host
# with nothing that is timed.
#
# EXIT CODES, the repo's table (moe/bench/exit_codes.py): 0 pushed (or, for
# `init`, ready); 1 committed here but not pushed (the next push carries it:
# every push is the whole snapshot); 2 refused (not initialised, a bad
# argument, the key or the branch missing); 4 a step crashed.
#
# TEST SEAMS: MOE_PUSH_HOME, MOE_PUSH_REMOTE (a local bare repository in the
# tests), MOE_PUSH_KEY, MOE_PUSH_SOURCES (space-separated, relative to $HOME),
# MOE_PUSH_MAX_BYTES, MOE_PUSH_TRIES, MOE_PUSH_BACKOFF_S.
set -euo pipefail

EXIT_DONE=0
EXIT_CLAIM_FAIL=1
EXIT_REFUSED=2
EXIT_ERROR=4

#: GitHub's ed25519 SSH host key, as https://api.github.com/meta lists it
#: (read 2026-09-26; fingerprint SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU).
GITHUB_HOST_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
DEFAULT_REMOTE="git@github.com:harshithkantamneni/moe-kernels.git"
#: What a push copies, relative to $HOME. Each is optional: a file the run has
#: not written yet is skipped, not an error.
DEFAULT_SOURCES="moe/results moe/session setup_vm.log driver580.log gh200-driver.out"

PUSH_HOME="${MOE_PUSH_HOME:-$HOME/moe}"
WORK="$PUSH_HOME/push"
CONF="$PUSH_HOME/push.env"
KEY="${MOE_PUSH_KEY:-$HOME/.ssh/moe_results_ed25519}"
KNOWN="$PUSH_HOME/push-known_hosts"
MAX_BYTES="${MOE_PUSH_MAX_BYTES:-95000000}"
TRIES="${MOE_PUSH_TRIES:-3}"
BACKOFF="${MOE_PUSH_BACKOFF_S:-15}"

say()    { printf '[push] %s\n' "$*"; }
refuse() { printf '[push] REFUSED: %s\n' "$*" >&2; exit "$EXIT_REFUSED"; }
utc()    { date -u +%Y-%m-%dT%H:%M:%SZ; }

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }

sha256_of() {   # the file's sha256, GNU or BSD
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

file_bytes() { wc -c < "$1" | tr -d ' '; }

#: ssh as git runs it for this remote: the VM's own key, only it, and only
#: GitHub's pinned host key.
git_ssh() {
  printf 'ssh -i %q -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 ' "$KEY"
  printf -- '-o StrictHostKeyChecking=yes -o UserKnownHostsFile=%q ' "$KNOWN"
  printf -- '-o HostKeyAlgorithms=ssh-ed25519'
}

is_ssh_remote() { [[ "$1" == *@*:* || "$1" == ssh://* ]]; }

load_conf() {
  [[ -f "$CONF" ]] || refuse "not initialised: run \`$0 init --branch B --run-id ID ...\` first ($CONF is missing)"
  # shellcheck disable=SC1090
  . "$CONF"
  [[ -d "$WORK/.git" ]] || refuse "$WORK is not a repository: run init again"
}

in_work() { git -C "$WORK" "$@"; }

# --------------------------------------------------------------------------
# init: the key, the pinned host key, the orphan branch's clone
# --------------------------------------------------------------------------
cmd_init() {
  local branch="" run_id="" name="" email="" remote="${MOE_PUSH_REMOTE:-$DEFAULT_REMOTE}"
  while (( $# )); do
    case "$1" in
      --branch) branch="${2:-}"; shift 2 ;;
      --run-id) run_id="${2:-}"; shift 2 ;;
      --name)   name="${2:-}"; shift 2 ;;
      --email)  email="${2:-}"; shift 2 ;;
      *) refuse "init: unknown argument $1" ;;
    esac
  done
  [[ "$branch" =~ ^run-[a-z0-9][a-z0-9.-]*$ ]] \
    || refuse "init: --branch must be run-<name> (lower case, digits, '.', '-'): results never go to main or a model/integrate branch; got '${branch}'"
  [[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || refuse "init: --run-id is letters, digits, '.', '_' and '-' only"
  [[ -n "$name" && -n "$email" ]] || refuse "init: --name and --email (the laptop checkout's git identity) are required"
  if [[ -f "$CONF" ]]; then
    # shellcheck disable=SC1090
    local old_branch; old_branch="$(. "$CONF"; printf '%s' "$BRANCH")"
    [[ "$old_branch" == "$branch" ]] \
      || refuse "init: already initialised for branch $old_branch; this VM pushes one run"
  fi
  command -v git >/dev/null || refuse "init: git is not installed"
  command -v ssh-keygen >/dev/null || refuse "init: ssh-keygen is not installed"
  mkdir -p "$PUSH_HOME" "$(dirname "$KEY")"
  chmod 700 "$(dirname "$KEY")"
  if [[ ! -f "$KEY" ]]; then
    ssh-keygen -q -t ed25519 -N '' -C "moe-results $run_id" -f "$KEY" >/dev/null
    say "made a new key on this machine ($KEY; only its public half leaves it)"
  else
    say "the key already exists ($KEY)"
  fi
  printf 'github.com %s\n' "$GITHUB_HOST_KEY" > "$KNOWN"
  if [[ ! -d "$WORK/.git" ]]; then
    git init -q "$WORK"
    in_work checkout -q --orphan "$branch"
  fi
  in_work config user.name "$name"
  in_work config user.email "$email"
  in_work config core.sshCommand "$(git_ssh)"
  if in_work remote get-url origin >/dev/null 2>&1; then
    in_work remote set-url origin "$remote"
  else
    in_work remote add origin "$remote"
  fi
  {
    printf 'BRANCH=%q\n' "$branch"
    printf 'RUN_ID=%q\n' "$run_id"
    printf 'REMOTE=%q\n' "$remote"
  } > "$CONF"
  say "ready: $WORK, orphan branch $branch -> $remote"
  printf 'PUBLIC-KEY %s\n' "$(cat "$KEY.pub")"
}

# --------------------------------------------------------------------------
# the snapshot
# --------------------------------------------------------------------------
readme_text() {
  local commit="unknown"
  [[ -f "$HOME/moe/session/commit.txt" ]] && commit="$(head -1 "$HOME/moe/session/commit.txt")"
  cat <<EOF
# $BRANCH: results pushed by the VM itself

Run \`$RUN_ID\`. Every file under \`vm/\` is a copy of the rented VM's own,
pushed by \`scripts/vm_results_push.sh\` after each step of
\`scripts/gh200_model_session.sh\` (docs/LAMBDA.md section 3c). Nothing here is
curated: this branch is the raw record the publish step reads from.

- measured commit: \`$commit\` (\`vm/session/commit.txt\`)
- \`vm/results/\` is \`~/moe/results\`, \`vm/session/\` is \`~/moe/session\`
- \`SHA256SUMS\`: the manifest of every file under \`vm/\` as this push committed
  it (the VM's files, copied with \`cp -p\` between two steps, when nothing is
  measuring). Check a fetched tree with \`sha256sum -c SHA256SUMS\`
  (\`shasum -a 256 -c\` on macOS): it prints \`OK\` for every line.
- \`HELD-BACK.txt\`: files over the size limit (compressed here, or left on the VM)
- \`PUSHES.txt\`: one line per push; \`DRIVER-DONE\` appears with the last one
- the driver's own ledger: \`vm/session/gh200-driver/status\`
EOF
}

#: Rebuild vm/ from the sources, then hold back what is too big, then hash.
snapshot() {
  local sources="${MOE_PUSH_SOURCES:-$DEFAULT_SOURCES}" src rel dest
  rm -rf "$WORK/vm"
  mkdir -p "$WORK/vm"
  for rel in $sources; do
    src="$HOME/$rel"
    [[ -e "$src" ]] || continue
    case "$rel" in
      moe/*) dest="$WORK/vm/${rel#moe/}" ;;
      *)     dest="$WORK/vm/$(basename "$rel")" ;;
    esac
    mkdir -p "$(dirname "$dest")"
    cp -pR "$src" "$dest"
  done
  : > "$WORK/HELD-BACK.txt.new"
  local f bytes sha
  while IFS= read -r -d '' f; do
    bytes="$(file_bytes "$f")"
    (( bytes > MAX_BYTES )) || continue
    sha="$(sha256_of "$f")"
    rel="${f#"$WORK"/}"
    if command -v xz >/dev/null 2>&1 && xz -T0 -6 -c "$f" > "$f.xz" 2>/dev/null \
        && (( $(file_bytes "$f.xz") <= MAX_BYTES )); then
      printf '%s\t%s bytes\tsha256 %s\tCOMPRESSED as %s.xz\n' "$rel" "$bytes" "$sha" "$rel" \
        >> "$WORK/HELD-BACK.txt.new"
    else
      rm -f "$f.xz"
      printf '%s\t%s bytes\tsha256 %s\tLEFT ON THE VM (copy it by hand before the terminate)\n' \
        "$rel" "$bytes" "$sha" >> "$WORK/HELD-BACK.txt.new"
    fi
    rm -f "$f"
  done < <(find "$WORK/vm" -type f -print0)
  mv "$WORK/HELD-BACK.txt.new" "$WORK/HELD-BACK.txt"
  (cd "$WORK" && find vm -type f -print0 | LC_ALL=C sort -z | while IFS= read -r -d '' f; do
     printf '%s  %s\n' "$(sha256_of "$f")" "$f"; done) > "$WORK/SHA256SUMS"
  readme_text > "$WORK/README.md"
}

push_branch() {
  local i rc
  for (( i = 1; i <= TRIES; i++ )); do
    rc=0
    in_work push -q origin "HEAD:refs/heads/$BRANCH" 2>"$WORK/.git/push-error.txt" || rc=$?
    (( rc )) || return 0
    say "push attempt $i of $TRIES failed (exit $rc): $(tail -1 "$WORK/.git/push-error.txt")"
    (( i < TRIES )) && sleep $(( BACKOFF * i ))
  done
  return 1
}

cmd_push() {
  local message="" final=""
  while (( $# )); do
    case "$1" in
      --message) message="${2:-}"; shift 2 ;;
      --final)   final="${2:-}"; shift 2 ;;
      *) refuse "push: unknown argument $1" ;;
    esac
  done
  [[ -n "$message" ]] || refuse "push: --message is required (the step this push closes)"
  [[ -z "$final" || "$final" =~ ^[0-9]+$ ]] || refuse "push: --final takes the driver's exit code"
  load_conf
  snapshot
  printf '%s %s\n' "$(utc)" "$message" >> "$WORK/PUSHES.txt"
  if [[ -n "$final" ]]; then
    printf 'driver exit %s at %s: %s\n' "$final" "$(utc)" "$message" > "$WORK/DRIVER-DONE"
  fi
  in_work add -A .
  if in_work diff --cached --quiet; then
    say "nothing new since the last commit"
  else
    in_work commit -q -m "$RUN_ID: $message"
  fi
  if push_branch; then
    say "pushed $(in_work rev-parse --short HEAD) to $BRANCH: $message"
    return "$EXIT_DONE"
  fi
  say "NOT PUSHED (committed here; the next push carries it): $message"
  return "$EXIT_CLAIM_FAIL"
}

# --------------------------------------------------------------------------
# check: GitHub takes the key, and the branch's first commit lands
# --------------------------------------------------------------------------
cmd_check() {
  load_conf
  if is_ssh_remote "$REMOTE"; then
    local out
    out="$(eval "$(git_ssh)" -T git@github.com 2>&1 || true)"
    if [[ "$out" != *"successfully authenticated"* ]]; then
      printf '[push] REFUSED: GitHub did not take this key: %s\n' "$(printf '%s' "$out" | tail -1)" >&2
      exit "$EXIT_REFUSED"
    fi
    say "GitHub took the key: $(printf '%s' "$out" | grep -o 'Hi [^!]*' | head -1)"
  fi
  cmd_push --message "the branch's first commit: the VM can push"
}

sub="${1:-}"
[[ -n "$sub" ]] && shift
case "$sub" in
  init)  cmd_init "$@" ;;
  check) cmd_check ;;
  push)  cmd_push "$@" ;;
  -h|--help|"") usage; [[ -n "$sub" ]] || exit "$EXIT_REFUSED" ;;
  *) refuse "unknown subcommand $sub (init, check, push)" ;;
esac
