#!/usr/bin/env bash
# THE LAPTOP COPIES THE RUN'S RAW TRACES OFF THE VM, BEFORE THE TERMINATE (review 2026-11-19 m6).
#
#   bash scripts/prod/pull_traces.sh <ip> <run> [--dest DIR] [--dry-run]
#
# main.sh writes every raw trace (*.nsys-rep, *.sqlite: GBs) under the VM's $MOE_HOME/traces/<run>/, OUTSIDE the
# pushed results tree, so scripts/vm_results_push.sh (pinned) never xz-compresses them on every push; each push
# records their sha256 in the results tree (<run>/traces.sha256.json). This script rsyncs ~/moe/traces/<run>/ from
# ubuntu@<ip> into DEST/<run>/ (default $PROD_TRACES_DEST, else ~/moe-kernels-exfil/traces), over ssh with the runs'
# known_hosts ($MOE_VM_STATE_ROOT/known_hosts, default ~/moe-kernels-exfil/runs/known_hosts: the host vm_run.sh
# already met; StrictHostKeyChecking=yes, so an unknown host is refused), then fetches the VM's manifest and checks
# every file's sha256 against it. Exit 0 all copied and matching, 3 a mismatch or a missing file, 2 refused (usage, an
# unknown host). Run it before the terminate; it never touches the VM's files.
set -euo pipefail

usage() { sed -n '4,4p' "$0" | sed 's/^# \{0,1\}//'; }
DRY=0; DEST="${PROD_TRACES_DEST:-$HOME/moe-kernels-exfil/traces}"
POS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "REFUSED pull_traces: unknown option $1" >&2; exit 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done
if [[ ${#POS[@]} -ne 2 ]]; then usage >&2; exit 2; fi
IP="${POS[0]}"; RUN="${POS[1]}"
if [[ ! "$RUN" =~ ^[A-Za-z0-9._-]+$ ]]; then echo "REFUSED pull_traces: run name '$RUN'" >&2; exit 2; fi
STATE_ROOT="${MOE_VM_STATE_ROOT:-$HOME/moe-kernels-exfil/runs}"
KNOWN="$STATE_ROOT/known_hosts"
SSH_CMD=(ssh -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$KNOWN" -o BatchMode=yes -o ConnectTimeout=10
         -o ServerAliveInterval=30)
REMOTE="ubuntu@$IP"
SRC="moe/traces/$RUN/"
MAN="moe/results/$RUN/traces.sha256.json"
OUT="$DEST/$RUN"
RSYNC=(rsync -a --partial -e "$(printf '%q ' "${SSH_CMD[@]}")" "$REMOTE:$SRC" "$OUT/")
if [[ $DRY -eq 1 ]]; then
  echo "PLAN pull_traces: $REMOTE:$SRC -> $OUT/ (known_hosts $KNOWN)"
  echo "PLAN $(printf '%q ' "${RSYNC[@]}")"
  echo "PLAN $(printf '%q ' "${SSH_CMD[@]}" "$REMOTE" cat "$MAN")> $OUT/traces.sha256.json"
  echo "PLAN verify every file's sha256 against $OUT/traces.sha256.json"
  exit 0
fi
if [[ ! -f "$KNOWN" ]] || ! ssh-keygen -F "$IP" -f "$KNOWN" >/dev/null 2>&1; then
  echo "REFUSED pull_traces: $IP is not in $KNOWN (the runs' known_hosts: vm_run.sh records the VM's key)" >&2
  exit 2
fi
mkdir -p "$OUT"
"${SSH_CMD[@]}" "$REMOTE" cat "$MAN" > "$OUT/traces.sha256.json" < /dev/null \
  || { echo "FAIL pull_traces: no manifest $MAN on the VM (did a push run?)"; exit 3; }
"${RSYNC[@]}" < /dev/null || { echo "FAIL pull_traces: rsync exit $?"; exit 3; }
python3 - "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
man = json.loads((out / "traces.sha256.json").read_text())
bad = []
for rel, v in man["files"].items():
    f = out / rel
    if not f.is_file():
        bad.append(f"{rel}: missing")
        continue
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for buf in iter(lambda: fh.read(1 << 24), b""):
            h.update(buf)
    if h.hexdigest() != v["sha256"]:
        bad.append(f"{rel}: sha256 differs (a trace written after the last push? push, then pull again)")
if bad:
    print("FAIL pull_traces: " + "; ".join(bad[:8]))
    sys.exit(3)
print(f"PASS pull_traces: {len(man['files'])} file(s), {man.get('total_bytes', 0) / 1e9:.2f} GB, sha256 match -> {out}")
PY
