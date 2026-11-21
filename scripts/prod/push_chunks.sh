#!/usr/bin/env bash
# CHUNKED PUSH (v10 blind run, 2026-11-21). GitHub refuses a push whose pack is over 2 GiB ("pack exceeds maximum
# allowed size"). scripts/vm_results_push.sh (registered and pinned: not edited) commits the whole snapshot as one
# commit, so a run whose captures add more than 2 GiB between two pushes can never land, and every later push of the
# run carries the same refused pack. kit_push (kitlib.sh) calls this script only after that refusal.
#
# It pushes the push clone's HEAD tree to the run's branch as a chain of commits on top of the branch's remote tip,
# each adding at most LIMIT bytes of new or changed files (their uncompressed size: an upper bound on the pack), the
# last one with HEAD's tree exactly and HEAD's message. Deletions go in the last commit. GitHub counts only the
# destination ref's own history as already present (another ref's objects do not shorten the pack), so the chunks go
# on the run's branch itself, each a fast-forward. Then the local branch is reset --soft to the last chunk (same tree;
# work tree untouched), so the next vm_results_push.sh push is an ordinary one.
#
#   bash scripts/prod/push_chunks.sh [--limit BYTES]
#
# The push clone and its branch come from vm_results_push.sh's own config ($MOE_PUSH_HOME/push.env, default
# ~/moe/push.env; the clone is $MOE_PUSH_HOME/push). Plumbing only, with an index file of its own until the reset.
# EXIT 0 pushed (or nothing to push); 1 a chunk was not pushed (what was pushed stays, the next run continues from
# it); 2 refused (no config, no remote tip, or the remote tip is not an ancestor of HEAD).
set -euo pipefail
PUSH_HOME="${MOE_PUSH_HOME:-$HOME/moe}"
CONF="$PUSH_HOME/push.env"
WORK="$PUSH_HOME/push"
LIMIT="${MOE_PUSH_CHUNK_BYTES:-1200000000}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    *) echo "[chunks] REFUSED: unknown option $1" >&2; exit 2 ;;
  esac
done
say() { echo "[chunks] $*"; }
refuse() { echo "[chunks] REFUSED: $*" >&2; exit 2; }
[[ -f "$CONF" ]] || refuse "no $CONF (vm_results_push.sh init writes it)"
# shellcheck disable=SC1090
. "$CONF"
[[ -n "${BRANCH:-}" && -d "$WORK/.git" ]] || refuse "$CONF names no BRANCH, or $WORK is not a repository"
[[ "$LIMIT" =~ ^[0-9]+$ && "$LIMIT" -gt 0 ]] || refuse "--limit takes a positive byte count"
g() { git -C "$WORK" "$@"; }

g fetch -q origin "refs/heads/$BRANCH" 2>/dev/null || refuse "origin has no branch $BRANCH (the first push is vm_results_push.sh check)"
BASE="$(g rev-parse FETCH_HEAD)"
HEAD="$(g rev-parse HEAD)"
TREE="$(g rev-parse 'HEAD^{tree}')"
if [[ "$(g rev-parse "$BASE^{tree}")" == "$TREE" ]]; then say "origin's $BRANCH already holds HEAD's tree: nothing to push"; exit 0; fi
g merge-base --is-ancestor "$BASE" "$HEAD" || refuse "origin's $BRANCH ($BASE) is not an ancestor of HEAD ($HEAD)"
MSG="$(g log -1 --format=%B HEAD)"

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
export GIT_INDEX_FILE="$TMPD/index"
g read-tree "$BASE"
# what HEAD adds or changes against the remote tip: "size<TAB>mode sha<TAB>path"; deletions: "path"
g diff-tree -r --no-renames -z "$BASE" "$HEAD" | python3 -c '
import subprocess, sys
work, out_add, out_del = sys.argv[1:4]
f = sys.stdin.buffer.read().split(b"\0")
adds, dels = [], []
i = 0
while i + 1 < len(f) and f[i]:
    meta, path = f[i].decode(), f[i + 1].decode()
    i += 2
    _, new_mode, _, new_sha, status = meta[1:].split(" ")
    if status == "D":
        dels.append(path)
    else:
        adds.append((new_mode, new_sha, path))
sizes = {}
if adds:
    q = "".join(sha + "\n" for _, sha, _ in adds).encode()
    r = subprocess.run(["git", "-C", work, "cat-file", "--batch-check=%(objectname) %(objectsize)"], input=q,
                       capture_output=True, check=True)
    sizes = dict(line.split() for line in r.stdout.decode().splitlines())
with open(out_add, "w") as fa:
    for mode, sha, path in adds:
        fa.write(f"{sizes[sha]}\t{mode} {sha}\t{path}\n")
with open(out_del, "w") as fd:
    for path in dels:
        fd.write(path + "\n")
' "$WORK" "$TMPD/add" "$TMPD/del"
say "HEAD $HEAD against origin's $BRANCH $BASE: $(wc -l < "$TMPD/add" | tr -d ' ') files to add or change" \
    "($(awk -F'\t' '{s+=$1} END{printf "%.0f", s}' "$TMPD/add") bytes), $(wc -l < "$TMPD/del" | tr -d ' ') to delete; limit $LIMIT bytes a chunk"

parent="$BASE"; acc=0; n=0; batch=()
push_one() {   # push_one MESSAGE: commit the index on $parent and push it to the run's branch (a fast-forward)
  local t c
  t="$(g write-tree)"
  c="$(printf '%s\n' "$1" | g commit-tree "$t" -p "$parent")"
  n=$((n + 1))
  if ! g push -q origin "$c:refs/heads/$BRANCH"; then
    say "chunk $n NOT PUSHED ($c); origin's $BRANCH stays at $parent"
    exit 1
  fi
  say "pushed chunk $n: $c ($acc bytes)"
  parent="$c"; acc=0; batch=()
}
flush() {
  [[ ${#batch[@]} -gt 0 ]] || return 0
  printf '%s\n' "${batch[@]}" | g update-index --index-info
  push_one "${RUN_ID:-run}: part $((n + 1)) of a push too large for one pack (the last part carries the push's tree)"
}
while IFS=$'\t' read -r size modesha path; do
  if (( acc + size > LIMIT && ${#batch[@]} > 0 )); then flush; fi
  batch+=("$modesha"$'\t'"$path")
  acc=$((acc + size))
done < "$TMPD/add"
# the last part: what is left, the deletions, and HEAD's message
if [[ ${#batch[@]} -gt 0 ]]; then printf '%s\n' "${batch[@]}" | g update-index --index-info; fi
while IFS= read -r path; do
  printf '0 0000000000000000000000000000000000000000\t%s\n' "$path"
done < "$TMPD/del" | g update-index --index-info
[[ "$(g write-tree)" == "$TREE" ]] || { echo "[chunks] FAIL: the chunks' tree is not HEAD's tree" >&2; exit 1; }
push_one "$MSG"
unset GIT_INDEX_FILE
g reset -q --soft "$parent"
say "origin's $BRANCH is at $parent in $n commits, tree $TREE (HEAD's); the local branch now points there"
