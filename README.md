# run-gh200-blind-ling-mini-2.0-20261120t175642z: results pushed by the VM itself

Run `blind-ling-mini-20-20261120t175642z`. Every file under `vm/` is a copy of the rented VM's own,
pushed by `scripts/vm_results_push.sh` after each step of
`scripts/gh200_model_session.sh` (docs/LAMBDA.md section 3c). Nothing here is
curated: this branch is the raw record the publish step reads from.

- measured commit: `unknown` (`vm/session/commit.txt`)
- `vm/results/` is `~/moe/results`, `vm/session/` is `~/moe/session`
- `SHA256SUMS`: the manifest of every file under `vm/` as this push committed
  it (the VM's files, copied with `cp -p` between two steps, when nothing is
  measuring). Check a fetched tree with `sha256sum -c SHA256SUMS`
  (`shasum -a 256 -c` on macOS): it prints `OK` for every line.
- `HELD-BACK.txt`: files over the size limit (compressed here, or left on the VM)
- `PUSHES.txt`: one line per push; `DRIVER-DONE` appears with the last one
- the driver's own ledger: `vm/session/gh200-driver/status`
