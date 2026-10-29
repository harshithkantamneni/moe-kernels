# run-gh200-tile1-20261029t0323: results pushed by the VM itself

Run `gh200-tile1-20261029T0323Z`. Every file under `vm/` is a copy of the rented VM's own,
pushed by `scripts/vm_results_push.sh` after each step of
`scripts/gh200_model_session.sh` (docs/LAMBDA.md section 3c). Nothing here is
curated: this branch is the raw record the publish step reads from.

- measured commit: `a398c6a0b6d4a158f854b689b32ba4320aa9a98d` (`vm/session/commit.txt`)
- `vm/results/` is `~/moe/results`, `vm/session/` is `~/moe/session`
- `SHA256SUMS`: the manifest of every file under `vm/` as this push committed
  it (the VM's files, copied with `cp -p` between two steps, when nothing is
  measuring). Check a fetched tree with `sha256sum -c SHA256SUMS`
  (`shasum -a 256 -c` on macOS): it prints `OK` for every line.
- `HELD-BACK.txt`: files over the size limit (compressed here, or left on the VM)
- `PUSHES.txt`: one line per push; `DRIVER-DONE` appears with the last one
- the driver's own ledger: `vm/session/gh200-driver/status`
