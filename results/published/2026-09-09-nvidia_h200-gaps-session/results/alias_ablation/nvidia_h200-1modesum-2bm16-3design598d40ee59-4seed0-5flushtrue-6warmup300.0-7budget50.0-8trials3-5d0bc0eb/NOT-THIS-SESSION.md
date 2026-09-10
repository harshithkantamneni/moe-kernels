# This directory is NOT the sum-mode half of the 2026-09-09 arm

It rode along from an earlier pod's `/workspace` and was committed with the
session tree. Nothing in it was measured, and nothing in it is from the
2026-09-09 H200 gaps session. Its own `provenance.json` says so on four fields:

| field | this directory | the 2026-09-09 session |
|---|---|---|
| `git_sha` | `1833c29` | `981a42f` (arms), `ab61e55` (calibration) |
| `hostname` | `be9a8c80d0d5` | `238b79f71e94` |
| `utc` | `2026-09-09T03:17:26+00:00` | `2026-09-09T16:47:55+00:00` |
| `ridge_source` | 712.3 TFLOP/s over 4374.8 GB/s = 162.8 FLOP/byte | 668.5 over 4374.5 = 152.8 |

The 712.3 TFLOP/s roof and the 162.8 FLOP/byte ridge in that `ridge_source`,
and the 4470 GB/s read roof `report.md` line 7 quotes, were all superseded by
the calibration this session published at `ab61e55`: 668.5 TFLOP/s at 1485 MHz
under a 700 W cap, ridge 152.8 with a 144.9-152.8 band, and a `read_stream`
roof of 4613 GB/s under the pattern names `calibrate` has written since
2026-09-02. Every figure on the page below is scored against a ruler this card
no longer has.

`report.md` here is a PLAN. It ends "Nothing was measured", there is no
`cells.jsonl`, and there are no gate RESULT lines: the arm printed its
prediction, its design, its cost and its MDE and stopped. Read as a plan it is
still correct arithmetic; read as a result it is nothing.

The 2026-09-09 session's alias arm wrote no sum-mode directory at all. It
probed six pinnings, no sum pinning cleared the read roof (best 5500 GB/s of
4613), `--dot-fallback allow` took the fastest dot pinning, and `main` moved
`out_dir` to the dot run before anything was saved. The arm's one output
directory is `nvidia_h200-1modedot-2bm16-...-f98327f2` beside this one.

Established by the 2026-09-09 invalid-arms audit; see
`docs/POD_RUNBOOK.md` and the dot directory's `report.md` for the arm itself.
