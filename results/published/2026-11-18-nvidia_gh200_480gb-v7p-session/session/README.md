# Model v7 prefill blind rental: one NVIDIA GH200 480GB, 14 units of the registered plan (2026-11-18-v7-prefill-blind-gh200)

One rental ran `scripts/plans/rentalV7P-prefill-blind-2026-11.plan` unattended through `scripts/gh200_v7p_session.sh`
(B3's driver `scripts/gh200_tileB3_session.sh`, pinned, plus lines marked V7P: the census gate `scripts/v7p_gate.py`),
started by `scripts/vm_run_v7p.sh` (`vm_run_v7b.sh` plus lines marked V7P). It set the GPU power limit to 700 W and
locked 1410 MHz (C2's prelude, B3's), ran calibrate's ruler, a compile-only CAL census of the three models with its
gate (v7pcensus), OLMoE-1B-7B's and Mixtral-8x7B's 128x128x64 w8 s3 byte pages (f128, the board's sag par), the band
and sag gates (G-BAND, b3sag), then B3's timed pages at 128x128x64 w8 s3 on Granite-3.0-1B-A400M, Granite-3.0-3B-A800M
and Qwen2-57B-A14B-tp8: a G 1 page at two treads and a G 16 page at n 9 a model, PT, PW, PT2, PW2 (OLMoE's fitted
shapes), uniform and balanced, NATIVE and SHARED arms at 9 declared copies; 54 NATIVE cells, 18 a model. The
registration (`docs/registered/2026-11-18-v7-prefill-blind-gh200.{txt,json}`, sha256 998a8c10..., REGISTERED with the
owner's go recorded) froze model v7 unchanged (every switch on, RR at G 1, round 5's prefill branch; the refit4 fix
registered verbatim, no price changed) and its 54 predictions at four reference pars before any page of the three
models, to be finalised at this board's own 128x128 par. It is the THIRD reopening of prefill and OVERRIDES the v7
gate's registered PREFILL NO-GO, with the gate's own pass estimate printed beside it (P(A) 0.2067, B 0.284, D 0.1885
at B2's bar). It was public at the push of 43a8802b (the registration, `--final`), and the run commit is 8c9e29a2
(a test-only commit on top of it), pushed before the 22:02:55Z launch (the guardian launches only from a clean,
pushed checkout, and the VM cloned 8c9e29a2 from GitHub at setup).

Every reading below is the registered scorer's, run as registered from the repository root with no post-page fix:
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/scoring/v7pblind/score_blind_v7p.py results/published/2026-11-18-nvidia_gh200_480gb-v7p-session <out>`
(it finalises and scores in one step, and refuses a dirty checkout). Its output is committed as
`scripts/scoring/v7pblind/prefill_blind.score.{json,txt}`, scored from a clean checkout with this directory committed
(provenance `git_dirty` false). Scored first on the fetched tree's `vm/` from a clean checkout of the run commit
8c9e29a2: the same verdicts, bars and cells; the JSON differs only in the session path and the recorded checkout.
`scripts/scoring/rescore.py v7pblind` re-scores this directory to the committed files. The notes are in
`docs/registered/README.md` ("RESULT 2026-11-18" under the v7 prefill blind) and `docs/FINDINGS.md` (Model v7
prefill blind).

## Where it came from

Lambda Cloud instance 365b52c0 (one GH200), launched 22:02:55Z on 2026-11-18, termination confirmed 23:28:45Z: about
86 min, about $3.70 at Lambda's list price of $2.59/h (about $3.28 at the registration's d2common.RATE of $2.29/h; an
estimate: the instance and the two times are the rental's records, not in the run files, and the bill is in neither).
Three earlier launch attempts, at 19:35:28Z, 19:44:36Z and 21:44:37Z, all in us-east-3, were refused for capacity
(insufficient-capacity, no instance created); the guardian's log names a region only for those refused attempts and
does not record the region of the launch that ran. The registration planned 84 unit-min and about 125 min of wall,
$4.77 at $2.29/h ($5.40 at $2.59/h), with a deadline of launch + 140 min and a hard cap of launch + 155 min ($5.92 /
$6.69 at the cap). The VM's setup ran 22:08:40Z (the guardian's STARTED) to 22:11:36Z (READY, 7 preflight checks
passed) and the driver 22:11:37Z to 23:27:11Z (driver end exit 0). The last push (DRIVER-DONE) was verified at
23:27:27Z, every one of the 335 files under the branch's `vm/` matching its SHA256SUMS, and termination was requested
then; the run's deploy key was deleted after the termination was confirmed. Board `8d8469` (`PREFLIGHT.txt`), new to
the study (no earlier published session names it). The plan's sha256 8b066bf7... is on the ledger's `plan:` line and
equals the committed plan's; the run commit 8c9e29a2 is in `commit.txt` and on the ledger's `driver start` line. This
directory is the branch `run-gh200-v7p-20261118t2202`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its
manifest and push log. Driver 580.105.08 (the VM booted 570.148.08; `vm_run` upgraded it and rebooted before setup,
the owner's decision of 2026-09-26), persistence on.

## The power limit, the lock and the gates

The scorer reads nothing past its power check unless `gh200-driver/c2-power.env` reads C2_POWER_W=700,
`gh200-driver/locks.env` LOCK_PROBE_1410=held, G-BAND PASS, and the census (v7pcensus) and sag (b3sag) gate records
PASS. Each does. `power-set.txt` reads a GPU Current Power Limit of 700.00 W (Requested 700.00 W; the module scope
reads 1000 W), and the driver's `clocks:` line after every unit reads 700.00 W. `locks.env` reads LOCK_PROBE_1410=held.
G-BAND PASS on its registered fallback band [1352.19, 1410.0] MHz (the primary band [1374.75, 1410.0] left OLMoE's
page with no cell in band, so its joint fit was None; the registration names the fallback for that case). The census
gate (v7pcensus) PASS: all six model-and-tread entries at 128x128x64 w8 s3 read 2 CTAs per SM, no MMA path mismatch,
no drift. The sag gate (b3sag) PASS with par A 64.325 us, tau 612.5 us (OLMoE 46.37 / 550, rms 6.66; Mixtral 82.28 /
675, rms 14.0).

Every timed page held the 1410 MHz lock (R3 exit 0 on all six, worst 1410.0 MHz on every page). No page was held out
and no cell voided: every page is counted in ALL and in CLEAN (rule 3), and the G 1 pages read 0 of 216 repeat rows
non-clean.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | start to end (UTC) | min | exit | page |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 1410 MHz, 700 W | 22:11:37 to 22:11:40 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b tp (the ruler) | 22:11:42 to 22:12:07 | 0 | 0 | triad 3726.2 GB/s |
| 3 | tilecensus | tp, 128x128x64w8s3, the three models | 22:12:09 to 22:12:18 | 0 | 0 | |
| 4 | gate | gcen (v7pcensus) | 22:12:20 to 22:12:21 | 0 | 0 | PASS, 2 CTAs/SM |
| 5 | bytes | olmoe-1b-7b f128 | 22:12:23 to 22:15:35 | 3 | 0 | |
| 6 | bytes | mixtral-8x7b f128 | 22:15:37 to 22:22:34 | 7 | 0 | |
| 7 | gate | gband (G-BAND) | 22:22:37 to 22:22:38 | 0 | 0 | PASS (fallback band) |
| 8 | gate | gsag (b3sag) | 22:22:40 to 22:22:40 | 0 | 0 | PASS, par 64.325 / 612.5 us |
| 9 | timed | granite-3.0-1b-a400m t128g1 | 22:22:42 to 22:36:41 | 14 | 0 | R3 cells=216 |
| 10 | timed | granite-3.0-3b-a800m t128g1 | 22:36:44 to 22:50:33 | 14 | 0 | R3 cells=216 |
| 11 | timed | qwen2-57b-a14b-tp8 t128g1 | 22:50:35 to 23:04:44 | 14 | 0 | R3 cells=216 |
| 12 | timed | granite-3.0-1b-a400m t128g16 | 23:04:47 to 23:12:05 | 7 | 0 | R3 cells=108 |
| 13 | timed | granite-3.0-3b-a800m t128g16 | 23:12:08 to 23:19:36 | 7 | 0 | R3 cells=108 |
| 14 | timed | qwen2-57b-a14b-tp8 t128g16 | 23:19:39 to 23:27:08 | 7 | 0 | R3 cells=108 |

14 units in about 76 min against 84 estimated; none dropped; every unit is nodrop.

## What the scorer reads (`scripts/scoring/v7pblind/prefill_blind.score.txt`)

Of the 54 registered NATIVE cells, 30 are dropped by the registered host-bound rule (G2: v3 + F240 under 0.40 ms):
all 12 G 1 cells of each Granite and the 6 tread-1 G 1 cells of Qwen2-57B-tp8. 24 are scored: each model's 6 G 16
cells and Qwen2-57B-tp8's 6 tread-2 G 1 cells. The error e is (predicted - measured) / measured.

| reading | outcome | numbers |
|---|---|---|
| validity | valid | power 700 W, lock 1410 held, G-BAND PASS (fallback), census PASS, sag PASS |
| par (on-board F1) | [64.325, 612.5] | from the two f128 pages, equal to the b3sag record; nothing unpriced |
| pooled (ALL and CLEAN, rule 3) | FAILS | 24 cells, rms 3.13%, worst -4.72%, mean -2.99% (bars 3% / 8% / 1.5%) |
| granite-3.0-1b-a400m | FAILS -> NOT CERTIFIED: BELOW THE MINIMUM | 6 cells, rms 4.25%, worst -4.72%, mean -4.24%; 6 in CLEAN, under 10 |
| granite-3.0-3b-a800m | FAILS -> NOT CERTIFIED: BELOW THE MINIMUM | 6 cells, rms 3.07%, worst -3.80%, mean -3.00%; 6 in CLEAN, under 10 |
| qwen2-57b-a14b-tp8 | FAILS -> NOT CERTIFIED (own bar FAILS; pooled bar FAILS) | 12 cells, rms 2.43%, worst -3.02%, mean -2.37% |
| G3 | PASS, EQUIVALENT; no stratum voided | 4 cells, pooled mean -0.05%, CI [-0.09%, -0.01%] inside the 0.36% margin |
| B3's E1-TILE (DESCRIPTIVE) | FAILS | printed beside, never the verdict |

Every one of the 24 scored cells is under-priced: e runs from -1.57% to -4.72%. The bias, not the spread, fails the
bars: each model's rms is within 0.1 point of its |mean|.

DESCRIPTIVE beside, no verdict: B2's bar (2% / 5% / 1%) on the same cells FAILS; round 5 (every switch off) on the
same cells reads rms 3.12%, worst -4.72%, mean -2.97%, so v7's switches move nothing here; the G 1 cells read rms
2.33%, mean -2.24% (6, all Qwen2-57B-tp8) and the G 16 cells rms 3.36%, mean -3.25% (18); per model and class, Qwen2-57B-tp8
G 16 reads rms 2.52%, mean -2.50%; with no stratum voided, the bars without G3's voids are the bars.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger, the per-unit exits and the
  consoles; `gh200-driver/c2-power.env`, `locks.env`, `gate-*.json`: the power, lock and gate records
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the `-q -d POWER` snapshots
- `locked-r3/<tag>/` and `logs/locked_r3-*.log`: the lock ladder's record per page
- `results/gaps-nvidia_gh200_480gb-<model>-t128g1/`, `-t128g16/`: the six timed pages
- `results/*-f128-r3-counters/`: the two byte pages (the sag par); `results/*-tile-census-*`: the CAL census
- `results/calibration/`: calibrate's ruler; `measured_nvidia_gh200_480gb.yaml` is its copy
- `census.json`, `census.profiles/`: the preflight's GEMM census; `PREFLIGHT.{txt,json}`, `probe.json`: the preflight
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session)

The timed pages are the blind's three models' first prefill-tile pages; once published they are SEEN at prefill
(the registration's census, computed over the sessions it records, is unchanged). The card's UUID is in
`PREFLIGHT.json`, the census files and `setup_vm.log`, as in every published session (the files are the run branch's,
byte for byte); prose here names the board by its hash. `HELD-BACK.txt` on the branch is empty: nothing was over the
size limit. The Triton caches' compiled `cuda_utils*.so` files (10) are left out, as in every earlier session.
Every other file of the run branch is here, byte for byte (325 files against SHA256SUMS, which lists 335).
