#!/usr/bin/env python
"""Can BLOCK_SIZE_M=128 ever show FIVE clean memory-bound treads? Arithmetic first.

    python scripts/bm128_depth.py --audit        # the answer, off GPU, from published data
    python scripts/bm128_depth.py --self-test    # four worlds, through the pod's own analysis
    python scripts/bm128_depth.py --dry-run      # the pod plan, the cost, and what it can detect
    python scripts/bm128_depth.py                # the pod run

THE APPARATUS, AND WHAT IT USED TO BE. Seven repairs landed on 2026-09-02, one
per heading below. Each is named where it lives, but a reader arriving here
should know them first.

  THE INSTRUMENT is `moe.bench.timing.time_kernel` and nothing else. Every
  tread used to go through a private loop that created its CUDA events inside
  the timed region, synchronised after every call, never flushed L2 and never
  read a clock -- while the compute reference every tread is classified against
  was measured queue-deep. That let 0.18-0.30 ms of host enqueue time per call
  inside the interval, a per-card bias in the fitted alpha of 8-16% at the
  smallest treads, the same size as the cross-card effect this study
  registered. `Sample` now carries the instrument's own columns and
  `analyse_run` carries them into the fit, so a tread timed below the clock the
  roof was measured at is EXCLUDED and counted rather than fitted.

  THE EXCLUSION REACHES EVERY GATE, not only the fit. It stopped at
  `fit_ladder` for one commit: V2's inversions, V3's slope sequence, V5's
  replication and the membership band were all still taken over the UNEXCLUDED
  medians, so a tread the fit had decided sits on another compute branch could
  still FAIL a VALIDITY gate -- INVALID, nothing quotable -- for the very reason
  the exclusion exists to discount. `analyse_run` now scores `fit_points` and a
  spread taken over them, and prints the whole ladder with the dropped treads
  marked.

  A MISSING POWER CALCULATION IS NOT A REFUTED CLAIM. `mde_lines` refused by
  `raise SystemExit(<str>)`, which exits 1 = CLAIM_FAIL without passing through
  `exit_codes.classify`. `--dry-run --plant-noise 0` returned 1 having measured
  nothing, and a pod run at `--reps 1` spent both ladders and then died between
  the last timing and the first `write_text`, leaving no report at all. It now
  raises the named `MdeNotStateable`, the plan degrades to one "not stateable"
  line as the sibling `tile_cap_test` does, and after the sweep the report is
  written and the exit code still comes from the gates.

  THE SELF TEST runs `analyse_run`, which is the function the pod run calls.
  It used to build `(tread, ms)` pairs by hand, count membership from the
  planted compute slope and call one helper; `compute_reference`, `fit_ladder`
  and every gate below were never touched, so a regression in the path that
  produced the 43.6x reference passed it. Each world now registers a verdict per
  gate AND THE EXIT CODE `classify` must return, which is the only thing the
  session driver can see; `--plant-noise` plants the published spread rather
  than silence, and a zero or over-ceiling spread is REFUSED with the reason.
  Two registrations are deliberately withheld and say so where they are made:
  V5 and C2 in the low-clock world, whose two surviving treads make the spread a
  median of two draws against a ceiling 10% away, so the seed would decide them.
  Everything else, `V1` included, is registered in every world that has it.

  UNDECIDED IS A THIRD ANSWER. `fit_ladder` has six named outcomes and two of
  them mean the sweep LOOKED AND COULD NOT SAY. Read as a blank,
  `undecided_parallel_branch` publishes "no depth" from a fit that named none,
  and `undecided_low_clock` publishes the card as the tile. Both now reach the
  report as UNKNOWN with the outcome printed beside them.

  `C3 the law predicts the depth` IS NOW `V6 fit self-consistency`. It compared
  `n*` computed from `(alpha, B/C)` against the tread count those same numbers
  were fitted on, which agrees by construction for any ladder that is two lines.
  As a CLAIM it read as a confirmed prediction; it is a check on the fit. ITS
  SCOPE IS THE ALPHA IN THE REPORT AND NOTHING ELSE: a report that publishes no
  alpha -- `undecided_parallel_branch`, which is what both cards return here --
  gets a vacuous PASS, because a VALIDITY gate that voids the page over a number
  the page does not contain makes this arm RETRY by construction on every card
  that exists. An alpha the gate cannot evaluate is still withheld.

  EXIT CODES AND THE ONE GREPPABLE LINE come from `moe.bench.exit_codes`, the
  run id from `moe.bench.provenance.run_id`, and `report.json` carries a
  provenance block naming the git sha, the device, the instrument and both
  halves of the roof with their sources.

  EVERY REFUSAL NOW EXITS 2, and until 2026-09-02 seven of them exited 1.
  `raise SystemExit(<str>)` sets `SystemExit.code` to the string, and the
  interpreter turns that into exit 1 -- CLAIM_FAIL, "measured; a pre-registered
  claim was refuted" -- so a renamed export in `block_m_crossing_sweep`, a model
  whose routing cannot form a full tile stack, an `--r-max` below one tile, an
  unreadable calibration, a missing one, and a resume across two cards were all
  ledgered as measured refutations of the depth claim by runs that measured
  nothing. They raise `RefusedBeforeMeasuring` now, which carries
  `code = exit_codes.REFUSED`, and `_refuse` prints the sentence at the raise
  site because an unhandled `SystemExit` carrying an int prints nothing.

  `--dry-run` EXITS 2 REFUSED, NOT 0 DONE, and the repository disagreed with
  itself about this until it was picked. A dry run scores no gate and prints no
  RESULT line, so `classify_text` over its log raises `NoGatesScored`, which is
  what a REFUSED log looks like from there; DONE means "measured; every gate
  PASSED". The `--dry-run` branch carries the full census of which scripts were
  on which side.

  NOTHING IS FOLDED INTO DONE. `--fail-on-gate` used to report a CLAIM_FAIL as
  0, and `--audit` is where it bit: over the published corpus C1 FAILS, the log
  said `RESULT: CLAIM C1 FAIL` and the process said DONE. The flag is retired,
  accepted and ignored; `_exit_code` returns what `classify` returns.

WHY THIS EXISTS, WITH ITS FOUNDING PREMISE RETRACTED ON 2026-09-02. This
paragraph used to open "BLOCK_SIZE_M=128 is the one tile height where the
study's arithmetic-intensity cap straddles the hardware ridge -- cap 150.4
against a calibrated ridge of 145.8 on the A100, 158.6 against 162.8 on the
H200". Both numbers were `2 BM / (b alpha)` at the two published BM=128 fits,
128/0.85078 on the A100 (qwen2 G=64) and 128/0.80718 on the H200
(deepseek-v2-lite G=16), and that expression omits `phi` -- one M-tile's
activation and output traffic, in units of one full weight read -- from its
denominator. WHICH ALPHA GOES INTO THE CORRECTION IS THE WHOLE OF IT,
and `premise_caps` got it wrong twice before this. The first version divided
the retracted cap by `ai_model.lin_overstatement = 1 + phi + delta`. The second
called `ai_model.exact_cap` with the report's `alpha-corrected` in the
`alpha_b` slot, on the strength of a sentence asserting that `alpha-corrected`
IS an alpha_b. It is not. `scripts/block_m_crossing_sweep.py` computes it as
`(slope_memory - activation_slope_ms) / load_ms`, and `load_ms = A + B` is the
fitted LEVEL. Dividing a slope by a level is precisely what makes a quantity
(EXA) rather than a weight miss fraction, which is the thesis of
`moe/bench/ai_model.py` and of the commit that made the mistake one function
over from where it fixed it.

SO THE INPUT IS THE RAW `alpha`, `B / (A + B)`, which the same ladder block
publishes, and which is by definition the estimator (EXA) describes.
`alpha-corrected` cannot be inverted here, and not only because it kept
alpha's denominator: its numerator subtracts `Act1` valued at the
CALIBRATION's streaming bandwidth, while (EXA)'s `phi` is `Act1` valued at the
rate the fit itself achieved, and the two are not the same rate. On the A100
ladder the subtraction actually made, `alpha - alpha-corrected` = 0.03334, is
about half the `phi / (1 + phi)` = 0.06459 the byte model charges. Inverting it
therefore means adding its own correction back, which returns `alpha`.
`alpha-corrected` stays on the page as the number the RETRACTED cap was built
from, and is used for nothing else.

AND `phi` IS THE FUSED LAYER'S, `Act1 / W`, not `ai_model.phi` on one GEMM. The
ladder times `E` experts through `x_perm`, `h_up`, `h_act` and `y_perm`, up and
down; the single-GEMM up-projection phi the second version evaluated is a
different byte model, and bracketing it over the unmeasured `alpha_a` was a
bracket about a parameter the fused model does not contain. The two disagree by
12-14% at the alpha_a = 0 end and by a factor of 17 to 30 at the other.
`scripts/memory_branch_anchor.py` uses `Act1 / W` for the same reason and says
so in `Bracket.phi`. `--audit` prints the result on every run, and it reads:

    A100 qwen2 G=64     alpha 0.88412  phi 0.06905  delta 0  alpha_b 0.87611
                        retracted 150.4 (1.032 of ridge 145.8)
                        corrected 135.4 (0.929 of ridge), factor 1.111
    H200 deepseek G=16  alpha 0.87218  phi 0.12311  delta 0  alpha_b 0.85644
                        retracted 158.6 (0.974 of ridge 162.8)
                        corrected 130.7 (0.803 of ridge), factor 1.214

THE STRADDLE DOES NOT EXIST, AND THE CORRECTED NUMBERS SAY SO BY MORE THAN THE
WITHDRAWN ONES DID: 7.1% and 19.7% below their own cards' ridges, where the
bracket this replaces printed 3.7% and 14.1%. BOTH INPUTS ARE FLOORS, and a
floor on either is a ceiling on the cap, so 135.4 and 130.7 are upper bounds
and the retraction holds a fortiori. `delta = 0` is a floor: the A100 ladder's
own `alpha_upper` pins `D / L = 0.13475`, so its delta is 0.16648, not zero.
`phi = Act1 / W` is a floor: it charges each activation tensor once per M-tile
and counts no N-tile re-read, and a re-read only adds to it. AT THAT LADDER'S
OWN DELTA THERE IS NO CAP AT ALL. The recovered `alpha_b` is 1.023, above 1,
and `ai_model.alpha_b_from_fitted` refuses it by name: the fitted per-tile
slope exceeds one full weight read plus one tile of activations and output, so
it carries traffic the three-term model does not name. `--audit` prints that
refusal beside the cap rather than quietly picking the delta that survives.
Even uncorrected only ONE of the two was ever above its ridge. A corpus
carrying no such alpha prints NOT RECOMPUTABLE and does not void the page; see
`PremiseNotRecomputable`. So this arm cannot be justified by "the cap lands on
the ridge at 128",
and that is a result rather than an inconvenience: it means no cap number
decides whether renting an H200 for five clean treads at BM=128 is worth doing.

WHAT SURVIVES, AND IT IS THE MEASUREMENT RATHER THAN THE ARITHMETIC.
BLOCK_SIZE_M=128 is the tile vLLM's fallback ladder actually runs in every
multi-tile decode cell (up to 32 M-tiles per expert among the arm's vLLM rows;
the 34 sometimes quoted counts cutlass and sglang rows, which run no Triton
tile), and the entire 128 row of the published alpha surface rests on TWO fits,
one per card. And the thing the retracted cap was standing in for is MEASURED
further down, with no ridge and no alpha in it: 19 of the 22 valid published
BM=128 ladders have `B/C` between 0.877 and 1.101 with a median of 0.991, so at
this tile the memory and compute branches ARE one line to about 1%. `B/C` is a
ratio of two fitted slopes; no cap, no ridge and no estimator's alpha enters it,
which is exactly why it survived the correction that took the premise. The
question this arm answers is therefore "can a BM=128 ladder be swept deep enough
to identify a memory branch at all", not "does the cap cross the ridge".

Both of those two fits are broken, and the same arithmetic explains why no third
one exists. This script is that arithmetic, the gates that would have caught the
two, and the pod run that would settle it if the arithmetic said it could be
settled. It does not.

THE MEASUREMENT THE LADDER FIT ACTUALLY MAKES. At `r = n BM` the layer's time is
`t(n) = D + max(A + B n, C n)`: a memory branch of slope `B` (one more M-tile
re-reads a fraction `alpha` of the expert's weights) and a compute branch of
slope `C` (one more M-tile does `BM` more rows of padded arithmetic). A tread is
called memory bound when it stands above `D + C n` by the margin, and
`scripts/block_m_crossing_sweep.py` then DISCARDS the whole memory branch when

    |B / C - 1| <= PARALLEL_BRANCH_TOLERANCE

because two branches within 15% of each other are one line, and a fit that reads
a stretch of the compute branch as a memory branch reports that branch's slope as
alpha. That rejection is right, and it is the thing that closes this experiment.

THE ONE IDENTITY EVERYTHING BELOW TURNS ON. Write `rho` for the ratio of the
achieved compute rate to the achieved memory rate of THIS kernel (its own ridge,
in FLOP per byte), `b` for the bytes in a weight element, `alpha` for the re-read
fraction. Then, per M-tile,

    B = (weight bytes) alpha / BW_achieved        C = (2 BM x flops/row) / peak_achieved

    B / C = alpha b rho / (2 BM) = ridge / ai_cap

and the expert count, the hidden size and the intermediate size CANCEL EXACTLY.
So `B / C` -- which decides both how many treads are memory bound and whether the
fit is allowed to speak at all -- does not depend on the model. Not on mixtral
against qwen2, not on a bigger expert, not on `--r-max`, not on how deep the
ladder is swept. That kills four of the five levers this experiment was handed.

THE TWO ESCAPE ROUTES, and both are closed at BM=128 on this hardware.

  ESCAPE UP  (B/C >= 1.15): the memory branch is the steeper line, nothing ever
      crosses, and EVERY tread is memory bound, so five of them is only a
      question of sweeping to 5 x 128 = 640 rows per expert. It needs

          alpha rho >= (1 + tol) 2 BM / b = 1.15 x 128 = 147.2 FLOP/byte

      Across the 22 published BM=128 ladders with a compute reference that
      survives a LEVEL check, measured `alpha rho` runs 101.8 to 150.4, median
      126.8 -- so the median has to move +16.1% to arrive. TWO ladders clear
      147.2 and NEITHER survives its own diagnostics: the A100 one runs
      backwards at its last tread, and the H200 one has a single memory tread
      and a per-tread slope of 1.01, 1.05, 1.01, 0.97, 1.37, 1.62, 1.31, which
      rises and then falls and so is not describable by any two-line model. On
      the A100 the route is additionally shut by physics rather than by margin:
      alpha would have to be >= 1.010 against that card's calibrated ridge of
      145.8, and alpha > 1 means an extra M-tile costs MORE than reading the
      whole expert once.

  ESCAPE DOWN (B/C <= 0.85): the memory branch is the shallower line, so it is a
      finite PREFIX, and the prefix length is

          n* = [ratio (1 - alpha) / alpha - D/C] / (1 + margin - ratio)

      Maximised under the tolerance constraint, `n` clean treads need
      `rho >= (1 + tol (n - 1)) 2 BM / b`, which at n = 5 is 1.6 x 128 = 204.8
      FLOP/byte against calibrated ridges of 145.8 (A100) and 162.8 (H200). But
      the same constraint has a form that needs NO CARD AT ALL:

          alpha <= (1 - tol) / [(1 - tol) + n (tol + margin)] = 0.85 / 1.70 = 0.500

      charging zero overhead, so 0.500 is a ceiling and not an estimate. The
      lowest activation-corrected alpha this study has measured at ANY block
      size, on either card, in any of 23 arms, is 0.596. Escape down is shut by
      numbers the study already published, with no ridge and no bandwidth in the
      argument.

WHAT IS LEFT IN BETWEEN is the discard band, and BM=128 sits in it: 19 of the 22
valid published ladders have `B/C` between 0.877 and 1.101, and the median is
0.991. At BLOCK_M=128 the two branches ARE the same line, to about 1%. That is
not a near miss to be tuned away -- it is what "the cap sits on the ridge" means,
measured. The levers move the median `B/C` by 3.2% (BLOCK_SIZE_N), 7.5%
(GROUP_SIZE_M) and 8.5% (card); the model moves it 3.4% in the median of eight
same-session mixtral/qwen2 pairs, against the 6.4x a "bigger weights to re-read"
mechanism would need. The requirement is +16.1%, in a direction no lever points.

SO THE ANSWER IS NO, and the deliverable is the arithmetic plus two gates the
study did not have:

  * MONOTONICITY. The A100 fit that the whole 128 row rests on runs BACKWARDS at
    its last tread: 25.8076 ms at 7 tiles, 25.4883 ms at 8. Time falling as rows
    rise is not a mechanism. That single tread is also the ONLY point on that
    ladder's compute branch, so it alone sets the `C` the memory branch is
    compared against.
  * TOLERANCE MARGIN. That comparison then clears the 15% tolerance by 1.0e-4:
    |B/C - 1| = 0.150101. Raise that one tread by 0.010% -- 25.4883 ms to
    25.4909 ms, 0.0026 ms -- and `memory_points` goes from 7 to 0, the memory
    branch is discarded, and the A100's only BLOCK_M=128 alpha ceases to exist.
    A verdict that survives on 0.0026 ms must not print the same word as one
    that survives on 0.3, so every margin here is reported as a number, as a
    fraction of the tolerance, and in units of its own bootstrap spread.

Both gates are run against every published BM=128 ladder by `--audit`, which
needs no GPU and is the evidence for everything above.

WHAT THE POD RUN ADDS IF IT IS RUN ANYWAY. `--dry-run` prints it: the BM=256
reference ladder FIRST (so a poisoned reference costs four cells and not the
whole run), a level check on it before the subject is measured at all, then the
BM=128 ladder measured `--reps` times in round-robin so that an inversion can be
told from noise -- which one pass cannot do, and which is exactly how the A100
ladder shipped. Nothing here can raise `B/C`; the run exists to measure the
margin honestly and to record the refusal.

A NOTE ON REUSE. The ladder fit, the compute reference and the tolerance are
IMPORTED from `scripts/block_m_crossing_sweep.py`, never copied and never
edited: this script has to be judged by the same fit the study publishes, and a
private copy would drift. `compute_reference` grew four required keyword-only
arguments while this file was being written, so `_load_sweep` probes SIGNATURES
and not only names: a rename must produce a sentence on a laptop rather than a
TypeError thirty seconds into a metered pod session.

`reference_level` here OVERLAPS that file's own level checks and is kept
deliberately, so do not delete it as a duplicate. The two do different jobs.
Theirs REFUSES a bad reference at qualification time and is what protects the pod
run; this one REPORTS the implied achieved TFLOP/s as a number and a fraction of
the card's ceiling, which is what `--audit` needs to score the 26 already-published
reports, whose references were qualified before any level test existed and cannot
be re-refused retrospectively. Theirs is three readings and strictly stronger on
live data; this one is one reading and works on a JSON file. If they are ever
merged, the audit path is the one that has to keep working.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.roofline import HARDWARE_DIR, load_hardware  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

#: `moe.bench.timing` is imported LAZILY, in `measure_setting` only, and this
#: comment is the reason. That module imports torch at module scope; this one is
#: documented to run `--audit`, `--self-test` and `--dry-run` on a laptop with
#: no torch at all, and an import here would turn three documented paths into an
#: ImportError before argparse ran. `exit_codes` and `provenance` import nothing
#: heavier than the standard library.


class RefusedBeforeMeasuring(SystemExit):
    """A precondition this run needs was not met, and nothing was measured.

    A `SystemExit` CARRYING `code = exit_codes.REFUSED`, and the code is the
    whole point of the class. Seven refusals in this file used to be
    `raise SystemExit(<str>)`, which sets `SystemExit.code` to the STRING; the
    interpreter prints it and exits 1, and 1 is `CLAIM_FAIL` in the table this
    study adopted -- "measured; a pre-registered claim was refuted". So a
    renamed export in `block_m_crossing_sweep`, a model whose routing cannot
    form a full tile stack, a missing calibration and a resume across two cards
    all reached the session driver as MEASURED REFUTATIONS of the depth claim,
    from runs that had measured nothing at all and spent no pod minutes. The
    ledger then marked the arm finished with a finding in it.

    `code` IS A CLASS ATTRIBUTE ON PURPOSE. `SystemExit.__init__` would set the
    instance's `code` to the message; a subclass attribute of the same name
    shadows that descriptor, so `str(exc)` still returns the sentence and the
    process still exits 2.

    THE MESSAGE IS PRINTED AT THE RAISE SITE by `_refuse`, once, because an
    unhandled `SystemExit` whose code is an int prints NOTHING. A refusal
    nobody can read is not a refusal. `main` therefore returns REFUSED without
    re-printing.

    IT IS A `SystemExit` AND NOT A `RuntimeError` FOR THE DELIVERY, the same
    reason `block_m_crossing_sweep.RetiredInstrument` is. `measure_setting`
    times inside a per-cell `except Exception`, so a `RuntimeError` raised
    anywhere it can reach would be swallowed into a `status="failed"` row and
    the arm would grind through its whole grid before reporting. `SystemExit`
    derives from `BaseException` and is outside that handler.
    """

    code = exit_codes.REFUSED


def _refuse(message: str) -> RefusedBeforeMeasuring:
    """Print the refusal once, then hand back the exception to raise.

    Returning rather than raising keeps `raise _refuse(...)` a `raise` at the
    call site, so the control flow reads normally and a linter still sees the
    function end there. The `REFUSED:` prefix is what the session driver greps
    out of the log when it ledgers the arm.
    """
    print(f"REFUSED: {message}")
    return RefusedBeforeMeasuring(message)


def _load_sweep():
    """Load `block_m_crossing_sweep` BY PATH, and name what is missing.

    `scripts/` is not a package, so a bare import works only when this file is
    the entry point and silently fails when a test loads it by path. Loading by
    path makes both work. The symbol check is not defensive noise: that file is
    under active edit by another workstream, and a renamed `fit_ladder` must
    produce a sentence naming it rather than an AttributeError three frames
    down inside a gate.
    """
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    needed = ("PARALLEL_BRANCH_TOLERANCE", "MEMORY_BRANCH_MARGIN",
              "MIN_MEMORY_TREADS", "FIXED", "ComputeReference", "fit_ladder",
              "ladder_points", "compute_reference", "make_cell", "model_ms",
              "tokens_for_rows", "rows_quantum", "results_root", "scaled_iters",
              "activation_slope_ms", "useful_flops", "_line",
              # Added 2026-09-02 with the instrument. `time_call` is retired
              # over there and raises; the clock the roof was measured at is
              # what `time_kernel` needs before it can say anything about
              # LEVEL, and this file must resolve it the same way the sweep
              # does or its rows are not comparable with the sweep's.
              "reference_clock_mhz", "timing_basis", "ladder_treads")
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise _refuse(
            "scripts/block_m_crossing_sweep.py no longer exports "
            f"{', '.join(missing)}. This script is deliberately scored by that "
            "file's fit rather than a private copy, so the two move together. "
            "Re-point the import; do not fork the fit.")
    # SIGNATURES, not only names. `compute_reference` grew four required
    # keyword-only arguments while this file was being written, and an existence
    # check passed it straight through to a TypeError raised thirty seconds into
    # a metered pod session. Everything callable from the GPU path is probed here
    # instead, on a laptop, with a sentence that names the drift.
    import inspect
    for name, required in (("compute_reference", ("cfg", "ridge",
                                                  "bandwidth_gbps", "b")),
                           ("fit_ladder", ("block_m",)),
                           ("make_cell", ("sm_count", "block_n"))):
        params = inspect.signature(getattr(module, name)).parameters
        gone = [p for p in required if p not in params]
        if gone:
            raise _refuse(
                f"block_m_crossing_sweep.{name} no longer takes "
                f"{', '.join(gone)}. That file is under active edit and this "
                "one calls into it on the pod path; re-check the call sites in "
                "analyse_run and main before spending GPU time.")
    return module


SWEEP = _load_sweep()

TOLERANCE = SWEEP.PARALLEL_BRANCH_TOLERANCE
MIN_MEMORY_TREADS = SWEEP.MIN_MEMORY_TREADS


# --------------------------------------------------------------------------
# The thresholds this script is arguing about, all stated before any code.
# --------------------------------------------------------------------------

#: The subject. Not a parameter: every sentence in the docstring is about 128,
#: and a `--block-m` flag would let a run answer a different question under this
#: script's name.
SUBJECT_BLOCK_M = 128

#: The compute reference. `C ~ BLOCK_M` with no free parameter, so one ladder
#: that is compute bound throughout gives `C` at every block size. 256 is the
#: only block size this study has ever found compute bound at tread 1.
REFERENCE_BLOCK_M = 256

#: The ask: five clean memory-bound treads at the subject block size.
TARGET_TREADS = 5

#: A compute reference must imply at least this fraction of the ATTACHED card's
#: calibrated bf16 ceiling.
#:
#: THE GATE THE STUDY DID NOT HAVE. `compute_reference` qualifies a ladder by
#: PROPORTIONALITY to its tile count and never by LEVEL, and a line 43.6x too
#: steep is still perfectly proportional. The A100 BLOCK_N=256 arm's reference
#: took 249.765 ms for one BLOCK_M=256 tile against 5.724 ms for the identical
#: setting in its BLOCK_N=64 twin; it implies 3.6 TFLOP/s, 1.4% of that card's
#: 262.4, and it classified every tread of every ladder in the arm as compute
#: bound. Its H200 twin implies 87.7 TFLOP/s, 12.3%. The 22 references that are
#: not corrupt imply 38.2% to 63.7%, so 25% separates them with room on both
#: sides and is not fitted to the failures.
REFERENCE_LEVEL_FLOOR = 0.25

#: An inversion bigger than this many across-repeat standard deviations is a
#: fault and not noise. Two sigma rather than three because the direction is
#: known a priori: `t(n)` is `D + max(A + Bn, Cn)` with `B, C > 0`, so time
#: cannot fall as tiles rise under EITHER branch, and a one-sided departure
#: needs less evidence than a two-sided one.
MONOTONE_SIGMA = 2.0

#: A tolerance verdict has to clear the tolerance by this many bootstrap
#: standard deviations before it may be read as a verdict. The published A100
#: fit clears by 0.000101, which is 0.07% of the tolerance and 0.02 sigma.
MARGIN_SIGMA = 3.0

#: Resamples used to put a spread on `|B/C - 1|`. Fixed rather than tuned, and
#: seeded, so two readers of the same ladder get the same margin.
BOOTSTRAP_DRAWS = 2000

#: `max(a + bn, D + cn)` is CONVEX, so its per-tread slope is non-decreasing and
#: takes at most two values. A slope that rises and then falls is not describable
#: by any two-line model -- but a per-tread slope is a DIFFERENCE of two timings,
#: so its noise is amplified by roughly `sqrt(2) n` over the ladder's own spread
#: and reaches 15% by tread 7 on the noisier H200 arms.
#:
#: A FIXED RELATIVE THRESHOLD IS THEREFORE NOT A GATE. At 5% this fired on 20 of
#: the 22 published ladders, which is a detector that has learned to say yes. The
#: threshold is propagated from the run's own timing spread instead, and a drop
#: has to clear BOTH this many standard deviations and the relative floor below.
SLOPE_DROP_SIGMA = 3.0

#: A drop inside this fraction is not worth reporting however small the spread
#: is, because the two-line model's own two slopes differ by more than it.
SLOPE_DROP_FLOOR = 0.05

#: Across-repeat spread above this makes a margin unreadable, whatever it is.
#: The published H200 ladders sit at 0.76-1.82% on ONE pass, the A100 ones at
#: 0.48-0.61%; 2% is a ceiling on the noise a replicated run is allowed to have
#: before its verdict is withdrawn rather than reported.
MAX_REPLICATE_SPREAD = 0.02

#: The per-tread spread this study actually produces, and the default
#: `--plant-noise`. The published H200 ladders run 0.76-1.82% on one pass and
#: the A100 ones 0.48-0.61%; 1.82% is the TOP of the H200 range, which is the
#: end a gate has to survive and the end the self-test therefore plants at.
#:
#: A ZERO DEFAULT IS THE DEFECT THIS REPLACES. The planted worlds were
#: noiseless, so every threshold propagated from a measured spread -- the
#: membership margin, the convexity floor, the margin's own sigma -- was
#: exercised only at 0.00%, the one value no pod produces.
PUBLISHED_LADDER_SPREAD = 0.0182

#: Calibration files, by the substring that identifies the card in a published
#: arm's directory name. Read rather than hardcoded: the A100's ridge is
#: 262.371/1.79936 = 145.81 and the H200's is 712.259/4.37476 = 162.81, and the
#: seven published A100 reports carry 160.3 -- an H200 band belonging to neither
#: card -- precisely because a number like that was written down once.
CALIBRATION_SLUGS = {"a100": "measured_nvidia_a100_sxm4_80gb",
                     "h200": "measured_nvidia_h200"}


# --------------------------------------------------------------------------
# The depth law. Pure arithmetic: no torch, no GPU, no files.
# --------------------------------------------------------------------------

def branch_ratio(block_m: int, b: int, alpha: float, rho: float) -> float:
    """`B / C = alpha b rho / (2 BM)`, the fitted-slope ratio.

    Also `ridge / ai_cap`, which is why it decides everything: below 1 the
    compute branch is steeper and the memory branch is a finite prefix, above 1
    nothing ever crosses, and within `TOLERANCE` of 1 the fit refuses to call
    them two branches at all. The model's E, F and H are not arguments because
    they cancel; see the module docstring.
    """
    return alpha * b * rho / (2.0 * block_m)


def escape_up_alpha_rho(block_m: int, b: int, tol: float = TOLERANCE) -> float:
    """`alpha rho` needed for the memory branch to be the steeper line.

    At or above this the ladder never crosses, every tread is memory bound, and
    depth is bounded only by `--r-max`. 147.2 FLOP/byte at BLOCK_M=128, bf16.
    """
    return (1.0 + tol) * 2.0 * block_m / b


def escape_down_rho(block_m: int, b: int, treads: int,
                    tol: float = TOLERANCE) -> float:
    """Achieved ridge needed for `treads` clean memory treads BELOW the crossing.

    In the prefix regime a tread is memory bound while `n < n*` with
    `n* = rho (1 - alpha) / (2 BM / b - alpha rho)`. `n*` rises with `alpha rho`,
    which the tolerance caps at `(1 - tol) 2 BM / b`; substituting that cap and
    solving `n* >= treads` gives

        rho >= (1 + tol (treads - 1)) 2 BM / b

    which is 204.8 FLOP/byte for five treads at BLOCK_M=128, bf16. The two cards
    in this study calibrate at 145.8 and 162.8.
    """
    return (1.0 + tol * (treads - 1)) * 2.0 * block_m / b


def prefix_depth(ratio: float, alpha: float, overhead_over_c: float,
                 margin: float) -> float | None:
    """How many leading treads stand above the compute branch, from the fit alone.

    A tread is memory bound when `A + B n > D + C n (1 + margin)`. With
    `A = B (1 - alpha) / alpha` and `ratio = B / C` that is

        n < [ratio (1 - alpha) / alpha - D / C] / (1 + margin - ratio)

    None means EVERY tread qualifies, which is a different answer from "many"
    and is what `ratio >= 1 + margin` says. Returns a float on purpose: the
    integer count is `floor` of it, and rounding here would hide how close a
    ladder sits to gaining or losing a tread.
    """
    denom = 1.0 + margin - ratio
    if denom <= 0:
        return None
    if alpha <= 0:
        raise ValueError("alpha must be positive to have a memory branch at all")
    return (ratio * (1.0 - alpha) / alpha - overhead_over_c) / denom


def escape_down_alpha(treads: int, tol: float = TOLERANCE,
                      margin: float = SWEEP.MEMORY_BRANCH_MARGIN) -> float:
    """The largest `alpha` that can give `treads` clean treads below the crossing.

    THE SAME BOUND AS `escape_down_rho`, RESTATED IN THE ONE QUANTITY THIS STUDY
    ACTUALLY MEASURES. `n*` rises with `ratio`, and the tolerance caps `ratio` at
    `1 - tol`, so substituting that cap and dropping the (non-negative) overhead
    term gives the most generous possible reading:

        (1 - tol)(1 - alpha) / alpha >= treads (tol + margin)
        alpha <= (1 - tol) / [(1 - tol) + treads (tol + margin)]

    At five treads that is 0.85 / 1.70 = 0.500. It needs no ridge, no bandwidth
    and no card: the study's own alphas settle it, and the lowest alpha it has
    measured anywhere is 0.596. Charging any overhead at all only lowers the
    bound further, so 0.500 is a ceiling and not an estimate.

    Independent of BLOCK_M, which is the surprising part and is correct: BLOCK_M
    moves `ratio`, and this bound is evaluated at the one `ratio` the tolerance
    allows, so the tile height has already been used up.
    """
    return (1.0 - tol) / ((1.0 - tol) + treads * (tol + margin))


@dataclass(frozen=True)
class DepthVerdict:
    """What the law says about one (block_m, b, alpha, rho) point."""

    block_m: int
    dtype_bytes: int
    alpha: float
    rho: float
    ratio: float
    regime: str
    reachable_treads: float | None
    needed_alpha_rho: float
    needed_rho: float

    @property
    def feasible(self) -> bool:
        return (self.regime == "escape-up"
                or (self.regime == "escape-down"
                    and self.reachable_treads is not None
                    and self.reachable_treads >= TARGET_TREADS))

    def line(self) -> str:
        depth = ("every tread" if self.reachable_treads is None
                 else f"{self.reachable_treads:.2f} treads")
        return (f"alpha {self.alpha:.3f} x rho {self.rho:7.1f} -> "
                f"B/C {self.ratio:.3f}  {self.regime:12s}  {depth:>13s}   "
                f"{'FEASIBLE' if self.feasible else 'no'}")


def depth_verdict(block_m: int, b: int, alpha: float, rho: float, *,
                  overhead_over_c: float = 0.0,
                  margin: float = SWEEP.MEMORY_BRANCH_MARGIN,
                  treads: int = TARGET_TREADS) -> DepthVerdict:
    """The law, evaluated. Three regimes, and the middle one is where 128 lives."""
    ratio = branch_ratio(block_m, b, alpha, rho)
    if abs(ratio - 1.0) <= TOLERANCE:
        # Not "few treads": the fit throws the memory branch away entirely, so
        # the ladder reports NO alpha however many treads stood above the line.
        return DepthVerdict(block_m, b, alpha, rho, ratio, "discarded", 0.0,
                            escape_up_alpha_rho(block_m, b),
                            escape_down_rho(block_m, b, treads))
    if ratio > 1.0:
        return DepthVerdict(block_m, b, alpha, rho, ratio, "escape-up", None,
                            escape_up_alpha_rho(block_m, b),
                            escape_down_rho(block_m, b, treads))
    return DepthVerdict(block_m, b, alpha, rho, ratio, "escape-down",
                        prefix_depth(ratio, alpha, overhead_over_c, margin),
                        escape_up_alpha_rho(block_m, b),
                        escape_down_rho(block_m, b, treads))


# --------------------------------------------------------------------------
# Ladder diagnostics. Everything here takes points and returns a NUMBER; none
# of it decides anything on its own, so a reader can check each separately.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Inversion:
    """One place where time fell as tiles rose."""

    n_lo: int
    n_hi: int
    ms_lo: float
    ms_hi: float
    rel: float
    sigma: float | None

    def line(self) -> str:
        sig = "spread unknown" if self.sigma is None else f"{self.sigma:.1f} sigma"
        return (f"n={self.n_lo}->{self.n_hi}  {self.ms_lo:9.4f} -> "
                f"{self.ms_hi:9.4f} ms  {self.rel:+.3%}  ({sig})")


def inversions(points, spread: float | None) -> list[Inversion]:
    """Every tread whose time is below its predecessor's.

    Under `t(n) = D + max(A + Bn, Cn)` with both slopes positive, time is
    strictly increasing in `n` on BOTH branches. So an inversion is never a
    mechanism; it is a clock, a thermal ramp, a cache that was warm for one
    point and not the next, or a single unlucky median. It matters because OLS
    hands the last tread the most leverage on the slope, and the slope IS alpha:
    on the published A100 ladder the inverted tread is also the only point on
    the compute branch, so it sets both lines at once.

    `spread` is the relative timing spread the run measured. None is carried
    through as None rather than replaced by a guess, because "we do not know how
    noisy this was" and "this was 0% noisy" are different states and only one of
    them lets an inversion be dismissed.
    """
    out = []
    for (n_lo, lo), (n_hi, hi) in zip(points, points[1:], strict=False):
        if hi >= lo:
            continue
        rel = hi / lo - 1.0
        out.append(Inversion(n_lo, n_hi, lo, hi, rel,
                             abs(rel) / spread if spread else None))
    return out


def slope_sequence(points) -> list[float]:
    """Per-tread marginal cost, `(t(n+1) - t(n)) / (n+1 - n)`."""
    return [(hi - lo) / (n_hi - n_lo)
            for (n_lo, lo), (n_hi, hi) in zip(points, points[1:], strict=False)]


@dataclass(frozen=True)
class SlopeDrop:
    index: int
    before: float
    after: float
    rel: float
    sigma: float | None

    def line(self) -> str:
        sig = "spread unknown" if self.sigma is None else f"{self.sigma:.1f} sigma"
        return (f"slope {self.before:.4f} -> {self.after:.4f} ms/tile after "
                f"tread {self.index + 2}  {self.rel:+.1%}  ({sig})")


def _slope_drop_sd(points, i: int, spread: float) -> float:
    """Propagated sd of `slope[i+1] - slope[i]` under multiplicative timing noise.

    Three timings are involved and the middle one appears in BOTH slopes with
    opposite sign, so its variance ADDS rather than cancelling. Writing
    `d1 = n[i+1] - n[i]` and `d2 = n[i+2] - n[i+1]`,

        Var = s^2 [ (t0^2 + t1^2)/d1^2 + (t1^2 + t2^2)/d2^2 + 2 t1^2/(d1 d2) ]

    Getting the covariance term wrong understates this by about a third, which
    is the difference between a gate that fires on one ladder and one that fires
    on twenty.
    """
    (n0, t0), (n1, t1), (n2, t2) = points[i], points[i + 1], points[i + 2]
    d1, d2 = float(n1 - n0), float(n2 - n1)
    var = spread ** 2 * ((t0 ** 2 + t1 ** 2) / d1 ** 2
                         + (t1 ** 2 + t2 ** 2) / d2 ** 2
                         + 2.0 * t1 ** 2 / (d1 * d2))
    return math.sqrt(var)


def slope_drops(points, spread: float | None,
                sigma_gate: float = SLOPE_DROP_SIGMA,
                floor: float = SLOPE_DROP_FLOOR) -> list[SlopeDrop]:
    """Where the ladder stops being convex, which max-affine cannot be.

    `max(A + Bn, D + Cn)` is a maximum of two increasing lines, so its slope is
    non-decreasing and takes at most two values. A slope that RISES and then
    FALLS is not that shape at all, and a two-line fit through it reports the
    slope of a curve rather than of a branch.

    A drop must clear the relative floor AND the propagated noise. With `spread`
    None there is no noise model, so every drop over the floor is returned with
    `sigma=None` and the gate reads UNKNOWN rather than passing.

    WHAT THIS GATE CANNOT DO ON THE PUBLISHED H200 LADDERS, said here because the
    obvious reading of it is wrong. H200 mixtral at GROUP_SIZE_M=64 runs
    1.01, 1.05, 1.01, 0.97, 1.37, 1.62, 1.31 ms per tile -- a 19.3% fall -- and
    this gate does NOT fire on it, because at that arm's 1.71% timing spread the
    propagated sd of a slope difference at tread 7 is 0.34 ms/tile against a
    slope of 1.3, so 19.3% is 0.9 sigma. That is not the gate being lenient; it
    is the measurement being unable to resolve a per-tread slope at all. A
    single-pass BLOCK_M=128 ladder on the H200 cannot distinguish a bend from
    noise, which is the same reason it cannot distinguish two branches, and it is
    why `--reps` exists.

    The spread used is the run's WITHIN-call spread, which is smaller than the
    across-pass spread a replicated run would report, so every sigma here is an
    over-estimate and this gate is conservative in the direction of not firing.
    """
    slopes = slope_sequence(points)
    out = []
    for i, (a, c) in enumerate(zip(slopes, slopes[1:], strict=False)):
        if a <= 0 or c >= a * (1.0 - floor):
            continue
        sd = _slope_drop_sd(points, i, spread) if spread else None
        sig = abs(c - a) / sd if sd and sd > 0 else None
        if sig is not None and sig < sigma_gate:
            continue
        out.append(SlopeDrop(i, a, c, c / a - 1.0, sig))
    return out


@dataclass(frozen=True)
class Margin:
    """How far a memory branch stands from being the compute branch again."""

    ratio: float
    distance: float
    margin: float
    sd: float | None
    sigma: float | None
    #: 'replicates' or 'parametric spread s=...'. Never empty: a margin whose
    #: noise model is unstated is a margin nobody can weigh.
    basis: str
    draws: int

    @property
    def clears(self) -> bool:
        return self.margin > 0

    @property
    def confident(self) -> bool:
        return self.clears and self.sigma is not None and self.sigma >= MARGIN_SIGMA

    def line(self) -> str:
        sig = "sigma unknown" if self.sigma is None else f"{self.sigma:.2f} sigma"
        return (f"B/C {self.ratio:.6f}  |B/C - 1| {self.distance:.6f}  "
                f"tolerance {TOLERANCE:.2f}  margin {self.margin:+.6f} "
                f"({self.margin / TOLERANCE:+.2%} of the tolerance, {sig})")


def _refit_ratio(points, k: int, compute_points, c_ref: float | None,
                 overhead: float) -> float | None:
    """`B / C` for a FIXED branch assignment, which is what a bootstrap needs.

    Membership is held at `k` across resamples on purpose. Letting it move would
    make the resampled quantity a mixture of "this slope under noise" and "a
    different set of treads", and the margin is a statement about the first.
    """
    if k < 2:
        return None
    _, b = SWEEP._line([float(n) for n, _ in points[:k]],
                       [ms for _, ms in points[:k]])
    c = c_ref
    if compute_points:
        c = SWEEP._through_origin([float(n) for n, _ in compute_points],
                                  [ms - overhead for _, ms in compute_points])
    if not c:
        return None
    return b / c


def margin_of(points, k: int, *, c_ref: float | None, overhead: float,
              spread: float | None = None,
              replicates: dict[int, list[float]] | None = None,
              draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> Margin:
    """`|B/C - 1| - tolerance`, with a spread on it.

    REFUSES rather than returning a bare pass. With neither replicates nor a
    measured spread there is no noise model, `sd` and `sigma` come back None,
    and the gate that reads this reports UNKNOWN. Returning 0.0 for an
    unmeasured spread would turn every knife-edge verdict into a confident one,
    which is the exact failure this whole file is about.

    With replicates the resample is nonparametric: each tread's value is drawn
    from its own repeated measurements. Without them it is parametric --
    lognormal at the run's own median relative spread -- and `basis` says so,
    because a parametric interval on one pass cannot see a drift that repeated
    passes would have caught.
    """
    compute_points = points[k:]
    ratio = _refit_ratio(points, k, compute_points, c_ref, overhead)
    if ratio is None:
        return Margin(float("nan"), float("nan"), float("nan"), None, None,
                      "no memory branch: fewer than 2 treads above the compute "
                      "branch, so there is no slope to compare", 0)
    distance = abs(ratio - 1.0)
    base = Margin(ratio, distance, distance - TOLERANCE, None, None, "", 0)

    if replicates:
        basis = f"replicates ({min(len(v) for v in replicates.values())}+ per tread)"
    elif spread and spread > 0:
        basis = f"parametric spread sigma={spread:.4f} (ONE pass, no replicates)"
    else:
        return Margin(ratio, distance, distance - TOLERANCE, None, None,
                      "no noise model: the run recorded neither replicates nor a "
                      "timing spread, so this margin cannot be weighed", 0)

    rng = random.Random(seed)
    dists = []
    for _ in range(draws):
        drawn = []
        for n, ms in points:
            if replicates and replicates.get(n):
                drawn.append((n, rng.choice(replicates[n])))
            else:
                drawn.append((n, ms * math.exp(rng.gauss(0.0, spread or 0.0))))
        # The reference slope carries the same noise as the ladder, so it is
        # resampled too. Holding it fixed understates the spread and would make
        # the margin gate more lenient than the data supports.
        c = c_ref * math.exp(rng.gauss(0.0, spread or 0.0)) if c_ref else None
        r = _refit_ratio(drawn, k, drawn[k:], c, overhead)
        if r is not None:
            dists.append(abs(r - 1.0))
    if len(dists) < 2:
        return Margin(ratio, distance, base.margin, None, None,
                      basis + "; every resample lost the branch, so no spread "
                              "could be formed", len(dists))
    sd = statistics.pstdev(dists)
    sigma = (distance - TOLERANCE) / sd if sd > 0 else None
    return Margin(ratio, distance, distance - TOLERANCE, sd, sigma, basis,
                  len(dists))


@dataclass(frozen=True)
class RefLevel:
    """Is the compute reference a compute branch, or just a straight slow line."""

    block_m: int
    slope_ms_per_tile: float
    implied_tflops: float
    ceiling_tflops: float
    fraction: float
    source: str

    @property
    def passes(self) -> bool:
        return REFERENCE_LEVEL_FLOOR <= self.fraction <= 1.0

    def line(self) -> str:
        return (f"BLOCK_M={self.block_m} reference {self.slope_ms_per_tile:.4f} "
                f"ms/tile implies {self.implied_tflops:7.1f} TFLOP/s = "
                f"{self.fraction:6.1%} of {self.ceiling_tflops:.1f} "
                f"({self.source})")


def reference_level(cfg, block_m: int, slope_ms_per_tile: float,
                    ceiling_tflops: float, source: str) -> RefLevel:
    """Turn a reference slope into achieved TFLOP/s and compare it to the card.

    One M-tile per expert at `block_m` rows is `E BM` padded rows and
    `6 E BM F H` flops, so the slope names an achieved rate directly. The whole
    point is that this is a LEVEL and not a shape: `compute_reference` already
    checks that the reference ladder is proportional to its tile count, and the
    A100 BLOCK_N=256 reference was proportional to 0.2% while being 43.6x too
    slow.
    """
    flops = SWEEP.useful_flops(cfg, cfg.num_experts * block_m)
    implied = flops / (slope_ms_per_tile * 1e-3) / 1e12
    return RefLevel(block_m, slope_ms_per_tile, implied, ceiling_tflops,
                    implied / ceiling_tflops if ceiling_tflops > 0 else float("inf"),
                    source)


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

VALIDITY, CLAIM = "VALIDITY", "CLAIM"


@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `passed=None` prints UNKNOWN and never PASS. `invalidates` is required on a
    VALIDITY gate and says what may not be quoted if it fails, because a failed
    gate whose consequence is unstated gets read as a warning.
    """

    kind: str
    name: str
    prediction: str
    rule: str
    passed: bool | None
    observed: str
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def tag(self) -> str:
        """The one-token name on the RESULT line: `V1`, `C3`, `S escape-up`.

        `name` is "V1 reference level" and `moe.bench.exit_codes.result_line`
        requires one run of non-whitespace, so the leading token is the name and
        the rest is prose. A gate whose name has no leading token would silently
        collide with another's on a driver's grep, so the whole name is
        squashed rather than truncated in that case.
        """
        parts = self.name.split()
        if not parts:
            return "gate"
        # `V1 reference level` -> `V1`. A leading token that is a letter-digit
        # code is the name; anything else (`S escape-up`) is hyphen-joined
        # whole, because three gates called `S` would collide on a driver's
        # grep and a collision is a gate that silently disappears.
        if re.fullmatch(r"[A-Za-z]+\d+", parts[0]):
            return parts[0]
        return "-".join(parts)

    def scored(self) -> tuple[str, str, str]:
        """`(kind, tag, verdict)` in `moe.bench.exit_codes`'s vocabulary.

        `passed=None` is UNKNOWN, never PASS, and UNKNOWN counts AGAINST the
        gate on both kinds: on a VALIDITY gate it means the instrument's
        soundness could not be shown, which is exactly as unquotable as a FAIL.
        """
        verdict = {True: exit_codes.PASS, False: exit_codes.FAIL,
                   None: exit_codes.UNKNOWN}[self.passed]
        return (exit_codes.VALIDITY if self.kind == VALIDITY
                else exit_codes.CLAIM, self.tag, verdict)

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        The `[PASS] VALIDITY ...` line below it is for a human and for
        `scripts/h200_gaps_session.sh`, whose `gate_from_log` has matched that
        shape since 2026-09-01; both are kept because they have different
        readers, and only this one is the machine contract. Nothing else this
        file prints starts with `RESULT: `.
        """
        detail = (f"[{self.kind}] {self.prediction} | saw {self.observed} "
                  f"| gate {self.rule}")
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> list[str]:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[self.passed]
        out = [self.result_line(),
               f"[{tag}] {self.kind:8s} {self.name}  {self.prediction}",
               f"         gate: {self.rule}",
               f"         saw:  {self.observed}"]
        if self.passed is not True and self.invalidates:
            out.append(f"         a FAIL here invalidates: {self.invalidates}")
        out += [f"         {line}" for line in self.lines]
        return out


def render_gates(gates: list[Gate]) -> list[str]:
    out: list[str] = []
    for g in gates:
        out += g.render()
    npass = sum(1 for g in gates if g.passed is True)
    nfail = sum(1 for g in gates if g.passed is False)
    nunk = sum(1 for g in gates if g.passed is None)
    out += ["", f"{npass} PASS, {nfail} FAIL, {nunk} UNKNOWN"]
    return out


def gate_reference_level(level: RefLevel | None) -> Gate:
    if level is None:
        return Gate(VALIDITY, "V1 reference level",
                    "the compute reference runs at a plausible rate",
                    f"implied TFLOP/s in [{REFERENCE_LEVEL_FLOOR:.0%}, 100%] of "
                    "the attached card's calibrated bf16 ceiling",
                    None, "no compute reference was qualified at all",
                    "every membership decision, hence every alpha and every "
                    "tread count in this report")
    return Gate(VALIDITY, "V1 reference level",
                "the compute reference runs at a plausible rate",
                f"implied TFLOP/s in [{REFERENCE_LEVEL_FLOOR:.0%}, 100%] of the "
                "card's calibrated bf16 ceiling",
                level.passes, level.line(),
                "every membership decision, hence every alpha and every tread "
                "count in this report: a reference too slow puts its line above "
                "every tread and calls the whole ladder compute bound",
                ["`compute_reference` tests PROPORTIONALITY and never LEVEL; a "
                 "line 43.6x too steep is still perfectly proportional."])


def gate_monotone(found: list[Inversion], boundaries: int,
                  spread: float | None) -> Gate:
    bad = [i for i in found if i.sigma is None or i.sigma >= MONOTONE_SIGMA]
    if boundaries <= 0:
        return Gate(VALIDITY, "V2 monotone ladder",
                    "time rises with every extra M-tile",
                    f"zero inversions beyond {MONOTONE_SIGMA:.0f} sigma",
                    None, "the ladder had fewer than two treads, so no boundary "
                          "was examined",
                    "the fitted slopes, which are what alpha is")
    obs = (f"{boundaries} tread boundaries examined, {len(found)} inversions, "
           f"{len(bad)} beyond {MONOTONE_SIGMA:.0f} sigma"
           + (f" (spread {spread:.3%})" if spread else " (spread unknown)"))
    return Gate(VALIDITY, "V2 monotone ladder",
                "time rises with every extra M-tile",
                f"zero inversions beyond {MONOTONE_SIGMA:.0f} sigma",
                not bad, obs,
                "the fitted slopes, which are what alpha is: OLS gives the last "
                "tread the most leverage, and on the published A100 ladder the "
                "inverted tread is also the only point on the compute branch",
                [i.line() for i in found])


def gate_convex(drops: list[SlopeDrop], slopes: list[float],
                spread: float | None) -> Gate:
    rule = (f"zero slope drops over {SLOPE_DROP_FLOOR:.0%} and "
            f"{SLOPE_DROP_SIGMA:.0f} sigma")
    invalidates = ("the two-line fit: max(A+Bn, D+Cn) is convex, so a slope "
                   "that rises then falls is not describable by ANY two-line "
                   "model and its fitted B/C is the slope of a bend")
    if len(slopes) < 2:
        return Gate(VALIDITY, "V3 max-affine shape",
                    "the per-tread slope never rises and then falls", rule,
                    None, "fewer than three treads, so no second difference "
                          "exists", invalidates)
    if spread is None:
        return Gate(VALIDITY, "V3 max-affine shape",
                    "the per-tread slope never rises and then falls", rule,
                    None, f"{len(drops)} drops over {SLOPE_DROP_FLOOR:.0%}, but "
                          "no timing spread to weigh them against",
                    invalidates, [d.line() for d in drops])
    return Gate(VALIDITY, "V3 max-affine shape",
                "the per-tread slope never rises and then falls", rule,
                not drops,
                f"{len(slopes)} per-tread slopes at spread {spread:.3%}, "
                f"{len(drops)} drops clearing both bars: "
                + " ".join(f"{s:.3f}" for s in slopes),
                invalidates, [d.line() for d in drops])


def gate_depth(memory_treads: int, ladder_treads: int,
               outcome: str = "", undecided: bool = False) -> Gate:
    """Did BLOCK_M=128 show `TARGET_TREADS` clean memory-bound treads.

    UNDECIDED IS A THIRD ANSWER AND IT IS NOT "NO". `fit_ladder` gained six
    named outcomes on 2026-09-02 precisely because the four that are not
    `identified` used to arrive here as the same blank, and two of them are
    states where the sweep LOOKED AND COULD NOT SAY rather than states where
    the tile is compute bound early:

      `undecided_parallel_branch` -- the memory branch runs within
      `PARALLEL_BRANCH_TOLERANCE` of the compute branch, which by the identity
      `B/C = ridge/ai_cap` means this tile's cap sits ON the ridge. That is the
      exact regime the whole BLOCK_M=128 question is about, and scoring it FAIL
      would publish "no depth" from a fit that declined to name one.
      `undecided_low_clock` -- enough treads were dropped for clock level that
      what remains is under the floor. The card is what that ladder measured.

    Both return UNKNOWN, which counts against the gate without asserting the
    opposite, and both NAME the outcome so the reader knows which next
    experiment settles it. Anything else keeps the plain comparison.
    """
    lines = ["A FAIL is the registered expectation and is a result: see the "
             "escape thresholds above."]
    if outcome:
        lines.append(f"ladder outcome: {outcome}")
    if undecided:
        return Gate(CLAIM, "C1 depth",
                    f"at least {TARGET_TREADS} clean memory-bound treads at "
                    f"BLOCK_M={SUBJECT_BLOCK_M}",
                    f">= {TARGET_TREADS} treads standing above the compute "
                    "branch", None,
                    f"UNDECIDED ({outcome}): the fit LOOKED and declined to "
                    f"name a memory branch, so the {memory_treads} treads it "
                    f"reports of {ladder_treads} are not a depth",
                    lines=lines + [
                        "This is NOT the same as a shallow ladder, and reading "
                        "it as one is how a fit that refused to answer becomes "
                        "a published 'no'. `undecided_parallel_branch` says the "
                        "cap sits ON the ridge, which the roofline arm can "
                        "settle and this one cannot; `undecided_low_clock` says "
                        "the card throttled and the run should be repeated on a "
                        "quiet one."])
    return Gate(CLAIM, "C1 depth",
                f"at least {TARGET_TREADS} clean memory-bound treads at "
                f"BLOCK_M={SUBJECT_BLOCK_M}",
                f">= {TARGET_TREADS} treads standing above the compute branch",
                memory_treads >= TARGET_TREADS,
                f"{memory_treads} of {ladder_treads} treads above the compute "
                f"branch (a verdict also needs {MIN_MEMORY_TREADS})",
                lines=lines)


def gate_margin(margin: Margin) -> Gate:
    if margin.sigma is None:
        return Gate(CLAIM, "C2 tolerance margin",
                    "the memory branch is not the compute branch again",
                    f"|B/C - 1| - {TOLERANCE:.2f} >= {MARGIN_SIGMA:.0f} sd",
                    None, margin.basis if math.isnan(margin.ratio) else margin.line(),
                    lines=[margin.basis])
    return Gate(CLAIM, "C2 tolerance margin",
                "the memory branch is not the compute branch again",
                f"|B/C - 1| - {TOLERANCE:.2f} >= {MARGIN_SIGMA:.0f} sd",
                margin.confident, margin.line(),
                lines=[f"noise model: {margin.basis}, {margin.draws} resamples, "
                       f"sd {margin.sd:.6f}",
                       "clearing the tolerance is not enough: the published A100 "
                       "fit clears by 0.000101, which is 0.07% of the tolerance "
                       "and 0.02 sd, and a 0.010% change in ONE tread reverses "
                       "it."])


def gate_non_vacuity(counts: dict[str, int],
                     optional: dict[str, int] | None = None) -> Gate:
    """A check that examined nothing also reports zero failures.

    Every gate above can pass by having no data. This one asserts the data
    existed, and it names the count so a reader can see WHICH work happened
    rather than trusting that some did.

    `counts` ARE THE REPORT'S OWN INPUTS and a zero in any of them is vacuity:
    no timings, no treads, no repeats, no setting that executed. `optional` are
    counts that belong to a gate that may legitimately DECLINE -- the bootstrap
    resample count is C2's, and C2 returns UNKNOWN when the ladder has no memory
    branch to resample. Scoring that zero here made "a gate declined" print as
    "this report examined nothing", which is the same conflation of a refusal
    with a null that `fit_ladder`'s UNDECIDED outcomes exist to prevent one
    level down. A zero in `optional` is REPORTED, named, and not scored; the
    declining gate already carries its own UNKNOWN into the exit code.
    """
    optional = optional or {}
    empty = sorted(k for k, v in counts.items() if v <= 0)
    declined = sorted(k for k, v in optional.items() if v <= 0)
    lines = []
    if empty:
        lines.append(f"nothing was counted for: {', '.join(empty)}")
    if declined:
        lines.append(
            f"not counted, and NOT scored here: {', '.join(declined)}. That "
            "count belongs to a gate that declined to answer, and its own "
            "UNKNOWN is what carries the refusal into the exit code.")
    return Gate(VALIDITY, "V0 non-vacuity", "this report examined real work",
                "every one of the report's own input counts is above zero",
                not empty,
                ", ".join(f"{k}={v}" for k, v in
                          sorted({**counts, **optional}.items())),
                "every gate in this report: a check with no input reports no "
                "failures",
                lines)


# --------------------------------------------------------------------------
# The registered predictions, printed with numbers before anything is measured.
# --------------------------------------------------------------------------

def predictions_text(b: int = 2) -> str:
    up = escape_up_alpha_rho(SUBJECT_BLOCK_M, b)
    down = escape_down_rho(SUBJECT_BLOCK_M, b, TARGET_TREADS)
    return f"""\
## Predictions, registered before anything is measured

THE LAW (arithmetic, not a prediction; check it, do not test it)
  B/C = alpha b rho / (2 BM) = ridge / ai_cap, and E, F, H cancel exactly.
  escape UP   needs alpha x rho >= {up:.1f} FLOP/byte  -> every tread memory bound
  escape DOWN needs rho        >= {down:.1f} FLOP/byte  -> {TARGET_TREADS} treads below the crossing
    equivalently alpha <= {escape_down_alpha(TARGET_TREADS):.3f}, a bound needing no card
  in between, |B/C - 1| <= {TOLERANCE:.2f}, the fit discards the memory branch entirely.

P1  A NEW BM=128 ladder lands at B/C in [0.94, 1.10].
    22 published ladders with a valid reference: 0.795 to 1.175, median 0.991,
    sd 0.078. FAIL means B/C at 128 is not stationary across arms and the
    feasibility argument below rests on a quantity that moves.
P2  C1 FAILS: fewer than {TARGET_TREADS} clean memory-bound treads.
    Escape up needs {up:.1f} and the corpus tops out at 150.4, on a ladder whose
    slope rises then falls. Escape down needs rho >= {down:.1f} and the highest
    achieved rho on any published ladder is 166.5, against calibrated ridges of
    145.8 (A100) and 162.8 (H200). FAIL here is the good outcome and would mean
    the cap CAN be measured at the tile vLLM actually runs.
P3  C2 FAILS: no BM=128 fit clears the tolerance by {MARGIN_SIGMA:.0f} sd.
P4  EXPERT SIZE does not enter B/C. mixtral's per-expert weight is 6.4x qwen2's
    (3 F H = 176.2 M elements against 27.5 M), so a "bigger weights to re-read"
    mechanism predicts B/C scaling like 6.4. The identity predicts NO size term
    at all: the model can still move B/C, but only by moving alpha or rho, which
    it does by tens of percent and not by factors. Gate: at matched card,
    GROUP_SIZE_M and BLOCK_N, |B/C(mixtral) / B/C(qwen2) - 1| < 0.20 in the
    median. FAIL at 6.4 means the identity is wrong and the feasibility argument
    goes with it; FAIL at 0.3 means alpha differs by more between models than
    this bound allows and the bound was too tight.
P5  --r-max does not move the memory-tread COUNT in the prefix regime: n* has no
    r_max in it. Sweeping deeper adds treads to measure, not treads that qualify.
    UNKNOWN in --audit unless the corpus holds two depths at one setting.
P6  V2 FAILS on exactly one published ladder -- A100 qwen2-57b-a14b at
    GROUP_SIZE_M=64, whose tread 8 is 1.237% below tread 7 at a 0.482% spread
    (2.6 sigma) -- and PASSES on the other 21.
P7  Of the ladders that HAVE a compute reference, the level bar fails on exactly
    the two BLOCK_N=256 arms -- 1.4% (A100) and 12.3% (H200) of their cards'
    calibrated ceilings -- and passes on the other 22 at 38.2% to 63.7%. Two
    further ladders have no reference at all to level-check and are counted
    separately, because "the reference is corrupt" and "there is no reference"
    are different states and only the first is what this bar is for.
P8  Of the two published BM=128 fits the study quotes, ZERO survive V1+V2+V3+C2.
    FAIL means one of them is admissible and the 128 row of the alpha surface
    stands.
P9  No alpha this study has measured, at ANY block size, is at or below
    {escape_down_alpha(TARGET_TREADS):.3f}. The lowest is 0.596 (H200
    deepseek-v2-lite, BLOCK_M=32, activation-corrected). FAIL means the
    escape-down route is open on hardware the study already owns and the pod run
    should be aimed at whichever arm produced it."""


# --------------------------------------------------------------------------
# The published corpus: the evidence for every number above.
# --------------------------------------------------------------------------

ARM_NAME = re.compile(r"^(?P<model>.+?)-(?P<dtype>bf16|fp16)-r(?P<r_max>\d+)"
                      r"-g(?P<group_m>\d+)-n(?P<block_n>\d+)-")


@dataclass
class LadderRecord:
    """One published BLOCK_M=128 ladder, with everything a gate needs."""

    arm: str
    card: str
    model: str
    dtype: str
    group_m: int
    block_n: int
    points: list[tuple[int, float]]
    spread: float
    overhead_ms: float
    ref_block_m: int | None
    ref_slope: float | None
    memory_points: int
    published_alpha: float | None
    ceiling_tflops: float
    path: str
    #: The lowest activation-corrected alpha this ARM measured at ANY block
    #: size, and the block size it came from. This is the evidence against the
    #: escape-down route, and it lives at BLOCK_M=32 and 64 -- the block sizes
    #: where alpha is identifiable -- not at 128, where it is not.
    min_alpha: float | None = None
    min_alpha_block_m: int | None = None
    #: This ladder's own activation-corrected alpha at BLOCK_M=128, and the
    #: ridge its report stamps. Both are here for `premise_caps` and nothing
    #: else: the arm's founding premise was a cap built from the first against
    #: the second, and a premise that cannot be recomputed from the files is a
    #: sentence rather than a result.
    alpha_corrected_128: float | None = None
    ridge: float | None = None
    #: `B / (L - D)` at BLOCK_M=128: alpha with the fused layer's fixed cost
    #: taken out of the level. `premise_caps` reads `delta` off it -- the fixed
    #: cost in weight-read units is recoverable from the PAIR as
    #: `(D/L)(1 + phi)/(1 - D/L)` with `D/L = 1 - alpha/alpha_upper` -- and
    #: prints what it implies rather than using it; see that function.
    alpha_upper_128: float | None = None

    @property
    def c_ref(self) -> float | None:
        """`C` at 128, scaled from the reference by `C ~ BLOCK_M`."""
        if self.ref_slope is None or not self.ref_block_m:
            return None
        return self.ref_slope * SUBJECT_BLOCK_M / self.ref_block_m


def _card_of(arm: str) -> str | None:
    low = arm.lower()
    for key in CALIBRATION_SLUGS:
        if key in low:
            return key
    return None


def _ceiling(card: str, dtype: str, directory: Path | None = None) -> float:
    hw = load_hardware(CALIBRATION_SLUGS[card], directory=directory)
    return hw.peak(dtype) / 1e12


def load_corpus(published: Path, dtype: str = "bf16",
                hardware_dir: Path | None = None
                ) -> tuple[list[LadderRecord], list[str]]:
    """Every published BLOCK_M=128 ladder, and a NAMED reason for each skip.

    Skips are returned rather than swallowed. A corpus audit that quietly drops
    the arms it cannot parse reports a cleaner picture than the data supports,
    and the two arms most worth looking at here are the two that are broken.
    """
    records: list[LadderRecord] = []
    skipped: list[str] = []
    for path in sorted(published.glob("*/*.report.json")):
        arm = path.parent.name
        card = _card_of(arm)
        if card is None:
            skipped.append(f"{arm}/{path.name}: no card in the arm name")
            continue
        m = ARM_NAME.match(path.name)
        if not m:
            skipped.append(f"{arm}/{path.name}: filename does not carry the arm")
            continue
        if m["dtype"] != dtype:
            skipped.append(f"{arm}/{path.name}: dtype {m['dtype']}, not {dtype}")
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            skipped.append(f"{arm}/{path.name}: unreadable ({exc})")
            continue
        fit = (payload.get("ladder") or {}).get(str(SUBJECT_BLOCK_M))
        if not fit or len(fit.get("points") or []) < 2:
            skipped.append(f"{arm}/{path.name}: no BLOCK_M={SUBJECT_BLOCK_M} ladder")
            continue
        if m["model"] not in MODEL_CONFIGS:
            skipped.append(f"{arm}/{path.name}: unknown model {m['model']!r}")
            continue
        ref = payload.get("compute_reference") or {}
        alphas = [(v.get("alpha_corrected") or v.get("alpha"), int(k))
                  for k, v in (payload.get("ladder") or {}).items()
                  if (v.get("alpha_corrected") or v.get("alpha"))]
        best = min(alphas) if alphas else (None, None)
        records.append(LadderRecord(
            arm=arm, card=card, model=m["model"], dtype=m["dtype"],
            group_m=int(m["group_m"]), block_n=int(m["block_n"]),
            points=[(int(n), float(ms)) for n, ms in fit["points"]],
            spread=float(payload.get("timing_spread_median") or 0.0),
            overhead_ms=float(payload.get("overhead_ms") or 0.0),
            ref_block_m=ref.get("block_m"), ref_slope=ref.get("slope_per_tile"),
            memory_points=int(fit.get("memory_points") or 0),
            published_alpha=fit.get("alpha"),
            ceiling_tflops=_ceiling(card, m["dtype"], hardware_dir),
            path=str(path), min_alpha=best[0], min_alpha_block_m=best[1],
            alpha_corrected_128=fit.get("alpha_corrected"),
            ridge=payload.get("ridge"),
            alpha_upper_128=fit.get("alpha_upper")))
    return records, skipped


# --------------------------------------------------------------------------
# The founding premise, recomputed instead of quoted. Retracted 2026-09-02.
# --------------------------------------------------------------------------

#: `delta`, the fused layer's fixed cost in units of one full weight read, as
#: `premise_caps` charges it. ZERO IS A FLOOR AND NOT A MEASUREMENT: two of the
#: two published BM=128 ladders pin a positive delta through their own
#: `alpha_upper`, and one of them pins a delta at which the corrected cap does
#: not exist. Zero is used because delta enters (EXA) only through the LEVEL, so
#: a larger delta recovers a LARGER alpha_b and a SMALLER cap: the floor is the
#: choice that gives the retracted straddle every benefit before refusing it.
#: `PremiseCap.delta_implied` carries what each ladder actually pins, and
#: `render_premise` prints it beside the cap.
PREMISE_DELTA_FLOOR = 0.0


def fused_phi(cfg, dtype: str, block_m: int = SUBJECT_BLOCK_M) -> float:
    """`Act1 / W`: one M-tile's activation and output traffic in units of one
    full weight read, on the FUSED layer the ladder actually timed.

    `W` is the layer's whole weight set, `E` experts of `3 F H` elements at the
    weight dtype. `Act1` is `E * BM` rows of `x_perm`, `h_up`, `h_act` and
    `y_perm`, which is `2 H + 3 F` elements per row at the ACTIVATION dtype --
    2 bytes in every arm this study has published, including the fp8-weight
    ones, because charging activations at the weight dtype reports traffic that
    was never moved.

    WHY NOT `ai_model.phi`. That evaluates phi on ONE GEMM, the up-projection
    `N = 2F`, `K = H`, and needs `alpha_a`, the miss fraction on the activation
    re-read, which has no measurement anywhere in this repository. The alpha it
    would be correcting is not a single-GEMM quantity. The first corrected
    version of this section paired the single-GEMM phi with a fused-layer alpha
    and bracketed the mismatch over alpha_a, which is a bracket about a
    parameter the fused byte model does not contain; the two phis differ by
    12-14% at alpha_a = 0 and by a factor of 17 to 30 at alpha_a = 1.
    `scripts/memory_branch_anchor.py` uses `Act1 / W` for exactly this reason
    and says so in `Bracket.phi`.

    IT IS A FLOOR, in the direction that makes the retraction harder to keep
    rather than easier: `Act1` charges each of the four activation tensors once
    per M-tile and counts no N-tile re-read, so a real re-read only adds to it,
    and a larger phi is a smaller cap.

    TRANSCRIBED, NOT IMPORTED, from `block_m_crossing_sweep.activation_bytes_per_row`
    and `memory_branch_anchor.activation_bytes_per_row`, both of which say the
    same thing: a premise that changes because a sibling script refactored a
    helper is not a premise. `tests/test_bm128_depth.py` cross-checks all three
    whenever the sweep is importable, so a divergence is caught rather than
    assumed away.
    """
    act1 = cfg.num_experts * block_m * (2 * cfg.hidden_size
                                        + 3 * cfg.intermediate_size) * 2
    return act1 / cfg.weight_bytes(dtype)


def implied_delta(alpha: float, alpha_upper: float | None, phi: float
                  ) -> float | None:
    """The fixed cost this ladder's OWN two alphas pin, in weight-read units.

    The report publishes `alpha = B / L` and `alpha_upper = B / (L - D)`, so

        D / L = 1 - alpha / alpha_upper

    with no bandwidth and no byte count in it. In (EXA) units the level is
    `1 + phi + delta`, and `delta / (1 + phi + delta) = D / L`, so

        delta = (D / L) (1 + phi) / (1 - D / L)

    Returns None when the pair pins nothing: no `alpha_upper` in the block, a
    non-positive one, or a ratio outside `[0, 1)`. None is a different answer
    from zero and is printed as one, because `D = 0` is a measurement (the H200
    ladder's `overhead_ms` really is 0.0) and a missing `alpha_upper` is not.
    """
    if alpha_upper is None or alpha_upper <= 0.0 or alpha <= 0.0:
        return None
    d_over_l = 1.0 - alpha / alpha_upper
    if not 0.0 <= d_over_l < 1.0:
        return None
    return d_over_l * (1.0 + phi) / (1.0 - d_over_l)


@dataclass(frozen=True)
class PremiseCap:
    """One published BLOCK_M=128 fit's cap, retracted form beside corrected.

    `lin_cap` is `2 BM / (b alpha_corrected)`: the expression the two numbers in
    this module's docstring were, at the alpha they were taken at. It is kept
    for exactly that, so the size of the correction is printed rather than
    described.

    `exa_cap` is `ai_model.cap_from_fitted` at `alpha_fitted`, the ladder's raw
    `B / (A + B)`, with `phi = Act1 / W` on the fused layer and
    `delta = PREMISE_DELTA_FLOOR`. THE INPUT IS THE RAW ALPHA AND THAT IS THE
    WHOLE OF THIS DATACLASS'S HISTORY. The first version divided `lin_cap` by
    `1 + phi + delta`; the second passed `alpha_corrected` into
    `ai_model.exact_cap`'s `alpha_b` slot, having asserted in this docstring
    that `alpha_corrected` is an alpha_b. It is `(B - Act1) / L`, a slope over
    the fitted LEVEL, which is (EXA)-shaped and not a miss fraction -- the same
    error the commit that wrote it existed to fix. `cap_from_fitted` takes the
    (EXA) quantity by name, inverts it through `ai_model.alpha_b_from_fitted`,
    and refuses whatever the three-term model cannot hold.

    `delta_implied` is what this ladder's own `alpha_upper` pins, and
    `exa_cap_at_delta_implied` / `delta_implied_refusal` are what the cap does
    there: one of the two published ladders has no cap at all at its own delta,
    because the recovered `alpha_b` lands above 1. That is reported, not
    hidden, and it is the reason the floor is used for the headline rather than
    the pinned value: the floor is the number the straddle would need, and it
    still does not reach the ridge.
    """

    arm: str
    model: str
    group_m: int
    alpha_fitted: float
    alpha_corrected: float
    ridge: float
    phi: float
    delta: float
    alpha_b: float
    lin_cap: float
    exa_cap: float
    delta_implied: float | None
    exa_cap_at_delta_implied: float | None
    delta_implied_refusal: str

    @property
    def factor(self) -> float:
        """`lin_cap / exa_cap`, which is `(alpha / alpha_corrected)(1 + phi + delta)`.

        NOT `ai_model.lin_overstatement` and not `(alpha_b + phi) / alpha_b`:
        the retracted cap and the corrected one were taken at DIFFERENT alphas,
        so the factor between them carries the ratio of those two as well as the
        level. Computed as the ratio of the two printed numbers so that it
        cannot disagree with them.
        """
        return self.lin_cap / self.exa_cap

    @property
    def lin_straddles(self) -> bool:
        """Did the RETRACTED cap sit above this card's ridge. The premise."""
        return self.lin_cap >= self.ridge

    @property
    def straddles(self) -> bool:
        """Does the CORRECTED cap reach the ridge.

        Read at the delta floor and at `Act1 / W`, both of which are floors, so
        `exa_cap` is the largest cap the corrected model allows for this ladder.
        False here is the retraction: the straddle does not exist even where it
        is given every benefit.
        """
        return self.exa_cap >= self.ridge


class PremiseNotRecomputable(Exception):
    """No published BLOCK_M=128 ladder can produce the founding premise.

    DELIBERATELY NOT A `RefusedBeforeMeasuring`, and the difference is the
    reason this class exists. `premise_caps` first refused through `_refuse`,
    which carries `exit_codes.REFUSED` and takes the process with it. Only 2 of
    the 22 valid published BM=128 ladders carry both an alpha and a stamped
    ridge, and those two are the fits this module's own docstring argues should
    be WITHDRAWN, so blanking them turned a fully scoreable `--audit` page into
    a total refusal: zero `RESULT:` lines, V0 and P4 through P9 never examined,
    a corpus voided by a paragraph none of those gates read. The premise is
    provenance for why the arm was proposed; the gates measure `B/C` and tread
    counts off the ladders themselves and take no cap, no ridge and no alpha
    from here.

    WHAT IS KEPT is the thing the refusal was for: "no BM=128 fit carries an
    alpha" and "the premise holds" still do not print the same way. This is
    raised rather than an empty list returned, `render_premise` catches it and
    prints a NOT RECOMPUTABLE section naming the reason, and the gates below it
    still score the page.
    """


def premise_caps(published: Path, hardware_dir: Path | None = None
                 ) -> list[PremiseCap]:
    """Re-derive the arm's founding premise from the published reports.

    WHY THIS IS CODE AND NOT A PARAGRAPH. Until 2026-09-02 this module opened by
    asserting that BLOCK_M=128 "is the one tile height where the study's
    arithmetic-intensity cap straddles the hardware ridge -- cap 150.4 against a
    calibrated ridge of 145.8 on the A100, 158.6 against 162.8 on the H200".
    Those two caps were `2 BM / (b alpha)` at the two published BM=128 fits, a
    form that leaves `phi` out of its denominator. Corrected, both fall below
    their own card's ridge and the straddle is gone. A premise that lived only
    in prose survived the retraction of the arithmetic under it at the same
    HEAD, in the same checkout, with no complaint from anything; one that is
    recomputed from the files on demand cannot.

    THE INPUT IS THE LADDER'S RAW `alpha`, AND THE TWO EARLIER VERSIONS OF THIS
    FUNCTION EACH PUT SOMETHING ELSE THERE. The first divided the retracted cap
    by `ai_model.lin_overstatement`. The second passed `alpha_corrected` into
    `ai_model.exact_cap`'s `alpha_b` slot. `alpha_corrected` is
    `(slope_memory - activation_slope_ms) / load_ms` and `load_ms = A + B` is
    the fitted LEVEL, so it is a slope over a level -- (EXA)-shaped, not a miss
    fraction -- and putting it in an `alpha_b` slot is the error the module it
    was calling exists to name.

    AND `alpha_corrected` CANNOT BE INVERTED BACK, which is why the raw alpha is
    used rather than a corrected form of the corrected one. Its numerator takes
    `Act1` out at the CALIBRATION's streaming bandwidth; (EXA)'s `phi` is `Act1`
    against the rate the fit itself achieved. On the A100 ladder the subtraction
    made is 0.03334 of the level where `phi / (1 + phi)` is 0.06459, so undoing
    it needs a bandwidth ratio the report does not carry -- and undoing it lands
    exactly on `alpha`, which the block publishes. So `alpha` goes in,
    `alpha_corrected` stays on the page as the number the retracted cap came
    from, and the arithmetic is `ai_model.cap_from_fitted`: one module owns the
    inverse and enforces the `[0, 1]` wall on the recovered miss fraction.

    RAISES `PremiseNotRecomputable` rather than returning an empty list,
    because "no BM=128 fit carries an alpha" and "the premise holds" must not
    print the same way. It is not a refusal of the RUN: see that class.
    """
    records, _ = load_corpus(published, hardware_dir=hardware_dir)
    out: list[PremiseCap] = []
    unusable: list[str] = []
    for rec in records:
        if (rec.published_alpha is None or rec.alpha_corrected_128 is None
                or not rec.ridge):
            continue
        alpha = float(rec.published_alpha)
        corrected = float(rec.alpha_corrected_128)
        if alpha <= 0 or corrected <= 0:
            unusable.append(
                f"{rec.arm}: alpha {alpha:.5f} / alpha-corrected "
                f"{corrected:.5f} is not positive, and neither "
                "2 BM / (b alpha) nor its correction has a value there")
            continue
        cfg = MODEL_CONFIGS[rec.model]
        b = dtype_bytes(rec.dtype)
        phi = fused_phi(cfg, rec.dtype)
        try:
            alpha_b = ai_model.alpha_b_from_fitted(
                alpha, phi=phi, delta=PREMISE_DELTA_FLOOR)
            exa = ai_model.cap_from_fitted(
                alpha, block_m=SUBJECT_BLOCK_M, b=b, phi=phi,
                delta=PREMISE_DELTA_FLOOR)
        except ai_model.AIModelRefused as exc:
            # A recovered alpha_b outside [0, 1] is not a miss fraction, and
            # `ai_model` says so by name. Named and dropped rather than
            # clamped: a cap computed from a number the model cannot hold is
            # not a cap.
            unusable.append(f"{rec.arm}: {exc}")
            continue
        implied = implied_delta(alpha, rec.alpha_upper_128, phi)
        at_implied: float | None = None
        refusal = ""
        if implied is not None:
            try:
                at_implied = ai_model.cap_from_fitted(
                    alpha, block_m=SUBJECT_BLOCK_M, b=b, phi=phi, delta=implied)
            except ai_model.AIModelRefused as exc:
                # NOT a reason to drop the ladder. The headline cap is read at
                # the delta FLOOR, which exists and is an upper bound; that the
                # ladder's own delta admits no cap at all is a stronger form of
                # the same retraction and is printed as one.
                refusal = str(exc)
        out.append(PremiseCap(
            arm=rec.arm, model=rec.model, group_m=rec.group_m,
            alpha_fitted=alpha, alpha_corrected=corrected,
            ridge=float(rec.ridge), phi=phi, delta=PREMISE_DELTA_FLOOR,
            alpha_b=alpha_b,
            lin_cap=2.0 * SUBJECT_BLOCK_M / (b * corrected), exa_cap=exa,
            delta_implied=implied, exa_cap_at_delta_implied=at_implied,
            delta_implied_refusal=refusal))
    if not out:
        raise PremiseNotRecomputable(
            f"no published BLOCK_M={SUBJECT_BLOCK_M} ladder under {published} "
            "carries both a fitted alpha this model can hold and a stamped "
            "ridge, so this arm's founding premise cannot be recomputed. The "
            "premise is unchecked, not upheld"
            + ("; ".join([""] + unusable) if unusable else "."))
    return out


def render_premise(published: Path, hardware_dir: Path | None = None
                   ) -> list[str]:
    """The founding premise, printed as arithmetic every time `--audit` runs.

    On the page rather than in the docstring because the docstring version
    stayed wrong at a HEAD where the file it cites already said so, and nothing
    noticed because nothing recomputed it. The retracted cap is printed BESIDE
    the corrected one, and both `phi` and `delta` are printed beside those: a
    cap whose two model parameters are not on the page is a cap whose next
    reader cannot tell which alpha went into it, which is how this section was
    wrong twice.
    """
    try:
        caps = premise_caps(published, hardware_dir=hardware_dir)
    except PremiseNotRecomputable as exc:
        return ["", "## The founding premise, NOT RECOMPUTABLE", "",
                # Wrapped because this sentence names the corpus path and every
                # ladder it could not use, and an audit page is read as text.
                textwrap.fill(str(exc), width=88, initial_indent="  ",
                              subsequent_indent="  "),
                "  This voids the PREMISE and not the page. Every gate below "
                "is scored off the",
                "  ladders themselves and takes no cap, no ridge and no alpha "
                "from this section,",
                "  so the audit continues; what is missing is the arithmetic "
                "for why the arm was",
                "  proposed, which the measured B/C near 1 replaced anyway."]
    out = ["", "## The founding premise, recomputed", "",
           "The retracted cap is `2 BM / (b alpha-corrected)`; the corrected "
           "one is",
           "`ai_model.cap_from_fitted` at the ladder's RAW alpha = B/(A+B), "
           "which is the estimator",
           "(EXA) describes, with phi = Act1/W on the fused layer and delta at "
           "its floor. Both of",
           "those are floors, so the corrected cap is an UPPER bound on what "
           "this ladder allows.", ""]
    for c in sorted(caps, key=lambda c: c.arm):
        out.append(f"  {c.arm[:38]:38s} {c.model[:16]:16s} G={c.group_m:2d} "
                   f"alpha {c.alpha_fitted:.5f}  ridge {c.ridge:6.1f}")
        out.append(f"  {'':38s} phi {c.phi:.5f} (Act1/W, fused)  "
                   f"delta {c.delta:.5f} (floor)  alpha_b {c.alpha_b:.5f}")
        out.append(f"  {'':38s} retracted {c.lin_cap:6.1f} "
                   f"({c.lin_cap / c.ridge:.3f} of ridge, "
                   f"{'ABOVE' if c.lin_straddles else 'below'}) "
                   f"at alpha-corrected {c.alpha_corrected:.5f}")
        out.append(f"  {'':38s} corrected {c.exa_cap:6.1f} "
                   f"({c.exa_cap / c.ridge:.3f} of ridge), "
                   f"factor {c.factor:.3f}, "
                   f"{'STILL STRADDLES' if c.straddles else 'no straddle'}")
        if c.delta_implied is None:
            out.append(f"  {'':38s} this ladder pins no delta: it publishes no "
                       "alpha-upper to read D/L from")
        elif c.delta_implied_refusal:
            out.append(f"  {'':38s} at its OWN delta {c.delta_implied:.5f} "
                       "there is NO cap:")
            out.append(textwrap.fill(c.delta_implied_refusal, width=88,
                                     initial_indent=" " * 43,
                                     subsequent_indent=" " * 43))
        else:
            out.append(f"  {'':38s} at its OWN delta {c.delta_implied:.5f} the "
                       f"cap is {c.exa_cap_at_delta_implied:6.1f} "
                       f"({c.exa_cap_at_delta_implied / c.ridge:.3f} of ridge)")
    if any(c.straddles for c in caps):
        out += ["", "  At least one corrected cap still reaches its card's "
                    "ridge, so the premise stands as written."]
    else:
        out += ["", "  NO corrected cap reaches its card's ridge, and both "
                    "inputs to the correction",
                "  are floors, so that is an upper bound refusing the "
                "straddle. What justifies the",
                "  arm is the MEASURED B/C near 1, which carries no cap and no "
                "ridge."]
    return out


@dataclass
class AuditRow:
    """One corpus ladder, after every diagnostic has been run on it."""

    record: LadderRecord
    level: RefLevel | None
    inversions: list[Inversion]
    drops: list[SlopeDrop]
    ratio: float | None
    margin: Margin
    admissible: bool
    reasons: list[str]


def audit_record(rec: LadderRecord, seed: int = 0,
                 draws: int = BOOTSTRAP_DRAWS) -> AuditRow:
    """Run V1, V2, V3 and C2 over one published ladder.

    The branch assignment used for `B` is the LEADING RUN the fit itself
    reported (`memory_points`), except that a ladder the fit gave 0 or 1 treads
    is still measured over its whole length -- not to claim it has a memory
    branch, but so its `B/C` can be read and compared against the escape
    thresholds. Which of the two happened is in `reasons`.
    """
    cfg = MODEL_CONFIGS[rec.model]
    level = None
    if rec.ref_slope and rec.ref_block_m:
        level = reference_level(cfg, rec.ref_block_m, rec.ref_slope,
                                rec.ceiling_tflops,
                                f"{rec.card.upper()} calibration")
    inv = inversions(rec.points, rec.spread or None)
    drops = slope_drops(rec.points, rec.spread or None)
    k = rec.memory_points if rec.memory_points >= 2 else len(rec.points)
    ratio = _refit_ratio(rec.points, k, rec.points[k:], rec.c_ref,
                         rec.overhead_ms)
    margin = margin_of(rec.points, k, c_ref=rec.c_ref,
                       overhead=rec.overhead_ms, spread=rec.spread or None,
                       draws=draws, seed=seed)

    reasons = []
    if level is not None and not level.passes:
        reasons.append(f"V1 reference at {level.fraction:.1%} of the ceiling")
    if level is None:
        reasons.append("V1 no compute reference")
    bad_inv = [i for i in inv if i.sigma is None or i.sigma >= MONOTONE_SIGMA]
    if bad_inv:
        reasons.append(f"V2 {len(bad_inv)} inversion(s) beyond "
                       f"{MONOTONE_SIGMA:.0f} sigma")
    if drops:
        reasons.append(f"V3 {len(drops)} slope drop(s) over "
                       f"{SLOPE_DROP_FLOOR:.0%} and {SLOPE_DROP_SIGMA:.0f} sigma")
    if rec.memory_points < MIN_MEMORY_TREADS:
        reasons.append(f"below MIN_MEMORY_TREADS: {rec.memory_points} treads "
                       f"stand above the compute branch, a verdict needs "
                       f"{MIN_MEMORY_TREADS}")
    if not margin.confident:
        reasons.append("C2 margin not " + (f"{MARGIN_SIGMA:.0f} sd clear"
                                           if margin.clears else "cleared"))
    return AuditRow(rec, level, inv, drops, ratio, margin, not reasons, reasons)


#: What a "bigger weights cost more to re-read" mechanism would predict for the
#: mixtral/qwen2 B/C ratio, and what the identity predicts instead. Stated as
#: constants so the gate names the alternative it rules out.
EXPERT_SIZE_RATIO_BOUND = 0.20


def _weight_elements(model: str) -> int:
    cfg = MODEL_CONFIGS[model]
    return 3 * cfg.intermediate_size * cfg.hidden_size


def _gate_escape_down_alpha(rows: list[AuditRow]) -> Gate:
    """P9: is any measured alpha low enough for the escape-down route to exist.

    Scored over EVERY block size in every arm, not only BLOCK_M=128, because
    alpha is identifiable at 32 and 64 and is not at 128 -- and the bound is a
    property of the model, not of the tile height (see `escape_down_alpha`). A
    PASS here means the route is shut by numbers this study already published,
    with no ridge, no bandwidth and no card in the argument.
    """
    bound = escape_down_alpha(TARGET_TREADS)
    seen = [(r.record.min_alpha, r.record.min_alpha_block_m, r.record)
            for r in rows if r.record.min_alpha is not None]
    if not seen:
        return Gate(CLAIM, "P9 escape-down is shut by alpha alone",
                    f"no measured alpha is at or below {bound:.3f}",
                    f"min measured alpha > {bound:.3f}", None,
                    "no arm reported an identifiable alpha at any block size")
    lo = min(seen)
    return Gate(CLAIM, "P9 escape-down is shut by alpha alone",
                f"no measured alpha is at or below {bound:.3f}",
                f"min measured alpha > {bound:.3f}", lo[0] > bound,
                f"lowest of {len(seen)} arms: alpha {lo[0]:.3f} at BLOCK_M="
                f"{lo[1]} ({lo[2].card.upper()} {lo[2].model} g{lo[2].group_m} "
                f"n{lo[2].block_n}), against the {bound:.3f} bound",
                lines=[f"{TARGET_TREADS} clean treads below the crossing need "
                       f"alpha <= {bound:.3f} even charging zero overhead; the "
                       "bound falls further once overhead is charged."])


def _gate_expert_size(valid: list[AuditRow]) -> Gate:
    """P4: does per-expert weight size move B/C at all.

    Paired inside an ARM directory, so the two models were measured on the same
    card in the same session at the same swizzle and BLOCK_N. An unpaired
    comparison would carry the card and the session, and the card alone is worth
    8.5%.
    """
    pairs = []
    by_key: dict[tuple, dict[str, float]] = {}
    for r in valid:
        if r.ratio is None:
            continue
        key = (r.record.arm, r.record.group_m, r.record.block_n)
        by_key.setdefault(key, {})[r.record.model] = r.ratio
    for key, models in sorted(by_key.items()):
        if "mixtral-8x7b" in models and "qwen2-57b-a14b" in models:
            pairs.append((key, models["mixtral-8x7b"] / models["qwen2-57b-a14b"]))
    if not pairs:
        return Gate(CLAIM, "P4 expert size does not enter B/C",
                    "a 6.4x bigger expert does not make B/C 6.4x bigger",
                    f"median |ratio - 1| < {EXPERT_SIZE_RATIO_BOUND:.2f}", None,
                    "no arm measured both models at one setting, so nothing is "
                    "paired and the comparison would carry the card")
    size = _weight_elements("mixtral-8x7b") / _weight_elements("qwen2-57b-a14b")
    worst = max(abs(v - 1.0) for _, v in pairs)
    med = statistics.median(abs(v - 1.0) for _, v in pairs)
    return Gate(CLAIM, "P4 expert size does not enter B/C",
                "a 6.4x bigger expert does not make B/C 6.4x bigger",
                f"median |B/C(mixtral)/B/C(qwen2) - 1| < "
                f"{EXPERT_SIZE_RATIO_BOUND:.2f}",
                med < EXPERT_SIZE_RATIO_BOUND,
                f"{len(pairs)} matched pairs, median |ratio-1| {med:.3f}, worst "
                f"{worst:.3f}; a per-expert-size mechanism predicts {size:.1f}",
                lines=[f"{key[0]} g{key[1]} n{key[2]}: {v:.3f}x"
                       for key, v in pairs])


def _gate_depth_invariance(valid: list[AuditRow]) -> Gate:
    """P5: does sweeping deeper add treads that QUALIFY as memory bound.

    `n*` has no `r_max` in it, so depth buys treads to measure and not treads
    that pass. Checkable only where the corpus holds two ladder lengths at one
    setting; UNKNOWN otherwise, because "we never varied it" is not evidence.
    """
    by_key: dict[tuple, set[int]] = {}
    for r in valid:
        key = (r.record.card, r.record.model, r.record.group_m,
               r.record.block_n)
        by_key.setdefault(key, set()).add(len(r.record.points))
    varied = {k: v for k, v in by_key.items() if len(v) > 1}
    if not varied:
        return Gate(CLAIM, "P5 depth does not buy qualifying treads",
                    "the memory-tread count does not rise with ladder length",
                    "no rise in memory_points across ladder lengths", None,
                    f"every setting in the corpus was swept to one depth "
                    f"({sorted({n for v in by_key.values() for n in v})} treads); "
                    "P5 needs two depths at one setting and the corpus has none")
    return Gate(CLAIM, "P5 depth does not buy qualifying treads",
                "the memory-tread count does not rise with ladder length",
                "no rise in memory_points across ladder lengths", None,
                f"{len(varied)} setting(s) hold two depths; this gate is not yet "
                "implemented for them and must not print PASS on that basis")


def audit_report(rows: list[AuditRow], skipped: list[str], b: int = 2
                 ) -> tuple[list[str], list[Gate], dict]:
    """The corpus table, the scored predictions and the gates over all of it."""
    out: list[str] = []
    up = escape_up_alpha_rho(SUBJECT_BLOCK_M, b)
    down = escape_down_rho(SUBJECT_BLOCK_M, b, TARGET_TREADS)

    valid = [r for r in rows if r.level is not None and r.level.passes]
    ratios = [r.ratio for r in valid if r.ratio is not None]

    out += ["", "## Every published BLOCK_M=128 ladder", "",
            f"{'card':5s} {'model':16s} {'G':>3s} {'BN':>4s} {'ref%':>6s} "
            f"{'n':>2s} {'mem':>3s} {'B/C':>7s} {'a*rho':>7s} {'margin':>9s} "
            f"{'inv':>3s} {'drop':>4s}  verdict"]
    for r in sorted(rows, key=lambda r: -(r.ratio or -1)):
        rec = r.record
        frac = f"{r.level.fraction:6.1%}" if r.level else "   n/a"
        ratio = f"{r.ratio:7.3f}" if r.ratio is not None else "    n/a"
        arho = f"{r.ratio * 2 * SUBJECT_BLOCK_M / b:7.1f}" if r.ratio else "    n/a"
        marg = (f"{r.margin.margin:+9.6f}" if not math.isnan(r.margin.margin)
                else "      n/a")
        verdict = "ADMISSIBLE" if r.admissible else "; ".join(r.reasons)
        out.append(f"{rec.card.upper():5s} {rec.model[:16]:16s} {rec.group_m:3d} "
                   f"{rec.block_n:4d} {frac} {len(rec.points):2d} "
                   f"{rec.memory_points:3d} {ratio} {arho} {marg} "
                   f"{len(r.inversions):3d} {len(r.drops):4d}  {verdict}")
    for line in skipped:
        out.append(f"      SKIPPED  {line}")
    out += ["",
            "  B/C on a row with fewer than 2 memory treads is the OLS slope of "
            "the WHOLE ladder over the",
            "  independent compute reference. That is not a claim that a memory "
            "branch exists -- it is the",
            "  marginal cost per M-tile asked whether it looks memory-like or "
            "compute-like, which is the only",
            "  form the question has when the ladder is one line. Rows with a "
            "memory branch use that branch,",
            "  and its own compute treads for C, exactly as `fit_ladder` does."]

    out += ["", "## The feasibility arithmetic", ""]
    if ratios:
        arhos = [x * 2 * SUBJECT_BLOCK_M / b for x in ratios]
        out += [f"valid-reference ladders: {len(ratios)}",
                f"  B/C        min {min(ratios):.4f}  median "
                f"{statistics.median(ratios):.4f}  max {max(ratios):.4f}  "
                f"sd {statistics.pstdev(ratios):.4f}",
                f"  alpha x rho  min {min(arhos):.1f}  median "
                f"{statistics.median(arhos):.1f}  max {max(arhos):.1f}",
                f"  escape UP needs alpha x rho >= {up:.1f}: reached by "
                f"{sum(1 for x in arhos if x >= up)} of {len(arhos)}",
                f"  escape DOWN needs B/C <= {1 - TOLERANCE:.2f}: reached by "
                f"{sum(1 for x in ratios if x <= 1 - TOLERANCE)} of {len(ratios)}",
                f"  the median must move {up / statistics.median(arhos) - 1:+.1%} "
                "to reach the escape-up threshold"]
        out += ["", "  what each lever is worth, as the spread of the median B/C:"]
        for key, get in (("card", lambda r: r.record.card.upper()),
                         ("model", lambda r: r.record.model),
                         ("GROUP_SIZE_M", lambda r: r.record.group_m),
                         ("BLOCK_SIZE_N", lambda r: r.record.block_n)):
            groups: dict = {}
            for r in valid:
                if r.ratio is not None:
                    groups.setdefault(get(r), []).append(r.ratio)
            meds = {k: statistics.median(v) for k, v in groups.items()}
            if len(meds) < 2:
                out.append(f"    {key:14s} only one level in the corpus")
                continue
            lo, hi = min(meds.values()), max(meds.values())
            body = "  ".join(f"{k}={v:.3f}" for k, v in sorted(meds.items(),
                                                              key=lambda kv: str(kv[0])))
            out.append(f"    {key:14s} {body}   spread {hi / lo - 1:+.1%}")
        out += ["    the `model` row is NOT evidence that expert size matters: "
                "it is dominated by",
                "    deepseek-v2-lite's single two-tread ladder. The matched "
                "mixtral/qwen2 pairs are P4."]

    counts = {"reports read": len(rows) + len(skipped),
              "BM=128 ladders": len(rows),
              "valid references": len(valid),
              "tread boundaries": sum(max(0, len(r.record.points) - 1) for r in rows),
              "bootstrap resamples": sum(r.margin.draws for r in rows)}
    gates = [gate_non_vacuity(counts)]

    # P6 / P7 / P8 are scored here because the corpus can settle them today.
    fail_v2 = [r for r in rows
               if any(i.sigma is None or i.sigma >= MONOTONE_SIGMA
                      for i in r.inversions)]
    gates.append(Gate(
        CLAIM, "P6 monotonicity discriminates",
        "exactly one published ladder runs backwards",
        "V2 fails on 1 ladder", len(fail_v2) == 1,
        f"{len(fail_v2)} of {len(rows)} fail V2: "
        + ("; ".join(f"{r.record.card.upper()} {r.record.model} "
                     f"g{r.record.group_m}" for r in fail_v2) or "none"),
        lines=[i.line() for r in fail_v2 for i in r.inversions]))

    # Split deliberately: a corrupt reference and a missing one fail the same
    # gate for different reasons, and lumping them turned this prediction into a
    # count nobody could check.
    no_ref = [r for r in rows if r.level is None]
    bad_level = [r for r in rows if r.level is not None and not r.level.passes]
    gates.append(Gate(
        CLAIM, "P7 the level bar finds the corrupt references",
        "exactly the two BLOCK_N=256 arms fail the level bar",
        "the level bar fails on 2 ladders that HAVE a reference, both "
        "BLOCK_SIZE_N=256",
        len(bad_level) == 2 and all(r.record.block_n == 256 for r in bad_level),
        f"{len(bad_level)} of {len(rows) - len(no_ref)} referenced ladders fail "
        "the level bar: "
        + ("; ".join(f"{r.record.card.upper()} BN={r.record.block_n} at "
                     f"{r.level.fraction:.1%}" for r in bad_level) or "none")
        + f"; {len(no_ref)} further ladder(s) have NO reference to check: "
        + ("; ".join(f"{r.record.card.upper()} {r.record.model} "
                     f"g{r.record.group_m} n{r.record.block_n}" for r in no_ref)
           or "none"),
        lines=["`compute_reference` tests proportionality only, so a reference "
               "43.6x too slow qualified and then classified every tread of "
               "every ladder in its arm as compute bound."]))

    gates.append(_gate_escape_down_alpha(rows))
    gates.append(_gate_expert_size(valid))
    gates.append(_gate_depth_invariance(valid))

    quoted = [r for r in rows if r.record.memory_points >= 2]
    survivors = [r for r in quoted if r.admissible]
    gates.append(Gate(
        CLAIM, "P8 no admissible BM=128 fit exists",
        "zero of the study's BLOCK_M=128 fits survive V1+V2+V3+C2",
        "0 survivors", len(survivors) == 0,
        f"{len(quoted)} ladder(s) reported a memory branch, {len(survivors)} "
        "survive",
        lines=[f"{r.record.card.upper()} {r.record.model} g{r.record.group_m} "
               f"n{r.record.block_n}: " + "; ".join(r.reasons) for r in quoted]))

    payload = {
        "subject_block_m": SUBJECT_BLOCK_M,
        "tolerance": TOLERANCE,
        "target_treads": TARGET_TREADS,
        "escape_up_alpha_rho": up,
        "escape_down_rho": down,
        "counts": counts,
        "ladders": [{"arm": r.record.arm, "card": r.record.card,
                     "model": r.record.model, "group_m": r.record.group_m,
                     "block_n": r.record.block_n,
                     "treads": len(r.record.points),
                     "memory_points": r.record.memory_points,
                     "reference_fraction": r.level.fraction if r.level else None,
                     "ratio": r.ratio, "margin": None if math.isnan(r.margin.margin)
                     else r.margin.margin,
                     "margin_sigma": r.margin.sigma,
                     "inversions": len(r.inversions), "slope_drops": len(r.drops),
                     "admissible": r.admissible, "reasons": r.reasons}
                    for r in rows],
        "skipped": skipped,
        "gates": [{"kind": g.kind, "name": g.name, "verdict":
                   {True: "PASS", False: "FAIL", None: "UNKNOWN"}[g.passed],
                   "observed": g.observed} for g in gates],
    }
    return out, gates, payload


# --------------------------------------------------------------------------
# The pod run: a replicated ladder at the subject block size.
# --------------------------------------------------------------------------

@dataclass
class Sample:
    """One timing of one tread in one repeat. The CSV row.

    THE STATE THE TIMING WAS TAKEN IN IS A COLUMN, not a property of the
    session. Until 2026-09-02 this row carried an iteration count and nothing
    else, and the loop that filled it created its CUDA events inside the timed
    region, synchronised after every call, never flushed L2 and never read a
    clock -- while the compute reference every tread here is classified against
    was measured queue-deep by `moe.bench.timing`. The audit bounded the host
    prefix that let inside the interval at 0.18 ms per `fused_experts` call on
    the H200 and 0.30 ms on the A100, which is a per-card bias in the fitted
    alpha of 8-16% at the smallest treads: the same size as the cross-card
    effect this study registered. Every row now comes from
    `timing.time_kernel` under `TIMING_BASIS`, and the fields below are what
    that instrument reported about the measurement it had just made.

    THE THREE CLOCK FIELDS ARE OPTIONAL AND None MEANS "NOT DETERMINED", never
    "fine". A container without NVML, a trial too short for the poller to land a
    sample, and a replay all produce None, and a reader that treats None as True
    re-admits exactly the rows the column exists to flag.
    """

    block_m: int
    tiles: int
    rows_per_expert: int
    tokens: int
    rep: int
    ms_p50: float
    ms_min: float
    ms_stdev: float
    iters: int
    status: str = "ok"
    detail: str = ""
    #: `moe.bench.timing.TIMING_BASIS` of the loop that produced `ms_p50`.
    #: Empty means a row from before the instrument had a name, or a planted one.
    instrument: str = ""
    #: MILLISECONDS of delivered GPU load the warmup ran for, not a call count.
    warmup_ms: float = 0.0
    trials: int = 0
    sm_clock_load_mhz: float | None = None
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    l2_flush: bool = False

    @property
    def clock_excluded(self) -> bool:
        """Was this timing taken below the clock the roof was measured at.

        False for None on purpose, and this is the one place that reading is
        correct: an EXCLUSION has to be positively established. A row with no
        clock is a row whose comparability is unknown, and the report says how
        many of those there are rather than dropping them.
        """
        return self.clock_level_ok is False


SAMPLE_FIELDS = list(Sample.__dataclass_fields__)


def ladder_rows(cfg, block_m: int, r_max: int) -> list[int]:
    """Exactly-full tile stacks only: `r = n BM`, zero padding, one per tread.

    No probes and no background grid. This experiment reads the ladder and
    nothing else, so a grid point that is not a tread top is GPU time spent on
    a number no gate here will look at. REFUSES when the model's routing makes
    `n BM` an impossible token count rather than nudging it: a nudged row is not
    a full tile stack, and a fit over partly-filled treads is a fit over
    padding.

    BOTH REFUSALS ARE `RefusedBeforeMeasuring`, so the process exits 2. They
    were `raise SystemExit(<str>)` until 2026-09-02, which exits 1 = CLAIM_FAIL,
    and a plan that could not be built announced itself to the driver as a
    measured refutation of the depth claim.
    """
    q = SWEEP.rows_quantum(cfg)
    rows = []
    for n in range(1, r_max // block_m + 1):
        r = n * block_m
        if r % q:
            raise _refuse(
                f"{cfg.num_experts} experts at top-k {cfg.top_k} need rows per "
                f"expert to be a multiple of {q}, and {r} (tread {n} at "
                f"BLOCK_M={block_m}) is not. This model cannot form an exactly "
                f"full tile stack at this block size; choose another --model.")
        rows.append(r)
    if not rows:
        raise _refuse(f"--r-max {r_max} is below one tile at BLOCK_M="
                      f"{block_m}; nothing to measure.")
    return rows


@dataclass(frozen=True)
class Plan:
    """Everything the pod run will do, computable on a laptop."""

    model: str
    dtype: str
    pinned: dict
    reference_rows: list[int]
    subject_rows: list[int]
    reps: int
    iters: int
    #: MILLISECONDS of delivered GPU load, not a call count. See `build_parser`.
    warmup: float
    trials: int
    l2_flush: bool
    cell_budget_ms: float
    estimated_seconds: float
    cells: int

    def lines(self, cfg) -> list[str]:
        return [
            f"model        {self.model} E={cfg.num_experts} k={cfg.top_k} "
            f"{self.dtype}",
            f"pinned       {self.pinned}",
            f"reference    BLOCK_M={REFERENCE_BLOCK_M} at rows "
            f"{self.reference_rows} -- MEASURED FIRST, and its level checked "
            "before the subject costs anything",
            f"subject      BLOCK_M={SUBJECT_BLOCK_M} at rows {self.subject_rows}",
            f"repeats      {self.reps} round-robin passes per setting, so an "
            "inversion can be told from noise",
            f"timing       moe.bench.timing.time_kernel: {self.warmup:.0f} ms "
            f"of delivered warmup load, then {self.trials} queue-deep trials "
            f"of {self.cell_budget_ms:.0f} ms each, L2 "
            + ("flushed" if self.l2_flush else "NOT flushed (--no-l2-flush)")
            + " between iterations, SM clock sampled under load",
            f"             --iters {self.iters} is RETIRED as a timing knob and "
            "is kept only in the run id; the instrument sizes each cell's "
            "iteration count from --cell-budget-ms and records it per row",
            f"cells        {self.cells} timings "
            f"({len(self.reference_rows) + len(self.subject_rows)} treads x "
            f"{self.reps} reps)",
            f"estimate     {self.estimated_seconds:.0f} s of GPU at the model's "
            "own timings, excluding compiles and allocation",
        ]


def build_plan(args, cfg) -> Plan:
    """The pod plan and its cost, priced the way `time_kernel` charges.

    Takes no alpha, ridge or bandwidth: under the instrument the bill is a
    warmup duration plus `--trials` budgets per cell and does not depend on how
    fast the kernel is. They used to be parameters and the estimate used to be
    `ms x (warmup + iters)`, which is a call-count formula and became wrong the
    moment `--warmup` became a duration.
    """
    pinned = dict(SWEEP.FIXED, num_stages=args.num_stages,
                  num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                  BLOCK_SIZE_N=args.block_n, BLOCK_SIZE_K=args.block_k)
    ref_rows = ladder_rows(cfg, REFERENCE_BLOCK_M, args.r_max)
    sub_rows = ladder_rows(cfg, SUBJECT_BLOCK_M, args.r_max)
    # PRICED AS THE INSTRUMENT CHARGES, not as the retired loop did. A cell
    # costs a fixed warmup DURATION plus `--trials` trials of `--cell-budget-ms`
    # of kernel time each, and it costs that whatever the kernel's own time is,
    # because `time_kernel` sizes the iteration count to fill the budget. The
    # old form multiplied the modelled milliseconds by a call count, which read
    # a duration as a count once the units changed and priced an 8-tread ladder
    # at seconds when it costs minutes. `alpha`, `ridge` and `bandwidth_gbps`
    # no longer enter the estimate at all and the docstring says so rather than
    # leaving three unused arguments looking load-bearing.
    per_cell_ms = args.warmup + args.trials * args.cell_budget_ms
    cells = args.reps * (len(ref_rows) + len(sub_rows))
    return Plan(args.model, args.dtype, pinned, ref_rows, sub_rows, args.reps,
                args.iters, args.warmup, args.trials, not args.no_l2_flush,
                args.cell_budget_ms, cells * per_cell_ms / 1e3, cells)


def collapse(samples: list[Sample], block_m: int
             ) -> tuple[list[tuple[int, float]], dict[int, list[float]], float | None]:
    """Per-tread median across repeats, the repeats themselves, and the spread.

    The median across REPEATS rather than the single-pass median: a repeat is a
    fresh compile-free call at a fresh point in the pod's thermal history, and
    the study's one non-monotone ladder is exactly what a single pass cannot
    distinguish from a mechanism.
    """
    by: dict[int, list[float]] = {}
    for s in samples:
        if s.block_m == block_m and s.status == "ok" and s.ms_p50 > 0:
            by.setdefault(s.tiles, []).append(s.ms_p50)
    points = [(n, statistics.median(v)) for n, v in sorted(by.items())]
    return points, by, _spread_of(by)


def _spread_of(reps: dict[int, list[float]]) -> float | None:
    """Median across-repeat relative spread over the treads handed in.

    SPLIT OUT OF `collapse` SO THE SPREAD CAN BE TAKEN OVER THE SCORED TREADS.
    `collapse` sees every tread the CSV holds; `analyse_run` scores only the
    treads `ladder_treads` admitted, and the spread it weighs an inversion or a
    slope drop against has to come from the same set. A spread taken over treads
    that were excluded for clock level is a noise band measured partly on
    another compute branch, and it is the denominator of every sigma on the
    page.

    None, never 0.0, when no tread has two repeats: "we do not know how noisy
    this was" and "this was 0% noisy" are different states and only one of them
    lets an inversion be dismissed.
    """
    spreads = [statistics.pstdev(v) / statistics.median(v)
               for v in reps.values() if len(v) > 1 and statistics.median(v) > 0]
    return statistics.median(spreads) if spreads else None


def tread_clock(samples: list[Sample], block_m: int
                ) -> dict[int, tuple[bool | None, bool | None, float | None]]:
    """Per tread: `(clock_level_ok, clock_drift_ok, median loaded SM clock)`.

    THE INSTRUMENT'S COLUMNS HAVE TO REACH THE FIT OR THEY ARE DECORATION.
    `collapse` takes a median across repeats and throws everything else away, so
    without this the clock a tread was timed at never reaches `make_cell` and
    `ladder_treads` -- which excludes a tread whose loaded clock came in below
    the clock the roof was measured at -- has nothing to exclude on. A run on a
    throttling card would then fit its ladder on treads taken at two different
    compute branches and report the result as one.

    MAJORITY, NOT ANY, and the reason is the same as the sweep's membership
    rule. A single throttled repeat out of seven is what the median across
    repeats exists to absorb; a tread is excluded when MOST of its repeats were
    taken below the reference clock, which is when its median is the throttled
    number. `None` (not determined) is not evidence either way and is returned
    as None: a reader that treats it as True re-admits exactly the rows the
    column exists to flag.
    """
    out: dict[int, tuple[bool | None, bool | None, float | None]] = {}
    by: dict[int, list[Sample]] = {}
    for s in samples:
        if s.block_m == block_m and s.status == "ok" and s.ms_p50 > 0:
            by.setdefault(s.tiles, []).append(s)
    for tiles, group in sorted(by.items()):
        def _majority(flags):
            known = [f for f in flags if f is not None]
            if not known:
                return None
            return sum(1 for f in known if f) * 2 > len(known)
        clocks = [s.sm_clock_load_mhz for s in group
                  if s.sm_clock_load_mhz is not None]
        out[tiles] = (_majority([s.clock_level_ok for s in group]),
                      _majority([s.clock_drift_ok for s in group]),
                      statistics.median(clocks) if clocks else None)
    return out


def drift(samples: list[Sample], block_m: int) -> float | None:
    """Median relative change from the first repeat to the last, per tread.

    A pod whose clocks ramp during a setting produces a ladder that bends, and a
    ladder that bends is read by a two-line fit as a second mechanism. Reported
    rather than gated because the sign is informative either way: a negative
    drift is a warm-up, a positive one is a thermal fade.
    """
    firsts: dict[int, float] = {}
    lasts: dict[int, float] = {}
    for s in sorted(samples, key=lambda s: s.rep):
        if s.block_m != block_m or s.status != "ok" or s.ms_p50 <= 0:
            continue
        firsts.setdefault(s.tiles, s.ms_p50)
        lasts[s.tiles] = s.ms_p50
    rel = [lasts[n] / firsts[n] - 1.0 for n in firsts if firsts[n] > 0]
    return statistics.median(rel) if rel else None


def append_sample(path: Path, sample: Sample) -> None:
    """One row, flushed. An abort costs the timing in flight and nothing else."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(asdict(sample))
        fh.flush()


def _opt_float(raw) -> float | None:
    """`None` for an empty cell, a float otherwise. None is NOT zero.

    A clock column read back as 0.0 would say the card ran at 0 MHz, which no
    filter can distinguish from a genuine reading; None says the sampler landed
    nothing, which is what an empty cell means.
    """
    text = (raw or "").strip()
    if not text or text.lower() in ("none", "null"):
        return None
    return float(text)


def _opt_bool(raw) -> bool | None:
    """Three states, because the clock verdicts have three. See `Sample`."""
    text = (raw or "").strip().lower()
    if not text or text in ("none", "null"):
        return None
    return text in ("1", "true", "yes")


def read_samples(path: Path) -> tuple[set[tuple[int, int, int]], list[Sample]]:
    if not path.exists():
        return set(), []
    out: list[Sample] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            # `.get` with a default on the instrument columns, and NOT on the
            # measurement ones: a row written before 2026-09-02 has no
            # instrument name, and the honest reading of that is "" (the
            # instrument was not recorded), not a crash and not an invented
            # name. A missing `ms_p50` is a corrupt file and still raises.
            out.append(Sample(
                block_m=int(row["block_m"]), tiles=int(row["tiles"]),
                rows_per_expert=int(row["rows_per_expert"]),
                tokens=int(row["tokens"]), rep=int(row["rep"]),
                ms_p50=float(row["ms_p50"]), ms_min=float(row["ms_min"]),
                ms_stdev=float(row["ms_stdev"]), iters=int(row["iters"]),
                status=row.get("status", "ok"), detail=row.get("detail", ""),
                instrument=row.get("instrument", ""),
                warmup_ms=float(row.get("warmup_ms") or 0.0),
                trials=int(row.get("trials") or 0),
                sm_clock_load_mhz=_opt_float(row.get("sm_clock_load_mhz")),
                clock_level_ok=_opt_bool(row.get("clock_level_ok")),
                clock_drift_ok=_opt_bool(row.get("clock_drift_ok")),
                l2_flush=(row.get("l2_flush", "") or "").strip()
                         .lower() in ("1", "true", "yes")))
    # Only successful timings count as done, for the same reason the sweep does
    # it: a failed cell is usually a pod that lost its device, and a real
    # failure fails again in milliseconds.
    return {(s.block_m, s.tiles, s.rep) for s in out if s.status == "ok"}, out


def measure_setting(args, cfg, block_m: int, rows: list[int], csv_path: Path,
                    cache_root: Path, pinned: dict, done, samples: list[Sample],
                    reference_clock: float | None = None) -> tuple[int, int]:
    """Time one block size, `--reps` round-robin passes over its treads.

    ROUND ROBIN INSIDE THE SETTING, not tread-by-tread to completion. Measuring
    tread 1 fifty times and then tread 8 fifty times puts every tread at a
    different point in the pod's thermal history, and the resulting monotone
    drift IS a slope -- the very quantity being fitted. One pass over all treads
    per repeat spreads that drift across the whole ladder instead of aligning it
    with the x axis.

    ONE INSTRUMENT, AND IT IS THE ROOF'S. Every timing goes through
    `moe.bench.timing.time_kernel`: queue-deep, one event pair per iteration
    primed before the loop, L2 evicted between iterations unless
    `--no-l2-flush`, the SM clock sampled UNDER LOAD against the clock the
    calibration's dense GEMM ran at, and the iteration count sized from
    `--cell-budget-ms` rather than asserted. `reference_clock` is resolved ONCE
    per run, before any tread, so the whole ladder is scored against one number
    and a mid-run yaml rewrite cannot move it; without it every row's
    `clock_level_ok` is None, which means "not determined" and excludes nothing.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.bench import timing
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    override_config, _ = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    SWEEP.arm_triton_cache(cache_root, block_m)
    seen: set[Path] = set()
    SWEEP.count_new(cache_root, seen)
    compiles = executed = 0
    built: dict[int, tuple] = {}

    for rep in range(1, args.reps + 1):
        for r in rows:
            tokens = SWEEP.tokens_for_rows(cfg, r)
            if (block_m, r // block_m, rep) in done:
                continue
            if tokens not in built:
                spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                                 routing=RoutingSpec("uniform", 0.0),
                                 seed=args.seed)
                x, weights = make_inputs(spec, device="cuda")
                ids = SWEEP.balanced_ids(cfg, tokens, "cuda")
                w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                               device="cuda")
                kw = vllm_call_kwargs(spec)
                kw["activation"] = MoEActivation(kw["activation"])
                built = {tokens: (x, weights, ids, w, kw)}   # one cell live
            x, weights, ids, w, kw = built[tokens]
            executed += 1
            conf = dict(pinned, BLOCK_SIZE_M=block_m)

            def call(_f=fused_experts, _x=x, _wt=weights, _w=w, _i=ids, _k=kw):
                return _f(hidden_states=_x, w1=_wt.w1, w2=_wt.w2,
                          topk_weights=_w, topk_ids=_i, **_k)

            try:
                with override_config(conf):
                    call()
                    torch.cuda.synchronize()
                    compiles += SWEEP.count_new(cache_root, seen)
                    # NO PILOT CALL AND NO `scaled_iters`. `time_kernel` sizes
                    # its own iteration count from the warmup's queue-deep
                    # per-call time, which is the same measurement the pilot was
                    # for and is taken with the instrument that will do the
                    # timing rather than with a different one.
                    t = timing.time_kernel(
                        call, warmup_ms=args.warmup,
                        target_ms=args.cell_budget_ms, trials=args.trials,
                        l2_flush=not args.no_l2_flush,
                        reference_clock_mhz=reference_clock)
                sample = Sample(block_m, r // block_m, r, tokens, rep, t.ms_p50,
                                t.ms_min, t.ms_std, t.iters,
                                instrument=t.instrument, warmup_ms=t.warmup_ms,
                                trials=t.trials,
                                sm_clock_load_mhz=t.sm_clock_load_mhz,
                                clock_level_ok=t.clock_level_ok,
                                clock_drift_ok=t.clock_drift_ok,
                                l2_flush=t.l2_flush)
                if t.clock_level_ok is False or t.host_bound:
                    print(f"  ^ {t.clock_note or ''} {t.host_note or ''}".rstrip())
            except timing.TimingRefused:
                # THE SECOND DOOR INTO THE SAME ROOM. `RefusedBeforeMeasuring`
                # is a `SystemExit` precisely because this handler swallows a
                # RuntimeError -- the class docstring says so -- and
                # `timing.TimingRefused` subclasses RuntimeError, so every
                # refusal the INSTRUMENT ITSELF raises walked through the door
                # that was left open beside it: no CUDA and no injected fakes,
                # trials=0, a warmup that makes the measurement meaningless.
                # Each is a fact about the RUN and identical for every tread, so
                # the ladder wrote a `status="failed"` sample per tread, ground
                # through both block sizes, and scored its gates over a page of
                # zeroes. Re-raised to `main`, which exits REFUSED: nothing was
                # measured and nothing was spent. The BASE class and not a
                # subclass, the way `driver.run_cell` names it, so a refusal
                # added to the instrument later cannot reintroduce the bug by
                # forgetting to add itself here.
                raise
            except Exception as exc:                    # noqa: BLE001
                sample = Sample(block_m, r // block_m, r, tokens, rep, 0.0, 0.0,
                                0.0, 0, "failed", f"{type(exc).__name__}: {exc}")
                print(f"  BM={block_m} n={r // block_m} rep={rep} FAILED "
                      f"{sample.detail}")
            samples.append(sample)
            append_sample(csv_path, sample)
            print(f"  BM={block_m:3d} n={r // block_m:2d} rep={rep:2d} "
                  f"r={r:5d} T={tokens:6d}  {sample.ms_p50:9.4f} ms "
                  f"({sample.iters} iters)")
    return compiles, executed


def analyse_run(samples, cfg, b: int, ceiling_tflops: float, ceiling_source: str,
                compiles: dict[int, int], executed: dict[int, int], *,
                ridge: float, bandwidth_gbps: float, pinned: dict | None = None,
                seed: int = 0, draws: int = BOOTSTRAP_DRAWS
                ) -> tuple[list[str], list[Gate], dict]:
    """The measured run's gates. Same diagnostics the audit runs on the corpus."""
    out: list[str] = []
    ref_points, _, ref_spread = collapse(samples, REFERENCE_BLOCK_M)
    sub_points, sub_reps, sub_spread = collapse(samples, SUBJECT_BLOCK_M)

    # The compute reference comes from the study's own qualification, run over
    # Cell objects built from these medians, so this run is judged by the same
    # code the published surface was.
    # THE CLOCK COLUMNS TRAVEL WITH THE MEDIAN. `ladder_treads` excludes a
    # tread whose loaded clock came in below the clock the roof was measured at,
    # and it can only do that if the verdict reaches the Cell. Before 2026-09-02
    # nothing here carried one, so a ladder fitted across a throttling episode
    # was indistinguishable from one that was not.
    cells = []
    clocks = {bm: tread_clock(samples, bm)
              for bm in (REFERENCE_BLOCK_M, SUBJECT_BLOCK_M)}
    for bm, points in ((REFERENCE_BLOCK_M, ref_points),
                       (SUBJECT_BLOCK_M, sub_points)):
        for n, ms in points:
            level, drift_ok, mhz = clocks[bm].get(n, (None, None, None))
            cells.append(SWEEP.make_cell(cfg, n * bm, bm, ms,
                                         sm_count=1, block_n=1,
                                         sm_clock_load_mhz=mhz,
                                         clock_level_ok=level,
                                         clock_drift_ok=drift_ok))
    ref = SWEEP.compute_reference(
        cells, (SUBJECT_BLOCK_M, REFERENCE_BLOCK_M), cfg=cfg, ridge=ridge,
        bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned)
    level = (reference_level(cfg, ref.block_m, ref.slope_per_tile,
                             ceiling_tflops, ceiling_source)
             if ref.block_m and ref.slope_per_tile else None)

    moved = drift(samples, SUBJECT_BLOCK_M)
    # THE FIT READS THE SIBLING'S LADDER, NOT THE RAW MEDIANS, and that is what
    # makes the clock columns matter. `ladder_treads` drops a tread whose loaded
    # clock came in below the clock the roof was measured at -- membership is
    # decided against `C = 2 BM N / peak` and a tread taken at 1000 MHz against
    # a roof measured at 1980 sits well above that line for a reason that is not
    # weight re-reads -- and returns how many it dropped. Passing `sub_points`
    # here instead, which is what this file used to do, fitted those treads in
    # and reported the card as the tile.
    fit_points, excluded = SWEEP.ladder_treads(
        [c for c in cells if c.block_m == SUBJECT_BLOCK_M], SUBJECT_BLOCK_M)
    # THE EXCLUSION REACHES EVERY GATE THAT READS THE SUBJECT LADDER, not just
    # the fit. Until 2026-09-02 it stopped at `fit_ladder`: V2's inversions, V3's
    # slope sequence, V5's replication and the membership band were all computed
    # over `sub_points`, the UNEXCLUDED medians. So a tread the fit had decided
    # sits on a different compute branch could still trip a VALIDITY gate --
    # INVALID, nothing quotable -- for exactly the reason the exclusion exists to
    # discount, and a throttled ladder came out INVALID-by-bend instead of
    # UNDECIDED-by-clock. Those point at different next steps: one says the
    # instrument is broken, the other says re-time on a quiet card. `ladder_treads`
    # is the one place a tread is admitted, and `kept_*` below is what every gate
    # scores. The full ladder is still PRINTED, marked, because a reader must be
    # able to see what was dropped.
    kept = {n for n, _ in fit_points}
    kept_reps = {n: v for n, v in sub_reps.items() if n in kept}
    kept_spread = _spread_of(kept_reps)
    margin_band = max(SWEEP.MEMORY_BRANCH_MARGIN, 3.0 * (kept_spread or 0.0))
    fit = SWEEP.fit_ladder(fit_points, SUBJECT_BLOCK_M, ref, margin=margin_band,
                           excluded_low_clock=excluded)
    c_ref = ref.slope_for(SUBJECT_BLOCK_M)
    k = fit.memory_points if fit.memory_points >= 2 else len(fit_points)
    margin = margin_of(fit_points, k, c_ref=c_ref, overhead=ref.overhead_ms,
                       spread=kept_spread, replicates=kept_reps, draws=draws,
                       seed=seed)
    inv = inversions(fit_points, kept_spread)
    drops = slope_drops(fit_points, kept_spread)

    out += ["", "## The measured ladder", "",
            f"reference    {ref.note}",
            f"             {level.line() if level else 'no level: no reference'}",
            f"subject      {len(sub_points)} treads ({len(fit_points)} scored "
            f"after {excluded} excluded for clock level), across-repeat spread "
            + (f"{kept_spread:.3%}" if kept_spread else "UNKNOWN (one repeat)")
            + " over the scored treads"
            + (f" (all {len(sub_points)}: "
               + (f"{sub_spread:.3%}" if sub_spread else "UNKNOWN") + ")"
               if excluded else ""),
            f"             membership margin {margin_band:.3%} "
            f"(floor {SWEEP.MEMORY_BRANCH_MARGIN:.0%}, 3x the measured spread)",
            "             drift first->last repeat "
            + (f"{moved:+.3%}" if moved is not None else "unknown"),
            f"             {fit.basis}",
            # THE OUTCOME IS PRINTED AS A WORD, not inferred from a blank alpha.
            # `fit_ladder` has six named outcomes and two of them are UNDECIDED
            # -- the sweep looked and could not say -- which reads identically
            # to "no treads" if only the alpha is shown.
            f"             ladder outcome: {fit.outcome}"
            + ("  (UNDECIDED: the fit declined to name a branch, which is not "
               "the same as finding none)" if fit.undecided else ""),
            f"             memory treads {fit.memory_points}, alpha "
            + (f"{fit.alpha:.4f}" if fit.alpha is not None
               else ("DECLINED, see the outcome above" if fit.undecided
                     else "NOT IDENTIFIABLE")),
            f"             {margin.line()}",
            (f"             {excluded} tread(s) EXCLUDED: timed below the clock "
             "the roof was measured at, so they sit on a different compute "
             "branch. A ladder that lost treads to a hot box must not look "
             "like a ladder that never had them."
             if excluded else
             "             no tread was excluded for clock level"), "",
            f"{'n':>3s} {'rows':>6s} {'ms':>10s} {'slope':>9s} {'reps':>5s} "
            f"{'spread':>8s}  scored"]
    # EVERY TREAD IS PRINTED, the excluded ones included and marked. The gates
    # score `fit_points`; a table that showed only those would hide the rows a
    # reader needs to see to judge the exclusion, and the count in the header
    # would have nothing to point at.
    display_slopes = slope_sequence(sub_points)
    for i, (n, ms) in enumerate(sub_points):
        reps = sub_reps.get(n, [])
        sp = (statistics.pstdev(reps) / ms) if len(reps) > 1 and ms > 0 else None
        out.append(f"{n:3d} {n * SUBJECT_BLOCK_M:6d} {ms:10.4f} "
                   + (f"{display_slopes[i - 1]:9.4f}" if i else "        -")
                   + f" {len(reps):5d} "
                   + (f"{sp:8.3%}" if sp is not None else "       -")
                   + ("  yes" if n in kept else "  NO: clock below the roof's"))

    gates = [
        gate_non_vacuity({
            "timings": len(samples),
            "subject treads": len(sub_points),
            # SCORED, not merely measured. Every subject gate reads
            # `fit_points`, so a ladder whose treads were all excluded for clock
            # level examined nothing, whatever the measured count says.
            "scored treads": len(fit_points),
            "reference treads": len(ref_points),
            "repeats per tread": min((len(v) for v in sub_reps.values()),
                                     default=0),
            "settings that executed": sum(1 for v in executed.values() if v > 0),
        }, optional={"bootstrap resamples": margin.draws}),
        _gate_override(compiles, executed),
        gate_reference_level(level),
        gate_monotone(inv, max(0, len(fit_points) - 1), kept_spread),
        gate_convex(drops, slope_sequence(fit_points), kept_spread),
        _gate_replication(kept_reps, kept_spread),
        gate_depth(fit.memory_points, len(fit_points),
                   outcome=fit.outcome, undecided=fit.undecided),
        gate_margin(margin),
        _gate_law(fit, margin, c_ref, ref.overhead_ms, margin_band),
    ]
    payload = {
        "subject_block_m": SUBJECT_BLOCK_M,
        "reference": {"block_m": ref.block_m, "slope_per_tile": ref.slope_per_tile,
                      "overhead_ms": ref.overhead_ms, "note": ref.note,
                      "level_fraction": level.fraction if level else None},
        "subject_points": sub_points,
        "scored_points": fit_points,
        "subject_spread": sub_spread,
        "scored_spread": kept_spread,
        "memory_points": fit.memory_points,
        # THE OUTCOME AND THE UNDECIDED FLAG ARE COLUMNS. A consumer reading
        # only `alpha: null` cannot tell a fit that found nothing from a fit
        # that declined to answer, and those point at different experiments.
        "ladder_outcome": fit.outcome,
        "ladder_undecided": fit.undecided,
        "excluded_low_clock": fit.excluded_low_clock,
        "alpha": fit.alpha, "alpha_upper": fit.alpha_upper,
        "slope_memory": fit.slope_memory, "slope_compute_ref": c_ref,
        "ratio": margin.ratio if not math.isnan(margin.ratio) else None,
        "margin": None if math.isnan(margin.margin) else margin.margin,
        "margin_sd": margin.sd, "margin_sigma": margin.sigma,
        "margin_basis": margin.basis,
        "inversions": [asdict(i) for i in inv],
        "slope_drops": [asdict(d) for d in drops],
        "gates": [{"kind": g.kind, "name": g.name, "verdict":
                   {True: "PASS", False: "FAIL", None: "UNKNOWN"}[g.passed],
                   "observed": g.observed} for g in gates],
    }
    return out, gates, payload


def _gate_override(compiles: dict[int, int], executed: dict[int, int]) -> Gate:
    """Did `override_config` change the kernel at each setting.

    Same assay as the sweep's gate 0 and for the same reason: if the override
    silently failed, both settings ran one kernel, `C` and `B` are the same
    line by construction, and the tolerance verdict is a comparison of a kernel
    with itself. A setting that executed nothing this session is UNDECIDED, not
    a failure: the assay belongs to whichever session ran the cells.
    """
    ran = [bm for bm, n in executed.items() if n > 0]
    resumed = [bm for bm, n in executed.items() if n <= 0]
    missing = [bm for bm in ran if compiles.get(bm, 0) <= 0]
    counts = ", ".join(f"BM={bm}:{compiles.get(bm, 0)}"
                       for bm in sorted(executed))
    if missing:
        return Gate(VALIDITY, "V4 override took effect",
                    "each setting compiled its own kernel",
                    ">= 1 fresh Triton artefact per setting that ran", False,
                    f"{counts}; {missing} ran cells and compiled nothing",
                    "both slopes: one kernel timed twice makes B/C exactly 1 "
                    "and the tolerance discard a foregone conclusion")
    if resumed and not ran:
        return Gate(VALIDITY, "V4 override took effect",
                    "each setting compiled its own kernel",
                    ">= 1 fresh Triton artefact per setting that ran", None,
                    f"{counts}; every setting was resumed from cells.csv, so "
                    "this session ran no assay",
                    "both slopes")
    return Gate(VALIDITY, "V4 override took effect",
                "each setting compiled its own kernel",
                ">= 1 fresh Triton artefact per setting that ran", True, counts,
                "both slopes")


def _gate_replication(reps: dict[int, list[float]], spread: float | None) -> Gate:
    worst = min((len(v) for v in reps.values()), default=0)
    if spread is None:
        return Gate(VALIDITY, "V5 replication",
                    "the ladder was measured often enough to weigh an inversion",
                    f">= 2 repeats per tread and spread <= "
                    f"{MAX_REPLICATE_SPREAD:.0%}", None,
                    f"{worst} repeat(s) on the thinnest tread, so no "
                    "across-repeat spread exists",
                    "V2 and C2, which both need a noise band; without one an "
                    "inversion cannot be told from a mechanism, which is how "
                    "the published A100 ladder shipped")
    return Gate(VALIDITY, "V5 replication",
                "the ladder was measured often enough to weigh an inversion",
                f">= 2 repeats per tread and spread <= "
                f"{MAX_REPLICATE_SPREAD:.0%}",
                worst >= 2 and spread <= MAX_REPLICATE_SPREAD,
                f"{worst} repeats on the thinnest tread, median across-repeat "
                f"spread {spread:.3%}",
                "V2 and C2, which both need a noise band")


def _gate_law(fit, margin: Margin, c_ref: float | None, overhead: float,
              band: float) -> Gate:
    """Are the fitted `(alpha, B/C)` and the observed tread count consistent.

    RELABELLED FROM A CLAIM ON 2026-09-02, and the relabelling is the point.
    This gate computes `n*` from `(alpha, B/C)` FITTED ON THE SAME LADDER whose
    tread count it then compares `n*` against. For any ladder that really is two
    lines the two agree by construction, so it can only fail when the two-line
    model does not describe the data -- which is a statement about the FIT, not
    about the world. Carried as `C3 the law predicts the depth` it read as a
    pre-registered prediction confirmed by the run, and a reader counting the
    CLAIM gates counted it as evidence for the law. It is a self-consistency
    check on the fit, it belongs with the VALIDITY gates that decide whether the
    alpha above may be quoted, and it is scored as one.

    WHAT WOULD MAKE IT A CLAIM AGAIN. `n*` computed from an alpha and a B/C
    registered BEFORE this ladder was measured -- the corpus median, say -- and
    compared with this ladder's count. That is a real prediction and a different
    gate; this file does not make it, and says so rather than implying it.

    A mismatch says the ladder is not two lines, which is the same thing V3
    reports from the other direction, and having both makes it checkable rather
    than asserted.

    WHAT IT DOES WHEN THERE IS NO ALPHA, and why that is a PASS. A VALIDITY gate
    that is not PASS exits INVALID, "nothing on this page may be quoted", and
    the driver reads it as a spent arm to be re-run. This gate's subject is the
    alpha in the report. When the fit named none -- `undecided_parallel_branch`,
    which is what BOTH cards produce at BLOCK_M=128 and is this arm's own
    registered expectation -- there is no alpha in the report, so there is
    nothing to certify AND NOTHING TO WITHHOLD, and voiding the page over it
    makes the twelve-minute arm RETRY by construction on every card that exists.
    That is exactly the defect this phase removes elsewhere; recreating it here
    would be the same defect in another file. The depth verdict is C1's, and C1
    already returns UNKNOWN for those outcomes, which is CLAIM_FAIL: a result,
    not a retry.

    So the rule is scoped rather than blanket, and the scope is "is there a
    number here that this gate is responsible for":

      no alpha at all              PASS, and the observed line says it is
                                   vacuous and names the outcome.
      an alpha this gate cannot    UNKNOWN. `undecided_low_clock` leaves a fit
      evaluate                     over the two treads that survived the
                                   exclusion, and a run with no compute
                                   reference leaves no `B/C` to put into the
                                   law. In both cases an alpha IS in the payload
                                   and this gate did not certify it, so it
                                   withholds and the page is INVALID.
      an alpha and a `B/C`         the comparison below, PASS or FAIL.
    """
    if fit.alpha is None:
        # VACUOUS, AND SAID OUT LOUD. `gate_non_vacuity` is the reason this is
        # safe to call a PASS: the counts that say this report examined real
        # work are scored there, not here, so a PASS here cannot be a check that
        # examined nothing pretending to be a check that found nothing.
        why = (f"the ladder fit is {fit.outcome}, so it declined to name an "
               "alpha rather than failing to find treads"
               if getattr(fit, "undecided", False) else
               "no memory branch was identified, so there is no alpha to put "
               "into the law")
        return Gate(VALIDITY, "V6 fit self-consistency",
                    "the observed tread count matches n* from the fitted "
                    "(alpha, B/C)",
                    "within 1 tread, or vacuous when the report carries no alpha",
                    True, f"VACUOUS: {why}",
                    "nothing: there is no alpha in this report for this gate to "
                    "certify, so it withholds nothing. C1 carries the depth "
                    "verdict and returns UNKNOWN for the same outcome",
                    lines=["This gate is a check on the FIT and not a "
                           "prediction about the world; see its docstring.",
                           "A PASS here is NOT a statement that the ladder is "
                           "two lines. It is a statement that the report "
                           "publishes no alpha, so no alpha can be mis-quoted "
                           "out of it."])
    if fit.undecided or margin.ratio is None or math.isnan(margin.ratio) \
            or not c_ref:
        # AN ALPHA THE GATE CANNOT EVALUATE IS WITHHELD, NOT WAIVED. A ladder
        # that lost most of its treads to clock level still yields an alpha from
        # the two that survived, `payload["alpha"]` carries it, and certifying
        # it would publish exactly the number the exclusion exists to withhold.
        why = (f"the ladder fit is {fit.outcome} and yet reports alpha "
               f"{fit.alpha:.4f}, fitted on the treads that survived; the "
               "instrument declined to name a branch, so that alpha is not "
               "certified here"
               if getattr(fit, "undecided", False) else
               f"alpha {fit.alpha:.4f} was fitted, but there is no usable "
               "compute slope B/C to put into the law, so n* cannot be computed")
        return Gate(VALIDITY, "V6 fit self-consistency",
                    "the observed tread count matches n* from the fitted "
                    "(alpha, B/C)",
                    "within 1 tread, or vacuous when the report carries no alpha",
                    None, why,
                    "the alpha and the tread count above, which are only "
                    "quotable if the ladder is the two lines the fit assumed",
                    lines=["This gate is a check on the FIT and not a "
                           "prediction about the world; see its docstring."])
    n_star = prefix_depth(margin.ratio, fit.alpha, overhead / c_ref, band)
    predicted = len(fit.points) if n_star is None else max(0, math.floor(n_star))
    return Gate(VALIDITY, "V6 fit self-consistency",
                "the observed tread count matches n* from the fitted (alpha, B/C)",
                "within 1 tread, or vacuous when the report carries no alpha",
                abs(predicted - fit.memory_points) <= 1,
                f"law predicts {predicted} "
                + ("(every tread: B/C is above the membership margin)"
                   if n_star is None else f"(n* = {n_star:.2f})")
                + f", fit found {fit.memory_points}",
                "the alpha and the tread count above, which are only quotable "
                "if the ladder is the two lines the fit assumed",
                lines=["A mismatch says the ladder is not two lines, which is "
                       "the same thing V3 reports from the other direction.",
                       "NOT A PREDICTION: n* is computed from an alpha and a "
                       "B/C fitted on this same ladder, so agreement is "
                       "self-consistency and not confirmation."])


# --------------------------------------------------------------------------
# Output paths, and whether git will keep them.
# --------------------------------------------------------------------------

def git_visibility(path: Path) -> str:
    """Say out loud whether git would keep this file.

    `.gitignore` ignores `results/*` and re-includes only `results/published/`,
    so a run that writes anywhere else under the repo produces files that `git
    add -A` silently drops. This project has already lost every published plot
    that way. Checked with `git check-ignore` rather than by re-implementing
    the pattern rules, because the pattern rules are what got it wrong.
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=ROOT, capture_output=True, timeout=15,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git check-ignore could not run ({exc}); path unverified"
    if proc.returncode == 0:
        return ("IGNORED by git. Nothing written here enters the repo. Publish "
                "with scripts/publish_results.sh, or point --out at "
                "results/published/<date>-<gpu>-bm128-depth")
    if proc.returncode == 1:
        return "git will keep this path"
    return (f"git check-ignore exited {proc.returncode}; path unverified "
            f"({proc.stderr.decode(errors='replace').strip()})")


#: The card slug a run id carries when NO device is attached, i.e. every
#: --dry-run on a laptop. Visible rather than blank, so a dry run cannot be
#: mistaken for printing the path a pod will really write to.
UNKNOWN_CARD_SLUG = "nocard"


def detect_card_slug() -> str | None:
    """Slug for the ATTACHED device, or None when there is no device.

    Resolved before the run id is built, because the card belongs IN the id and
    the id is the directory a resumed run reads its treads back out of.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def default_run_id(args, card: str) -> str:
    """Derived from EVERY swept parameter, so two settings cannot collide.

    The sweep this script is built beside lost a whole arm to a run id that
    omitted GROUP_SIZE_M: the second run resumed into the first's directory,
    found every cell present, skipped all of them, and printed the first run's
    timings under the second's heading. BLOCK_SIZE_K, num_warps and --reps are
    in here for the same reason, even though only one value of each has ever
    been run.

    THE CARD IS THE NEXT FIELD IN THAT SAME LIST, and the one that would have
    bitten hardest here. It is not swept by this script, it is swept by the
    operator moving to another pod, and `$MOE_RESULTS_DIR` is a network volume
    that outlives a pod. Every number this file computes -- B/C, alpha x rho,
    both escape thresholds -- is scored against a per-card calibrated ridge,
    145.8 on the A100 against 162.8 on the H200. Two cards sharing one directory
    would have the second silently report the first's treads against its own
    ridge, which is a hybrid of two machines and is exactly the defect that put
    a stale H200 band into seven published A100 reports.
    """
    swept = {
        "model": args.model, "dtype": args.dtype, "r": args.r_max,
        "reps": args.reps, "iters": args.iters, "warmup": args.warmup,
        "trials": args.trials, "l2flush": not args.no_l2_flush,
        "budget": args.cell_budget_ms, "seed": args.seed, "g": args.group_m,
        "n": args.block_n, "k": args.block_k, "stages": args.num_stages,
        "warps": args.num_warps, "subject": SUBJECT_BLOCK_M,
        "reference": REFERENCE_BLOCK_M,
    }
    return PV.run_id(card=card, **swept)


# --------------------------------------------------------------------------
# Self test: plant four worlds, check the gates tell them apart.
# --------------------------------------------------------------------------

#: What the planted BLOCK_M=256 reference achieves, as a fraction of the planted
#: card's calibrated ceiling. Under 1 SO THAT V1 HAS A PASS BRANCH: the gate's
#: window is `[REFERENCE_LEVEL_FLOOR, 100%]` and a reference generated at the
#: world's own `rho` implies exactly 100%, which is the wall, so the seed decided
#: the verdict. 0.97 also happens to be true of every real grouped GEMM.
REFERENCE_KERNEL_EFFICIENCY = 0.97

#: The clock the planted roof was measured at, and the clock a throttled tread
#: was timed at. The ratio is applied to the COMPUTE branch of those treads, not
#: to their traffic, because that is what a lower SM clock does.
SELF_TEST_REFERENCE_CLOCK_MHZ = 1980.0
SELF_TEST_LOW_CLOCK_MHZ = 1000.0


def planted_ladder(treads: int, *, alpha: float, rho: float, block_m: int,
                   b: int, load_ms: float, overhead_ms: float,
                   noise: float = 0.0, seed: int = 0) -> list[tuple[int, float]]:
    """A ladder generated FROM the law, so the gates have a known answer.

    `t(n) = D + max(L (1 + alpha(n-1)), C n)` with `C = L alpha / ratio`, which
    is the two-line model with `B / C` set to exactly what `branch_ratio` says.

    KEPT FOR THE ARITHMETIC, NOT FOR THE SELF TEST. `margin_of` and
    `prefix_depth` are checked against it directly, where a hand-built ladder is
    the right fixture. The self test no longer uses it: see `planted_samples`.
    """
    rng = random.Random(seed)
    ratio = branch_ratio(block_m, b, alpha, rho)
    c = load_ms * alpha / ratio
    out = []
    for n in range(1, treads + 1):
        ms = overhead_ms + max(load_ms * (1.0 + alpha * (n - 1)), c * n)
        out.append((n, ms * (math.exp(rng.gauss(0.0, noise)) if noise else 1.0)))
    return out


def planted_samples(cfg, *, alpha: float, rho: float, bandwidth_gbps: float,
                    b: int = 2, reps: int = 5, noise: float = 0.0,
                    seed: int = 0, r_max: int = 1024,
                    low_clock_treads: tuple[int, ...] = ()) -> list[Sample]:
    """A whole replicated run, as the CSV rows a pod would have written.

    GENERATED THROUGH `SWEEP.model_ms`, NOT BY HAND, and that is not a
    convenience. `compute_reference` level-checks its candidate against the
    roof, against one full weight read and against the smaller ladders, so a
    hand-rolled ladder with a plausible-looking slope is REFUSED for being
    physically inconsistent with the card it claims to be from. Both settings
    come from one model at one `(alpha, rho, bandwidth)`, which keeps the
    planted world self-consistent: its implied ceiling is `rho x bandwidth`.

    `instrument` is left EMPTY on every planted row, deliberately. These rows
    were not measured, and stamping them with `TIMING_BASIS` would make a
    planted world indistinguishable from a pod's in the one column that says
    which instrument produced a number.

    THE REFERENCE LADDER IS PLANTED A FEW PER CENT UNDER THE CEILING, and that
    is what makes V1 registrable at all. `reference_level` accepts a compute
    reference whose implied rate lands in `[REFERENCE_LEVEL_FLOOR, 100%]` of the
    card's calibrated ceiling. `model_ms` computes its compute branch as
    `flops / (ridge x bandwidth)`, so a reference generated at the world's own
    `rho` implies EXACTLY `rho x bandwidth`, which is the gate's upper wall: at
    any planted spread the seed decides which side of 100% it lands on, and at
    the default spread seeds 0, 3 and 5 gave V1 FAIL while 1, 2 and 4 gave PASS.
    Registering that is registering a coin flip, and leaving it unregistered
    left the one gate the 43.6x reference exists to catch with no PASS branch
    planted anywhere. `REFERENCE_KERNEL_EFFICIENCY` is the fix and it is also
    the physics: a Triton grouped GEMM does not achieve the calibrated cuBLAS
    ceiling, and a fixture in which it does is a fixture no pod can produce.

    `low_clock_treads` marks the named SUBJECT treads as having been timed below
    the clock the roof was measured at, which is how the throttled world is
    planted. Every other row carries `clock_level_ok=True`, because a world
    where the verdict is None everywhere excludes nothing and would not
    exercise the exclusion at all.

    A THROTTLED TREAD IS SLOWER, NOT MERELY LABELLED. Until 2026-09-02 this
    function stamped the clock columns on the low rows and computed their `ms`
    identically to every other row, so the fixture planted the LABEL and not the
    physics: no gate could tell a throttled ladder from a clean one, and the
    claim in `ladder_treads`' docstring -- that a ladder which lost treads to a
    hot box must not look like a ladder that never had them -- was demonstrated
    by nothing. A card at `SELF_TEST_LOW_CLOCK_MHZ` runs its COMPUTE branch
    slower in exactly that proportion and its traffic at the same DRAM
    bandwidth, which is a lower `ridge` and nothing else, so those rows are
    generated at `rho x low / reference` and the shape of the ladder bends where
    the throttling starts. That bend is what V2 and V3 would score if the
    exclusion did not reach them.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    throttled = SELF_TEST_LOW_CLOCK_MHZ / SELF_TEST_REFERENCE_CLOCK_MHZ
    for rep in range(1, reps + 1):
        for block_m in (REFERENCE_BLOCK_M, SUBJECT_BLOCK_M):
            for rows in ladder_rows(cfg, block_m, r_max):
                tiles = rows // block_m
                low = (block_m == SUBJECT_BLOCK_M
                       and tiles in low_clock_treads)
                # ONE KNOB CARRIES BOTH EFFECTS because both are the same
                # physics: `ridge` is `peak / bandwidth`, so scaling it scales
                # the compute branch and leaves the traffic branch alone. The
                # reference kernel runs under the calibrated peak; a throttled
                # tread runs under its own card's peak.
                scale = REFERENCE_KERNEL_EFFICIENCY if block_m == REFERENCE_BLOCK_M else 1.0
                if low:
                    scale *= throttled
                ms = SWEEP.model_ms(cfg, rows, block_m, alpha=alpha,
                                    ridge=rho * scale,
                                    bandwidth_gbps=bandwidth_gbps, b=b,
                                    overhead_ms=0.05)
                if noise:
                    ms *= math.exp(rng.gauss(0.0, noise))
                out.append(Sample(block_m, tiles, rows,
                                  SWEEP.tokens_for_rows(cfg, rows), rep, ms, ms,
                                  0.0, 0, clock_level_ok=not low,
                                  clock_drift_ok=True,
                                  sm_clock_load_mhz=(
                                      SELF_TEST_LOW_CLOCK_MHZ if low
                                      else SELF_TEST_REFERENCE_CLOCK_MHZ)))
    return out


@dataclass(frozen=True)
class PlantedWorld:
    """A planted `(alpha, rho)`, what it demonstrates, and what must come back.

    `expect` maps a gate TAG (`C1`, `V1`, ...) to `True` / `False` / `None`, the
    three values `Gate.passed` takes. A tag absent from `expect` is deliberately
    unregistered and is not asserted; a tag NAMED in `expect` that the report
    does not contain is itself a mismatch, because a registration that silently
    matches nothing is the check-that-examined-nothing shape one level up.

    `exit_code` IS REGISTERED TOO, AND IT IS THE POINT OF THIS FIXTURE. Per-gate
    verdicts are not what a session driver sees; it sees one integer, and
    `scripts/h200_gaps_session.sh` turns anything outside this arm's done list
    into RETRY with "nothing on the page may be quoted". A fixture that scored
    only the gates let all four worlds pass while every one of them would have
    exited INVALID on a pod -- which is what happened when `C3` was relabelled
    to a VALIDITY gate, and it is what this field catches. It is
    `moe.bench.exit_codes.classify` over the SAME gates, so the world registers
    the number the driver reads and not a paraphrase of it.
    """

    name: str
    alpha: float
    rho: float
    why: str
    expect: dict[str, bool | None]
    #: The `moe.bench.exit_codes` value `classify` must return for this world.
    exit_code: int = exit_codes.DONE
    #: Subject treads planted as having been timed below the roof's clock.
    low_clock_treads: tuple[int, ...] = ()

    def check(self, gates: list[Gate]) -> list[str]:
        """The registrations that did not come back. Empty is a pass."""
        got = {g.tag: g.passed for g in gates}
        word = {True: "PASS", False: "FAIL", None: "UNKNOWN"}
        bad = []
        for tag, want in sorted(self.expect.items()):
            if tag not in got:
                bad.append(f"{tag}: registered {word[want]}, but the report "
                           f"has no gate {tag}")
            elif got[tag] is not want:
                bad.append(f"{tag}: registered {word[want]}, got {word[got[tag]]}")
        rc = exit_codes.classify(g.scored() for g in gates)
        if rc != self.exit_code:
            bad.append(f"exit code: registered {exit_codes.describe(self.exit_code)}, "
                       f"got {exit_codes.describe(rc)}")
        return bad


class SelfTestRefused(RuntimeError):
    """The planted design cannot exercise what the registrations are about.

    A REFUSAL, NOT A FAILURE, and the difference is the exit code: `main`
    returns `exit_codes.REFUSED`, which the driver reads as "nothing was
    measured, this cost nothing, read the first line and fix it". Raised as a
    `RuntimeError` rather than a `SystemExit` so that a caller inside the test
    suite can name it, and so it cannot exit 1 -- CLAIM_FAIL -- which would tell
    a driver the worlds were run and a claim was refuted.
    """


#: The bandwidth every planted world is generated at. The A100's calibrated
#: figure, so `rho x bandwidth` is a ceiling a real card could have and the
#: level check has something physical to accept or refuse.
SELF_TEST_BANDWIDTH = 1799.4


#: The worlds, the verdicts the REAL gates must return in each, and the exit
#: code the driver would read.
SELF_TEST_WORLDS: tuple[PlantedWorld, ...] = (
    PlantedWorld(
        "escape-up", 0.95, 175.0,
        "alpha x rho = 166.2, over the 147.2 the escape needs. rho = 175 Op/B "
        "is a card NEITHER OF THIS STUDY'S HAS -- the A100 calibrates at 145.8 "
        "and the H200 at 162.8 -- which is the finding stated as a fixture: to "
        "plant a world where the depth claim is reachable, hardware has to be "
        "invented. The one world that exits DONE, so DONE is planted too",
        # V1 IS REGISTERED IN EVERY WORLD THAT HAS A REFERENCE, and it can be
        # because `planted_samples` now generates the reference ladder at
        # `REFERENCE_KERNEL_EFFICIENCY` of the planted ceiling. Generated at the
        # ceiling itself, which is `reference_level`'s own upper wall, the seed
        # decided the verdict -- 0, 3 and 5 gave FAIL and 1, 2 and 4 gave PASS
        # -- so this gate, the one the 43.6x reference exists to catch, had its
        # PASS branch planted in no world at all.
        {"V0": True, "V1": True, "V2": True, "V3": True, "V4": True,
         "V5": True, "C1": True, "C2": True, "V6": True},
        exit_code=exit_codes.DONE),
    PlantedWorld(
        "straddle", 0.88, 145.813,
        "where BOTH cards actually sit at BLOCK_M=128: alpha x rho = 128.3 "
        "against the 128 that makes the two branches one line. The depth must "
        "NOT be found, the tolerance margin must not be confident either, and "
        "THE ARM MUST STILL EXIT ON A RESULT rather than on a void page",
        # C1 IS UNKNOWN HERE AND THAT IS THE REGISTRATION. B/C comes out inside
        # PARALLEL_BRANCH_TOLERANCE, so `fit_ladder` returns
        # `undecided_parallel_branch` -- it LOOKED and declined -- and the
        # identity `B/C = ridge/ai_cap` says the cap sits ON the ridge, which is
        # the roofline arm's question and not this one's. Registering FAIL here
        # would publish "no depth" from a fit that named none, which is the
        # defect this world pins.
        #
        # V6 PASSES HERE, VACUOUSLY, AND THE EXIT CODE IS WHY IT MATTERS. This
        # is the outcome every published H200 BLOCK_M=128 arm returns. A V6 that
        # scored UNKNOWN over a fit that published no alpha made the world
        # INVALID (3), which the session driver reads as RETRY, so the twelve
        # metered minutes were RETRY by construction on both cards. There is no
        # alpha in this report, so there is nothing for a VALIDITY gate to
        # withhold; C1's UNKNOWN carries the verdict and CLAIM_FAIL is a result.
        {"V0": True, "V1": True, "V2": True, "V3": True, "V4": True,
         "V5": True, "C1": None, "C2": False, "V6": True},
        exit_code=exit_codes.CLAIM_FAIL),
    PlantedWorld(
        "escape-down", 0.30, 320.0,
        "alpha x rho = 96, under the 108.8 cap, and rho = 320 is 2.2x the "
        "A100's calibrated 145.8. Reachable by the gates and by NO CARD IN "
        "THIS STUDY, which is the whole finding: the gates can see a depth the "
        "hardware cannot produce",
        # V1 UNKNOWN, deterministically: at rho = 320 the reference ladder is
        # memory bound at every tread, so no compute reference qualifies at all.
        # That is a property of the world, not of the seed.
        #
        # AND THAT IS WHY THIS WORLD IS REGISTERED INVALID. A run with no
        # compute reference has no instrument -- membership falls back to a
        # split search, which invents an alpha rather than declining to -- so
        # the C1 PASS below is a statement about the gate's arithmetic and is
        # NOT a quotable result. Registering DONE here would be registering that
        # a page with no ruler on it may be read.
        #
        # C2 IS NOT REGISTERED: with no compute reference the bootstrap has no
        # `B/C` to resample and the verdict alternates between UNKNOWN and FAIL
        # with the seed. A registration on a coin flip is worse than none, and
        # C2's three branches are planted in the other worlds.
        {"V0": True, "V1": None, "V2": True, "V3": True, "V4": True,
         "V5": True, "C1": True, "V6": None},
        exit_code=exit_codes.INVALID),
    PlantedWorld(
        "low-clock", 0.95, 175.0,
        "the escape-up world again, with six of its eight subject treads TIMED "
        f"AT {SELF_TEST_LOW_CLOCK_MHZ:.0f} MHz against a roof measured at "
        f"{SELF_TEST_REFERENCE_CLOCK_MHZ:.0f}. Those treads are not merely "
        "labelled slow, they ARE slow: their compute branch is generated at the "
        "throttled clock and their traffic is not, so the ladder BENDS where "
        "the throttling starts. `ladder_treads` excludes them and what remains "
        "is under the floor an alpha may decide on. THE VERDICT MUST BE "
        "UNDECIDED AND NOT 'no depth': the card is what that ladder measured, "
        "and a run on a quiet card is the next step. This world is why the "
        "clock columns travel from the CSV into the fit AND INTO EVERY GATE",
        # V3 IS UNKNOWN HERE, AND THAT IS THE FIX IT PINS. Scored over all eight
        # treads this ladder is not convex -- the throttle onset makes the slope
        # rise 1.9 -> 3.0 and fall back to 2.2, a drop that clears both the floor
        # and the sigma bar at every seed tried -- so V3 FAILED, which is
        # INVALID, for exactly the reason the exclusion exists to discount.
        # Scored over the two treads that survive it there is no second
        # difference and the honest answer is UNKNOWN. Both are non-PASS and
        # both make the arm INVALID, but they say opposite things about what to
        # do next, and only one of them is true.
        #
        # V5 AND C2 ARE NOT REGISTERED HERE. Two surviving treads make the
        # across-repeat spread a median of two draws at a planted 1.82% against
        # a 2% ceiling, so V5 straddles its own bar with the seed (5 of 40
        # seeds FAIL); C2 follows the same two treads. Both are registered in
        # the worlds where they are deterministic, and a registration on a coin
        # flip is what the V1 comment above exists to warn against.
        {"V0": True, "V1": True, "V2": True, "V3": None, "V4": True,
         "C1": None, "V6": None},
        exit_code=exit_codes.INVALID,
        low_clock_treads=(3, 4, 5, 6, 7, 8)),
)


def self_test(b: int = 2, *, noise: float = 0.0, seed: int = 0, draws: int = 400
              ) -> tuple[list[str], list[Gate]]:
    """Four worlds, routed through the code the pod runs, with registered verdicts.

    WHAT THIS USED TO DO AND WHY IT PROVED NOTHING ABOUT THE POD PATH. It built
    a bare list of `(tread, ms)` pairs from `planted_ladder`, computed the
    membership count BY HAND from the planted compute slope, and called
    `margin_of` on it. `SWEEP.compute_reference`, `SWEEP.fit_ladder`,
    `gate_reference_level`, `gate_monotone`, `gate_convex`,
    `_gate_replication`, `gate_depth` and `_gate_law` -- every function the
    twelve metered minutes actually run, and the path that produced the 43.6x
    reference the level check exists to catch -- were never called. A
    regression in any of them passed this self test.

    WHAT IT DOES NOW. Generates a full replicated run per world through
    `planted_samples` and hands it to `analyse_run`, which is the one function
    `main` calls after the sweep. The gates that come back are the pod's gates,
    scored on data whose answer is known, and `PlantedWorld.expect` registers
    that answer per world so a wrong verdict is a failure rather than a line to
    eyeball. The `S <world>` gates below carry the comparison; the world's own
    gates are printed under it so a mismatch names itself.

    AND IT SCORES THE EXIT CODE EACH WORLD WOULD RETURN ON A POD. Per-gate
    verdicts are not what the session driver sees; it sees one integer. For one
    commit every world here reported four S gates PASS while all four would have
    exited INVALID on a pod, because relabelling `C3` to a VALIDITY gate turned
    the arm's own expected outcome into a void page and nothing in this fixture
    looked at the number. `PlantedWorld.exit_code` is that number, computed by
    `moe.bench.exit_codes.classify` over the same gate objects, and a world that
    returns a different one is a mismatch like any other.
    """
    if noise <= 0:
        raise SelfTestRefused(
            "--plant-noise 0 with --self-test: a noiseless world cannot "
            "exercise a single one of this file's noise-floored thresholds. "
            "The membership margin is max(2%, 3 x the measured spread), the "
            "convexity floor is propagated from it, and C2's margin needs an "
            "across-repeat spread before it has a sigma at all -- at 0.00% "
            "every one of them is evaluated at the one value no pod produces, "
            "and C2 comes back UNKNOWN in every world for want of a noise "
            f"model. Use the default {PUBLISHED_LADDER_SPREAD}, the top of the "
            "published H200 range, or a spread you have measured.")
    if noise > MAX_REPLICATE_SPREAD:
        raise SelfTestRefused(
            f"--plant-noise {noise} with --self-test: above "
            f"{MAX_REPLICATE_SPREAD:.0%} gate V5 WITHDRAWS its verdict by "
            "design -- an across-repeat spread that large makes every margin on "
            "the page unreadable -- so every world would report V5 FAIL and the "
            "registrations below, which are for a run whose noise is inside what "
            "a replicated run is allowed to have, would all mismatch. That is V5 "
            "working, not the worlds failing, and this refuses rather than "
            "printing four mismatches that mean the opposite of what they say. "
            "Plant at or below the ceiling, or fix the card.")
    out = ["", "## Self test: planted worlds, through the pod's own analysis", "",
           f"planted at a per-tread spread of {noise:.2%}; every world is "
           f"generated from SWEEP.model_ms at {SELF_TEST_BANDWIDTH:.0f} GB/s "
           "and routed through analyse_run, which is the function main() calls "
           "after the sweep"]
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    gates: list[Gate] = []
    for world in SELF_TEST_WORLDS:
        samples = planted_samples(cfg, alpha=world.alpha, rho=world.rho,
                                  bandwidth_gbps=SELF_TEST_BANDWIDTH, b=b,
                                  noise=noise, seed=seed,
                                  low_clock_treads=world.low_clock_treads)
        ceiling = world.rho * SELF_TEST_BANDWIDTH * 1e9 / 1e12
        _, world_gates, payload = analyse_run(
            samples, cfg, b, ceiling_tflops=ceiling,
            ceiling_source=f"planted world {world.name}",
            compiles={SUBJECT_BLOCK_M: 1, REFERENCE_BLOCK_M: 1},
            executed={SUBJECT_BLOCK_M: len(samples), REFERENCE_BLOCK_M: len(samples)},
            ridge=world.rho, bandwidth_gbps=SELF_TEST_BANDWIDTH, seed=seed,
            draws=draws)
        law = depth_verdict(SUBJECT_BLOCK_M, b, world.alpha, world.rho)
        bad = world.check(world_gates)
        rc = exit_codes.classify(g.scored() for g in world_gates)
        out += ["",
                f"### {world.name}  alpha {world.alpha} rho {world.rho}",
                f"    {world.why}",
                f"    law says: {law.line()}",
                "    real gates: "
                + ", ".join(f"{g.tag}="
                            + {True: 'PASS', False: 'FAIL', None: 'UNKNOWN'}[g.passed]
                            for g in world_gates),
                f"    memory treads {payload['memory_points']}, ladder outcome "
                f"{payload['ladder_outcome']}",
                # THE INTEGER THE DRIVER WOULD READ, printed beside the verdicts
                # it came from. The gates are for a human; this is the contract.
                f"    on a pod this world exits {exit_codes.describe(rc)} "
                f"(registered {exit_codes.describe(world.exit_code)})"]
        out += [f"    MISMATCH {line}" for line in bad]
        gates.append(Gate(
            VALIDITY, f"S {world.name}",
            f"the {world.name} world returns its registered verdicts AND its "
            "registered exit code through analyse_run",
            f"{len(world.expect)} registered gate(s) match and "
            f"classify() returns {exit_codes.CODE_NAMES[world.exit_code]}",
            not bad,
            (f"every registered verdict matched; classify() returned "
             f"{exit_codes.CODE_NAMES[rc]}") if not bad else "; ".join(bad),
            "the gates themselves: a self test that exercises a different code "
            "path from the pod cannot certify the pod's, and one that scores "
            "gates but never the exit code cannot see an arm that is RETRY by "
            "construction",
            lines=[f"registered: {world.expect}",
                   f"registered exit: {exit_codes.describe(world.exit_code)}"]))
    return out, gates


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true",
                    help="run the gates over every published BLOCK_M=128 "
                         "ladder and print the feasibility arithmetic. No GPU, "
                         "and the evidence for the docstring")
    ap.add_argument("--published", type=Path,
                    default=ROOT / "results" / "published",
                    help="where the published arms live")
    ap.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR,
                    help="calibration yaml directory; the ceiling every "
                         "reference level is scored against")
    ap.add_argument("--self-test", action="store_true",
                    help="plant four worlds from the law and check the gates "
                         "tell them apart, off GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the predictions and the cost, then stop")
    ap.add_argument("--model", default="qwen2-57b-a14b",
                    choices=sorted(MODEL_CONFIGS),
                    help="qwen2 by default: its BLOCK_M=128 ladder is the one "
                         "the A100 surface quotes, so a re-run is comparable")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="not fp8: halving the weight bytes DOUBLES the AI cap, "
                         "which moves B/C the wrong way and off the corpus")
    ap.add_argument("--r-max", type=int, default=1024,
                    help="largest rows per expert; 1024 is 8 treads at "
                         "BLOCK_M=128 and 4 at 256. Raising it cannot change "
                         "WHICH treads are memory bound (see P5), only how many "
                         "exist to be measured")
    ap.add_argument("--reps", type=int, default=7,
                    help="round-robin passes over the ladder. The published "
                         "ladders had ONE, which is why a 1.24%% inversion could "
                         "not be told from noise")
    ap.add_argument("--group-m", type=int, default=SWEEP.FIXED["GROUP_SIZE_M"],
                    help="the swizzle width. Moves alpha, and so moves B/C -- "
                         "by 6.1%% across the published corpus, against the "
                         "16.1%% the escape needs")
    ap.add_argument("--block-n", type=int, default=SWEEP.FIXED["BLOCK_SIZE_N"],
                    help="N tile. 256 is the arm whose BLOCK_M=256 reference is "
                         "43.6x too slow on the A100; V1 refuses it")
    ap.add_argument("--block-k", type=int, default=SWEEP.FIXED["BLOCK_SIZE_K"],
                    help="K tile. Degrading it lowers the achieved compute rate, "
                         "which lowers B/C -- the wrong direction for escape up "
                         "and the wrong direction for escape down as well")
    ap.add_argument("--num-stages", type=int, default=SWEEP.FIXED["num_stages"])
    ap.add_argument("--num-warps", type=int, default=SWEEP.FIXED["num_warps"])
    ap.add_argument("--iters", type=int, default=50,
                    help="RETIRED as a timing knob and kept in the run id. "
                         "moe.bench.timing.time_kernel sizes each cell's "
                         "iteration count from --cell-budget-ms and the "
                         "warmup's own queue-deep per-call time; the count a "
                         "cell really used is a column in cells.csv")
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. UNITS CHANGED with the instrument: "
                         "a count of 20 and a duration of 20 ms are both "
                         "accepted by argparse and only one of them is a warmup")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per timing; the percentiles are "
                         "over iters x trials samples")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do NOT evict L2 between timed iterations. Off by "
                         "default because the compute reference and the card's "
                         "ceiling were both measured flushed and a warm-L2 "
                         "tread is not comparable with either. Recorded per row")
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="target measured KERNEL time per trial; the "
                         "instrument sizes its iteration count from it")
    ap.add_argument("--plant-noise", type=float, default=PUBLISHED_LADDER_SPREAD,
                    metavar="SIGMA",
                    help="lognormal sigma planted on every --self-test tread, "
                         "and the noise assumption every MDE in the plan is "
                         f"derived from. Defaults to {PUBLISHED_LADDER_SPREAD}, "
                         "the top of the published H200 across-repeat spread; "
                         "0 plants a world no pod produces")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS,
                    help="bootstrap resamples behind every reported margin")
    ap.add_argument("--alpha", type=float, default=SWEEP.ALPHA,
                    help="only used to COST the run in --dry-run")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="only used to cost the run; 0 reads the attached "
                         "device's calibration")
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--card", default="",
                    help="card slug the run id is built from. Read from the "
                         "attached device by default and REFUSED if it "
                         "contradicts one; its only real use is printing a "
                         "pod's exact path from a laptop dry run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--require-git-visible", action="store_true",
                    help="refuse to run when the output path is git-ignored")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="RETIRED 2026-09-02 and accepted so old driver lines "
                         f"still parse. A failed CLAIM gate now always exits "
                         f"{exit_codes.CLAIM_FAIL} CLAIM_FAIL, which the ledger "
                         "reads as a finished result rather than a retry; "
                         "folding it into 0 made the log disagree with the "
                         "process, and --audit was the live instance")
    return ap


def _cost_inputs(args):
    """Ridge and bandwidth used ONLY to price the run, named with their source."""
    ridge, ridge_src = args.ridge, "given on the command line"
    bandwidth, bw_src = args.bandwidth_gbps, "given on the command line"
    if not ridge or not bandwidth:
        try:
            from moe.bench.roofline import load_measured
            hw = load_measured()
        except Exception as exc:                        # noqa: BLE001
            hw, exc_note = None, str(exc)
        else:
            exc_note = ""
        if hw is not None:
            if not ridge:
                ridge = hw.ridge_point(args.dtype)
                ridge_src = f"this machine's calibration ({hw.name})"
            if not bandwidth:
                bandwidth = hw.bandwidth_bytes_s / 1e9
                bw_src = f"this machine's calibration ({hw.name})"
        else:
            # A cost estimate is not a measurement, so a hypothesis is allowed
            # here where it would be refused in a verdict -- and it is labelled
            # so it can never be mistaken for one.
            if not ridge:
                ridge, ridge_src = 145.813, (
                    "HYPOTHESIS: the A100 calibration in this repo, no device "
                    "attached" + (f" ({exc_note})" if exc_note else ""))
            if not bandwidth:
                bandwidth, bw_src = 1799.4, (
                    "HYPOTHESIS: the A100 calibration in this repo, no device "
                    "attached")
    return ridge, ridge_src, bandwidth, bw_src


def _exit_code(gates: list[Gate]) -> int:
    """The shared table's verdict over the scored gates, and nothing folded.

    `moe.bench.exit_codes.classify` decides it, over the SAME gate objects that
    printed the RESULT lines, so `classify_text` on the log recomputes the code
    the process returned and a disagreement is itself a defect.

    NOTHING IS FOLDED INTO DONE ANY MORE, and `--fail-on-gate` is why this is a
    paragraph rather than a branch. Until 2026-09-02 a CLAIM_FAIL was described
    in words and RETURNED AS 0 unless the flag was passed, and `--audit` is a
    live instance: over the published corpus C1 FAILS, the log carried
    `RESULT: CLAIM C1 FAIL` and the process said DONE. That is exactly the
    log-versus-exit-code split this module was adopted to end, printed by the
    docstring that claimed it could not happen here. The masking was obsolete
    the moment the shared table landed: CLAIM_FAIL (1) is in `FINISHED_CODES`
    and `ledger_state(1)` is "CLAIM_FAIL", so 1 already tells the driver "this
    is a result, do not retry it", which is the whole thing 0 was protecting.
    The flag is accepted by the parser, ignored, and NOT PASSED IN HERE any
    more, so no future edit can reach for it; its help text says it is retired.
    A VALIDITY failure is INVALID either way: nothing on the page may be quoted
    after one.
    """
    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    if rc == exit_codes.CLAIM_FAIL:
        print("         a CLAIM gate that did not pass is a RESULT and the "
              "arm is FINISHED. This arm's")
        print("         registered expectation is that C1 fails; "
              "`--fail-on-gate` is retired and ignored.")
    return rc


class MdeNotStateable(RuntimeError):
    """`mde_lines` was asked for a power calculation its inputs cannot carry.

    A NAMED REFUSAL AND NOT A `SystemExit`, and the difference is an exit code
    that used to be a lie. This raised `SystemExit(<str>)` until 2026-09-02,
    which exits the process with 1 -- CLAIM_FAIL, "a pre-registered claim was
    refuted" -- from a run that had refuted nothing. `--dry-run --plant-noise 0`
    returned 1 after measuring nothing, and worse, `main` builds
    `payload["mde"]` AFTER both ladders are timed and BEFORE `report.json` is
    written, so a pod run at `--reps 1` spent the twelve metered minutes, exited
    1, and wrote no report at all. Neither number reached
    `moe.bench.exit_codes.classify`; both were chosen by the interpreter.

    Raised as a `RuntimeError` so a caller can name it. Both call sites do, and
    they answer it differently ON PURPOSE: in the plan the MDE is one section of
    a document and the sibling's degradation is right (`tile_cap_test` prints
    "MINIMUM DETECTABLE EFFECT: not stateable" and carries on), while after the
    sweep the report must still be written and the exit code must still come
    from the gates that were scored. Under no arrangement may the absence of a
    power calculation decide the arm's verdict.
    """


def mde_lines(*, spread: float, reps: int, treads: int) -> list[str]:
    """The minimum detectable effect of this arm's gates, before it is paid for.

    HOEFLER & BELLI RULE 12. Not one arm in this study stated an MDE, and this
    one carries two thresholds that are multiples of a measured spread -- V2's
    `MONOTONE_SIGMA` inversion test and C2's `MARGIN_SIGMA` tolerance margin --
    so what they can see is entirely a function of the spread and the repeat
    count, and neither was ever written down.

    ONE ASSUMPTION, `--plant-noise`, and everything derived from it.

      V2 scores an inversion, a difference of two per-tread MEDIANS over `reps`
      repeats each, so its standard error is `spread * sqrt(2 / reps)` and it
      fires at `MONOTONE_SIGMA` of that. One-sided, because `t(n)` cannot fall
      under either branch.
      C2 scores `|B/C - 1| - TOLERANCE` at `MARGIN_SIGMA` bootstrap sd. The sd
      is a property of the fitted ladder and is not knowable before the run;
      what IS knowable is the sd the ladder's own points contribute, which is
      the slope's relative sd, and it is stated so the run can be sized.
      C1 is a COUNT and has no MDE: five treads either stand above the compute
      branch or they do not. Said out loud rather than left as a gap, because a
      missing line reads as an oversight.
    """
    if spread <= 0:
        raise MdeNotStateable(
            f"--plant-noise {spread}: an MDE is a multiple of a standard "
            "deviation, and at a spread of zero every effect is detectable, "
            "which is a statement about the planted world and not about any "
            "pod. State the spread you believe the card has.")
    if reps < 2 or treads < 2:
        raise MdeNotStateable(
            f"--reps {reps} over {treads} treads: an across-repeat spread needs "
            "two repeats and a slope needs two treads, so no MDE is stateable.")
    se_median = spread * math.sqrt(2.0 / reps)
    inversion = MONOTONE_SIGMA * se_median
    slope_rel = spread * math.sqrt(12.0 / (treads * (treads ** 2 - 1.0)))
    return [
        "",
        "## Minimum detectable effect, from ONE stated assumption",
        "",
        f"  assumed per-tread relative spread {spread:.2%} (--plant-noise; the "
        "published H200 ladders run 0.76-1.82% on one pass and the A100 ones "
        f"0.48-0.61%, so the default {PUBLISHED_LADDER_SPREAD:.2%} is the top "
        "of the H200 range, which is the end a gate has to survive)",
        f"  V2 inversions   a per-tread median over {reps} repeats carries "
        f"{se_median:.3%}; at {MONOTONE_SIGMA:.0f} sigma this run can see an "
        f"inversion of {inversion:.2%} or more. The published A100 ladder's "
        "1.24% inversion is "
        + ("INSIDE that and would be called noise here"
           if 0.0124 < inversion else
           "OUTSIDE that and would be called a mechanism here"),
        f"  C2 margin       the slope's relative sd over {treads} treads is "
        f"{slope_rel:.3%} (equally spaced OLS), so B/C carries at least that "
        f"and the {MARGIN_SIGMA:.0f}-sigma bar needs the margin to clear "
        f"{MARGIN_SIGMA * slope_rel:.3%} of B/C on top of the "
        f"{TOLERANCE:.0%} tolerance. The published A100 fit clears the "
        "tolerance by 0.007%, which is inside this before the bootstrap is run",
        f"  C1 depth        a COUNT, not an effect: {TARGET_TREADS} treads "
        "either stand above the compute branch or they do not, so it has no "
        "MDE. What noise buys it is membership churn at the boundary, which is "
        f"why the membership margin is max({SWEEP.MEMORY_BRANCH_MARGIN:.0%}, "
        "3 x the measured spread) rather than a constant",
    ]


def mde_block(*, spread: float, reps: int, treads: int) -> tuple[list[str], str]:
    """The MDE section, or the one line that says why this run has none.

    ONE FUNCTION SO BOTH CALL SITES DEGRADE THE SAME WAY, and so the pod path's
    degradation can be exercised off GPU. `main` needs the MDE twice -- in the
    plan, where it is a section of a document, and after the sweep, where it is
    a key in `report.json` -- and for one commit only the first of those had any
    handling at all: the second called `mde_lines` bare, between the last timing
    and the first `write_text`, so a `--reps 1` pod run paid for both ladders,
    raised `SystemExit`, wrote no report and returned 1 = CLAIM_FAIL from a run
    that had refuted nothing.

    Returns `(lines, reason)`. `reason` is empty when the MDE was stateable and
    is the refusal's own words otherwise, so the caller can put it in the report
    as a named field rather than leaving a reader to parse the prose.
    """
    try:
        return mde_lines(spread=spread, reps=reps, treads=treads), ""
    except MdeNotStateable as exc:
        return ["", f"MINIMUM DETECTABLE EFFECT: not stateable. {exc}"], str(exc)


def _instrument_refusal(exc: BaseException) -> bool:
    """True when `exc` is `moe.bench.timing.TimingRefused`, asked lazily.

    BY CLASS AND NOT BY NAME, because a name test would also catch a class some
    other library happened to spell the same way. LAZILY, because
    `moe.bench.timing` imports torch at module scope and this file is documented
    to run `--audit`, `--self-test` and `--dry-run` on a laptop with no torch at
    all: the import that answers the question must not be the thing that breaks
    those three. On such a box the import fails and the answer is False, which
    is the right answer anyway -- a run that never reached the instrument cannot
    have been refused by it.
    """
    try:
        from moe.bench.timing import TimingRefused
    except Exception:                                   # noqa: BLE001
        return False
    return isinstance(exc, TimingRefused)


def main(argv=None) -> int:
    """`_main` with the escapes that were exiting ONE, which is CLAIM_FAIL.

    AN UNPLANNED CRASH IS ERROR, WHICH IS THE ONLY RETRYABLE CODE. Left to
    propagate, an unexpected exception exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it is in
    FINISHED_CODES, the session driver records it, and it is never retried. This
    arm pays for two full ladders before it writes anything, and the file's own
    header already records one way it lost that booking -- a `SystemExit`
    between the last timing and the first `write_text`. An OOM in the same gap
    was the other way, and it was filed as this experiment's registered answer
    to "can BLOCK_M=128 show clean memory-bound treads". ERROR (4) is outside
    FINISHED_CODES precisely so the driver can tell "the apparatus broke" from
    "the claim did not hold". The traceback is printed first and not swallowed,
    because a code without one tells an operator nothing about what to fix.

    A STRING `SystemExit` IS A REFUSAL, and this file already fought that half
    of the defect: fourteen `raise SystemExit(<str>)` became
    `RefusedBeforeMeasuring`, which carries `code = exit_codes.REFUSED`. The
    branch is kept anyway, because a refusal added later, or one raised by a
    library this imports, must not land as a refuted claim just for spelling
    itself the old way. A `SystemExit` carrying an INT already says what it
    means and is re-raised untouched, argparse's included.

    A `TimingRefused` IS A REFUSAL TOO, and it is the one the per-cell handler
    in `measure_setting` now lets past it. It is the same fact for every tread,
    so nothing was measured and nothing was spent, which is REFUSED and not
    ERROR.
    """
    try:
        return _main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = (exc.code if exc.code.startswith("REFUS")
                   else f"REFUSED: {exc.code}")
            print(msg, file=sys.stderr)
            return exit_codes.REFUSED
        raise
    except Exception as exc:                            # noqa: BLE001
        if _instrument_refusal(exc):
            print(f"REFUSED: {exc}", file=sys.stderr)
            return exit_codes.REFUSED
        traceback.print_exc()
        print("ERROR: bm128_depth crashed before it could reach a verdict. "
              "This is the apparatus failing, not a claim failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, the cells already timed are "
              "in the run directory the header printed, and the arm may be "
              "re-run.", file=sys.stderr)
        return exit_codes.ERROR


def _main(argv=None) -> int:
    """The one place an exit code is chosen for a run that reached a verdict.

    Every return is a member of `moe.bench.exit_codes`'s table: REFUSED (2)
    before anything is measured, and otherwise `_exit_code` over the scored
    gates.

    THE ONE HANDLER FOR `RefusedBeforeMeasuring` LIVES HERE, so that the deep
    refusals -- `ladder_rows` on a model whose routing cannot form a full tile
    stack, `--r-max` below one tile -- become a RETURNED 2 rather than an
    exception escaping through a caller. The message is not re-printed: `_refuse`
    printed it at the raise site, before the exception existed. An escape past
    this handler would still exit 2 and would still have printed its reason,
    because the class carries the code and `_refuse` did the printing; the
    handler is here because a caller that invokes `main` as a function -- every
    test in `tests/test_bm128_depth.py` -- wants the integer back rather than an
    exception.
    """
    try:
        return _run(argv)
    except RefusedBeforeMeasuring:
        return exit_codes.REFUSED


def _run(argv=None) -> int:
    """`main` without the refusal handler. Every path here returns a table code."""
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    lines: list[str] = [f"experiment  bm128_depth: can BLOCK_M={SUBJECT_BLOCK_M} "
                        f"show {TARGET_TREADS} clean memory-bound treads?", "",
                        predictions_text(b)]
    gates: list[Gate] = []
    payload: dict = {"predictions": predictions_text(b)}

    if args.audit or args.self_test:
        if args.self_test:
            try:
                more, g = self_test(b, noise=args.plant_noise, seed=args.seed,
                                    draws=args.draws)
            except SelfTestRefused as exc:
                print("\n".join(lines))
                print(f"\nREFUSED: {exc}")
                return exit_codes.REFUSED
            lines += more
            gates += g
        if args.audit:
            records, skipped = load_corpus(args.published, args.dtype,
                                           args.hardware_dir)
            if records:
                # Only when the corpus HAS BM=128 ladders. With none, V0 already
                # voids the page for vacuity and that is the gate that owns the
                # verdict; a second refusal from here would answer a question
                # nobody could have asked. When it has them but none carries a
                # usable alpha and a ridge, `render_premise` prints NOT
                # RECOMPUTABLE and returns, and the gates below still score:
                # see `PremiseNotRecomputable` for why that is not a default.
                lines += render_premise(args.published, args.hardware_dir)
            rows = [audit_record(r, seed=args.seed, draws=args.draws)
                    for r in records]
            more, g, pay = audit_report(rows, skipped, b)
            lines += more
            gates += g
            payload["audit"] = pay
        lines += ["", "## Gates", ""] + render_gates(gates)
        print("\n".join(lines))
        return _exit_code(gates)

    ridge, ridge_src, bandwidth, bw_src = _cost_inputs(args)
    plan = build_plan(args, cfg)
    # THE CARD, RESOLVED BEFORE THE RUN ID. See default_run_id: the results root
    # is a network volume shared between pods and every verdict here is scored
    # against a per-card ridge, so two cards must not derive one directory.
    detected = detect_card_slug()
    card = args.card or detected or UNKNOWN_CARD_SLUG
    if args.card and detected and args.card != detected:
        print(f"REFUSED: --card {args.card!r} but the attached device is "
              f"{detected!r}. --card may name a card that is ABSENT, so a "
              "laptop can print the pod's real path; it may never contradict "
              "one that is present. Nothing measured.")
        return exit_codes.REFUSED
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "bm128_depth" / run_id
    csv_path = out_dir / "cells.csv"
    card_path = out_dir / "CARD"
    cache_root = out_dir / "triton-cache"

    lines += ["", "## The plan", ""] + plan.lines(cfg) + [
        f"costing      ridge {ridge:.3f} ({ridge_src}); bandwidth "
        f"{bandwidth:.1f} GB/s ({bw_src})",
        f"card         {card}" + ("" if detected else
                                  f"  (NO DEVICE ATTACHED: the id above is the "
                                  f"{UNKNOWN_CARD_SLUG!r} one and is not what a "
                                  "pod derives; pass --card <slug> for that)"),
        f"WRITES TO    {out_dir}",
        f"             {git_visibility(out_dir)}",
        "             cells.csv (one row per tread per repeat, flushed), "
        "CARD, report.txt, report.json, triton-cache/"]

    if args.dry_run:
        law = depth_verdict(SUBJECT_BLOCK_M, b, args.alpha, ridge)
        lines += ["", "## What the law says about this plan before it runs", "",
                  "  " + law.line(),
                  f"  escape up needs alpha x rho >= "
                  f"{escape_up_alpha_rho(SUBJECT_BLOCK_M, b):.1f}; this plan "
                  f"sits at {args.alpha * ridge:.1f}",
                  f"  escape down needs rho >= "
                  f"{escape_down_rho(SUBJECT_BLOCK_M, b, TARGET_TREADS):.1f}; "
                  f"this card calibrates at {ridge:.1f}, or equivalently "
                  f"alpha <= {escape_down_alpha(TARGET_TREADS):.3f} with no "
                  "card in the argument",
                  "",
                  "  READ THE COSTING ALPHA AND RIDGE AS A COSTING, NOT AS A "
                  "PREDICTION. The pooled",
                  f"  alpha {args.alpha:.3f} is fitted over BLOCK_M <= 64 and "
                  f"{ridge:.1f} is a CALIBRATION ridge, not the",
                  f"  achieved one; together they say alpha x rho = "
                  f"{args.alpha * ridge:.1f}, while the 22 measured",
                  "  BLOCK_M=128 ladders say 126.8 in the median. The measured "
                  "number is the one that",
                  "  decides feasibility; this pair only decides how many "
                  "seconds the run takes.",
                  "  Nothing in --r-max, --model or --block-n appears in either "
                  "threshold. Run --audit for the measured corpus."]
        # THE MDE IS PART OF THE PLAN, not of the post mortem: the only cheap
        # moment to find that a gate cannot resolve the effect it is registered
        # against is before the pod is rented. A plan whose MDE is not stateable
        # is still a plan, and it says so in the one line where the number would
        # have been -- the sibling `tile_cap_test` degrades identically. What it
        # may NOT do is choose the exit code: a `--dry-run` measured nothing, so
        # a missing power calculation cannot make it CLAIM_FAIL.
        lines += mde_block(spread=args.plant_noise, reps=args.reps,
                           treads=len(plan.subject_rows))[0]
        # REFUSED (2) AND NOT DONE (0). A dry run prints a plan and times
        # nothing, so it scores no gate and prints no RESULT line, and
        # `exit_codes.classify_text` over this log raises `NoGatesScored` --
        # which that module documents as what a REFUSED log looks like from
        # there. Returning DONE made the two disagree in the one direction that
        # matters: DONE is "measured; every gate PASSED", and nothing was
        # measured.
        #
        # THE REPOSITORY DISAGREED WITH ITSELF AND THIS IS THE SIDE THAT WON.
        # On 2026-09-02 six scripts returned DONE from `--dry-run` -- this file,
        # `bm128_roofline`, `bn_decomposition`, `tile_cap_test`, and
        # `block_m_crossing_sweep` and `ruler_rebaseline` as the bare literal 0
        # -- and seven returned REFUSED (`calibrate_hardware`,
        # `dtype_tile_confound`, `memory_branch_anchor`, `occupancy_vs_swizzle`,
        # `rescore_published_reports`, `span_extent_separation`, `tile_sweep`).
        # REFUSED is the only one of the two that a log can be checked against,
        # and `dtype_tile_confound` had already written the argument out at its
        # own dry-run branch. The driver reads both as finished, so nothing is
        # re-queued either way: `dry_state` maps 0 to PLANNED and 2 to
        # PLAN_REFUSED and `arm` retries neither.
        lines += ["", "=" * 72,
                  "REFUSED. Nothing was measured and nothing was written.",
                  "  reason: --dry-run was given",
                  "  Everything above is arithmetic over the published corpus "
                  "and this repo's",
                  "  calibration. No gate was scored, so no RESULT line was "
                  "printed and none",
                  "  of it is a result. Run --audit for the gates over the "
                  "published ladders,",
                  "  --self-test for the planted worlds, or the bare command "
                  "on the pod.",
                  "=" * 72]
        print("\n".join(lines))
        return exit_codes.REFUSED

    visibility = git_visibility(out_dir)
    if args.require_git_visible and visibility.startswith("IGNORED"):
        print("\n".join(lines))
        print(f"\nREFUSING: {visibility}")
        return exit_codes.REFUSED

    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n".join(lines))
        # `missing_gpu_stack` points at the SWEEP's own off-GPU modes, and its
        # `--self-test` takes a float where this script's takes no argument.
        # Naming this file's modes here stops a reader following an invocation
        # that does not exist.
        first = missing.split(".")[0]
        print(f"\n{first}.\n"
              "Off GPU, this script's whole argument is still available:\n"
              "  --audit      the gates over every published BLOCK_M=128 ladder\n"
              "  --self-test  four planted worlds through the pod's own "
              "analysis, each with a registered verdict per gate and a "
              "registered exit code\n"
              "  --dry-run    the pod plan, the grid and the cost")
        return exit_codes.REFUSED

    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    print("\n".join(lines))

    try:
        from moe.bench.roofline import load_measured
        hw = load_measured()
    except Exception as exc:                            # noqa: BLE001
        raise _refuse(
            f"no usable calibration for the attached device ({exc}). V1 scores "
            "the compute reference against THIS card's measured bf16 ceiling "
            "and there is nothing to refuse or accept it with. Run "
            "scripts/calibrate_hardware.py first.") from exc
    if hw is None:
        raise _refuse(
            "no calibration for the attached device. V1 needs this card's own "
            "bf16 ceiling; the seven published A100 reports were scored against "
            "an H200 number precisely because a missing calibration was allowed "
            "to fall back. Run scripts/calibrate_hardware.py first.")
    ceiling = hw.peak(args.dtype) / 1e12
    # RE-DERIVED FROM THE ATTACHED DEVICE, overriding whatever `_cost_inputs`
    # was willing to assume. Those values are allowed to be a labelled
    # HYPOTHESIS because they only price the run; from here they also feed the
    # compute reference's qualification, which is a measurement decision, and a
    # hypothesis ridge reaching a qualification is exactly how seven published
    # A100 reports came to be scored against an H200 number.
    ridge = hw.ridge_point(args.dtype)
    bandwidth = hw.bandwidth_bytes_s / 1e9
    print(f"\nqualification ridge {ridge:.3f} Op/B and bandwidth "
          f"{bandwidth:.1f} GB/s, both from {hw.name}")

    # THE CLOCK THE ROOF WAS MEASURED AT, resolved ONCE, before any tread. It is
    # the reference `timing.clock_flags` compares against, and without it every
    # row's LEVEL verdict is None, which means "not determined" and excludes
    # nothing. Resolved here rather than per tread so the whole ladder is scored
    # against one number and a mid-run yaml rewrite cannot move it.
    reference_clock, clock_source = SWEEP.reference_clock_mhz(hw.name)
    print("reference clock: "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT RESOLVED ({clock_source}); every row's clock LEVEL "
                  "verdict will be None and no tread can be excluded for it"))

    # ONE PROVENANCE BLOCK PER RUN, built after the rulers resolve so it carries
    # their sources and before any measurement so every artefact of one run
    # carries one block. `iters` is None because --iters is retired as a timing
    # knob: on a pod each row's count comes from `time_kernel`, sized from
    # --cell-budget-ms, and recording the dead argparse default would put a
    # number in the report that no row used.
    prov = PV.provenance_block(
        instrument=SWEEP.timing_basis(),
        ridge=ridge, ridge_source=f"calibration: {hw.name}",
        bandwidth=bandwidth, bandwidth_source=f"calibration: {hw.name}",
        warmup_ms=args.warmup, iters=None, target_ms=args.cell_budget_ms)

    # THE RESUME GUARD, belt as well as braces. The card is already in the run
    # id, so another card lands in another directory and cannot normally reach
    # this cells.csv at all. This catches the ways it could anyway: an explicit
    # --out or --run-id aiming two cards at one place, a directory copied
    # between pods, or a cells.csv written before the card entered the id. It
    # REFUSES rather than starting over, because silently discarding measured
    # treads is its own way to lose an arm.
    if csv_path.exists():
        written_by = card_path.read_text().strip() if card_path.exists() else ""
        if written_by != card:
            raise _refuse(
                f"refusing to resume {csv_path}: written by card "
                f"{written_by or '<unrecorded, pre-card-in-id>'!r} and this run "
                f"is {card!r}. Resuming would report one card's treads against "
                "the other's ridge -- 145.8 against 162.8 Op/B -- which is a "
                "hybrid of two machines. Move or delete that directory "
                "deliberately. Nothing measured.")
    card_path.write_text(card + "\n")

    done, samples = read_samples(csv_path)
    compiles: dict[int, int] = {}
    executed: dict[int, int] = {}
    started = time.time()

    # THE REFERENCE FIRST, and its level checked before the subject costs
    # anything. A reference 43.6x too slow classifies every subject tread, so
    # measuring the subject before knowing the reference is sound is spending
    # the expensive half of the run on cells that cannot be read.
    print(f"\n-- reference ladder, BLOCK_M={REFERENCE_BLOCK_M} --")
    c, e = measure_setting(args, cfg, REFERENCE_BLOCK_M, plan.reference_rows,
                           csv_path, cache_root, plan.pinned, done, samples,
                           reference_clock)
    compiles[REFERENCE_BLOCK_M], executed[REFERENCE_BLOCK_M] = c, e

    ref_points, _, _ = collapse(samples, REFERENCE_BLOCK_M)
    ref_cells = [SWEEP.make_cell(cfg, n * REFERENCE_BLOCK_M, REFERENCE_BLOCK_M,
                                 ms, sm_count=1, block_n=1)
                 for n, ms in ref_points]
    early = SWEEP.compute_reference(
        ref_cells, (REFERENCE_BLOCK_M,), cfg=cfg, ridge=ridge,
        bandwidth_gbps=bandwidth, b=b, pinned=plan.pinned)
    if early.block_m and early.slope_per_tile:
        lvl = reference_level(cfg, early.block_m, early.slope_per_tile, ceiling,
                              hw.name)
        print(f"\n{lvl.line()}")
        if not lvl.passes:
            print("REFUSING to measure the subject: the compute reference is "
                  f"outside [{REFERENCE_LEVEL_FLOOR:.0%}, 100%] of this card's "
                  "ceiling, so every membership decision it would make is void. "
                  "This is the A100 BLOCK_N=256 failure, caught before the "
                  "expensive half of the run.")
            print(f"cells   {csv_path}")
            # INVALID, not CLAIM_FAIL and not REFUSED. The reference WAS
            # measured and it is unsound, so the arm cost pod minutes and
            # nothing on the page may be quoted; re-running it without changing
            # the world repeats the failure. REFUSED would have the driver treat
            # a spent half-run as free.
            print(f"exit     {exit_codes.describe(exit_codes.INVALID)}")
            return exit_codes.INVALID
    else:
        print(f"\nreference did not qualify: {early.note}")

    print(f"\n-- subject ladder, BLOCK_M={SUBJECT_BLOCK_M} --")
    c, e = measure_setting(args, cfg, SUBJECT_BLOCK_M, plan.subject_rows,
                           csv_path, cache_root, plan.pinned, done, samples,
                           reference_clock)
    compiles[SUBJECT_BLOCK_M], executed[SUBJECT_BLOCK_M] = c, e
    print(f"\nmeasured in {time.time() - started:.0f} s")

    more, g, pay = analyse_run(samples, cfg, b, ceiling, hw.name, compiles,
                               executed, ridge=ridge, bandwidth_gbps=bandwidth,
                               pinned=plan.pinned, seed=args.seed,
                               draws=args.draws)
    gates += g
    payload["run"] = pay
    payload["gpu"] = torch.cuda.get_device_name(0)
    # AFTER TWELVE METERED MINUTES THE REPORT GETS WRITTEN. This line used to
    # raise `SystemExit` out of `main` between the last timing and the first
    # `write_text`, so a pod run at `--reps 1` paid for both ladders, wrote no
    # report.json, and exited 1. The MDE is a section of a document; it is not a
    # gate, and it does not get a vote on the arm's verdict.
    payload["mde"], not_stateable = mde_block(
        spread=args.plant_noise, reps=args.reps,
        treads=len(plan.subject_rows))
    payload["mde_not_stateable"] = not_stateable or None
    # The block goes in LAST and through `stamp`, which raises if the payload
    # already carries a different value for any of the five keys it puts at the
    # top level. That is how the block and the report are made to agree rather
    # than merely coexist.
    payload = prov.stamp(payload)
    text = "\n".join(lines + more + ["", "## Gates", ""] + render_gates(gates))
    print("\n".join(more + ["", "## Gates", ""] + render_gates(gates)))
    (out_dir / "report.txt").write_text(text)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    for label, path in (("cells", csv_path),
                        ("report", out_dir / "report.txt"),
                        ("json", out_dir / "report.json")):
        # Every path, not just the directory: `.gitignore` re-includes
        # `results/published/` under a blanket `results/*` exclusion, and this
        # repo has already lost every published figure to a pattern that
        # matched at a depth nobody checked.
        print(f"{label:8s} {path}\n         {git_visibility(path)}")
    return _exit_code(gates)


if __name__ == "__main__":
    sys.exit(main())
