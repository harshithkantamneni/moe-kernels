#!/usr/bin/env python3
"""Rescore the published block_m reports against each card's OWN ridge.

    python scripts/rescore_published_reports.py --dry-run
    python scripts/rescore_published_reports.py --write

WHAT WAS WRONG. All 26 committed `*.report.json` files under
`results/published/` carry `ridge = 160.3` and `ridge_band = [160.3, 176.2]`.
That is the 2026-08-26 H200 figure, and it was passed on the command line to
every sweep in three later arms, including seven that ran on an A100. The A100's
own contemporaneous calibration puts its ridge at 145.8, 9.9% away; the H200
s3/s4 arms' own is 162.8, inside the old band. `results/published/
CALIBRATION_PROVENANCE.md` lists all three arms as unknown or refused,
`ANCHOR_RESCORE.txt` and `docs/COUNTERS.md` disclose it, and nothing INSIDE the
arm directories said a word (audit B2/R13/P2, fix B4).

WHAT IS AND IS NOT TAINTED, because this rescoring is deliberately narrow. The
audit's refuters established that every gate verdict in those reports is
ridge-INDEPENDENT: gate 4 scores BLOCK_M=64's peak against the run's own
BLOCK_M=256 plateau (100.02/165.80 = 0.603 on the A100 mixtral cell), gate 3 is
a ratio of two crossings at one ridge so the ridge cancels, and the ladder fits
are slopes. `ai_cap` is `2 BM / (alpha b)` and has no ridge in it. What the
ridge does reach is exactly four fields:

    ridge, ridge_band                      the quoted ceiling itself
    predictions[BM].crossing_rows_ridge_lo   solved at the low end of the band
    predictions[BM].crossing_rows_ridge_hi   solved at the high end
    predictions[BM].first_compute_tread      the tread the low end crosses on
    bracketing.horizon_rows                  2x the retracted alpha's crossing

and those are the only ones this script touches. Everything else is copied
byte for byte, which is checked rather than asserted: gate `fields_confined`
re-serialises the untouched half of every report and compares it with the
original, and a rescoring that moved anything else FAILS the gate instead of
shipping.

WHY NOT REGENERATE THE REPORTS INSTEAD. Because the cells are gone. No
`cells.csv` was published beside any of the 26 (audit B6), so the ladders,
plateaus and gate inputs cannot be recomputed from the repository at all. What
CAN be recomputed is the prediction block, because a prediction is arithmetic
over `(BLOCK_M, alpha, ridge, dtype bytes)` and every one of those is in the
file. Rescoring the recomputable half and stamping its source is strictly more
than the alternative, which is 26 files quoting a ceiling that belongs to no
attached device.

IDEMPOTENCE IS A GATE, NOT A HOPE. The script computes the new payload, compares
it with the file on disk, and writes NOTHING when they agree -- so the second
run leaves the tree clean and `rescored_utc` does not churn. `--self-test`
proves the FAIL branch of every gate by planting it.

Off-GPU, like everything else in this slice: it reads two YAML calibrations and
26 JSON files, and refuses per report rather than defaulting when a card cannot
be resolved.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench import exit_codes  # noqa: E402
from moe.bench import roofline as RL  # noqa: E402
from moe.bench.provenance import provenance_block  # noqa: E402
from moe.bench.published import two_sample_mde  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = REPO_ROOT / "results" / "published"

#: The fields this script is allowed to write. Named as data so
#: `fields_confined` can check the claim instead of trusting the code that
#: makes it.
RESCORED_TOP = ("ridge", "ridge_band", "ridge_source", "ridge_band_source",
                "bandwidth_source", "rescored_utc", "rescored_from")
RESCORED_PREDICTION = ("crossing_rows_ridge_lo", "crossing_rows_ridge_hi",
                       "first_compute_tread")
RESCORED_BRACKETING = ("horizon_rows",)

#: The calibration files' own `ridge_by_pattern` is published to one decimal, so
#: a rescored report quoting more digits would claim a precision its cited source
#: does not. The reports being replaced carry `160.3`, the same convention.
RIDGE_DECIMALS = 1

#: `json.dumps(payload, indent=2)` with no trailing newline reproduces every one
#: of the 26 committed files byte for byte. Pinned here, and asserted by
#: `fields_confined`, so a formatting change cannot masquerade as a rescoring.
JSON_INDENT = 2


class CardUnresolved(LookupError):
    """This arm's directory name matches no committed calibration.

    Raised per report rather than defaulted, because the only thing a default
    could be is another card's ceiling, which is the defect being repaired.
    """


def load_sweep():
    """`scripts/block_m_crossing_sweep.py`, as a module.

    IMPORTED RATHER THAN REIMPLEMENTED. `predict_tile` scans for the first tread
    where padded rows meet `ridge b Q(n) / 2`, and both sides step, so a
    "simpler" closed form here would round differently from the sweep and the
    rescored numbers would not be the sweep's answers. The file is a script and
    not a package module, so it is loaded by path; its module-level code is
    constants and definitions only, and its CLI sits behind a `__main__` guard.
    """
    path = REPO_ROOT / "scripts" / "block_m_crossing_sweep.py"
    spec = importlib.util.spec_from_file_location("SWEEP", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("SWEEP", module)
    spec.loader.exec_module(module)
    return module


def measured_profiles() -> dict[str, str]:
    """`{device slug: profile stem}` for every committed calibration.

    Built from the hardware directory rather than hard-coded, so adding a third
    card's calibration is enough to make its arms rescorable.
    """
    out = {}
    for path in sorted(RL.HARDWARE_DIR.glob("measured_*.yaml")):
        out[path.stem[len("measured_"):]] = path.stem
    return out


def profile_for_arm(arm_name: str, profiles: dict[str, str]) -> str:
    """The calibration whose device slug the arm directory names.

    LONGEST MATCH WINS, and a tie REFUSES. Arm names are
    `<date>-<device slug>-<label>`, so `nvidia_h200` and a hypothetical
    `nvidia_h200_nvl` would both match an NVL arm and the longer one is the
    right answer; two slugs of the SAME length matching one arm means the naming
    scheme no longer identifies a card, and guessing between them is how another
    machine's ceiling got into these files in the first place.
    """
    hits = sorted((slug for slug in profiles if slug in arm_name),
                  key=len, reverse=True)
    if not hits:
        raise CardUnresolved(
            f"arm {arm_name!r} names none of the calibrated devices "
            f"({', '.join(sorted(profiles))}). Its reports cannot be rescored: "
            "the only ridge available would be another card's.")
    if len(hits) > 1 and len(hits[0]) == len(hits[1]):
        raise CardUnresolved(
            f"arm {arm_name!r} matches {hits[0]!r} and {hits[1]!r} equally "
            "well, so its card is not decidable from its name")
    return profiles[hits[0]]


def calibration(profile: str, dtype: str, sweep):
    """`(ridge, band, ridge source, band source, bandwidth source)`.

    The ridge is `peak(dtype) / bandwidth` off the committed yaml, and the band
    is `ridge_band_from_detail`: the SAME device measured against its own
    surviving DRAM rulers, carried as a ratio against its ceiling pattern. Not
    the old `[160.3, 176.2]`, which is one card's triad ridge beside another
    card's read ridge and is a band across two machines.
    """
    import yaml

    doc = yaml.safe_load((RL.HARDWARE_DIR / f"{profile}.yaml").read_text())
    hw = RL.load_hardware(profile)
    ridge = round(hw.ridge_point(dtype), RIDGE_DECIMALS)
    detail = doc.get("detail") or {}
    band, band_source = sweep.ridge_band_from_detail(detail, ridge)
    band = [round(band[0], RIDGE_DECIMALS), round(band[1], RIDGE_DECIMALS)]
    pattern = detail.get("ceiling_pattern") or "unnamed"
    ridge_source = f"{profile}.yaml"
    bandwidth_source = (
        f"{profile}.yaml, {detail.get('achieved_bandwidth_gbps', 0):.1f} GB/s "
        f"({pattern} pattern), the denominator of the ridge above")
    return ridge, band, ridge_source, band_source, bandwidth_source


def rescore(payload: dict, ridge: float, band: list[float], sweep) -> dict:
    """A NEW payload with the four ridge-dependent quantities recomputed.

    Pure: the input dict is not mutated, so a caller can diff the two.

    `crossing_rows` is None when the tile's `ai_cap` sits at or below the ridge,
    and that null is a PREDICTION -- "this block size never crosses at any batch
    size" -- not a missing value. It is written as null for the same reason the
    sweep writes it that way, and lowering the A100's ridge from 160.3 to 145.8
    moves cells INTO crossing, which is exactly the kind of change the old
    figure was hiding.
    """
    out = json.loads(json.dumps(payload))    # deep copy through the wire format
    alpha = float(payload["alpha"])
    b = int(payload["dtype_bytes"])
    block_sizes = sorted(int(k) for k in payload["predictions"])

    lo = sweep.predictions(block_sizes, alpha, band[0], b)
    hi = sweep.predictions(block_sizes, alpha, band[1], b)
    for bm in block_sizes:
        pred = out["predictions"][str(bm)]
        pred["crossing_rows_ridge_lo"] = lo[bm].crossing_rows
        pred["crossing_rows_ridge_hi"] = hi[bm].crossing_rows
        pred["first_compute_tread"] = lo[bm].first_compute_tread

    # `bracketing` is registered for the tile gate 4 asks its absence question
    # about, which `null_block_m` names, and the horizon is twice the crossing
    # the RETRACTED alpha predicts for it at the ridge. Both halves come from the
    # sweep so the number is the sweep's, not a re-derivation of it.
    null_bm = sweep.null_block_m(block_sizes)
    retracted = sweep.predict_tile(null_bm, sweep.RETRACTED_ALPHA, ridge, b)
    out["bracketing"]["horizon_rows"] = 2.0 * (retracted.crossing_rows or 0.0)

    out["ridge"] = ridge
    out["ridge_band"] = list(band)
    return out


def untouched(payload: dict) -> str:
    """Everything the rescoring is NOT allowed to change, as a stable string.

    The comparison `fields_confined` makes. Built by deleting the permitted
    fields from a copy and serialising what is left, so "nothing else moved" is
    a computed fact about the two documents rather than a claim about the code.
    """
    rest = json.loads(json.dumps(payload))
    for key in RESCORED_TOP:
        rest.pop(key, None)
    for pred in rest.get("predictions", {}).values():
        for key in RESCORED_PREDICTION:
            pred.pop(key, None)
    for key in RESCORED_BRACKETING:
        rest.get("bracketing", {}).pop(key, None)
    return json.dumps(rest, indent=JSON_INDENT, sort_keys=True)


def stamp(payload: dict, original: dict, ridge_source: str, band_source: str,
          bandwidth_source: str, now: str) -> dict:
    """Add the provenance fields, in a FIXED order, without losing history.

    TWO THINGS HERE EXIST FOR IDEMPOTENCE AND BOTH ARE LOAD-BEARING.

    The keys are popped and re-set rather than assigned, so a report that
    already carries them ends up with them in the same positions as one that
    does not. `json.dumps` preserves insertion order, so without this a second
    pass would reorder the tail of the document and the file would differ by
    layout while agreeing on every value.

    `rescored_from` is preserved when it is already there. It records what the
    report said BEFORE the first rescoring, and a second pass reading its own
    output would otherwise overwrite `160.3` with `145.8` and erase the very
    fact it was written to keep.
    """
    for key in ("ridge_source", "ridge_band_source", "bandwidth_source",
                "rescored_utc", "rescored_from"):
        payload.pop(key, None)
    payload["ridge_source"] = ridge_source
    payload["ridge_band_source"] = band_source
    payload["bandwidth_source"] = bandwidth_source
    payload["rescored_utc"] = now
    payload["rescored_from"] = original.get("rescored_from") or {
        "ridge": original["ridge"],
        "ridge_band": list(original["ridge_band"]),
        "why": ("the 2026-08-26 H200 figure, passed on the command line to "
                "arms on two different cards; see this arm's NOTE.md"),
    }
    return payload


def without_utc(payload: dict | None) -> str:
    """The document with `rescored_utc` removed, as a string.

    The comparison that decides whether a report needs rewriting at all. A
    timestamp that changed because the clock moved is not a rescoring, and
    treating it as one would make every run dirty the tree -- which is the
    failure mode audit A6 catalogues, arriving through the fix for B4.
    """
    if payload is None:
        return ""
    rest = json.loads(json.dumps(payload))
    rest.pop("rescored_utc", None)
    return json.dumps(rest, indent=JSON_INDENT)


def rescored_payload(before: dict, path: Path, sweep, now: str) -> dict | None:
    """The rescored document for one report, or None when its card is unresolved.

    THE ONLY PLACE A RESCORED PAYLOAD IS BUILT. `plan` and the idempotence gate
    both call it, so the gate is checking the transform the writer applies
    rather than a second implementation of it that could agree by luck and
    diverge later.

    The timestamp is carried over when nothing else moved, which is what makes
    a second pass a no-op instead of a one-field rewrite.
    """
    try:
        profile = profile_for_arm(path.parent.name, measured_profiles())
        ridge, band, r_src, b_src, bw_src = calibration(
            profile, str(before.get("dtype", "bf16")), sweep)
    except (CardUnresolved, ValueError, KeyError, FileNotFoundError):
        return None
    after = stamp(rescore(before, ridge, band, sweep), before,
                  r_src, b_src, bw_src, now)
    if without_utc(before) == without_utc(after) and before.get("rescored_utc"):
        after["rescored_utc"] = before["rescored_utc"]
    return after


def report_paths(root: Path) -> list[Path]:
    """Both committed layouts, deduplicated. Same rule as `alpha_surface.py`."""
    return sorted({*root.rglob("report.json"), *root.rglob("*.report.json")})


class Outcome:
    """One report's rescoring: what it was, what it becomes, and whether it moved."""

    def __init__(self, path: Path, before: dict, after: dict | None,
                 refusal: str = ""):
        self.path = path
        self.before = before
        self.after = after
        self.refusal = refusal

    @property
    def changed(self) -> bool:
        return self.after is not None and self.serialised != self.original_text

    @property
    def serialised(self) -> str:
        return json.dumps(self.after, indent=JSON_INDENT)

    @property
    def original_text(self) -> str:
        return json.dumps(self.before, indent=JSON_INDENT)

    @property
    def confined(self) -> bool:
        return self.after is not None and \
            untouched(self.before) == untouched(self.after)


def plan(paths: list[Path], sweep, now: str) -> list[Outcome]:
    """Compute every rescoring without writing anything."""
    profiles = measured_profiles()
    out = []
    for path in paths:
        before = json.loads(path.read_text())
        try:
            profile_for_arm(path.parent.name, profiles)
        except CardUnresolved as exc:
            out.append(Outcome(path, before, None, f"{type(exc).__name__}: {exc}"))
            continue
        after = rescored_payload(before, path, sweep, now)
        if after is None:
            out.append(Outcome(path, before, None,
                               "the card resolved but its calibration did not"))
            continue
        out.append(Outcome(path, before, after))
    return out


def mde_report(outcomes: list[Outcome]) -> list[str]:
    """The MDE line, from the noise these reports themselves recorded.

    `timing_spread_median` is each sweep's within-cell pstdev over p50, so it is
    the relative noise on ONE ladder point. It is the only noise model these
    files carry, it is a WITHIN-process spread and therefore a floor, and it is
    what the ridge shift below has to be read against: a rescoring that moves a
    prediction by less than the instrument's own spread has changed a digit and
    not a conclusion (audit B14).
    """
    spreads = [float(o.before.get("timing_spread_median") or 0.0)
               for o in outcomes if o.before.get("timing_spread_median")]
    if not spreads:
        return ["  MDE: NOT STATED -- no report records timing_spread_median, "
                "so these files carry",
                "  no noise model and the size of the rescoring cannot be "
                "judged against one."]
    sd = statistics.median(spreads)
    lines = [f"  noise assumption: median timing_spread_median "
             f"{sd:.2%} over {len(spreads)} report(s), the within-cell pstdev "
             "over p50",
             f"  MDE {two_sample_mde(sd):.1%} relative (two independent cells, "
             "90% two-sided, 80% power).",
             "  WITHIN one process, so it is a floor: it prices neither the "
             "session nor the card."]
    shifts = [abs(o.after["ridge"] - o.before["ridge"]) / o.before["ridge"]
              for o in outcomes if o.after is not None and o.before.get("ridge")]
    if shifts:
        lines.append(
            f"  ridge shifts here: {min(shifts):.1%} to {max(shifts):.1%}; "
            + ("the largest is above that MDE"
               if max(shifts) > two_sample_mde(sd)
               else "ALL of them sit inside that MDE"))
    return lines


def gates(outcomes: list[Outcome], sweep, now: str) -> list[tuple[str, str, str, str]]:
    """`(kind, name, verdict, detail)` per gate. Every one can PASS and FAIL."""
    scored = []

    refused = [o for o in outcomes if o.after is None]
    scored.append((
        exit_codes.VALIDITY, "card_resolved",
        exit_codes.PASS if not refused and outcomes else exit_codes.FAIL,
        f"{len(outcomes) - len(refused)} of {len(outcomes)} report(s) resolved "
        "a card"
        + ("" if not refused else
           f"; refused: {refused[0].path.parent.name}/{refused[0].path.name} "
           f"({refused[0].refusal})")))

    written = [o for o in outcomes if o.after is not None]
    leaked = [o for o in written if not o.confined]
    scored.append((
        exit_codes.VALIDITY, "fields_confined",
        exit_codes.PASS if written and not leaked else exit_codes.FAIL,
        f"{len(written) - len(leaked)} of {len(written)} report(s) differ from "
        "the original in the registered fields only"
        + ("" if not leaked else f"; leaked: {leaked[0].path.name}")))

    # Idempotence, MEASURED by re-running the transform on its own output rather
    # than argued from the code's shape.
    unstable = []
    for outcome in written:
        again = rescored_payload(json.loads(json.dumps(outcome.after)),
                                 outcome.path, sweep, now)
        if again is None or json.dumps(again, indent=JSON_INDENT) != outcome.serialised:
            unstable.append(outcome)
    scored.append((
        exit_codes.CLAIM, "idempotent",
        exit_codes.PASS if written and not unstable else exit_codes.FAIL,
        f"{len(written) - len(unstable)} of {len(written)} report(s) are "
        "unchanged by a second pass"
        + ("" if not unstable else f"; first drifting: {unstable[0].path.name}")))

    wrong = [o for o in written
             if o.after["ridge_source"] != f"measured_{_slug(o.path)}.yaml"]
    scored.append((
        exit_codes.CLAIM, "ridge_is_own_card",
        exit_codes.PASS if written and not wrong else exit_codes.FAIL,
        f"{len(written) - len(wrong)} of {len(written)} report(s) cite the "
        "calibration of the card their arm names"
        + ("" if not wrong else f"; first mismatch: {wrong[0].path.name}")))
    return scored


def _slug(path: Path) -> str:
    """The device slug this report's ARM directory names, for the gate above.

    Derived from the path rather than from the payload on purpose: the gate is
    checking that the ridge came from the right card, and reading the card out
    of the same field being checked would make the check vacuous.
    """
    profiles = measured_profiles()
    stem = profile_for_arm(path.parent.name, profiles)
    return stem[len("measured_"):]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", type=Path, default=PUBLISHED,
                    help="directory to walk (default results/published)")
    ap.add_argument("--write", action="store_true",
                    help="write the rescored reports. Without it the run is a "
                         "plan and touches nothing")
    ap.add_argument("--report", type=Path, default=None,
                    help="write this run's own JSON report, with a provenance "
                         "block, to PATH")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise every gate off-GPU, including the FAIL "
                         "branch of each")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    sweep = load_sweep()
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    paths = report_paths(args.root)
    if not paths:
        print(f"REFUSED: no report.json or *.report.json under {args.root}")
        return exit_codes.REFUSED

    outcomes = plan(paths, sweep, now)
    print(f"# rescoring {len(paths)} published report(s) under "
          f"{_relative(args.root)}")
    print()
    print("## the plan")
    print()
    for line in mde_report(outcomes):
        print(line)
    print()
    by_arm: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        by_arm.setdefault(outcome.path.parent.name, []).append(outcome)
    for arm, group in sorted(by_arm.items()):
        first = group[0]
        if first.after is None:
            print(f"  {arm}: REFUSED -- {first.refusal}")
            continue
        changed = sum(1 for o in group if o.changed)
        print(f"  {arm}: {len(group)} report(s), {changed} to rewrite")
        print(f"    ridge {first.before['ridge']} -> {first.after['ridge']} "
              f"  band {first.before['ridge_band']} -> "
              f"{first.after['ridge_band']}")
        print(f"    source {first.after['ridge_source']}")

    print()
    print("## the gates")
    print()
    scored = gates(outcomes, sweep, now)
    for kind, name, verdict, detail in scored:
        print(exit_codes.result_line(kind, name, verdict, detail))

    rc = exit_codes.classify([(k, v) for k, _n, v, _d in scored])
    print()
    if args.write and rc in (exit_codes.DONE, exit_codes.CLAIM_FAIL):
        wrote = 0
        for outcome in outcomes:
            if outcome.after is None or not outcome.changed:
                continue
            outcome.path.write_text(outcome.serialised)
            wrote += 1
        print(f"wrote {wrote} report(s); "
              f"{sum(1 for o in outcomes if o.after is not None) - wrote} "
              "already carried the rescored values and were left alone")
    elif args.write:
        print("NOT WRITING: a gate did not pass, and a rescoring whose own "
              "checks failed must not reach the published tree")
    else:
        print("plan only; pass --write to rewrite the reports")

    if args.report:
        args.report.write_text(json.dumps({
            "schema": "moe-kernels/rescore/1",
            "root": _relative(args.root),
            "reports": len(paths),
            "rewritten": sum(1 for o in outcomes if o.changed),
            "gates": [{"kind": k, "name": n, "verdict": v, "detail": d}
                      for k, n, v, d in scored],
            "exit_code": rc,
            "provenance": provenance_block(
                repo_root=REPO_ROOT, instrument="analysis/rescore",
                ridge_source="per-card measured_*.yaml",
                bandwidth_source="per-card measured_*.yaml").as_dict(),
        }, indent=2) + "\n")
    return rc


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# self-test: every gate, PASS branch and planted FAIL branch
# --------------------------------------------------------------------------

def _synthetic(ridge: float = 160.3) -> dict:
    """A minimal report with the shape the 26 published ones share."""
    return {
        "alpha": 0.558, "retracted_alpha": 0.1, "ridge": ridge,
        "ridge_band": [ridge, 176.2], "dtype_bytes": 2, "dtype": "bf16",
        "model": "mixtral-8x7b",
        "timing_spread_median": 0.005,
        "predictions": {
            "64": {"ai_cap": 114.7, "crossing_rows_ridge_lo": None,
                   "crossing_rows_ridge_hi": None, "first_compute_tread": None},
            "256": {"ai_cap": 458.8, "crossing_rows_ridge_lo": 160.3,
                    "crossing_rows_ridge_hi": 176.2, "first_compute_tread": 1},
        },
        "bracketing": {"horizon_rows": 416.78, "reached_rows": 1024.0},
        "gates": [{"number": 4, "verdict": "PASS"}],
    }


def self_test() -> int:
    """Returns 0 when every check agrees, 1 otherwise. Prints one line each.

    Each gate is exercised twice: once on a corpus that should pass it and once
    on a corpus with its failure PLANTED, because a gate that has only ever been
    seen to pass has not been shown to be capable of failing. The proof that
    THIS function can return 1 lives in
    `tests/test_rescore_published.py::test_self_test_fails_when_a_gate_is_broken`.
    """
    sweep = load_sweep()
    now = "1970-01-01T00:00:00Z"
    bad = 0

    def check(label: str, ok: bool) -> None:
        nonlocal bad
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    tmp = Path(__file__).resolve().parent.parent / "results" / "published"
    a100 = next(p for p in report_paths(tmp) if "a100" in p.parent.name)

    # card_resolved, both branches
    ok_outcome = plan([a100], sweep, now)
    check("card_resolved PASS on a real A100 report",
          gates(ok_outcome, sweep, now)[0][2] == exit_codes.PASS)
    fake = Outcome(Path("results/published/2026-01-01-nvidia_unknown-x/r.json"),
                   _synthetic(), None, "planted: no such card")
    check("card_resolved FAIL on a planted unresolvable arm",
          gates([fake], sweep, now)[0][2] == exit_codes.FAIL)

    # fields_confined, both branches
    check("fields_confined PASS on a real rescoring",
          gates(ok_outcome, sweep, now)[1][2] == exit_codes.PASS)
    leaky = Outcome(ok_outcome[0].path, ok_outcome[0].before,
                    json.loads(json.dumps(ok_outcome[0].after)))
    leaky.after["plateau_tflops"] = -1.0        # a field nobody may touch
    check("fields_confined FAIL when a non-registered field moves",
          gates([leaky], sweep, now)[1][2] == exit_codes.FAIL)

    # idempotent, both branches
    check("idempotent PASS on a real rescoring",
          gates(ok_outcome, sweep, now)[2][2] == exit_codes.PASS)
    drifting = Outcome(ok_outcome[0].path, ok_outcome[0].before,
                       json.loads(json.dumps(ok_outcome[0].after)))
    drifting.after["ridge"] = 999.9             # a second pass would put it back
    check("idempotent FAIL when a second pass would move the file",
          gates([drifting], sweep, now)[2][2] == exit_codes.FAIL)

    # ridge_is_own_card, both branches
    check("ridge_is_own_card PASS on a real rescoring",
          gates(ok_outcome, sweep, now)[3][2] == exit_codes.PASS)
    borrowed = Outcome(ok_outcome[0].path, ok_outcome[0].before,
                       json.loads(json.dumps(ok_outcome[0].after)))
    borrowed.after["ridge_source"] = "measured_nvidia_h200.yaml"
    check("ridge_is_own_card FAIL when a report cites the other card",
          gates([borrowed], sweep, now)[3][2] == exit_codes.FAIL)

    # the A100's own ridge, which is the number this whole script is about
    after = ok_outcome[0].after
    check(f"the A100 report is rescored to 145.8, got {after['ridge']}",
          after["ridge"] == 145.8)
    check("gate 4's verdict is untouched by the rescoring",
          after["gates"] == ok_outcome[0].before["gates"])

    # classify wiring: a planted VALIDITY FAIL must not read as DONE
    planted = exit_codes.classify(
        [(k, v) for k, _n, v, _d in gates([fake], sweep, now)])
    check(f"a refused card exits INVALID, not DONE (got {planted})",
          planted == exit_codes.INVALID)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
