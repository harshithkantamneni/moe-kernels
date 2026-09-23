#!/usr/bin/env python
"""The Python half of scripts/alpha_g_chain.sh: every read of a report.json, a
log or a ledger that the chain needs, as one small command each, so the shell
stays a sequencer and the reading is testable off GPU.

    alpha_g_chain_helpers.py card                   -> card slug, why not (the probe's reason)
    alpha_g_chain_helpers.py device                 -> this card's UUID, as R3's DEVICE file keys it
    alpha_g_chain_helpers.py run-id LOG EXPERIMENT  -> the run id the plan page printed
    alpha_g_chain_helpers.py estimate LOG           -> seconds the arm's own --dry-run priced
    alpha_g_chain_helpers.py estimate-basis LOG     -> what that figure is made of, in words
    alpha_g_chain_helpers.py reading REPORT         -> one PAIRS row's ratio half (R3)
    alpha_g_chain_helpers.py pairs REPORT...        -> the reports that FORMED a ratio, one per line
    alpha_g_chain_helpers.py eta REPORT             -> eta, lo, hi, band, exit word (R1)
    alpha_g_chain_helpers.py eta-for SESSION RESULTS G  -> the same, for the G's R1 log
    alpha_g_chain_helpers.py gate REPORT TAG        -> that gate's verdict, `absent` or `unreadable`
    alpha_g_chain_helpers.py probe-note REPORT      -> the alignment probe's note and graph_calls
    alpha_g_chain_helpers.py verdict LOG            -> what its RESULT lines imply (2nd opinion)
    alpha_g_chain_helpers.py pairs-table SESSION RESULTS LADDER SEEDS [key=value...]
                                                    -> rewrites PAIRS.tsv, PAIRS-by-G.tsv,
                                                       PAIRS-fixed.tsv and PAIRS-README.txt
    alpha_g_chain_helpers.py calibration-dir LOG    -> the run directory calibrate wrote

Every command prints tab-separated fields on ONE line (or one path per line for
`pairs`) and exits 0; a report that cannot be read prints `unreadable` in the
first field rather than crashing the chain.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from moe.bench import exit_codes  # noqa: E402

RUN_ID_LINE = re.compile(r"^experiment\s+(\S+)\s*/\s*(\S+)\s*$", re.M)
ESTIMATE_LINES = (
    re.compile(r"takes about (\d+) s"),                  # R3 at a duty below 1: the wall figure
    re.compile(r"estimated GPU time (\d+) s"),           # R3 at full duty
    re.compile(r"estimated wall time (\d+) s"),          # R1
)
#: R3's plan names the alignment probe's seconds INSIDE its GPU figure, and
#: below full duty its wall line leaves them OUT ("the probe is timed at full
#: duty"): the price of a run below full duty is the wall line plus these.
PROBE_SECONDS = re.compile(r"includes the alignment probe's (\d+) s")
#: The one line calibrate_hardware.py prints for the ruler it measured, under
#: its untracked run directory (the published copy is a second line, PUBLISHED).
CALIBRATE_WROTE = re.compile(r"^\[calibrate\] wrote (\S+\.yaml)\s*$", re.M)

#: The R1 exit words a regime word may be read off. INVALID is a page whose
#: own validity gates refused it (R1's V7 says an interval above 1 means
#: "something other than the SM clock moved"), and REFUSED, ERROR and an
#: unscored page scored nothing: the word is withheld, never computed.
BAND_QUOTABLE = (exit_codes.CODE_NAMES[exit_codes.DONE],
                 exit_codes.CODE_NAMES[exit_codes.CLAIM_FAIL])
#: What an R1 column reads when the G has no R1 report on disk yet.
ETA_UNMEASURED = ["none", "none", "none", "unmeasured", "unscored"]


def _load(path: str | Path) -> dict | None:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _fmt(v, spec: str = ".4f") -> str:
    return "none" if v is None else format(float(v), spec)


def _one_line(text: str) -> str:
    """No tab and no newline: the field goes into a TSV row or a ledger note."""
    return " ".join(str(text).split())


def _exit_word(payload: dict) -> str:
    gates = payload.get("gates") or []
    if not gates:
        return "unscored"
    try:
        rc = exit_codes.classify((g["kind"], g.get("tag") or g.get("number"), g["verdict"])
                                 for g in gates)
    except Exception:                                             # noqa: BLE001
        return "unscored"
    return exit_codes.CODE_NAMES.get(rc, str(rc))


def card() -> list[str]:
    """`slug  reason`: the card as the driver and every report name it, through
    moe/bench/provenance, and why not when it cannot be named. The reason is
    the probe's own text, because "nocard" alone hid a torch wheel built for a
    newer driver than the host's behind a STOP that named the wrong row."""
    try:
        from moe.bench.provenance import card_slug, provenance_block
    except Exception as exc:                                      # noqa: BLE001
        return ["nocard", _one_line(f"moe.bench.provenance is not importable from "
                                    f"{ROOT}: {exc.__class__.__name__}: {exc}")]
    try:
        prov = provenance_block(repo_root=ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return ["nocard", _one_line(f"provenance_block raised {exc.__class__.__name__}: {exc}")]
    if not prov.gpu_name:
        return ["nocard", _one_line(prov.missing.get("gpu_name", "no reason recorded"))]
    return [card_slug(prov.gpu_name), ""]


def _bare_uuid(text: str) -> str:
    """torch prints a card's UUID bare and nvidia-smi prefixes it `GPU-`: one
    spelling, so a pass that read one source and a resume that read the other
    still agree on the card."""
    text = text.strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def device() -> str:
    """This card's identity: R3's own `device_identity` (torch's UUID, so the
    chain's DEVICE and the ratio arm's DEVICE name the card one way), then
    nvidia-smi's when torch gave none, else whatever weaker identity R3 would
    record, else ''."""
    import private_weight_reference as PWR
    ident = PWR.device_identity()
    if ident and not ident.startswith(PWR.NO_UUID_PREFIX):
        return _bare_uuid(ident)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            out = None
        if out is not None and out.returncode == 0 and out.stdout.strip():
            return _bare_uuid(out.stdout.strip().splitlines()[0])
    return ident


def run_id(log: str | Path, experiment: str) -> str:
    """The run id the arm's plan page printed for `experiment`, or ''."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return ""
    for name, rid in RUN_ID_LINE.findall(text):
        if name == experiment:
            return rid
    return ""


def _estimate_terms(log: str | Path) -> list[tuple[int, str]]:
    """The seconds an arm's own --dry-run priced, term by term, off its plan
    page, or [] when it priced nothing. R3 below full duty is TWO terms: the
    ladder's wall line at the duty and the alignment probe's seconds, which
    that line leaves out because the probe is timed at full duty (the driver
    books the same arm as their sum)."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return []
    for k, pattern in enumerate(ESTIMATE_LINES):
        m = pattern.search(text)
        if not m:
            continue
        secs = int(m.group(1))
        if k == 0:
            probe = PROBE_SECONDS.search(text)
            terms = [(secs, "the ladder's wall at the plan's duty")]
            if probe:
                terms.append((int(probe.group(1)),
                              "the alignment probe, timed at full duty on top of it"))
            return terms
        if k == 1:
            return [(secs, "the plan's GPU figure at full duty, the alignment probe inside it")]
        return [(secs, "the plan's wall figure")]
    return []


def estimate(log: str | Path) -> str:
    """Seconds the arm's own --dry-run priced, off its plan page; '' if none.
    The sum of `_estimate_terms`."""
    terms = _estimate_terms(log)
    return str(sum(s for s, _ in terms)) if terms else ""


def estimate_basis(log: str | Path) -> str:
    """What `estimate` is made of, in words: `581 s the ladder's wall ... +
    9 s the alignment probe ...`; '' if the plan priced nothing."""
    return " + ".join(f"{s} s {what}" for s, what in _estimate_terms(log))


def reading_header() -> list[str]:
    """The ratio half of PAIRS.tsv's header, the arm names read from the arm."""
    import private_weight_reference as PWR
    return ["G", "seed", "ratio", "lo", "hi", "exit", "exit_scope", "duty", "run_id",
            "rep_n", "rep_spread", "rep_sd", "env_lo", "env_hi", "joint",
            *[f"clk_{arm}" for arm in PWR.ARMS], "low_cells"]


def _formed(p: dict) -> bool:
    """The report formed its own ratio and interval: the rule `run_reading`
    and `pairs` apply."""
    iv = p.get("ratio_interval") or [None, None]
    return p.get("ratio") is not None and iv[0] is not None and iv[1] is not None


def _joint_is_its_own(p: dict) -> bool:
    """Is the report's `replicates` block a reading THIS run is in: the run
    formed its own ratio (R3 builds the block from the replicates alone when
    it did not) and, where the block lists its runs by id, this run is one."""
    rep = p.get("replicates")
    if not isinstance(rep, dict) or not _formed(p):
        return False
    ids = [r.get("run_id") for r in rep.get("runs") or [] if isinstance(r, dict)]
    return not any(ids) or p.get("run_id") in ids


def reading(report: str | Path) -> list[str]:
    """One PAIRS row's ratio half for a private_weight_reference report:

    `G seed ratio lo hi exit exit_scope duty run_id`, the run's own within-run
    reading and its exit word; `exit_scope` says how C1 inside that word was
    scored: `alone` on this run's interval, `envelope` on the envelope of
    this run's and the earlier seeds' it was given through --replicate-of;
    `rep_n rep_spread rep_sd env_lo env_hi joint`, the JOINT reading over this
    seed and those earlier ones (the report's `replicates` block), FILLED ONLY
    when the block is a reading this run is in (`_joint_is_its_own`): a run
    that formed no ratio of its own gets a block built from the other runs
    alone, and its row says `none` rather than quote their envelope as its
    own. The within-run interval is a bootstrap over repeats and understates
    the cross-run spread (DESIGN DECISION 14); the per-G joint over every
    seed, whatever order they ran in, is PAIRS-by-G.tsv's (`by_g`);
    `clk_<arm>...` each arm's median under-load clock over the ladder, off
    `treads_table`, and `low_cells`, the (arm, tread) cells whose
    `level_sides` carries LEVEL LOW. At a duty below 1 every cell is expected
    at the ceiling, so a LOW cell is the throttle signature no R3 gate reads.
    """
    p = _load(report)
    if p is None or p.get("experiment") != "private_weight_reference":
        return ["unreadable"]
    import private_weight_reference as PWR

    from moe.bench.timing import LEVEL_LOW

    iv = p.get("ratio_interval") or [None, None]
    own = _joint_is_its_own(p)
    row = [str((p.get("pinned") or {}).get("GROUP_SIZE_M", "?")),
           str(p.get("seed") if p.get("seed") is not None else "unrecorded"),
           _fmt(p.get("ratio")), _fmt(iv[0]), _fmt(iv[1]), _exit_word(p),
           "envelope" if own else "alone",
           str(p.get("duty", 1.0)), str(p.get("run_id") or Path(report).parent.name)]

    rep = p.get("replicates")
    if own:
        env = rep.get("envelope") or [None, None]
        row += [str(rep.get("n", "none")), _fmt(rep.get("spread")), _fmt(rep.get("sd")),
                _fmt(env[0]), _fmt(env[1]), str(rep.get("verdict") or "none")]
    else:
        row += ["none"] * 6

    table = [r for r in (p.get("treads_table") or []) if isinstance(r, dict)]
    for arm in PWR.ARMS:
        clocks = [float(r["sm_clock_load_mhz"]) for r in table
                  if r.get("arm") == arm and r.get("sm_clock_load_mhz")]
        row.append(_fmt(statistics.median(clocks), ".0f") if clocks else "none")
    sided = [r for r in table if "level_sides" in r]
    row.append(str(sum(1 for r in sided if LEVEL_LOW in (r.get("level_sides") or [])))
               if sided else "none")
    return row


def pairs(reports: list[str]) -> list[str]:
    """The reports a later seed may be scored WITH: this arm's, measured, with a
    ratio and an interval formed. The same rule `load_replicates` applies, so
    a seed-1 line never names a report that would refuse it."""
    out = []
    for r in reports:
        p = _load(r)
        if p is None or p.get("experiment") != "private_weight_reference":
            continue
        if p.get("synthetic") or not _formed(p):
            continue
        out.append(r)
    return out


def eta(report: str | Path) -> list[str]:
    """`eta lo hi band exit` for a clock_elasticity report: the per-M-tile
    elasticity the arm registered and the band its interval landed in, read
    through the arm's own `band_of` so the edges are never restated.

    THE BAND IS READ ONLY OFF A PAGE WHOSE GATES STAND BEHIND IT: exit DONE or
    CLAIM_FAIL. Any other exit prints `withheld:<EXIT>` in the band field.
    `band_of` alone would put CLOCK-CARRIES on an interval above V7's
    admissible edge, on a page that exits INVALID, and that word asserts the
    clock mechanism the page has just disclaimed. The all-tread reading of
    session 4's G=16 cells (printed beside the claim, never gated) is such an
    interval. The claim itself, over treads 2 and deeper (D2), passes V7 on
    those cells, exits CLAIM_FAIL, and its CLOCK-CARRIES is quoted here.
    """
    p = _load(report)
    if p is None or "elasticity" not in p:
        return ["unreadable"]
    est = p["elasticity"] or {}
    value, lo, hi = est.get("value"), est.get("lo"), est.get("hi")
    word = _exit_word(p)
    if word not in BAND_QUOTABLE:
        band = f"withheld:{word}"
    elif lo is None or hi is None:
        band = "unresolved"
    else:
        import clock_elasticity as CE
        got = CE.band_of(float(lo), float(hi))
        band = got[0] if got else "STRADDLES"
    return [_fmt(value), _fmt(lo), _fmt(hi), band, word]


#: PAIRS-by-G.tsv's header: the R3 half off `by_g`, then R1's columns.
BY_G_HEADER = ["G", "n", "seeds", "mean", "sd", "env_lo", "env_hi", "joint",
               "invalid_in_envelope", "eta", "eta_lo", "eta_hi", "band", "eta_exit", "note"]


def by_g(g: str, reports: list[str]) -> tuple[list[str], str]:
    """One PAIRS-by-G row's R3 half, and its note: EVERY report of this G
    that formed a ratio, whatever order its seeds ran in, read together by
    R3's own cross-run machinery the way `--read RUN --replicate-of ...`
    reads them (`run_reading` of the first, `load_replicates` of the rest
    against its design, `cross_run` over all), so the envelope, the sd and
    the joint verdict are R3's and are written nowhere else.

    `n seeds mean sd env_lo env_hi joint invalid_in_envelope`: `sd` is the
    points' (R3 forms one from three); `joint` is C1's rule on the envelope;
    `invalid_in_envelope` names the seeds inside it whose own page exited
    INVALID, which `load_replicates` admits by design (their spread is
    information, their ratio is not quotable alone). A refusal by
    `load_replicates` (two duties, say) is `REFUSED` with its reason.
    """
    import private_weight_reference as PWR
    formed = pairs(reports)
    if not formed:
        return [g, "0"] + ["none"] * 7, "no report of this G formed a ratio yet"
    payload = _load(formed[0]) or {}
    seeds = " ".join(str(s) if (s := (_load(r) or {}).get("seed")) is not None
                     else "unrecorded" for r in formed)
    try:
        this = PWR.run_reading(payload, Path(formed[0]))
        reps = PWR.load_replicates(
            formed[1:], card_known=True, this=this,
            design={k: payload.get(k, PWR.DESIGN_KEY_DEFAULTS.get(k)) for k in PWR.DESIGN_KEYS})
    except PWR.PrivateWeightRefusal as exc:
        return ([g, str(len(formed)), seeds] + ["none"] * 4 + ["REFUSED", "none"],
                _one_line(f"load_replicates refused: {exc}"))
    cross = PWR.cross_run([this, *reps])
    lo, hi = cross.envelope
    invalid = [str(r.seed if r.seed is not None else r.name) for r in cross.readings
               if r.exit_code == exit_codes.INVALID]
    note = ("one run: the envelope is its own interval" if len(cross.readings) == 1
            else "an sd needs three points" if cross.sd is None else "")
    return ([g, str(len(cross.readings)), seeds, _fmt(statistics.mean(cross.points)),
             _fmt(cross.sd), _fmt(lo), _fmt(hi), cross.verdict,
             ("seed " + ", seed ".join(invalid)) if invalid else "none"], note or "none")


def pairs_readme(r1_duty: str = "none", r3_duty: str = "none") -> str:
    """PAIRS-README.txt: what every column of the three tables means, the
    interval widths read off the arms (R3's INTERVAL_PCT and ALPHA_BAND), and
    the duty states R1's word is a secant across and R3's duty, as
    PAIRS-fixed.tsv's r1_duty and r3_duty rows read them (pairs_table passes
    those values). The sentence on the chain's defaults names R1_DUTY's and
    R3_DUTY's defaults in scripts/alpha_g_chain.sh; a test holds it to them."""
    import private_weight_reference as PWR
    pct = f"{PWR.INTERVAL_PCT:.0f}%"
    band = f"[{PWR.ALPHA_BAND[0]}, {PWR.ALPHA_BAND[1]})"
    return f"""\
THE alpha(G) TABLES, rebuilt from the reports on disk by
scripts/alpha_g_chain_helpers.py pairs-table at the end of every chain pass and
before every STOP, and again on the laptop after exfil by the same command.

PAIRS.tsv, one row per ratio run (private_weight_reference), G then seed:
  G seed duty run_id   the swizzle the run pinned (GROUP_SIZE_M), its seed, its
                       duty and its run id.
  ratio lo hi          the run's OWN reading, slope(shared) / slope(private), and
                       its {pct} percentile bootstrap over repeats WITHIN the run
                       (R3's INTERVAL_PCT). It understates the run-to-run spread.
  exit                 the run's own exit word: classify over its page's gates.
  exit_scope           how C1 inside that word was scored: `alone` on this run's
                       interval; `envelope` on the envelope of this run's interval
                       and those of the earlier seeds it was given through
                       --replicate-of. Seed 0's exit and seed 1's can differ by
                       scope, not by result.
  rep_n rep_spread rep_sd env_lo env_hi joint
                       the joint reading on that run's page, over itself and those
                       earlier seeds: n, the spread and sd of the points (an sd
                       needs three), the envelope, and `joint`, C1's verdict on the
                       envelope against R3's refit band ALPHA_BAND {band}. `joint`
                       is NOT a quotability flag: NO-REUSE, expected at G=1, reads
                       FAIL. `none` on a run scored alone, and on a run whose page
                       formed no ratio of its own (its page's replicates block is
                       then the other runs' reading, not one this run is in).
  clk_<arm> low_cells  each arm's median under-load clock (MHz) over the ladder, and
                       the (arm, tread) cells whose level record reads LEVEL LOW.
  eta eta_lo eta_hi band eta_exit
                       R1 (clock_elasticity) at this G: the per-M-tile elasticity
                       over treads 2 and deeper, its 95% percentile bootstrap
                       (clock_elasticity.fit's 2.5th and 97.5th percentiles), the
                       regime word off that interval, and R1's own exit word. The
                       word is withheld (`withheld:<EXIT>`) from a page whose gates
                       did not stand behind it. It is a SECANT across R1's duty
                       states, not a local reading at R3's duty. This session's R1
                       states: {r1_duty}; R3's duty: {r3_duty} (PAIRS-fixed.tsv's
                       r1_duty and r3_duty). At the chain's defaults, R1_DUTY
                       1.0 0.5 0.25 and R3_DUTY 0.25, the secant runs from the
                       capped clock at 1.0 to the clocks at 0.5 and 0.25 (on
                       session 5's H200 only 1.0 held the 700 W cap), and R3's
                       0.25 is the top of that range.

PAIRS-by-G.tsv, one row per G, THE PER-G VALUE: every seed of that G whose report
formed a ratio, whatever order the seeds ran in, read together by R3's own
cross-run machinery (what `--read RUN --replicate-of ...` prints):
  n seeds              how many runs, and their seeds.
  mean sd              the points' mean, and their sd (three points or more).
  env_lo env_hi joint  the envelope of the runs' {pct} intervals and C1's verdict on
                       it, the same rule and band as `joint` above.
  invalid_in_envelope  the seeds inside the envelope whose own page exited INVALID:
                       R3 admits them by design, their spread is information and
                       their ratio is not quotable alone.
  eta .. eta_exit      R1 at this G, as above.
  note                 why a G has no joint (no report formed a ratio, or
                       load_replicates refused the set, e.g. two duties).

PAIRS-fixed.tsv: the coordinates every row shares (model, tile, pinned config,
treads, repeats, duty), their value and where each was read; MIXED when the
reports disagree.
"""


def _log(session: str | Path, step: str) -> Path:
    return Path(session) / "chain-logs" / f"{step}.log"


def _report(results: str | Path, log: Path, experiment: str) -> Path | None:
    rid = run_id(log, experiment)
    if not rid:
        return None
    path = Path(results) / experiment / rid / "report.json"
    return path if path.is_file() else None


def eta_for(session: str | Path, results: str | Path, g: str) -> list[str]:
    """`eta` for the G's R1 run, found through the run id its log printed, or
    the `unmeasured` row when no R1 report for that G is on disk."""
    rep = _report(results, _log(session, f"r1-g{g}"), "clock_elasticity")
    return eta(rep) if rep is not None else list(ETA_UNMEASURED)


def gate(report: str | Path, tag: str) -> str:
    """The verdict of the gate `tag` on a report, `absent` when the report
    scored no such gate, `unreadable` when there is no report to read."""
    p = _load(report)
    if p is None:
        return "unreadable"
    for g in p.get("gates") or []:
        if isinstance(g, dict) and (g.get("tag") or g.get("number")) == tag:
            return str(g.get("verdict"))
    return "absent"


def probe_note(report: str | Path) -> str:
    """The alignment probe's own record off an R3 report, for the V8 STOP:
    its note (a refused capture is named there), the `graph_calls` its cells
    were timed with (0 is eager) and how many cells the instrument called
    host-bound. '' when the report carries no probe."""
    p = _load(report)
    probe = (p or {}).get("align_probe")
    if not isinstance(probe, dict):
        return ""
    cells = [c for c in probe.get("cells") or [] if isinstance(c, dict)]
    parts = [f"note: {probe.get('note') or 'none'}"]
    if cells:
        calls = sorted({int(c.get("graph_calls") or 0) for c in cells})
        judged = [c for c in cells if c.get("host_bound") is not None]
        parts.append("graph_calls " + ", ".join(map(str, calls))
                     + (" (0 is eager)" if 0 in calls else ""))
        parts.append(f"host-bound {sum(1 for c in judged if c['host_bound'])} "
                     f"of {len(judged)} judged cells")
    return _one_line("; ".join(parts))


def verdict(log: str) -> str:
    """What the log's RESULT lines imply, as an exit code, or NONE / UNREADABLE."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return "UNREADABLE"
    try:
        return str(exit_codes.classify_text(text))
    except exit_codes.NoGatesScored:
        return "NONE"


def _render(value) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _fixed(key: str, payloads: list[tuple[str, dict]], field: str, drop=(),
           fallback: str | None = None) -> list[str]:
    """One PAIRS-fixed row: the field's value across every report that carries
    it, MIXED when they disagree, the chain's own command line when none does."""
    seen: dict[str, list[str]] = {}
    for step, p in payloads:
        if field not in p:
            continue
        value = p[field]
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in drop}
        seen.setdefault(_render(value), []).append(step)
    if not seen:
        if fallback is None:
            return [key, "none", "no report carries it yet"]
        return [key, fallback, "the chain's command line; no report carries it"]
    if len(seen) == 1:
        (value, steps), = seen.items()
        return [key, value, f"report.json of {len(steps)} run(s)"]
    return [key, "MIXED: " + " | ".join(seen),
            "; ".join(f"{v}: {', '.join(s)}" for v, s in seen.items())]


def pairs_table(session: str | Path, results: str | Path, ladder: str, seeds: str,
                settings: dict[str, str] | None = None) -> int:
    """REWRITE PAIRS.tsv, PAIRS-by-G.tsv and PAIRS-fixed.tsv from the reports
    on disk, and PAIRS-README.txt, the legend, beside them.

    Rebuilt, never appended: the chain calls this at the end of every pass and
    before every STOP, so a step re-run on --resume cannot leave a second row
    for one run id, and an R1 that finished on a later pass is joined to every
    row of its G. PAIRS.tsv: one row per (G, seed) whose ratio report is on
    disk, in ladder order; the R1 columns are `eta` for the G. PAIRS-by-G.tsv:
    one row per G of the ladder, every seed read together (`by_g`). The
    coordinates every row shares (model, tile, pinned config, treads,
    repeats, duty) go in the sidecar with where each came from, and the legend
    names the sidecar's r1_duty and r3_duty. Returns the number of PAIRS.tsv
    rows.
    """
    session, settings = Path(session), dict(settings or {})
    rows, by_g_rows, r3_payloads, r1_payloads = [], [], [], []
    for g in ladder.split():
        eta_cols = eta_for(session, results, g)
        found = []
        for seed in seeds.split():
            step = f"r3-g{g}-s{seed}"
            rep = _report(results, _log(session, step), "private_weight_reference")
            if rep is None:
                continue
            found.append(str(rep))
            payload = _load(rep)
            if payload is not None:
                r3_payloads.append((step, payload))
            rows.append(reading(rep) + eta_cols)
        r3_half, note = by_g(g, found)
        by_g_rows.append(r3_half + eta_cols + [note])
        r1 = _report(results, _log(session, f"r1-g{g}"), "clock_elasticity")
        if r1 is not None and (payload := _load(r1)) is not None:
            r1_payloads.append((f"r1-g{g}", payload))

    header = reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit"]
    _write_tsv(session / "PAIRS.tsv", [header, *rows])
    _write_tsv(session / "PAIRS-by-G.tsv", [BY_G_HEADER, *by_g_rows])
    fixed = [
        ["key", "value", "source"],
        _fixed("model", r3_payloads, "model", fallback=settings.get("model")),
        _fixed("block_m", r3_payloads, "block_m", fallback=settings.get("block_m")),
        _fixed("r3_pinned", r3_payloads, "pinned", drop=("GROUP_SIZE_M",)),
        ["group_m", "varies: " + ladder, "the G column; the swizzle each run pinned"],
        _fixed("r3_treads", r3_payloads, "treads", fallback=settings.get("r3_treads")),
        _fixed("r3_repeats", r3_payloads, "repeats", fallback=settings.get("r3_repeats")),
        _fixed("r3_duty", r3_payloads, "duty", fallback=settings.get("r3_duty")),
        _fixed("r1_pinned", r1_payloads, "pinned", drop=("GROUP_SIZE_M",)),
        _fixed("r1_duty", r1_payloads, "duty", fallback=settings.get("r1_duty")),
        # R1's report does not record its treads or repeats: the command line is the record
        ["r1_treads", settings.get("r1_treads", "none"), "the chain's command line"],
        ["r1_repeats", settings.get("r1_repeats", "none"), "the chain's command line"],
    ]
    _write_tsv(session / "PAIRS-fixed.tsv", fixed)
    # the legend names the duties the fixed rows read, without the flag a
    # value off the chain's command line carries
    duty = {r[0]: r[1].removeprefix("--duty ") for r in fixed[1:]}
    tmp = session / "PAIRS-README.txt.tmp"
    tmp.write_text(pairs_readme(duty["r1_duty"], duty["r3_duty"]))
    os.replace(tmp, session / "PAIRS-README.txt")
    return len(rows)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    """Whole-file and atomic: a reader never sees half a table."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join("\t".join(_one_line(c) for c in r) + "\n" for r in rows))
    os.replace(tmp, path)


def calibration_dir(log: str | Path) -> str:
    """The run directory of the newest ruler calibrate_hardware.py wrote, off
    its own `[calibrate] wrote <yaml>` line, or ''."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return ""
    found = CALIBRATE_WROTE.findall(text)
    return str(Path(found[-1]).parent) if found else ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "card" and not rest:
        print("\t".join(card()))
    elif cmd == "device" and not rest:
        print(device())
    elif cmd == "run-id" and len(rest) == 2:
        print(run_id(rest[0], rest[1]))
    elif cmd == "estimate" and len(rest) == 1:
        print(estimate(rest[0]))
    elif cmd == "estimate-basis" and len(rest) == 1:
        print(estimate_basis(rest[0]))
    elif cmd == "reading" and len(rest) == 1:
        print("\t".join(reading(rest[0])))
    elif cmd == "pairs":
        print("\n".join(pairs(rest)))
    elif cmd == "eta" and len(rest) == 1:
        print("\t".join(eta(rest[0])))
    elif cmd == "eta-for" and len(rest) == 3:
        print("\t".join(eta_for(*rest)))
    elif cmd == "gate" and len(rest) == 2:
        print(gate(rest[0], rest[1]))
    elif cmd == "probe-note" and len(rest) == 1:
        print(probe_note(rest[0]))
    elif cmd == "verdict" and len(rest) == 1:
        print(verdict(rest[0]))
    elif cmd == "pairs-table" and len(rest) >= 4:
        settings = dict(kv.split("=", 1) for kv in rest[4:] if "=" in kv)
        n = pairs_table(rest[0], rest[1], rest[2], rest[3], settings)
        print(f"{n} row(s)")
    elif cmd == "calibration-dir" and len(rest) == 1:
        print(calibration_dir(rest[0]))
    else:
        print(f"unknown command {argv!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
