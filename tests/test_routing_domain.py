"""Pooling routing regimes into one crossing must not be possible silently.

`AI = 2R/b` describes UNIFORM routing. `R = T*k/E` is a mean, and under skew no
expert experiences the mean: the busy experts are compute-bound while the quiet
ones are still memory-bound AT THE SAME BATCH, so the layer straddles the ridge
and there is no single crossing to find. Uniform is 1344 of the 9408 cells in
each cross-card arm, 14.3%; the other 86% are outside the model's domain.

C5 was read off pooled rows for three days and the pooling was invisible, so
these tests pin the two demonstrations that pooling is INVALID rather than
merely noisy, and then pin that both reports say so out loud:

  (a) the cross-card ratio moves 4.3x across routings both cards ran identically,
      on a quantity whose two candidate answers are 0.83 and 0.82;
  (b) the pooled deepseek-v3 crossing moves 238x depending on the saturation
      floor, where the uniform one does not move at all.

The script tests also pin what the guard deliberately does NOT do: it never
changes the default filter. A report that answers differently than it did
yesterday with no flag changed is its own failure mode, so it warns and names
`--routing uniform` instead of applying it.
"""
from __future__ import annotations

import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from moe.bench.crossing import (
    UNIFORM_ROUTING,
    RoutingDomain,
    crossing_from_points,
    routing_domain,
    timed_rows,
)
from moe.bench.ridge import saturation_batch
from moe.bench.schema import COLUMNS, UNRECORDED

REPO = Path(__file__).resolve().parent.parent
A100 = REPO / "results/published/2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card"
H200 = REPO / "results/published/2026-08-28-nvidia_h200-h200-whole-layer"

needs_published = pytest.mark.skipif(
    not (A100.exists() and H200.exists()),
    reason="published cross-card arms are not in this checkout")


def rows(*kinds: str) -> list[dict]:
    """One row per named kind, carrying nothing but the routing column.

    The guard reads one column, so a fixture that builds whole rows would be
    asserting things about the schema instead of about the guard.
    """
    return [{"routing_kind": k} for k in kinds]


# --------------------------------------------------------------------------
# The guard itself
# --------------------------------------------------------------------------

def test_uniform_only_rows_are_inside_the_domain_and_say_nothing():
    dom = routing_domain(rows("uniform", "uniform", "uniform"))
    assert dom.inside and not dom.mixed
    assert dom.kinds == (UNIFORM_ROUTING,)
    assert dom.uniform_fraction == 1.0
    assert dom.warning_lines() == []
    assert dom.crossing_note() == []


def test_pooled_routing_kinds_are_called_invalid_and_not_merely_noisy():
    """The distinction the whole guard exists for. Noise is something a band
    covers; a blend of regimes is not a crossing at all, so a band around it
    would be a resolution claim on a number that measures nothing."""
    dom = routing_domain(rows("uniform", "zipf", "hot", "dirichlet"))
    assert dom.mixed and not dom.inside
    banner = "\n".join(dom.warning_lines())
    assert "INVALID" in banner and "not merely noisy" in banner
    assert "straddles the ridge" in banner


def test_one_skewed_kind_is_outside_the_domain_even_though_nothing_is_pooled():
    """`--routing zipf` pools nothing and is still not what `2R/b` predicts, so
    a guard keyed only on mixing would wave it through."""
    dom = routing_domain(rows("zipf", "zipf"))
    assert not dom.mixed and not dom.inside
    banner = "\n".join(dom.warning_lines())
    assert "OUTSIDE THE DOMAIN" in banner
    assert "no uniform rows" in banner


def test_the_census_says_how_many_rows_each_kind_contributed():
    """Counts, not a set: "one stray zipf row" and "six sevenths of the input"
    are the same set, and only one of them is worth stopping for."""
    dom = routing_domain(rows(*(["uniform"] * 99 + ["zipf"])))
    assert dom.census == "uniform x99, zipf x1"
    assert dom.total == 100
    assert dom.uniform_fraction == pytest.approx(0.99)


def test_a_blank_routing_kind_is_unrecorded_and_never_counted_as_uniform():
    """A row that does not say which regime it ran is exactly the case this
    guard exists to catch, so reading a blank as the in-domain value would
    defeat it. Missing key and empty string both land on the sentinel."""
    dom = routing_domain([{"routing_kind": ""}, {}, {"routing_kind": "  "}])
    assert dom.kinds == (UNRECORDED,)
    assert not dom.inside
    assert dom.uniform_rows == 0
    assert dom.warning_lines()


def test_an_empty_selection_is_not_inside_the_domain_and_warns_about_nothing():
    """`inside` is a claim about rows that exist. With none, the caller already
    prints "nothing to report" and a routing banner would be noise."""
    dom = routing_domain([])
    assert not dom.inside and dom.total == 0
    assert dom.uniform_fraction == 0.0
    assert dom.warning_lines() == [] and dom.crossing_note() == []


def test_every_warning_names_the_flag_that_fixes_it():
    """Both scripts spell it the same way, and a warning that does not say what
    to do instead gets read as unavoidable."""
    for kinds in (("uniform", "zipf"), ("zipf",), ("",)):
        dom = routing_domain(rows(*kinds))
        assert any("--routing uniform" in ln for ln in dom.warning_lines())
        assert any("--routing uniform" in ln for ln in dom.crossing_note())


def test_the_note_beside_the_number_says_what_the_number_is_not():
    """A header banner does not travel with a figure that gets quoted on its
    own, and every crossing this study retracted was quoted on its own."""
    note = "\n".join(routing_domain(rows("uniform", "zipf", "hot")).crossing_note())
    assert "POOLED OVER 3 ROUTING KINDS" in note
    assert "not a crossing" in note
    assert "not a noisy estimate of one" in note


def test_the_domain_can_be_built_from_counts_without_replaying_the_rows():
    """`RoutingDomain` is a value, so an aggregator that already counted does
    not have to hold 70k row dicts alive to ask the question again."""
    dom = RoutingDomain({"uniform": 1344, "zipf": 4032})
    assert dom.uniform_fraction == pytest.approx(1344 / 5376)
    assert dom.mixed and not dom.inside


# --------------------------------------------------------------------------
# (a) and (b), from the published rows
# --------------------------------------------------------------------------

def published_rows(arm: Path) -> list[dict]:
    """Every timed, correct, unthrottled `vllm_fused_experts` bf16 row.

    The per-run files rather than `merged.csv`, which is their concatenation:
    reading both would weight every row twice, which is the bug
    `moe.bench.published` exists for.
    """
    out = []
    for path in sorted(arm.glob("*.csv")):
        if path.name == "merged.csv":
            continue
        with path.open(newline="") as fh:
            for r in timed_rows(list(csv.DictReader(fh))):
                if r["impl"] != "vllm_fused_experts" or r["dtype"] != "bf16":
                    continue
                if r.get("correctness_passed") not in ("True", "true", "1", ""):
                    continue
                if r.get("throttled") in ("True", "true", "1"):
                    continue
                out.append(r)
    return out


def curve(rs: list[dict], model: str, kind: str | None) -> list[tuple[float, float]]:
    """One median per token count, the way `crossing_report` aggregates."""
    by_t: dict[int, list[float]] = {}
    for r in rs:
        if r["model"] != model:
            continue
        if kind is not None and r["routing_kind"] != kind:
            continue
        by_t.setdefault(int(r["num_tokens"]), []).append(float(r["ms_p50"]))
    return [(t, statistics.median(v)) for t, v in sorted(by_t.items())]


@needs_published
def test_both_cards_ran_the_identical_seven_routing_distributions():
    """The premise of demonstration (a). If the two arms had swept different
    distributions, a ratio that moved with routing would be trivially explained
    and would say nothing about pooling."""
    def census(arm: Path) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        with (arm / "merged.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["routing_kind"], f"{float(r['routing_param']):g}")
                counts[key] = counts.get(key, 0) + 1
        return counts

    a, h = census(A100), census(H200)
    assert a == h
    assert len(a) == 7 and set(a.values()) == {1344}


@needs_published
def test_uniform_is_one_seventh_of_the_cross_card_cells():
    """14.3% in the domain, 86% outside it. The number in the guard's docstring
    and in the C5 section of docs/FINDINGS.md."""
    for arm in (A100, H200):
        with (arm / "merged.csv").open(newline="") as fh:
            dom = routing_domain(csv.DictReader(fh))
        assert dom.total == 9408
        assert dom.uniform_rows == 1344
        assert dom.uniform_fraction == pytest.approx(0.143, abs=0.001)
        assert dom.mixed and not dom.inside


@needs_published
def test_the_cross_card_ratio_swings_4x_across_routings_both_cards_ran():
    """DEMONSTRATION (a). mixtral-8x7b, `vllm_fused_experts` bf16, one card
    each, the crossing ratio computed per routing kind.

    Same hardware pair, same seven distributions, same filter. Only the regime
    changes, and the answer moves by more than the entire question."""
    a100, h200 = published_rows(A100), published_rows(H200)
    floor = saturation_batch("mixtral-8x7b")

    def ratio(kind: str | None) -> float:
        a = crossing_from_points(curve(a100, "mixtral-8x7b", kind), min_tokens=floor)
        h = crossing_from_points(curve(h200, "mixtral-8x7b", kind), min_tokens=floor)
        assert a is not None and h is not None
        return a / h

    by_kind = {k: ratio(k) for k in ("uniform", "zipf", "hot", "dirichlet")}
    assert by_kind["uniform"] == pytest.approx(0.73, abs=0.01)
    assert by_kind["zipf"] == pytest.approx(0.44, abs=0.01)
    assert by_kind["hot"] == pytest.approx(0.46, abs=0.01)
    assert by_kind["dirichlet"] == pytest.approx(1.92, abs=0.01)
    spread = max(by_kind.values()) / min(by_kind.values())
    assert spread == pytest.approx(4.3, abs=0.1)


@needs_published
def test_the_routing_spread_dwarfs_the_gap_between_the_two_hypotheses():
    """Why (a) is fatal rather than untidy. The cross-card ratio is asked to
    decide between ridge scaling (0.83) and the SM-count rival (0.82), two
    values 0.01 apart. Routing choice moves the same number by 1.48, more than a
    hundred times the gap it is meant to resolve, so the measurement answers the
    analyst's flag and not the hardware."""
    a100, h200 = published_rows(A100), published_rows(H200)
    floor = saturation_batch("mixtral-8x7b")
    vals = []
    for kind in ("uniform", "zipf", "hot", "dirichlet"):
        a = crossing_from_points(curve(a100, "mixtral-8x7b", kind), min_tokens=floor)
        h = crossing_from_points(curve(h200, "mixtral-8x7b", kind), min_tokens=floor)
        vals.append(a / h)
    hypotheses = 0.83 - 0.82
    assert (max(vals) - min(vals)) > 100 * hypotheses


@needs_published
def test_pooling_makes_the_deepseek_v3_crossing_swing_238x_under_the_floor():
    """DEMONSTRATION (b). The saturation floor discards points below `E/k`,
    where a batch does not yet reach every expert. On a curve that is genuinely
    weight-bound at small T that cut moves nothing, because those points are
    flat. The pooled curve is still STEEP there -- it is a blend of regimes, and
    the skewed ones are already compute-bound on their hot experts -- so the
    slope detector finds a crossing at 15 tokens and the floor is the only thing
    standing between the report and quoting it."""
    pooled = curve(published_rows(H200), "deepseek-v3", None)
    floor = saturation_batch("deepseek-v3")
    with_floor = crossing_from_points(pooled, min_tokens=floor)
    without = crossing_from_points(pooled, min_tokens=0.0)
    assert with_floor == pytest.approx(3474, abs=2)
    assert without == pytest.approx(14.6, abs=0.1)
    assert with_floor / without == pytest.approx(238, abs=2)


@needs_published
def test_the_uniform_deepseek_v3_crossing_does_not_move_under_the_floor_at_all():
    """The control for (b), and the reason 238x is a verdict on pooling rather
    than on the floor. Same card, same kernel, same model, same detector: in
    the domain the floor changes the answer by nothing whatsoever."""
    uniform = curve(published_rows(H200), "deepseek-v3", "uniform")
    floor = saturation_batch("deepseek-v3")
    with_floor = crossing_from_points(uniform, min_tokens=floor)
    without = crossing_from_points(uniform, min_tokens=0.0)
    assert with_floor == pytest.approx(3010, abs=2)
    assert with_floor == without


# --------------------------------------------------------------------------
# The reports
# --------------------------------------------------------------------------

def write_csv(path: Path, rs: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rs:
            base = {c: "" for c in COLUMNS}
            base.update(r)
            w.writerow(base)


def row(kind: str, t: int, ms: float, impl: str = "vllm_fused_experts") -> dict:
    return {"impl": impl, "num_tokens": t, "ms_p50": f"{ms}",
            "model": "mixtral-8x7b", "dtype": "bf16", "routing_kind": kind,
            "routing_param": "0", "seed": "0", "l2_flush": "True",
            "cuda_graph": "False", "correctness_passed": "True",
            "throttled": "False", "covers": "all",
            "load_active_experts": "8", "load_total_rows": f"{2 * t}"}


#: T, then a uniform curve that goes flat before it climbs and a skewed one that
#: climbs from the start. Their MEDIAN crosses somewhere neither of them does,
#: which is the whole objection to pooling in eight numbers.
TOKENS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
UNIFORM_MS = (1.00, 1.02, 1.05, 1.10, 1.60, 3.00, 6.00, 12.00)
SKEWED_MS = (1.00, 1.40, 2.00, 2.90, 4.20, 6.10, 8.80, 12.80)


def mixed_csv(path: Path) -> Path:
    write_csv(path, [row("uniform", t, u) for t, u in zip(TOKENS, UNIFORM_MS, strict=True)]
              + [row("zipf", t, s) for t, s in zip(TOKENS, SKEWED_MS, strict=True)])
    return path


def report(*args: str) -> str:
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crossing_report.py"), *args],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


def measured(stdout: str) -> float:
    m = re.search(r"measured \(slope crosses 0\.5\):\s+([0-9.]+) tokens", stdout)
    assert m, stdout
    return float(m.group(1))


def test_crossing_report_warns_loudly_when_its_input_mixes_routing_kinds(tmp_path):
    out = report(str(mixed_csv(tmp_path / "mixed.csv")), "--ridge", "160.3")
    assert "ROUTING KINDS POOLED: uniform x8, zipf x8" in out
    assert "INVALID" in out
    assert "--routing uniform" in out


def test_crossing_report_repeats_the_warning_beside_the_crossing(tmp_path):
    """Prominence is about where, not just whether. The banner sits above four
    model sections in a real run, and the number is what gets copied out."""
    out = report(str(mixed_csv(tmp_path / "mixed.csv")), "--ridge", "160.3")
    lines = out.splitlines()
    idx = next(i for i, ln in enumerate(lines) if "measured (slope crosses" in ln)
    assert "POOLED OVER 2 ROUTING KINDS" in lines[idx + 1]


def test_crossing_report_does_not_silently_switch_to_the_uniform_rows(tmp_path):
    """The failure mode the warning is chosen INSTEAD of. Yesterday's command
    must still print yesterday's number, or an analysis quietly changes under
    an unchanged invocation. So the pooled default still pools, and it is still
    the median of the two curves rather than the uniform one."""
    p = mixed_csv(tmp_path / "mixed.csv")
    pooled = measured(report(str(p), "--ridge", "160.3"))
    uniform = measured(report(str(p), "--ridge", "160.3", "--routing", "uniform"))
    assert pooled != uniform

    expected = crossing_from_points(
        [(t, statistics.median([u, s]))
         for t, u, s in zip(TOKENS, UNIFORM_MS, SKEWED_MS, strict=True)],
        min_tokens=saturation_batch("mixtral-8x7b"))
    assert pooled == pytest.approx(expected, abs=1.0)


def test_crossing_report_says_nothing_extra_when_the_input_is_uniform_only(tmp_path):
    """The guard is silent in the domain, so the byte-for-byte output of every
    in-domain run this study has published is untouched."""
    out = report(str(mixed_csv(tmp_path / "mixed.csv")), "--ridge", "160.3",
                 "--routing", "uniform")
    assert "ROUTING" not in out
    assert "POOLED" not in out


def compare(*args: str) -> str:
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compare.py"), *args],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_compare_warns_when_routing_any_pools_regimes(tmp_path):
    """`--routing any` is the one flag that pools regimes into a single table,
    and its `regime` column is a mean rows/expert scored against the ridge."""
    mixed_csv(tmp_path / "run_mixed.csv")
    out = compare("--results", str(tmp_path), "--routing", "any")
    assert "ROUTING KINDS POOLED: uniform x8, zipf x8" in out
    assert "regime" in out and "MEAN rows/expert" in out


def test_compare_warns_on_a_single_skewed_kind_too(tmp_path):
    mixed_csv(tmp_path / "run_mixed.csv")
    out = compare("--results", str(tmp_path), "--routing", "zipf")
    assert "ROUTING OUTSIDE THE DOMAIN: zipf x8" in out


def test_compare_is_silent_on_its_uniform_default(tmp_path):
    mixed_csv(tmp_path / "run_mixed.csv")
    out = compare("--results", str(tmp_path))
    assert "ROUTING KINDS POOLED" not in out
    assert "OUTSIDE THE DOMAIN" not in out
