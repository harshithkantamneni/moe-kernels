"""scripts/cross_model_score.py: another model's timed pages priced with one
card's fitted timing parameters and the target's own counted bytes."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cross_model_score as S  # noqa: E402
import r3_timing_model as TM  # noqa: E402

PUB = ROOT / "results" / "published"
S27 = PUB / "2026-09-27-nvidia_gh200_480gb-session"
SQ = PUB / "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session"


def _pages(session, keep):
    out = []
    for rep in sorted(session.rglob("private_weight_reference/*/report.json")):
        j = json.loads(rep.read_text())
        if any(j["session_tag"].endswith(f"-{t}-lock1710") and j["pinned"]["GROUP_SIZE_M"] == g
               for t, g in keep):
            out.append(rep.parent)
    return out


SO = PUB / "2026-09-29-nvidia_gh200_480gb-olmoe-session"
SRC = [("deep", 2), ("p5", 3), ("p2", 32), ("deep", 4), ("p2", 8)]
C27 = S27 / "results" / "2026-09-27-nvidia_gh200_480gb-r3-counters" / "lock1710"


def _qwen(no_cta_fixed):
    return S.score(_pages(S27, SRC), C27, "qwen2-57b-a14b",
                   _pages(SQ, [("p5", 3), ("p2", 8), ("p2b", 32), ("deep", 4)]),
                   SQ / "results" / "2026-09-28-nvidia_gh200_480gb-r3-counters" / "lock1710",
                   no_cta_fixed=no_cta_fixed)


@pytest.fixture(scope="module")
def qwen():
    """The 2026-09-29 pins, before the per-CTA fixed cost."""
    return _qwen(True)


def test_qwen_with_its_own_bytes_is_the_timing_structures_score(qwen):
    """Qwen2-57B's four VALID lock-1710 pages, 8x7B's parameters held: 58
    SHARED+PRIVATE cells at 1.52% rms (diagnosis data since 2026-09-29: the
    dead-CTA term was chosen on these pages), none beyond 5%."""
    s = qwen["sets"]["shared+private"]
    assert s["cells"] == 58 and s["beyond_5pct"] == 0
    assert s["rms"] == pytest.approx(0.0152, abs=0.0005)
    assert qwen["source_model"] == "mixtral-8x7b" and qwen["target"] == "qwen2-57b-a14b"
    assert TM.MODEL == "mixtral-8x7b", "the scorer restores the timing model's shapes"


def test_nothing_is_fitted_on_the_target(qwen):
    """Every cell is priced with the source fit's parameters: a cell's predicted
    time moves with them, and the reported parameters are the source's."""
    src = TM.build(TM.build_parser().parse_args(
        [*map(str, _pages(S27, SRC)), "--counters", str(C27), "--no-cta-fixed"]))
    assert qwen["timing_params"] == src["main"].params


def _slope(rows, key, G):
    import numpy as np
    d = {r["n"]: r[key] for r in rows if r["arm"] == "shared" and r["G"] == G}
    return float(np.polyfit([2, 3, 4, 5, 6], [d[n] for n in (2, 3, 4, 5, 6)], 1)[0])


def test_the_per_cta_fixed_cost_closes_qwens_slope():
    """With CTA_FIXED (PHI measured on four models' counters, Qwen2-57B's among
    them, so this is diagnosis): 58 cells 1.52% -> 1.06% rms, and the G >= 8
    SHARED slope 0.6838 -> 0.6982 ms per tread against 0.6962 measured."""
    d = _qwen(False)
    s = d["sets"]["shared+private"]
    assert s["cells"] == 58 and s["beyond_5pct"] == 0
    assert s["rms"] == pytest.approx(0.0106, abs=0.0005)
    assert d["cta_fixed_ksteps"] == TM.CTA_FIXED_KSTEPS
    assert _slope(d["cells"], "predicted_ms", 8) == pytest.approx(0.6982, abs=0.0003)
    assert _slope(d["cells"], "measured_ms", 8) == pytest.approx(0.6962, abs=0.0003)


@pytest.mark.parametrize("no_cta_fixed, rms, beyond, slope", [
    (True, 0.0353, 12, 0.1597), (False, 0.0193, 0, 0.1705)])
def test_olmoe_with_and_without_the_per_cta_fixed_cost(no_cta_fixed, rms, beyond, slope):
    """OLMoE's five VALID pages, 76 cells: the registered test's 3.53% and 12
    beyond 5% (slope 0.1597 against 0.1705 measured) without the term; 1.93% and
    none with it (slope 0.1705). Diagnosis: OLMoE's counters are in PHI."""
    if not SO.exists():
        pytest.skip("the OLMoE session is not in this tree")
    d = S.score(_pages(S27, SRC), C27, "olmoe-1b-7b",
                _pages(SO, [("deep", 2), ("p5", 3), ("p2", 32), ("deep", 4), ("p2", 8)]),
                SO / "results" / "2026-09-29-nvidia_gh200_480gb-r3-counters" / "lock1710",
                no_cta_fixed=no_cta_fixed)
    s = d["sets"]["shared+private"]
    assert s["cells"] == 76 and s["beyond_5pct"] == beyond
    assert s["rms"] == pytest.approx(rms, abs=0.0005)
    assert _slope(d["cells"], "predicted_ms", 8) == pytest.approx(slope, abs=0.0003)
    assert _slope(d["cells"], "measured_ms", 8) == pytest.approx(0.1705, abs=0.0003)
    assert TM.CTA_FIXED, "the scorer restores CTA_FIXED"
