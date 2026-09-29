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


@pytest.fixture(scope="module")
def qwen():
    return S.score(_pages(S27, [("deep", 2), ("p5", 3), ("p2", 32), ("deep", 4), ("p2", 8)]),
                   S27 / "results" / "2026-09-27-nvidia_gh200_480gb-r3-counters" / "lock1710",
                   "qwen2-57b-a14b",
                   _pages(SQ, [("p5", 3), ("p2", 8), ("p2b", 32), ("deep", 4)]),
                   SQ / "results" / "2026-09-28-nvidia_gh200_480gb-r3-counters" / "lock1710")


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
        [*map(str, _pages(S27, [("deep", 2), ("p5", 3), ("p2", 32), ("deep", 4), ("p2", 8)])),
         "--counters", str(S27 / "results" / "2026-09-27-nvidia_gh200_480gb-r3-counters"
                           / "lock1710")]))
    assert qwen["timing_params"] == src["main"].params
