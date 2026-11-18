"""Model v7's gate result as published: the registered gate's official output (gate_v7.py run 2026-11-18 at 6ed4c57d
after the registration was public) is committed byte for byte and unchanged, and reads R3 GO, PREFILL NO-GO
(REFUSED: no module named 'refit4'), DECODE GO. The DESCRIPTIVE prefill re-run (prefill_rerun_v7.py, post-registration,
not a verdict) reproduces its committed output (labels exact, floats within tests/_reproduce.py's tolerance), quotes
the registered PREFILL line as standing, and refuses when the committed gate output is not the registered run's."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V7DIR = REPO / "scripts" / "scoring" / "v7"
for p in (REPO / "tests", REPO, REPO / "scripts", V7DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _reproduce as RT  # noqa: E402
import prefill_rerun_v7 as PR  # noqa: E402
import round5_chain as R5  # noqa: E402

#: the official run's outputs (moe-kernels-work 2026-11-18 gate-v7 gate.txt and gate.json), recorded at publication
GATE_TXT_SHA = "882c749163b1b622b4a077ea0e3db2750cb014a4cf8cfc5b3a8a1fae68f56df7"
GATE_JSON_SHA = "db84baf428bb1a23f1d27170d79a8a97ce95d2faf564ec0450994a8b42364670"
REFUSAL = "PREFILL: NO-GO: REFUSED: ModuleNotFoundError: No module named 'refit4': no blind for this regime"

PUB = REPO / "results" / "published"
HAVE = all((PUB / s / "results").is_dir() for s in R5.RUNS.values())
need = pytest.mark.skipif(not HAVE, reason="the A2, S1, C2 or C4 session is not published in this checkout")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_official_gate_output_is_committed_and_unchanged():
    assert _sha(V7DIR / "gate_v7.out.txt") == GATE_TXT_SHA
    assert _sha(V7DIR / "gate_v7.json") == GATE_JSON_SHA
    assert (PR.GATE_TXT_SHA, PR.GATE_JSON_SHA) == (GATE_TXT_SHA, GATE_JSON_SHA)


def test_the_official_gate_output_reads_r3_go_prefill_refused_decode_go():
    assert (V7DIR / "gate_v7.out.txt").read_text().splitlines() == [
        "R3: GO: this regime may go to ONE blind", REFUSAL, "DECODE: GO: this regime may go to ONE blind"]
    d = json.loads((V7DIR / "gate_v7.json").read_text())
    assert d["registration"] == "2026-11-17-v7-gate-gh200" and d["freeze"] == {"verdict": "HOLDS", "files": 148}
    assert d["registration_sha256"] == _sha(REPO / "docs" / "registered" / "2026-11-17-v7-gate-gh200.json")
    assert d["r3"]["verdict"] == "GO" and d["r3"]["G-R3"]["compared"] == 300
    assert d["prefill"] == {"verdict": "NO-GO", "why": ["REFUSED: ModuleNotFoundError: No module named 'refit4'"]}
    assert d["decode"]["verdict"] == "GO"
    assert d["decode"]["CAL"]["E1_CAL_IN_SAMPLE"] == {"cells": 91, "rms": 0.016032, "worst": -0.053955, "mean": -0.001724}
    assert d["decode"]["RE-SCORE"]["pooled"]["cells"] == 220 and d["decode"]["RE-SCORE"]["pooled"]["rms"] == 0.031928


def test_the_rerun_refuses_when_the_committed_gate_output_is_not_the_registered_runs(tmp_path, monkeypatch):
    assert PR.registered_prefill_line() == REFUSAL
    bad = tmp_path / "gate_v7.out.txt"
    bad.write_text((V7DIR / "gate_v7.out.txt").read_text().replace("REFUSED", "GO"))
    monkeypatch.setattr(PR, "GATE_TXT", bad)
    with pytest.raises(PR.V7.Refused):
        PR.registered_prefill_line()


@need
def test_the_descriptive_prefill_rerun_reproduces_its_committed_output(tmp_path):
    r = subprocess.run([sys.executable, str(V7DIR / "prefill_rerun_v7.py"), "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    for name, kind in (("prefill_rerun_v7.json", "json"), ("prefill_rerun_v7.txt", "txt")):
        RT.assert_reproduces((tmp_path / name).read_text(), (V7DIR / name).read_text(), kind, what=name)
    d = json.loads((V7DIR / "prefill_rerun_v7.json").read_text())
    assert d["label"] == PR.LABEL and d["label"].startswith("DESCRIPTIVE, post-registration, NOT A VERDICT")
    assert d["registered_verdict (stands)"] == REFUSAL
    assert "verdict" not in d["prefill"] and d["prefill"]["reads (DESCRIPTIVE, not a verdict)"] == "NO-GO"
    off = d["prefill"]["ablation (DESCRIPTIVE)"]["off (round 5)"]
    # with every switch off the re-run is round 5: its table and its five-model estimate as B3 froze them
    assert {k: off["G1"]["means_pct"][k] for k in R5.ROUND5_MEANS} == R5.ROUND5_MEANS
    assert off["G1"]["means_pct"][R5.M_C2] == R5.ROUND5_M_C2 and off["G2"]["P_A"] == R5.ROUND5_PASS_A
