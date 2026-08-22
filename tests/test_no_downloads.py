"""The benchmark path must never download anything.

A sweep generates random weights for one MoE layer; only
`scripts/capture_traces.py` pulls a real checkpoint. That separation is what
makes a 256-expert model cost 27 GB of device memory and zero disk, and it is
easy to break by accident: one `from_pretrained` in a baseline shim and a
sweep starts pulling 90 GB mid-session on a metered box.

These tests pin it three ways: no source references, no imported modules, and
no network-capable modules loaded after a real cell runs.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "moe"

# Names that can reach the network or a model hub.
FORBIDDEN_SOURCE = (
    "transformers", "huggingface_hub", "from_pretrained", "snapshot_download",
    "hf_hub_download", "AutoModel", "AutoTokenizer", "datasets",
    "urllib.request", "requests.get",
)


def test_no_download_capable_references_in_the_package():
    """`hf_repo` is allowed: it is a metadata string recording provenance, not
    a fetch. Anything that could actually retrieve a model is not."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text()
        for name in FORBIDDEN_SOURCE:
            if name in text:
                offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert not offenders, (
        "the benchmark package must not be able to download:\n  "
        + "\n  ".join(offenders))


def test_capture_script_is_the_only_place_that_loads_a_model():
    hits = [p.relative_to(ROOT) for p in sorted((ROOT / "scripts").rglob("*.py"))
            if "from_pretrained" in p.read_text()]
    assert hits == [pathlib.Path("scripts/capture_traces.py")], hits


def test_running_a_real_cell_loads_no_hub_modules():
    """The airtight version: a fresh interpreter runs an actual benchmark cell
    end to end, then reports whether anything hub-shaped got imported."""
    script = """
import sys, json
import moe
moe.bootstrap("reference", "kernels", "baselines")
from moe import pipeline as P
from moe.bench import driver as D
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec

spec = BenchSpec(MODEL_CONFIGS["toy"], 32, "fp32", RoutingSpec("zipf", 1.2))
x, w = R.make_inputs(spec, device="cpu")
pipe = P.build(P.reference_pipeline_names(), spec=spec)
result, st, golden, tol = D.check_correctness(spec, pipe, x, w, None,
                                              D.RunConfig(device="cpu"))
bad = sorted(m for m in sys.modules
             if m.split(".")[0] in {"transformers", "huggingface_hub",
                                    "datasets", "requests", "urllib3"})
print(json.dumps({"passed": bool(result.passed), "hub_modules": bad}))
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    import json
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["passed"], "the cell itself should have passed the oracle"
    assert out["hub_modules"] == [], (
        f"running a cell imported {out['hub_modules']}; a sweep must not be "
        "able to reach a model hub")


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_run_all_forces_offline_for_the_sweep(var):
    """Belt and braces: even if something did try to fetch, the sweep runs with
    the hub disabled so it fails loudly in seconds instead of quietly pulling
    90 GB. capture_traces.py is invoked separately and is not affected."""
    text = (ROOT / "scripts/run_all.sh").read_text()
    assert f"export {var}=1" in text, f"{var} should be exported before the sweep"
