"""The suite on a GPU box: the `no_gpu` marker mirrors `gpu`, the results root
is sandboxed, and the two are registered where `--strict-markers` can see them.

Session 4's pod pytest failed 23 tests that asserted the laptop path on a box
with a card, wrote under the session's real results root through an inherited
MOE_RESULTS_DIR, and started a measurement from a `--run` inside a test.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import conftest as CF  # noqa: E402


class _Item:
    def __init__(self, keywords, nodeid="x"):
        self.keywords = set(keywords)
        self.nodeid = nodeid
        self.marks = []

    def add_marker(self, mark):
        self.marks.append(mark)


def _skips(item):
    return [m.kwargs.get("reason", "") for m in item.marks]


@pytest.mark.parametrize("has_cuda", [True, False])
def test_a_no_gpu_test_is_skipped_exactly_when_a_device_is_attached(monkeypatch, has_cuda):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: has_cuda)
    gpu, no_gpu, plain = _Item({"gpu"}), _Item({"no_gpu"}), _Item({"other"})
    CF.pytest_collection_modifyitems(None, [gpu, no_gpu, plain])
    assert bool(_skips(gpu)) == (not has_cuda)
    assert bool(_skips(no_gpu)) == has_cuda
    if has_cuda:
        assert "laptop path" in _skips(no_gpu)[0]
    assert _skips(plain) == []
    with pytest.raises(pytest.UsageError, match="both gpu and no_gpu"):
        CF.pytest_collection_modifyitems(None, [_Item({"gpu", "no_gpu"}, "t::both")])


def test_both_markers_are_registered_in_pyproject_and_strict():
    opts = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    markers = opts["markers"]
    assert any(m.startswith("gpu:") for m in markers)
    assert any(m.startswith("no_gpu:") for m in markers)
    assert "--strict-markers" in opts["addopts"]


def test_the_shared_no_cuda_plant_hides_the_device_from_every_detector(no_cuda):
    from moe.bench import roofline, timing
    assert torch.cuda.is_available() is False
    assert roofline.current_gpu_name() == ""
    with pytest.raises(Exception, match="CUDA|cuda"):
        timing.require_cuda()


def test_the_shared_card_plant_shows_every_detector_the_same_h200(an_h200):
    """The mirror of the plant above: one fixture, and every detector in the
    repo reads the same card from it, so a test can walk the pod's detection
    path on a laptop. Session 5's pod suite failed ten tests on paths no laptop
    had walked."""
    from moe.bench import provenance, roofline
    assert torch.cuda.is_available() is True
    assert roofline.current_gpu_name() == an_h200.name
    name, _, reason = provenance._gpu(torch)
    assert name == an_h200.name and reason is None
    sys.path.insert(0, str(ROOT / "scripts"))
    import block_m_crossing_sweep as SWEEP
    assert SWEEP.detect_card_slug() == "nvidia_h200"


def test_the_suite_sandboxes_its_results_root_inside_the_repo():
    root = Path(os.environ["MOE_RESULTS_DIR"])
    assert root.is_relative_to(ROOT / "results" / "_pytest"), root
    assert root.is_dir()
    got = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(root)])
    assert got.returncode == 0, (
        "the sandbox must be git-ignored, or every 'IGNORED by git' line lies")


def test_the_hermetic_child_environment_hides_cuda_and_pins_the_interpreters():
    from _hermetic import LAPTOP_ENV, laptop_env
    assert LAPTOP_ENV["CUDA_VISIBLE_DEVICES"] == ""
    assert LAPTOP_ENV["PY_BASE"] == LAPTOP_ENV["PY_VLLM"] == sys.executable
    env = laptop_env(EXTRA="1")
    assert env["PATH"] == os.environ["PATH"] and env["EXTRA"] == "1"
    child = subprocess.run([sys.executable, "-c",
                            "import torch; print(torch.cuda.is_available())"],
                           capture_output=True, text=True, env=env)
    assert child.stdout.strip() == "False"
