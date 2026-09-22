"""Shared fixtures and reference fusions used across the suite.

The two fused spans below are real torch implementations, not stubs. They exist
so that the fusion accounting in bytes_model and the numerical equivalence of
fused vs unfused tilings can both be tested on a laptop, without CUDA.
"""
import os
import pathlib
import shutil

import pytest
import torch

import moe
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import StageSpan, register
from moe.state import MoEState

moe.bootstrap("reference")


class _RefFused(StageSpan):
    requires_cuda = False
    dtypes = ("fp32", "fp16", "bf16")


@register
class RefFusedUpAct(_RefFused):
    """up_gemm + act in one span: h_up is never materialised."""

    name = "ref_fused_up_act"
    covers = ("up_gemm", "act")

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        x_perm, offsets = st.require("x_perm", "expert_offsets")
        h_up = R.grouped_gemm_loop(x_perm, st.weights.w1, offsets,
                                   2 * cfg.intermediate_size)
        st.h_act = R.swiglu(h_up)


@register
class RefFusedDownScatter(_RefFused):
    """down_gemm + unpermute in one span: y_perm is never materialised.

    This is the shape of the kernel the bassrehab work could not express in
    Triton. Having a correct reference for it lets the harness compare against
    the right target before any CUDA is written.
    """

    name = "ref_fused_down_scatter"
    covers = ("down_gemm", "unpermute")

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        h_act, offsets, perm, w = st.require(
            "h_act", "expert_offsets", "perm_index", "topk_weights")
        y_perm = R.grouped_gemm_loop(h_act, st.weights.w2, offsets, cfg.hidden_size)
        st.y = R.combine(y_perm, perm, w, st.spec.num_tokens, cfg.top_k)


@pytest.fixture
def toy_spec():
    return BenchSpec(MODEL_CONFIGS["toy"], num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("uniform"), seed=0)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line(
        "markers",
        "no_gpu: asserts the laptop path; skipped when a CUDA device is attached")


def pytest_collection_modifyitems(config, items):
    """Auto-skip `gpu` tests off the box and `no_gpu` tests on it.

    On the H200 the `gpu` tests run for real and are the only verification the
    CUDA timing paths ever get. `no_gpu` is the mirror: a test whose SUBJECT is
    the machine fact "no card is attached" (the real resolver's no-card answer,
    or a door behind an absent PACKAGE such as pynvml, which hiding CUDA does
    not close) and that cannot plant it. A test whose subject is a script's
    refusal logic does not take this marker: it plants the world with the
    `no_cuda` fixture below, or spawns its child under `_hermetic.LAPTOP_ENV`,
    so the pod checks the door too. Session 4's pod suite failed 23 tests that
    asserted the laptop path on a box with a card, and STARTED a measurement
    from one of them.
    """
    has_cuda = torch.cuda.is_available()
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    skip_no_gpu = pytest.mark.skip(
        reason="a CUDA device is attached; this test asserts the laptop path")
    for item in items:
        wants_gpu = "gpu" in item.keywords
        wants_no_gpu = "no_gpu" in item.keywords
        if wants_gpu and wants_no_gpu:
            raise pytest.UsageError(f"{item.nodeid}: marked both gpu and no_gpu")
        if wants_gpu and not has_cuda:
            item.add_marker(skip_gpu)
        if wants_no_gpu and has_cuda:
            item.add_marker(skip_no_gpu)


@pytest.fixture
def no_cuda(monkeypatch):
    """Plant the no-card world in THIS process. Every detector in the repo asks
    `torch.cuda.is_available()` first (roofline.current_gpu_name,
    timing.require_cuda, provenance's gpu_name, each script's detect_card_slug
    / missing_gpu_stack / resolve_card), so one attribute makes a box with a
    card walk the laptop path. Moved here from tests/test_timing.py so every
    file plants it the same way. It hides CUDA, not NVML or an absent package:
    a door behind those takes the `no_gpu` marker instead."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True, scope="session")
def _results_root_sandbox(tmp_path_factory):
    """The pod exports MOE_RESULTS_DIR=/workspace/results/gaps-<card> to every
    arm, and pytest inherited it: in-process dry runs and self-tests printed
    and wrote under the session's real results root (session 4). Sandbox it
    INSIDE the repo, under results/* which .gitignore excludes, so every
    `IGNORED by git` line the dry runs print stays true. Tests that assert the
    default-root branch already delenv/setenv it themselves."""
    root = (pathlib.Path(__file__).resolve().parents[1] / "results" / "_pytest"
            / tmp_path_factory.getbasetemp().name)
    root.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("MOE_RESULTS_DIR")
    os.environ["MOE_RESULTS_DIR"] = str(root)
    yield root
    if previous is None:
        os.environ.pop("MOE_RESULTS_DIR", None)
    else:
        os.environ["MOE_RESULTS_DIR"] = previous
    shutil.rmtree(root, ignore_errors=True)
