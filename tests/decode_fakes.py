"""Off-GPU fakes for the decode stage's tests: a backend that answers r3_decode.run_page with planted
times (so the real page writer, report and cells.csv are exercised end to end), a perfect spy record,
and a builder of a synthetic results tree that the decode scorers read as they would a rental's."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r3_decode as RD  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

CARD = "nvidia_gh200_480gb"


def perfect_record(cfg, cell, ids_flat, *, declared: int, copies: int) -> dict:
    """The spy record vLLM would produce for this cell: what expected_readback says, as kernel inputs."""
    exp = RD.expected_readback(cfg, cell.T, cell.counts, ids_flat, declared=declared, copies=copies)
    gem = []
    for i in range(2):
        if exp["path"] == RD.NAIVE:
            srt = None
        else:
            srt = [exp["numel"]] * exp["sorted_len"]
            pos = 0
            for e, v in enumerate(cell.counts):
                if not v:
                    continue
                tiles = -(-int(v) // exp["config"]["BLOCK_SIZE_M"])
                for j in range(int(v)):
                    srt[pos + j] = 0
                pos += tiles * exp["config"]["BLOCK_SIZE_M"]
        eids = list(exp["expert_order"]) + [-1] * 4
        gem.append({"sorted": srt, "expert_ids": eids, "post_padded": exp["post_padded"], "grid": exp["grid"][i],
                    "use_td": False, "config": dict(exp["config"], SPLIT_K=1), "top_k": cfg.top_k if i == 0 else 1})
    return {"gemms": gem, "tuned": None, "override": None}


class FakeBackend:
    """run_page's backend with no GPU: ids from cell_ids (CPU), a perfect readback (or a planted bad one),
    GD4 true, a host probe of `host_ms`, and times truth(cell, arm) x (1 + noise); `slip` holds
    (label, arm, n): the n-th timing of that cell and arm reads 60 MHz under the clock."""

    def __init__(self, cfg, page, *, truth, copies: int = 1, calls: int = 1, noise: float = 0.0, seed: int = 0,
                 host_ms: float = 0.002, clock: float = 1710.0, bad_readback=(), slip=(), R: int = 1, same_copy=()):
        self.cfg, self.page, self.truth, self.copies, self.calls = cfg, page, truth, copies, calls
        self.noise, self.host_ms, self.clock = noise, host_ms, clock
        self.rng = random.Random(seed)
        self.bad, self.slip = set(bad_readback), set(slip)
        self.R, self.same_copy = R, set(same_copy)
        self.timed = []

    def build(self, cell, arm):
        ids = RD.cell_ids(self.cfg, cell, self.page.shuffle_seed)
        return ids, [(cell, arm)] * self.calls

    def readback(self, cell, arm, ids, calls):
        flat = ids.reshape(-1).tolist()
        declared = self.cfg.num_experts * (self.copies if arm == RD.DECL else 1)
        if arm == RD.DECL:
            flat = [int(v) * self.copies for v in flat]
        rec = perfect_record(self.cfg, cell, flat, declared=declared, copies=self.copies if arm == RD.DECL else 1)
        if cell.label in self.bad:
            rec["gemms"][0]["post_padded"] += 1
        rec["copy_ptrs"] = [1000 + j for j in range(self.R)] if cell.label not in self.same_copy else [1000] * self.R
        return rec

    def replay_fn(self, calls):
        return calls, len(calls), None

    def fidelity(self, calls, replay, holder):
        return True, "fake"

    def host_probe(self, replay):
        return self.host_ms

    def time(self, replay, per):
        cell, arm = replay[0]
        ms = self.truth(cell, arm) * (1 + self.rng.gauss(0.0, self.noise))
        self.timed.append((cell.label, arm))
        nth = self.timed.count((cell.label, arm))
        clk = self.clock - (60.0 if (cell.label, arm, nth) in self.slip else 0.0)
        return SimpleNamespace(ms_p50=ms * per, ms_min=ms * per, ms_std=0.0, samples=30, trials=3,
                               sm_clock_load_mhz=clk, clock_samples_mhz=(clk,) * 8, clock_drift_ok=True,
                               clock_level_ok=True, host_bound=False, host_enqueue_ms=0.01, duty_achieved=0.25,
                               power_w=300.0, clock_note="")


def write_page(root: Path, model: str, label: str, page_path: Path, inst: str, truth, *, noise=0.0, seed=0,
               copies=1, bad_readback=(), session_tag="T-lock1710", repeats=9) -> Path:
    """One decode page in a results tree, as the VM writes it: <root>/gaps-<card>-<model>-<label>/
    private_weight_reference/<run id>/{report.json, cells.csv}."""
    cfg = MODEL_CONFIGS[model]
    page = RD.load_page(page_path, cfg, model)
    out = root / f"gaps-{CARD}-{model}-{label}" / "private_weight_reference" / f"run-{label}-{inst}"
    out.mkdir(parents=True, exist_ok=True)
    store = RD.Store(out / "cells.csv")
    be = FakeBackend(cfg, page, truth=truth, copies=copies, noise=noise, seed=seed, bad_readback=bad_readback)
    res = RD.run_page(cfg, page, be, inst=inst, repeats=repeats, store=store, lock_mhz=1710, min_clean=0,
                      say=lambda m: None)
    store.close()
    args = argparse.Namespace(model=model, dtype="bf16", inst=inst, duty=0.25, repeats=repeats, group_m=1, seed=0,
                              session_tag=session_tag, min_clean_repeats=0, rot="auto")
    RD.write_report(args, cfg, page, res, out_dir=out, run_id=out.name, card=CARD, R=1, calls=1, copies=copies,
                    slip_lock=1710, header=["DECODE PAGE (fake)"], seconds=1.0)
    return out


def truth_v4d(model: str, prm, *, plus_ms: float = 0.0, decl_copies: int | None = None):
    """A truth function: v4d at prm (+ a constant), per (cell, arm)."""
    import r3_timing_model_v4d as V4D
    E = MODEL_CONFIGS[model].num_experts

    def f(cell, arm):
        p = V4D.price(model, cell.T, cell.counts, prm, declared=None if arm == RD.NATIVE else E * int(decl_copies or 1))
        return (p["ms"] if p["ms"] is not None else p["body_ms"] + 0.02) + plus_ms
    return f
