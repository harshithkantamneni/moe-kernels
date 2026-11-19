"""scripts/prod/prompts.py (the prompt set: deterministic, pinned, tamper-evident) and serve_capture.py's
concurrency design (levels, the V1-scheduler T-histogram estimator, the coverage gate), off the GPU."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import prodlib as PL  # noqa: E402
import prompts as PR  # noqa: E402
import serve_capture as S  # noqa: E402


def enc(text: str) -> list[int]:
    return [sum(map(ord, w)) % 50000 for w in text.split()]


def sharegpt(tmp_path, n=300, seed=1) -> Path:
    rng = random.Random(seed)
    words = ["the", "router", "expert", "token", "graph", "kernel", "tile", "clock", "lock", "batch"]
    doc = []
    for i in range(n):
        conv = [{"from": "human", "value": " ".join(rng.choice(words) for _ in range(rng.randint(1, 60)))},
                {"from": "gpt", "value": " ".join(rng.choice(words) for _ in range(rng.randint(1, 200)))}]
        if i % 17 == 0:
            conv = conv[:1]                         # no reply: ineligible
        if i % 23 == 0:
            conv = [{"from": "gpt", "value": "hi"}] + conv
        doc.append({"id": f"c{i}", "conversations": conv})
    p = tmp_path / "sg.json"
    p.write_text(json.dumps(doc))
    return p


def test_the_set_is_a_function_of_its_inputs_only(tmp_path):
    p = sharegpt(tmp_path)
    a = PR.build(kind="sharegpt", data_path=p, encode=enc, n=100, seed=0, max_input=40, out_dist="sharegpt")
    b = PR.build(kind="sharegpt", data_path=p, encode=enc, n=100, seed=0, max_input=40, out_dist="sharegpt")
    c = PR.build(kind="sharegpt", data_path=p, encode=enc, n=100, seed=1, max_input=40, out_dist="sharegpt")
    PR.write(a, tmp_path / "a.jsonl")
    PR.write(b, tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert a["header"]["body_sha256"] != c["header"]["body_sha256"]
    assert all(PR.MIN_INPUT <= r["input_len"] <= 40 for r in a["rows"])
    assert all(4 <= r["max_tokens"] <= 1024 for r in a["rows"])
    assert [r["i"] for r in a["rows"]] == list(range(100))


def test_sharegpt_pairs_take_the_first_human_turn_and_its_reply():
    doc = [{"id": "x", "conversations": [{"from": "gpt", "value": "a"}, {"from": "human", "value": "q1"},
                                         {"from": "gpt", "value": "r1"}, {"from": "human", "value": "q2"}]},
           {"id": "y", "conversations": [{"from": "human", "value": "only"}]}]
    assert PR.sharegpt_pairs(doc) == [("x", "q1", "r1")]


def test_an_edited_file_is_refused_on_load(tmp_path):
    a = PR.build(kind="sharegpt", data_path=sharegpt(tmp_path), encode=enc, n=20, seed=0, max_input=40,
                 out_dist="fixed:16")
    PR.write(a, tmp_path / "p.jsonl")
    assert PR.load(tmp_path / "p.jsonl")["rows"] == a["rows"]
    lines = (tmp_path / "p.jsonl").read_text().splitlines()
    row = json.loads(lines[3])
    row["max_tokens"] += 1
    lines[3] = json.dumps(row, sort_keys=True)
    (tmp_path / "p.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(PR.PromptError, match="body sha256"):
        PR.load(tmp_path / "p.jsonl")


def test_an_unpinned_source_is_refused_without_the_smoke_flag(tmp_path, monkeypatch):
    p = sharegpt(tmp_path)
    monkeypatch.setitem(PR.SOURCES["sharegpt"], "sha256", None)
    with pytest.raises(PR.PromptError, match="no pinned sha256"):
        PR.check_pin("sharegpt", p, allow_unpinned=False)
    assert PR.check_pin("sharegpt", p, allow_unpinned=True) == PR.sha256_file(p)
    monkeypatch.setitem(PR.SOURCES["sharegpt"], "sha256", "0" * 64)
    with pytest.raises(PR.PromptError, match="!= pinned"):
        PR.check_pin("sharegpt", p, allow_unpinned=False)


def test_text_fallback_and_out_dist_parsing(tmp_path):
    t = tmp_path / "t.txt"
    t.write_text(" ".join(f"w{i}" for i in range(5000)))
    a = PR.build(kind="text", data_path=t, encode=enc, n=30, seed=3, max_input=200, out_dist="uniform:8,64")
    assert len(a["rows"]) == 30 and all(8 <= r["max_tokens"] <= 64 for r in a["rows"])
    with pytest.raises(PR.PromptError):
        PR.build(kind="text", data_path=t, encode=enc, n=3, seed=0, max_input=50, out_dist="sharegpt")
    for bad in ("uniform:9,3", "fixed:0", "nope", "lognormal:x"):
        with pytest.raises(PR.PromptError):
            PR.parse_out_dist(bad)
    assert PR.parse_out_dist("lognormal:5,1") == ("lognormal", (5.0, 1.0))


# ---------------------------------------------------------------- the concurrency design

def test_levels_parse_with_repeats():
    assert S.parse_levels("1,8,1024x3") == [1, 8, 1024, 1024, 1024]
    for bad in ("0", "2000", "8x0", ""):
        with pytest.raises(ValueError):
            S.parse_levels(bad)


def test_the_scheduler_model_decodes_first_and_chunks_prefill():
    assert S.simulate_level(1, [(10, 5)]) == [10, 1, 1, 1, 1]
    # one prompt larger than the budget: chunked, the first token at the end of its last chunk
    assert S.simulate_level(1, [(20000, 2)], budget=16384) == [16384, 3616, 1]
    # two in flight: decodes ride with the second request's prefill
    assert S.simulate_level(2, [(4, 3), (6, 2)]) == [10, 2, 1]
    # closed loop: a finished request is replaced by the next prompt
    assert S.simulate_level(1, [(4, 1), (4, 1), (4, 1)]) == [4, 4, 4]


def test_padding_follows_the_capture_sizes():
    sizes = PL.capture_sizes()
    assert sizes[:4] == [1, 2, 4, 8] and 248 in sizes and 256 in sizes and sizes[-1] == 512 and 264 not in sizes
    assert PL.padded_T(3, "graphs") == 4 and PL.padded_T(250, "graphs") == 256 and PL.padded_T(513, "graphs") == 513
    assert PL.padded_T(3, "eager") == 3


def test_the_registered_levels_give_every_tile_30_steps_on_a_sharegpt_like_mix():
    """CRITIQUE M9 on the estimator, for ShareGPT-like lengths (input median ~150, output median ~180)."""
    rng = random.Random(0)
    rows = [{"input_len": max(4, min(2048, int(math.exp(rng.gauss(5.0, 1.0))))),
             "max_tokens": max(4, min(1024, int(math.exp(rng.gauss(5.2, 0.9)))))} for _ in range(4096)]
    est = S.estimate_histogram(S.parse_levels(S.DEFAULT_LEVELS), rows, 64)
    gate = S.coverage_gate(est["per_tile"], S.MIN_STEPS_PER_TILE)
    assert gate["verdict"] == "PASS", gate
    assert set(est["per_tile"]) == {lab for lab, _lo, _hi in PL.TILE_BINS}


def test_the_coverage_gate_fails_loudly_on_a_short_tile():
    g = S.coverage_gate({"a": 31, "b": 2}, 30)
    assert g["verdict"] == "FAIL" and "'b': 2" in g["why"]


def test_serve_capture_dry_run_prints_the_plan_without_vllm(tmp_path, capsys):
    a = PR.build(kind="sharegpt", data_path=sharegpt(tmp_path), encode=enc, n=50, seed=0, max_input=40,
                 out_dist="fixed:16")
    PR.write(a, tmp_path / "p.jsonl")
    rc = S.main(["--model", "olmoe-1b-7b", "--prompts", str(tmp_path / "p.jsonl"), "--concurrency", "1,8",
                 "--requests-per-level", "4", "--out", str(tmp_path / "out"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN" in out and "16x64x128w4s4" in out
    assert not (tmp_path / "out").exists()
    assert S.main(["--model", "olmoe-1b-7b", "--prompts", str(tmp_path / "p.jsonl"), "--mode", "graphs",
                   "--torch-profiler", "8:10", "--out", str(tmp_path / "o2"), "--dry-run"]) == 2
    assert S.main(["--model", "olmoe-1b-7b", "--prompts", str(tmp_path / "p.jsonl"), "--max-num-batched-tokens",
                   "8192", "--out", str(tmp_path / "o3"), "--dry-run"]) == 2


def test_tiles_branches_and_strata():
    for T in (1, 32, 33, 64, 65, 96, 97, 128, 129, 512, 513, 8255, 8256, 16384):
        assert PL.tile_label(T, 64) == PL.tile_bin(T)
    assert PL.branch(128) == "decode" and PL.branch(129) == "prefill"
    assert PL.strata(150, 6) == ["band_64x128", "small_batch_align_on_prefill_tile"]
    assert PL.strata(171, 6) == ["band_64x128"] and PL.strata(150, 8) == ["band_64x128"]
    assert PL.strata(8200, 6) == ["refused_window_8192_8255"] and PL.strata(64, 6) == []


def test_hf_config_check_reads_ernie_and_olmoe_keys():
    ernie = {"hidden_size": 2560, "moe_intermediate_size": 1536, "moe_num_experts": [64, 64], "moe_k": 6,
             "num_hidden_layers": 28}
    assert PL.check_hf_config(PL.MODELS["ernie-4.5-21b-a3b"], ernie) == []
    assert PL.check_hf_config(PL.MODELS["ernie-4.5-21b-a3b"], dict(ernie, moe_k=8))
    olmoe = {"hidden_size": 2048, "intermediate_size": 1024, "num_experts": 64, "num_experts_per_tok": 8,
             "num_hidden_layers": 16}
    assert PL.check_hf_config(PL.MODELS["olmoe-1b-7b"], olmoe) == []
