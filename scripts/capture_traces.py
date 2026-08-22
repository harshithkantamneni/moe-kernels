#!/usr/bin/env python
"""Capture real expert-routing distributions from a real MoE model.

Run on the GPU box. Writes a compact per-layer expert-count histogram to
traces/<id>.npz (kilobytes), never model weights. Every later benchmark replays
that histogram offline and for free.

    python scripts/capture_traces.py --model mixtral-8x7b --phase decode \\
        --corpus chat --batches 16 --out traces/

Which models can be captured on ONE H200 (141 GB), bf16:
    mixtral-8x7b      93.4 GB   yes, comfortably
    qwen2-57b-a14b   114.8 GB   yes, but only ~26 GB left for KV and activations
    deepseek-v2-lite  31.4 GB   yes, and it is the cheap 64-expert proxy
    deepseek-v3     1369 GB     NO. Needs 5+ H200s. Its routing CANNOT be
                                captured here, and the repo must never claim
                                otherwise. Benchmark its GEOMETRY with
                                parametric routing and say so.

Decode is the interesting phase: single-token steps with many experts is the
memory-bound weight-loading regime this project targets, and decode-time routing
traces are what almost nobody publishes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench.schema import git_provenance  # noqa: E402
from moe.routing.capture import GateRecorder, find_gate_modules  # noqa: E402
from moe.routing.imbalance import expert_load  # noqa: E402
from moe.routing.traces import write_trace  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

# Small built-in corpora so a first capture needs no dataset download. Expert
# specialisation differs by domain, which is the point of having more than one.
CORPORA: dict[str, list[str]] = {
    "chat": [
        "Explain why the sky appears blue, and what would change on a planet\n"
        "with a thicker atmosphere.",
        "I have chicken thighs, rice, and a lemon. What should I cook tonight?",
        "Summarise the causes of the 1929 crash for someone who knows no\n"
        "economics.",
        "My colleague keeps taking credit for my work. How do I handle it\n"
        "without escalating?",
    ],
    "code": [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]",
        "Write a CUDA kernel that performs a warp-level reduction using\n"
        "__shfl_down_sync.",
        "template <typename T>\nclass RingBuffer {\npublic:\n"
        "    explicit RingBuffer(size_t capacity)",
        "Explain the difference between a Python generator and an async\n"
        "generator, with examples.",
    ],
    "math": [
        "Prove that the square root of two is irrational.",
        "Compute the eigenvalues of [[2, 1], [1, 2]] and explain what they\n"
        "mean geometrically.",
        "A fair coin is flipped 10 times. What is the probability of at least 7 heads?",
        "Show that the sum of the first n odd integers equals n squared.",
    ],
    "prose": [
        "The lighthouse had not been staffed since the war, and the path up\n"
        "to it had long since",
        "Describe a city at dawn from the point of view of someone who has not slept.",
        "Write an opening paragraph for a novel about a cartographer who cannot read maps.",
        "It rained for nine days. On the tenth, the river took the bridge.",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--corpus", default="chat", choices=sorted(CORPORA))
    ap.add_argument("--corpus-file", type=Path, default=None,
                    help="one prompt per line; overrides --corpus")
    ap.add_argument("--phase", default="decode", choices=("prefill", "decode"))
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32,
                    help="decode phase only: steps captured per batch")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("traces"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--trace-id", default=None)
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    if cfg.hf_repo is None:
        raise SystemExit(f"{args.model} has no hf_repo; nothing to capture")
    if not torch.cuda.is_available():
        raise SystemExit("capture needs a GPU")

    est_gb = cfg.weight_bytes("bf16") * cfg.num_moe_layers / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[capture] {args.model}: expert weights alone are about "
          f"{est_gb:.0f} GB; this device has {total_gb:.0f} GB")
    if est_gb > total_gb:
        raise SystemExit(
            f"{args.model} does not fit on this device. Its routing cannot be "
            "captured here. Benchmark its geometry with parametric routing and "
            "label the results as synthetic routing.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[capture] loading {cfg.hf_repo} (first run downloads to HF_HOME)")
    tok = AutoTokenizer.from_pretrained(cfg.hf_repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_repo, torch_dtype=getattr(torch, args.dtype),
        device_map="auto", trust_remote_code=True)
    model.eval()

    gates = find_gate_modules(model, cfg.num_experts)
    if not gates:
        raise SystemExit(
            f"found no gate module with {cfg.num_experts} outputs. The model "
            "layout is not what this script expects; inspect named_modules().")
    print(f"[capture] hooked {len(gates)} MoE gates "
          f"(config says {cfg.num_moe_layers} MoE layers of {cfg.num_layers})")

    prompts = (args.corpus_file.read_text().splitlines() if args.corpus_file
               else CORPORA[args.corpus])
    prompts = [p for p in prompts if p.strip()]

    recorder = GateRecorder(cfg, len(gates))
    recorder.attach(gates)

    all_counts = np.zeros((args.batches, len(gates), cfg.num_experts), dtype=np.int32)
    started = time.time()
    with torch.no_grad():
        for b in range(args.batches):
            batch = [prompts[(b * args.batch_size + i) % len(prompts)]
                     for i in range(args.batch_size)]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length).to(model.device)
            recorder.reset()
            if args.phase == "prefill":
                model(**enc)
            else:
                # Warm the cache without recording, then capture only the
                # single-token decode steps, which is the regime of interest.
                out = model(**enc, use_cache=True)
                past, next_tok = out.past_key_values, out.logits[:, -1:].argmax(-1)
                recorder.reset()
                for _ in range(args.max_new_tokens):
                    out = model(input_ids=next_tok, past_key_values=past,
                                use_cache=True)
                    past = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(-1)
            all_counts[b] = recorder.snapshot()
            load = expert_load(all_counts[b].sum(axis=0).tolist())
            print(f"[capture] batch {b + 1}/{args.batches} "
                  f"max/mean={load.max_over_mean:.2f} "
                  f"empty={load.empty_experts}/{cfg.num_experts} "
                  f"entropy={load.entropy_norm:.3f}")
    recorder.detach()

    trace_id = args.trace_id or f"{args.model}-{args.corpus}-{args.phase}"
    # git_provenance runs with -C <repo root>, so it records the right SHA even
    # when the script is invoked from elsewhere, and it will not hang.
    sha, dirty = git_provenance()
    meta = {
        "trace_id": trace_id,
        "model": cfg.name,
        "hf_repo": cfg.hf_repo,
        "corpus": args.corpus if not args.corpus_file else str(args.corpus_file),
        "phase": args.phase,
        "num_experts": cfg.num_experts,
        "top_k": cfg.top_k,
        "gate_fn": cfg.gate_fn,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "n_moe_layers_hooked": len(gates),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens if args.phase == "decode" else 0,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0),
        "capture_commit": sha,
        "capture_dirty": dirty,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = write_trace(args.out / f"{trace_id}.npz", all_counts, meta)
    size_kb = path.stat().st_size / 1024
    print(f"[capture] wrote {path} ({size_kb:.1f} KB) in "
          f"{time.time() - started:.0f}s")
    print(json.dumps({"trace": str(path), "shape": list(all_counts.shape)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
