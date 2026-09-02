#!/usr/bin/env python
"""Capture real expert-routing distributions from a real MoE model.

Run on the GPU box. Writes a compact per-layer expert-count histogram to
traces/<id>.npz (kilobytes), never model weights. Every later benchmark replays
that histogram offline and for free.

    python scripts/capture_traces.py --model deepseek-v2-lite --phase decode \\
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
memory-bound weight-loading regime this project targets, and a captured decode
histogram is the input the parametric distributions are standing in for.

BEFORE RENTING THE BOX, read docs and `requirements/base.txt`: this script is
the only thing in the repo that downloads a checkpoint, it needs
`transformers`, `accelerate` and `safetensors` installed, and it needs HF_HOME
pointed at the persistent volume rather than the container disk.
"""
from __future__ import annotations

import argparse
import json
import random
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
#
# WHY 64 EACH: the default run is 16 batches of 4, and a batch axis is only
# worth sweeping if its 16 entries are 16 different things. 64 distinct prompts
# is exactly one pass with no prompt reused, which is what `build_batches`
# spends them on. Shrinking a corpus below batches*batch_size does not fail, it
# quietly turns the batch axis into a regrouping of the same text.
CORPORA: dict[str, list[str]] = {
    "chat": [
        "Explain why the sky is blue, and what changes on a planet with a thicker air.",
        "I have chicken thighs, rice, and a lemon. What should I cook tonight?",
        "Summarise the causes of the 1929 crash for someone who knows no economics.",
        "My colleague keeps taking credit for my work. How do I handle it without escalating?",
        "What is the difference between a virus and a bacterium, in plain language?",
        "Plan a three-day trip to Lisbon for someone who hates museums.",
        "How do noise-cancelling headphones actually cancel noise?",
        "My sourdough starter smells like acetone. Is it dead?",
        "Explain compound interest to a fifteen-year-old with a first job.",
        "What should I ask a landlord before signing a lease?",
        "Why does coffee taste bitter when I brew it too long?",
        "Help me write a polite email declining a wedding invitation.",
        "What are the trade-offs between renting and buying a home?",
        "How do I get a cat to stop scratching the sofa?",
        "Explain what a credit score is and what actually moves it.",
        "I keep waking up at 4am. What is worth trying before seeing a doctor?",
        "What is the difference between weather and climate?",
        "Describe how to change a bicycle tyre without a repair stand.",
        "Why do onions make you cry, and what stops it?",
        "How should I structure a one-on-one with a manager I do not trust?",
        "What is the fastest way to learn to swim as an adult?",
        "Explain the offside rule in football to somebody who has never watched it.",
        "My laptop fan runs constantly. How do I work out what is causing it?",
        "What questions should I ask when buying a used car?",
        "How does a fridge make things cold?",
        "Explain the difference between a stock and a bond.",
        "What is the etiquette for splitting a restaurant bill with colleagues?",
        "Why does bread go stale faster in the fridge than on the counter?",
        "How do I start running if I get winded climbing stairs?",
        "Explain what happens to a plane when it hits turbulence.",
        "My houseplant's leaves are yellowing from the bottom. What does that mean?",
        "How do I politely tell a friend they talk over me constantly?",
        "What does a good cover letter actually need to say?",
        "Explain the electoral college as if I grew up somewhere else.",
        "Why do some people get jet lag worse flying east than west?",
        "How much water does a person actually need in a day?",
        "Explain what inflation does to somebody living on savings.",
        "What is the difference between a hurricane, a typhoon and a cyclone?",
        "How do I remove a red wine stain from a wool rug?",
        "Explain why airlines overbook flights.",
        "What is a reasonable emergency fund for a single person renting?",
        "How do vaccines produce immunity without causing the disease?",
        "My dog is scared of thunder. What actually helps?",
        "Explain the difference between olive oil grades and when each matters.",
        "What should I look for when reading a rental car agreement?",
        "How do submarines control their depth?",
        "Explain why some medicines must be taken with food.",
        "What is the difference between a debit card and a prepaid card?",
        "How do I make small talk when I find it excruciating?",
        "Explain how tides work and why some places get four a day.",
        "What is the healthiest way to cool down a hot house without air conditioning?",
        "Why do phone batteries degrade, and what slows it down?",
        "Explain what an index fund is and why people recommend them.",
        "How do I tell whether a mushroom in my garden is dangerous?",
        "What is the point of a mattress topper?",
        "Explain the difference between espresso, ristretto and lungo.",
        "How should I prepare for a flight with a nervous four-year-old?",
        "Why does the moon look bigger near the horizon?",
        "What is the safest way to defrost a chicken?",
        "Explain how a heat pump can be more than 100% efficient.",
        "How do I negotiate a salary offer without seeming ungrateful?",
        "What causes hiccups and does anything reliably stop them?",
        "Explain the difference between a cold, flu and COVID by symptoms.",
        "How do I choose between a gas and an induction hob?",
    ],
    "code": [
        "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    pivot = a[len(a)//2]",
        "Write a CUDA kernel that performs a warp-level reduction using __shfl_down_sync.",
        "template <typename T>\nclass RingBuffer {\npublic:\n    explicit RingBuffer(size_t cap)",
        "Explain the difference between a Python generator and an async generator, with examples.",
        "Why does this Rust fail to compile: `let r = &mut v; v.push(1); println!(\"{r:?}\");`",
        "Write a SQL query returning the second-highest salary per department, ties included.",
        "import asyncio\n\nasync def fetch_all(urls):\n    async with ClientSession() as s:",
        "What is the difference between `git rebase --onto` and a plain rebase?",
        "Implement a lock-free single-producer single-consumer queue in C++20.",
        "Explain what a memory barrier does on ARM versus x86.",
        "def __init__(self, *, timeout: float = 5.0) -> None:\n    self._timeout = timeout",
        "Write a Triton kernel for a fused softmax over the last dimension.",
        "Why is my Docker image 3 GB when the binary is 12 MB?",
        "Explain Python's GIL and what actually releases it.",
        "func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {",
        "Write a regex that matches an IPv4 address but not 999.1.1.1.",
        "What does `volatile` mean in C, and why is it not a threading primitive?",
        "Explain the difference between `useMemo` and `useCallback` in React.",
        "Write a bash script that retries a command with exponential backoff.",
        "How do I profile a PyTorch model to find which op dominates the step time?",
        "class Node:\n    __slots__ = (\"key\", \"value\", \"prev\", \"next\")",
        "Explain how a bloom filter trades memory for false positives.",
        "Write a Makefile rule that rebuilds only when the header changes.",
        "What is the difference between `COPY` and `ADD` in a Dockerfile?",
        "Implement binary search that returns the insertion point, not just a hit.",
        "Explain what happens when you `kill -9` a process holding a mutex.",
        "SELECT customer_id, COUNT(*) FROM orders GROUP BY 1 HAVING COUNT(*) > 3",
        "Why does my Kubernetes pod get OOMKilled with no memory spike in the graphs?",
        "Write a Python decorator that caches on disk with a TTL.",
        "Explain the difference between fp16 and bf16 and when each overflows.",
        "int main(void) {\n    char buf[16];\n    gets(buf);\n    return 0;\n}",
        "How do I write a property-based test for a serialiser round-trip?",
        "Explain what CUDA occupancy is and when raising it does not help.",
        "Write a Go function that fans out to N workers and collects results in order.",
        "What is the difference between a shallow and a deep copy in JavaScript?",
        "Implement a rate limiter using a token bucket, in Python, thread-safe.",
        "Explain why `float` equality comparisons fail and what to use instead.",
        "async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {",
        "What does `git bisect run` do and how do I script the test?",
        "Explain the actor model and how it differs from shared-memory threading.",
        "Write a CMakeLists.txt for a header-only library with tests.",
        "How does Python's `__slots__` reduce memory, and what breaks?",
        "Explain what a segmentation fault actually is at the hardware level.",
        "Write a JAX function using `vmap` to batch a per-example gradient.",
        "What is the difference between `INNER JOIN` and `WHERE EXISTS` for performance?",
        "Explain tail-call optimisation and why CPython does not do it.",
        "for (int i = 0; i < n; i += 4) {\n    __m256d a = _mm256_loadu_pd(&x[i]);",
        "How do I debug a flaky test that only fails in CI?",
        "Explain what a monad is using only Python's `Optional` as the example.",
        "Write a Terraform module that creates an S3 bucket with versioning.",
        "What is the difference between `git reset --soft` and `--mixed`?",
        "Explain how TLS certificate chain validation works, step by step.",
        "Implement Dijkstra's algorithm with a binary heap and explain the complexity.",
        "Why does my NumPy code slow down after adding a `.copy()`?",
        "Explain the C++ rule of five and when you can use the rule of zero.",
        "Write a systemd unit that restarts a service on failure with a backoff.",
        "What is the difference between a coroutine and a green thread?",
        "Explain how Merkle trees let Git detect a changed subtree cheaply.",
        "def forward(self, x):\n    return self.norm(x + self.attn(self.norm(x)))",
        "How do I choose a shard count for a Kafka topic?",
        "Explain what `perf stat` reports and which counters actually matter.",
        "Write a Python context manager that times a block and logs it.",
        "What causes a deadlock in a database transaction, and how do I read the log?",
        "Explain the difference between structural and nominal typing with TypeScript examples.",
    ],
    "math": [
        "Prove that the square root of two is irrational.",
        "Compute the eigenvalues of [[2, 1], [1, 2]] and explain what they mean geometrically.",
        "A fair coin is flipped 10 times. What is the probability of at least 7 heads?",
        "Show that the sum of the first n odd integers equals n squared.",
        "Prove that there are infinitely many primes.",
        "Find the derivative of x^x and explain the step that trips people up.",
        "What is the expected number of rolls of a fair die to see all six faces?",
        "Show that the harmonic series diverges without using the integral test.",
        "Compute the integral of e^(-x^2) over the whole real line.",
        "Explain why the determinant is the signed volume scaling factor.",
        "Solve the recurrence T(n) = 2T(n/2) + n and state the method used.",
        "Prove that a continuous function on a closed interval attains its maximum.",
        "What is the probability two random people in a room of 23 share a birthday?",
        "Show that the Fibonacci numbers satisfy Binet's formula.",
        "Explain the difference between pointwise and uniform convergence, with an example.",
        "Find all solutions to x^2 = 1 in the integers modulo 8.",
        "Prove that the rationals are countable and the reals are not.",
        "Compute the volume of the region bounded by z = x^2 + y^2 and z = 4.",
        "Explain what a p-value is and what it is not.",
        "Show that every finite group of prime order is cyclic.",
        "What is the Fourier transform of a Gaussian, and why is that convenient?",
        "Prove that the diagonals of a rhombus are perpendicular.",
        "Explain Bayes' theorem using the false-positive medical-test example.",
        "Find the general solution of y'' + 4y = 0 and interpret it physically.",
        "Show that a matrix is diagonalisable if it has n distinct eigenvalues.",
        "What is the variance of a sum of two correlated random variables?",
        "Prove that the Cantor set is uncountable but has measure zero.",
        "Explain why the central limit theorem needs finite variance.",
        "Compute the number of ways to seat 8 people around a circular table.",
        "Show that the exponential function is its own derivative from the series.",
        "What is the difference between a metric and a norm?",
        "Prove that a bounded monotone sequence converges.",
        "Explain the intuition behind Lagrange multipliers.",
        "Find the Taylor series of ln(1 + x) and state its radius of convergence.",
        "Show that the sum over 1/n^2 converges, and state its value.",
        "What is the probability of a run of five heads in 100 fair flips?",
        "Prove that the composition of two injective functions is injective.",
        "Explain the difference between correlation and independence with a counterexample.",
        "Compute the arc length of y = cosh(x) from 0 to 1.",
        "Show that every polynomial of odd degree has a real root.",
        "Explain what an eigenvector of a Markov transition matrix means.",
        "Find the least squares solution when the design matrix is rank deficient.",
        "Prove the AM-GM inequality for three positive numbers.",
        "What does the condition number of a matrix tell you about solving Ax = b?",
        "Explain the difference between a permutation and a combination with a card example.",
        "Show that the gradient points in the direction of steepest ascent.",
        "Compute the expected value of the maximum of two uniform [0,1] variables.",
        "Prove that a graph with n vertices and more than n-1 edges contains a cycle.",
        "Explain what a confidence interval actually covers over repeated sampling.",
        "Find the fixed points of the logistic map and their stability.",
        "Show that the trace of a matrix equals the sum of its eigenvalues.",
        "What is the difference between big-O, big-Theta and big-Omega?",
        "Prove that the median minimises the mean absolute deviation.",
        "Explain how the singular value decomposition relates to eigendecomposition.",
        "Compute the probability that a random permutation of n items has no fixed point.",
        "Show that the set of invertible matrices is open in the space of matrices.",
        "Explain why the sample variance divides by n-1.",
        "Find the stationary distribution of a two-state Markov chain.",
        "Prove that a convex function on an interval is continuous on its interior.",
        "What is the geometric meaning of the cross product magnitude?",
        "Explain the pigeonhole principle with a non-obvious application.",
        "Compute the surface area of a sphere by integration.",
        "Show that a symmetric positive definite matrix has a unique Cholesky factor.",
        "Explain what entropy measures for a discrete probability distribution.",
    ],
    "prose": [
        "The lighthouse had not been staffed since the war, and the path up to it had long since",
        "Describe a city at dawn from the point of view of someone who has not slept.",
        "Write an opening paragraph for a novel about a cartographer who cannot read maps.",
        "It rained for nine days. On the tenth, the river took the bridge.",
        "She counted the stairs on the way down, as she had every evening for eleven years.",
        "Write a scene in which two strangers share a taxi and neither speaks.",
        "The orchard belonged to nobody now, which is why the fruit was so good.",
        "Describe the smell of a house that has been closed up for a decade.",
        "He had been an excellent liar until the winter his mother died.",
        "Write the last paragraph of a letter that will never be sent.",
        "The train was forty minutes late and nobody on the platform looked surprised.",
        "Describe a wedding from the point of view of the caterer.",
        "There were three keys on the ring and she had never known what the third one opened.",
        "Write a scene set entirely inside a lift that has stopped between floors.",
        "The dog had been waiting at the door since the funeral.",
        "Describe an argument in which neither person raises their voice.",
        "By the time the tide came back, the footprints belonged to somebody else.",
        "Write an opening line for a story about a town that forgot its own name.",
        "The photograph was of a room she had never been in, and yet.",
        "Describe the moment a musician realises the audience has stopped listening.",
        "He kept the receipts for everything, which is how they eventually found him.",
        "Write a scene in which a child explains something to an adult correctly.",
        "The garden had been designed by someone who expected to live much longer.",
        "Describe a border crossing at three in the morning.",
        "She had rehearsed the sentence for a week and still said it wrong.",
        "Write the opening of a ghost story where the ghost is not the frightening part.",
        "The library kept one book that nobody was permitted to catalogue.",
        "Describe a room from the point of view of the person cleaning it after a party.",
        "They had agreed not to talk about it, and so they talked about the weather beautifully.",
        "Write a scene set in a launderette at closing time.",
        "The map was accurate in every detail except the one that mattered.",
        "Describe the first cold morning of the year in a house with no heating.",
        "He inherited the shop, the debts, and the cat, in that order of surprise.",
        "Write a paragraph in which nothing happens and it is unbearable.",
        "The letter arrived nineteen years after it was posted.",
        "Describe a football match from the point of view of the groundskeeper.",
        "She had never seen the sea and had opinions about it anyway.",
        "Write an opening for a story about the last person to leave a village.",
        "The clock in the hall had been wrong for so long that it was right again.",
        "Describe a hospital waiting room without mentioning illness.",
        "They found the boat two valleys inland and nobody could explain it.",
        "Write a scene in which someone gives away something they cannot afford to lose.",
        "The recipe was in her handwriting but the measurements made no sense.",
        "Describe a market town on the day the factory closes.",
        "He had one photograph of his father and had studied it into meaninglessness.",
        "Write the opening of a novel set entirely during a single power cut.",
        "The bees left in March, all at once, and the hives stayed exactly as they were.",
        "Describe a long drive in which the passenger is asleep.",
        "She wrote the address from memory and it was, astonishingly, correct.",
        "Write a scene in which two people fail to recognise each other.",
        "The house had been built facing away from the view, deliberately.",
        "Describe the last day of a job someone has held for thirty years.",
        "There was a word for what he was feeling and he refused to learn it.",
        "Write an opening paragraph in which the weather does all the work.",
        "The suitcase had been packed for six months and stood by the door.",
        "Describe a funeral at which somebody laughs and is right to.",
        "They planted the tree over the septic tank, which explains everything after.",
        "Write a scene set in a border town where two languages are spoken badly.",
        "The violin was worth more than the house it had been kept in.",
        "Describe an empty swimming pool in February.",
        "He had learned to sleep through anything except silence.",
        "Write the closing paragraph of a story about a promise that was kept too literally.",
        "The road ended at a gate, and beyond the gate the road continued.",
        "Describe the inside of a shop that sells only one thing.",
    ],
}


def padding_side_for(phase: str) -> str:
    """Which side a batch should be padded on for this phase.

    Decode needs LEFT padding. The loop takes `logits[:, -1]` as the next token
    and then feeds one token at a time, so with right padding a short sequence's
    continuation starts from a PAD position: the model is asked what follows the
    padding, and every routing count after that describes a sequence no human
    wrote. Left padding puts the real final token last in every row.

    Prefill pads on the right, the tokeniser default, because nothing reads a
    last position there -- the pad rows are excluded from the counts by the
    recorder's token mask instead.
    """
    return "left" if phase == "decode" else "right"


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that ignore padding, the way generation computes them.

    Without this, a left-padded row is told its first real token sits at
    position 3 rather than 0, because the default position ids are a plain
    arange over the padded length. The rotary embedding then rotates every row
    by a different amount depending on how much padding it happened to get,
    which changes the hidden states and therefore the routing.
    """
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def build_batches(prompts: list[str], batches: int, batch_size: int,
                  seed: int = 0) -> list[list[str]]:
    """Distinct prompt sets, one per batch.

    The old expression was `prompts[(b * batch_size + i) % len(prompts)]`. With
    a 4-prompt corpus and the default batch_size of 4 that reduces to
    `prompts[i]` for every b: all 16 batches byte-identical, decoded greedily to
    the same tokens, one sample recorded sixteen times. It passes `write_trace`,
    and `Trace.select` picks a batch with `seed % B`, so a seed sweep over that
    file resamples the same slice while appearing to sample sixteen.

    Walk a deterministically shuffled corpus instead, taking consecutive
    windows. One pass yields `len(prompts) // batch_size` batches sharing no
    prompt. If more are asked for, reshuffle under a new key so the next pass at
    least groups different prompts together -- and note that a corpus of exactly
    `batch_size` prompts cannot do even that, which is why the caller still
    checks the counts before writing.
    """
    if batches <= 0 or batch_size <= 0:
        raise ValueError(
            f"batches={batches} and batch_size={batch_size} must both be positive")
    if len(prompts) < batch_size:
        raise ValueError(
            f"corpus has {len(prompts)} prompts, fewer than one batch of "
            f"{batch_size}; a batch cannot be filled without repeating a prompt "
            "inside it")

    out: list[list[str]] = []
    pass_no = 0
    while len(out) < batches:
        order = list(range(len(prompts)))
        random.Random(f"{seed}:{pass_no}").shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            out.append([prompts[j] for j in order[start:start + batch_size]])
            if len(out) == batches:
                break
        pass_no += 1
    return out


def distinct_batch_count(batches: list[list[str]]) -> int:
    """How many of these batches are a different set of prompts from the rest.

    Order within a batch does not change the per-layer histogram, since the
    counts are summed over the batch, so a reordering is not a new sample.
    """
    return len({tuple(sorted(b)) for b in batches})


def assert_batches_differ(counts) -> None:
    """Refuse to write a trace whose batch axis is one sample repeated.

    `Trace.select` picks a batch with `seed % B`. An all-identical trace makes
    a seed sweep look like B samples of real routing when it is one, and every
    downstream number inherits that. Nothing in `write_trace` can see it: the
    file is well-formed, the metadata is true, the histogram is a real
    histogram.

    Identical counts do not strictly prove identical inputs, but no real corpus
    produces them across 16 batches, and the file is the expensive mistake.
    """
    counts = np.asarray(counts)
    if counts.shape[0] < 2:
        return
    if np.array_equal(counts, np.broadcast_to(counts[0], counts.shape)):
        raise SystemExit(
            f"every one of the {counts.shape[0]} captured batches has identical "
            "per-layer expert counts, so the batch axis carries one sample and "
            "Trace.select would return it for every seed. The corpus needs at "
            "least batches*batch_size distinct prompts. Nothing was written.")


def _disable_torchvision() -> None:
    """Make `import torchvision` fail cleanly, BEFORE transformers reaches it.

    THE FAILURE THIS FIXES, and it cost a whole step-7 slot on 2026-09-01:

        RuntimeError: operator torchvision::nms does not exist
        ModuleNotFoundError: Could not import module 'MixtralForCausalLM'

    transformers imports torchvision unconditionally from `image_utils`, and the
    torchvision on the path is the IMAGE's, installed against a different torch
    than the venv's pinned 2.13.0. Registering its fake kernels against the wrong
    torch raises at import time, and transformers reports the result as "could
    not import MixtralForCausalLM. Are this object's requirements defined
    correctly?" -- which reads as a model problem and is a torch/torchvision ABI
    problem. Both models this study captures died that way.

    Setting the entry to None makes `import torchvision` raise ImportError, which
    is the failure transformers ALREADY handles: its is_torchvision_available()
    path degrades to no image support, and nothing here processes images. That is
    strictly better than uninstalling the image's torchvision, which is shared
    with everything else on the box, and better than pinning a matching build,
    which would pull a second torch.
    """
    import sys as _sys
    if "torchvision" in _sys.modules and _sys.modules["torchvision"] is not None:
        return                       # already imported and working; leave it
    _sys.modules["torchvision"] = None
    print("[capture] torchvision disabled: transformers imports it "
          "unconditionally and the one on this path was built against a "
          "different torch. Nothing here processes images.")


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
    ap.add_argument("--corpus-seed", type=int, default=0,
                    help="which shuffle of the corpus fills the batches")
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
    print(f"[capture] {args.model}: routed-expert weights alone are about "
          f"{est_gb:.0f} GB, before attention, embeddings, shared experts and "
          f"the KV cache; this device has {total_gb:.0f} GB")
    if est_gb > total_gb:
        raise SystemExit(
            f"{args.model} does not fit on this device. Its routing cannot be "
            "captured here. Benchmark its geometry with parametric routing and "
            "label the results as synthetic routing.")

    # Build the batches BEFORE the download. A corpus too small for the
    # requested sweep is a two-second failure, not a 93 GB one.
    prompts = (args.corpus_file.read_text().splitlines() if args.corpus_file
               else CORPORA[args.corpus])
    # Deduplicate: a repeated prompt is not a second sample, and a corpus file
    # with duplicates would otherwise shrink the batch axis invisibly.
    prompts = list(dict.fromkeys(p for p in prompts if p.strip()))
    wanted = args.batches * args.batch_size
    if len(prompts) < wanted:
        print(f"[capture] WARNING: corpus has {len(prompts)} distinct prompts "
              f"but {args.batches} batches of {args.batch_size} want {wanted}. "
              "Batches will regroup the same text, so the batch axis samples "
              "grouping rather than new prompts.")
    try:
        batch_prompts = build_batches(prompts, args.batches, args.batch_size,
                                      seed=args.corpus_seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    n_distinct = distinct_batch_count(batch_prompts)
    print(f"[capture] {len(prompts)} distinct prompts -> {n_distinct} distinct "
          f"batches of {args.batch_size}")

    _disable_torchvision()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[capture] loading {cfg.hf_repo} (first run downloads to HF_HOME)")
    tok = AutoTokenizer.from_pretrained(cfg.hf_repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side_for(args.phase)
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
    if len(gates) != cfg.num_moe_layers:
        print("[capture] WARNING: that is not the layer count the config "
              "predicts. The trace's layer axis will not line up with the "
              "model's, so an @bNlM slice would name the wrong layer.")

    recorder = GateRecorder(cfg, len(gates))
    recorder.attach(gates)

    all_counts = np.zeros((args.batches, len(gates), cfg.num_experts), dtype=np.int32)
    started = time.time()
    with torch.no_grad():
        for b, batch in enumerate(batch_prompts):
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length).to(model.device)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            pos = position_ids_from_mask(attn)
            recorder.reset()
            # Pad positions are scored by the gate exactly like real tokens.
            # Excluding them is the difference between a trace of the corpus
            # and a trace of the pad embedding.
            recorder.set_token_mask(attn.reshape(-1).bool())
            if args.phase == "prefill":
                model(input_ids=input_ids, attention_mask=attn, position_ids=pos)
            else:
                # Warm the cache without recording, then capture only the
                # single-token decode steps, which is the regime of interest.
                out = model(input_ids=input_ids, attention_mask=attn,
                            position_ids=pos, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(-1)
                next_pos = pos[:, -1:] + 1
                recorder.reset()
                # One real token per sequence from here on: nothing to mask.
                recorder.set_token_mask(None)
                for _ in range(args.max_new_tokens):
                    attn = torch.cat(
                        [attn, torch.ones_like(attn[:, :1])], dim=-1)
                    out = model(input_ids=next_tok, attention_mask=attn,
                                position_ids=next_pos, past_key_values=past,
                                use_cache=True)
                    past = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(-1)
                    next_pos = next_pos + 1
            all_counts[b] = recorder.snapshot()
            load = expert_load(all_counts[b].sum(axis=0).tolist())
            print(f"[capture] batch {b + 1}/{args.batches} "
                  f"max/mean={load.max_over_mean:.2f} "
                  f"empty={load.empty_experts}/{cfg.num_experts} "
                  f"entropy={load.entropy_norm:.3f}")
    recorder.detach()

    assert_batches_differ(all_counts)

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
        "corpus_seed": args.corpus_seed,
        "distinct_prompts": len(prompts),
        "distinct_batches": n_distinct,
        "padding_side": tok.padding_side,
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
