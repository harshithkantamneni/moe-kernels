Laptop-side Frontier work after rental prodmain-20261120 (2026-11-20).

Why: extras.sh X3 trained only Frontier's cli and cli-fills arms for Qwen3-30B-A3B; the simpath arm
(features total_routed_tokens, num_tokens_post_padded), Frontier's best on ERNIE prefill, was never in the X3 plan.

What: Frontier db5a85b cloned to scratch; venv python 3.12.13, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.3.5
(the VM trainer's versions.json; python 3.12.16 on the VM). Code: ~/code/moe-kernels at fe96486c, frontier_op.py.
- qwen3-simpath/: frontier_op.py simpath on the VM's Qwen3 CSV (extras/qwen3/frontier/profile/.../moe_kernel_only.csv)
- qwen3-pred/: frontier_op.py predict, arms cli + cli-fills (VM-trained models) + simpath (laptop-trained)
  -> cli and cli-fills predictions are BIT-IDENTICAL to the VM's (np.array_equal), so predict reproduces exactly.
- score-qwen3-simpath/: score_prod.py qwen3 with qwen3-pred (all three arms)
- repro/: ERNIE simpath retrained on the laptop from the VM's CSV. NOT bit-identical to the VM model: per-call
  prediction difference median 0.34%, p95 2.6%, mean signed -0.62% (A1410). Scored (score-ernie-laptopsimpath/):
  A1410 simpath MAPE 1.24% laptop vs 1.32% VM-trained. Training drift moves the MAPE by < 0.1 point.
