"""Environment for a child process that must walk the LAPTOP path on any box.

CUDA_VISIBLE_DEVICES="" makes torch.cuda.is_available() False in the child,
which is what every card detector in this repo asks first; PY_BASE/PY_VLLM pin
the gaps-session driver's interpreters to the one running pytest instead of
/workspace/venvs/*, so an arm's dry run refuses at once rather than planning
for ~55 s under the vllm venv, and a bare `--run` in a test cannot start a
measurement from inside pytest. What this does NOT hide: NVML and nvidia-smi,
and any package the pod has that the laptop lacks (pynvml, triton, vllm). A
child whose asserted sentence comes from one of those doors is a `no_gpu` test,
not a hermetic one.
"""
import os
import sys

LAPTOP_ENV = {"CUDA_VISIBLE_DEVICES": "", "PY_BASE": sys.executable,
              "PY_VLLM": sys.executable}


def laptop_env(**extra) -> dict:
    """`os.environ` with the laptop world laid over it: MERGED, never replaced,
    so PATH, HOME and the results-root sandbox still reach the child."""
    return {**os.environ, **LAPTOP_ENV, **extra}
