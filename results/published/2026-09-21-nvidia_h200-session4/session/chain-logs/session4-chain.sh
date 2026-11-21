#!/bin/bash
# Session 4 chain: G=16 companion -> R2 smoke -> R1 smoke -> GPU test suite.
S=/workspace/session/gaps-nvidia_h200-20260921T235000Z
REPO=/workspace/moe-kernels
PY_VLLM=/workspace/venvs/vllm/bin/python
PY_BASE=/workspace/venvs/base/bin/python
export MOE_RESULTS_DIR=/workspace/results/gaps-nvidia_h200
LOG=/workspace/session4-chain.log
stage() { echo "$(date -u +%H:%M:%S) START $1" >> $LOG; }
done_() { echo "$(date -u +%H:%M:%S) END   $1 rc=$2" >> $LOG; }
cd $REPO
stage g16
$PY_VLLM scripts/private_weight_reference.py --model mixtral-8x7b --block-m 32 --treads 6 --repeats 9 --group-m 16 --session-tag "$(basename $S)" > $S/logs/private-mixtral-bm32-g16.log 2>&1; done_ g16 $?
stage r2-r1
SESSION=$S bash scripts/h200_gaps_session.sh --only thermal,calibrate,pin_probe-n64-g1,blockk-w4,elasticity-m32-n64-g16 > /workspace/session4-r1r2.log 2>&1; done_ r2-r1 $?
stage pytest
if $PY_VLLM -c "import pytest" 2>/dev/null; then PY=$PY_VLLM; else PY=$PY_BASE; fi
echo "pytest interpreter: $PY" >> $LOG
PYTHONPATH=$REPO $PY -m pytest -q -p no:cacheprovider -x --maxfail=50 > /workspace/session4-pytest.log 2>&1; done_ pytest $?
touch /workspace/session4-chain.DONE
