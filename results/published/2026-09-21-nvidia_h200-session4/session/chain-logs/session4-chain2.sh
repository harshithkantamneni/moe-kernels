#!/bin/bash
# Session 4 chain 2: clock-lock probe (+ locked re-runs if granted), cross-run
# replicate, second model, counter route. Runs only after chain 1 is DONE.
S=/workspace/session/gaps-nvidia_h200-20260921T235000Z
REPO=/workspace/moe-kernels
PY_VLLM=/workspace/venvs/vllm/bin/python
export MOE_RESULTS_DIR=/workspace/results/gaps-nvidia_h200
LOG=/workspace/session4-chain2.log
stage() { echo "$(date -u +%H:%M:%S) START $1" >> $LOG; }
done_() { echo "$(date -u +%H:%M:%S) END   $1 rc=$2" >> $LOG; }
R3() { $PY_VLLM scripts/private_weight_reference.py --model mixtral-8x7b --block-m 32 --treads 6 --repeats 9 --session-tag "$(basename $S)" "$@"; }
cd $REPO
stage clocklock-probe
nvidia-smi -lgc 1425,1425 > /workspace/session4-clocklock.log 2>&1; LRC=$?
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics,clocks_throttle_reasons.active --format=csv >> /workspace/session4-clocklock.log 2>&1
done_ clocklock-probe $LRC
if [ $LRC -eq 0 ]; then
  stage locked-g1
  R3 --group-m 1 --seed 2 > $S/logs/private-mixtral-bm32-locked1425-g1.log 2>&1; done_ locked-g1 $?
  stage locked-g16
  R3 --group-m 16 --seed 2 > $S/logs/private-mixtral-bm32-locked1425-g16.log 2>&1; done_ locked-g16 $?
  nvidia-smi -rgc >> /workspace/session4-clocklock.log 2>&1; echo "reset rc=$?" >> $LOG
fi
stage seed1
R3 --seed 1 > $S/logs/private-mixtral-bm32-seed1.log 2>&1; done_ seed1 $?
stage qwen2
$PY_VLLM scripts/private_weight_reference.py --model qwen2-57b-a14b --block-m 32 --treads 6 --repeats 9 --session-tag "$(basename $S)" > $S/logs/private-qwen2-bm32.log 2>&1; done_ qwen2 $?
stage counter_plan
SESSION=$S bash scripts/h200_gaps_session.sh --only thermal,calibrate,counter_plan > /workspace/session4-counter.log 2>&1; done_ counter_plan $?
touch /workspace/session4-chain2.DONE
