#!/bin/bash
# Session 4 chain 3: the WHOLE suite on the GPU box, uncapped, with a per-test summary.
REPO=/workspace/moe-kernels
PY=/workspace/venvs/base/bin/python
LOG=/workspace/session4-chain3.log
cd $REPO
echo "$(date -u +%H:%M:%S) START pytest-full" >> $LOG
PYTHONPATH=$REPO $PY -m pytest -q -p no:cacheprovider -rfE --durations=15 > /workspace/session4-pytest-full.log 2>&1
echo "$(date -u +%H:%M:%S) END   pytest-full rc=$?" >> $LOG
touch /workspace/session4-chain3.DONE
