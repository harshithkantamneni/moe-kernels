#!/bin/bash
# The moe-kernels Lambda H100 plan on this VM, with no laptop:
#   driver -> setup -> bytes -> pre -> locked -> done
# pre:    the timing chain at G=1 seed 0 (preflight, thermal + calibrate, counter probe,
#         gpu-tests, probe-check, then its duty-0.25 R3 page at G=1, kept as the same-card
#         duty-vs-lock record), then the chain is stopped before R1.
# locked: ~/locked_r3.py finds the highest SM clock lock this card holds during R3's
#         bursts and runs R3 at G = 1 2 4 16 64 under it, then resets the clock.
# State in ~/autopilot.state, resumable; an @reboot crontab line restarts it after the
# driver reboot. It never terminates the VM: the laptop copies results off first.
# A failed stage touches ~/AUTOPILOT_FAILED so the laptop still copies off and terminates.
set -u
ST=~/autopilot.state; LOG=~/autopilot.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
stage() { cat "$ST" 2>/dev/null || echo driver; }
set_stage() { echo "$1" > "$ST"; say "stage -> $1"; }
fail() { set_stage "failed-$1"; sudo nvidia-smi -rgc > /dev/null 2>&1; touch ~/AUTOPILOT_FAILED; exit 1; }
COMMIT=${COMMIT:-f49a2133c942c20c3b1b72e002b03af1ccd216a3}
BUNDLE=${BUNDLE:-$HOME/moe3.bundle}
RATE=${RATE:-4.29}
exec 9>~/autopilot.lock
flock -n 9 || { echo "autopilot already running"; exit 0; }
say "autopilot start at stage $(stage)"
while :; do
  case "$(stage)" in
    driver)
      v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | cut -d. -f1)
      if [ "${v:-0}" -ge 580 ]; then set_stage setup; continue; fi
      say "driver ${v:-unread}: installing nvidia-driver-580-server-open from Lambda's repo"
      sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 update -qq >> ~/driver580.log 2>&1
      if ! sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y nvidia-driver-580-server-open >> ~/driver580.log 2>&1; then
        fail driver; fi
      set_stage setup; say "rebooting for the new driver"; sudo systemctl reboot; exit 0 ;;
    setup)
      v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | cut -d. -f1)
      [ "${v:-0}" -ge 580 ] || { say "driver after reboot: ${v:-unread}"; fail driver-after-reboot; }
      if [ -f ~/moe/env.sh ] && grep -q "^EXIT 0" ~/setup_vm.log 2>/dev/null; then set_stage bytes; continue; fi
      bash ~/setup_vm.sh --bundle "$BUNDLE" --commit "$COMMIT" > ~/setup_vm.log 2>&1; rc=$?
      echo "EXIT $rc" >> ~/setup_vm.log
      [ $rc -eq 0 ] || { say "setup_vm rc=$rc"; fail setup; }
      set_stage bytes ;;
    bytes)
      if ! grep -q '^DONE' ~/run_counters.status 2>/dev/null; then
        if ! pgrep -f "[b]ash .*run_counters.sh" > /dev/null; then timeout 3h bash ~/run_counters.sh > ~/run_counters.out 2>&1; fi
        t0=$(date +%s)
        until grep -q '^DONE' ~/run_counters.status 2>/dev/null; do
          [ $(( $(date +%s) - t0 )) -gt 10800 ] && { say "bytes: no DONE after 3 h"; break; }
          sleep 30; done
      fi
      sudo nvidia-smi -rgc > /dev/null 2>&1
      set_stage pre ;;
    pre)
      . ~/moe/env.sh; unset MOE_RESULTS_DIR
      sudo nvidia-smi -rgc > /dev/null 2>&1
      t0=$(date +%s)
      setsid bash -c "cd '$REPO' && G_LADDER=1 SEEDS=0 RATE_USD_H=$RATE bash scripts/alpha_g_chain.sh >> ~/moe/alpha_g_chain.out 2>&1 < /dev/null" &
      sleep 20
      while :; do
        S=$(ls -td "$SESSION_ROOT"/alpha_g-"$MOE_CARD"-*/ 2>/dev/null | head -1)
        if [ -n "$S" ] && awk -F'\t' '$1=="r3-g1-s0"{f=1} END{exit !f}' "$S/CHAIN.tsv" 2>/dev/null; then
          say "pre: r3-g1-s0 row written in $S; stopping the chain before R1"; break; fi
        if ! pgrep -f "[a]lpha_g_chain.sh" > /dev/null; then say "pre: the chain exited before r3-g1-s0"; break; fi
        if [ $(( $(date +%s) - t0 )) -gt 3600 ]; then say "pre: over 60 min; stopping the chain"; break; fi
        sleep 10
      done
      pid=$(pgrep -f "[a]lpha_g_chain.sh" | head -1)
      if [ -n "$pid" ]; then
        sid=$(ps -o sid= -p "$pid" | tr -d ' '); me=$(ps -o sid= -p $$ | tr -d ' ')
        if [ -n "$sid" ] && [ "$sid" != 0 ] && [ "$sid" != "$me" ]; then
          pkill -TERM -s "$sid"; sleep 20; pkill -KILL -s "$sid" 2>/dev/null; say "pre: chain session $sid stopped"
        fi
      fi
      n=0; until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || [ $n -gt 60 ]; do sleep 5; n=$((n+1)); done
      sudo nvidia-smi -rgc > /dev/null 2>&1
      [ -n "${S:-}" ] && cut -f1-5 "$S/CHAIN.tsv" >> "$LOG"
      if ls "$REPO/moe/bench/hardware/measured_${MOE_CARD}.yaml" > /dev/null 2>&1; then set_stage locked
      else say "pre: no calibration for $MOE_CARD; R3 cannot run"; fail pre; fi ;;
    locked)
      . ~/moe/env.sh
      S=$(ls -td "$SESSION_ROOT"/alpha_g-"$MOE_CARD"-*/ 2>/dev/null | head -1)
      MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD CHAIN_SESSION=$(basename "${S:-alpha_g-$MOE_CARD-nochain}") \
        python3 ~/locked_r3.py >> ~/locked_r3.out 2>&1; rc=$?
      sudo nvidia-smi -rgc > /dev/null 2>&1
      say "locked_r3.py rc=$rc"
      [ $rc -eq 0 ] || fail locked
      set_stage done ;;
    done)
      sudo nvidia-smi -rgc > /dev/null 2>&1; touch ~/AUTOPILOT_DONE; say "ALL DONE: results wait on disk for the laptop to copy off"; exit 0 ;;
    *) say "stopped at $(stage)"; touch ~/AUTOPILOT_FAILED; exit 1 ;;
  esac
done
