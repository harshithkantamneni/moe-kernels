#!/bin/bash
# The moe-kernels Lambda plan on this VM, with no laptop: driver -> setup -> bytes
# -> timing chain -> extras (only if delivered) -> done. State in ~/autopilot.state,
# resumable; it never terminates the VM (the laptop copies results off first).
set -u
ST=~/autopilot.state; LOG=~/autopilot.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
stage() { cat "$ST" 2>/dev/null || echo driver; }
set_stage() { echo "$1" > "$ST"; say "stage -> $1"; }
COMMIT=${COMMIT:-f49a2133c942c20c3b1b72e002b03af1ccd216a3}
BUNDLE=${BUNDLE:-$HOME/moe3.bundle}
RATE=${RATE:-2.29}
exec 9>~/autopilot.lock
flock -n 9 || { echo "autopilot already running"; exit 0; }
while :; do
  case "$(stage)" in
    driver)
      v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
      if [ "${v:-0}" -ge 580 ]; then set_stage setup; continue; fi
      say "driver $v: installing nvidia-driver-580-server-open from Lambda's repo"
      sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> ~/driver580.log 2>&1
      if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-580-server-open >> ~/driver580.log 2>&1; then
        set_stage failed-driver; exit 1; fi
      set_stage setup; say "rebooting for the new driver"; sudo systemctl reboot; exit 0 ;;
    setup)
      if [ -f ~/moe/env.sh ] && grep -q "^EXIT 0" ~/setup_vm.log 2>/dev/null; then set_stage bytes; continue; fi
      bash ~/setup_vm.sh --bundle "$BUNDLE" --commit "$COMMIT" > ~/setup_vm.log 2>&1; rc=$?
      echo "EXIT $rc" >> ~/setup_vm.log
      [ $rc -eq 0 ] || { say "setup_vm rc=$rc"; set_stage failed-setup; exit 1; }
      set_stage bytes ;;
    bytes)
      if ! grep -q '^DONE' ~/run_counters.status 2>/dev/null; then
        if ! pgrep -f "bash .*run_counters.sh" > /dev/null; then bash ~/run_counters.sh > ~/run_counters.out 2>&1; fi
        until grep -q '^DONE' ~/run_counters.status 2>/dev/null; do sleep 30; done
      fi
      set_stage chain ;;
    chain)
      ( . ~/moe/env.sh; unset MOE_RESULTS_DIR; cd "$REPO" || exit 9
        sudo nvidia-smi -rgc > /dev/null 2>&1
        G_LADDER="1 2 4 16 64" SEEDS=0 RATE_USD_H=$RATE timeout 7h \
          bash scripts/alpha_g_chain.sh >> ~/moe/alpha_g_chain.out 2>&1 < /dev/null )
      say "chain rc=$?"; set_stage extras ;;
    extras)
      if [ -x ~/extras/run.sh ]; then timeout 3h bash ~/extras/run.sh >> ~/extras.log 2>&1; say "extras rc=$?"
      else say "no extras delivered"; fi
      set_stage done ;;
    done)
      sudo nvidia-smi -rgc > /dev/null 2>&1; touch ~/AUTOPILOT_DONE; say "ALL DONE: results wait on disk for the laptop to copy off"; exit 0 ;;
    *) say "stopped at $(stage)"; exit 1 ;;
  esac
done
