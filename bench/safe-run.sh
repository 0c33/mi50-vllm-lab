#!/bin/bash
# Run a GPU job under a temperature watchdog (mandatory for every test/benchmark).
#
#   safe-run.sh [-l LIMIT_C] [-s] [-n] -- <command...>      (-n: skip the start check; tests only)
#
# Polls every sensor (edge, junction, HBM) of both MI50s 4x per second. The job is
# stopped the moment any sensor reaches LIMIT_C (default 80), or a card's sensors can no
# longer be read (card dropping off the bus):
#   - the command's whole process group is killed,
#   - every container started with  --label gfxjob=1  is killed (docker run benchmarks),
#   - with -s also the vLLM server container (vllm-qwen38), for tests that load the server.
# The LXC cannot lower power caps (host guard does that); this only removes the load.
# Log: $SAFE_RUN_LOG (default ./safe-run.log): start/stop, peak per sensor, trips.
set -u
LIMIT=80
SERVER=0
NOCHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    -l) LIMIT=$2; shift 2 ;;
    -s) SERVER=1; shift ;;
    -n) NOCHECK=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: safe-run.sh [-l LIMIT_C] [-s] -- <command...>"; exit 2; }
LOG=${SAFE_RUN_LOG:-safe-run.log}
SENSORS=$(ls /sys/class/drm/card[01]/device/hwmon/hwmon*/temp[123]_input 2>/dev/null)
[ "$(echo "$SENSORS" | wc -w)" -eq 6 ] || { echo "safe-run: expected 6 sensors, found: $SENSORS" | tee -a "$LOG"; exit 3; }

stop_all() {  # $1 = reason
  echo "$(date +%T) TRIP: $1 -> stopping load" | tee -a "$LOG"
  kill -KILL -- -"$PGID" 2>/dev/null
  ids=$(docker ps -q --filter label=gfxjob=1)
  [ -n "$ids" ] && docker kill $ids >/dev/null 2>&1
  [ "$SERVER" = 1 ] && docker kill vllm-qwen38 >/dev/null 2>&1
}

# hottest sensor right now; refuse to start a job on hot cards
hottest() {
  local m=0 v t
  for t in $SENSORS; do
    v=$(cat "$t" 2>/dev/null) || { echo -1; return; }
    v=$((v / 1000)); [ "$v" -gt "$m" ] && m=$v
  done
  echo $m
}
h=$(hottest)
if [ "$h" -lt 0 ] || { [ "$NOCHECK" = 0 ] && [ "$h" -ge $((LIMIT - 20)) ]; }; then
  echo "$(date +%T) refusing to start: hottest sensor ${h} C (must be < $((LIMIT - 20)) C)" | tee -a "$LOG"
  exit 4
fi

echo "$(date +%T) START limit ${LIMIT} C server=${SERVER}: $*" >> "$LOG"
setsid "$@" &
PID=$!
PGID=$PID
declare -A PEAK
rc=0
while kill -0 "$PID" 2>/dev/null; do
  for t in $SENSORS; do
    if ! v=$(cat "$t" 2>/dev/null); then
      stop_all "cannot read $t (card unreachable?)"; rc=5; break 2
    fi
    v=$((v / 1000))
    [ "${PEAK[$t]:-0}" -lt "$v" ] && PEAK[$t]=$v
    if [ "$v" -ge "$LIMIT" ]; then
      stop_all "$t = ${v} C >= ${LIMIT} C"; rc=6; break 2
    fi
  done
  sleep 0.25
done
if [ $rc = 0 ]; then wait "$PID"; rc=$?; fi
peaks=""
for t in $SENSORS; do peaks+="$(echo "$t" | grep -oE 'card[01]').$(basename "$t" _input)=${PEAK[$t]:-?} "; done
echo "$(date +%T) END rc=$rc peaks: $peaks" | tee -a "$LOG"
exit $rc
