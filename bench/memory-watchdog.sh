#!/usr/bin/env bash
# Kill the case that is about to take the box, so the run survives it.
#
# The memory budget in `bench/conftest.py` is a projection made *between* rungs:
# it multiplies the rung that just finished by the next one's growth factor and
# stops the arm if the product is over. It cannot see a cell while that cell
# runs, and it counts one copy where a measurement holds two — the timed rounds
# in the pytest process, and `benchmem(isolate=True)` again in a child, with
# glibc returning neither to the OS in between. So `transport/w100` projected
# 8.4 GB from `w10`, took 14.3 of its own, needed about twice that of the
# machine, and took a 32 GB box down twice (runs 12 and 16 of the published
# benchmark).
#
# This is a backstop rather than the fix. It samples, so it cannot catch an
# allocation faster than its interval — run 16 climbed 10 GB between two
# one-second samples, which is why the default here is four times a second. A
# cgroup cap or a box with room for both copies is what would *guarantee* it.
# What this buys is the difference between a dead case and a dead runner: a
# killed case leaves the ones after it their turn, and leaves `report`, `plot`
# and the artifact something to run on.
set -uo pipefail

#: The running case, by the flag only its pytest carries.
CASE='--benchmark-memory'
#: **The memory is in the child, and the child cannot be found by that flag.**
#: `benchmem(isolate=True)` measures in a `multiprocessing` *spawn*, so the
#: process holding the model is `python -c 'from multiprocessing.spawn import
#: spawn_main…'` with none of pytest's arguments on it. Killing the pytest
#: alone orphans it, still holding its 14 GB — which is run 18: `transport` was
#: killed at 24 GB used, `storage` started on a box that had freed nothing, and
#: the runner died fifteen seconds later.
SPAWNED='multiprocessing.spawn import spawn_main'

available() { free -m | awk '/^Mem:/{print $7}'; }

# Children first: once the pytest is gone its child is reparented, and only its
# own argv is left to find it by.
stop_the_case() {
  local signal=$1 pid
  for pid in $(pgrep -f -- "$CASE" 2>/dev/null); do
    pkill "-$signal" -P "$pid" 2>/dev/null || true
  done
  pkill "-$signal" -f -- "$CASE" 2>/dev/null || true
  pkill "-$signal" -f -- "$SPAWNED" 2>/dev/null || true
}

# `free` is procps, so this samples on Linux and nowhere else. Standing down is
# the honest answer on a machine it cannot watch: the ladder is runnable by hand
# and a watchdog that exits non-zero would take the run with it.
if ! command -v free >/dev/null 2>&1; then
  echo "memory watchdog: no \`free\` here, so nothing is watching — a cell too big for this machine will take it down"
  exit 0
fi

#: A killed cell is a measurement, not an accident: one library needing the
#: whole machine where another needs half a gigabyte is what the ladder is for.
#: The process that would record it is the one being killed, so the cell's name
#: comes from the breadcrumb `bench/conftest.py` writes before each test, and
#: the record is appended here — where it survives the kill and rides out with
#: the case's artifact.
INFLIGHT=bench/results/.inflight
CASUALTIES=${BENCH_CASUALTIES:-bench/results/casualties.json}

record_the_casualty() {
  local avail=$1 peak=$2 cell
  cell=$(cat "$INFLIGHT" 2>/dev/null) || cell=''
  [ -n "$cell" ] || cell='unknown'
  mkdir -p "$(dirname "$CASUALTIES")"
  [ -s "$CASUALTIES" ] || printf '[]' > "$CASUALTIES"
  python3 - "$CASUALTIES" "$cell" "$avail" "$peak" <<'PYEOF'
import json, sys
path, cell, avail, peak = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = json.loads(open(path).read() or '[]')
rows.append({'record': 'casualty', 'cell': cell, 'available_mb': avail, 'peak_mb': peak})
open(path, 'w').write(json.dumps(rows, indent=1))
PYEOF
  echo "recorded: ${cell} did not fit — ${avail} MB free at the kill, ${peak} MB high-water"
}

interval=${BENCH_MEMORY_SAMPLE_SECONDS:-0.25}
#: A line on the clock as well as on a new maximum. The high-water mark prints
#: only when it moves, so a quiet watchdog and a dead one read alike — run 19
#: went six minutes without a word and it took an orphaned `sleep` in the
#: runner's cleanup to establish it had been sampling the whole time.
heartbeat=${BENCH_MEMORY_HEARTBEAT_SECONDS:-60}
total=$(free -m | awk '/^Mem:/{print $2}')
floor=${BENCH_MEMORY_FLOOR_MB:-$((total / 4))}
echo "memory watchdog: ${total} MB total, a case is killed under ${floor} MB available"

peak=0
beat=$SECONDS
while sleep "$interval"; do
  read -r used avail <<<"$(free -m | awk '/^Mem:/{print $3, $7}')"
  if [ "$used" -gt "$peak" ]; then
    peak=$used
    echo "MEM high-water ${peak} MB"
  fi
  if [ $((SECONDS - beat)) -ge "$heartbeat" ]; then
    beat=$SECONDS
    echo "MEM ${used} MB used, ${avail} MB available, high-water ${peak} MB"
  fi
  [ "$avail" -ge "$floor" ] && continue

  echo "MEM ${avail} MB available, under the ${floor} MB floor — killing this case before it takes the box"
  record_the_casualty "$avail" "$peak"
  stop_the_case TERM
  # Never a blind sleep here. The next case starts the moment this one dies, so
  # the seconds after a kill are exactly when the box is still full and still
  # falling — run 18 died inside a 30-second one.
  waited=0
  while :; do
    sleep "$interval"
    waited=$((waited + 1))
    avail=$(available)
    if [ "$avail" -ge "$floor" ]; then
      echo "MEM ${avail} MB available again — the ladder goes on"
      break
    fi
    if [ "$waited" -eq 20 ]; then
      echo "MEM ${avail} MB still under the floor — the case has not let go, killing harder"
      stop_the_case KILL
    fi
    if [ "$waited" -ge 240 ]; then
      echo "MEM ${avail} MB still under the floor with nothing left to kill — the box is on its own"
      break
    fi
  done
done
