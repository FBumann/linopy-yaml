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

interval=${BENCH_MEMORY_SAMPLE_SECONDS:-0.25}
total=$(free -m | awk '/^Mem:/{print $2}')
floor=${BENCH_MEMORY_FLOOR_MB:-$((total / 4))}
echo "memory watchdog: ${total} MB total, a case is killed under ${floor} MB available"

peak=0
while sleep "$interval"; do
  read -r used avail <<<"$(free -m | awk '/^Mem:/{print $3, $7}')"
  if [ "$used" -gt "$peak" ]; then
    peak=$used
    echo "MEM high-water ${peak} MB"
  fi
  if [ "$avail" -lt "$floor" ]; then
    echo "MEM ${avail} MB available, under the ${floor} MB floor — killing this case before it takes the box"
    # The pytest of the running case and nothing else: `ladder-ci`'s own shell
    # carries the case list in its argv and has to outlive this to reach them.
    pkill -f -- '--benchmark-memory' || true
    sleep 30
  fi
done
