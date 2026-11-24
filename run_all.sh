#!/bin/bash
# runs the pointing study with the GPU heat watchdog (stops at 84 C; the run is resumable)
cd "$(dirname "$0")"
python -I point_study.py run > results/run.log 2>&1 &
P=$!; sleep 5; python -I watchdog.py "$(cat /proc/$P/winpid)" &
wait $P
