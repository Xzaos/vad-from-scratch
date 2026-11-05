#!/usr/bin/env bash
# Train and benchmark. ~20 min on a T4, about an hour on a laptop CPU.
set -euo pipefail
ARGS="--speech data/speech --noise data/noise"

vad dump $ARGS --n 8                       # listen to a few mixtures first
vad train $ARGS --epochs 20 --steps 200 --out results/run
vad bench $ARGS --ckpt results/run/best.pt --n 100 --out results
cat results/bench.md
