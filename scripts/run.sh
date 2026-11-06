#!/usr/bin/env bash
# Train and benchmark. About 20 min of training on a T4, plus a few minutes
# of benchmarking (Silero runs on CPU, one 32 ms chunk at a time).
set -euo pipefail
ARGS="--speech data/speech --speech-test data/speech_test --noise data/noise"

vad dump $ARGS --n 8                       # listen to a few mixtures first
vad train $ARGS --epochs 20 --steps 200 --out results/run
vad bench $ARGS --ckpt results/run/best.pt --n 100 --out results
cat results/bench.md
