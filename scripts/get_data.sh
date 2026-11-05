#!/usr/bin/env bash
# Speech: Hindi and Tamil FLEURS train clips (read speech, clean enough for
# energy-based clip labels). Noise: ESC-50 minus the classes that contain
# human vocal sounds, since those would make the labels ambiguous.
set -euo pipefail
mkdir -p data

if [ ! -d data/speech ]; then
  pip install -q "asrbench[models] @ git+https://github.com/Xzaos/indic-asr-bench.git"
  asrbench prepare --dataset google/fleurs --config hi_in --split train --text-field raw_transcription --lang hi --n 1500 --tag hi --out data/speech
  asrbench prepare --dataset google/fleurs --config ta_in --split train --text-field raw_transcription --lang ta --n 1500 --tag ta --out data/speech
fi

if [ ! -d data/noise ]; then
  curl -L -o data/esc50.zip https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
  unzip -q data/esc50.zip -d data/
  python scripts/filter_esc50.py data/ESC-50-master data/noise
fi
echo "speech: $(find data/speech -name '*.wav' | wc -l) files, noise: $(find data/noise -name '*.wav' | wc -l) files"
