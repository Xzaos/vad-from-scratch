#!/usr/bin/env bash
# Speech: Hindi and Tamil FLEURS read speech, clean enough for energy-based
# clip labels. Training clips come from FLEURS train, test clips from FLEURS
# test, so no sentence appears on both sides.
# Noise: ESC-50 minus the classes that contain human vocal sounds, since those
# would make the labels ambiguous.
set -euo pipefail
mkdir -p data
pip install -q huggingface_hub

if [ ! -d data/speech ]; then
  python scripts/get_fleurs.py hi_in train 1500 data/speech/hi
  python scripts/get_fleurs.py ta_in train 1500 data/speech/ta
fi
if [ ! -d data/speech_test ]; then
  python scripts/get_fleurs.py hi_in test 300 data/speech_test/hi
  python scripts/get_fleurs.py ta_in test 300 data/speech_test/ta
fi

if [ ! -d data/noise ]; then
  curl -L -o data/esc50.zip https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
  unzip -q data/esc50.zip -d data/
  python scripts/filter_esc50.py data/ESC-50-master data/noise
  rm -rf data/esc50.zip data/ESC-50-master
fi
echo "speech: $(find data/speech -name '*.wav' | wc -l) train, $(find data/speech_test -name '*.wav' | wc -l) test; noise: $(find data/noise -name '*.wav' | wc -l) files"
