"""Copy ESC-50 clips into noise/<class>/, skipping human vocal classes.

One folder per class lets `vad --unseen-noise` hold whole classes out of
training. File names are kept as-is, because the split groups takes of the
same source recording by parsing them (see synth.source_key).
"""
import csv
import shutil
import sys
from pathlib import Path

VOCAL = {"crying_baby", "laughing", "sneezing", "coughing", "snoring", "breathing"}
# Listen to a sample of what's left before training (`vad dump`); if other
# classes turn out to contain voices, add them here.

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
kept = 0
with open(src / "meta" / "esc50.csv") as f:
    for row in csv.DictReader(f):
        if row["category"] in VOCAL:
            continue
        d = dst / row["category"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "audio" / row["filename"], d / row["filename"])
        kept += 1
print(f"kept {kept} noise clips")
