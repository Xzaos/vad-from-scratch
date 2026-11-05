"""Copy ESC-50 clips into a noise folder, skipping human vocal classes."""
import csv
import shutil
import sys
from pathlib import Path

VOCAL = {"crying_baby", "laughing", "sneezing", "coughing", "snoring", "breathing"}
# Listen to a sample of what's left before training (`vad dump`); if other
# classes turn out to contain voices, add them here.

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
kept = 0
with open(src / "meta" / "esc50.csv") as f:
    for row in csv.DictReader(f):
        if row["category"] in VOCAL:
            continue
        shutil.copy(src / "audio" / row["filename"], dst / row["filename"])
        kept += 1
print(f"kept {kept} noise clips")
