"""Download FLEURS audio for one language and split, and keep the first N wavs.

Reads the tarball straight from the Hugging Face repo, so it needs neither
the `datasets` loading script (dropped in datasets 4.x) nor anything else
from this project.

    python scripts/get_fleurs.py hi_in train 1500 data/speech/hi
"""
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

lang, split, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
tgz = hf_hub_download("google/fleurs", f"data/{lang}/audio/{split}.tar.gz", repo_type="dataset")

kept = 0
with tarfile.open(tgz) as tf:
    for m in tf:
        if not (m.isfile() and m.name.endswith(".wav")):
            continue
        # Only the bytes are written, under the bare file name, so nothing in
        # the archive can choose where it lands.
        (out / Path(m.name).name).write_bytes(tf.extractfile(m).read())
        kept += 1
        if kept >= n:
            break
print(f"{lang}/{split}: kept {kept} wavs in {out}")
