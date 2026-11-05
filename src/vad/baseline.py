"""Energy + zero-crossing VAD with an adaptive noise floor and hangover.

No training. The noise floor is a low percentile of frame energy over the
clip, so it adapts to the recording, and the score is how far a frame sits
above that floor. Zero-crossing rate vetoes frames that are loud but look
like broadband noise (lots of crossings, little energy below 1 kHz).

Use the score for ROC curves and `decide` for the hard decision.
"""
from __future__ import annotations

import numpy as np

from .synth import HOP, WIN, frame_energy_db, n_frames


def zcr(x: np.ndarray) -> np.ndarray:
    n = n_frames(len(x))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    s = np.sign(x[idx])
    return np.mean(np.abs(np.diff(s, axis=1)) > 0, axis=1)


def score(x: np.ndarray, floor_pct: float = 10.0, zcr_max: float = 0.35) -> np.ndarray:
    e = frame_energy_db(x)
    floor = np.percentile(e, floor_pct)
    s = e - floor
    s[zcr(x) > zcr_max] -= 10.0
    return s


def decide(x: np.ndarray, thresh_db: float = 9.0, hangover: int = 15, min_on: int = 3) -> np.ndarray:
    raw = score(x) > thresh_db
    out = np.zeros_like(raw, dtype=np.uint8)
    run, hold = 0, 0
    for i, v in enumerate(raw):
        run = run + 1 if v else 0
        if run >= min_on:
            out[i - run + 1 : i + 1] = 1
            hold = hangover
        elif hold > 0:
            out[i] = 1
            hold -= 1
    return out
