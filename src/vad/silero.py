"""Silero VAD as the reference system.

Silero scores 512-sample chunks (32 ms at 16 kHz). Our labels are 10 ms
frames with a 25 ms window, so each frame takes the score of the chunk that
contains its centre.
"""
from __future__ import annotations

import numpy as np
import torch

from .synth import HOP, SR, WIN, n_frames

CHUNK = 512


class Silero:
    def __init__(self):
        try:
            from silero_vad import load_silero_vad
        except ImportError as e:
            raise RuntimeError("pip install silero-vad") from e
        self.model = load_silero_vad()

    @torch.inference_mode()
    def chunk_probs(self, audio: np.ndarray) -> np.ndarray:
        self.model.reset_states()
        x = torch.from_numpy(audio.astype(np.float32))
        n = len(x) // CHUNK
        return np.array([float(self.model(x[i * CHUNK : (i + 1) * CHUNK], SR)) for i in range(n)])

    def frame_scores(self, audio: np.ndarray) -> np.ndarray:
        probs = self.chunk_probs(audio)
        nf = n_frames(len(audio))
        centres = HOP * np.arange(nf) + WIN // 2
        idx = np.minimum(centres // CHUNK, len(probs) - 1)
        return probs[idx]
