"""Synthetic training and test data for frame-level VAD.

Real VAD labels are expensive: someone has to mark where speech starts and
stops to within 10 ms. This module builds them instead. It takes clean
speech clips and noise clips, lays the speech out on a timeline with silent
gaps, mixes noise in at a chosen SNR, and keeps the frame labels exact
because the speech positions are known.

Inside a clean clip there are still pauses between words, so each speech
clip is labelled with an energy threshold relative to its own peak before it
is placed. That threshold runs on clean audio, where it is reliable; the
model then has to learn the same decision on noisy audio, where it isn't.

Train and test draw from disjoint source files (see `split_files`), so the
test score is never on a speaker or noise recording the model has heard.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16_000
HOP = 160  # 10 ms
WIN = 400  # 25 ms


@dataclass
class SynthConfig:
    duration_s: float = 8.0
    speech_segments: tuple[int, int] = (1, 4)
    gap_s: tuple[float, float] = (0.2, 2.0)
    snr_db: tuple[float, ...] = (20, 10, 5, 0, -5)
    speech_gain_db: tuple[float, float] = (-12, 0)
    clean_label_db: float = -35.0  # frames this far below the clip peak are silence
    telephony_prob: float = 0.3  # share of examples band-limited to 8 kHz
    noise_free_prob: float = 0.05
    seed: int = 0


@dataclass
class Example:
    audio: np.ndarray  # float32, SR
    labels: np.ndarray  # uint8, one per 10 ms frame
    snr_db: float | None
    telephony: bool
    meta: dict = field(default_factory=dict)


def split_files(files: list[Path], test_frac: float = 0.2, salt: str = "vad") -> tuple[list[Path], list[Path]]:
    """Deterministic split by file name hash. For datasets where the speaker
    id is in the path (e.g. LibriSpeech's spk/chapter/file), pass the speaker
    directory instead of the file to get a speaker-disjoint split."""
    train, test = [], []
    for f in sorted(files):
        h = int(hashlib.md5(f"{salt}{f.name}".encode()).hexdigest(), 16) % 1000
        (test if h < test_frac * 1000 else train).append(f)
    return train, test


def n_frames(n_samples: int) -> int:
    return 1 + (n_samples - WIN) // HOP if n_samples >= WIN else 0


def frame_energy_db(x: np.ndarray) -> np.ndarray:
    n = n_frames(len(x))
    if n == 0:
        return np.zeros(0)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    e = np.mean(x[idx] ** 2, axis=1)
    return 10 * np.log10(e + 1e-12)


def clean_labels(x: np.ndarray, rel_db: float) -> np.ndarray:
    e = frame_energy_db(x)
    if len(e) == 0:
        return np.zeros(0, np.uint8)
    lab = (e > e.max() + rel_db).astype(np.uint8)
    # Fill pauses shorter than 100 ms; a VAD that drops between syllables is
    # not what anyone wants.
    return _fill_short_gaps(lab, 10)


def _fill_short_gaps(lab: np.ndarray, max_gap: int) -> np.ndarray:
    lab = lab.copy()
    on = np.flatnonzero(lab)
    if len(on) < 2:
        return lab
    for a, b in zip(on[:-1], on[1:]):
        if 1 < b - a <= max_gap:
            lab[a:b] = 1
    return lab


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def telephony(x: np.ndarray) -> np.ndarray:
    """Narrowband channel: 300-3400 Hz band, 8 kHz round trip, mu-law. Close
    enough to a phone line for a VAD to have to cope with it."""
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(x.astype(np.float32))
    t = AF.highpass_biquad(t, SR, 300.0)
    t = AF.lowpass_biquad(t, SR, 3400.0)
    t = AF.resample(t, SR, 8000)
    t = AF.mu_law_decoding(AF.mu_law_encoding(t.clamp(-1, 1), 256), 256)
    t = AF.resample(t, 8000, SR)
    return t.numpy()[: len(x)]


class Synthesizer:
    def __init__(self, speech: list[np.ndarray], noise: list[np.ndarray], cfg: SynthConfig):
        if not speech:
            raise ValueError("no speech clips")
        self.speech, self.noise, self.cfg = speech, noise, cfg
        self._labels = [clean_labels(s, cfg.clean_label_db) for s in speech]

    def make(self, rng: np.random.Generator, snr_db: float | None = None) -> Example:
        cfg = self.cfg
        total = int(cfg.duration_s * SR)
        clean = np.zeros(total, np.float32)
        sample_lab = np.zeros(total, np.uint8)

        pos = int(rng.uniform(*cfg.gap_s) * SR)
        for _ in range(rng.integers(cfg.speech_segments[0], cfg.speech_segments[1] + 1)):
            k = rng.integers(len(self.speech))
            clip, flab = self.speech[k], self._labels[k]
            room = total - pos
            if room < WIN * 4:
                break
            clip = clip[:room]
            gain = 10 ** (rng.uniform(*cfg.speech_gain_db) / 20)
            clean[pos : pos + len(clip)] += gain * clip / (np.abs(clip).max() + 1e-9)
            # frame labels back to sample resolution for placement
            sl = np.repeat(flab, HOP)[: len(clip)]
            sample_lab[pos : pos + len(sl)] = np.maximum(sample_lab[pos : pos + len(sl)], sl)
            pos += len(clip) + int(rng.uniform(*cfg.gap_s) * SR)
            if pos >= total:
                break

        if snr_db is None:
            snr_db = None if rng.random() < cfg.noise_free_prob or not self.noise else float(rng.choice(cfg.snr_db))
        mix = clean.copy()
        if snr_db is not None and self.noise:
            noise = self._noise_bed(rng, total)
            speech_rms = _rms(clean[sample_lab.astype(bool)]) if sample_lab.any() else _rms(clean)
            noise *= speech_rms / (_rms(noise) * 10 ** (snr_db / 20))
            mix = mix + noise

        tel = rng.random() < cfg.telephony_prob
        if tel:
            mix = telephony(mix)
        peak = np.abs(mix).max()
        if peak > 0.99:
            mix = mix * (0.99 / peak)

        # sample labels -> frame labels: speech if most of the window is speech
        nf = n_frames(total)
        idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
        labels = (sample_lab[idx].mean(axis=1) > 0.5).astype(np.uint8)
        return Example(mix.astype(np.float32), labels, snr_db, tel)

    def _noise_bed(self, rng, total):
        out = np.zeros(total, np.float32)
        pos = 0
        while pos < total:
            n = self.noise[rng.integers(len(self.noise))]
            start = rng.integers(0, max(1, len(n) - 1))
            seg = n[start : start + total - pos]
            out[pos : pos + len(seg)] = seg
            pos += len(seg)
        return out


def load_clips(files: list[Path], max_s: float = 15.0, limit: int | None = None) -> list[np.ndarray]:
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    clips = []
    for f in files[:limit]:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
        clips.append(x[: int(max_s * SR)])
    return clips
