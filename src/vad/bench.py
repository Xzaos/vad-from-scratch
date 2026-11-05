"""Compare the energy baseline, TinyVAD and Silero on held-out mixtures.

For each SNR the test set is regenerated from the held-out speech and noise
files with a fixed seed, so every system sees identical audio. Thresholds
are tuned once on the validation set and then frozen; tuning on the test
set would flatter whichever system has the flattest score distribution.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import baseline
from .metrics import best_threshold, frame_f1, roc_auc
from .model import Streamer, n_params
from .train import fixed_set, predict


def _systems(model, use_silero):
    sys = {
        "energy": lambda a: baseline.score(a),
        "tinyvad": lambda a: predict(model, a),
    }
    if use_silero:
        from .silero import Silero

        s = Silero()
        sys["silero"] = s.frame_scores
    return sys


def latency(model, n_frames: int = 2000) -> dict:
    """CPU wall time per 10 ms frame in streaming mode, single thread."""
    torch.set_num_threads(1)
    st = Streamer(model)
    frames = torch.randn(n_frames, model.features.mel.n_mels)
    for f in frames[:100]:
        st.step(f)
    t0 = time.perf_counter()
    for f in frames:
        st.step(f)
    ms = (time.perf_counter() - t0) / n_frames * 1000
    return {"ms_per_10ms_frame": ms, "real_time_factor": ms / 10, "params": n_params(model)}


def run(model, val_synth, test_synth, snrs=(20, 10, 5, 0, -5), n_per_snr: int = 100, use_silero: bool = True, out: str = "results") -> dict:
    systems = _systems(model, use_silero)
    val = fixed_set(val_synth, 100, 1234)
    gold_v = np.concatenate([e.labels for e in val])
    thresholds = {}
    for name, fn in systems.items():
        s = np.concatenate([fn(e.audio) for e in val])
        grid = np.quantile(s, np.linspace(0.02, 0.98, 97))
        thresholds[name] = best_threshold(s, gold_v, grid)

    results = {"thresholds": thresholds, "by_snr": {}, "latency": latency(model)}
    for snr in snrs:
        test = fixed_set(test_synth, n_per_snr, 5000 + int(snr * 10), snr_db=snr)
        gold = np.concatenate([e.labels for e in test])
        row = {}
        for name, fn in systems.items():
            s = np.concatenate([fn(e.audio) for e in test])
            row[name] = {**frame_f1(s > thresholds[name], gold), "auc": roc_auc(s, gold)}
        results["by_snr"][str(snr)] = row
        print(snr, {k: round(v["f1"], 3) for k, v in row.items()})

    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "bench.json").write_text(json.dumps(results, indent=1))
    Path(out, "bench.md").write_text(to_markdown(results))
    return results


def to_markdown(res: dict) -> str:
    names = list(next(iter(res["by_snr"].values())).keys())
    lines = ["| SNR (dB) | " + " | ".join(f"{n} F1" for n in names) + " | " + " | ".join(f"{n} AUC" for n in names) + " |"]
    lines.append("|" + "---|" * (1 + 2 * len(names)))
    for snr, row in res["by_snr"].items():
        lines.append(f"| {snr} | " + " | ".join(f"{row[n]['f1']:.3f}" for n in names) + " | " + " | ".join(f"{row[n]['auc']:.3f}" for n in names) + " |")
    lat = res["latency"]
    lines.append(f"\nTinyVAD: {lat['params']} parameters, {lat['ms_per_10ms_frame']:.3f} ms per 10 ms frame on one CPU thread (RTF {lat['real_time_factor']:.4f}).")
    return "\n".join(lines) + "\n"
