"""Compare the energy baseline, TinyVAD and Silero on held-out mixtures.

For each SNR the test set is regenerated from the held-out speech and noise
files with a fixed seed, so every system sees identical audio. Thresholds
are tuned once on the validation set and then frozen; tuning on the test
set would flatter whichever system has the flattest score distribution.

There are two test conditions. "seen" uses held-out recordings from noise
classes that also appear in training. "unseen" uses noise classes the model
never trained on. TinyVAD is trained on this exact mixing recipe and Silero
is not, so the seen condition favours TinyVAD; the unseen condition is the
fairer comparison.

Differences between systems come with a 95% paired bootstrap interval over
test clips: both systems are rescored on the same resampled clips, so the
interval reflects which clips are hard as well as how many frames there are.
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

SNRS = (20, 10, 5, 0, -5)


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


def clip_counts(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """(tp, fp, fn) for one clip."""
    p, g = pred.astype(bool), gold.astype(bool)
    return np.array([np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)], dtype=np.int64)


def _f1(c: np.ndarray) -> np.ndarray:
    """F1 from summed counts; c has (tp, fp, fn) on the last axis."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def paired_f1_diff(ca: np.ndarray, cb: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """F1(a) - F1(b) with a 95% bootstrap interval. ca, cb: (n_clips, 3)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ca), size=(n_boot, len(ca)))
    diffs = _f1(ca[idx].sum(axis=1)) - _f1(cb[idx].sum(axis=1))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(_f1(ca.sum(0)) - _f1(cb.sum(0))), "lo": float(lo), "hi": float(hi)}


def score_condition(systems, thresholds, synth, snrs, n_per_snr, seed_base) -> dict:
    out = {}
    for snr in snrs:
        test = fixed_set(synth, n_per_snr, seed_base + int(snr * 10), snr_db=snr)
        gold = np.concatenate([e.labels for e in test])
        row, counts = {}, {}
        for name, fn in systems.items():
            per_clip = [fn(e.audio) for e in test]
            s = np.concatenate(per_clip)
            row[name] = {**frame_f1(s > thresholds[name], gold), "auc": roc_auc(s, gold)}
            counts[name] = np.stack([clip_counts(p > thresholds[name], e.labels) for p, e in zip(per_clip, test)])
        diffs = {}
        for other in ("silero", "energy"):
            if other in counts:
                diffs[f"tinyvad-{other}"] = paired_f1_diff(counts["tinyvad"], counts[other], seed=1000 + int(snr * 10))
        out[str(snr)] = {"systems": row, "f1_diff": diffs}
        print(snr, {k: round(v["f1"], 3) for k, v in row.items()}, {k: round(v["diff"], 3) for k, v in diffs.items()})
    return out


def run(model, val_synth, test_synth, unseen_synth=None, snrs=SNRS, n_per_snr: int = 100, use_silero: bool = True, out: str = "results") -> dict:
    systems = _systems(model, use_silero)
    val = fixed_set(val_synth, 100, 1234)
    gold_v = np.concatenate([e.labels for e in val])
    thresholds = {}
    for name, fn in systems.items():
        s = np.concatenate([fn(e.audio) for e in val])
        grid = np.quantile(s, np.linspace(0.02, 0.98, 97))
        thresholds[name] = best_threshold(s, gold_v, grid)

    results = {"thresholds": thresholds, "n_per_snr": n_per_snr, "conditions": {}, "latency": latency(model)}
    print("seen noise classes")
    results["conditions"]["seen"] = score_condition(systems, thresholds, test_synth, snrs, n_per_snr, 5000)
    if unseen_synth is not None:
        print("unseen noise classes")
        results["conditions"]["unseen"] = score_condition(systems, thresholds, unseen_synth, snrs, n_per_snr, 7000)

    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "bench.json").write_text(json.dumps(results, indent=1))
    Path(out, "bench.md").write_text(to_markdown(results))
    return results


def _table(cond: dict) -> list[str]:
    names = list(next(iter(cond.values()))["systems"].keys())
    diff_keys = list(next(iter(cond.values()))["f1_diff"].keys())
    head = ["SNR (dB)"] + [f"{n} F1" for n in names] + [f"{n} AUC" for n in names] + [f"{k} F1 (95% CI)" for k in diff_keys]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for snr, r in cond.items():
        cells = [snr]
        cells += [f"{r['systems'][n]['f1']:.3f}" for n in names]
        cells += [f"{r['systems'][n]['auc']:.3f}" for n in names]
        cells += [f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" for d in (r["f1_diff"][k] for k in diff_keys)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def to_markdown(res: dict) -> str:
    lines = []
    titles = {"seen": "Held-out recordings, noise classes seen in training", "unseen": "Noise classes never seen in training"}
    for key, cond in res["conditions"].items():
        lines += [f"### {titles.get(key, key)}", ""] + _table(cond) + [""]
    lat = res["latency"]
    lines.append(f"{res['n_per_snr']} test mixtures per SNR. TinyVAD: {lat['params']} parameters, {lat['ms_per_10ms_frame']:.3f} ms per 10 ms frame on one CPU thread (RTF {lat['real_time_factor']:.4f}).")
    return "\n".join(lines) + "\n"
