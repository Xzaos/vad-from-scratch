from __future__ import annotations

import numpy as np


def frame_f1(pred: np.ndarray, gold: np.ndarray) -> dict:
    pred, gold = pred.astype(bool), gold.astype(bool)
    tp = int(np.sum(pred & gold))
    fp = int(np.sum(pred & ~gold))
    fn = int(np.sum(~pred & gold))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    # Miss rate and false-alarm rate are what a downstream ASR feels: misses
    # cut words, false alarms feed noise to the recogniser.
    return {"precision": p, "recall": r, "f1": f, "miss": 1 - r, "false_alarm": fp / max(1, int(np.sum(~gold)))}


def roc_auc(scores: np.ndarray, gold: np.ndarray) -> float:
    """Mann-Whitney U form of AUC, with ties counted as half."""
    s, g = np.asarray(scores, float), np.asarray(gold).astype(bool)
    pos, neg = g.sum(), (~g).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[g].sum() - pos * (pos + 1) / 2) / (pos * neg))


def best_threshold(scores: np.ndarray, gold: np.ndarray, grid: np.ndarray) -> float:
    return float(max(grid, key=lambda t: frame_f1(scores > t, gold)["f1"]))
