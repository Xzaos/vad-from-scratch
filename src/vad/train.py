"""Train TinyVAD on examples generated on the fly.

Every epoch sees fresh mixtures (new placement, new noise, new SNR), so the
training set is effectively unlimited and the model can't memorise
positions. The validation set is fixed by seed so epochs are comparable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .metrics import frame_f1, roc_auc
from .model import TinyVAD, n_params
from .synth import Synthesizer


class SynthStream(IterableDataset):
    def __init__(self, synth: Synthesizer, n: int, seed: int):
        self.synth, self.n, self.seed = synth, n, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        # One long iterator covers the whole run, so a fixed seed per worker
        # still gives fresh examples every step and a reproducible run.
        rng = np.random.default_rng([self.seed, wid])
        per_worker = self.n // (info.num_workers if info else 1)
        for _ in range(per_worker):
            ex = self.synth.make(rng)
            yield torch.from_numpy(ex.audio), torch.from_numpy(ex.labels.astype(np.float32))


def fixed_set(synth: Synthesizer, n: int, seed: int, snr_db=None):
    rng = np.random.default_rng(seed)
    return [synth.make(rng, snr_db=snr_db) for _ in range(n)]


@torch.inference_mode()
def predict(model: TinyVAD, audio: np.ndarray, device="cpu") -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(audio).unsqueeze(0).to(device))
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def evaluate(model, examples, device="cpu", thresh=0.5) -> dict:
    s = np.concatenate([predict(model, e.audio, device) for e in examples])
    g = np.concatenate([e.labels for e in examples])
    return {**frame_f1(s > thresh, g), "auc": roc_auc(s, g)}


def train(
    train_synth: Synthesizer,
    val_synth: Synthesizer,
    out_dir: str,
    epochs: int = 20,
    steps_per_epoch: int = 200,
    batch_size: int = 32,
    lr: float = 2e-3,
    hidden: int = 64,
    seed: int = 0,
    workers: int = 2,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyVAD(hidden=hidden).to(device)
    print(f"params: {n_params(model)}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)

    # Speech is usually the majority class in these mixtures; weight the
    # loss so neither class dominates.
    val = fixed_set(val_synth, 200, seed + 1000)
    pos_frac = float(np.mean(np.concatenate([e.labels for e in val])))
    pos_weight = torch.tensor((1 - pos_frac) / max(pos_frac, 1e-3), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(SynthStream(train_synth, epochs * steps_per_epoch * batch_size, seed), batch_size=batch_size, num_workers=workers)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    it = iter(loader)
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(steps_per_epoch):
            wav, lab = next(it)
            wav, lab = wav.to(device), lab.to(device)
            logits = model(wav)
            n = min(logits.shape[1], lab.shape[1])
            loss = loss_fn(logits[:, :n], lab[:, :n])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(model, val, device)
        row = {"epoch": ep, "loss": tot / steps_per_epoch, **m, "sec": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row))
        if m["f1"] > best:
            best = m["f1"]
            torch.save({"state": model.state_dict(), "hidden": hidden, "epoch": ep, "val": m}, out / "best.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    return out / "best.pt"


def load(path: str, device="cpu") -> TinyVAD:
    ck = torch.load(path, map_location=device)
    m = TinyVAD(hidden=ck["hidden"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()
