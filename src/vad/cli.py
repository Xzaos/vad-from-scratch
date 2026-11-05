from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .synth import SynthConfig, Synthesizer, load_clips, split_files


def _audio_files(root: str) -> list[Path]:
    return sorted(p for ext in ("wav", "flac", "mp3") for p in Path(root).rglob(f"*.{ext}"))


def _synths(a):
    speech_tr, speech_te = split_files(_audio_files(a.speech))
    noise_tr, noise_te = split_files(_audio_files(a.noise), salt="noise") if a.noise else ([], [])
    # Validation comes out of the training files; the test files stay unseen
    # until bench.
    k = max(1, len(speech_tr) // 10)
    speech_val, speech_tr = speech_tr[:k], speech_tr[k:]
    kn = max(1, len(noise_tr) // 10) if noise_tr else 0
    noise_val, noise_tr = noise_tr[:kn], noise_tr[kn:]
    print(f"speech files train/val/test: {len(speech_tr)}/{len(speech_val)}/{len(speech_te)}; noise: {len(noise_tr)}/{len(noise_val)}/{len(noise_te)}")
    cfg = SynthConfig(telephony_prob=a.telephony_prob)
    load = lambda fs: load_clips(fs, limit=a.limit)
    return (
        Synthesizer(load(speech_tr), load(noise_tr), cfg),
        Synthesizer(load(speech_val), load(noise_val), cfg),
        Synthesizer(load(speech_te), load(noise_te), cfg),
    )


def cmd_train(a):
    from .train import train

    tr, va, _ = _synths(a)
    train(tr, va, a.out, epochs=a.epochs, steps_per_epoch=a.steps, batch_size=a.batch_size, hidden=a.hidden, workers=a.workers)


def cmd_bench(a):
    from .bench import run
    from .train import load

    _, va, te = _synths(a)
    run(load(a.ckpt), va, te, n_per_snr=a.n, use_silero=not a.no_silero, out=a.out)


def cmd_dump(a):
    """Write a few synthetic examples to wav + label files for listening."""
    import soundfile as sf

    tr, _, _ = _synths(a)
    rng = np.random.default_rng(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        ex = tr.make(rng)
        sf.write(out / f"ex{i:02d}_snr{ex.snr_db}_tel{int(ex.telephony)}.wav", ex.audio, 16000)
        np.save(out / f"ex{i:02d}_labels.npy", ex.labels)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vad")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--speech", required=True, help="folder of clean speech (wav/flac)")
    common.add_argument("--noise", help="folder of noise (MUSAN noise/, ESC-50 audio/)")
    common.add_argument("--telephony-prob", type=float, default=0.3)
    common.add_argument("--limit", type=int, help="cap clips per split, for quick runs")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("train", parents=[common])
    s.add_argument("--out", default="results/run")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--hidden", type=int, default=64)
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("bench", parents=[common])
    s.add_argument("--ckpt", required=True)
    s.add_argument("--n", type=int, default=100, help="test mixtures per SNR")
    s.add_argument("--no-silero", action="store_true")
    s.add_argument("--out", default="results")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("dump", parents=[common])
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--out", default="results/samples")
    s.set_defaults(fn=cmd_dump)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
