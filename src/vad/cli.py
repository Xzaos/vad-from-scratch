from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .synth import SynthConfig, Synthesizer, load_clips, split_files


def _audio_files(root: str) -> list[Path]:
    return sorted(p for ext in ("wav", "flac", "mp3") for p in Path(root).rglob(f"*.{ext}"))


# Noise classes that never appear in training or validation. Picked to cover
# stationary (rain, vacuum), periodic (clock, church bells), impulsive
# (footsteps, keyboard) and tonal (siren, dog) noise, so the unseen-noise
# test covers more than one easy kind.
UNSEEN_NOISE = "rain,vacuum_cleaner,clock_tick,church_bells,footsteps,keyboard_typing,siren,dog"


def _split_val(files, salt):
    """Validation comes out of the training files by the same grouped hash,
    so a source never straddles train and val either."""
    tr, va = split_files(files, test_frac=0.1, salt=salt + "-val")
    return (tr, va) if va else (files[1:], files[:1])


def _synths(a):
    """Train, val and test synthesizers, plus an unseen-noise test synthesizer
    (None when there is no noise folder or no held-out classes).

    Noise files are expected under noise/<class>/; the class is the parent
    folder name."""
    if a.speech_test:
        speech_tr, speech_te = _audio_files(a.speech), _audio_files(a.speech_test)
    else:
        speech_tr, speech_te = split_files(_audio_files(a.speech))
    speech_tr, speech_val = _split_val(speech_tr, "speech")

    unseen = {c for c in (a.unseen_noise or "").split(",") if c}
    noise_all = _audio_files(a.noise) if a.noise else []
    noise_unseen = [f for f in noise_all if f.parent.name in unseen]
    noise_tr, noise_te = split_files([f for f in noise_all if f.parent.name not in unseen], salt="noise")
    noise_tr, noise_val = _split_val(noise_tr, "noise") if noise_tr else ([], [])

    print(
        f"speech files train/val/test: {len(speech_tr)}/{len(speech_val)}/{len(speech_te)}; "
        f"noise train/val/test/unseen-class: {len(noise_tr)}/{len(noise_val)}/{len(noise_te)}/{len(noise_unseen)}"
    )
    cfg = SynthConfig(telephony_prob=a.telephony_prob)
    load = lambda fs: load_clips(fs, limit=a.limit)
    test_speech = load(speech_te)
    return (
        Synthesizer(load(speech_tr), load(noise_tr), cfg),
        Synthesizer(load(speech_val), load(noise_val), cfg),
        Synthesizer(test_speech, load(noise_te), cfg),
        Synthesizer(test_speech, load(noise_unseen), cfg) if noise_unseen else None,
    )


def cmd_train(a):
    from .train import train

    tr, va, _, _ = _synths(a)
    train(tr, va, a.out, epochs=a.epochs, steps_per_epoch=a.steps, batch_size=a.batch_size, hidden=a.hidden, workers=a.workers)


def cmd_bench(a):
    from .bench import run
    from .train import load

    _, va, te, unseen = _synths(a)
    run(load(a.ckpt), va, te, unseen_synth=unseen, n_per_snr=a.n, use_silero=not a.no_silero, out=a.out)


def cmd_dump(a):
    """Write a few synthetic examples to wav + label files for listening."""
    import soundfile as sf

    tr, _, _, _ = _synths(a)
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
    common.add_argument("--speech-test", help="separate folder of test speech (e.g. FLEURS test split); default is a hash split of --speech")
    common.add_argument("--noise", help="folder of noise, one subfolder per class (see scripts/filter_esc50.py)")
    common.add_argument("--unseen-noise", default=UNSEEN_NOISE, help="comma-separated noise classes kept out of training; empty string to disable")
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
