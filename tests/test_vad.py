import numpy as np
import pytest
import torch

from vad import baseline
from vad.metrics import frame_f1, roc_auc
from vad.model import N_MELS, Streamer, TinyVAD, n_params
from vad.synth import SR, SynthConfig, Synthesizer, clean_labels, n_frames, split_files


def tone(sec, f=220.0, amp=0.5):
    t = np.arange(int(sec * SR)) / SR
    # a few harmonics with an envelope so it looks a bit like voiced speech
    x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6))
    return (amp * x / np.abs(x).max()).astype(np.float32)


def white(sec, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, int(sec * SR)).astype(np.float32)


def test_clean_labels_find_the_tone():
    x = np.concatenate([np.zeros(SR // 2, np.float32), tone(1.0), np.zeros(SR // 2, np.float32)])
    lab = clean_labels(x, -35)
    assert len(lab) == n_frames(len(x))
    assert lab[:40].sum() == 0 and lab[60:140].all() and lab[-40:].sum() == 0


def test_synth_shapes_and_snr():
    s = Synthesizer([tone(1.5), tone(1.0, 300)], [white(3.0)], SynthConfig(duration_s=4.0, telephony_prob=0.0, noise_free_prob=0.0))
    ex = s.make(np.random.default_rng(0), snr_db=0)
    assert ex.audio.shape == (4 * SR,)
    assert ex.labels.shape == (n_frames(4 * SR),)
    assert 0 < ex.labels.mean() < 1
    assert np.abs(ex.audio).max() <= 1.0


def test_synth_telephony_runs():
    s = Synthesizer([tone(1.0)], [white(2.0)], SynthConfig(duration_s=2.0, telephony_prob=1.0))
    ex = s.make(np.random.default_rng(1), snr_db=10)
    assert ex.telephony and ex.audio.shape == (2 * SR,)


def test_split_is_disjoint_and_stable():
    from pathlib import Path

    files = [Path(f"spk{i}.wav") for i in range(200)]
    a1, b1 = split_files(files)
    a2, b2 = split_files(list(reversed(files)))
    assert a1 == a2 and b1 == b2 and not set(a1) & set(b1)
    assert 20 < len(b1) < 60


def test_energy_baseline_separates_tone_from_silence():
    x = np.concatenate([white(1.0) * 0.01, tone(1.0), white(1.0) * 0.01])
    s = baseline.score(x)
    third = len(s) // 3
    assert s[third + 10 : 2 * third - 10].mean() > s[:third - 10].mean() + 20
    d = baseline.decide(x)
    assert d[third + 10 : 2 * third - 10].mean() > 0.9


def test_model_is_small_and_frame_aligned():
    m = TinyVAD()
    assert n_params(m) < 100_000
    wav = torch.randn(2, 3 * SR)
    assert m(wav).shape == (2, n_frames(3 * SR))


def test_streaming_matches_batch():
    torch.manual_seed(0)
    m = TinyVAD().eval()
    with torch.no_grad():
        # give BatchNorm non-trivial running stats
        m.norm.running_mean.uniform_(-1, 1)
        m.norm.running_var.uniform_(0.5, 2)
        wav = torch.randn(1, SR)
        mel = m.features(wav)
        batch = torch.sigmoid(m.forward_feats(mel)[0]).squeeze(0)
    st = Streamer(m)
    stream = torch.tensor([st.step(mel[0, :, i]) for i in range(mel.shape[2])])
    assert torch.allclose(batch, stream, atol=1e-5)


def test_metrics():
    g = np.array([0, 0, 1, 1])
    assert frame_f1(np.array([0, 0, 1, 1]), g)["f1"] == 1.0
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), g) == 1.0
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), g) == pytest.approx(0.5)
    assert frame_f1(np.array([1, 1, 1, 1]), g)["false_alarm"] == 1.0


def test_esc50_takes_of_one_source_stay_together():
    from pathlib import Path

    from vad.synth import source_key

    # takes A, B, C of each source recording, across many sources
    files = [Path(f"1-{100000 + s}-{t}-0.wav") for s in range(300) for t in "ABC"]
    tr, te = split_files(files)
    assert {source_key(f) for f in tr}.isdisjoint({source_key(f) for f in te})
    assert source_key(Path("4-171207-A-12.wav")) == source_key(Path("4-171207-C-12.wav"))
    assert source_key(Path("10017.wav")) == "10017.wav"


def test_paired_bootstrap_interval():
    from vad.bench import _f1, paired_f1_diff

    rng = np.random.default_rng(0)
    a = np.stack([rng.integers(80, 100, 50), rng.integers(0, 5, 50), rng.integers(0, 5, 50)], 1)
    b = np.stack([rng.integers(50, 70, 50), rng.integers(20, 30, 50), rng.integers(20, 30, 50)], 1)
    d = paired_f1_diff(a, b)
    assert d["lo"] < d["diff"] < d["hi"] and d["lo"] > 0
    same = paired_f1_diff(a, a)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    assert _f1(np.array([10, 0, 0])) == 1.0
