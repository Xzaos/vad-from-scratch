"""A small streaming VAD: log-mel -> two causal 1D convs -> GRU -> logit.

Every layer only looks backwards in time, so the model can run one 10 ms
frame at a time with a carried state and give the same output as on the
whole clip. `test_streaming_matches_batch` checks that.

58,513 parameters at the defaults.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from .synth import HOP, SR, WIN

N_MELS = 40


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        # center=False keeps frame i aligned with samples [i*HOP, i*HOP+WIN),
        # the same convention the labels use.
        # n_fft == WIN so the frame count matches synth.n_frames exactly.
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=WIN, win_length=WIN, hop_length=HOP, n_mels=N_MELS, center=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, N_MELS, F)
        return torch.log(self.mel(wav) + 1e-6)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.pad = k - 1
        self.conv = nn.Conv1d(cin, cout, k)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TinyVAD(nn.Module):
    def __init__(self, hidden: int = 64, k: int = 5):
        super().__init__()
        self.features = LogMel()
        # Per-mel-bin normalisation learned from data, applied before the
        # convs. Cheaper than per-utterance CMVN and works when streaming.
        self.norm = nn.BatchNorm1d(N_MELS)
        self.conv = nn.Sequential(
            CausalConv(N_MELS, hidden, k), nn.ReLU(),
            CausalConv(hidden, hidden, k), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.k = k

    def forward_feats(self, mel: torch.Tensor, h=None):
        x = self.conv(self.norm(mel))  # (B, H, F)
        y, h = self.gru(x.transpose(1, 2), h)  # (B, F, H)
        return self.head(y).squeeze(-1), h

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward_feats(self.features(wav))[0]


class Streamer:
    """Feeds the model one mel frame at a time. Keeps the last k-1 frames of
    each conv's input and the GRU state between calls."""

    def __init__(self, model: TinyVAD):
        self.m = model.eval()
        c1, c2 = model.conv[0], model.conv[2]
        self.buf1 = torch.zeros(1, c1.conv.in_channels, model.k - 1)
        self.buf2 = torch.zeros(1, c2.conv.in_channels, model.k - 1)
        self.h = None

    @torch.inference_mode()
    def step(self, mel_frame: torch.Tensor) -> float:  # (N_MELS,)
        m = self.m
        x = m.norm(mel_frame.view(1, -1, 1))
        x1 = torch.cat([self.buf1, x], dim=2)
        self.buf1 = x1[:, :, 1:]
        y1 = torch.relu(m.conv[0].conv(x1))
        x2 = torch.cat([self.buf2, y1], dim=2)
        self.buf2 = x2[:, :, 1:]
        y2 = torch.relu(m.conv[2].conv(x2))
        out, self.h = m.gru(y2.transpose(1, 2), self.h)
        return float(torch.sigmoid(m.head(out)).item())


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
