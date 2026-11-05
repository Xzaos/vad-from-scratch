# vad-from-scratch

A 58K-parameter streaming voice activity detector trained only on synthetic mixtures, compared against Silero VAD and an energy baseline at five noise levels.

Headline result: _ (fill in from results/bench.md after the run)

## The data problem, and the workaround

Frame-level VAD labels are expensive. Someone has to mark where speech starts and stops to within 10 ms, and on noisy audio even careful annotators disagree. So this repo doesn't use labelled VAD data at all. It builds it.

`synth.py` takes clean speech clips and noise clips and does three things:

1. Labels each clean clip with an energy threshold relative to its own peak (35 dB below by default), then fills pauses under 100 ms. On clean audio this is reliable.
2. Lays one to four clips out on an 8-second timeline with random silent gaps, so the speech positions, and therefore the frame labels, are known exactly.
3. Mixes in noise at a chosen SNR (20, 10, 5, 0 or -5 dB), and passes 30% of examples through a phone-line channel (300 to 3400 Hz band, 8 kHz round trip, mu-law).

The model then has to make the energy threshold's decision on noisy audio, where an energy threshold fails. Training draws a fresh mixture for every example, so the training set is effectively unlimited.

Speech and noise files are split into train and test by a hash of the file name before anything is generated, so the test numbers are always on recordings the model has never heard.

## Model

Log-mel (40 bins, 25 ms window, 10 ms hop), then two causal 1D convolutions, a unidirectional GRU and a linear head. Every layer only looks backwards in time, so it runs frame by frame with carried state. `Streamer` does exactly that, and a test checks that frame-by-frame output matches whole-clip output to 1e-5.

| | |
|---|---|
| parameters | _ |
| CPU time per 10 ms frame (1 thread) | _ ms |
| real-time factor | _ |

## Results

Thresholds for every system are tuned once on the validation mixtures and frozen before the test set is generated. Test mixtures are regenerated per SNR from held-out files with a fixed seed, so all three systems see identical audio.

| SNR (dB) | energy F1 | TinyVAD F1 | Silero F1 | TinyVAD AUC | Silero AUC |
|---|---|---|---|---|---|
| 20 | _ | _ | _ | _ | _ |
| 10 | _ | _ | _ | _ | _ |
| 5 | _ | _ | _ | _ | _ |
| 0 | _ | _ | _ | _ | _ |
| -5 | _ | _ | _ | _ | _ |

Full numbers, including miss rate and false-alarm rate per SNR, are in [results/bench.json](results/bench.json).

What I read from these: _

## Running it

```bash
pip install -e ".[silero,dev]"
pytest -q
bash scripts/get_data.sh     # FLEURS hi/ta train clips + ESC-50 noise
bash scripts/run.sh          # dump samples, train, benchmark
```

`vad dump` writes a few mixtures to results/samples so you can listen to what the model trains on before spending GPU time. Training at the defaults takes about 20 minutes on a T4.

## Choices worth questioning

The labels come from an energy threshold on clean audio, so they inherit its mistakes: very quiet word endings get labelled silence. The model can't learn to be better than those labels. A cleaner version would use forced alignment on the clean clips to get word boundaries.

ESC-50 classes with human vocal sounds (laughing, coughing, crying baby, sneezing, snoring, breathing) are excluded from the noise pool. Whether a laugh should count as speech depends on what the VAD feeds into; for an ASR front end, excluding it from both classes is the honest choice.

Silero scores 32 ms chunks while the labels are 10 ms frames. Each frame takes the score of the chunk containing its centre, which costs Silero a little at speech boundaries.

The loss uses a positive-class weight computed from the validation mixtures, since speech is usually the majority class in these timelines.

## Limitations

Read speech placed with gaps is easier than conversation: no overlapping talkers, no backchannels, no music. Real telephone recordings would be the next test.

The telephony channel is a simulation. Real codecs (AMR-NB, Opus at low bitrate) and packet loss behave differently.
