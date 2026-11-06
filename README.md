# VAD from scratch

A 58K-parameter streaming voice activity detector trained only on synthetic mixtures, compared against Silero VAD and an energy baseline at five noise levels, on noise types it trained on and on noise types it has never heard.

Headline result: on noise types it never trained on, TinyVAD beats Silero's frame F1 at every SNR, by 0.02 to 0.03 at 0 dB and above and by 0.085 at -5 dB. All the 95% intervals exclude zero. Most of the high-SNR gap is probably label convention rather than better detection (see below); the -5 dB gap is not.

## The data problem, and the workaround

Frame-level VAD labels are expensive. Someone has to mark where speech starts and stops to within 10 ms, and on noisy audio even careful annotators disagree. So this repo doesn't use labelled VAD data at all. It builds it.

`synth.py` takes clean speech clips and noise clips and does three things:

1. Labels each clean clip with an energy threshold relative to its own peak (35 dB below by default), then fills pauses under 100 ms. On clean audio this is reliable.
2. Lays one to four clips out on an 8-second timeline with random silent gaps, so the speech positions, and therefore the frame labels, are known exactly. Each clip is a random 0.8 to 3 second crop. FLEURS utterances run around 10 seconds, and without the crop a single clip fills the whole timeline, most frames are speech, and a detector that says "speech" to everything scores a high F1.
3. Mixes in noise at a chosen SNR (20, 10, 5, 0 or -5 dB), and passes 30% of examples through a phone-line channel (300 to 3400 Hz band, 8 kHz round trip, mu-law).

The model then has to make the energy threshold's decision on noisy audio, where an energy threshold fails. Training draws a fresh mixture for every example, so the training set is effectively unlimited.

Speech is Hindi and Tamil read speech from FLEURS. Training and validation clips come from the FLEURS train split and test clips from the FLEURS test split, so no sentence is read on both sides. Noise is ESC-50 with the human vocal classes removed.

## Keeping the test set honest

ESC-50 has a trap. Its file names look like `1-100038-A-14.wav`, where `100038` is the source recording and A, B, C are different 5-second cuts of it. 795 of its 2,000 clips share a source with another clip. A split on file names puts cuts of the same recording on both sides, and the model gets tested on noise it has already heard. `synth.source_key` groups cuts by source before splitting, and a test checks that no source straddles the split.

The second problem is the comparison itself. TinyVAD trains on exactly this mixing recipe, and Silero was trained on something else entirely. Beating Silero on held-out recordings of the same noise classes says more about distribution match than about the model. So the bench runs two conditions:

- seen: held-out recordings from noise classes that also appear in training
- unseen: eight classes held out of training entirely (rain, vacuum cleaner, clock tick, church bells, footsteps, keyboard typing, siren, dog), picked to mix stationary, periodic, impulsive and tonal noise

The unseen condition is the one I'd trust. The gap between the two conditions shows how much TinyVAD's edge comes from having seen the noise.

## Model

Log-mel (40 bins, 25 ms window, 10 ms hop), then two causal 1D convolutions, a unidirectional GRU and a linear head. Every layer only looks backwards in time, so it runs frame by frame with carried state. `Streamer` does exactly that, and a test checks that frame-by-frame output matches whole-clip output to 1e-5.

| | |
|---|---|
| parameters | 58,513 |
| CPU time per 10 ms frame (1 thread, Colab CPU) | 0.37 ms |
| real-time factor | 0.037 |

## Results

Thresholds for every system are tuned once on the validation mixtures and frozen before the test set is generated. Test mixtures are regenerated per SNR from held-out files with a fixed seed, so all three systems see identical audio. Each TinyVAD vs Silero gap comes with a 95% paired bootstrap interval over test clips: both systems are rescored on the same resampled clips, so the interval reflects which clips are hard as well as how many frames there are.

Noise classes never seen in training:

| SNR (dB) | all-speech F1 | energy F1 | TinyVAD F1 | Silero F1 | TinyVAD minus Silero (95% CI) |
|---|---|---|---|---|---|
| 20 | 0.629 | 0.769 | 0.959 | 0.931 | +0.028 [+0.019, +0.039] |
| 10 | 0.598 | 0.704 | 0.957 | 0.925 | +0.032 [+0.021, +0.042] |
| 5 | 0.612 | 0.673 | 0.917 | 0.900 | +0.017 [+0.005, +0.029] |
| 0 | 0.626 | 0.692 | 0.904 | 0.873 | +0.032 [+0.015, +0.048] |
| -5 | 0.612 | 0.635 | 0.858 | 0.773 | +0.085 [+0.054, +0.117] |

Held-out recordings, noise classes seen in training:

| SNR (dB) | all-speech F1 | energy F1 | TinyVAD F1 | Silero F1 | TinyVAD minus Silero (95% CI) |
|---|---|---|---|---|---|
| 20 | 0.596 | 0.730 | 0.957 | 0.929 | +0.028 [+0.020, +0.037] |
| 10 | 0.629 | 0.721 | 0.942 | 0.911 | +0.031 [+0.019, +0.043] |
| 5 | 0.628 | 0.693 | 0.925 | 0.901 | +0.024 [+0.015, +0.034] |
| 0 | 0.649 | 0.698 | 0.890 | 0.855 | +0.036 [+0.020, +0.052] |
| -5 | 0.616 | 0.620 | 0.850 | 0.754 | +0.097 [+0.064, +0.133] |

The all-speech column is the F1 of a detector that calls every frame speech. Any system near that number isn't detecting much. AUC, miss rate and false-alarm rate per SNR and condition are in [results/bench.json](results/bench.json), and the full tables are in [results/bench.md](results/bench.md).

Speech is 42 to 48% of frames in every test set, so a detector that calls everything speech scores about 0.6 F1. The energy baseline barely clears that: at -5 dB it scores 0.620 against a floor of 0.616, and its AUC falls from 0.85 to 0.63 as the noise gets louder.

TinyVAD does as well on the eight noise classes it never trained on as on the held-out recordings of classes it did see. At 10 dB it scores 0.957 unseen and 0.942 seen. So its lead doesn't come from having heard the noise.

That doesn't make it a better VAD than Silero. At 20 dB the AUCs are 0.995 and 0.977, so both systems rank frames almost perfectly, and the F1 gap there probably comes from where each puts speech boundaries. TinyVAD learned this repo's convention (energy 35 dB below peak, pauses under 100 ms filled, 10 ms frames). Silero has its own convention and scores 32 ms chunks. I haven't measured how much F1 that mismatch costs, but it could plausibly account for a 0.03 gap. Scoring both systems against forced-alignment boundaries would settle it.

The -5 dB result is different. The AUC gap widens to 0.950 against 0.881, which is a ranking difference, not a threshold or boundary one. Silero falls off faster in heavy noise than TinyVAD does, at least on this mix of read Indic speech, ESC-50 noise and a 30% share of phone-line audio. Whether that holds on real noisy recordings is the open question.

The whole model costs 0.37 ms of one CPU thread per 10 ms of audio, about 4% of real time.

## Running it

```bash
pip install -e ".[silero,dev]"
pytest -q
bash scripts/get_data.sh     # FLEURS hi/ta train and test clips + ESC-50 noise
bash scripts/run.sh          # dump samples, train, benchmark
```

`vad dump` writes a few mixtures to results/samples so you can listen to what the model trains on before spending GPU time. Training at the defaults takes about 20 minutes on a T4. `--unseen-noise` changes the held-out classes (pass an empty string to train on all of them).

## Choices worth questioning

The labels come from an energy threshold on clean audio, so they inherit its mistakes: very quiet word endings get labelled silence. The model can't learn to be better than those labels. A cleaner version would use forced alignment on the clean clips to get word boundaries.

FLEURS clips aren't studio clean. Some have room noise or hum under the speech, and that ends up inside the "clean" side of the mixture.

FLEURS doesn't publish speaker IDs, so I can only promise that train and test sentences differ, not that the speakers do.

ESC-50 classes with human vocal sounds (laughing, coughing, crying baby, sneezing, snoring, breathing) are excluded from the noise pool. Whether a laugh should count as speech depends on what the VAD feeds into; for an ASR front end, excluding it from both classes is the honest choice.

Silero scores 32 ms chunks while the labels are 10 ms frames. Each frame takes the score of the chunk containing its centre, which costs Silero a little at speech boundaries.

The loss uses a positive-class weight computed from the validation mixtures, since speech is usually the majority class in these timelines.

## Limitations

Read speech placed with gaps is easier than conversation. There are no overlapping talkers, no backchannels and no music. Real telephone recordings with hand labels would be the next test, and the unseen-noise condition is only a partial stand-in for that.

The telephony channel is a simulation. Real codecs (AMR-NB, Opus at low bitrate) and packet loss behave differently.
