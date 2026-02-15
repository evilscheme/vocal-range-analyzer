# Pitch Detector Shootout — Design

## Goal

Compare 7 pitch detectors on "The Cuckoo (Remastered 2025)" to find the best replacement for CREPE. Measure both accuracy and speed. Pick a single winner and integrate it.

## Reference Song

- File: `~/Downloads/The Cuckoo (Remastered 2025) [ns1_1jU4Ycw].mp4`
- Known dominant melody notes: C4, E4, A3 (primary); D4, E3, G3 (secondary)
- Has vocal harmony

## Contenders

| Detector | Year | Type | Separation needed? | Install |
|----------|------|------|--------------------|---------|
| CREPE (baseline) | 2018 | Monophonic | Yes (Demucs) | already installed |
| BasicPitch (baseline) | 2022 | Polyphonic | No | already installed |
| SwiftF0 | 2025 | Monophonic | Yes (Demucs) | `pip install swift-f0` |
| FCPE | 2024 | Monophonic | Yes (Demucs) | `pip install torchfcpe` |
| PENN | 2023 | Monophonic | Yes (Demucs) | `pip install penn` |
| PESTO | 2023 | Monophonic | Yes (Demucs) | `pip install pesto-pitch` |
| RMVPE | 2023 | Vocal from mix | No | clone from GitHub |

## Structure

```
experiments/
├── results/                  # gitignored, raw outputs + cached vocals
├── run_crepe.py
├── run_basicpitch.py
├── run_swiftf0.py
├── run_fcpe.py
├── run_penn.py
├── run_pesto.py
├── run_rmvpe.py
├── compare.py
└── common.py
```

## Workflow

1. Run Demucs once, cache vocals WAV in `experiments/results/vocals.wav`
2. Each `run_*.py` loads audio, times the detector, saves `results/<name>.json`
3. `compare.py` reads all JSON results, produces summary table and plots

## Each run script outputs

JSON with: `{detector, elapsed_seconds, sample_rate, settings, frames: [{time, freq, confidence}, ...]}`

## Evaluation

- **Speed**: wall-clock elapsed time per detector
- **Accuracy (sanity check)**: top notes in histogram vs known melody notes (C4, E4, A3)
- **Accuracy (consensus)**: per-frame majority vote across all detectors; measure each detector's deviation
- **Visual**: overlay pitch contour plots, side-by-side histograms

## Outcome

Pick one winner. Integrate it into the main project, replacing CREPE in pipeline A. Throw away experiment scripts.
