# Vocal Range Analyzer - Implementation Plan

## Context

Build a Python CLI tool that analyzes the vocal range of a song from an audio file. Two pipelines allow quality/speed comparison: Pipeline A (Demucs vocal isolation + torchcrepe pitch detection) and Pipeline B (Spotify's BasicPitch, single-stage). Both feed into a shared analysis/visualization module.

Key research findings that shaped this plan:
- **torchcrepe over CREPE**: Avoids TensorFlow dependency, stays in PyTorch ecosystem alongside Demucs
- **BasicPitch over RMVPE**: Pip-installable, no manual model weight downloads, no cloning repos
- **Optional dependencies via pyproject.toml**: Demucs+torchcrepe pulls ~2GB of PyTorch; let users install only what they need

## Project Structure

```
vocal-range-analyzer/
├── pyproject.toml
├── analyze.py              # CLI entry point (argparse)
├── src/
│   ├── __init__.py
│   ├── analysis.py         # PitchFrame/VocalRangeResult dataclasses, Hz→note, histogram, range calc
│   ├── separation.py       # Demucs vocal isolation wrapper
│   ├── pitch_detection.py  # torchcrepe pitch detection wrapper
│   ├── basic_pitch_detect.py  # BasicPitch wrapper
│   └── visualization.py    # Matplotlib horizontal bar chart + pitch contour
└── tests/
    ├── conftest.py         # Synthetic sine wave fixture
    └── test_analysis.py    # Unit tests (no ML deps required)
```

## Build Order

### Phase 1: Project skeleton + pyproject.toml

Create directory structure and `pyproject.toml` with:
- **Core deps**: numpy, librosa, soundfile, matplotlib
- **Optional `pipeline-a`**: demucs, torchcrepe
- **Optional `pipeline-b`**: basic-pitch
- **Optional `all`**: both pipelines
- **Optional `dev`**: pytest, pytest-cov

All module stubs + `src/__init__.py`.

### Phase 2: `src/analysis.py` — Core math module (no ML deps)

The heart of the project. Build and test first since both pipelines feed into it.

**Data structures:**
- `PitchFrame(time_seconds, frequency_hz, confidence)` — universal interface between pipelines and analysis
- `NoteEvent(midi_number, note_name, start_time, duration, mean_frequency_hz)`
- `VocalRangeResult(lowest_note, highest_note, range_semitones, range_display, note_histogram, note_events, pitch_frames)`

**Key functions:**
- `hz_to_midi(freq_hz) -> int` — standard formula: `69 + 12*log2(freq/440)`, clamped to 0-127
- `midi_to_note_name(midi) -> str` — e.g., 60 → "C4"
- `build_note_histogram(frames, confidence_threshold, hop_duration) -> dict[str, float]` — note name → total seconds
- `compute_vocal_range(frames, confidence_threshold, percentile_trim) -> VocalRangeResult` — full analysis with 2% outlier trim
- `group_into_note_events(frames, confidence_threshold, min_duration) -> list[NoteEvent]`
- `InsufficientDataError` — raised when <10 confident frames

### Phase 3: `tests/test_analysis.py` — Unit tests

Tests for all conversion functions and edge cases. No ML dependencies needed.
- Hz↔MIDI round-trip accuracy (A4=440→69, C4=261.63→60)
- Zero/negative frequency → ValueError
- MIDI boundary clamping (0, 127)
- Empty/single-frame inputs
- Confidence filtering
- Histogram accuracy (100 frames of A4 at 10ms = 1.0s)
- `InsufficientDataError` on sparse data

`conftest.py`: Generate synthetic 440Hz sine WAV fixture for integration tests.

### Phase 4: `src/separation.py` + `src/pitch_detection.py` — Pipeline A

**separation.py** — `isolate_vocals(audio_path, output_dir, model, device) -> (np.ndarray, int)`:
- Lazy `import demucs.api` with clear ImportError message
- `Separator(model="htdemucs").separate_audio_file(path)`
- Extract `separated["vocals"]` tensor → mono numpy array
- Optionally save vocals WAV to output_dir

**pitch_detection.py** — `detect_pitch(audio, sample_rate, hop_length, fmin, fmax, model, confidence_threshold, device) -> list[PitchFrame]`:
- Lazy `import torchcrepe`
- Convert numpy → torch tensor shape (1, samples)
- `torchcrepe.predict(..., return_periodicity=True)`
- Apply `torchcrepe.filter.median(periodicity, win_length=3)`
- Apply `torchcrepe.threshold.At(0.21)` for voicing detection
- Convert NaN frames (unvoiced) to freq=0, confidence=0
- Default fmin=65Hz (C2), fmax=2000Hz (well above soprano range)

### Phase 5: `src/basic_pitch_detect.py` — Pipeline B

`detect_pitch_basic(audio_path, ...) -> list[PitchFrame]`:
- Lazy `from basic_pitch.inference import predict`
- `predict(audio_path)` returns `(model_output, midi_data, note_events)`
- Expand each note event `(start, end, midi_pitch, amplitude, pitch_bends)` into PitchFrames at 10ms resolution
- Deduplicate overlapping polyphonic notes: keep highest amplitude per time slot
- Sort by time

### Phase 6: `src/visualization.py` — Charts

**`plot_vocal_range(result, title, output_path)`**:
- Horizontal bar chart: note names on Y-axis (low→high), duration seconds on X-axis
- Color gradient by octave
- Subtitle showing range (e.g., "E2 – C5 | 2 oct + 8 semi")
- Top 10 notes labeled with duration

**`plot_pitch_contour(result, title, output_path)`**:
- Raw F0 over time, log-scale Y-axis with note name labels
- Useful for visual quality inspection

Both use `matplotlib.use("Agg")` for non-interactive CLI rendering.

### Phase 7: `analyze.py` — CLI entry point

```
python analyze.py song.mp3 --pipeline a --output results/
python analyze.py song.mp3 --pipeline b --output results/
python analyze.py song.mp3 --pipeline both --output results/
```

argparse with: `audio_file` (positional), `--pipeline {a,b,both}`, `--output`, `--device`, `--confidence`, `--no-plot`

Console output:
```
==================================================
  Pipeline A (Demucs + torchcrepe) Results
==================================================
  Lowest note:  E2 (MIDI 40)
  Highest note: C5 (MIDI 72)
  Range:        2 octaves + 8 semitones

  Top notes by duration:
    A3:  4.23s ######################
    G3:  3.87s ####################
    ...
```

## Key Design Decisions

1. **`PitchFrame` is the universal interface** — both pipelines output `list[PitchFrame]`, analysis only consumes this format
2. **Lazy ML imports** — all `demucs`, `torchcrepe`, `basic_pitch` imports inside functions, not at module level. This lets `analysis.py` and `visualization.py` work without ML deps installed
3. **Percentile trimming (2%)** in `compute_vocal_range` handles octave errors and pitch detection artifacts without manual range clamping
4. **argparse over click** — zero extra dependencies for a simple CLI

## Verification Plan

1. `pip install -e ".[all,dev]"`
2. `pytest tests/test_analysis.py -v` — unit tests, no ML downloads
3. Generate test sine wave and run both pipelines:
   ```bash
   python -c "import numpy as np, soundfile as sf; sr=44100; t=np.linspace(0,5,sr*5,False); sf.write('test_sine.wav', (0.5*np.sin(2*np.pi*440*t)).astype('float32'), sr)"
   python analyze.py test_sine.wav --pipeline both --output test_output/
   ```
   Expected: both detect A4 as the only note
4. Test with a real song: `python analyze.py song.mp3 --pipeline both --output results/`
5. Test edge case: instrumental-only track → should print `InsufficientDataError` gracefully
