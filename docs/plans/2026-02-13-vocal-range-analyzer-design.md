# Vocal Range Analyzer — Design Document

**Date**: 2026-02-13
**Status**: Approved

## Problem

Given a song audio file (MP3, WAV, FLAC), determine the vocal range required to sing it. Output the highest note, lowest note, total range, and a histogram of note occurrence weighted by duration. Target use case: "Can I sing this song?" — personal curiosity, speed over lab-grade accuracy.

## Decisions

- **Architecture**: Monolithic, function-based. No abstract classes or plugin system. Each pipeline is a function returning `list[PitchFrame]`.
- **Two pipelines** for comparison:
  - **Pipeline A**: Demucs vocal isolation + torchcrepe pitch detection (higher quality, slower)
  - **Pipeline B**: Spotify BasicPitch direct analysis (simpler, faster)
- **torchcrepe over CREPE**: Stays in PyTorch ecosystem with Demucs, avoids TensorFlow dependency
- **BasicPitch over RMVPE**: Pip-installable, no manual model weight downloads
- **argparse** for CLI (no extra deps)
- **pyproject.toml** with optional dependency groups so users only install what they need
- **Target hardware**: Mac with Apple Silicon (MPS acceleration), CPU fallback
- **Lazy ML imports**: analysis/visualization modules work without ML deps installed

## Data Model

```python
@dataclass
class PitchFrame:
    time_seconds: float      # position in song
    frequency_hz: float      # detected pitch (0 = unvoiced)
    confidence: float        # 0.0-1.0

@dataclass
class NoteEvent:
    midi_number: int         # 69
    note_name: str           # "A4"
    start_time: float        # seconds
    duration: float          # seconds
    mean_frequency_hz: float

@dataclass
class VocalRangeResult:
    lowest_note: str
    highest_note: str
    lowest_midi: int
    highest_midi: int
    range_semitones: int
    range_display: str               # "2 octaves + 8 semitones"
    note_histogram: dict[str, float] # note_name -> total_seconds
    note_events: list[NoteEvent]
    pitch_frames: list[PitchFrame]   # raw data for contour plot
```

`PitchFrame` is the universal interface — both pipelines produce it, all analysis consumes it.

## Pipeline A: Demucs + torchcrepe

```
audio file → Demucs htdemucs → vocals (mono, 44.1kHz)
           → torchcrepe.predict() → list[PitchFrame]
```

- `demucs.api.Separator(model="htdemucs")` for vocal isolation
- `torchcrepe.predict(audio, sr, hop_length=512, fmin=65, fmax=2000, return_periodicity=True)`
- Periodicity filtering: `torchcrepe.filter.median(win_length=3)` + `torchcrepe.threshold.At(0.21)`
- NaN frames (unvoiced) → freq=0, confidence=0
- Device auto-detection: mps > cuda > cpu

## Pipeline B: BasicPitch

```
audio file → basic_pitch.inference.predict() → note events → list[PitchFrame]
```

- No vocal isolation — works on polyphonic audio directly
- Returns `(model_output, midi_data, note_events)`
- Note events expanded to PitchFrames at 10ms resolution
- Overlapping polyphonic notes: keep highest amplitude per time slot

## Analysis

- `hz_to_midi(freq)`: standard formula `69 + 12*log2(freq/440)`, clamped 0-127
- `midi_to_note_name(midi)`: e.g., 60 → "C4"
- `build_note_histogram()`: accumulate frame durations per note
- `compute_vocal_range()`: filter by confidence, 2% percentile trim for outliers, find min/max
- `group_into_note_events()`: merge consecutive same-note frames, drop events < 50ms
- `InsufficientDataError` when < 10 confident frames

## Visualization

**Note Histogram** (horizontal bar chart):
- Y-axis: note names (low to high)
- X-axis: total duration in seconds
- Color gradient by octave
- Shaded background regions for voice type ranges:
  - Bass: E2–E4, Baritone: A2–A5, Tenor: C3–C5
  - Alto: F3–F5, Mezzo-Soprano: A3–A5, Soprano: C4–C6
- Subtitle with range display

**Pitch Contour** (scatter plot):
- X-axis: time (seconds)
- Y-axis: frequency (log scale) with note name grid lines
- Only voiced frames plotted

Both charts use `matplotlib.use("Agg")` for non-interactive CLI rendering.

## CLI Interface

```
python analyze.py song.mp3 --pipeline a --output results/
python analyze.py song.mp3 --pipeline b --output results/
python analyze.py song.mp3 --pipeline both --output results/
```

Options: `--pipeline {a,b,both}`, `--output DIR`, `--device {cpu,cuda,mps}`, `--confidence FLOAT`, `--no-plot`

## Project Structure

```
vocal-range-analyzer/
├── pyproject.toml
├── analyze.py
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── separation.py
│   ├── pitch_detection.py
│   ├── basic_pitch_detect.py
│   └── visualization.py
└── tests/
    ├── conftest.py
    └── test_analysis.py
```

## Dependencies

Core: numpy, librosa, soundfile, matplotlib
Pipeline A (optional): demucs, torchcrepe
Pipeline B (optional): basic-pitch
Dev: pytest, pytest-cov
