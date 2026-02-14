# Vocal Range Analyzer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that analyzes the vocal range of a song, outputting highest/lowest notes, range, and a duration-weighted note histogram with voice type overlay.

**Architecture:** Monolithic, function-based. Two pipelines (Demucs+torchcrepe and BasicPitch) both produce `list[PitchFrame]`, which feeds into a shared analysis/visualization module. Lazy ML imports so the core math works without heavy dependencies.

**Tech Stack:** Python 3.10+, numpy, librosa, soundfile, matplotlib, demucs, torchcrepe, basic-pitch, pytest

---

### Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/analysis.py` (stub)
- Create: `src/separation.py` (stub)
- Create: `src/pitch_detection.py` (stub)
- Create: `src/basic_pitch_detect.py` (stub)
- Create: `src/visualization.py` (stub)
- Create: `analyze.py` (stub)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py` (stub)
- Create: `tests/test_analysis.py` (stub)
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "vocal-range-analyzer"
version = "0.1.0"
description = "Analyze the vocal range of a song from an audio file"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "librosa>=0.10",
    "soundfile>=0.12",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
pipeline-a = [
    "demucs>=4.0",
    "torchcrepe>=0.0.22",
]
pipeline-b = [
    "basic-pitch>=0.3",
]
all = [
    "vocal-range-analyzer[pipeline-a,pipeline-b]",
]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.wav
*.mp3
*.flac
*.ogg
test_output/
results/
.venv/
venv/
```

**Step 3: Create all stub files**

`src/__init__.py`: empty file

`src/analysis.py`:
```python
"""Pitch analysis: Hz-to-note conversion, histogram building, range calculation."""
```

`src/separation.py`:
```python
"""Vocal isolation using Demucs."""
```

`src/pitch_detection.py`:
```python
"""Pitch detection using torchcrepe."""
```

`src/basic_pitch_detect.py`:
```python
"""Pitch detection using Spotify BasicPitch."""
```

`src/visualization.py`:
```python
"""Visualization: note histogram and pitch contour charts."""
```

`analyze.py`:
```python
"""CLI entry point for Vocal Range Analyzer."""
```

`tests/__init__.py`: empty file

`tests/conftest.py`:
```python
"""Shared test fixtures."""
```

`tests/test_analysis.py`:
```python
"""Tests for src/analysis.py."""
```

**Step 4: Create virtual environment and install**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: Install succeeds, core deps available.

**Step 5: Verify pytest runs**

Run: `pytest -v`
Expected: "no tests ran" or "collected 0 items"

**Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/ analyze.py tests/
git commit -m "feat: project skeleton with pyproject.toml and module stubs"
```

---

### Task 2: Analysis Module — Data Structures and Hz/MIDI Conversion

**Files:**
- Modify: `src/analysis.py`
- Modify: `tests/test_analysis.py`

**Step 1: Write failing tests for hz_to_midi and midi_to_note_name**

`tests/test_analysis.py`:
```python
"""Tests for src/analysis.py."""
from __future__ import annotations

import pytest
from src.analysis import hz_to_midi, midi_to_note_name, hz_to_note_name, PitchFrame


class TestHzToMidi:
    def test_a4_is_69(self):
        assert hz_to_midi(440.0) == 69

    def test_middle_c(self):
        assert hz_to_midi(261.63) == 60

    def test_a3(self):
        assert hz_to_midi(220.0) == 57

    def test_c5(self):
        assert hz_to_midi(523.25) == 72

    def test_rounding_up(self):
        # 445 Hz is slightly sharp of A4 but should round to 69
        assert hz_to_midi(445.0) == 69

    def test_rounding_down(self):
        # 435 Hz is slightly flat of A4 but should round to 69
        assert hz_to_midi(435.0) == 69

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            hz_to_midi(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            hz_to_midi(-100.0)

    def test_very_low_clamps_to_zero(self):
        assert hz_to_midi(5.0) == 0

    def test_very_high_clamps_to_127(self):
        assert hz_to_midi(20000.0) == 127


class TestMidiToNoteName:
    def test_middle_c(self):
        assert midi_to_note_name(60) == "C4"

    def test_a4(self):
        assert midi_to_note_name(69) == "A4"

    def test_c_sharp_4(self):
        assert midi_to_note_name(61) == "C#4"

    def test_b4(self):
        assert midi_to_note_name(71) == "B4"

    def test_c5(self):
        assert midi_to_note_name(72) == "C5"

    def test_midi_0(self):
        assert midi_to_note_name(0) == "C-1"

    def test_midi_127(self):
        assert midi_to_note_name(127) == "G9"

    def test_e2(self):
        assert midi_to_note_name(40) == "E2"


class TestHzToNoteName:
    def test_a4(self):
        assert hz_to_note_name(440.0) == "A4"

    def test_middle_c(self):
        assert hz_to_note_name(261.63) == "C4"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py -v`
Expected: FAIL — ImportError, functions not defined.

**Step 3: Implement data structures and conversion functions**

`src/analysis.py`:
```python
"""Pitch analysis: Hz-to-note conversion, histogram building, range calculation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


class InsufficientDataError(Exception):
    """Raised when too few pitched frames are detected."""
    pass


@dataclass
class PitchFrame:
    """A single pitch observation."""
    time_seconds: float
    frequency_hz: float
    confidence: float


@dataclass
class NoteEvent:
    """A discrete note with duration."""
    midi_number: int
    note_name: str
    start_time: float
    duration: float
    mean_frequency_hz: float


@dataclass
class VocalRangeResult:
    """Complete analysis output."""
    lowest_note: str
    highest_note: str
    lowest_midi: int
    highest_midi: int
    range_semitones: int
    range_display: str
    note_histogram: dict[str, float]
    note_events: list[NoteEvent]
    pitch_frames: list[PitchFrame]


def hz_to_midi(freq_hz: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number (0-127)."""
    if freq_hz <= 0:
        raise ValueError(f"Frequency must be positive, got {freq_hz}")
    midi = 69 + 12 * np.log2(freq_hz / 440.0)
    return int(np.clip(round(midi), 0, 127))


def midi_to_note_name(midi_number: int) -> str:
    """Convert MIDI note number to scientific pitch notation (e.g., 60 -> 'C4')."""
    octave = (midi_number // 12) - 1
    note = NOTE_NAMES[midi_number % 12]
    return f"{note}{octave}"


def hz_to_note_name(freq_hz: float) -> str:
    """Convert frequency in Hz to note name (e.g., 440.0 -> 'A4')."""
    return midi_to_note_name(hz_to_midi(freq_hz))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: add PitchFrame dataclasses and Hz/MIDI conversion functions"
```

---

### Task 3: Analysis Module — Histogram and Range Calculation

**Files:**
- Modify: `src/analysis.py`
- Modify: `tests/test_analysis.py`

**Step 1: Write failing tests for build_note_histogram**

Append to `tests/test_analysis.py`:
```python
from src.analysis import build_note_histogram, InsufficientDataError


class TestBuildNoteHistogram:
    def test_empty_input(self):
        assert build_note_histogram([]) == {}

    def test_single_note_100_frames(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]
        hist = build_note_histogram(frames, hop_duration_seconds=0.01)
        assert "A4" in hist
        assert hist["A4"] == pytest.approx(1.0, abs=0.02)

    def test_two_notes(self):
        frames = (
            [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(50)]
            + [PitchFrame(t * 0.01 + 0.5, 261.63, 0.9) for t in range(50)]
        )
        hist = build_note_histogram(frames, hop_duration_seconds=0.01)
        assert "A4" in hist
        assert "C4" in hist
        assert hist["A4"] == pytest.approx(0.5, abs=0.02)
        assert hist["C4"] == pytest.approx(0.5, abs=0.02)

    def test_confidence_filtering(self):
        frames = [PitchFrame(0.0, 440.0, 0.1)]
        assert build_note_histogram(frames, confidence_threshold=0.5) == {}

    def test_zero_freq_skipped(self):
        frames = [PitchFrame(0.0, 0.0, 0.9)]
        assert build_note_histogram(frames) == {}
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py::TestBuildNoteHistogram -v`
Expected: FAIL — ImportError.

**Step 3: Implement build_note_histogram**

Add to `src/analysis.py`:
```python
def build_note_histogram(
    pitch_frames: list[PitchFrame],
    confidence_threshold: float = 0.5,
    hop_duration_seconds: float = 0.01,
) -> dict[str, float]:
    """Build histogram of note occurrence weighted by duration.

    Returns dict mapping note names (e.g., "A4") to total seconds.
    """
    histogram: dict[str, float] = {}
    for frame in pitch_frames:
        if frame.confidence < confidence_threshold or frame.frequency_hz <= 0:
            continue
        note_name = hz_to_note_name(frame.frequency_hz)
        histogram[note_name] = histogram.get(note_name, 0.0) + hop_duration_seconds
    return histogram
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py::TestBuildNoteHistogram -v`
Expected: All PASS.

**Step 5: Write failing tests for group_into_note_events**

Append to `tests/test_analysis.py`:
```python
from src.analysis import group_into_note_events


class TestGroupIntoNoteEvents:
    def test_single_sustained_note(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]
        events = group_into_note_events(frames, hop_duration_seconds=0.01)
        assert len(events) == 1
        assert events[0].note_name == "A4"
        assert events[0].duration == pytest.approx(1.0, abs=0.02)

    def test_two_notes_sequential(self):
        frames = (
            [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(50)]
            + [PitchFrame(t * 0.01 + 0.5, 261.63, 0.9) for t in range(50)]
        )
        events = group_into_note_events(frames, hop_duration_seconds=0.01)
        assert len(events) == 2
        assert events[0].note_name == "A4"
        assert events[1].note_name == "C4"

    def test_short_notes_filtered(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(2)]
        events = group_into_note_events(frames, min_duration_seconds=0.05, hop_duration_seconds=0.01)
        assert len(events) == 0

    def test_empty_input(self):
        assert group_into_note_events([]) == []

    def test_low_confidence_skipped(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.1) for t in range(100)]
        events = group_into_note_events(frames, confidence_threshold=0.5)
        assert len(events) == 0
```

**Step 6: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py::TestGroupIntoNoteEvents -v`
Expected: FAIL — ImportError.

**Step 7: Implement group_into_note_events**

Add to `src/analysis.py`:
```python
def group_into_note_events(
    pitch_frames: list[PitchFrame],
    confidence_threshold: float = 0.5,
    min_duration_seconds: float = 0.05,
    hop_duration_seconds: float = 0.01,
) -> list[NoteEvent]:
    """Group consecutive same-note frames into NoteEvent objects.

    Events shorter than min_duration_seconds are discarded.
    """
    if not pitch_frames:
        return []

    events: list[NoteEvent] = []
    current_midi: int | None = None
    current_start: float = 0.0
    current_freqs: list[float] = []
    frame_count: int = 0

    def flush():
        if current_midi is not None and frame_count > 0:
            duration = frame_count * hop_duration_seconds
            if duration >= min_duration_seconds:
                events.append(NoteEvent(
                    midi_number=current_midi,
                    note_name=midi_to_note_name(current_midi),
                    start_time=current_start,
                    duration=duration,
                    mean_frequency_hz=float(np.mean(current_freqs)),
                ))

    for frame in pitch_frames:
        if frame.confidence < confidence_threshold or frame.frequency_hz <= 0:
            flush()
            current_midi = None
            frame_count = 0
            current_freqs = []
            continue

        midi = hz_to_midi(frame.frequency_hz)
        if midi != current_midi:
            flush()
            current_midi = midi
            current_start = frame.time_seconds
            current_freqs = [frame.frequency_hz]
            frame_count = 1
        else:
            current_freqs.append(frame.frequency_hz)
            frame_count += 1

    flush()
    return events
```

**Step 8: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py::TestGroupIntoNoteEvents -v`
Expected: All PASS.

**Step 9: Write failing tests for compute_vocal_range**

Append to `tests/test_analysis.py`:
```python
from src.analysis import compute_vocal_range, VocalRangeResult


class TestComputeVocalRange:
    def test_insufficient_data_raises(self):
        frames = [PitchFrame(0.0, 440.0, 0.9)]
        with pytest.raises(InsufficientDataError):
            compute_vocal_range(frames)

    def test_single_note(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]
        result = compute_vocal_range(frames, hop_duration_seconds=0.01)
        assert result.lowest_note == "A4"
        assert result.highest_note == "A4"
        assert result.range_semitones == 0

    def test_one_octave_range(self):
        frames = (
            [PitchFrame(t * 0.01, 261.63, 0.9) for t in range(50)]
            + [PitchFrame(t * 0.01 + 0.5, 523.25, 0.9) for t in range(50)]
        )
        result = compute_vocal_range(frames, percentile_trim=0.0, hop_duration_seconds=0.01)
        assert result.lowest_note == "C4"
        assert result.highest_note == "C5"
        assert result.range_semitones == 12
        assert "1 octave" in result.range_display

    def test_range_display_format(self):
        # E2 (MIDI 40) to C5 (MIDI 72) = 32 semitones = 2 oct + 8 semi
        frames = (
            [PitchFrame(t * 0.01, 82.41, 0.9) for t in range(50)]  # E2
            + [PitchFrame(t * 0.01 + 0.5, 523.25, 0.9) for t in range(50)]  # C5
        )
        result = compute_vocal_range(frames, percentile_trim=0.0, hop_duration_seconds=0.01)
        assert result.range_semitones == 32
        assert "2 octaves" in result.range_display
        assert "8 semitones" in result.range_display

    def test_all_low_confidence(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.1) for t in range(100)]
        with pytest.raises(InsufficientDataError):
            compute_vocal_range(frames, confidence_threshold=0.5)

    def test_histogram_populated(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]
        result = compute_vocal_range(frames, hop_duration_seconds=0.01)
        assert "A4" in result.note_histogram
        assert result.note_histogram["A4"] > 0

    def test_note_events_populated(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]
        result = compute_vocal_range(frames, hop_duration_seconds=0.01)
        assert len(result.note_events) > 0
        assert result.note_events[0].note_name == "A4"
```

**Step 10: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py::TestComputeVocalRange -v`
Expected: FAIL — ImportError.

**Step 11: Implement compute_vocal_range**

Add to `src/analysis.py`:
```python
def _format_range_display(semitones: int) -> str:
    """Format semitone count as 'N octaves + M semitones'."""
    if semitones == 0:
        return "0 semitones"
    octaves = semitones // 12
    remaining = semitones % 12
    parts = []
    if octaves == 1:
        parts.append("1 octave")
    elif octaves > 1:
        parts.append(f"{octaves} octaves")
    if remaining == 1:
        parts.append("1 semitone")
    elif remaining > 1:
        parts.append(f"{remaining} semitones")
    return " + ".join(parts)


def compute_vocal_range(
    pitch_frames: list[PitchFrame],
    confidence_threshold: float = 0.5,
    percentile_trim: float = 2.0,
    hop_duration_seconds: float = 0.01,
    min_note_duration: float = 0.05,
) -> VocalRangeResult:
    """Compute full vocal range analysis from pitch frames.

    Filters by confidence, trims outliers by percentile, computes range,
    histogram, and note events.
    """
    # Filter confident, voiced frames
    voiced = [f for f in pitch_frames
              if f.confidence >= confidence_threshold and f.frequency_hz > 0]

    if len(voiced) < 10:
        raise InsufficientDataError(
            f"Only {len(voiced)} voiced frames detected (need at least 10). "
            "The audio may not contain vocals."
        )

    freqs = np.array([f.frequency_hz for f in voiced])

    # Percentile trim to remove outliers
    if percentile_trim > 0 and len(freqs) > 20:
        low_cut = np.percentile(freqs, percentile_trim)
        high_cut = np.percentile(freqs, 100 - percentile_trim)
        trimmed_frames = [f for f in voiced if low_cut <= f.frequency_hz <= high_cut]
        if len(trimmed_frames) >= 10:
            voiced = trimmed_frames
            freqs = np.array([f.frequency_hz for f in voiced])

    midi_notes = np.array([hz_to_midi(f) for f in freqs])
    lowest_midi = int(midi_notes.min())
    highest_midi = int(midi_notes.max())
    range_semitones = highest_midi - lowest_midi

    histogram = build_note_histogram(
        pitch_frames, confidence_threshold, hop_duration_seconds
    )
    note_events = group_into_note_events(
        pitch_frames, confidence_threshold, min_note_duration, hop_duration_seconds
    )

    return VocalRangeResult(
        lowest_note=midi_to_note_name(lowest_midi),
        highest_note=midi_to_note_name(highest_midi),
        lowest_midi=lowest_midi,
        highest_midi=highest_midi,
        range_semitones=range_semitones,
        range_display=_format_range_display(range_semitones),
        note_histogram=histogram,
        note_events=note_events,
        pitch_frames=pitch_frames,
    )
```

**Step 12: Run all tests**

Run: `pytest tests/test_analysis.py -v`
Expected: All PASS.

**Step 13: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: add histogram, note events, and vocal range analysis"
```

---

### Task 4: Test Fixtures

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Create synthetic audio fixtures**

`tests/conftest.py`:
```python
"""Shared test fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def sine_440hz_wav(tmp_path_factory) -> Path:
    """Generate a 3-second 440Hz (A4) sine wave WAV file."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path_factory.mktemp("fixtures") / "sine_440hz.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture(scope="session")
def two_tone_wav(tmp_path_factory) -> Path:
    """Generate a 4-second WAV: 2s of C4 (261.63Hz), then 2s of A4 (440Hz)."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    c4 = (0.5 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)
    a4 = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    audio = np.concatenate([c4, a4])
    path = tmp_path_factory.mktemp("fixtures") / "two_tone.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture(scope="session")
def silence_wav(tmp_path_factory) -> Path:
    """Generate a 2-second silent WAV file."""
    sr = 44100
    duration = 2.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    path = tmp_path_factory.mktemp("fixtures") / "silence.wav"
    sf.write(str(path), audio, sr)
    return path
```

**Step 2: Run to verify fixtures work**

Run: `pytest tests/test_analysis.py -v`
Expected: All PASS (fixtures created but not yet used by any test).

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add synthetic audio test fixtures"
```

---

### Task 5: Pipeline A — Vocal Separation (Demucs)

**Files:**
- Modify: `src/separation.py`

**Step 1: Implement isolate_vocals**

`src/separation.py`:
```python
"""Vocal isolation using Demucs."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def isolate_vocals(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "htdemucs",
    device: str | None = None,
) -> tuple[np.ndarray, int]:
    """Separate vocals from an audio file using Demucs.

    Args:
        audio_path: Path to input audio file.
        output_dir: If provided, save isolated vocals WAV here.
        model: Demucs model name.
        device: 'cpu', 'cuda', 'mps', or None (auto-detect).

    Returns:
        Tuple of (vocals_mono_float32, sample_rate).
    """
    try:
        import demucs.api
    except ImportError:
        raise ImportError(
            "Demucs is required for Pipeline A. "
            "Install with: pip install 'vocal-range-analyzer[pipeline-a]'"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    separator = demucs.api.Separator(model=model, device=device)
    _, separated = separator.separate_audio_file(str(audio_path))
    vocals_tensor = separated["vocals"]

    # Convert to mono numpy float32
    vocals_np = vocals_tensor.cpu().numpy()
    if vocals_np.ndim == 2 and vocals_np.shape[0] > 1:
        vocals_np = vocals_np.mean(axis=0)
    elif vocals_np.ndim == 2:
        vocals_np = vocals_np[0]

    sample_rate = separator.samplerate

    if output_dir is not None:
        import soundfile as sf
        out_path = Path(output_dir) / "vocals.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), vocals_np, sample_rate)

    return vocals_np, sample_rate
```

**Step 2: Commit**

No unit test for this module — it wraps an external ML model. Integration testing in Task 9.

```bash
git add src/separation.py
git commit -m "feat: add Demucs vocal isolation wrapper"
```

---

### Task 6: Pipeline A — Pitch Detection (torchcrepe)

**Files:**
- Modify: `src/pitch_detection.py`

**Step 1: Implement detect_pitch**

`src/pitch_detection.py`:
```python
"""Pitch detection using torchcrepe."""
from __future__ import annotations

import numpy as np

from src.analysis import PitchFrame


def detect_pitch(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 2000.0,
    model: str = "full",
    confidence_threshold: float = 0.21,
    device: str | None = None,
) -> list[PitchFrame]:
    """Detect pitch contour from audio using torchcrepe.

    Args:
        audio: Mono float32 numpy array.
        sample_rate: Sample rate in Hz.
        hop_length: Hop length in samples.
        fmin: Minimum frequency (Hz). Default 65 = C2.
        fmax: Maximum frequency (Hz). Default 2000, well above soprano.
        model: 'full' (accurate) or 'tiny' (fast).
        confidence_threshold: Periodicity threshold for voicing.
        device: 'cpu', 'cuda', 'mps', or None (auto-detect).

    Returns:
        List of PitchFrame objects.
    """
    try:
        import torch
        import torchcrepe
    except ImportError:
        raise ImportError(
            "torchcrepe is required for Pipeline A. "
            "Install with: pip install 'vocal-range-analyzer[pipeline-a]'"
        )

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    pitch, periodicity = torchcrepe.predict(
        audio_tensor,
        sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model,
        return_periodicity=True,
        device=device,
    )

    # Smooth periodicity and apply voicing threshold
    periodicity = torchcrepe.filter.median(periodicity, win_length=3)
    pitch = torchcrepe.threshold.At(confidence_threshold)(pitch, periodicity)

    pitch_np = pitch.squeeze().cpu().numpy()
    periodicity_np = periodicity.squeeze().cpu().numpy()
    hop_seconds = hop_length / sample_rate

    frames: list[PitchFrame] = []
    for i, (f, p) in enumerate(zip(pitch_np, periodicity_np)):
        if np.isnan(f) or f <= 0:
            frames.append(PitchFrame(
                time_seconds=i * hop_seconds,
                frequency_hz=0.0,
                confidence=0.0,
            ))
        else:
            frames.append(PitchFrame(
                time_seconds=i * hop_seconds,
                frequency_hz=float(f),
                confidence=float(p),
            ))

    return frames
```

**Step 2: Commit**

```bash
git add src/pitch_detection.py
git commit -m "feat: add torchcrepe pitch detection wrapper"
```

---

### Task 7: Pipeline B — BasicPitch

**Files:**
- Modify: `src/basic_pitch_detect.py`

**Step 1: Implement detect_pitch_basic**

`src/basic_pitch_detect.py`:
```python
"""Pitch detection using Spotify BasicPitch."""
from __future__ import annotations

from pathlib import Path

from src.analysis import PitchFrame


def detect_pitch_basic(
    audio_path: str | Path,
    minimum_frequency: float | None = 65.0,
    maximum_frequency: float | None = 2000.0,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 50.0,
) -> list[PitchFrame]:
    """Detect pitch from audio using BasicPitch (single-stage, no separation).

    Args:
        audio_path: Path to audio file.
        minimum_frequency: Low frequency cutoff in Hz.
        maximum_frequency: High frequency cutoff in Hz.
        onset_threshold: Note onset sensitivity.
        frame_threshold: Frame-level activation threshold.
        minimum_note_length_ms: Minimum note duration in ms.

    Returns:
        List of PitchFrame objects sorted by time.
    """
    try:
        from basic_pitch.inference import predict
    except ImportError:
        raise ImportError(
            "basic-pitch is required for Pipeline B. "
            "Install with: pip install 'vocal-range-analyzer[pipeline-b]'"
        )

    import numpy as np

    _, _, note_events = predict(
        str(audio_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    HOP_SECONDS = 0.01
    frames: list[PitchFrame] = []

    for start, end, midi_pitch, amplitude, _pitch_bends in note_events:
        freq_hz = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))
        t = start
        while t < end:
            frames.append(PitchFrame(
                time_seconds=t,
                frequency_hz=freq_hz,
                confidence=float(amplitude),
            ))
            t += HOP_SECONDS

    # Sort by time, deduplicate overlapping notes (keep highest amplitude)
    frames.sort(key=lambda f: (f.time_seconds, -f.confidence))
    deduplicated: list[PitchFrame] = []
    seen_times: set[float] = set()
    for f in frames:
        t_key = round(f.time_seconds, 3)
        if t_key not in seen_times:
            seen_times.add(t_key)
            deduplicated.append(f)

    return deduplicated
```

**Step 2: Commit**

```bash
git add src/basic_pitch_detect.py
git commit -m "feat: add BasicPitch pitch detection wrapper"
```

---

### Task 8: Visualization

**Files:**
- Modify: `src/visualization.py`

**Step 1: Implement plot_vocal_range and plot_pitch_contour**

`src/visualization.py`:
```python
"""Visualization: note histogram and pitch contour charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.analysis import VocalRangeResult, midi_to_note_name


# Standard voice type ranges as (name, low_midi, high_midi, color)
VOICE_TYPES = [
    ("Bass",     40, 64, "#2196F3"),    # E2-E4
    ("Baritone", 45, 69, "#4CAF50"),    # A2-A4
    ("Tenor",    48, 72, "#FF9800"),    # C3-C5
    ("Alto",     53, 77, "#9C27B0"),    # F3-F5
    ("Mezzo",    57, 81, "#E91E63"),    # A3-A5
    ("Soprano",  60, 84, "#F44336"),    # C4-C6
]


def plot_vocal_range(
    result: VocalRangeResult,
    title: str = "Vocal Range Analysis",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Create horizontal bar chart of note occurrence with voice type overlays."""
    if not result.note_histogram:
        raise ValueError("No notes to plot — histogram is empty.")

    # Sort notes by MIDI number (low to high)
    from src.analysis import hz_to_midi, NOTE_NAMES

    def note_sort_key(name: str) -> int:
        """Parse note name to MIDI for sorting."""
        for i, n in enumerate(NOTE_NAMES):
            if name.startswith(n) and (len(n) == len(name.rstrip("0123456789-"))):
                octave = int(name[len(n):])
                return (octave + 1) * 12 + i
        return 0

    sorted_notes = sorted(result.note_histogram.items(), key=lambda x: note_sort_key(x[0]))
    note_names = [n for n, _ in sorted_notes]
    durations = [d for _, d in sorted_notes]

    # Color by octave
    octaves = [int(n[-1]) if n[-1].isdigit() else int(n[-2:]) for n in note_names]
    cmap = plt.cm.viridis
    unique_octaves = sorted(set(octaves))
    if len(unique_octaves) > 1:
        octave_norm = [(o - min(unique_octaves)) / (max(unique_octaves) - min(unique_octaves))
                       for o in octaves]
    else:
        octave_norm = [0.5] * len(octaves)
    colors = [cmap(v) for v in octave_norm]

    # Figure size scales with number of notes
    fig_height = max(4, len(note_names) * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Voice type background shading
    for vtype_name, low_midi, high_midi, color in VOICE_TYPES:
        low_note = midi_to_note_name(low_midi)
        high_note = midi_to_note_name(high_midi)
        # Find Y positions that fall within this range
        y_positions = []
        for i, n in enumerate(note_names):
            midi = note_sort_key(n)
            if low_midi <= midi <= high_midi:
                y_positions.append(i)
        if y_positions:
            ax.axhspan(
                min(y_positions) - 0.4, max(y_positions) + 0.4,
                alpha=0.08, color=color, label=f"{vtype_name} ({low_note}-{high_note})",
            )

    # Bar chart
    bars = ax.barh(range(len(note_names)), durations, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(note_names)))
    ax.set_yticklabels(note_names, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Duration (seconds)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(durations) * 1.15)

    # Subtitle with range info
    subtitle = f"Range: {result.lowest_note} \u2013 {result.highest_note} | {result.range_display}"
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=10, color="gray")

    # Legend for voice types
    handles = [mpatches.Patch(color=c, alpha=0.3, label=f"{n} ({midi_to_note_name(lo)}-{midi_to_note_name(hi)})")
               for n, lo, hi, c in VOICE_TYPES]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_pitch_contour(
    result: VocalRangeResult,
    title: str = "Pitch Contour",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot raw F0 contour over time."""
    voiced = [f for f in result.pitch_frames if f.frequency_hz > 0 and f.confidence > 0]
    if not voiced:
        raise ValueError("No voiced frames to plot.")

    times = [f.time_seconds for f in voiced]
    freqs = [f.frequency_hz for f in voiced]
    confs = [f.confidence for f in voiced]

    fig, ax = plt.subplots(figsize=(12, 5))
    scatter = ax.scatter(times, freqs, c=confs, s=1, cmap="viridis", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add note name gridlines
    reference_notes = [("C2", 65.41), ("E2", 82.41), ("A2", 110.0),
                       ("C3", 130.81), ("E3", 164.81), ("A3", 220.0),
                       ("C4", 261.63), ("E4", 329.63), ("A4", 440.0),
                       ("C5", 523.25), ("E5", 659.25), ("A5", 880.0),
                       ("C6", 1046.5)]
    for note_name, freq in reference_notes:
        if min(freqs) * 0.8 <= freq <= max(freqs) * 1.2:
            ax.axhline(y=freq, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
            ax.text(max(times) * 1.01, freq, note_name, fontsize=7, va="center", color="gray")

    plt.colorbar(scatter, ax=ax, label="Confidence", shrink=0.8)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
```

**Step 2: Write a quick smoke test**

Append to `tests/test_analysis.py`:
```python
from src.visualization import plot_vocal_range, plot_pitch_contour


class TestVisualization:
    def _make_result(self) -> VocalRangeResult:
        """Create a synthetic VocalRangeResult for testing."""
        frames = (
            [PitchFrame(t * 0.01, 261.63, 0.9) for t in range(200)]   # C4
            + [PitchFrame(t * 0.01 + 2.0, 329.63, 0.9) for t in range(150)]  # E4
            + [PitchFrame(t * 0.01 + 3.5, 440.0, 0.9) for t in range(100)]   # A4
        )
        return compute_vocal_range(frames, percentile_trim=0.0, hop_duration_seconds=0.01)

    def test_plot_vocal_range_saves_png(self, tmp_path):
        result = self._make_result()
        out = tmp_path / "range.png"
        plot_vocal_range(result, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_pitch_contour_saves_png(self, tmp_path):
        result = self._make_result()
        out = tmp_path / "contour.png"
        plot_pitch_contour(result, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_empty_histogram_raises(self):
        result = VocalRangeResult(
            lowest_note="A4", highest_note="A4", lowest_midi=69,
            highest_midi=69, range_semitones=0, range_display="0 semitones",
            note_histogram={}, note_events=[], pitch_frames=[],
        )
        with pytest.raises(ValueError, match="No notes"):
            plot_vocal_range(result)
```

**Step 3: Run tests**

Run: `pytest tests/test_analysis.py -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add src/visualization.py tests/test_analysis.py
git commit -m "feat: add note histogram and pitch contour visualization"
```

---

### Task 9: CLI Entry Point

**Files:**
- Modify: `analyze.py`

**Step 1: Implement CLI**

`analyze.py`:
```python
"""CLI entry point for Vocal Range Analyzer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.analysis import PitchFrame, compute_vocal_range, VocalRangeResult, InsufficientDataError


def run_pipeline_a(
    audio_path: Path,
    output_dir: Path | None,
    device: str | None,
) -> list[PitchFrame]:
    """Pipeline A: Demucs vocal separation + torchcrepe pitch detection."""
    from src.separation import isolate_vocals
    from src.pitch_detection import detect_pitch

    print("[Pipeline A] Separating vocals with Demucs...")
    vocals, sr = isolate_vocals(audio_path, output_dir=output_dir, device=device)
    print(f"[Pipeline A] Vocals isolated ({len(vocals) / sr:.1f}s at {sr}Hz)")

    print("[Pipeline A] Detecting pitch with torchcrepe...")
    frames = detect_pitch(vocals, sr, device=device)
    voiced = sum(1 for f in frames if f.confidence > 0)
    print(f"[Pipeline A] {len(frames)} frames, {voiced} voiced")
    return frames


def run_pipeline_b(audio_path: Path) -> list[PitchFrame]:
    """Pipeline B: BasicPitch direct analysis."""
    from src.basic_pitch_detect import detect_pitch_basic

    print("[Pipeline B] Analyzing with BasicPitch...")
    frames = detect_pitch_basic(audio_path)
    print(f"[Pipeline B] {len(frames)} pitch frames detected")
    return frames


def print_results(result: VocalRangeResult, pipeline_label: str) -> None:
    """Pretty-print analysis results to stdout."""
    print(f"\n{'=' * 50}")
    print(f"  {pipeline_label}")
    print(f"{'=' * 50}")
    print(f"  Lowest note:  {result.lowest_note} (MIDI {result.lowest_midi})")
    print(f"  Highest note: {result.highest_note} (MIDI {result.highest_midi})")
    print(f"  Range:        {result.range_display}")
    print(f"\n  Top notes by duration:")
    sorted_notes = sorted(result.note_histogram.items(), key=lambda x: -x[1])
    for note, dur in sorted_notes[:10]:
        bar = "#" * int(dur * 5)
        print(f"    {note:>4s}: {dur:5.2f}s {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the vocal range of a song.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python analyze.py song.mp3 --pipeline a
  python analyze.py song.mp3 --pipeline b --output results/
  python analyze.py song.mp3 --pipeline both --output results/
        """,
    )
    parser.add_argument("audio_file", type=Path, help="Path to audio file (mp3, wav, flac)")
    parser.add_argument("--pipeline", choices=["a", "b", "both"], default="a",
                        help="Pipeline to use (default: a)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for charts and isolated vocals")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for ML inference: cpu, cuda, mps (default: auto)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for note detection (default: 0.5)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")

    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    pipelines = []
    if args.pipeline in ("a", "both"):
        pipelines.append(("Pipeline A (Demucs + torchcrepe)", "a"))
    if args.pipeline in ("b", "both"):
        pipelines.append(("Pipeline B (BasicPitch)", "b"))

    for label, pipe_id in pipelines:
        t0 = time.time()
        try:
            if pipe_id == "a":
                frames = run_pipeline_a(args.audio_file, args.output, args.device)
            else:
                frames = run_pipeline_b(args.audio_file)

            result = compute_vocal_range(frames, confidence_threshold=args.confidence)
            print_results(result, label)

            if not args.no_plot and args.output:
                from src.visualization import plot_vocal_range, plot_pitch_contour
                plot_vocal_range(
                    result,
                    title=f"{args.audio_file.stem} - {label}",
                    output_path=args.output / f"range_{pipe_id}.png",
                )
                plot_pitch_contour(
                    result,
                    title=f"{args.audio_file.stem} - Pitch Contour ({label})",
                    output_path=args.output / f"contour_{pipe_id}.png",
                )
                print(f"  Charts saved to {args.output}/")

        except InsufficientDataError as e:
            print(f"  {label}: {e}", file=sys.stderr)
        except ImportError as e:
            print(f"  {label}: {e}", file=sys.stderr)
            sys.exit(1)

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test the CLI help**

Run: `python analyze.py --help`
Expected: Shows help text with arguments and examples.

**Step 3: Commit**

```bash
git add analyze.py
git commit -m "feat: add CLI entry point with pipeline orchestration"
```

---

### Task 10: End-to-End Integration Test

**Step 1: Generate test audio**

Run:
```bash
python -c "
import numpy as np, soundfile as sf
sr = 44100
t = np.linspace(0, 5, sr * 5, endpoint=False)
audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype('float32')
sf.write('test_sine.wav', audio, sr)
print('Created test_sine.wav (5s, 440Hz A4)')
"
```

**Step 2: Test Pipeline A**

Run: `python analyze.py test_sine.wav --pipeline a --output test_output/`
Expected: Detects A4 as dominant note, saves charts. May take 30-60s for Demucs model download on first run.

**Step 3: Test Pipeline B**

Run: `python analyze.py test_sine.wav --pipeline b --output test_output/`
Expected: Detects A4 as dominant note, saves charts. Faster than Pipeline A.

**Step 4: Test both pipelines**

Run: `python analyze.py test_sine.wav --pipeline both --output test_output/`
Expected: Both pipelines run, results printed for each, 4 chart PNGs in test_output/.

**Step 5: Verify charts exist**

Run: `ls -la test_output/`
Expected: `range_a.png`, `contour_a.png`, `range_b.png`, `contour_b.png` (and optionally `vocals.wav`).

**Step 6: Clean up test files**

Run: `rm -f test_sine.wav && rm -rf test_output/`

**Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

**Step 8: Final commit**

```bash
git add -A
git commit -m "chore: integration testing complete"
```

---

### Task 11: Verify with Real Audio

This is a manual verification step — find a real song and test both pipelines.

**Step 1: Run on a real song**

Run: `python analyze.py /path/to/song.mp3 --pipeline both --output results/`

**Step 2: Inspect results**

- Check that the detected range seems reasonable for the song
- Open `results/range_a.png` and `results/range_b.png` — do the histograms look sensible?
- Open `results/contour_a.png` and `results/contour_b.png` — is the pitch contour clean?
- Compare Pipeline A vs B — Pipeline A should be more accurate (cleaner histogram)

**Step 3: Adjust thresholds if needed**

If results look noisy, try adjusting confidence:
Run: `python analyze.py song.mp3 --pipeline a --confidence 0.6 --output results_strict/`
