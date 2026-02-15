"""Pitch analysis: Hz-to-note conversion, histogram building, range calculation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

# Human vocal range hard limits
_VOCAL_MIN_MIDI = 40  # E2 (~82.4 Hz)
_VOCAL_MAX_MIDI = 84  # C6 (~1046.5 Hz)


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
class CleaningConfig:
    """Configuration for the pitch-frame cleaning pipeline."""
    min_midi: int = _VOCAL_MIN_MIDI
    max_midi: int = _VOCAL_MAX_MIDI
    min_note_duration: float = 0.1       # seconds
    max_gap_to_bridge: float = 0.08     # seconds (80 ms)
    max_slide_duration: float = 0.3      # seconds
    slide_min_semitones: int = 2
    hop_duration: float = 0.01           # seconds (10 ms)


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


def filter_vocal_range(
    pitch_frames: list[PitchFrame],
    min_midi: int = _VOCAL_MIN_MIDI,
    max_midi: int = _VOCAL_MAX_MIDI,
) -> list[PitchFrame]:
    """Remove frames outside the plausible human vocal range (E2–C6)."""
    result = []
    for f in pitch_frames:
        if f.frequency_hz <= 0 or f.confidence <= 0:
            result.append(f)
            continue
        midi = hz_to_midi(f.frequency_hz)
        if min_midi <= midi <= max_midi:
            result.append(f)
    return result


def bridge_gaps(
    pitch_frames: list[PitchFrame],
    max_gap: float = 0.08,
    hop_duration: float = 0.01,
) -> list[PitchFrame]:
    """Bridge short silence gaps between runs of the same note.

    When singing words on a sustained pitch, consonants create tiny gaps
    (20-80 ms) in the pitch detection.  This function fills those gaps so
    that 'A4 [gap] A4' becomes one continuous A4 run.

    Gap frames are replaced with copies that carry the bridging note's
    frequency and a confidence of 1.0.
    """
    if not pitch_frames:
        return []

    max_gap_frames = max(1, int(max_gap / hop_duration))
    result = list(pitch_frames)  # shallow copy

    # Build (start, length, midi|None) run list
    runs: list[tuple[int, int, int | None]] = []
    run_start = 0
    run_midi: int | None = None
    run_len = 0

    for i, f in enumerate(pitch_frames):
        if f.frequency_hz <= 0 or f.confidence <= 0:
            midi = None
        else:
            midi = hz_to_midi(f.frequency_hz)

        if midi == run_midi:
            run_len += 1
        else:
            if run_len > 0:
                runs.append((run_start, run_len, run_midi))
            run_start = i
            run_midi = midi
            run_len = 1

    if run_len > 0:
        runs.append((run_start, run_len, run_midi))

    # Scan for pattern: voiced_run, short_gap, same_voiced_run
    i = 0
    while i < len(runs) - 2:
        s1, l1, m1 = runs[i]
        s2, l2, m2 = runs[i + 1]
        s3, l3, m3 = runs[i + 2]

        if m1 is not None and m2 is None and m3 == m1 and l2 <= max_gap_frames:
            # Bridge the gap: fill silence frames with the surrounding note
            freq = pitch_frames[s1].frequency_hz
            for k in range(s2, s2 + l2):
                result[k] = PitchFrame(
                    time_seconds=pitch_frames[k].time_seconds,
                    frequency_hz=freq,
                    confidence=1.0,
                )
            # Merge the three runs into one and re-check from this position
            merged = (s1, l1 + l2 + l3, m1)
            runs[i] = merged
            del runs[i + 1:i + 3]
        else:
            i += 1

    return result


def filter_short_notes(
    pitch_frames: list[PitchFrame],
    min_duration: float = 0.1,
    hop_duration: float = 0.01,
) -> list[PitchFrame]:
    """Remove runs of consecutive same-note frames shorter than *min_duration*.

    Unlike group_into_note_events, this returns PitchFrames (not NoteEvents)
    so downstream functions can still consume frames.
    """
    if not pitch_frames:
        return []

    min_frames = max(1, int(min_duration / hop_duration))

    # First pass: identify runs (start_index, length, midi)
    runs: list[tuple[int, int, int | None]] = []
    run_start = 0
    run_midi: int | None = None
    run_len = 0

    for i, f in enumerate(pitch_frames):
        if f.frequency_hz <= 0 or f.confidence <= 0:
            if run_len > 0:
                runs.append((run_start, run_len, run_midi))
            # Silence is always kept
            runs.append((i, 1, None))
            run_len = 0
            run_midi = None
            continue

        midi = hz_to_midi(f.frequency_hz)
        if midi == run_midi:
            run_len += 1
        else:
            if run_len > 0:
                runs.append((run_start, run_len, run_midi))
            run_start = i
            run_midi = midi
            run_len = 1

    if run_len > 0:
        runs.append((run_start, run_len, run_midi))

    # Second pass: keep frames from runs that are long enough (or silence)
    result: list[PitchFrame] = []
    for start, length, midi in runs:
        if midi is None or length >= min_frames:
            result.extend(pitch_frames[start:start + length])
    return result


def collapse_slides(
    pitch_frames: list[PitchFrame],
    max_slide_duration: float = 0.3,
    min_semitones: int = 2,
    hop_duration: float = 0.01,
) -> list[PitchFrame]:
    """Detect monotonic pitch slides and collapse to endpoint frames.

    A slide is a sequence of voiced frames where the MIDI note changes by
    exactly +1 or -1 each step, spans >= *min_semitones*, and lasts
    <= *max_slide_duration*.  The intermediate frames are dropped; only the
    first and last frames of the slide are kept.
    """
    if not pitch_frames:
        return []

    max_frames = int(max_slide_duration / hop_duration)

    # Convert to (index, midi_or_none) pairs for analysis
    midi_seq: list[int | None] = []
    for f in pitch_frames:
        if f.frequency_hz > 0 and f.confidence > 0:
            midi_seq.append(hz_to_midi(f.frequency_hz))
        else:
            midi_seq.append(None)

    n = len(midi_seq)
    keep = [True] * n

    i = 0
    while i < n:
        if midi_seq[i] is None:
            i += 1
            continue

        # Try to extend a monotonic run starting at i
        j = i + 1
        while j < n and midi_seq[j] is not None:
            diff = midi_seq[j] - midi_seq[j - 1]  # type: ignore[operator]
            if j == i + 1:
                direction = diff
                if direction not in (1, -1):
                    break
            else:
                if diff != direction:
                    break
            j += 1

        run_len = j - i
        semitones = abs(midi_seq[j - 1] - midi_seq[i]) if run_len > 1 else 0  # type: ignore[operator]

        if run_len > 2 and semitones >= min_semitones and run_len <= max_frames:
            # Mark intermediate frames for removal (keep first and last)
            for k in range(i + 1, j - 1):
                keep[k] = False
            i = j
        else:
            i += 1

    return [f for f, k in zip(pitch_frames, keep) if k]


def clean_pitch_frames(
    pitch_frames: list[PitchFrame],
    config: CleaningConfig | None = None,
) -> list[PitchFrame]:
    """Run the full cleaning pipeline: range → bridge gaps → duration → slides."""
    if config is None:
        config = CleaningConfig()

    frames = filter_vocal_range(pitch_frames, config.min_midi, config.max_midi)
    frames = bridge_gaps(frames, config.max_gap_to_bridge, config.hop_duration)
    frames = filter_short_notes(frames, config.min_note_duration, config.hop_duration)
    frames = collapse_slides(
        frames, config.max_slide_duration, config.slide_min_semitones, config.hop_duration
    )
    return frames


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
    cleaning_config: CleaningConfig | None = CleaningConfig(),
) -> VocalRangeResult:
    """Compute full vocal range analysis from pitch frames.

    Filters by confidence, applies probabilistic cleaning, trims outliers
    by percentile, then computes range, histogram, and note events.

    Pass ``cleaning_config=None`` to disable the cleaning pipeline.
    """
    # Filter confident, voiced frames
    voiced = [f for f in pitch_frames
              if f.confidence >= confidence_threshold and f.frequency_hz > 0]

    if len(voiced) < 10:
        raise InsufficientDataError(
            f"Only {len(voiced)} voiced frames detected (need at least 10). "
            "The audio may not contain vocals."
        )

    # Apply cleaning pipeline (range limits, duration, slides)
    if cleaning_config is not None:
        cleaned = clean_pitch_frames(voiced, cleaning_config)
        # Re-filter to only voiced after cleaning (filter_vocal_range may
        # have removed some frames entirely)
        cleaned = [f for f in cleaned if f.frequency_hz > 0 and f.confidence > 0]
        if len(cleaned) >= 10:
            voiced = cleaned

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

    # Build histogram and events from cleaned voiced frames
    histogram = build_note_histogram(
        voiced, confidence_threshold, hop_duration_seconds
    )
    note_events = group_into_note_events(
        voiced, confidence_threshold, min_note_duration, hop_duration_seconds
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
        pitch_frames=pitch_frames,  # raw frames preserved for contour viz
    )
