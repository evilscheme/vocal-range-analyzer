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
