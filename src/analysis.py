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
