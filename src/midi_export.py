"""Export NoteEvents to a standard MIDI file."""
from __future__ import annotations

from pathlib import Path

from midiutil import MIDIFile

from src.analysis import NoteEvent


def export_midi(
    note_events: list[NoteEvent],
    output_path: str | Path,
    tempo: float = 120.0,
    velocity: int = 100,
    program: int = 52,  # General MIDI "Choir Aahs"
) -> None:
    """Write note events to a MIDI file.

    Parameters
    ----------
    note_events:
        Cleaned note events from the analysis pipeline.
    output_path:
        Path to write the .mid file.
    tempo:
        BPM used to map real-time seconds to MIDI ticks.
    velocity:
        MIDI velocity (loudness) for all notes.
    program:
        General MIDI program number.  Default 52 = "Choir Aahs".
    """
    if not note_events:
        raise ValueError("No note events to export.")

    midi = MIDIFile(1)  # single track
    track = 0
    channel = 0

    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, program)

    beats_per_second = tempo / 60.0

    for event in note_events:
        start_beat = event.start_time * beats_per_second
        duration_beats = event.duration * beats_per_second
        midi.addNote(track, channel, event.midi_number, start_beat, duration_beats, velocity)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        midi.writeFile(f)
