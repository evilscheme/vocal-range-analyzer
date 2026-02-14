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
