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
