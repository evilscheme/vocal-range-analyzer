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
