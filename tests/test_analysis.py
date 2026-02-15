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


from src.analysis import (
    compute_vocal_range, VocalRangeResult, CleaningConfig,
    filter_vocal_range, bridge_gaps, filter_short_notes, collapse_slides, clean_pitch_frames,
)


class TestFilterVocalRange:
    def test_removes_below_e2(self):
        # MIDI 30 = F#1, well below E2 (MIDI 40)
        frames = [PitchFrame(0.0, 46.25, 1.0)]  # F#1
        result = filter_vocal_range(frames)
        assert len(result) == 0

    def test_removes_above_c6(self):
        # MIDI 96 = C7, well above C6 (MIDI 84)
        frames = [PitchFrame(0.0, 2093.0, 1.0)]  # C7
        result = filter_vocal_range(frames)
        assert len(result) == 0

    def test_keeps_valid_range(self):
        frames = [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(20)]  # A4
        result = filter_vocal_range(frames)
        assert len(result) == 20

    def test_preserves_silence_frames(self):
        frames = [PitchFrame(0.0, 0.0, 0.0), PitchFrame(0.01, 440.0, 1.0)]
        result = filter_vocal_range(frames)
        assert len(result) == 2
        assert result[0].frequency_hz == 0.0

    def test_boundary_e2_kept(self):
        frames = [PitchFrame(0.0, 82.41, 1.0)]  # E2
        result = filter_vocal_range(frames)
        assert len(result) == 1

    def test_boundary_c6_kept(self):
        frames = [PitchFrame(0.0, 1046.5, 1.0)]  # C6
        result = filter_vocal_range(frames)
        assert len(result) == 1


class TestBridgeGaps:
    def test_bridges_short_gap_between_same_note(self):
        # A4 (50ms) → gap (30ms) → A4 (50ms) should become continuous A4
        frames = (
            [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(5)]
            + [PitchFrame(t * 0.01 + 0.05, 0.0, 0.0) for t in range(3)]
            + [PitchFrame(t * 0.01 + 0.08, 440.0, 1.0) for t in range(5)]
        )
        result = bridge_gaps(frames, max_gap=0.08, hop_duration=0.01)
        voiced = [f for f in result if f.frequency_hz > 0]
        assert len(voiced) == 13  # all frames now voiced

    def test_does_not_bridge_different_notes(self):
        # A4 → gap → C4 should NOT be bridged
        frames = (
            [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(5)]
            + [PitchFrame(t * 0.01 + 0.05, 0.0, 0.0) for t in range(3)]
            + [PitchFrame(t * 0.01 + 0.08, 261.63, 1.0) for t in range(5)]
        )
        result = bridge_gaps(frames, max_gap=0.08, hop_duration=0.01)
        gaps = [f for f in result if f.frequency_hz == 0.0]
        assert len(gaps) == 3  # gap preserved

    def test_does_not_bridge_long_gap(self):
        # A4 → long gap (200ms) → A4 should NOT be bridged
        frames = (
            [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(5)]
            + [PitchFrame(t * 0.01 + 0.05, 0.0, 0.0) for t in range(20)]
            + [PitchFrame(t * 0.01 + 0.25, 440.0, 1.0) for t in range(5)]
        )
        result = bridge_gaps(frames, max_gap=0.08, hop_duration=0.01)
        gaps = [f for f in result if f.frequency_hz == 0.0]
        assert len(gaps) == 20  # gap preserved

    def test_bridges_multiple_consecutive_gaps(self):
        # A4 → gap → A4 → gap → A4 should all merge
        frames = (
            [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(5)]
            + [PitchFrame(0.05, 0.0, 0.0), PitchFrame(0.06, 0.0, 0.0)]
            + [PitchFrame(t * 0.01 + 0.07, 440.0, 1.0) for t in range(5)]
            + [PitchFrame(0.12, 0.0, 0.0), PitchFrame(0.13, 0.0, 0.0)]
            + [PitchFrame(t * 0.01 + 0.14, 440.0, 1.0) for t in range(5)]
        )
        result = bridge_gaps(frames, max_gap=0.08, hop_duration=0.01)
        voiced = [f for f in result if f.frequency_hz > 0]
        assert len(voiced) == 19  # all frames now voiced

    def test_empty_input(self):
        assert bridge_gaps([]) == []

    def test_pipeline_saves_gapped_note(self):
        # A note sung across syllables: short runs separated by consonant gaps
        # Without bridging, each 50ms run would be filtered out by min_duration=0.1
        # With bridging, they merge into one long run
        frames = (
            [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(5)]       # 50ms A4
            + [PitchFrame(0.05, 0.0, 0.0), PitchFrame(0.06, 0.0, 0.0)]  # 20ms gap
            + [PitchFrame(t * 0.01 + 0.07, 440.0, 1.0) for t in range(5)]  # 50ms A4
            + [PitchFrame(0.12, 0.0, 0.0), PitchFrame(0.13, 0.0, 0.0)]  # 20ms gap
            + [PitchFrame(t * 0.01 + 0.14, 440.0, 1.0) for t in range(5)]  # 50ms A4
        )
        config = CleaningConfig(min_note_duration=0.1, hop_duration=0.01)
        result = clean_pitch_frames(frames, config)
        voiced = [f for f in result if f.frequency_hz > 0]
        # After bridging + duration filter, the merged note should survive
        assert len(voiced) > 0
        assert all(hz_to_midi(f.frequency_hz) == 69 for f in voiced)


class TestFilterShortNotes:
    def test_removes_short_runs(self):
        # 3 frames at 10ms = 30ms, below 100ms threshold
        frames = [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(3)]
        result = filter_short_notes(frames, min_duration=0.1, hop_duration=0.01)
        voiced = [f for f in result if f.frequency_hz > 0]
        assert len(voiced) == 0

    def test_keeps_sustained_notes(self):
        # 20 frames at 10ms = 200ms, above 100ms threshold
        frames = [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(20)]
        result = filter_short_notes(frames, min_duration=0.1, hop_duration=0.01)
        voiced = [f for f in result if f.frequency_hz > 0]
        assert len(voiced) == 20

    def test_gap_separates_runs(self):
        # Two short runs of same note separated by silence — both removed
        frames = (
            [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(3)]
            + [PitchFrame(0.03, 0.0, 0.0)]
            + [PitchFrame(t * 0.01 + 0.04, 440.0, 1.0) for t in range(3)]
        )
        result = filter_short_notes(frames, min_duration=0.1, hop_duration=0.01)
        voiced = [f for f in result if f.frequency_hz > 0]
        assert len(voiced) == 0

    def test_empty_input(self):
        assert filter_short_notes([]) == []

    def test_mixed_short_and_long(self):
        # Short A4 (3 frames) then long C4 (20 frames)
        frames = (
            [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(3)]
            + [PitchFrame(t * 0.01 + 0.03, 261.63, 1.0) for t in range(20)]
        )
        result = filter_short_notes(frames, min_duration=0.1, hop_duration=0.01)
        voiced = [f for f in result if f.frequency_hz > 0]
        assert len(voiced) == 20
        assert all(hz_to_midi(f.frequency_hz) == 60 for f in voiced)


class TestCollapseSlides:
    def test_ascending_slide_collapsed(self):
        # A3 (57) -> A#3 (58) -> B3 (59) -> C4 (60): 3 semitones ascending
        freqs = [220.0, 233.08, 246.94, 261.63]  # A3, A#3, B3, C4
        frames = [PitchFrame(t * 0.01, f, 1.0) for t, f in enumerate(freqs)]
        result = collapse_slides(frames, max_slide_duration=0.3, min_semitones=2, hop_duration=0.01)
        midis = [hz_to_midi(f.frequency_hz) for f in result if f.frequency_hz > 0]
        # Should keep only endpoints: A3 and C4
        assert midis == [57, 60]

    def test_descending_slide_collapsed(self):
        # C4 (60) -> B3 (59) -> A#3 (58) -> A3 (57): 3 semitones descending
        freqs = [261.63, 246.94, 233.08, 220.0]
        frames = [PitchFrame(t * 0.01, f, 1.0) for t, f in enumerate(freqs)]
        result = collapse_slides(frames, max_slide_duration=0.3, min_semitones=2, hop_duration=0.01)
        midis = [hz_to_midi(f.frequency_hz) for f in result if f.frequency_hz > 0]
        assert midis == [60, 57]

    def test_sustained_note_unchanged(self):
        # 30 frames of the same A4 — not a slide
        frames = [PitchFrame(t * 0.01, 440.0, 1.0) for t in range(30)]
        result = collapse_slides(frames)
        assert len(result) == 30

    def test_vibrato_not_collapsed(self):
        # Alternating A4/A#4 — not monotonic
        frames = []
        for t in range(10):
            freq = 440.0 if t % 2 == 0 else 466.16
            frames.append(PitchFrame(t * 0.01, freq, 1.0))
        result = collapse_slides(frames)
        assert len(result) == 10  # All preserved

    def test_empty_input(self):
        assert collapse_slides([]) == []

    def test_short_slide_below_threshold_kept(self):
        # Only 1 semitone change — below min_semitones=2
        frames = [
            PitchFrame(0.0, 440.0, 1.0),   # A4 (69)
            PitchFrame(0.01, 466.16, 1.0),  # A#4 (70)
        ]
        result = collapse_slides(frames, min_semitones=2)
        assert len(result) == 2


class TestCleanPitchFrames:
    def test_full_pipeline(self):
        # Mix of: out-of-range note, short stray note, a slide, and sustained valid notes
        frames = (
            # Sustained C4 (200ms)
            [PitchFrame(t * 0.01, 261.63, 1.0) for t in range(20)]
            # Short stray G#5 (30ms)
            + [PitchFrame(t * 0.01 + 0.20, 830.61, 1.0) for t in range(3)]
            # Slide from E4 to G4 (E4=64, F4=65, F#4=66, G4=67) over 40ms
            + [PitchFrame(0.23, 329.63, 1.0),    # E4
               PitchFrame(0.24, 349.23, 1.0),    # F4
               PitchFrame(0.25, 369.99, 1.0),    # F#4
               PitchFrame(0.26, 392.00, 1.0)]    # G4
            # Sustained A4 (200ms)
            + [PitchFrame(t * 0.01 + 0.27, 440.0, 1.0) for t in range(20)]
        )
        config = CleaningConfig(min_note_duration=0.1, hop_duration=0.01)
        result = clean_pitch_frames(frames, config)
        midis = [hz_to_midi(f.frequency_hz) for f in result if f.frequency_hz > 0]
        # C4 sustained should survive, short G#5 removed, slide collapsed to E4+G4,
        # A4 sustained should survive
        assert 60 in midis   # C4
        assert 69 in midis   # A4
        # The intermediate slide notes (F4=65, F#4=66) should be gone
        assert 65 not in midis
        assert 66 not in midis


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


class TestComputeVocalRangeCleaning:
    def test_cleaning_removes_outlier_from_range(self):
        # Sustained A4 + short stray C2 (below duration threshold)
        frames = (
            [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]   # A4 = 1s
            + [PitchFrame(t * 0.01 + 1.0, 65.41, 0.9) for t in range(3)]  # C2 = 30ms
        )
        result = compute_vocal_range(frames, percentile_trim=0.0, hop_duration_seconds=0.01)
        # With cleaning, the short C2 should be filtered out
        assert result.lowest_note == "A4"
        assert result.highest_note == "A4"

    def test_no_cleaning_keeps_outlier(self):
        frames = (
            [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]
            + [PitchFrame(t * 0.01 + 1.0, 65.41, 0.9) for t in range(3)]
        )
        result = compute_vocal_range(
            frames, percentile_trim=0.0, hop_duration_seconds=0.01,
            cleaning_config=None,
        )
        # Without cleaning, C2 is included
        assert result.lowest_midi < 69  # lower than A4

    def test_raw_frames_preserved(self):
        frames = [PitchFrame(t * 0.01, 440.0, 0.9) for t in range(100)]
        result = compute_vocal_range(frames, hop_duration_seconds=0.01)
        assert result.pitch_frames is frames  # exact same object


from src.visualization import plot_vocal_range, plot_pitch_contour, plot_piano_roll


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

    def test_plot_piano_roll_saves_png(self, tmp_path):
        result = self._make_result()
        out = tmp_path / "piano_roll.png"
        plot_piano_roll(result, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_piano_roll_empty_events_raises(self):
        result = VocalRangeResult(
            lowest_note="A4", highest_note="A4", lowest_midi=69,
            highest_midi=69, range_semitones=0, range_display="0 semitones",
            note_histogram={}, note_events=[], pitch_frames=[],
        )
        with pytest.raises(ValueError, match="No note events"):
            plot_piano_roll(result)


from src.midi_export import export_midi
from src.analysis import NoteEvent


class TestMidiExport:
    def test_export_creates_file(self, tmp_path):
        events = [
            NoteEvent(midi_number=60, note_name="C4", start_time=0.0, duration=1.0, mean_frequency_hz=261.63),
            NoteEvent(midi_number=64, note_name="E4", start_time=1.0, duration=0.5, mean_frequency_hz=329.63),
            NoteEvent(midi_number=67, note_name="G4", start_time=1.5, duration=0.5, mean_frequency_hz=392.00),
        ]
        out = tmp_path / "test.mid"
        export_midi(events, out)
        assert out.exists()
        assert out.stat().st_size > 0
        # MIDI files start with "MThd"
        assert out.read_bytes()[:4] == b"MThd"

    def test_export_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No note events"):
            export_midi([], tmp_path / "empty.mid")
