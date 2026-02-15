"""CLI entry point for Vocal Range Analyzer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.analysis import (
    CleaningConfig, PitchFrame, compute_vocal_range, VocalRangeResult, InsufficientDataError,
)


def run_pipeline_a(
    audio_path: Path,
    output_dir: Path | None,
    device: str | None,
) -> list[PitchFrame]:
    """Pipeline A: Demucs vocal separation + FCPE pitch detection."""
    from src.separation import isolate_vocals
    from src.pitch_detection import detect_pitch

    print("[Pipeline A] Separating vocals with Demucs...")
    vocals, sr = isolate_vocals(audio_path, output_dir=output_dir, device=device)
    print(f"[Pipeline A] Vocals isolated ({len(vocals) / sr:.1f}s at {sr}Hz)")
    if output_dir:
        print(f"[Pipeline A] Saved vocals.wav and karaoke.wav to {output_dir}/")

    print("[Pipeline A] Detecting pitch with FCPE...")
    frames = detect_pitch(vocals, sr, device=device)
    voiced = sum(1 for f in frames if f.confidence > 0)
    print(f"[Pipeline A] {len(frames)} frames, {voiced} voiced")
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
  python analyze.py song.mp3 --output results/
  python analyze.py song.mp3 --confidence 0.6 --output results/
        """,
    )
    parser.add_argument("audio_file", type=Path, help="Path to audio file (mp3, wav, flac)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for charts and isolated vocals")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for ML inference: cpu, cuda, mps (default: auto)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for note detection (default: 0.5)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--min-duration", type=float, default=0.1,
                        help="Minimum note duration in seconds (default: 0.1)")
    parser.add_argument("--no-cleaning", action="store_true",
                        help="Disable probabilistic note cleaning")

    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        frames = run_pipeline_a(args.audio_file, args.output, args.device)

        cleaning_config = None if args.no_cleaning else CleaningConfig(
            min_note_duration=args.min_duration,
        )
        result = compute_vocal_range(
            frames,
            confidence_threshold=args.confidence,
            cleaning_config=cleaning_config,
        )
        print_results(result, "Demucs + FCPE")

        if not args.no_plot and args.output:
            from src.visualization import plot_vocal_range, plot_pitch_contour, plot_piano_roll
            stem = args.audio_file.stem
            plot_vocal_range(
                result,
                title=f"{stem} - Vocal Range",
                output_path=args.output / "range.png",
            )
            plot_pitch_contour(
                result,
                title=f"{stem} - Pitch Contour",
                output_path=args.output / "contour.png",
            )
            if result.note_events:
                plot_piano_roll(
                    result,
                    title=f"{stem} - Piano Roll",
                    output_path=args.output / "piano_roll.png",
                )
            print(f"  Charts saved to {args.output}/")

        if args.output and result.note_events:
            from src.midi_export import export_midi
            midi_path = args.output / "notes.mid"
            export_midi(result.note_events, midi_path)
            print(f"  MIDI saved to {midi_path}")

    except InsufficientDataError as e:
        print(f"  Error: {e}", file=sys.stderr)
    except ImportError as e:
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
