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
    """Pipeline A: Demucs vocal separation + FCPE pitch detection."""
    from src.separation import isolate_vocals
    from src.pitch_detection import detect_pitch

    print("[Pipeline A] Separating vocals with Demucs...")
    vocals, sr = isolate_vocals(audio_path, output_dir=output_dir, device=device)
    print(f"[Pipeline A] Vocals isolated ({len(vocals) / sr:.1f}s at {sr}Hz)")

    print("[Pipeline A] Detecting pitch with FCPE...")
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
        pipelines.append(("Pipeline A (Demucs + FCPE)", "a"))
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
