"""Visualization: note histogram and pitch contour charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.analysis import VocalRangeResult, midi_to_note_name


# Standard voice type ranges as (name, low_midi, high_midi, color)
VOICE_TYPES = [
    ("Bass",     40, 64, "#2196F3"),    # E2-E4
    ("Baritone", 45, 69, "#4CAF50"),    # A2-A4
    ("Tenor",    48, 72, "#FF9800"),    # C3-C5
    ("Alto",     53, 77, "#9C27B0"),    # F3-F5
    ("Mezzo",    57, 81, "#E91E63"),    # A3-A5
    ("Soprano",  60, 84, "#F44336"),    # C4-C6
]


def plot_vocal_range(
    result: VocalRangeResult,
    title: str = "Vocal Range Analysis",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Create horizontal bar chart of note occurrence with voice type overlays."""
    if not result.note_histogram:
        raise ValueError("No notes to plot — histogram is empty.")

    # Sort notes by MIDI number (low to high)
    from src.analysis import NOTE_NAMES

    def note_sort_key(name: str) -> int:
        """Parse note name to MIDI for sorting."""
        for i, n in enumerate(NOTE_NAMES):
            if name.startswith(n) and (len(n) == len(name.rstrip("0123456789-"))):
                octave = int(name[len(n):])
                return (octave + 1) * 12 + i
        return 0

    sorted_notes = sorted(result.note_histogram.items(), key=lambda x: note_sort_key(x[0]))
    note_names = [n for n, _ in sorted_notes]
    durations = [d for _, d in sorted_notes]

    # Color by octave
    octaves = [int(n[-1]) if n[-1].isdigit() else int(n[-2:]) for n in note_names]
    cmap = plt.cm.viridis
    unique_octaves = sorted(set(octaves))
    if len(unique_octaves) > 1:
        octave_norm = [(o - min(unique_octaves)) / (max(unique_octaves) - min(unique_octaves))
                       for o in octaves]
    else:
        octave_norm = [0.5] * len(octaves)
    colors = [cmap(v) for v in octave_norm]

    # Figure size scales with number of notes
    fig_height = max(4, len(note_names) * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Voice type background shading
    for vtype_name, low_midi, high_midi, color in VOICE_TYPES:
        low_note = midi_to_note_name(low_midi)
        high_note = midi_to_note_name(high_midi)
        # Find Y positions that fall within this range
        y_positions = []
        for i, n in enumerate(note_names):
            midi = note_sort_key(n)
            if low_midi <= midi <= high_midi:
                y_positions.append(i)
        if y_positions:
            ax.axhspan(
                min(y_positions) - 0.4, max(y_positions) + 0.4,
                alpha=0.08, color=color, label=f"{vtype_name} ({low_note}-{high_note})",
            )

    # Bar chart
    bars = ax.barh(range(len(note_names)), durations, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(note_names)))
    ax.set_yticklabels(note_names, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Duration (seconds)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(durations) * 1.15)

    # Subtitle with range info
    subtitle = f"Range: {result.lowest_note} \u2013 {result.highest_note} | {result.range_display}"
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=10, color="gray")

    # Legend for voice types
    handles = [mpatches.Patch(color=c, alpha=0.3, label=f"{n} ({midi_to_note_name(lo)}-{midi_to_note_name(hi)})")
               for n, lo, hi, c in VOICE_TYPES]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_pitch_contour(
    result: VocalRangeResult,
    title: str = "Pitch Contour",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot raw F0 contour over time."""
    voiced = [f for f in result.pitch_frames if f.frequency_hz > 0 and f.confidence > 0]
    if not voiced:
        raise ValueError("No voiced frames to plot.")

    times = [f.time_seconds for f in voiced]
    freqs = [f.frequency_hz for f in voiced]
    confs = [f.confidence for f in voiced]

    fig, ax = plt.subplots(figsize=(12, 5))
    scatter = ax.scatter(times, freqs, c=confs, s=1, cmap="viridis", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add note name gridlines
    reference_notes = [("C2", 65.41), ("E2", 82.41), ("A2", 110.0),
                       ("C3", 130.81), ("E3", 164.81), ("A3", 220.0),
                       ("C4", 261.63), ("E4", 329.63), ("A4", 440.0),
                       ("C5", 523.25), ("E5", 659.25), ("A5", 880.0),
                       ("C6", 1046.5)]
    for note_name, freq in reference_notes:
        if min(freqs) * 0.8 <= freq <= max(freqs) * 1.2:
            ax.axhline(y=freq, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
            ax.text(max(times) * 1.01, freq, note_name, fontsize=7, va="center", color="gray")

    plt.colorbar(scatter, ax=ax, label="Confidence", shrink=0.8)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
